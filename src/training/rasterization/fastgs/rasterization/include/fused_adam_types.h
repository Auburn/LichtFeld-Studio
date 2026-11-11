/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace fast_lfs::rasterization {

    // Optimizer moments use joint (u, log_s) packing plus float4 bounds per
    // 256-splat block. joint_bits == 0 disables updates; 8 or 16 selects the codec.
    // SH value quant: when sh_value_bits==16, `param` is uint16 codes
    // (pad-dropped cell-linear swizzle) and sh_value_bounds is float2 per 256-prim block.
    struct FusedAdamParam {
        // Optional raw gradient destination. When present, capture instead of updating Adam.
        float* gradient_out = nullptr;
        float* param = nullptr; // float params OR bitcast uint16* when sh_value_bits==16
        // Joint codec: packed (u,log_s) bytes + float4 bounds[n_bounds]
        std::uint8_t* joint_packed = nullptr;
        float* joint_bounds = nullptr; // float4 as 4 floats per bound
        int joint_bits = 0;            // 0=disabled, 8=SH, 16=non-SH
        // SH value quant — only meaningful on fused.shN
        float* sh_value_bounds = nullptr; // float2 as 2 floats per 256-prim block
        int sh_value_bits = 0;            // 0=fp32 param, 16=uint16 codes
        int sh_value_n_cells = 0;         // cells per prim (45 for SH3 pad-dropped)
        int n_primitives = 0;             // live splat count (bounds index = prim/256)
        const bool* frozen_mask = nullptr;
        int frozen_mask_size = 0;
        float frozen_lr_scale = 0.0f;
        const bool* crop_damping_mask = nullptr;
        int crop_damping_mask_size = 0;
        float cropbox_lr_scale = 1.0f;
        int n_elements = 0;
        int n_attributes = 0;
        float step_size = 0.0f;
        float bias_correction2_sqrt_rcp = 1.0f;
        bool enabled = false;
        const float* screen_share_max = nullptr;
        int screen_share_n = 0;
        float screen_share_limit = 0.0f;
        float screen_share_penalty = 0.0f;
        int screen_share_axis = 0; // 0 max: hinge pushes all 3 log-scales; 1 area: not the shortest (screen_share.cuh)
    };

    struct FusedAdamSettings {
        bool enabled = false;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float scale_reg_weight = 0.0f;
        bool scale_reg_log = false;
        float scale_reg_normalizer = 1.0f;
        float erank_reg_weight = 0.0f;
        float dc_reg_weight = 0.0f;
        float sh_rest_reg_weight = 0.0f;
        float flatten_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        // weight * mean(o (1 - o)): pushes opacities toward 0 or 1. 0 = off.
        float opacity_binarize_weight = 0.0f;
        float* rendered_count = nullptr;
        // Optional persistent device scalars (caller zeros). Filled by preprocess_backward
        // via block-reduce + atomicAdd so the trainer can skip loss-only reg kernels.
        float* scale_reg_loss_out = nullptr;
        float* opacity_reg_loss_out = nullptr;
        float* opacity_binarize_loss_out = nullptr;
        float* erank_reg_loss_out = nullptr;
        float* sh_rest_reg_loss_out = nullptr;
        float* dc_reg_loss_out = nullptr;
        // --loss-footprint (LFQPLCYKMO; lfs/training/game_footprint.cuh): lambda x (1 / (W H)) x the sum over the view's visible
        // splats of the quad area the game draws for each. footprint_weight = lambda (ramped; 0 = off: nothing below is read).
        // Gradient into the log-scales through the projected 2D covariance; into the raw opacity through the reach only with
        // footprint_opacity (shaped splats whose alpha-floor contour lies inside the cut). Beta, means and rotations get none.
        float footprint_weight = 0.0f;
        bool footprint_opacity = false;
        float footprint_max_px_per_fx = 0.0f;           // the game's per-axis quad cap / the view's fx (kGameMaxPixelRadiusPerFx)
        const float* footprint_shape_raw = nullptr;     // --kernel-shape raw r per splat (beta); null = every splat a Gaussian
        float* footprint_loss_out = nullptr;            // optional: += lambda x sum A / (W H)
        float* footprint_area_out = nullptr;            // optional: += sum A / (W H) (mean splats drawn per pixel)
        bool per_splat_mean_step = false;
        float mean_step_median_extent = 0.0f;
        const bool* mean_step_far_mask = nullptr;
        int mean_step_far_mask_n = 0;
        // --grad-scale-near D0 (LFQPLCYKMGS; after Philip & Deschaintre, "Floaters No More: Radiance Field Gradient Scaling",
        // EGSR 2023): every gradient this view gives a visible splat's own parameters (means, log-scales, rotation, raw opacity,
        // sh0, shN and, through grad_scale_near_shape_grad, the --kernel-shape beta) is multiplied by s = min(1, d^2 / D0^2),
        // d = the distance of the splat's centre from the view's camera centre (cam_position; a distance, not camera z: the
        // lens is a ~180-degree fisheye). That covers every per-view term that reaches the splat: the image loss, the depth
        // distortion term, depth / normal supervision and the footprint loss. Left as they are: the per-splat regularisers
        // (scale / opacity / binarize / erank / flatten / dc / sh-rest, the screen-share hinge), every camera-side gradient
        // (pose, rolling-shutter motion and readout, exposure scale, grad_w2c) and the blend backward's densification /
        // removal-impact sums. Splats at d >= D0 are untouched (s = 1: not even multiplied). 0 = off: nothing below is read.
        float grad_scale_near_inv_d0_sq = 0.0f;      // 1 / D0^2 (D0 in scene units)
        float* grad_scale_near_shape_grad = nullptr; // set by backward(): this view's dL/dbeta buffer, scaled in place per splat
        float* grad_scale_near_stats_out = nullptr;  // optional [3]: += visible splats, += those with s < 1, += the sum of their s
        // --aniso-max-ratio r / --aniso-weight lambda (LFQPLCYKMAN; after PhysGaussian, Xie et al. 2024): a per-splat regulariser
        // like erank, evaluated for EVERY splat (visible or not) once per step: loss = lambda / N x sum max(0, s_max / s_mid - r),
        // s = exp(log-scales). Gradient into the log-scales only: d/dl_max = +c rho, d/dl_mid = -c rho (rho = s_max / s_mid,
        // c = lambda / N) for splats over r; zero otherwise. Not scaled by --grad-scale-near (a parameter prior). r = 0 = off.
        float aniso_max_ratio = 0.0f;
        float aniso_weight = 0.0f;
        float* aniso_loss_out = nullptr;  // optional: += the loss (lambda x mean excess)
        float* aniso_stats_out = nullptr; // optional [2]: += splats, += splats with rho > r
        // --thick-max-ratio R / --thick-weight lambda (LFQPLCYKMT; a thickness floor, PhysGaussian's anisotropy term on the THIN axis
        // in log form): a per-splat regulariser like erank, evaluated for EVERY splat (visible or not) once per step:
        // loss = lambda / N x sum max(0, (l_mid - l_min) - ln R), l = the log-scales (s = exp(l)): ln(s_mid / s_min) over ln R.
        // Gradient into the log-scales only: d/dl_mid = +c, d/dl_min = -c (c = lambda / N) for splats over R; zero otherwise.
        // Bounded whatever the ratio (paper-thin discs reach s_mid / s_min ~ 1e15). Not scaled by --grad-scale-near. R = 0 = off.
        float thick_max_ratio = 0.0f; // R (> 0 = on)
        float thick_log_ratio = 0.0f; // ln R (set with it)
        float thick_weight = 0.0f;
        float* thick_loss_out = nullptr;  // optional: += the loss (lambda x mean log excess)
        float* thick_stats_out = nullptr; // optional [2]: += splats, += splats with s_ref / s_min > R
        bool thick_longest = false;       // --thick-longest: s_ref = s_max (d/dl_max = +c) instead of s_mid

        FusedAdamParam means;
        FusedAdamParam scaling;
        FusedAdamParam rotation;
        FusedAdamParam opacity;
        FusedAdamParam sh0;
        FusedAdamParam shN;
    };

} // namespace fast_lfs::rasterization
