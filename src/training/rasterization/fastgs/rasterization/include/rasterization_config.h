/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/radial_lens.h" // --lens-radial (LFRL): the radial lens table view in FastGSLens

#include "helper_math.h"

#define DEF inline constexpr

enum class DensificationType : int { None = 0,
                                     MCMC = 1,
                                     MRNF = 2 };

namespace fast_lfs::rasterization {
    // dL/dalpha of the background blend (output = color + (1 - alpha) * bg).
    // A precomputed map wins; otherwise the blend backward derives it per pixel
    // from the image gradient and the background (CHW image, else solid color)
    // and adds the optional extra alpha gradient.
    struct BackgroundAlphaGradient {
        const float* grad_alpha_map = nullptr;
        const float* bg_color = nullptr;
        const float* bg_image = nullptr;
        const float* grad_alpha_extra = nullptr;
    };

    // The camera projection the preprocess kernels apply to splat centres and covariances.
    //   kLensPinhole: the rasterizer's own pinhole projection (u = fx x / z + cx).
    //   kLensFisheye: COLMAP's OPENCV_FISHEYE. With rho = |(x, y)| and theta = atan2(rho, z),
    //     theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),
    //     u = fx theta_d x / rho + cx, v = fy theta_d y / rho + cy (k = 0: the equidistant lens).
    //     The 2D covariance is J W Sigma W^T J^T with J the analytic Jacobian of (u, v) at the
    //     centre; depth (sort key, depth map) is the distance from the camera, not z.
    //     Centres more than theta_max off the axis (rho^2 > max_w z^2, max_w = tan^2 theta_max),
    //     or behind the camera, are culled.
    // Pixel centres sit at integer + 0.5 for both, as in COLMAP.
    //   radial.seg != nullptr (--lens-radial, LFRL): theta_d = theta P(theta^2) with P from the radial lens table
    //     (core/radial_lens.h) instead of the k-polynomial; everything else as kLensFisheye. nullptr: the polynomial.
    enum : int { kLensPinhole = 0,
                 kLensFisheye = 1 };
    struct FastGSLens {
        int model = kLensPinhole;
        float k1 = 0.0f;
        float k2 = 0.0f;
        float k3 = 0.0f;
        float k4 = 0.0f;
        float max_w = 0.0f;
        lfs::core::RadialLensView radial{}; // device table (--lens-radial); seg == nullptr: the k-polynomial
    };

    // Depth distortion loss (2DGS / Mip-NeRF 360) folded into the blend kernels. Per pixel
    //   D = sum_i sum_{j<i} w_i w_j (m_i - m_j)^2
    // over the splats that take part in the term, with w the blending weights front to back (alpha_i T_i;
    // T includes every splat, taking part or not) and m the splat's distortion depth d mapped to [0, 1]
    // by m = far/(far - near) * (1 - near/d), d clamped to [near, far].
    // The distortion depth is its own per-splat value (depths[], written by the preprocess), used ONLY by
    // this term; sorting and the depth output keep the rasterizer's depth (camera z for pinhole, distance
    // for fisheye):
    //   depth_mode kDistortionDepthDistance: the distance from the camera centre |p_cam| (pinhole and fisheye).
    //   depth_mode kDistortionDepthZ: the sort depth (camera z for pinhole, distance for fisheye); the
    //     behaviour before 2026-11-01, kept for comparison. Splats beside a pinhole camera have tiny z
    //     whatever their distance, so they map to m ~ 0 and the term rewards opaque near-plane "blobs".
    // plane_sigma > 0 leaves out of the term (for that view) every splat whose plane_sigma-sigma extent
    // along the camera axis crosses the camera plane: z - plane_sigma * sigma_z < 0, sigma_z^2 = a^T Sigma a
    // with a the camera axis (stored as depths[] = -1). They still occlude (their alpha is in T).
    // min_dist > 0 also leaves out (the same way) every splat whose distortion depth d is below min_dist
    // (2026-11-01, round 2). With near = 0.2 the mapping is so steep below ~1 unit that the edges of a very
    // close object against the background (|m_i - m_j| ~ 0.3-1) outweigh everything else in the term, and
    // the cheapest optimum turns the object's blurry multi-view fringe into a wide opaque wall; below near,
    // m is clamped at 0 and the term cannot even push such splats back. Leaving them out keeps the term for
    // the floor and floaters further away (d >= min_dist).
    // The forward keeps A = sum w, M1 = sum w m and M2 = sum w m^2 per pixel over the splats taking part
    // and adds w_i (m_i^2 A + M2 - 2 m_i M1) for every new one (A, M1, M2 in front of it).
    // moments == nullptr turns it off; the default kernels are then used unchanged.
    enum : int { kDistortionDepthZ = 0,
                 kDistortionDepthDistance = 1 };
    struct FastGSDistortion {
        float* moments = nullptr;            // [3*H*W]: M1, M2, then A per pixel; forward writes, backward reads
        float* loss_out = nullptr;           // [1]: forward adds loss_scale * sum over pixels of D (caller zeros); may be null
        const float* pixel_weight = nullptr; // [H*W] optional per-pixel factor on D (null = 1)
        float* depths = nullptr;             // [N] per model splat: distortion depth (< 0 = left out); preprocess writes. Required with moments.
        float* grad_depths = nullptr;        // [N] per model splat: dL/d(distortion depth); blend backward adds (caller zeros). Required with moments.
        float* n_left_out = nullptr;         // [2] optional: preprocess adds 1 per visible splat left out: [0] plane-crossing, [1] nearer than min_dist (and not plane-crossing) (caller zeros)
        float loss_scale = 0.0f;             // forward: loss weight / number of pixels
        float grad_scale = 0.0f;             // backward: dL/dD of every pixel (loss weight / number of pixels)
        float near_n = 0.2f;
        float far_n = 1000.0f;
        int depth_mode = kDistortionDepthDistance;
        float plane_sigma = 2.0f; // 0 = no splat is left out
        float min_dist = 0.0f;    // splats with distortion depth < min_dist are left out; 0 = none
    };

    // Dynamic rolling shutter (2026-11-02; native fisheye only). Sensor row v (this render's pixels, pixel centres at
    // +0.5) is read at t = clamp((v + row_offset) / H - 1/2, -1/2, 1/2) tau, top row first (H = sensor_height, or the
    // render height when 0). The camera at time t relative to its mid-readout pose (the w2c the rasterizer is given):
    // camera-space p(t) = E(t) (p_mid - d(t)), E = exp(-[r(t)]x), r(t) = omega t + alpha t^2 / 2, d(t) = vel t + acc t^2 / 2
    // (mid-readout camera axes). Each splat is drawn at ONE time, the fixed point of its centre's row (iters
    // re-projections from the mid pose): its centre, covariance, depth and normal go through that pose; colours (SH)
    // keep the mid-readout camera centre. The backward holds each splat's time fixed and, when grad is bound, adds
    // dL/d(motion, tau). motion == nullptr = off: the unchanged kernels run. See rolling_shutter.cuh.
    struct FastGSRollingShutter {
        const float* motion = nullptr;  // device [12]: omega, alpha, vel, acc (rad/s, rad/s^2, units/s, units/s^2)
        const float* readout = nullptr; // device [1]: tau in seconds (required with motion)
        float* grad = nullptr;          // device [13] optional: the backward adds dL/d(motion[12]) and dL/dtau (caller zeros)
        float* times = nullptr;         // device [N] optional (checks): per model splat row fraction s (t = s tau); the
                                        // forward writes it for the splats it draws, or reads it when times_fixed != 0
        int times_fixed = 0;
        int iters = 2;                  // fixed-point re-projections (>= 1)
        float row_offset = 0.0f;        // sensor rows above this render's row 0 (tiled renders)
        float sensor_height = 0.0f;     // sensor rows in this render's pixels (0 = the render height)
        // --motion-blur (LFQPLCYM): the exposure on top of the row times. A row read at time t integrates its pose over
        // [t - Te/2, t + Te/2] (its rolling-shutter time is the middle of its exposure). Screen-space model
        // (rolling_shutter.cuh, blur_*): each splat's dilated 2D covariance S0 gets + dmu dmu^T / 12 (the variance of
        // a box of length |dmu|), dmu = J (p(t + Te/2) - p(t - Te/2)) the smear of its centre (p(.) the camera-space
        // centre through the motion's pose, rotation and travel; J the lens Jacobian at p(t)), and its opacity
        // x sqrt(det S0 / det(S0 + dmu dmu^T / 12)) so its integrated coverage is unchanged.
        const float* exposure = nullptr;       // device [1]: T, seconds (null = no blur: the unchanged results)
        const float* exposure_scale = nullptr; // device [1] optional: this camera's learned scale (Te = T x scale; null = 1)
        float* grad_exposure = nullptr;        // device [1] optional: the backward adds dL/d(exposure scale) (caller zeros)
        float* smear = nullptr;                // device [N] optional (checks): |dmu| in pixels per drawn model splat
        float time_offset = 0.0f;              // seconds added to every splat's time (sub-exposure reference renders)
        // --rs-time-map (LFQPLCYKMR, 11 Nov 2026): a per-pixel readout time in place of the row rule. A camera-space point
        // imaged at (u, v) of this render is read at s(fu, fv), fu = u / render width, fv = (v + row_offset) / H (H as above;
        // both clamped to [0, 1]: 0 = the image's left / top edge, 1 = its right / bottom edge, pixel-corner coordinates),
        // and t = s tau. Two forms (training/rs_time_map.hpp):
        //   time_radial != 0: the camera's radial remap of its sensor rows, x = 2 fu - 1 - tr_cx, y = (2 fv - 1) tr_aspect - tr_cy
        //     (both in half image widths), s = tr_k y (1 + tr_a r^2 + tr_b r^4 + tr_c r^6) - tr_s0, r^2 = x^2 + y^2;
        //   time_map != nullptr: device [time_map_ny * time_map_nx] s at grid nodes, node (i, j) at (i / (nx - 1), j / (ny - 1)),
        //     bilinear between them.
        // The host scales the time by 1 / (its linear-equivalent readout) and removes the centre's time, so tau keeps its
        // meaning and t = 0 at the image centre (a linear form at readout R gives s = v / H - 1/2, the row rule). The backward
        // holds each splat's time fixed as with the row rule (dL/dtau = s x ...). Neither set = the row rule, the unchanged
        // arithmetic.
        const float* time_map = nullptr;
        int time_map_nx = 0;
        int time_map_ny = 0;
        int time_radial = 0;
        float tr_k = 0.0f, tr_a = 0.0f, tr_b = 0.0f, tr_c = 0.0f, tr_cx = 0.0f, tr_cy = 0.0f;
        float tr_aspect = 0.75f;
        float tr_s0 = 0.0f;
    };
} // namespace fast_lfs::rasterization

namespace fast_lfs::rasterization::config {
    // rendering constants
    DEF float dilation = 0.3f;            // Standard dilation when mip_filter OFF
    DEF float dilation_mip_filter = 0.1f; // Smaller dilation when mip_filter ON
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float max_fragment_alpha = 0.999f;                          // 0.99f in original 3dgs
    DEF float min_cov2d_determinant = 1e-6f;
    DEF float transmittance_threshold = 1e-4f;
    DEF float max_raw_scale = 20.0f;  // exp(40) ≈ 2.35e17, safe margin before overflow
    DEF float max_blend_color = 4.0f; // SH output is typically < 2.0
    // block size constants
    DEF int block_size_preprocess = 128;
    DEF int visibility_block_size = 256;
    // 256 matches joint-Adam quant block size so blockIdx.x == splat_block for bounds.
    DEF int block_size_preprocess_backward = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int block_size_adam_step_invisible = 256;
    DEF int tile_width = 16;
    DEF int tile_height = 16;
    DEF int block_size_blend = tile_width * tile_height;
    // Four warps with two pixels per thread cover all 8×4 sub-tiles without a
    // 256-thread block.
    DEF int block_size_blend_backward = 128;
    // Forward blend matches the backward shape: 128 threads, 4 warps x 2 pixels
    // cover the 16x16 tile, halving shared-memory traffic per pixel.
    DEF int block_size_blend_forward = 128;
    // Forward blend fetch batch (multiple of 32, <= block_size_blend_forward).
    // Capped by the 128-thread dual-pixel block: one thread loads one splat.
    DEF int blend_batch_size = 128;
    // Backward blend fetch batch (warp-cull reverse walk). Cap at block_size (128).
    DEF int blend_backward_batch_size = 128;
    // Warp sub-tile geometry for forward/backward blend culling (32 threads = 1 warp).
    DEF int warp_subtile_width = 8;
    DEF int warp_subtile_height = 4;

    // SH coefficient swizzle (vksplat float4 layout): 32 primitives per block, 12 float4 slots
    // per primitive packing 15 float3 SH-rest coefficients with 3 floats of tail padding.
    DEF unsigned int sh_reorder_size = 32u;
    DEF unsigned int sh_rest_float4_per_primitive = 12u;
    DEF unsigned int sh_max_coeffs_rest = 15u;
} // namespace fast_lfs::rasterization::config

namespace config = fast_lfs::rasterization::config;

namespace fast_lfs::rasterization {
    // --max-std-dev (LFQPLCY): the splat cut of the game renderer fpv-render (shaders/draw.wgsl, project.wgsl; gpu/mod.rs:
    // max_std_dev sqrt(5), min_alpha 0.5/255, dilation 0.3 px^2): a pixel takes part in a splat only when z^2 / 2 <=
    // half_z2_max, z = its Mahalanobis distance in the splat's dilated 2D covariance (the conic the blend evaluates), and
    // alpha = opacity exp(-z^2 / 2) >= min_alpha. Tile bounds, the exact tile count and the warp / sub-tile culls use
    // min(half_z2_max, log(opacity / min_alpha)), so the screen radius shrinks with the cut. The defaults are every
    // kernel's original test (no geometric cut, alpha >= 1/255). Set process-wide by set_splat_cutoff (rasterization_api.h).
    struct SplatCutoffParams {
        float half_z2_max = 3.0e38f;
        float min_alpha = config::min_alpha_threshold;
        float min_alpha_rcp = config::min_alpha_threshold_rcp;
        // --max-std-dev-distance (LFQPLCYKD): the cap on log(opacity / min_alpha) as a power threshold (half_z2_max with
        // plain --max-std-dev, no cap in the distance mode) and the distance mode itself (1: every bound takes
        // min(P_eff(beta), half_z2_max) as its d^2/2 radius; P_eff = kshape::power_threshold of the UNCAPPED P). The per-pixel
        // tests are the same in both modes: E = 1/2 d^beta <= P (the alpha floor) and d^2/2 <= half_z2_max.
        float pt_cap = 3.0e38f;
        int distance_cap = 0;
    };
    inline constexpr float kGameMinAlpha = 0.5f / 255.0f; // fpv-render gpu/mod.rs min_alpha
} // namespace fast_lfs::rasterization

#undef DEF
