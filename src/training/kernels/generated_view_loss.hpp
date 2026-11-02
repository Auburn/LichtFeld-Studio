/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::training::kernels {

    // Alignment-tolerant loss for generated (e.g. diffusion-made) training views, which invent fine detail that
    // shifts slightly from view to view. Instead of matching pixels it matches the low-pass image and the local
    // amount of texture:
    //   L = loss_weight * ( lowpass_weight * mean_c,p m_p |G(r)_c,p - G(t)_c,p|
    //                     + texture_weight * mean_p   m_p |s(r)_p - s(t)_p|
    //                     + pixel_weight   * mean_c,p m_p |r_c,p - t_c,p| )
    // r the render, t the target, m the per-pixel weight (mask / ROI; 1 without one). The means are weighted by m:
    // sum(m * term) / (C * sum(m)) for the per-channel terms and sum(m * term) / sum(m) for the texture term.
    // G is a normalized (masked) Gaussian blur of standard deviation lowpass_sigma per channel:
    //   G(x)_p = sum_q k_pq m_q x_q / sum_q k_pq m_q,
    // so pixels outside the mask never leak in. s is the local standard deviation of the luma (Rec. 709) in a
    // (2 texture_radius + 1)^2 box window, with the same masked normalization:
    //   s_p = sqrt(max(E_p[y^2] - E_p[y]^2, GENERATED_VIEW_VAR_EPS)).
    // Both filters are separable and zero-padded, i.e. truncated at the image border and renormalized.

    inline constexpr int GENERATED_VIEW_MAX_RADIUS = 32;
    inline constexpr float GENERATED_VIEW_MAX_SIGMA = 10.0f;
    inline constexpr float GENERATED_VIEW_VAR_EPS = 1e-6f;
    inline constexpr float GENERATED_VIEW_LUMA[3] = {0.2126f, 0.7152f, 0.0722f};

    struct GeneratedViewLossParams {
        float lowpass_sigma = 2.0f;
        int texture_radius = 3;
        float lowpass_weight = 1.0f;
        float texture_weight = 1.0f;
        float pixel_weight = 0.0f;
        float loss_weight = 1.0f;
    };

    // Running sums for the log (device float[GENERATED_VIEW_STAT_COUNT], accumulated per call, zeroed by the owner).
    enum GeneratedViewStat : int {
        GENERATED_VIEW_STAT_VIEWS = 0,      // number of generated views
        GENERATED_VIEW_STAT_TOTAL = 1,      // the weighted loss L
        GENERATED_VIEW_STAT_LOWPASS = 2,    // unweighted low-pass L1
        GENERATED_VIEW_STAT_TEXTURE = 3,    // unweighted texture term
        GENERATED_VIEW_STAT_PIXEL = 4,      // unweighted pixel L1 (computed even when pixel_weight is 0)
        GENERATED_VIEW_STAT_STD_RENDER = 5, // m-weighted mean of s(render)
        GENERATED_VIEW_STAT_STD_TARGET = 6, // m-weighted mean of s(target)
        GENERATED_VIEW_STAT_REAL_LOSS = 7,  // photometric loss of the real views (generated_view_stats_add_real)
        GENERATED_VIEW_STAT_REAL_VIEWS = 8, // number of real views
        GENERATED_VIEW_STAT_COUNT = 9
    };

    struct LFS_CUDA_API GeneratedViewLossWorkspace {
        lfs::core::Tensor planes;  // [12, H, W] filter inputs / outputs, then the backward seeds
        lfs::core::Tensor tmp;     // [12, H, W] between the two separable passes
        lfs::core::Tensor grad;    // [3, H, W] gradient w.r.t. the render
        lfs::core::Tensor scalars; // [8] per-call device sums
        lfs::core::Tensor loss;    // [1]
        void ensure_size(size_t height, size_t width);
    };

    struct GeneratedViewLossResult {
        lfs::core::Tensor loss; // [1] device scalar (a workspace view)
        lfs::core::Tensor grad; // shaped like the render (a workspace view, valid until the next call)
    };

    // render: Float32 [3,H,W] or [1,3,H,W]; target: the same shape, Float32 or UInt8 (/255); pixel_weight: invalid
    // (all 1) or [H,W] / [1,H,W] Float32 (weights) or UInt8 / Bool (nonzero = 1). stats: null or a CUDA Float32
    // [GENERATED_VIEW_STAT_COUNT] tensor the call adds to (re-homed to the loss stream). No host synchronization.
    LFS_CUDA_API GeneratedViewLossResult generated_view_loss(
        const lfs::core::Tensor& render,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& pixel_weight,
        const GeneratedViewLossParams& params,
        GeneratedViewLossWorkspace& workspace,
        lfs::core::Tensor* stats = nullptr);

    // Adds a real view's photometric loss (device scalar) to stats[REAL_LOSS] and 1 to stats[REAL_VIEWS].
    LFS_CUDA_API void generated_view_stats_add_real(const lfs::core::Tensor& loss, lfs::core::Tensor& stats);

} // namespace lfs::training::kernels
