/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --max-std-dev (LFQPLCY): this translation unit's copy of the splat cut (SplatCutoffParams, rasterization_config.h),
// read by the preprocess, the blend kernels and the backward. Its initial value is the original test; forward.cu and
// backward.cu upload the process-wide setting (set_splat_cutoff). Other units (pop_scores.cu) keep the initial value.

#include "rasterization_config.h"
#include <cuda_runtime.h>

namespace fast_lfs::rasterization {
    namespace {
        __constant__ SplatCutoffParams c_splat_cutoff = {3.0e38f, config::min_alpha_threshold, config::min_alpha_threshold_rcp};

        // power threshold (z^2 / 2) of a splat: the alpha contour log(opacity / min_alpha), capped by the cut. Off: the
        // original value exactly (a NaN passes through as before).
        __device__ __forceinline__ float splat_power_threshold(const float opacity) {
            const float pt = logf(opacity * c_splat_cutoff.min_alpha_rcp);
            return pt > c_splat_cutoff.half_z2_max ? c_splat_cutoff.half_z2_max : pt;
        }

        [[maybe_unused]] cudaError_t upload_splat_cutoff_this_unit(const SplatCutoffParams& p) {
            return cudaMemcpyToSymbol(c_splat_cutoff, &p, sizeof(p));
        }
    } // namespace
} // namespace fast_lfs::rasterization
