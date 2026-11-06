/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // --kernel-shape (training/kernel_shape.hpp), on the device without a host sync.
    //  kernel_shape_adam: one Adam step of the raw shape r [N] (beta = 1 + 7 sigmoid(r - ln 6)) from the rasterizer's
    //    dL/dbeta grad [N] (chained to r by d beta / d r), state m, v interleaved in adam [N, 2]. Rows whose gradient is
    //    exactly 0 (not drawn by this view, or soft-deleted) are left alone, as the fused Adam leaves the splats a view
    //    does not draw; every row's gradient is zeroed for the next view.
    //  kernel_shape_histogram: counts of beta over kBetaHistBins bins (edges kBetaHistEdges), skipping rows with
    //    deleted[i] != 0 (nullptr = none), plus the sum of beta (double) and the count of rows with beta exactly 2.
    struct KernelShapeAdam {
        float lr = 0.0f;
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-15f;
        float bias1 = 1.0f, bias2 = 1.0f; // 1 - beta1^t, 1 - beta2^t
    };

    constexpr int kBetaHistBins = 12;
    inline constexpr float kBetaHistEdges[kBetaHistBins + 1] = {1.0f, 1.25f, 1.5f, 1.75f, 1.95f, 2.05f, 2.5f, 3.0f,
                                                                4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    struct KernelShapeHistogram {
        unsigned long long bins[kBetaHistBins] = {};
        unsigned long long exactly_two = 0;
        unsigned long long count = 0;
        double sum = 0.0;
        float min = 0.0f, max = 0.0f;
    };

    LFS_CUDA_API void kernel_shape_adam(float* raw, float* adam, float* grad, std::int64_t n, const KernelShapeAdam& a,
                                        cudaStream_t stream);

    // Blocking (copies the result back): for the periodic log and the tests.
    LFS_CUDA_API KernelShapeHistogram kernel_shape_histogram(const float* raw, const std::uint8_t* deleted, std::int64_t n,
                                                             cudaStream_t stream);

    // beta [n] from raw [n] (PLY export, the tests).
    LFS_CUDA_API void kernel_shape_beta(const float* raw, float* beta, std::int64_t n, cudaStream_t stream);

} // namespace lfs::training::kernels
