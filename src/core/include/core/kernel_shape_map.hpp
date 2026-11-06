/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// --kernel-shape (LFQPLCYK): the per-splat shape parameter map, shared by the host (PLY import / export, checkpoints)
// and the device (the FastGS kernels, training/kernels/kernel_shape_update.cu).
//
// The kernel: alpha = o exp(-1/2 d^beta), d = the Mahalanobis distance in the rasterizer's 2D conic.
// The parameter: an unconstrained raw r per splat, beta = 1 + 7 sigmoid(r - ln 6), in (1, 8). r = 0 gives beta = 2
// EXACTLY (today's Gaussian): the form below is 2 + 6 (1 - e^-r) / (1 + 6 e^-r) for r >= 0 and 2 + 6 (e^r - 1) / (e^r + 6)
// for r < 0, whose numerator is exactly 0 at r = 0 (e^0 = 1 exactly, also with fast-math exp).
// d beta / d r = 7 sigmoid (1 - sigmoid) = (beta - 1) (8 - beta) / 7 (6/7 at beta = 2).
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define LFS_KSHAPE_HD __host__ __device__ __forceinline__
#else
#define LFS_KSHAPE_HD inline
#endif

namespace lfs::core::kshape {

    constexpr float kBetaMin = 1.0f;
    constexpr float kBetaMax = 8.0f;
    constexpr const char* kPlyProperty = "shape"; // the PLY vertex property: beta itself

    LFS_KSHAPE_HD float beta_from_raw(const float r) {
        if (r >= 0.0f) {
            const float u = expf(-r);
            return 2.0f + 6.0f * (1.0f - u) / (1.0f + 6.0f * u);
        }
        const float t = expf(r);
        return 2.0f + 6.0f * (t - 1.0f) / (t + 6.0f);
    }

    LFS_KSHAPE_HD float dbeta_draw(const float beta) {
        return (beta - kBetaMin) * (kBetaMax - beta) * (1.0f / 7.0f);
    }

    // The inverse, for PLY import (host): r = ln(6 (beta - 1) / (8 - beta)), beta clamped just inside (1, 8);
    // beta = 2 -> r = 0 exactly. Non-finite beta -> 0 (the Gaussian).
    LFS_KSHAPE_HD float raw_from_beta(float beta) {
        if (beta == 2.0f || !(beta == beta) || beta > 3.0e38f || beta < -3.0e38f)
            return 0.0f;
        beta = fminf(fmaxf(beta, kBetaMin + 1.0e-6f), kBetaMax - 1.0e-5f);
        return logf(6.0f * (beta - kBetaMin) / (kBetaMax - beta));
    }

} // namespace lfs::core::kshape
