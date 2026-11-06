/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// --kernel-shape (LFQPLCYK): a per-splat trainable kernel shape (a generalized Gaussian), written from the maths.
//
//   alpha = o exp(-1/2 d^beta),  d = the Mahalanobis distance in the (dilated) 2D conic the rasterizer already uses.
//
// The rasterizer's quadric is s = 1/2 d^2 (FastGS's "sigma_over_2"), so the exponent is E = 1/2 (2 s)^(beta / 2).
// beta = 2 is exactly today's kernel (E = s, taken as is: the same bits). beta > 2: flat top, sharp edge; beta < 2: a
// peak with long tails.
//
// Parameter: an unconstrained per-splat raw value r, beta = 1 + 7 sigmoid(r - ln 6) in (1, 8), r = 0 -> beta = 2 exactly
// (core/kernel_shape_map.hpp).
//
// Extent: alpha >= 1/255  <=>  1/2 d^beta <= P = ln(255 o)  <=>  d^2 <= (2P)^(2/beta) =: 2 P_eff. Every bound the
// rasterizer derives from P (tile bounds, the exact tile count, the pixel box, the warp sub-tile test) takes P_eff in
// its place; the per-pixel tests compare E with the unchanged P.
#pragma once

#include "core/kernel_shape_map.hpp"
#include <cuda_runtime.h>

namespace fast_lfs::rasterization::kshape {

    using lfs::core::kshape::beta_from_raw;
    using lfs::core::kshape::dbeta_draw;

    // P_eff: the d^2/2 radius of the alpha = 1/255 contour (P itself for beta = 2). A 1e-5 relative slack keeps the
    // bounds conservative against the per-pixel __powf (whose error is far smaller).
    __device__ __forceinline__ float power_threshold(const float P, const float beta) {
        if (beta == 2.0f || !(P > 0.0f))
            return P;
        return 0.5f * exp2f(log2f(2.0f * P) * (2.0f / beta)) * 1.00001f;
    }

    // The exponent E(s) = 1/2 (2 s)^(beta/2) for the quadric s = d^2/2 >= 0 (s itself, bit for bit, at beta = 2).
    __device__ __forceinline__ float exponent(const float s, const float beta) {
        if (beta == 2.0f)
            return s;
        return 0.5f * __powf(2.0f * s, 0.5f * beta);
    }

    // dE/ds and dE/dbeta at the exponent E = exponent(s, beta):
    //   dE/ds    = (beta/2) (2s)^(beta/2 - 1) = beta E / (2 s)        (1 exactly at beta = 2)
    //   dE/dbeta = 1/2 (2s)^(beta/2) ln(2s)/2 = E ln(2s) / 2  = 1/2 d^beta ln d
    // so d alpha / d beta = -alpha dE/dbeta = -1/2 alpha d^beta ln d. At d -> 0 (s = 0): dE/dbeta -> 0 (d^beta ln d -> 0)
    // and dE/ds is taken as 0 there: its product with ds/d(mean, conic) ~ d^(beta - 1) or d^beta, which -> 0 for beta > 1
    // (and is bounded at beta = 1).
    __device__ __forceinline__ void exponent_grads(const float s, const float beta, const float E, float& dE_ds, float& dE_dbeta) {
        if (!(s > 0.0f)) {
            dE_ds = beta == 2.0f ? 1.0f : 0.0f;
            dE_dbeta = 0.0f;
            return;
        }
        dE_ds = beta == 2.0f ? 1.0f : 0.5f * beta * E / s;
        dE_dbeta = 0.5f * E * __logf(2.0f * s);
    }

} // namespace fast_lfs::rasterization::kshape
