/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Radial lens table (--lens-radial FILE; LFRL fork, 6 Nov 2026): a free radial profile theta -> theta_d that replaces the
// OPENCV_FISHEYE k-polynomial of the native-fisheye path. Header-only, host + device, no dependencies, so the FastGS
// projection (fisheye_projection.cuh), MRNF blob seeding (blob_seeding.hpp) and the host code all evaluate exactly the
// same function. The file format and the builder are abq/lens_radial.py (Videos/splat/work); the Python tools evaluate
// the same table the same way (pose_opt_fish.py FC_LENS_RADIAL, abq/fish_proj_rl_cuda.cu).
//
// The lens: theta_d = theta P(u), u = theta^2 (OPENCV_FISHEYE: P = 1 + k1 u + k2 u^2 + k3 u^3 + k4 u^4). P is tabulated on
// nodes uniform in theta (theta_i = i dtheta, i = 0 .. n-1, theta_{n-1} = theta_end); on each interval [u_i, u_{i+1}] it is
// the quintic Hermite interpolant in u of P, dP/du, d2P/du2 at the two nodes (C2 in u), stored as the coefficients of a
// polynomial in t = (u - u_i) inv_h_i. Beyond theta_end theta_d is linear in theta (theta_d = A + B theta, C1).

#include <cmath>

#ifdef __CUDACC__
#define LFS_RL_HD __host__ __device__ __forceinline__
#else
#define LFS_RL_HD inline
#endif

namespace lfs::core {

    // A POD view of a loaded table (the pointer is host or device memory, as the caller needs).
    struct RadialLensView {
        const float* seg = nullptr; // [n - 1][8]: u0, inv_h, c0 .. c5
        int n = 0;                  // nodes
        float inv_dth = 0.0f;       // 1 / dtheta
        float th_end = 0.0f;        // theta of the last node (radians)
        float A = 0.0f, B = 0.0f;   // theta_d = A + B theta beyond th_end
    };

    // P(u), dP/du, d2P/du2 at u = theta^2 >= 0.
    LFS_RL_HD void radial_lens_P(const RadialLensView& L, const float u, float& P, float& P1, float& P2) {
        const float th = sqrtf(fmaxf(u, 0.0f));
        if (th < L.th_end) {
            int i = static_cast<int>(th * L.inv_dth);
            i = i < 0 ? 0 : (i > L.n - 2 ? L.n - 2 : i);
            const float* s = L.seg + 8 * i;
#ifdef __CUDA_ARCH__
            const float u0 = __ldg(s), ih = __ldg(s + 1), c0 = __ldg(s + 2), c1 = __ldg(s + 3), c2 = __ldg(s + 4),
                        c3 = __ldg(s + 5), c4 = __ldg(s + 6), c5 = __ldg(s + 7);
#else
            const float u0 = s[0], ih = s[1], c0 = s[2], c1 = s[3], c2 = s[4], c3 = s[5], c4 = s[6], c5 = s[7];
#endif
            const float t = (u - u0) * ih;
            P = c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
            P1 = (c1 + t * (2.0f * c2 + t * (3.0f * c3 + t * (4.0f * c4 + t * (5.0f * c5))))) * ih;
            P2 = (2.0f * c2 + t * (6.0f * c3 + t * (12.0f * c4 + t * (20.0f * c5)))) * ih * ih;
        } else {
            const float it = 1.0f / th;
            const float it2 = it * it;
            P = L.A * it + L.B;
            P1 = -0.5f * L.A * it2 * it;
            P2 = 0.75f * L.A * it2 * it2 * it;
        }
    }

    // theta_d(theta) and its slope d theta_d / d theta.
    LFS_RL_HD float radial_lens_thetad(const RadialLensView& L, const float th) {
        float P, P1, P2;
        radial_lens_P(L, th * th, P, P1, P2);
        return th * P;
    }

    LFS_RL_HD float radial_lens_slope(const RadialLensView& L, const float th) {
        float P, P1, P2;
        const float u = th * th;
        radial_lens_P(L, u, P, P1, P2);
        return P + 2.0f * u * P1;
    }

} // namespace lfs::core
