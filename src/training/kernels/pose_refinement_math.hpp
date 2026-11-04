/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --pose-opt: the per-camera pose offset and its gradient chain, shared by the device update kernels
// (pose_refinement_update.cu) and the host checks (src/app/lfe_pose_tool.cpp). Templated on the scalar type.
//
// A camera's refined world-to-camera transform, from its start pose (rows W0 | b0, centre c0 = -W0^T b0) and its offset
// (r, d) in the START camera's axes (r: rotation vector in radians, d: the camera centre's move in scene units):
//   W = E W0,  b = E (b0 - d),  E = exp(-[r]x);   centre c = c0 + W0^T d   (so |d| = the move, |r| = the turn).
// The rasterizer's backward returns g = dL/d(dr, dd) of a LEFT perturbation of the transform it was given,
//   W(delta) = exp(-[dr]x) W,  b(delta) = exp(-[dr]x) (b - dd),
// and this chains it to the offset:  dL/dr = J_l(-r)^T dL/ddr  (exact SO(3) left Jacobian),  dL/dd = E^T dL/ddd.
// With r = d = 0 the transform is the start pose bit for bit (E = I exactly).

#include <cmath>

#if defined(__CUDACC__)
#define LFS_POSE_HD __host__ __device__ __forceinline__
#else
#define LFS_POSE_HD inline
#endif

namespace lfs::training::pose_math {

    // E = exp([phi]x), rows (row-major 3x3); a series near 0, exactly the identity at 0.
    template <class T>
    LFS_POSE_HD void exp_so3(const T phi[3], T E[9]) {
        const T th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
        T A, B;
        if (th2 < T(1e-8)) {
            A = T(1) - th2 * (T(1) / T(6));
            B = T(0.5) - th2 * (T(1) / T(24));
        } else {
            const T th = std::sqrt(th2);
            A = std::sin(th) / th;
            B = (T(1) - std::cos(th)) / th2;
        }
        const T x = phi[0], y = phi[1], z = phi[2];
        E[0] = T(1) + B * (x * x - th2);
        E[1] = B * x * y - A * z;
        E[2] = B * x * z + A * y;
        E[3] = B * x * y + A * z;
        E[4] = T(1) + B * (y * y - th2);
        E[5] = B * y * z - A * x;
        E[6] = B * x * z - A * y;
        E[7] = B * y * z + A * x;
        E[8] = T(1) + B * (z * z - th2);
    }

    // The refined transform: w2c0 (row-major 4x4 start pose), c0 (start centre), off = (r[3], d[3]) ->
    // w2c (row-major 4x4) and the centre c.
    template <class T>
    LFS_POSE_HD void compose(const T w2c0[16], const T c0[3], const T off[6], T w2c[16], T c[3]) {
        const T phi[3] = {-off[0], -off[1], -off[2]};
        T E[9];
        exp_so3(phi, E);
        const T u[3] = {w2c0[3] - off[3], w2c0[7] - off[4], w2c0[11] - off[5]};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                w2c[i * 4 + j] = E[i * 3] * w2c0[j] + E[i * 3 + 1] * w2c0[4 + j] + E[i * 3 + 2] * w2c0[8 + j];
            w2c[i * 4 + 3] = E[i * 3] * u[0] + E[i * 3 + 1] * u[1] + E[i * 3 + 2] * u[2];
        }
        for (int j = 0; j < 4; ++j)
            w2c[12 + j] = w2c0[12 + j];
        for (int j = 0; j < 3; ++j)
            c[j] = c0[j] + (w2c0[j] * off[3] + w2c0[4 + j] * off[4] + w2c0[8 + j] * off[5]);
    }

    // g = dL/d(dr[3], dd[3]) of the left perturbation at the offset `off` -> dL/d(off).
    template <class T>
    LFS_POSE_HD void chain(const T off[6], const T g[6], T out[6]) {
        // J_l(phi)^T gr = gr - a (phi x gr) + b (phi x (phi x gr)),  phi = -r
        const T phi[3] = {-off[0], -off[1], -off[2]};
        const T th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
        T a, b;
        if (th2 < T(1e-6)) {
            a = T(0.5) - th2 * (T(1) / T(24));
            b = T(1) / T(6) - th2 * (T(1) / T(120));
        } else {
            const T th = std::sqrt(th2);
            a = (T(1) - std::cos(th)) / th2;
            b = (th - std::sin(th)) / (th2 * th);
        }
        const T pg[3] = {phi[1] * g[2] - phi[2] * g[1], phi[2] * g[0] - phi[0] * g[2], phi[0] * g[1] - phi[1] * g[0]};
        const T ppg[3] = {phi[1] * pg[2] - phi[2] * pg[1], phi[2] * pg[0] - phi[0] * pg[2], phi[0] * pg[1] - phi[1] * pg[0]};
        for (int k = 0; k < 3; ++k)
            out[k] = g[k] - a * pg[k] + b * ppg[k];
        // dL/dd = E^T dL/ddd
        T E[9];
        exp_so3(phi, E);
        for (int k = 0; k < 3; ++k)
            out[3 + k] = E[k] * g[3] + E[3 + k] * g[4] + E[6 + k] * g[5];
    }

} // namespace lfs::training::pose_math
