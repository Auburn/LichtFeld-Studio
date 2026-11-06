/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// OPENCV_FISHEYE projection of splat centres for the FastGS preprocess kernels (forward and
// backward). See FastGSLens in rasterization_config.h for the model.
//
// With q = x^2 + y^2 and theta = atan2(sqrt(q), z), the projection is u = fx x h + cx,
// v = fy y h + cy with h(q, z) = theta_d / rho. h is smooth in (q, z) and finite on the axis, so the
// Jacobian and the Hessian of (x h, y h) are written in h and its partial derivatives in q and z,
// with no division by rho anywhere:
//   d(x h)/d(x, y, z) = (h + 2 x^2 h_q, 2 x y h_q, x h_z)
//   d(y h)/d(x, y, z) = (2 x y h_q, h + 2 y^2 h_q, y h_z)
// h is homogeneous of degree -1, h = H(w) / z with w = q / z^2 = tan^2 theta, and
//   H(w) = phi(w) P(u),  phi(w) = atan(sqrt w) / sqrt w,  u = theta^2 = w phi^2,
//   P(u) = 1 + k1 u + k2 u^2 + k3 u^3 + k4 u^4 (--lens-radial, LFRL: terms_radial below takes P, dP/du, d2P/du2 from
//   the radial lens table instead, core/radial_lens.h; C2 in u, so the backward's second derivatives stay continuous),
// so every partial derivative of h follows from phi, phi', phi'' (a series near the axis, where the
// closed forms cancel, and the closed forms elsewhere):
//   h = H / z,  h_q = H' / z^3,  h_z = -(H + 2 w H') / z^2,
//   h_qq = H'' / z^5,  h_qz = -(2 w H'' + 3 H') / z^4,  h_zz = (2 H + 10 w H' + 4 w^2 H'') / z^3.
// Valid for z > 0 (theta < 90 degrees); the preprocess culls everything else.

#include "helper_math.h"
#include "rasterization_config.h"

namespace fast_lfs::rasterization::kernels::fisheye {

    // phi(w) = atan(sqrt(w)) / sqrt(w) and its first two derivatives in w.
    __device__ __forceinline__ void phi_terms(const float w, float& phi, float& d1, float& d2) {
        if (w < 0.1f) {
            // phi = sum_n (-1)^n w^n / (2n + 1); through n = 10 the truncation is < 1e-8 for
            // w < 0.1 (phi'' included), where the closed forms lose ~1e-5 to cancellation.
            constexpr float c0 = 1.0f, c1 = -1.0f / 3.0f, c2 = 1.0f / 5.0f, c3 = -1.0f / 7.0f,
                            c4 = 1.0f / 9.0f, c5 = -1.0f / 11.0f, c6 = 1.0f / 13.0f, c7 = -1.0f / 15.0f,
                            c8 = 1.0f / 17.0f, c9 = -1.0f / 19.0f, c10 = 1.0f / 21.0f;
            phi = c0 + w * (c1 + w * (c2 + w * (c3 + w * (c4 + w * (c5 + w * (c6 + w * (c7 + w * (c8 + w * (c9 + w * c10)))))))));
            d1 = c1 + w * (2.0f * c2 + w * (3.0f * c3 + w * (4.0f * c4 + w * (5.0f * c5 + w * (6.0f * c6 + w * (7.0f * c7 + w * (8.0f * c8 + w * (9.0f * c9 + w * (10.0f * c10)))))))));
            d2 = 2.0f * c2 + w * (6.0f * c3 + w * (12.0f * c4 + w * (20.0f * c5 + w * (30.0f * c6 + w * (42.0f * c7 + w * (56.0f * c8 + w * (72.0f * c9 + w * (90.0f * c10))))))));
        } else {
            const float s = sqrtf(w);
            const float a = atanf(s);
            phi = a / s;
            // n = s / (1 + w) - atan(s); phi' = n / (2 s^3); phi'' = -1 / (2 w (1 + w)^2) - 3 n / (4 s^5)
            const float n = s / (1.0f + w) - a;
            const float sw = s * w;
            d1 = n / (2.0f * sw);
            const float opw = 1.0f + w;
            d2 = -1.0f / (2.0f * w * opw * opw) - 3.0f * n / (4.0f * sw * w);
        }
    }

    struct Terms {
        float h, hq, hz;       // h and its first derivatives in q and z
        float hqq, hqz, hzz;   // second derivatives (backward only)
    };

    // h and its derivatives at (q, z), z > 0.
    template <bool SECOND>
    __device__ __forceinline__ Terms terms(const FastGSLens& lens, const float q, const float z) {
        const float iz = 1.0f / z;
        const float iz2 = iz * iz;
        const float w = q * iz2;
        float phi, p1, p2;
        phi_terms(w, phi, p1, p2);
        const float u = w * phi * phi;                    // theta^2
        const float u1 = phi * phi + 2.0f * w * phi * p1; // du/dw
        const float P = 1.0f + u * (lens.k1 + u * (lens.k2 + u * (lens.k3 + u * lens.k4)));
        const float P1 = lens.k1 + u * (2.0f * lens.k2 + u * (3.0f * lens.k3 + u * (4.0f * lens.k4)));
        const float H = phi * P;
        const float H1 = p1 * P + phi * P1 * u1;
        Terms t;
        t.h = H * iz;
        t.hq = H1 * iz2 * iz;
        t.hz = -(H + 2.0f * w * H1) * iz2;
        if constexpr (SECOND) {
            const float P2 = 2.0f * lens.k2 + u * (6.0f * lens.k3 + u * (12.0f * lens.k4));
            const float u2 = 4.0f * phi * p1 + 2.0f * w * p1 * p1 + 2.0f * w * phi * p2;
            const float H2 = p2 * P + 2.0f * p1 * P1 * u1 + phi * (P2 * u1 * u1 + P1 * u2);
            const float iz3 = iz2 * iz;
            const float iz4 = iz2 * iz2;
            t.hqq = H2 * iz4 * iz;
            t.hqz = -(2.0f * w * H2 + 3.0f * H1) * iz4;
            t.hzz = (2.0f * H + 10.0f * w * H1 + 4.0f * w * w * H2) * iz3;
        } else {
            t.hqq = t.hqz = t.hzz = 0.0f;
        }
        return t;
    }

    // --lens-radial (LFRL): terms<> with the radial lens table's P(u), dP/du, d2P/du2 instead of the k-polynomial. Used only
    // by the RADIAL = true kernel instantiations (terms_lens), so terms<> above and every kernel without the option are
    // compiled exactly as before.
    template <bool SECOND>
    __device__ __forceinline__ Terms terms_radial(const FastGSLens& lens, const float q, const float z) {
        const float iz = 1.0f / z;
        const float iz2 = iz * iz;
        const float w = q * iz2;
        float phi, p1, p2;
        phi_terms(w, phi, p1, p2);
        const float u = w * phi * phi;                    // theta^2
        const float u1 = phi * phi + 2.0f * w * phi * p1; // du/dw
        float P, P1, P2;
        lfs::core::radial_lens_P(lens.radial, u, P, P1, P2);
        const float H = phi * P;
        const float H1 = p1 * P + phi * P1 * u1;
        Terms t;
        t.h = H * iz;
        t.hq = H1 * iz2 * iz;
        t.hz = -(H + 2.0f * w * H1) * iz2;
        if constexpr (SECOND) {
            const float u2 = 4.0f * phi * p1 + 2.0f * w * p1 * p1 + 2.0f * w * phi * p2;
            const float H2 = p2 * P + 2.0f * p1 * P1 * u1 + phi * (P2 * u1 * u1 + P1 * u2);
            const float iz3 = iz2 * iz;
            const float iz4 = iz2 * iz2;
            t.hqq = H2 * iz4 * iz;
            t.hqz = -(2.0f * w * H2 + 3.0f * H1) * iz4;
            t.hzz = (2.0f * H + 10.0f * w * H1 + 4.0f * w * w * H2) * iz3;
        } else {
            t.hqq = t.hqz = t.hzz = 0.0f;
        }
        return t;
    }

    // The lens of a kernel instantiation: the radial lens table (RADIAL) or the k-polynomial.
    template <bool SECOND, bool RADIAL>
    __device__ __forceinline__ Terms terms_lens(const FastGSLens& lens, const float q, const float z) {
        if constexpr (RADIAL)
            return terms_radial<SECOND>(lens, q, z);
        else
            return terms<SECOND>(lens, q, z);
    }

    // The projected centre (pixels) and the rows of the Jacobian d(u, v)/d(x, y, z) at camera-space p.
    __device__ __forceinline__ void project(
        const Terms& t, const float3 p,
        const float fx, const float fy, const float cx, const float cy,
        float2& mean2d, float3& j1, float3& j2) {
        mean2d = make_float2(fx * p.x * t.h + cx, fy * p.y * t.h + cy);
        const float xy2 = 2.0f * p.x * p.y * t.hq;
        j1 = make_float3(fx * (t.h + 2.0f * p.x * p.x * t.hq), fx * xy2, fx * p.x * t.hz);
        j2 = make_float3(fy * xy2, fy * (t.h + 2.0f * p.y * p.y * t.hq), fy * p.y * t.hz);
    }

    // dL/dp through the Jacobian's own dependence on p: sum_ij G_ij dJ_ij/dp, with G1, G2 = dL/dJ
    // (rows) and J_1. = fx d(x h)/dp, J_2. = fy d(y h)/dp (Hessians of x h and y h, symmetric).
    __device__ __forceinline__ float3 jacobian_backward(
        const Terms& t, const float3 p, const float fx, const float fy, const float3 G1, const float3 G2) {
        const float x = p.x, y = p.y;
        const float x2 = x * x, y2 = y * y, xy = x * y;
        // Hessian of x h
        const float ua = 6.0f * x * t.hq + 4.0f * x2 * x * t.hqq;     // xx
        const float ub = 2.0f * y * t.hq + 4.0f * x2 * y * t.hqq;     // xy
        const float uc = t.hz + 2.0f * x2 * t.hqz;                    // xz
        const float ud = 2.0f * x * t.hq + 4.0f * x * y2 * t.hqq;     // yy
        const float ue = 2.0f * xy * t.hqz;                           // yz
        const float uf = x * t.hzz;                                   // zz
        // Hessian of y h
        const float va = ub;                                          // xx = 2 y hq + 4 x^2 y hqq
        const float vb = ud;                                          // xy = 2 x hq + 4 x y^2 hqq
        const float vc = ue;                                          // xz = 2 x y hqz
        const float vd = 6.0f * y * t.hq + 4.0f * y2 * y * t.hqq;     // yy
        const float ve = t.hz + 2.0f * y2 * t.hqz;                    // yz
        const float vf = y * t.hzz;                                   // zz
        return make_float3(
            fx * (ua * G1.x + ub * G1.y + uc * G1.z) + fy * (va * G2.x + vb * G2.y + vc * G2.z),
            fx * (ub * G1.x + ud * G1.y + ue * G1.z) + fy * (vb * G2.x + vd * G2.y + ve * G2.z),
            fx * (uc * G1.x + ue * G1.y + uf * G1.z) + fy * (vc * G2.x + ve * G2.y + vf * G2.z));
    }

} // namespace fast_lfs::rasterization::kernels::fisheye
