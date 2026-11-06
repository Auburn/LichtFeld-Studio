/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Dynamic rolling shutter for the FastGS preprocess kernels (forward and backward). See FastGSRollingShutter in
// rasterization_config.h for the model. Each splat is captured at ONE time: its centre is projected at the
// mid-readout pose, the row it lands on gives a time, the centre is re-projected at the pose of that time, and so
// on (rs.iters re-projections); then the splat's world-to-camera rows become those of the pose at its time,
//   W' = E W,  b' = E (b - d),  E = exp(-[r(t)]x),  r(t) = omega t + alpha t^2 / 2,  d(t) = vel t + acc t^2 / 2,
// so its centre, covariance, depth and normal all come from that pose; colours (SH) keep the mid-readout centre.
// The backward holds each splat's time fixed (dt/d(splat, motion) is O(image speed x tau / H), < 1% of the
// gradient) and adds dL/d(omega, alpha, vel, acc, tau) from dL/dW' and dL/db':
//   dL/dE = G_W W^T + G_b (b - d)^T;  with dE = [delta]x E:  dL/ddelta = vee(M - M^T), M = dL/dE E^T;
//   phi = -r: dL/dphi = J_l(phi)^T dL/ddelta (exact SO(3) left Jacobian), dL/dr = -dL/dphi;  dL/dd = -E^T G_b;
//   dL/domega = t dL/dr, dL/dalpha = t^2/2 dL/dr, dL/dvel = t dL/dd, dL/dacc = t^2/2 dL/dd,
//   dL/dtau = s (dL/dr . (omega + alpha t) + dL/dd . (vel + acc t)),  t = s tau.
// With zero motion E = I and d = 0 exactly, so the rows (and every result) equal the global-shutter ones bit for bit.

#include "fisheye_projection.cuh"
#include "helper_math.h"
#include "rasterization_config.h"

namespace fast_lfs::rasterization::kernels::rolling_shutter {

    struct Motion {
        float3 w;   // angular velocity (camera axes, rad/s)
        float3 al;  // angular acceleration (rad/s^2)
        float3 v;   // linear velocity (camera axes, units/s)
        float3 a;   // linear acceleration (units/s^2)
        float tau;  // readout time, top row to bottom row (s)
    };

    __device__ __forceinline__ Motion load_motion(const FastGSRollingShutter& rs) {
        Motion m;
        const float* p = rs.motion;
        m.w = make_float3(__ldg(p + 0), __ldg(p + 1), __ldg(p + 2));
        m.al = make_float3(__ldg(p + 3), __ldg(p + 4), __ldg(p + 5));
        m.v = make_float3(__ldg(p + 6), __ldg(p + 7), __ldg(p + 8));
        m.a = make_float3(__ldg(p + 9), __ldg(p + 10), __ldg(p + 11));
        m.tau = __ldg(rs.readout);
        return m;
    }

    // E = exp([phi]x) (Rodrigues; a series near 0; exactly the identity at 0), as rows.
    __device__ __forceinline__ void exp_so3(const float3 phi, float3 E[3]) {
        const float th2 = dot(phi, phi);
        float A, B;
        if (th2 < 1e-8f) {
            A = 1.0f - th2 * (1.0f / 6.0f);
            B = 0.5f - th2 * (1.0f / 24.0f);
        } else {
            const float th = sqrtf(th2);
            A = sinf(th) / th;
            B = (1.0f - cosf(th)) / th2;
        }
        // I + A S + B S^2 with S = [phi]x and S^2 = phi phi^T - th2 I
        const float x = phi.x, y = phi.y, z = phi.z;
        E[0] = make_float3(1.0f + B * (x * x - th2), B * x * y - A * z, B * x * z + A * y);
        E[1] = make_float3(B * x * y + A * z, 1.0f + B * (y * y - th2), B * y * z - A * x);
        E[2] = make_float3(B * x * z - A * y, B * y * z + A * x, 1.0f + B * (z * z - th2));
    }

    // The pose at time t relative to the mid-readout pose: camera-space p(t) = E (p_mid - d).
    __device__ __forceinline__ void pose_at(const Motion& m, const float t, float3 E[3], float3& d) {
        const float h = 0.5f * t * t;
        const float3 r = m.w * t + m.al * h;
        exp_so3(make_float3(-r.x, -r.y, -r.z), E);
        d = m.v * t + m.a * h;
    }

    __device__ __forceinline__ float3 rotate(const float3 E[3], const float3 p) {
        return make_float3(dot(E[0], p), dot(E[1], p), dot(E[2], p));
    }

    // The sensor row (this render's pixels) where camera-space p is imaged; false if p is not in front.
    template <bool FISHEYE, bool RADIAL = false>
    __device__ __forceinline__ bool row_of(const FastGSLens& lens, const float fy, const float cy, const float3 p, float& v) {
        if (!(p.z > 0.0f))
            return false;
        if constexpr (FISHEYE) {
            const auto tm = fisheye::terms_lens<false, RADIAL>(lens, p.x * p.x + p.y * p.y, p.z);
            v = fy * p.y * tm.h + cy;
        } else {
            v = fy * p.y / p.z + cy;
        }
        return true;
    }

    // Solves the splat's capture time and replaces rows (in: the mid-readout world-to-camera rows; out: the rows at the
    // splat's time). Returns s, the row fraction (t = s tau, s in [-1/2, 1/2]).
    template <bool FISHEYE, bool RADIAL = false>
    __device__ __forceinline__ float splat_rows(const FastGSRollingShutter& rs, const Motion& m, const FastGSLens& lens,
                                                const float fy, const float cy, const float render_height,
                                                const float3 mean, float4 rows[3], const uint primitive_idx) {
        const float3 pm = make_float3(
            rows[0].x * mean.x + rows[0].y * mean.y + rows[0].z * mean.z + rows[0].w,
            rows[1].x * mean.x + rows[1].y * mean.y + rows[1].z * mean.z + rows[1].w,
            rows[2].x * mean.x + rows[2].y * mean.y + rows[2].z * mean.z + rows[2].w);
        float s = 0.0f;
        if (rs.times != nullptr && rs.times_fixed != 0) {
            s = rs.times[primitive_idx];
        } else {
            const float H = rs.sensor_height > 0.0f ? rs.sensor_height : render_height;
            float3 p = pm;
            for (int k = 0; k < rs.iters; ++k) {
                float v;
                if (!row_of<FISHEYE, RADIAL>(lens, fy, cy, p, v))
                    break;
                s = fminf(fmaxf((v + rs.row_offset) / H - 0.5f, -0.5f), 0.5f);
                if (k + 1 < rs.iters) {
                    float3 E[3], d;
                    pose_at(m, s * m.tau, E, d);
                    p = rotate(E, pm - d);
                }
            }
        }
        float3 E[3], d;
        pose_at(m, s * m.tau, E, d);
        const float3 b = make_float3(rows[0].w - d.x, rows[1].w - d.y, rows[2].w - d.z);
        const float4 r0 = rows[0], r1 = rows[1], r2 = rows[2];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const float3 e = E[i];
            rows[i] = make_float4(
                e.x * r0.x + e.y * r1.x + e.z * r2.x,
                e.x * r0.y + e.y * r1.y + e.z * r2.y,
                e.x * r0.z + e.y * r1.z + e.z * r2.z,
                e.x * b.x + e.y * b.y + e.z * b.z);
        }
        return s;
    }

    // Adds this splat's dL/d(omega[3], alpha[3], vel[3], acc[3], tau) to acc13, given dL/d(rows at its time):
    // GW = dL/dW' (rows), Gb = dL/db', and the mid-readout rows.
    __device__ __forceinline__ void motion_grad(const Motion& m, const float s, const float4 mid[3],
                                                const float3 GW[3], const float3 Gb, float acc13[13]) {
        const float t = s * m.tau;
        float3 E[3], d;
        pose_at(m, t, E, d);
        const float3 u = make_float3(mid[0].w - d.x, mid[1].w - d.y, mid[2].w - d.z);
        const float3 W0 = make_float3(mid[0].x, mid[0].y, mid[0].z);
        const float3 W1 = make_float3(mid[1].x, mid[1].y, mid[1].z);
        const float3 W2 = make_float3(mid[2].x, mid[2].y, mid[2].z);
        const float Gbv[3] = {Gb.x, Gb.y, Gb.z};
        float GE[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            GE[i][0] = dot(GW[i], W0) + Gbv[i] * u.x;
            GE[i][1] = dot(GW[i], W1) + Gbv[i] * u.y;
            GE[i][2] = dot(GW[i], W2) + Gbv[i] * u.z;
        }
        // M = GE E^T
        float M[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int j = 0; j < 3; ++j)
                M[i][j] = GE[i][0] * E[j].x + GE[i][1] * E[j].y + GE[i][2] * E[j].z;
        const float3 g = make_float3(M[2][1] - M[1][2], M[0][2] - M[2][0], M[1][0] - M[0][1]);
        // phi = -r; J_l(phi)^T g = g - a (phi x g) + b (phi x (phi x g))
        const float h = 0.5f * t * t;
        const float3 r = m.w * t + m.al * h;
        const float3 phi = make_float3(-r.x, -r.y, -r.z);
        const float th2 = dot(phi, phi);
        float ca, cb;
        if (th2 < 1e-6f) {
            ca = 0.5f - th2 * (1.0f / 24.0f);
            cb = (1.0f / 6.0f) - th2 * (1.0f / 120.0f);
        } else {
            const float th = sqrtf(th2);
            ca = (1.0f - cosf(th)) / th2;
            cb = (th - sinf(th)) / (th2 * th);
        }
        const float3 pg = cross(phi, g);
        const float3 gphi = g - ca * pg + cb * cross(phi, pg);
        const float3 gr = make_float3(-gphi.x, -gphi.y, -gphi.z);
        // dL/dd = -E^T Gb
        const float3 gd = make_float3(
            -(E[0].x * Gb.x + E[1].x * Gb.y + E[2].x * Gb.z),
            -(E[0].y * Gb.x + E[1].y * Gb.y + E[2].y * Gb.z),
            -(E[0].z * Gb.x + E[1].z * Gb.y + E[2].z * Gb.z));
        acc13[0] += t * gr.x;
        acc13[1] += t * gr.y;
        acc13[2] += t * gr.z;
        acc13[3] += h * gr.x;
        acc13[4] += h * gr.y;
        acc13[5] += h * gr.z;
        acc13[6] += t * gd.x;
        acc13[7] += t * gd.y;
        acc13[8] += t * gd.z;
        acc13[9] += h * gd.x;
        acc13[10] += h * gd.y;
        acc13[11] += h * gd.z;
        acc13[12] += s * (dot(gr, m.w + m.al * t) + dot(gd, m.v + m.a * t));
    }

    // Block-wide sum of 13 values per thread, added atomically to out[13] (one atomic per value per block). Every thread of
    // the block must call it (it synchronises the block).
    __device__ __forceinline__ void block_add13(float v[13], float* out) {
        __shared__ float part[32][13];
        const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
        for (int i = 0; i < 13; ++i) {
            float x = v[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                x += __shfl_down_sync(0xffffffffu, x, o);
            if (lane == 0)
                part[warp][i] = x;
        }
        __syncthreads();
        if (warp == 0) {
            const unsigned n_warps = (blockDim.x + 31u) >> 5;
#pragma unroll
            for (int i = 0; i < 13; ++i) {
                float x = lane < n_warps ? part[lane][i] : 0.0f;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    x += __shfl_down_sync(0xffffffffu, x, o);
                if (lane == 0 && x != 0.0f)
                    atomicAdd(out + i, x);
            }
        }
        __syncthreads();
    }

} // namespace fast_lfs::rasterization::kernels::rolling_shutter
