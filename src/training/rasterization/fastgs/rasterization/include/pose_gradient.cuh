/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Per-camera pose refinement (--pose-opt; FastGSPose in rasterization_config.h): the preprocess backward's gradient of
// the loss with respect to a LEFT perturbation of the world-to-camera transform (W | b) it was given,
//   W(delta) = exp(-[dr]x) W,  b(delta) = exp(-[dr]x) (b - dd),  camera centre c(delta) = c + W^T dd,
// at delta = 0 (dr a rotation in radians, dd the camera centre's move in scene units, both in the camera's axes).
// A splat is drawn through rows W' = E W, b' = E (b - d) (its rolling-shutter time; E = I, d = 0 without the rolling
// shutter, held fixed as for the motion). With G_W = dL/dW' (rows: through its centre W' x + b', its covariance
// J W' Sigma W'^T J^T and its normal W' n) and G_b = dL/db':
//   dW' = -E [dr]x W  ->  dL/ddr += -vee(N),  N = W G_W^T E,  vee(N) = (N12 - N21, N20 - N02, N01 - N10)
//   db' = -E [dr]x b - E dd  ->  dL/ddr += (E^T G_b) x b,  dL/ddd += -E^T G_b
// and through its colour's view direction (mean - c): dL/ddd += -W g_col, g_col = dL/dmean from the colour.
// training/kernels/pose_refinement_math.hpp chains it to the camera's offset.

#include "helper_math.h"

namespace fast_lfs::rasterization::kernels::pose {

    __device__ __forceinline__ void splat_grad(const float3 E[3], const float4 mid[3], const float3 GW[3], const float3 Gb,
                                               const float3 g_col, float acc6[6]) {
        // Ge = E^T G_b
        const float3 Ge = E[0] * Gb.x + E[1] * Gb.y + E[2] * Gb.z;
        const float3 W0 = make_float3(mid[0].x, mid[0].y, mid[0].z);
        const float3 W1 = make_float3(mid[1].x, mid[1].y, mid[1].z);
        const float3 W2 = make_float3(mid[2].x, mid[2].y, mid[2].z);
        // P_il = W_i . G_W,l
        const float P00 = dot(W0, GW[0]), P01 = dot(W0, GW[1]), P02 = dot(W0, GW[2]);
        const float P10 = dot(W1, GW[0]), P11 = dot(W1, GW[1]), P12 = dot(W1, GW[2]);
        const float P20 = dot(W2, GW[0]), P21 = dot(W2, GW[1]), P22 = dot(W2, GW[2]);
        // N_ij = sum_l P_il E_lj (E rows E[l] = (E_l0, E_l1, E_l2))
        const float N01 = P00 * E[0].y + P01 * E[1].y + P02 * E[2].y;
        const float N02 = P00 * E[0].z + P01 * E[1].z + P02 * E[2].z;
        const float N10 = P10 * E[0].x + P11 * E[1].x + P12 * E[2].x;
        const float N12 = P10 * E[0].z + P11 * E[1].z + P12 * E[2].z;
        const float N20 = P20 * E[0].x + P21 * E[1].x + P22 * E[2].x;
        const float N21 = P20 * E[0].y + P21 * E[1].y + P22 * E[2].y;
        const float3 b = make_float3(mid[0].w, mid[1].w, mid[2].w);
        const float3 gr = make_float3(N21 - N12, N02 - N20, N10 - N01) + cross(Ge, b);
        const float3 gd = make_float3(-Ge.x - dot(W0, g_col), -Ge.y - dot(W1, g_col), -Ge.z - dot(W2, g_col));
        acc6[0] += gr.x;
        acc6[1] += gr.y;
        acc6[2] += gr.z;
        acc6[3] += gd.x;
        acc6[4] += gd.y;
        acc6[5] += gd.z;
    }

    // Block-wide sum of 6 values per thread, added atomically to out[6] (one atomic per value per block). Every thread of
    // the block must call it (it synchronises the block).
    __device__ __forceinline__ void block_add6(float v[6], float* out) {
        __shared__ float part[32][6];
        const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
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
            for (int i = 0; i < 6; ++i) {
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

} // namespace fast_lfs::rasterization::kernels::pose
