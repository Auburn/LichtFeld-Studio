/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Evaluates the fisheye projection maths the preprocess kernels use (fisheye_projection.cuh) at
// given camera-space points, on the GPU, so a host-side double-precision reference can check it.

#include "fisheye_projection.cuh"
#include "rasterization_api.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
    // in: per point p (3), G1 (3), G2 (3); out: per point mean2d (2), J row 1 (3), J row 2 (3),
    // d(G1 . J1 + G2 . J2)/dp through J (3), h, h_q, h_z, h_qq, h_qz, h_zz (6) = 17 floats
    template <bool RADIAL>
    __global__ void fisheye_selftest_cu(
        const fast_lfs::rasterization::FastGSLens lens,
        const float fx, const float fy, const float cx, const float cy,
        const float* __restrict__ in, float* __restrict__ out, const int n) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        namespace fe = fast_lfs::rasterization::kernels::fisheye;
        const float* a = in + 9 * i;
        const float3 p = make_float3(a[0], a[1], a[2]);
        const float3 g1 = make_float3(a[3], a[4], a[5]);
        const float3 g2 = make_float3(a[6], a[7], a[8]);
        const auto t = fe::terms_lens<true, RADIAL>(lens, p.x * p.x + p.y * p.y, p.z);
        float2 m;
        float3 j1, j2;
        fe::project(t, p, fx, fy, cx, cy, m, j1, j2);
        const float3 jb = fe::jacobian_backward(t, p, fx, fy, g1, g2);
        float* o = out + 17 * i;
        o[0] = m.x;
        o[1] = m.y;
        o[2] = j1.x;
        o[3] = j1.y;
        o[4] = j1.z;
        o[5] = j2.x;
        o[6] = j2.y;
        o[7] = j2.z;
        o[8] = jb.x;
        o[9] = jb.y;
        o[10] = jb.z;
        o[11] = t.h;
        o[12] = t.hq;
        o[13] = t.hz;
        o[14] = t.hqq;
        o[15] = t.hqz;
        o[16] = t.hzz;
    }
} // namespace

namespace fast_lfs::rasterization {

    void fisheye_projection_selftest(
        const FastGSLens& lens, const float fx, const float fy, const float cx, const float cy,
        const float* host_in, const int n, float* host_out) {
        if (n <= 0)
            return;
        float* d_in = nullptr;
        float* d_out = nullptr;
        auto check = [](cudaError_t e, const char* what) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("fisheye_projection_selftest: ") + what + ": " + cudaGetErrorString(e));
        };
        check(cudaMalloc(&d_in, sizeof(float) * 9 * n), "cudaMalloc");
        check(cudaMalloc(&d_out, sizeof(float) * 17 * n), "cudaMalloc");
        check(cudaMemcpy(d_in, host_in, sizeof(float) * 9 * n, cudaMemcpyHostToDevice), "cudaMemcpy");
        if (lens.radial.seg != nullptr) // --lens-radial (LFRL)
            fisheye_selftest_cu<true><<<(n + 127) / 128, 128>>>(lens, fx, fy, cx, cy, d_in, d_out, n);
        else
            fisheye_selftest_cu<false><<<(n + 127) / 128, 128>>>(lens, fx, fy, cx, cy, d_in, d_out, n);
        check(cudaGetLastError(), "launch");
        check(cudaMemcpy(host_out, d_out, sizeof(float) * 17 * n, cudaMemcpyDeviceToHost), "cudaMemcpy");
        cudaFree(d_in);
        cudaFree(d_out);
    }

} // namespace fast_lfs::rasterization
