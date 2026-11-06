/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// LFQPLCYR: lfe-register's image loss (see lfe_register_kernels.hpp). Plain CUDA, no dependency on the trainer.

#include "lfe_register_kernels.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace lfs::app::regk {

    namespace {
        __constant__ float c_g[11];
        __constant__ float c_sol[12];
        bool g_window_set = false;

        void check(cudaError_t e, const char* what) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("lfe-register loss: ") + what + ": " + cudaGetErrorString(e));
        }

        void set_window() {
            if (g_window_set)
                return;
            double w[11], s = 0.0;
            for (int k = 0; k < 11; ++k) {
                w[k] = std::exp(-((k - 5) * (k - 5)) / (2.0 * 1.5 * 1.5));
                s += w[k];
            }
            float g[11];
            for (int k = 0; k < 11; ++k)
                g[k] = static_cast<float>(w[k] / s);
            check(cudaMemcpyToSymbol(c_g, g, sizeof(g)), "window");
            g_window_set = true;
        }

        __device__ __forceinline__ int refl(int k, int n) { return k < 0 ? -k : (k >= n ? 2 * (n - 1) - k : k); }

        __device__ __forceinline__ double warp_sum(double v) {
            for (int o = 16; o > 0; o >>= 1)
                v += __shfl_down_sync(0xffffffffu, v, o);
            return v;
        }

        __global__ void k_photo(const uint8_t* __restrict__ rgb, float* __restrict__ out, int HW) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= HW)
                return;
            for (int c = 0; c < 3; ++c)
                out[c * HW + i] = static_cast<float>(rgb[3 * i + c]) / 255.0f;
        }

        __global__ void k_affine(const float* __restrict__ r, const float* __restrict__ p, const uint8_t* __restrict__ m,
                                 int H, int W, int stride, int nsx, int nsy, double* __restrict__ acc) {
            const int t = blockIdx.x * blockDim.x + threadIdx.x;
            double v[28];
#pragma unroll
            for (int k = 0; k < 28; ++k)
                v[k] = 0.0;
            if (t < nsx * nsy) {
                const int y = (t / nsx) * stride, x = (t % nsx) * stride;
                const int i = y * W + x, HW = H * W;
                if (m[i]) {
                    const double X[4] = {r[i], r[HW + i], r[2 * HW + i], 1.0};
                    const double P[3] = {p[i], p[HW + i], p[2 * HW + i]};
#pragma unroll
                    for (int a = 0; a < 4; ++a) {
#pragma unroll
                        for (int b = 0; b < 4; ++b)
                            v[a * 4 + b] = X[a] * X[b];
#pragma unroll
                        for (int c = 0; c < 3; ++c)
                            v[16 + a * 3 + c] = X[a] * P[c];
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < 28; ++k) {
                const double s = warp_sum(v[k]);
                if ((threadIdx.x & 31) == 0 && s != 0.0)
                    atomicAdd(acc + k, s);
            }
        }

        __global__ void k_transform(const float* __restrict__ r, float* __restrict__ x, int HW) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= HW)
                return;
            const float r0 = r[i], r1 = r[HW + i], r2 = r[2 * HW + i];
#pragma unroll
            for (int c = 0; c < 3; ++c)
                x[c * HW + i] = c_sol[c] * r0 + c_sol[3 + c] * r1 + c_sol[6 + c] * r2 + c_sol[9 + c];
        }

        // horizontal Gaussian pass (reflect padding) of x, y, x^2, y^2, xy per channel -> h[(c*5 + k)]
        __global__ void k_hpass(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ h, int H, int W) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int HW = H * W;
            if (idx >= HW)
                return;
            const int i = idx / W, j = idx % W;
            for (int c = 0; c < 3; ++c) {
                const float* xr = x + c * HW + i * W;
                const float* yr = y + c * HW + i * W;
                float sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
#pragma unroll
                for (int b = 0; b < 11; ++b) {
                    const int jj = refl(j + b - 5, W);
                    const float g = c_g[b], xv = xr[jj], yv = yr[jj];
                    sx += g * xv;
                    sy += g * yv;
                    sxx += g * xv * xv;
                    syy += g * yv * yv;
                    sxy += g * xv * yv;
                }
                float* o = h + (c * 5) * HW + idx;
                o[0] = sx;
                o[HW] = sy;
                o[2 * HW] = sxx;
                o[3 * HW] = syy;
                o[4 * HW] = sxy;
            }
        }

        // vertical pass, SSIM map, L1, sums and the weighted partials d[(c*3 + {mu_x, E[x^2], E[xy]})]
        __global__ void k_vpass(const float* __restrict__ x, const float* __restrict__ y, const uint8_t* __restrict__ m,
                                const float* __restrict__ h, float* __restrict__ d, int H, int W, float coef_s,
                                double* __restrict__ sums) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int HW = H * W;
            double l1 = 0.0, ss = 0.0;
            if (idx < HW) {
                const int i = idx / W, j = idx % W;
                const bool kept = m[idx] != 0;
                const float C1 = 0.01f * 0.01f, C2 = 0.03f * 0.03f;
                float smap = 0.0f, l1p = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    float mx = 0, my = 0, ex2 = 0, ey2 = 0, exy = 0;
                    const float* hc = h + (c * 5) * HW;
#pragma unroll
                    for (int a = 0; a < 11; ++a) {
                        const int o = refl(i + a - 5, H) * W + j;
                        const float g = c_g[a];
                        mx += g * hc[o];
                        my += g * hc[HW + o];
                        ex2 += g * hc[2 * HW + o];
                        ey2 += g * hc[3 * HW + o];
                        exy += g * hc[4 * HW + o];
                    }
                    const float sxx = ex2 - mx * mx, syy = ey2 - my * my, sxy = exy - mx * my;
                    const float A1 = 2.0f * mx * my + C1, A2 = 2.0f * sxy + C2;
                    const float B1 = mx * mx + my * my + C1, B2 = sxx + syy + C2;
                    const float map = (A1 * A2) / (B1 * B2);
                    smap += map;
                    const float xv = x[c * HW + idx], yv = y[c * HW + idx];
                    l1p += fabsf(xv - yv);
                    const float w = kept ? coef_s : 0.0f;
                    const float dmu = (2.0f * my * A2 - 2.0f * my * A1) / (B1 * B2) - map * (2.0f * mx / B1 - 2.0f * mx / B2);
                    const float dex2 = -map / B2;
                    const float dexy = 2.0f * A1 / (B1 * B2);
                    d[(c * 3 + 0) * HW + idx] = w * dmu;
                    d[(c * 3 + 1) * HW + idx] = w * dex2;
                    d[(c * 3 + 2) * HW + idx] = w * dexy;
                }
                if (kept) {
                    l1 = static_cast<double>(l1p / 3.0f);
                    ss = static_cast<double>(smap / 3.0f);
                }
            }
            l1 = warp_sum(l1);
            ss = warp_sum(ss);
            if ((threadIdx.x & 31) == 0) {
                if (l1 != 0.0)
                    atomicAdd(sums, l1);
                if (ss != 0.0)
                    atomicAdd(sums + 1, ss);
            }
        }

        // adjoint of "reflect-pad by 5, then valid 11-tap convolution" along rows: t(q, j) = sum over the output rows i
        // that read source row q (directly or through the reflection) of g[a] d(i, j)
        __device__ __forceinline__ float adj_gather(const float* __restrict__ col, int q, int n, int step) {
            float s = 0.0f;
#pragma unroll
            for (int a = 0; a < 11; ++a) {
                const float g = c_g[a];
                const int i1 = q - a + 5;
                if (i1 >= 0 && i1 < n)
                    s += g * col[i1 * step];
                if (q > 0) {
                    const int i2 = -q - a + 5;
                    if (i2 >= 0 && i2 < n)
                        s += g * col[i2 * step];
                }
                if (q < n - 1) {
                    const int i3 = 2 * (n - 1) - q - a + 5;
                    if (i3 >= 0 && i3 < n)
                        s += g * col[i3 * step];
                }
            }
            return s;
        }

        __global__ void k_vadj(const float* __restrict__ d, float* __restrict__ t, int H, int W) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int HW = H * W;
            if (idx >= HW)
                return;
            const int q = idx / W, j = idx % W;
            for (int k = 0; k < 9; ++k)
                t[k * HW + idx] = adj_gather(d + k * HW + j, q, H, W);
        }

        __global__ void k_hadj(const float* __restrict__ t, const float* __restrict__ x, const float* __restrict__ y,
                               const uint8_t* __restrict__ m, float* __restrict__ grad_r, int H, int W, float coef_l1) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int HW = H * W;
            if (idx >= HW)
                return;
            const int i = idx / W, q = idx % W;
            const float wl = m[idx] ? coef_l1 : 0.0f;
            float gx[3];
            for (int c = 0; c < 3; ++c) {
                const float amu = adj_gather(t + (c * 3 + 0) * HW + i * W, q, W, 1);
                const float ae2 = adj_gather(t + (c * 3 + 1) * HW + i * W, q, W, 1);
                const float aexy = adj_gather(t + (c * 3 + 2) * HW + i * W, q, W, 1);
                const float xv = x[c * HW + idx], yv = y[c * HW + idx];
                const float diff = xv - yv;
                const float sgn = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                gx[c] = amu + 2.0f * xv * ae2 + yv * aexy + wl * sgn;
            }
#pragma unroll
            for (int k = 0; k < 3; ++k)
                grad_r[k * HW + idx] = c_sol[k * 3 + 0] * gx[0] + c_sol[k * 3 + 1] * gx[1] + c_sol[k * 3 + 2] * gx[2];
        }

        inline int blocks(int n) { return (n + 255) / 256; }
    } // namespace

    void photo_to_chw(const uint8_t* rgb_hwc, float* out_chw, int H, int W, cudaStream_t stream) {
        k_photo<<<blocks(H * W), 256, 0, stream>>>(rgb_hwc, out_chw, H * W);
        check(cudaGetLastError(), "photo");
    }

    void affine_normal_equations(const float* r_chw, const float* p_chw, const uint8_t* mask, int H, int W, int stride,
                                 double* acc, cudaStream_t stream) {
        const int nsx = (W + stride - 1) / stride, nsy = (H + stride - 1) / stride;
        k_affine<<<blocks(nsx * nsy), 256, 0, stream>>>(r_chw, p_chw, mask, H, W, stride, nsx, nsy, acc);
        check(cudaGetLastError(), "affine");
    }

    void LossWorkspace::ensure(int height, int width) {
        if (height == H && width == W)
            return;
        cudaFree(x);
        cudaFree(h);
        cudaFree(d);
        cudaFree(t);
        if (!sums)
            check(cudaMalloc(&sums, 2 * sizeof(double)), "alloc");
        H = height;
        W = width;
        const size_t hw = static_cast<size_t>(H) * W * sizeof(float);
        check(cudaMalloc(&x, 3 * hw), "alloc");
        check(cudaMalloc(&h, 15 * hw), "alloc");
        check(cudaMalloc(&d, 9 * hw), "alloc");
        check(cudaMalloc(&t, 9 * hw), "alloc");
    }

    LossWorkspace::~LossWorkspace() {
        cudaFree(x);
        cudaFree(h);
        cudaFree(d);
        cudaFree(t);
        cudaFree(sums);
    }

    void loss_and_grad(const float* r_chw, const float* p_chw, const uint8_t* mask, int H, int W, const float sol[12],
                       double n_kept, LossWorkspace& ws, float* grad_r_chw, cudaStream_t stream) {
        if (H < 6 || W < 6)
            throw std::runtime_error("lfe-register loss: image too small");
        set_window();
        ws.ensure(H, W);
        check(cudaMemcpyToSymbolAsync(c_sol, sol, 12 * sizeof(float), 0, cudaMemcpyHostToDevice, stream), "sol");
        check(cudaMemsetAsync(ws.sums, 0, 2 * sizeof(double), stream), "zero");
        const int HW = H * W;
        const double n = n_kept < 1.0 ? 1.0 : n_kept;
        const float coef_s = static_cast<float>(-0.2 / (3.0 * n)); // loss = ... + 0.2 (1 - S): dLoss/dS = -0.2
        const float coef_l1 = static_cast<float>(0.8 / (3.0 * n));
        k_transform<<<blocks(HW), 256, 0, stream>>>(r_chw, ws.x, HW);
        k_hpass<<<blocks(HW), 256, 0, stream>>>(ws.x, p_chw, ws.h, H, W);
        k_vpass<<<blocks(HW), 256, 0, stream>>>(ws.x, p_chw, mask, ws.h, ws.d, H, W, coef_s, ws.sums);
        k_vadj<<<blocks(HW), 256, 0, stream>>>(ws.d, ws.t, H, W);
        k_hadj<<<blocks(HW), 256, 0, stream>>>(ws.t, ws.x, p_chw, mask, grad_r_chw, H, W, coef_l1);
        check(cudaGetLastError(), "loss kernels");
    }

} // namespace lfs::app::regk
