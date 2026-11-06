/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// --loss-ycc (LFQPLCY): see ycc_loss.hpp.

#include "ycc_loss.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {

        // Cb, Cr as linear combinations of R', G', B' (BT.709): Cb = (B - Y) / 1.8556, Cr = (R - Y) / 1.5748
        constexpr float kCbR = -kYccKr / kYccCbDen, kCbG = -kYccKg / kYccCbDen, kCbB = (1.0f - kYccKb) / kYccCbDen;
        constexpr float kCrR = (1.0f - kYccKr) / kYccCrDen, kCrG = -kYccKg / kYccCrDen, kCrB = -kYccKb / kYccCrDen;

        __device__ __forceinline__ float load_px(const void* p, const bool u8, const size_t i) {
            return u8 ? static_cast<float>(static_cast<const uint8_t*>(p)[i]) * (1.0f / 255.0f) : static_cast<const float*>(p)[i];
        }

        __device__ __forceinline__ float load_w(const void* w, const int kind, const size_t i) {
            if (kind == kYccWeightFloat)
                return static_cast<const float*>(w)[i];
            if (kind == kYccWeightU8)
                return static_cast<float>(static_cast<const uint8_t*>(w)[i]);
            return 1.0f;
        }

        __device__ __forceinline__ float sgn(const float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

        // block (bx, by) -> weight w_b and the box-averaged chroma differences (pred - target)
        __device__ __forceinline__ float block_terms(const float* pred, const void* target, const bool tu8, const void* w,
                                                     const int wk, const int H, const int W, const int border, const int bx,
                                                     const int by, float& dcb, float& dcr) {
            const int x0 = 2 * bx, y0 = 2 * by;
            float wb = 0.0f;
            if (wk == kYccWeightNone) {
                if (border > 0 && (x0 < border || y0 < border || x0 + 1 > W - 1 - border || y0 + 1 > H - 1 - border))
                    return 0.0f;
                wb = 1.0f;
            } else {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float v = load_w(w, wk, static_cast<size_t>(y0 + (k >> 1)) * W + (x0 + (k & 1)));
                    if (!(v > 0.0f))
                        return 0.0f;   // only fully kept blocks
                    s += v;
                }
                wb = 0.25f * s;
            }
            const size_t HW = static_cast<size_t>(H) * W;
            float pr = 0.0f, pg = 0.0f, pb = 0.0f, tr = 0.0f, tg = 0.0f, tb = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const size_t i = static_cast<size_t>(y0 + (k >> 1)) * W + (x0 + (k & 1));
                pr += pred[i];
                pg += pred[HW + i];
                pb += pred[2 * HW + i];
                tr += load_px(target, tu8, i);
                tg += load_px(target, tu8, HW + i);
                tb += load_px(target, tu8, 2 * HW + i);
            }
            const float dr = 0.25f * (pr - tr), dg = 0.25f * (pg - tg), db = 0.25f * (pb - tb);
            dcb = kCbR * dr + kCbG * dg + kCbB * db;
            dcr = kCrR * dr + kCrG * dg + kCrB * db;
            return wb;
        }

        __global__ void luma_kernel(const void* __restrict__ rgb, const bool u8, const int HW, float* __restrict__ y) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= HW)
                return;
            y[i] = kYccKr * load_px(rgb, u8, i) + kYccKg * load_px(rgb, u8, static_cast<size_t>(HW) + i) +
                   kYccKb * load_px(rgb, u8, 2 * static_cast<size_t>(HW) + i);
        }

        __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                x += __shfl_down_sync(0xffffffffu, x, o);
            return x;
        }

        __global__ void chroma_sums_kernel(const float* __restrict__ pred, const void* __restrict__ target, const bool tu8,
                                           const void* __restrict__ w, const int wk, const int H, const int W,
                                           const int border, float* __restrict__ sums) {
            const int BW = W / 2, BH = H / 2;
            const int b = blockIdx.x * blockDim.x + threadIdx.x;
            float a = 0.0f, ws = 0.0f;
            if (b < BW * BH) {
                float dcb, dcr;
                const float wb = block_terms(pred, target, tu8, w, wk, H, W, border, b % BW, b / BW, dcb, dcr);
                if (wb > 0.0f) {
                    a = wb * (fabsf(dcb) + fabsf(dcr));
                    ws = wb;
                }
            }
            a = warp_sum(a);
            ws = warp_sum(ws);
            if ((threadIdx.x & 31u) == 0u) {
                if (a != 0.0f)
                    atomicAdd(sums + 0, a);
                if (ws != 0.0f)
                    atomicAdd(sums + 1, ws);
            }
        }

        __global__ void chroma_finalize_kernel(const float* __restrict__ sums, const float scale, float* __restrict__ loss,
                                               float* __restrict__ coef) {
            const float c = scale / (2.0f * fmaxf(sums[1], 1e-12f));
            coef[0] = c;
            loss[0] = c * sums[0];
        }

        __global__ void chroma_backward_kernel(const float* __restrict__ pred, const void* __restrict__ target, const bool tu8,
                                               const void* __restrict__ w, const int wk, const int H, const int W,
                                               const int border, const float* __restrict__ coef, float* __restrict__ grad) {
            const int BW = W / 2, BH = H / 2;
            const int b = blockIdx.x * blockDim.x + threadIdx.x;
            if (b >= BW * BH)
                return;
            const int bx = b % BW, by = b / BW;
            float dcb, dcr;
            const float wb = block_terms(pred, target, tu8, w, wk, H, W, border, bx, by, dcb, dcr);
            if (!(wb > 0.0f))
                return;
            // d/dpred of w_b (|dCb| + |dCr|): each of the 4 pixels gets 1/4 of the block's chroma derivative
            const float c = coef[0] * wb * 0.25f;
            const float scb = sgn(dcb) * c, scr = sgn(dcr) * c;
            const float gr = kCbR * scb + kCrR * scr, gg = kCbG * scb + kCrG * scr, gb = kCbB * scb + kCrB * scr;
            const size_t HW = static_cast<size_t>(H) * W;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const size_t i = static_cast<size_t>(2 * by + (k >> 1)) * W + (2 * bx + (k & 1));
                grad[i] += gr;
                grad[HW + i] += gg;
                grad[2 * HW + i] += gb;
            }
        }

        __global__ void luma_grad_kernel(const float* __restrict__ gy, const float scale, const int HW, float* __restrict__ g,
                                         const bool accumulate) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= HW)
                return;
            const float v = scale * gy[i];
            if (accumulate) {
                g[i] += kYccKr * v;
                g[static_cast<size_t>(HW) + i] += kYccKg * v;
                g[2 * static_cast<size_t>(HW) + i] += kYccKb * v;
            } else {
                g[i] = kYccKr * v;
                g[static_cast<size_t>(HW) + i] = kYccKg * v;
                g[2 * static_cast<size_t>(HW) + i] = kYccKb * v;
            }
        }

        __global__ void combine_kernel(const float* __restrict__ luma, const float scale, const float* __restrict__ chroma,
                                       float* __restrict__ total) {
            total[0] = scale * luma[0] + chroma[0];
        }

    } // namespace

    void ycc_combine_loss(const float* luma_loss, const float scale, const float* chroma_loss, float* total, cudaStream_t stream) {
        stream = resolve_stream(stream);
        combine_kernel<<<1, 1, 0, stream>>>(luma_loss, scale, chroma_loss, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ycc.combine");
    }

    void ycc_luma(const void* rgb, const bool rgb_u8, const int H, const int W, float* y, cudaStream_t stream) {
        stream = resolve_stream(stream);
        const int HW = H * W;
        luma_kernel<<<(HW + 255) / 256, 256, 0, stream>>>(rgb, rgb_u8, HW, y);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ycc.luma");
    }

    void ycc_chroma_sums(const float* pred, const void* target, const bool target_u8, const void* weight, const int weight_kind,
                         const int H, const int W, const int border, float* sums, cudaStream_t stream) {
        stream = resolve_stream(stream);
        const int nb = (W / 2) * (H / 2);
        if (nb <= 0)
            return;
        chroma_sums_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(pred, target, target_u8, weight, weight_kind, H, W, border, sums);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ycc.chroma_sums");
    }

    void ycc_chroma_finalize(const float* sums, const float scale, float* loss_out, float* coef_out, cudaStream_t stream) {
        stream = resolve_stream(stream);
        chroma_finalize_kernel<<<1, 1, 0, stream>>>(sums, scale, loss_out, coef_out);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ycc.chroma_finalize");
    }

    void ycc_chroma_backward(const float* pred, const void* target, const bool target_u8, const void* weight,
                             const int weight_kind, const int H, const int W, const int border, const float* coef, float* grad_rgb,
                             cudaStream_t stream) {
        stream = resolve_stream(stream);
        const int nb = (W / 2) * (H / 2);
        if (nb <= 0)
            return;
        chroma_backward_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(pred, target, target_u8, weight, weight_kind, H, W, border,
                                                                      coef, grad_rgb);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ycc.chroma_backward");
    }

    void ycc_luma_grad_to_rgb(const float* grad_y, const float scale, const int HW, float* grad_rgb, const bool accumulate,
                              cudaStream_t stream) {
        stream = resolve_stream(stream);
        luma_grad_kernel<<<(HW + 255) / 256, 256, 0, stream>>>(grad_y, scale, HW, grad_rgb, accumulate);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.ycc.luma_grad");
    }

} // namespace lfs::training::kernels
