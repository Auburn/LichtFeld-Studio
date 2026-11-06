/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --loss-ycc (LFQPLCY): the photometric loss in Y'CbCr. The footage is 4:2:0 video (HEVC Main, BT.709, tv range, chroma at
// half resolution); our decode upsamples its chroma to full-resolution R'G'B'. These kernels take exactly the image the RGB
// loss sees (after the cubemap composite and PPISP) and its target, and give
//   luma    Y  = 0.2126 R' + 0.7152 G' + 0.0722 B'          (BT.709, full range on the encoded values: the exact inverse of
//   chroma  Cb = (B' - Y) / 1.8556,  Cr = (R' - Y) / 1.5748  the decode's matrix after its tv -> pc range expansion)
// The luma loss (L1 + D-SSIM at full resolution) reuses the trainer's existing fused kernels on Y images; here: the
// conversion, the half-resolution chroma L1 and the mapping of the Y gradient back to R'G'B'.
// Chroma: Cb, Cr of render and target are averaged over 2x2 blocks (rows 2i, 2i+1, columns 2j, 2j+1: the band limit of the
// half-resolution chroma both sides share), L_c = sum_b w_b (|dCb_b| + |dCr_b|) / (2 sum_b w_b). Block weight: with a
// pixel-weight map (mask / ROI / loss weights), w_b = the mean of its 4 pixel weights if ALL four are > 0, else 0 (only fully
// kept blocks: no masked pixel's colour enters a block average); without one, 1, except blocks within `border` pixels of
// the image edge (the unmasked loss's valid padding). An odd last row / column is not used by the chroma term.

#include "core/export.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace lfs::training::kernels {

    constexpr float kYccKr = 0.2126f, kYccKg = 0.7152f, kYccKb = 0.0722f;
    constexpr float kYccCbDen = 1.8556f, kYccCrDen = 1.5748f;

    // y[H*W] = luma of rgb [3, H, W] (planar); rgb is float32, or uint8 when rgb_u8 (value / 255)
    LFS_CUDA_API void ycc_luma(const void* rgb, bool rgb_u8, int H, int W, float* y, cudaStream_t stream);

    // weight kinds for the chroma term
    enum YccWeightKind : int { kYccWeightNone = 0, kYccWeightFloat = 1, kYccWeightU8 = 2 };

    // sums[0] += sum_b w_b (|dCb_b| + |dCr_b|), sums[1] += sum_b w_b  (caller zeros sums[2])
    LFS_CUDA_API void ycc_chroma_sums(const float* pred, const void* target, bool target_u8, const void* weight, int weight_kind,
                                      int H, int W, int border, float* sums, cudaStream_t stream);

    // loss_out[0] = scale * sums[0] / (2 max(sums[1], eps));  coef_out[0] = scale / (2 max(sums[1], eps))  (device, no sync)
    LFS_CUDA_API void ycc_chroma_finalize(const float* sums, float scale, float* loss_out, float* coef_out, cudaStream_t stream);

    // grad_rgb [3, H, W] += coef * d/d(pred) of sum_b w_b (|dCb_b| + |dCr_b|)   (each pixel belongs to one block: no atomics)
    LFS_CUDA_API void ycc_chroma_backward(const float* pred, const void* target, bool target_u8, const void* weight,
                                          int weight_kind, int H, int W, int border, const float* coef, float* grad_rgb,
                                          cudaStream_t stream);

    // grad_rgb [3, H, W] (+)= scale * (Kr, Kg, Kb) * grad_y [H, W]   (accumulate = false: overwrite)
    LFS_CUDA_API void ycc_luma_grad_to_rgb(const float* grad_y, float scale, int HW, float* grad_rgb, bool accumulate,
                                           cudaStream_t stream);

    // total[0] = scale * luma_loss[0] + chroma_loss[0]  (device scalars; total may alias neither input)
    LFS_CUDA_API void ycc_combine_loss(const float* luma_loss, float scale, const float* chroma_loss, float* total,
                                       cudaStream_t stream);

} // namespace lfs::training::kernels
