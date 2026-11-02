/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::training::kernels {

    // --loss-weight-dir: per-pixel photometric loss weights.
    //
    // Bilinear resample of a row-major [src_height, src_width] float map (device memory) to a [height, width]
    // Float32 CUDA tensor: pixel centres map to pixel centres (align_corners = false, as OpenCV INTER_LINEAR),
    // clamped at the edges. A map already at the output size is copied unchanged. `output` is (re)allocated
    // when its shape differs; the kernel runs on output's stream.
    LFS_CUDA_API void resample_loss_weight_map(const float* source, int src_width, int src_height,
                                               lfs::core::Tensor& output, int width, int height);

    // output = weight * base * inside, per pixel of a [H, W] map. base is read the way the masked photometric loss
    // reads a mask (UInt8 / Bool: 1 where != 0; Float32: the value; invalid tensor: 1). inside is 0 within `border`
    // pixels of the image edge (as the unmasked fused L1 + SSIM's valid padding, only when H and W are both larger
    // than 2 * border) and 1 elsewhere; border 0 = no border. output may be base itself (in place) and is
    // (re)allocated when it is not a [H, W] Float32 tensor.
    LFS_CUDA_API void combine_loss_weight(const lfs::core::Tensor& weight, const lfs::core::Tensor& base,
                                          int border, lfs::core::Tensor& output);

} // namespace lfs::training::kernels
