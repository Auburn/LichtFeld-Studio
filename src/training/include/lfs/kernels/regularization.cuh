/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused scale regularization: loss = weight * mean(exp(scaling_raw))
     *
     * Computes loss and gradients in a single pass using warp-level reductions.
     *
     * @param params Input parameters (scaling_raw) [N*3]
     * @param param_grads Output gradients (accumulated) [N*3]
     * @param loss_out Device buffer for scalar loss (1 element)
     * @param temp_buffer Temporary buffer for partial sums (min(1024, (n+255)/256) elements)
     * @param n Number of elements
     * @param weight Regularization weight
     * @param stream CUDA stream
     */
    void launch_fused_scale_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused opacity regularization: loss = weight * mean(sigmoid(opacity_raw))
     *
     * Computes loss and gradients in a single pass using warp-level reductions.
     *
     * @param params Input parameters (opacity_raw) [N]
     * @param param_grads Output gradients (accumulated) [N]
     * @param loss_out Device buffer for scalar loss (1 element)
     * @param temp_buffer Temporary buffer for partial sums (min(1024, (n+255)/256) elements)
     * @param n Number of elements
     * @param weight Regularization weight
     * @param stream CUDA stream
     */
    void launch_fused_opacity_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused opacity binarization: loss = weight * mean(o * (1 - o)), o = sigmoid(opacity_raw)
     *
     * Pushes opacities toward 0 or 1. Gradient: (weight / n) * (1 - 2 o) * o * (1 - o).
     * Same buffers and reductions as launch_fused_opacity_regularization; param_grads may be null
     * (loss only).
     */
    void launch_fused_opacity_binarize(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
