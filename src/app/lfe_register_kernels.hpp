/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// LFQPLCYR: the image loss of `LichtFeld-Studio lfe-register` (src/app/lfe_render_tool.cpp), a CUDA port of the
// Python registration refiner's loss (abq/pose_opt_fish_fast.py with FC_FAST=1, FC_COLOR=affine; abq/rs_render_fast.py
// colour_fit_ns / ssim_map_b / _loss_one), so that registering test cameras through LichtFeld's rasterizer optimises
// the same objective as the gsplat path:
//   colour affine  x = A r + b  (A 3x3, b 3) fitted by weighted least squares on r[::4, ::4] (kept pixels), detached;
//   loss = 0.8 * sum_kept mean_c |x - p| / n + 0.2 * (1 - sum_kept S(p) / n),  n = number of kept pixels;
//   S = the channel mean of the SSIM map over the WHOLE image (11-tap Gaussian, sigma 1.5, reflect padding of 5,
//   C1 = 0.01^2, C2 = 0.03^2), masked afterwards (masked-out pixels still enter the windows, as in the Python code).
// Not used by the trainer.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::app::regk {

    // rgb u8 [H, W, 3] (RGB) -> float [3, H, W] / 255
    void photo_to_chw(const uint8_t* rgb_hwc, float* out_chw, int H, int W, cudaStream_t stream);

    // acc[28] (device doubles, caller zeros): acc[0..15] = sum_w X^T X (row-major 4x4, X = [r0, r1, r2, 1]),
    // acc[16..27] = sum_w X^T p (row-major 4x3), over the pixels (y % stride == 0, x % stride == 0) with mask != 0.
    void affine_normal_equations(const float* r_chw, const float* p_chw, const uint8_t* mask, int H, int W, int stride,
                                 double* acc, cudaStream_t stream);

    struct LossWorkspace {
        int H = 0, W = 0;
        float* x = nullptr;  // [3, H, W]  A r + b
        float* h = nullptr;  // [15, H, W] horizontal pass / adjoint scratch
        float* d = nullptr;  // [9, H, W]  per-pixel SSIM partials (weighted)
        float* t = nullptr;  // [9, H, W]  adjoint vertical pass
        double* sums = nullptr; // [2] device: sum_kept mean_c |x - p|, sum_kept S
        void ensure(int height, int width);
        ~LossWorkspace();
    };

    // sol[12] = the least-squares solution (row-major 4x3: x_c = sum_k sol[k*3+c] r_k + sol[9+c]).
    // Writes dLoss/dr into grad_r_chw [3, H, W]; ws.sums gets the two sums (zeroed here). n_kept = number of kept pixels.
    void loss_and_grad(const float* r_chw, const float* p_chw, const uint8_t* mask, int H, int W, const float sol[12],
                       double n_kept, LossWorkspace& ws, float* grad_r_chw, cudaStream_t stream);

} // namespace lfs::app::regk
