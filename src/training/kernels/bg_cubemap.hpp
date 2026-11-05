/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --bg-mode cubemap (LFC): a learned environment cubemap behind the splats. See bg_cubemap.cu for the model.
//
// Texels: 6 faces x R x R x RGB, float, interleaved: texel (face f, column i, row j) channel c at ((f * R + j) * R + i) * 3 + c.
// Faces in the usual cube-map order and orientation (OpenGL / Vulkan / D3D / wgpu): 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z;
// a direction d (world axes of the dataset = COLMAP / OpenCV) is looked up on the face of its largest |component| with
//   +X: s = -z/|x|, t = -y/|x|    -X: s =  z/|x|, t = -y/|x|
//   +Y: s =  x/|y|, t =  z/|y|    -Y: s =  x/|y|, t = -z/|y|
//   +Z: s =  x/|z|, t = -y/|z|    -Z: s = -x/|z|, t = -y/|z|
// and s, t in [-1, 1] map to column (s + 1) / 2 * R and row (t + 1) / 2 * R (row 0 at the top of the face image),
// texel centres at half-integers. Bilinear across face edges (a neighbour beyond an edge is the texel of the adjacent
// face under the direction of that virtual texel centre).

#include "core/export.hpp"
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    struct CubemapView {
        int width = 0;  // render pixels
        int height = 0;
        float fx = 0.0f, fy = 0.0f, cx = 0.0f, cy = 0.0f;
        int fisheye = 0;           // 1: OPENCV_FISHEYE, theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
        float k1 = 0.0f, k2 = 0.0f, k3 = 0.0f, k4 = 0.0f;
        float theta_max = 1.5359f; // fisheye: the lens polynomial is inverted on [0, theta_max] (the rasterizer's limit)
        const float* w2c = nullptr;     // device [16]: row-major world-to-camera [R | t] of the (mid-readout) pose
        const float* motion = nullptr;  // device [12] rolling-shutter motion (omega, alpha, vel, acc; camera axes) or null
        const float* readout = nullptr; // device [1] readout time tau (s); required with motion
        float row_offset = 0.0f;        // as FastGSRollingShutter (tiled renders; 0 here)
        float sensor_height = 0.0f;     // 0 = the render height
    };

    // bg_chw [3, H, W] = cube(world direction of each pixel's ray).
    LFS_CUDA_API void cubemap_render_background(const float* texels, int res, const CubemapView& view, float* bg_chw,
                                                cudaStream_t stream);

    // grad_texels += sum over pixels of (1 - alpha) * dL/d(image) scattered bilinearly to the texels each pixel sampled.
    LFS_CUDA_API void cubemap_backward(const float* grad_image_chw, const float* alpha_hw, int res,
                                       const CubemapView& view, float* grad_texels, cudaStream_t stream);

    // Smoothness (squared-difference total variation over the 4 neighbours, across face edges):
    //   L = weight * mean over directed neighbour pairs of |t_a - t_n|^2 (RGB summed);  dL/dt_a = weight / N * sum_n (t_a - t_n)
    // with N = 6 R^2 texels. Adds the gradient to grad_texels; adds L to *loss_out when loss_out != null.
    LFS_CUDA_API void cubemap_smooth_grad(const float* texels, int res, float weight, float* grad_texels, float* loss_out,
                                          cudaStream_t stream);

    // Adam on all n floats, then clamp to [lo, hi] and zero the gradient. Non-finite gradients are treated as 0 and
    // counted into *nonfinite_count (device float).
    LFS_CUDA_API void cubemap_adam_step(float* texels, float* grad, float* m, float* v, int n, float lr, float beta1,
                                        float beta2, float eps, float bias1, float bias2, float lo, float hi,
                                        float* nonfinite_count, cudaStream_t stream);

    LFS_CUDA_API void cubemap_fill(float* texels, int res, float r, float g, float b, cudaStream_t stream);

    // out3 (device, caller zeros): [0] += sum |g|, [1] = max |g| (atomicMax on the bits of non-negative floats),
    // [2] += number of non-finite entries
    LFS_CUDA_API void cubemap_grad_stats(const float* grad, int n, float* out3, cudaStream_t stream);

} // namespace lfs::training::kernels
