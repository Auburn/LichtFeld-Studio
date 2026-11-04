/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // --pose-opt (training/pose_refinement.hpp), on the device without a host sync:
    //  pose_compose: camera `row`'s refined world-to-camera matrix and centre from its start pose and its offset
    //    (pose_refinement_math.hpp compose) into w2c_out [16] / cam_out [3], which the rasterizer then draws with;
    //  pose_adam_step: after the backward of that camera, one Adam step of its offset (r [rad], d [scene units], the
    //    start camera's axes) from the rasterizer's dL/d(left perturbation) grad [6], chained exactly to the offset,
    //    plus prior * |q|^2 and smooth * |q - q_n|^2 to each time neighbour n (nb[2 row], nb[2 row + 1], -1 = none;
    //    the neighbour's offset taken through the world frame), q = focal_px * (r, d) (pixels at the image centre /
    //    one scene unit away). Per-camera Adam state m, v [N, 6] and step count [N].
    LFS_CUDA_API void pose_compose(const float* pose, const float* w2c0, const float* c0, int row,
                                   float* w2c_out, float* cam_out, cudaStream_t stream);

    struct PoseRefinementUpdate {
        int row = 0;
        float lr_rot = 0.0f;   // radians per camera update (schedule applied)
        float lr_trans = 0.0f; // scene units per camera update (schedule applied)
        float prior = 0.0f;
        float smooth = 0.0f;
        float focal_px = 1.0f;
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    };

    LFS_CUDA_API void pose_adam_step(float* pose, float* adam_m, float* adam_v, float* steps, const float* grad,
                                     const int* nb, const float* w2c0, const PoseRefinementUpdate& u, cudaStream_t stream);

} // namespace lfs::training::kernels
