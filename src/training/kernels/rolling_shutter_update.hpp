/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // --rolling-shutter with --rs-learn: one Adam step of one camera's rolling-shutter motion and of the shared readout,
    // on the device (no host sync), from the gradient the rasterizer's backward accumulated (grad[13]: dL/d(motion[12]),
    // dL/dtau). The motion is learned as a correction q to the camera path's motion init, in "pixels at the readout
    // ends" (motion = init + scale * q per component), with a prior prior * |q|^2 and a smoothness term
    // smooth * |q - q_n|^2 to each time neighbour n (nb[2 * row], nb[2 * row + 1], -1 = none). The readout is learned in
    // milliseconds and clamped to [readout_min_ms, readout_max_ms]. Per-camera Adam state (m, v, step count) lives in
    // adam_m / adam_v [N, 12] and steps [N]; the readout's in readout_state [3] (m, v, step).
    struct RollingShutterUpdate {
        int row = 0;
        unsigned learn_mask = 0;     // bit k: motion component k (0-11) is learned
        bool learn_readout = false;
        float lr = 0.05f;            // per camera update, in q units (pixels)
        float readout_lr_ms = 0.002f;
        float prior = 1e-4f;
        float smooth = 0.0f;
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        float readout_min_ms = 0.0f, readout_max_ms = 40.0f;
        float scale[12] = {};
    };

    LFS_CUDA_API void rolling_shutter_adam_step(float* motion, const float* init, float* adam_m, float* adam_v, float* steps,
                                                float* readout, float* readout_state, const float* grad, const int* nb,
                                                const RollingShutterUpdate& u, cudaStream_t stream);

    // --motion-blur-learn-exposure: one Adam step of camera row's exposure scale from grad[0] = dL/d(scale) (state [N,3]:
    // m, v, step count), clamped to [lo, hi]; a non-finite gradient skips the step.
    LFS_CUDA_API void motion_blur_exposure_adam_step(float* scale, float* state, const float* grad, int row, float lr, float lo,
                                                     float hi, cudaStream_t stream);
} // namespace lfs::training::kernels
