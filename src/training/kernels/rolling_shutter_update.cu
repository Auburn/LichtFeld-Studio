/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// --rolling-shutter --rs-learn: the per-camera motion / shared readout Adam step (see rolling_shutter_update.hpp).

#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "rolling_shutter_update.hpp"

namespace lfs::training::kernels {
    namespace {
        __global__ void rolling_shutter_adam_kernel(float* motion, const float* init, float* adam_m, float* adam_v, float* steps,
                                                    float* readout, float* readout_state, const float* grad, const int* nb,
                                                    const RollingShutterUpdate u) {
            const int j = threadIdx.x;
            const int row = u.row;
            if (j < 12 && ((u.learn_mask >> j) & 1u)) {
                const float sc = u.scale[j];
                const int i = row * 12 + j;
                const float q = (motion[i] - init[i]) / sc;
                float g = grad[j] * sc + 2.0f * u.prior * q;
                if (u.smooth > 0.0f) {
                    for (int k = 0; k < 2; ++k) {
                        const int n = nb[row * 2 + k];
                        if (n >= 0)
                            g += 2.0f * u.smooth * (q - (motion[n * 12 + j] - init[n * 12 + j]) / sc);
                    }
                }
                if (isfinite(g)) {
                    const float step = steps[row] + 1.0f;
                    const float m = u.beta1 * adam_m[i] + (1.0f - u.beta1) * g;
                    const float v = u.beta2 * adam_v[i] + (1.0f - u.beta2) * g * g;
                    adam_m[i] = m;
                    adam_v[i] = v;
                    const float mh = m / (1.0f - powf(u.beta1, step));
                    const float vh = v / (1.0f - powf(u.beta2, step));
                    motion[i] = init[i] + sc * (q - u.lr * mh / (sqrtf(vh) + u.eps));
                }
            }
            if (j == 12 && u.learn_readout) {
                const float g = grad[12] * 1e-3f; // dL/d(tau in ms)
                if (isfinite(g)) {
                    const float step = readout_state[2] + 1.0f;
                    const float m = u.beta1 * readout_state[0] + (1.0f - u.beta1) * g;
                    const float v = u.beta2 * readout_state[1] + (1.0f - u.beta2) * g * g;
                    readout_state[0] = m;
                    readout_state[1] = v;
                    readout_state[2] = step;
                    const float mh = m / (1.0f - powf(u.beta1, step));
                    const float vh = v / (1.0f - powf(u.beta2, step));
                    const float ms = readout[0] * 1000.0f - u.readout_lr_ms * mh / (sqrtf(vh) + u.eps);
                    readout[0] = fminf(fmaxf(ms, u.readout_min_ms), u.readout_max_ms) * 1e-3f;
                }
            }
            __syncthreads();
            if (j == 0 && u.learn_mask != 0u)
                steps[row] += 1.0f;
        }
    } // namespace

    void rolling_shutter_adam_step(float* motion, const float* init, float* adam_m, float* adam_v, float* steps,
                                   float* readout, float* readout_state, const float* grad, const int* nb,
                                   const RollingShutterUpdate& u, cudaStream_t stream) {
        stream = resolve_stream(stream);
        rolling_shutter_adam_kernel<<<1, 32, 0, stream>>>(motion, init, adam_m, adam_v, steps, readout, readout_state, grad, nb, u);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.rolling_shutter.adam");
    }

    namespace {
        __global__ void motion_blur_exposure_adam_kernel(float* scale, float* state, const float* grad, const int row, const float lr,
                                                         const float lo, const float hi) {
            const float g = grad[0];
            if (!isfinite(g))
                return;
            float* st = state + 3 * row;
            const float step = st[2] + 1.0f;
            st[2] = step;
            const float m = 0.9f * st[0] + 0.1f * g;
            const float v = 0.999f * st[1] + 0.001f * g * g;
            st[0] = m;
            st[1] = v;
            const float mh = m / (1.0f - powf(0.9f, step));
            const float vh = v / (1.0f - powf(0.999f, step));
            scale[row] = fminf(fmaxf(scale[row] - lr * mh / (sqrtf(vh) + 1e-10f), lo), hi);
        }
    } // namespace

    void motion_blur_exposure_adam_step(float* scale, float* state, const float* grad, const int row, const float lr, const float lo,
                                        const float hi, cudaStream_t stream) {
        stream = resolve_stream(stream);
        motion_blur_exposure_adam_kernel<<<1, 1, 0, stream>>>(scale, state, grad, row, lr, lo, hi);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.motion_blur.exposure_adam");
    }

} // namespace lfs::training::kernels
