/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// --pose-opt: the per-camera pose compose and Adam kernels (see pose_refinement_update.hpp).

#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "pose_refinement_math.hpp"
#include "pose_refinement_update.hpp"

namespace lfs::training::kernels {
    namespace {
        __global__ void pose_compose_kernel(const float* pose, const float* w2c0, const float* c0, const int row,
                                            float* w2c_out, float* cam_out) {
            if (threadIdx.x != 0)
                return;
            float off[6], W0[16], C0[3], W[16], C[3];
            for (int k = 0; k < 6; ++k)
                off[k] = pose[row * 6 + k];
            for (int k = 0; k < 16; ++k)
                W0[k] = w2c0[row * 16 + k];
            for (int k = 0; k < 3; ++k)
                C0[k] = c0[row * 3 + k];
            pose_math::compose<float>(W0, C0, off, W, C);
            for (int k = 0; k < 16; ++k)
                w2c_out[k] = W[k];
            for (int k = 0; k < 3; ++k)
                cam_out[k] = C[k];
        }

        __global__ void pose_adam_kernel(float* pose, float* adam_m, float* adam_v, float* steps, const float* grad,
                                         const int* nb, const float* w2c0, const PoseRefinementUpdate u) {
            if (threadIdx.x != 0)
                return;
            const int row = u.row;
            float off[6], g[6], dpar[6];
            for (int k = 0; k < 6; ++k) {
                off[k] = pose[row * 6 + k];
                g[k] = grad[k];
            }
            pose_math::chain<float>(off, g, dpar);
            // prior f^2 |(r, d)|^2 and smoothness f^2 |(r, d) - M (r_n, d_n)|^2, M = W0_row W0_n^T (the neighbour's offset
            // through the world frame into this camera's axes)
            const float f2 = u.focal_px * u.focal_px;
            if (u.prior > 0.0f)
                for (int k = 0; k < 6; ++k)
                    dpar[k] += 2.0f * u.prior * f2 * off[k];
            if (u.smooth > 0.0f) {
                const float* A = w2c0 + row * 16;
                for (int s = 0; s < 2; ++s) {
                    const int n = nb[row * 2 + s];
                    if (n < 0)
                        continue;
                    const float* B = w2c0 + n * 16;
                    for (int part = 0; part < 2; ++part) {
                        const float* on = pose + n * 6 + 3 * part;
                        // world vector B^T on, then into this camera's axes: A (B^T on)
                        float wv[3];
                        for (int j = 0; j < 3; ++j)
                            wv[j] = B[j] * on[0] + B[4 + j] * on[1] + B[8 + j] * on[2];
                        for (int i = 0; i < 3; ++i) {
                            const float mapped = A[i * 4] * wv[0] + A[i * 4 + 1] * wv[1] + A[i * 4 + 2] * wv[2];
                            dpar[3 * part + i] += 2.0f * u.smooth * f2 * (off[3 * part + i] - mapped);
                        }
                    }
                }
            }
            bool finite = true;
            for (int k = 0; k < 6; ++k)
                finite = finite && isfinite(dpar[k]);
            if (!finite)
                return;
            const float step = steps[row] + 1.0f;
            const float bc1 = 1.0f - powf(u.beta1, step), bc2 = 1.0f - powf(u.beta2, step);
            for (int k = 0; k < 6; ++k) {
                const int i = row * 6 + k;
                const float m = u.beta1 * adam_m[i] + (1.0f - u.beta1) * dpar[k];
                const float v = u.beta2 * adam_v[i] + (1.0f - u.beta2) * dpar[k] * dpar[k];
                adam_m[i] = m;
                adam_v[i] = v;
                const float lr = k < 3 ? u.lr_rot : u.lr_trans;
                pose[i] = off[k] - lr * (m / bc1) / (sqrtf(v / bc2) + u.eps);
            }
            steps[row] = step;
        }
    } // namespace

    void pose_compose(const float* pose, const float* w2c0, const float* c0, const int row, float* w2c_out, float* cam_out,
                      cudaStream_t stream) {
        stream = resolve_stream(stream);
        pose_compose_kernel<<<1, 32, 0, stream>>>(pose, w2c0, c0, row, w2c_out, cam_out);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.pose_refinement.compose");
    }

    void pose_adam_step(float* pose, float* adam_m, float* adam_v, float* steps, const float* grad, const int* nb,
                        const float* w2c0, const PoseRefinementUpdate& u, cudaStream_t stream) {
        stream = resolve_stream(stream);
        pose_adam_kernel<<<1, 32, 0, stream>>>(pose, adam_m, adam_v, steps, grad, nb, w2c0, u);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.pose_refinement.adam");
    }

} // namespace lfs::training::kernels
