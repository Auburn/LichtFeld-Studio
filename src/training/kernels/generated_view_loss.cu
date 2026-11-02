/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Alignment-tolerant loss for generated training views (see generated_view_loss.hpp).
//
// Forward and backward in seven launches, all on the render's stream, no host synchronization:
//   1 prepare   per pixel: the 12 filter input planes (masked render / target / mask for the Gaussian, masked shifted
//               luma and its square of render and target plus the mask for the box) and the mask and pixel-L1 sums.
//   2,3 conv    separable filter, rows then columns, over all 12 planes (Gaussian taps on planes 0-6, box on 7-11).
//   4 seed      per pixel: the normalized filter outputs, the low-pass and texture differences and their sums, and
//               the backward seeds dL/d(filter output) / (filter normalization), written over planes 0-4.
//   5,6 conv    the same separable filter on the 5 seed planes (the zero-padded symmetric filter is its own
//               transpose), Gaussian on 0-2, box on 3-4.
//   7 grad      per pixel: dL/dr = m * (filtered low-pass seed) + luma * m * (E[y] seed + 2 y E[y^2] seed) + pixel term.
//   8 finalize  one thread: the loss scalar and the logging sums.

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "generated_view_loss.hpp"
#include "kernel_stream.hpp"
#include "lfs/core/warp_reduce.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace lfs::training::kernels {
    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_BLOCKS = 1024;
        constexpr int CONV_TX = 32;
        constexpr int CONV_TY = 8;

        // Forward planes: 0-2 m*r_c, 3-5 m*t_c, 6 m (Gaussian); 7 m*yr, 8 m*yr^2, 9 m*yt, 10 m*yt^2, 11 m (box).
        constexpr int FWD_PLANES = 12;
        constexpr int FWD_GAUSS_PLANES = 7;
        // Backward seed planes (over 0-4): 0-2 low-pass seeds (Gaussian), 3 E[y] seed, 4 E[y^2] seed (box).
        constexpr int BWD_PLANES = 5;
        constexpr int BWD_GAUSS_PLANES = 3;

        // Per-call device sums.
        constexpr int SC_MASK = 0;
        constexpr int SC_LOW = 1;
        constexpr int SC_TEX = 2;
        constexpr int SC_PIX = 3;
        constexpr int SC_STD_R = 4;
        constexpr int SC_STD_T = 5;
        constexpr int SC_COUNT = 8;

        constexpr float LUMA_R = 0.2126f;
        constexpr float LUMA_G = 0.7152f;
        constexpr float LUMA_B = 0.0722f;
        // The luma is shifted by this before the local moments, which keeps E[y^2] - E[y]^2 out of float cancellation.
        constexpr float LUMA_SHIFT = 0.5f;

        struct Taps {
            float w[GENERATED_VIEW_MAX_RADIUS + 1];
            int radius;
        };

        template <typename T>
        __device__ __forceinline__ float target_value(const T* target, const int i) {
            if constexpr (std::is_same_v<T, uint8_t>)
                return static_cast<float>(target[i]) * (1.0f / 255.0f);
            else
                return target[i];
        }

        template <typename M>
        __device__ __forceinline__ float mask_weight(const M* mask, const int i) {
            if (mask == nullptr)
                return 1.0f;
            if constexpr (std::is_same_v<M, uint8_t>)
                return mask[i] != 0 ? 1.0f : 0.0f;
            else
                return mask[i];
        }

        __device__ __forceinline__ float sign_of(const float x) {
            return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        }

        __device__ __forceinline__ float luma_shifted(const float r, const float g, const float b) {
            return LUMA_R * r + LUMA_G * g + LUMA_B * b - LUMA_SHIFT;
        }

        template <typename T, typename M>
        __global__ void prepare_kernel(const float* __restrict__ render, const T* __restrict__ target,
                                       const M* __restrict__ mask, float* __restrict__ planes,
                                       float* __restrict__ scalars, const int n) {
            float mask_sum = 0.0f;
            float pix_sum = 0.0f;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const float m = mask_weight(mask, i);
                float r[3], t[3];
#pragma unroll
                for (int c = 0; c < 3; ++c) {
                    r[c] = render[c * n + i];
                    t[c] = target_value(target, c * n + i);
                    planes[c * n + i] = m * r[c];
                    planes[(3 + c) * n + i] = m * t[c];
                    pix_sum += m * fabsf(r[c] - t[c]);
                }
                planes[6 * n + i] = m;
                const float yr = luma_shifted(r[0], r[1], r[2]);
                const float yt = luma_shifted(t[0], t[1], t[2]);
                planes[7 * n + i] = m * yr;
                planes[8 * n + i] = m * yr * yr;
                planes[9 * n + i] = m * yt;
                planes[10 * n + i] = m * yt * yt;
                planes[11 * n + i] = m;
                mask_sum += m;
            }
            mask_sum = lfs::core::warp_ops::block_reduce_sum(mask_sum);
            pix_sum = lfs::core::warp_ops::block_reduce_sum(pix_sum);
            if (threadIdx.x == 0) {
                atomicAdd(&scalars[SC_MASK], mask_sum);
                atomicAdd(&scalars[SC_PIX], pix_sum);
            }
        }

        // One 1-D pass of the separable filter over planes [0, gridDim.z): taps a on planes < split, b on the rest.
        // Zero padding: taps outside the image are dropped (the normalization planes carry the matching weights).
        __global__ void conv_kernel(const float* __restrict__ in, float* __restrict__ out, const int height,
                                    const int width, const int split, const Taps a, const Taps b,
                                    const bool along_x) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            const int plane = blockIdx.z;
            if (x >= width || y >= height)
                return;
            const Taps& taps = plane < split ? a : b;
            const size_t offset = static_cast<size_t>(plane) * height * width;
            const float* src = in + offset;
            const int i = y * width + x;
            float acc = taps.w[0] * src[i];
            if (along_x) {
                for (int d = 1; d <= taps.radius; ++d) {
                    const float w = taps.w[d];
                    if (x - d >= 0)
                        acc += w * src[i - d];
                    if (x + d < width)
                        acc += w * src[i + d];
                }
            } else {
                for (int d = 1; d <= taps.radius; ++d) {
                    const float w = taps.w[d];
                    if (y - d >= 0)
                        acc += w * src[i - d * width];
                    if (y + d < height)
                        acc += w * src[i + d * width];
                }
            }
            out[offset + i] = acc;
        }

        // low_scale = loss_weight * lowpass_weight / C, tex_scale = loss_weight * texture_weight; both / sum(m) here.
        template <typename M>
        __global__ void seed_kernel(const M* __restrict__ mask, float* __restrict__ planes,
                                    float* __restrict__ scalars, const int n, const float low_scale,
                                    const float tex_scale, const float var_eps) {
            const float mask_total = scalars[SC_MASK];
            const float inv_mask_total = mask_total > 0.0f ? 1.0f / mask_total : 0.0f;
            float low_sum = 0.0f, tex_sum = 0.0f, std_r_sum = 0.0f, std_t_sum = 0.0f;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const float m = mask_weight(mask, i);
                float seed[BWD_PLANES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                // m > 0 implies both normalizations > 0 (the centre tap sees the pixel itself).
                if (m > 0.0f) {
                    const float inv_g = 1.0f / planes[6 * n + i];
#pragma unroll
                    for (int c = 0; c < 3; ++c) {
                        const float d = (planes[c * n + i] - planes[(3 + c) * n + i]) * inv_g;
                        low_sum += m * fabsf(d);
                        seed[c] = low_scale * inv_mask_total * m * sign_of(d) * inv_g;
                    }
                    const float inv_b = 1.0f / planes[11 * n + i];
                    const float e1r = planes[7 * n + i] * inv_b;
                    const float var_r = planes[8 * n + i] * inv_b - e1r * e1r;
                    const float e1t = planes[9 * n + i] * inv_b;
                    const float var_t = planes[10 * n + i] * inv_b - e1t * e1t;
                    const float std_r = sqrtf(fmaxf(var_r, var_eps));
                    const float std_t = sqrtf(fmaxf(var_t, var_eps));
                    const float ds = std_r - std_t;
                    tex_sum += m * fabsf(ds);
                    std_r_sum += m * std_r;
                    std_t_sum += m * std_t;
                    // dL/dvar_r; the clamp has zero derivative below var_eps.
                    const float dvar = var_r > var_eps
                                           ? tex_scale * inv_mask_total * m * sign_of(ds) * 0.5f / std_r
                                           : 0.0f;
                    // var = E2 - E1^2: dL/dE1 = -2 E1 dvar, dL/dE2 = dvar; then / normalization.
                    seed[3] = -2.0f * e1r * dvar * inv_b;
                    seed[4] = dvar * inv_b;
                }
                // Safe in place: this thread alone reads pixel i of every plane in this kernel.
#pragma unroll
                for (int k = 0; k < BWD_PLANES; ++k)
                    planes[k * n + i] = seed[k];
            }
            low_sum = lfs::core::warp_ops::block_reduce_sum(low_sum);
            tex_sum = lfs::core::warp_ops::block_reduce_sum(tex_sum);
            std_r_sum = lfs::core::warp_ops::block_reduce_sum(std_r_sum);
            std_t_sum = lfs::core::warp_ops::block_reduce_sum(std_t_sum);
            if (threadIdx.x == 0) {
                atomicAdd(&scalars[SC_LOW], low_sum);
                atomicAdd(&scalars[SC_TEX], tex_sum);
                atomicAdd(&scalars[SC_STD_R], std_r_sum);
                atomicAdd(&scalars[SC_STD_T], std_t_sum);
            }
        }

        // pix_scale = loss_weight * pixel_weight / C (then / sum(m)).
        template <typename T, typename M>
        __global__ void grad_kernel(const float* __restrict__ render, const T* __restrict__ target,
                                    const M* __restrict__ mask, const float* __restrict__ planes,
                                    const float* __restrict__ scalars, float* __restrict__ grad, const int n,
                                    const float pix_scale) {
            const float mask_total = scalars[SC_MASK];
            const float pix_coef = mask_total > 0.0f ? pix_scale / mask_total : 0.0f;
            const float luma[3] = {LUMA_R, LUMA_G, LUMA_B};
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
                const float m = mask_weight(mask, i);
                float r[3];
#pragma unroll
                for (int c = 0; c < 3; ++c)
                    r[c] = render[c * n + i];
                const float yr = luma_shifted(r[0], r[1], r[2]);
                const float dy = m * (planes[3 * n + i] + 2.0f * yr * planes[4 * n + i]);
#pragma unroll
                for (int c = 0; c < 3; ++c) {
                    float g = m * planes[c * n + i] + luma[c] * dy;
                    if (pix_coef != 0.0f)
                        g += pix_coef * m * sign_of(r[c] - target_value(target, c * n + i));
                    grad[c * n + i] = g;
                }
            }
        }

        __global__ void finalize_kernel(const float* __restrict__ scalars, float* __restrict__ loss,
                                        float* __restrict__ stats, const float loss_weight,
                                        const float lowpass_weight, const float texture_weight,
                                        const float pixel_weight) {
            const float mask_total = scalars[SC_MASK];
            const float inv = mask_total > 0.0f ? 1.0f / mask_total : 0.0f;
            const float low = scalars[SC_LOW] * inv / 3.0f;
            const float tex = scalars[SC_TEX] * inv;
            const float pix = scalars[SC_PIX] * inv / 3.0f;
            const float total = loss_weight * (lowpass_weight * low + texture_weight * tex + pixel_weight * pix);
            loss[0] = total;
            if (stats != nullptr) {
                stats[GENERATED_VIEW_STAT_VIEWS] += 1.0f;
                stats[GENERATED_VIEW_STAT_TOTAL] += total;
                stats[GENERATED_VIEW_STAT_LOWPASS] += low;
                stats[GENERATED_VIEW_STAT_TEXTURE] += tex;
                stats[GENERATED_VIEW_STAT_PIXEL] += pix;
                stats[GENERATED_VIEW_STAT_STD_RENDER] += scalars[SC_STD_R] * inv;
                stats[GENERATED_VIEW_STAT_STD_TARGET] += scalars[SC_STD_T] * inv;
            }
        }

        __global__ void add_real_kernel(const float* __restrict__ loss, float* __restrict__ stats) {
            stats[GENERATED_VIEW_STAT_REAL_LOSS] += loss[0];
            stats[GENERATED_VIEW_STAT_REAL_VIEWS] += 1.0f;
        }

        Taps gaussian_taps(const float sigma) {
            Taps taps{};
            if (!(sigma > 0.0f)) {
                taps.radius = 0;
                taps.w[0] = 1.0f;
                return taps;
            }
            taps.radius = std::min(GENERATED_VIEW_MAX_RADIUS, static_cast<int>(std::ceil(3.0f * sigma)));
            double sum = 0.0;
            double w[GENERATED_VIEW_MAX_RADIUS + 1];
            for (int d = 0; d <= taps.radius; ++d) {
                w[d] = std::exp(-0.5 * static_cast<double>(d) * d / (static_cast<double>(sigma) * sigma));
                sum += d == 0 ? w[d] : 2.0 * w[d];
            }
            for (int d = 0; d <= taps.radius; ++d)
                taps.w[d] = static_cast<float>(w[d] / sum);
            return taps;
        }

        Taps box_taps(const int radius) {
            Taps taps{};
            taps.radius = radius;
            for (int d = 0; d <= radius; ++d)
                taps.w[d] = 1.0f / static_cast<float>(2 * radius + 1);
            return taps;
        }

        lfs::core::Tensor as_chw(const lfs::core::Tensor& image) {
            auto out = image.contiguous();
            if (out.ndim() == 4)
                out = out.squeeze(0);
            return out;
        }
    } // namespace

    void GeneratedViewLossWorkspace::ensure_size(const size_t height, const size_t width) {
        using namespace lfs::core;
        if (!scalars.is_valid()) {
            scalars = Tensor::zeros({static_cast<size_t>(SC_COUNT)}, Device::CUDA);
            loss = Tensor::zeros({1}, Device::CUDA);
        }
        if (planes.is_valid() && planes.shape()[1] == height && planes.shape()[2] == width)
            return;
        planes = Tensor::empty({static_cast<size_t>(FWD_PLANES), height, width}, Device::CUDA);
        tmp = Tensor::empty({static_cast<size_t>(FWD_PLANES), height, width}, Device::CUDA);
        grad = Tensor::empty({3, height, width}, Device::CUDA);
    }

    GeneratedViewLossResult generated_view_loss(
        const lfs::core::Tensor& render_input,
        const lfs::core::Tensor& target_input,
        const lfs::core::Tensor& pixel_weight_input,
        const GeneratedViewLossParams& params,
        GeneratedViewLossWorkspace& workspace,
        lfs::core::Tensor* stats_tensor) {
        using namespace lfs::core;
        LFS_ASSERT_MSG(render_input.is_valid() && render_input.device() == Device::CUDA &&
                           render_input.dtype() == DataType::Float32,
                       "generated-view loss: render must be a CUDA Float32 tensor");
        LFS_ASSERT_MSG(render_input.ndim() == 3 || (render_input.ndim() == 4 && render_input.shape()[0] == 1),
                       "generated-view loss: render must be [3,H,W] or [1,3,H,W]");
        LFS_ASSERT_MSG(target_input.is_valid() && target_input.device() == Device::CUDA &&
                           target_input.shape() == render_input.shape() &&
                           (target_input.dtype() == DataType::Float32 || target_input.dtype() == DataType::UInt8),
                       "generated-view loss: target must match the render (Float32 or UInt8)");
        LFS_ASSERT_MSG(std::isfinite(params.lowpass_sigma) && params.lowpass_sigma >= 0.0f &&
                           params.lowpass_sigma <= GENERATED_VIEW_MAX_SIGMA,
                       "generated-view loss: lowpass sigma out of range");
        LFS_ASSERT_MSG(params.texture_radius >= 1 && params.texture_radius <= GENERATED_VIEW_MAX_RADIUS,
                       "generated-view loss: texture radius out of range");

        const Tensor render = as_chw(render_input);
        const Tensor target = as_chw(target_input);
        LFS_ASSERT_MSG(render.shape()[0] == 3, "generated-view loss needs 3-channel images");
        const size_t height = render.shape()[1];
        const size_t width = render.shape()[2];
        LFS_ASSERT_MSG(height * width <= static_cast<size_t>(INT32_MAX / FWD_PLANES),
                       "generated-view loss: image too large");
        const int n = static_cast<int>(height * width);

        Tensor weight;
        if (pixel_weight_input.is_valid() && pixel_weight_input.numel() > 0) {
            weight = pixel_weight_input.contiguous();
            if (weight.ndim() == 3)
                weight = weight.squeeze(0);
            LFS_ASSERT_MSG(weight.device() == Device::CUDA && weight.ndim() == 2 && weight.shape()[0] == height &&
                               weight.shape()[1] == width &&
                               (weight.dtype() == DataType::Float32 || weight.dtype() == DataType::UInt8 ||
                                weight.dtype() == DataType::Bool),
                           "generated-view loss: pixel weight must be [H,W] Float32, UInt8 or Bool");
        }

        workspace.ensure_size(height, width);
        const cudaStream_t stream = resolve_stream(render.stream());
        for (auto* t : {&workspace.planes, &workspace.tmp, &workspace.grad, &workspace.scalars, &workspace.loss})
            t->set_stream(stream);
        float* stats = nullptr;
        if (stats_tensor != nullptr) {
            LFS_ASSERT_MSG(stats_tensor->is_valid() && stats_tensor->device() == Device::CUDA &&
                               stats_tensor->dtype() == DataType::Float32 &&
                               stats_tensor->numel() >= static_cast<size_t>(GENERATED_VIEW_STAT_COUNT),
                           "generated-view loss: stats must be a CUDA Float32 tensor of GENERATED_VIEW_STAT_COUNT");
            stats_tensor->set_stream(stream);
            stats = stats_tensor->ptr<float>();
        }
        target.sync_to_stream(stream);
        if (weight.is_valid())
            weight.sync_to_stream(stream);

        float* planes = workspace.planes.ptr<float>();
        float* tmp = workspace.tmp.ptr<float>();
        float* scalars = workspace.scalars.ptr<float>();
        LFS_CUDA_CHECK(cudaMemsetAsync(scalars, 0, SC_COUNT * sizeof(float), stream));

        const Taps gauss = gaussian_taps(params.lowpass_sigma);
        const Taps box = box_taps(params.texture_radius);
        const int blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        const dim3 conv_threads(CONV_TX, CONV_TY);
        const auto conv_grid = [&](const int planes_count) {
            return dim3(static_cast<unsigned>((width + CONV_TX - 1) / CONV_TX),
                        static_cast<unsigned>((height + CONV_TY - 1) / CONV_TY),
                        static_cast<unsigned>(planes_count));
        };
        const float low_scale = params.loss_weight * params.lowpass_weight / 3.0f;
        const float tex_scale = params.loss_weight * params.texture_weight;
        const float pix_scale = params.loss_weight * params.pixel_weight / 3.0f;
        const int h = static_cast<int>(height);
        const int w = static_cast<int>(width);

        auto run = [&]<typename T, typename M>(const T* target_ptr, const M* mask_ptr) {
            prepare_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(render.ptr<float>(), target_ptr, mask_ptr, planes,
                                                               scalars, n);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.prepare");
            conv_kernel<<<conv_grid(FWD_PLANES), conv_threads, 0, stream>>>(planes, tmp, h, w, FWD_GAUSS_PLANES,
                                                                             gauss, box, true);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.conv_x");
            conv_kernel<<<conv_grid(FWD_PLANES), conv_threads, 0, stream>>>(tmp, planes, h, w, FWD_GAUSS_PLANES,
                                                                             gauss, box, false);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.conv_y");
            seed_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(mask_ptr, planes, scalars, n, low_scale, tex_scale,
                                                            GENERATED_VIEW_VAR_EPS);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.seed");
            conv_kernel<<<conv_grid(BWD_PLANES), conv_threads, 0, stream>>>(planes, tmp, h, w, BWD_GAUSS_PLANES,
                                                                             gauss, box, true);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.seed_conv_x");
            conv_kernel<<<conv_grid(BWD_PLANES), conv_threads, 0, stream>>>(tmp, planes, h, w, BWD_GAUSS_PLANES,
                                                                             gauss, box, false);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.seed_conv_y");
            grad_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(render.ptr<float>(), target_ptr, mask_ptr, planes,
                                                            scalars, workspace.grad.ptr<float>(), n, pix_scale);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.grad");
            finalize_kernel<<<1, 1, 0, stream>>>(scalars, workspace.loss.ptr<float>(), stats, params.loss_weight,
                                                 params.lowpass_weight, params.texture_weight, params.pixel_weight);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.finalize");
        };
        auto run_masked = [&]<typename T>(const T* target_ptr) {
            if (!weight.is_valid())
                run(target_ptr, static_cast<const float*>(nullptr));
            else if (weight.dtype() == DataType::Float32)
                run(target_ptr, weight.ptr<float>());
            else
                run(target_ptr, weight.ptr<uint8_t>());
        };
        if (target.dtype() == DataType::UInt8)
            run_masked(target.ptr<uint8_t>());
        else
            run_masked(target.ptr<float>());

        Tensor grad = workspace.grad;
        if (render_input.ndim() == 4)
            grad = grad.unsqueeze(0);
        return {.loss = workspace.loss, .grad = grad};
    }

    void generated_view_stats_add_real(const lfs::core::Tensor& loss, lfs::core::Tensor& stats) {
        using namespace lfs::core;
        if (!loss.is_valid() || loss.numel() == 0)
            return;
        LFS_ASSERT_MSG(loss.device() == Device::CUDA && loss.dtype() == DataType::Float32,
                       "generated-view stats: loss must be a CUDA Float32 scalar");
        LFS_ASSERT_MSG(stats.is_valid() && stats.device() == Device::CUDA && stats.dtype() == DataType::Float32 &&
                           stats.numel() >= static_cast<size_t>(GENERATED_VIEW_STAT_COUNT),
                       "generated-view stats must be a CUDA Float32 tensor of GENERATED_VIEW_STAT_COUNT");
        const cudaStream_t stream = resolve_stream(loss.stream());
        stats.set_stream(stream);
        add_real_kernel<<<1, 1, 0, stream>>>(loss.ptr<float>(), stats.ptr<float>());
        LFS_CUDA_LAUNCH_CHECK(stream, "training.generated_view_loss.add_real");
    }
} // namespace lfs::training::kernels
