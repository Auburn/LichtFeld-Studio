/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Per-pixel photometric loss weights (--loss-weight-dir; see loss_weight_map.hpp).

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "loss_weight_map.hpp"
#include <cstdint>
#include <type_traits>

namespace lfs::training::kernels {
    namespace {
        constexpr int BLOCK_SIZE = 256;

        __global__ void resample_loss_weight_kernel(const float* __restrict__ src, const int src_w, const int src_h,
                                                    float* __restrict__ dst, const int w, const int h) {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<int64_t>(w) * h)
                return;
            const int x = static_cast<int>(i % w);
            const int y = static_cast<int>(i / w);
            if (src_w == w && src_h == h) {
                dst[i] = src[i];
                return;
            }
            // Pixel centre to pixel centre, clamped to the source's outer centres.
            const float sx = fminf(fmaxf((static_cast<float>(x) + 0.5f) * (static_cast<float>(src_w) / static_cast<float>(w)) - 0.5f, 0.0f),
                                   static_cast<float>(src_w - 1));
            const float sy = fminf(fmaxf((static_cast<float>(y) + 0.5f) * (static_cast<float>(src_h) / static_cast<float>(h)) - 0.5f, 0.0f),
                                   static_cast<float>(src_h - 1));
            const int x0 = static_cast<int>(floorf(sx));
            const int y0 = static_cast<int>(floorf(sy));
            const int x1 = min(x0 + 1, src_w - 1);
            const int y1 = min(y0 + 1, src_h - 1);
            const float fx = sx - static_cast<float>(x0);
            const float fy = sy - static_cast<float>(y0);
            const float top = (1.0f - fx) * src[y0 * src_w + x0] + fx * src[y0 * src_w + x1];
            const float bottom = (1.0f - fx) * src[y1 * src_w + x0] + fx * src[y1 * src_w + x1];
            dst[i] = (1.0f - fy) * top + fy * bottom;
        }

        template <typename BaseT>
        __global__ void combine_loss_weight_kernel(const float* __restrict__ weight, const BaseT* base,
                                                   float* out, const int w, const int h, const int border) {
            const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<int64_t>(w) * h)
                return;
            float b = 1.0f;
            if constexpr (!std::is_void_v<BaseT>) {
                if constexpr (std::is_same_v<BaseT, uint8_t>)
                    b = base[i] != 0 ? 1.0f : 0.0f;
                else
                    b = static_cast<float>(base[i]);
            }
            bool inside = true;
            if (border > 0 && h > 2 * border && w > 2 * border) {
                const int x = static_cast<int>(i % w);
                const int y = static_cast<int>(i / w);
                inside = x >= border && x < w - border && y >= border && y < h - border;
            }
            out[i] = inside ? weight[i] * b : 0.0f;
        }
    } // namespace

    void resample_loss_weight_map(const float* source, const int src_width, const int src_height,
                                  lfs::core::Tensor& output, const int width, const int height) {
        using namespace lfs::core;
        LFS_ASSERT_MSG(source != nullptr && src_width > 0 && src_height > 0 && width > 0 && height > 0,
                       "loss weight resample: empty source or output");
        const TensorShape shape{static_cast<size_t>(height), static_cast<size_t>(width)};
        if (!output.is_valid() || output.shape() != shape || output.dtype() != DataType::Float32 ||
            output.device() != Device::CUDA)
            output = Tensor::empty(shape, Device::CUDA, DataType::Float32);
        const cudaStream_t stream = resolve_stream(output.stream());
        output.set_stream(stream);
        const int64_t n = static_cast<int64_t>(width) * height;
        const unsigned blocks = static_cast<unsigned>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        resample_loss_weight_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(source, src_width, src_height,
                                                                       output.ptr<float>(), width, height);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.loss_weight.resample");
    }

    void combine_loss_weight(const lfs::core::Tensor& weight, const lfs::core::Tensor& base, const int border,
                             lfs::core::Tensor& output) {
        using namespace lfs::core;
        LFS_ASSERT_MSG(weight.is_valid() && weight.device() == Device::CUDA && weight.dtype() == DataType::Float32 &&
                           weight.ndim() == 2 && weight.is_contiguous(),
                       "loss weight must be a contiguous [H,W] Float32 CUDA tensor");
        const bool has_base = base.is_valid() && base.numel() > 0;
        if (has_base) {
            LFS_ASSERT_MSG(base.device() == Device::CUDA && base.shape() == weight.shape() && base.is_contiguous() &&
                               (base.dtype() == DataType::Float32 || base.dtype() == DataType::UInt8 ||
                                base.dtype() == DataType::Bool),
                           "loss weight base must be a contiguous [H,W] Float32, UInt8 or Bool CUDA tensor");
        }
        const int h = static_cast<int>(weight.shape()[0]);
        const int w = static_cast<int>(weight.shape()[1]);
        const bool aliases_base = has_base && output.is_valid() && output.data_ptr() == base.data_ptr();
        if (!aliases_base && (!output.is_valid() || output.shape() != weight.shape() ||
                              output.dtype() != DataType::Float32 || output.device() != Device::CUDA))
            output = Tensor::empty(weight.shape(), Device::CUDA, DataType::Float32);
        LFS_ASSERT_MSG(output.dtype() == DataType::Float32, "loss weight output must be Float32");
        const cudaStream_t stream = resolve_stream(weight.stream());
        output.set_stream(stream);
        if (has_base)
            base.sync_to_stream(stream);
        const int64_t n = static_cast<int64_t>(w) * h;
        const unsigned blocks = static_cast<unsigned>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (!has_base) {
            combine_loss_weight_kernel<void><<<blocks, BLOCK_SIZE, 0, stream>>>(
                weight.ptr<float>(), nullptr, output.ptr<float>(), w, h, border);
        } else if (base.dtype() == DataType::Float32) {
            combine_loss_weight_kernel<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
                weight.ptr<float>(), base.ptr<float>(), output.ptr<float>(), w, h, border);
        } else {
            combine_loss_weight_kernel<uint8_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                weight.ptr<float>(), base.ptr<uint8_t>(), output.ptr<float>(), w, h, border);
        }
        LFS_CUDA_LAUNCH_CHECK(stream, "training.loss_weight.combine");
    }

} // namespace lfs::training::kernels
