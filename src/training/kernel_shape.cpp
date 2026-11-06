/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "kernel_shape.hpp"

#include "core/kernel_shape_map.hpp"
#include "core/logger.hpp"
#include "kernels/kernel_shape_update.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

namespace lfs::training {

    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::core::TensorShape;

        size_t rows_of(const core::SplatData& model) { return static_cast<size_t>(model.size()); }
    } // namespace

    std::expected<std::unique_ptr<KernelShape>, std::string> KernelShape::create(const KernelShapeSettings& settings,
                                                                                 core::SplatData& model) {
        if (!std::isfinite(settings.lr) || settings.lr < 0.0f)
            return std::unexpected(std::format("--kernel-shape-lr must be finite and >= 0 (got {})", settings.lr));
        const size_t n = rows_of(model);
        if (model._shape.is_valid() && static_cast<size_t>(model._shape.numel()) != n)
            return std::unexpected(std::format("the model's kernel shape has {} rows for {} splats", model._shape.numel(), n));
        if (!model._shape.is_valid())
            model._shape = Tensor::zeros(TensorShape({n}), Device::CUDA, DataType::Float32);
        if (!model._shape_adam.is_valid() || static_cast<size_t>(model._shape_adam.size(0)) != n) {
            model._shape_adam = Tensor::zeros(TensorShape({n, 2}), Device::CUDA, DataType::Float32);
            model._shape_step = 0;
        }
        return std::unique_ptr<KernelShape>(new KernelShape(settings));
    }

    std::string KernelShape::description(const core::SplatData& model) const {
        cudaDeviceSynchronize(); // once, at start-up: the state's allocation / zeroing may be on another stream
        const auto h = kernels::kernel_shape_histogram(model._shape.ptr<float>(), nullptr,
                                                       static_cast<std::int64_t>(model._shape.numel()), nullptr);
        return std::format(
            "Kernel shape (--kernel-shape): beta in [1, 8], init 2, lr {} (Adam on r, beta = 1 + 7 sigmoid(r - ln 6); own "
            "state: beta1 0.9, beta2 0.999, eps 1e-15), updates from iteration {}; alpha = o exp(-1/2 d^beta) in the "
            "forward and backward of every FastGS render of this model; extent d_max = (2 ln(255 o))^(1/beta); the 0.3 px "
            "dilation still added to the 2D covariance (an approximation for beta != 2); {} splats, {} at beta = 2 now, "
            "Adam step {}",
            settings_.lr, settings_.start, h.count, h.exactly_two, model._shape_step);
    }

    float* KernelShape::grad_for(core::SplatData& model, cudaStream_t /*stream*/) {
        const size_t n = rows_of(model);
        if (!model._shape.is_valid()) { // a model replaced after create() (resume from a checkpoint without the state)
            LOG_WARN("--kernel-shape: the model carries no kernel shape state (resumed from a checkpoint without it?); "
                     "starting every splat at beta = 2");
            model._shape = Tensor::zeros(TensorShape({n}), Device::CUDA, DataType::Float32);
            model._shape_adam = Tensor::zeros(TensorShape({n, 2}), Device::CUDA, DataType::Float32);
            model._shape_step = 0;
        }
        kernel_shape_rows::check(model, "kernel shape gradient");
        if (!grad_.is_valid() || static_cast<size_t>(grad_.numel()) < n) {
            // zero on allocation; the step zeroes what the backward wrote
            grad_ = Tensor::zeros(TensorShape({std::max<size_t>(n + n / 4, 1)}), Device::CUDA, DataType::Float32);
        }
        return grad_.ptr<float>();
    }

    void KernelShape::step(core::SplatData& model, const int iter, cudaStream_t stream) {
        if (!grad_.is_valid())
            return;
        const size_t n = rows_of(model);
        kernel_shape_rows::check(model, "kernel shape step");
        if (n == 0)
            return;
        if (iter < settings_.start || settings_.lr == 0.0f) {
            // frozen: drop this view's gradient
            if (cudaMemsetAsync(grad_.ptr<float>(), 0, sizeof(float) * n, stream) != cudaSuccess)
                throw std::runtime_error("kernel shape: gradient reset failed");
            return;
        }
        ++model._shape_step;
        kernels::KernelShapeAdam a;
        a.lr = settings_.lr;
        a.bias1 = 1.0f - static_cast<float>(std::pow(static_cast<double>(a.beta1), static_cast<double>(model._shape_step)));
        a.bias2 = 1.0f - static_cast<float>(std::pow(static_cast<double>(a.beta2), static_cast<double>(model._shape_step)));
        kernels::kernel_shape_adam(model._shape.ptr<float>(), model._shape_adam.ptr<float>(), grad_.ptr<float>(),
                                   static_cast<std::int64_t>(n), a, stream);
    }

    std::string KernelShape::summary(const core::SplatData& model, cudaStream_t stream) const {
        const std::uint8_t* deleted = nullptr;
        if (model.has_deleted_mask() && static_cast<size_t>(model.deleted().numel()) == rows_of(model) &&
            model.deleted().dtype() == DataType::Bool)
            deleted = reinterpret_cast<const std::uint8_t*>(model.deleted().data_ptr());
        const auto h = kernels::kernel_shape_histogram(model._shape.ptr<float>(), deleted,
                                                       static_cast<std::int64_t>(model._shape.numel()), stream);
        std::string bins;
        for (int k = 0; k < kernels::kBetaHistBins; ++k)
            bins += std::format("{}[{:g},{:g}) {}", k ? ", " : "", kernels::kBetaHistEdges[k], kernels::kBetaHistEdges[k + 1],
                                h.bins[k]);
        unsigned long long sharp = 0, soft = 0;
        for (int k = 0; k < kernels::kBetaHistBins; ++k) {
            if (kernels::kBetaHistEdges[k] >= 2.05f)
                sharp += h.bins[k];
            if (kernels::kBetaHistEdges[k + 1] <= 1.95f)
                soft += h.bins[k];
        }
        const double cnt = h.count ? static_cast<double>(h.count) : 1.0;
        return std::format("kernel shape beta over {} live splats: mean {:.4f}, min {:.4f}, max {:.4f}; soft (< 1.95) {:.2f}%, "
                           "~Gaussian [1.95, 2.05) {:.2f}%, sharp (>= 2.05) {:.2f}%, exactly 2 {}; histogram {}; Adam step {}",
                           h.count, h.sum / cnt, h.min, h.max, 100.0 * soft / cnt,
                           100.0 * (static_cast<double>(h.count) - soft - sharp) / cnt, 100.0 * sharp / cnt, h.exactly_two,
                           bins, model._shape_step);
    }

    namespace kernel_shape_rows {

        Tensor gather_raw(const core::SplatData& model, const Tensor& rows) {
            if (!model._shape.is_valid() || !rows.is_valid())
                return {};
            if (rows.numel() == 0)
                return Tensor::zeros(TensorShape({size_t{0}}), Device::CUDA, DataType::Float32);
            return model._shape.index_select(0, rows);
        }

        void fill(core::SplatData& model, const Tensor& rows, const Tensor& raw) {
            if (!model._shape.is_valid() || !rows.is_valid() || rows.numel() == 0)
                return;
            const size_t k = static_cast<size_t>(rows.numel());
            Tensor values = raw.is_valid() ? raw.slice(0, 0, k).contiguous()
                                           : Tensor::zeros(TensorShape({k}), Device::CUDA, DataType::Float32);
            model._shape.index_put_(rows, values);
            if (model._shape_adam.is_valid())
                model._shape_adam.index_put_(rows, Tensor::zeros(TensorShape({k, 2}), Device::CUDA, DataType::Float32));
        }

        void append(core::SplatData& model, const Tensor& raw, const size_t start, const size_t end) {
            if (!model._shape.is_valid() || end <= start)
                return;
            const size_t k = end - start;
            Tensor values = raw.is_valid() ? raw.slice(0, start, end).contiguous()
                                           : Tensor::zeros(TensorShape({k}), Device::CUDA, DataType::Float32);
            model._shape = Tensor::cat({model._shape, values}, 0);
            if (model._shape_adam.is_valid())
                model._shape_adam = Tensor::cat(
                    {model._shape_adam, Tensor::zeros(TensorShape({k, 2}), Device::CUDA, DataType::Float32)}, 0);
        }

        void reset_adam(core::SplatData& model, const Tensor& rows) {
            if (!model._shape_adam.is_valid() || !rows.is_valid() || rows.numel() == 0)
                return;
            const size_t k = static_cast<size_t>(rows.numel());
            model._shape_adam.index_put_(rows, Tensor::zeros(TensorShape({k, 2}), Device::CUDA, DataType::Float32));
        }

        void check(const core::SplatData& model, const char* where) {
            if (!model._shape.is_valid())
                return;
            const size_t n = rows_of(model);
            const bool ok = static_cast<size_t>(model._shape.numel()) == n &&
                            (!model._shape_adam.is_valid() || static_cast<size_t>(model._shape_adam.size(0)) == n);
            if (!ok)
                throw std::runtime_error(std::format("{}: the kernel shape state has {} rows (Adam {}) for {} splats; a row "
                                                     "operation did not carry it",
                                                     where, model._shape.numel(),
                                                     model._shape_adam.is_valid() ? model._shape_adam.size(0) : 0, n));
        }

    } // namespace kernel_shape_rows

} // namespace lfs::training
