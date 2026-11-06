/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// --kernel-shape (LFQPLCYK): a per-splat trainable kernel shape, alpha = o exp(-1/2 d^beta) (FastGS kernel_shape.cuh,
// core/kernel_shape_map.hpp).
//
// The state lives on the model (core::SplatData, so every row operation, checkpoint and export can carry it):
//   _shape      [N] fp32  raw r, beta = 1 + 7 sigmoid(r - ln 6) in (1, 8); r = 0 -> beta = 2 exactly;
//   _shape_adam [N, 2]    Adam first / second moments of r;
//   _shape_step           Adam steps taken (bias correction).
// Every FastGS render of a model that carries _shape draws it with its beta (training/rasterization/fast_rasterizer.cpp);
// this class adds the training side: the gradient buffer the backward fills (dL/dbeta per splat), the Adam step after
// each backward, and the beta histogram for the log.
#pragma once

#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <string>

namespace lfs::training {

    struct KernelShapeSettings {
        float lr = 0.005f;    // Adam learning rate of the raw r (beta moves ~6/7 of that per step near 2)
        int start = 0;        // first iteration that updates beta (before it: frozen, gradients dropped)
        int log_every = 1000; // beta histogram every this many iterations
    };

    class KernelShape {
    public:
        // Ensures the model carries the shape state: r = 0 (beta = 2) for every splat unless the model already has one
        // (a PLY with a `shape` property, a checkpoint); Adam moments zero if missing.
        static std::expected<std::unique_ptr<KernelShape>, std::string> create(const KernelShapeSettings& settings,
                                                                               core::SplatData& model);

        // The feature line: "Kernel shape (--kernel-shape): beta in [1, 8], init 2, lr ..."
        [[nodiscard]] std::string description(const core::SplatData& model) const;

        // The backward's dL/dbeta buffer for this view ([N], zero on entry; the step zeroes it again).
        [[nodiscard]] float* grad_for(core::SplatData& model, cudaStream_t stream);

        // One Adam step of r from the buffer (after each backward); frozen (gradients dropped) before settings.start or
        // with lr 0.
        void step(core::SplatData& model, int iter, cudaStream_t stream);

        // "beta: mean ..., [1,1.25) n ..., ..." (blocking; the log, every settings.log_every)
        [[nodiscard]] std::string summary(const core::SplatData& model, cudaStream_t stream) const;

        [[nodiscard]] const KernelShapeSettings& settings() const { return settings_; }

    private:
        explicit KernelShape(const KernelShapeSettings& s) : settings_(s) {}
        KernelShapeSettings settings_;
        core::Tensor grad_;
    };

    // Row operations on the model's shape state (each a no-op when the model carries none), for the densification
    // strategy (MRNF) and the reorders. Children inherit their parent's r with fresh (zero) Adam moments, as the
    // strategy treats every other parameter of a new row.
    namespace kernel_shape_rows {
        // r of the given rows (int64 [K]) -> [K] (an invalid tensor without a shape state)
        [[nodiscard]] core::Tensor gather_raw(const core::SplatData& model, const core::Tensor& rows);
        // r = raw[k] at rows[k] for k < rows.numel() (raw may be longer), Adam moments zeroed there; raw invalid = r 0
        void fill(core::SplatData& model, const core::Tensor& rows, const core::Tensor& raw);
        // appends raw[start, end) (raw invalid = r 0) with zero Adam moments
        void append(core::SplatData& model, const core::Tensor& raw, size_t start, size_t end);
        // zeroes the Adam moments at rows
        void reset_adam(core::SplatData& model, const core::Tensor& rows);
        // throws std::runtime_error naming `where` if the shape state does not have one row per splat
        void check(const core::SplatData& model, const char* where);
    } // namespace kernel_shape_rows

} // namespace lfs::training
