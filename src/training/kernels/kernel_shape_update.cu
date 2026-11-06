/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// --kernel-shape (LFQPLCYK): the Adam step and the histogram of the per-splat shape (kernel_shape_update.hpp).

#include "core/cuda_error.hpp"
#include "core/kernel_shape_map.hpp"
#include "kernel_shape_update.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>

namespace lfs::training::kernels {

    namespace {
        namespace ks = lfs::core::kshape;

        __global__ void kernel_shape_adam_kernel(float* __restrict__ raw, float2* __restrict__ adam, float* __restrict__ grad,
                                                 const std::int64_t n, const KernelShapeAdam a) {
            const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const float g_beta = grad[i];
            if (g_beta == 0.0f)
                return;
            grad[i] = 0.0f;
            const float r = raw[i];
            const float g = g_beta * ks::dbeta_draw(ks::beta_from_raw(r));
            if (!(g == g) || a.lr == 0.0f) // non-finite: drop this view's gradient for this splat
                return;
            float2 mv = adam[i];
            mv.x = a.beta1 * mv.x + (1.0f - a.beta1) * g;
            mv.y = a.beta2 * mv.y + (1.0f - a.beta2) * g * g;
            adam[i] = mv;
            const float m_hat = mv.x / a.bias1;
            const float v_hat = mv.y / a.bias2;
            raw[i] = r - a.lr * m_hat / (sqrtf(v_hat) + a.eps);
        }

        __global__ void kernel_shape_beta_kernel(const float* __restrict__ raw, float* __restrict__ beta, const std::int64_t n) {
            const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n)
                beta[i] = ks::beta_from_raw(raw[i]);
        }

        struct HistDevice {
            unsigned long long bins[kBetaHistBins];
            unsigned long long exactly_two;
            unsigned long long count;
            double sum;
            unsigned int min_bits, max_bits; // positive floats order as their bits
        };

        __global__ void kernel_shape_hist_kernel(const float* __restrict__ raw, const std::uint8_t* __restrict__ deleted,
                                                 const std::int64_t n, HistDevice* out) {
            __shared__ unsigned int s_bins[kBetaHistBins];
            __shared__ unsigned int s_two, s_count;
            __shared__ float s_sum[256];
            if (threadIdx.x < kBetaHistBins)
                s_bins[threadIdx.x] = 0u;
            if (threadIdx.x == 0) {
                s_two = 0u;
                s_count = 0u;
            }
            __syncthreads();
            float local = 0.0f;
            unsigned int lmin = 0x7f7fffffu, lmax = 0u;
            for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
                if (deleted != nullptr && deleted[i] != 0)
                    continue;
                const float b = ks::beta_from_raw(raw[i]);
                // the bin edges of kBetaHistEdges (a device-local copy)
                constexpr float edges[kBetaHistBins + 1] = {1.0f, 1.25f, 1.5f, 1.75f, 1.95f, 2.05f, 2.5f, 3.0f,
                                                            4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                int k = 0;
                while (k < kBetaHistBins - 1 && b >= edges[k + 1])
                    ++k;
                atomicAdd(&s_bins[k], 1u);
                atomicAdd(&s_count, 1u);
                if (b == 2.0f)
                    atomicAdd(&s_two, 1u);
                local += b;
                lmin = min(lmin, __float_as_uint(b));
                lmax = max(lmax, __float_as_uint(b));
            }
            s_sum[threadIdx.x] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (static_cast<int>(threadIdx.x) < stride)
                    s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
                __syncthreads();
            }
            atomicMin(&out->min_bits, lmin);
            atomicMax(&out->max_bits, lmax);
            if (threadIdx.x < kBetaHistBins)
                atomicAdd(&out->bins[threadIdx.x], static_cast<unsigned long long>(s_bins[threadIdx.x]));
            if (threadIdx.x == 0) {
                atomicAdd(&out->exactly_two, static_cast<unsigned long long>(s_two));
                atomicAdd(&out->count, static_cast<unsigned long long>(s_count));
                atomicAdd(&out->sum, static_cast<double>(s_sum[0]));
            }
        }

        void check(const cudaError_t e, const char* what) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("kernel shape: ") + what + ": " + cudaGetErrorString(e));
        }
    } // namespace

    void kernel_shape_adam(float* raw, float* adam, float* grad, const std::int64_t n, const KernelShapeAdam& a,
                           cudaStream_t stream) {
        if (n <= 0)
            return;
        constexpr int kBlock = 256;
        const auto blocks = static_cast<unsigned int>((n + kBlock - 1) / kBlock);
        kernel_shape_adam_kernel<<<blocks, kBlock, 0, stream>>>(raw, reinterpret_cast<float2*>(adam), grad, n, a);
        check(cudaGetLastError(), "adam launch");
    }

    void kernel_shape_beta(const float* raw, float* beta, const std::int64_t n, cudaStream_t stream) {
        if (n <= 0)
            return;
        constexpr int kBlock = 256;
        const auto blocks = static_cast<unsigned int>((n + kBlock - 1) / kBlock);
        kernel_shape_beta_kernel<<<blocks, kBlock, 0, stream>>>(raw, beta, n);
        check(cudaGetLastError(), "beta launch");
    }

    KernelShapeHistogram kernel_shape_histogram(const float* raw, const std::uint8_t* deleted, const std::int64_t n,
                                                cudaStream_t stream) {
        KernelShapeHistogram h{};
        if (n <= 0 || raw == nullptr)
            return h;
        HistDevice init{};
        init.min_bits = 0x7f7fffffu;
        init.max_bits = 0u;
        HistDevice* d = nullptr;
        check(cudaMallocAsync(reinterpret_cast<void**>(&d), sizeof(HistDevice), stream), "histogram alloc");
        check(cudaMemcpyAsync(d, &init, sizeof(HistDevice), cudaMemcpyHostToDevice, stream), "histogram init");
        const auto blocks = static_cast<unsigned int>(std::min<std::int64_t>((n + 255) / 256, 1024));
        kernel_shape_hist_kernel<<<blocks, 256, 0, stream>>>(raw, deleted, n, d);
        check(cudaGetLastError(), "histogram launch");
        HistDevice out{};
        check(cudaMemcpyAsync(&out, d, sizeof(HistDevice), cudaMemcpyDeviceToHost, stream), "histogram copy");
        check(cudaStreamSynchronize(stream), "histogram sync");
        check(cudaFreeAsync(d, stream), "histogram free");
        for (int k = 0; k < kBetaHistBins; ++k)
            h.bins[k] = out.bins[k];
        h.exactly_two = out.exactly_two;
        h.count = out.count;
        h.sum = out.sum;
        h.min = out.count ? std::bit_cast<float>(out.min_bits) : 0.0f;
        h.max = out.count ? std::bit_cast<float>(out.max_bits) : 0.0f;
        return h;
    }

} // namespace lfs::training::kernels
