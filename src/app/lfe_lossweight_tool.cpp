/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Developer check of the per-pixel loss weights (--loss-weight-dir; training/kernels/loss_weight_map.hpp).
//   LichtFeld-Studio lfe-lossweight check
//     Runs the trainer's weighted path (combine_loss_weight with the mask, structure_base_denominator, masked fused
//     L1 + SSIM forward and backward) on small random images with smooth weight maps, without / with a UInt8 / with a
//     soft Float32 mask, and compares it with:
//       1. a double-precision reference of the definition (loss and, by central differences, the gradient):
//            L = sum_p m_p w_p sum_c [(1 - l) |r - t| + l (1 - SSIM_cp)] / (3 sum_p m_p)
//          with the masked loss's SSIM (window statistics weighted by m w; outside the image weight 1, value 0);
//          without a mask m is 1 inside the 5 px valid-padding border and 0 on it, and the denominator counts the
//          valid region, as the unmasked loss;
//       2. the "hand-weighted" loss: the same sum with the SSIM map of the unweighted loss (window statistics weighted
//          by m only), i.e. per-pixel weighting of the default loss's terms;
//       3. weights all 1: the trainer's default loss (masked kernel with the mask alone, or the unmasked fused kernel);
//     plus the bilinear resample against a CPU reference.

#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "lfs/kernels/ssim.cuh"
#include "training/kernels/loss_weight_map.hpp"
#include "training/kernels/thin_structure.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace lfs::app {

    namespace {
        namespace tk = lfs::training::kernels;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr double LAMBDA = 0.2;
        constexpr double C1 = 0.01 * 0.01;
        constexpr double C2 = 0.03 * 0.03;
        constexpr int HALO = 5;
        constexpr double GAUSS[11] = {0.001028380123898387, 0.0075987582094967365, 0.036000773310661316,
                                      0.10936068743467331, 0.21300552785396576, 0.26601171493530273,
                                      0.21300552785396576, 0.10936068743467331, 0.036000773310661316,
                                      0.0075987582094967365, 0.001028380123898387};

        enum class MaskKind { None, UInt8, Float };

        struct WeightCase {
            std::string name;
            int height;
            int width;
            MaskKind mask;
            int grid_w; // weight map source size (resampled to H x W on the GPU)
            int grid_h;
            int fd_coordinates;
        };

        // Per-pixel SSIM (per channel) with window weights a (outside the image: weight 1, value 0).
        std::vector<double> ssim_maps(const std::vector<double>& r, const std::vector<double>& t,
                                      const std::vector<double>& a, const int H, const int W) {
            const int n = H * W;
            std::vector<double> s(3 * n, 1.0);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        double M = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
                        for (int dy = -HALO; dy <= HALO; ++dy)
                            for (int dx = -HALO; dx <= HALO; ++dx) {
                                const int yy = y + dy, xx = x + dx;
                                const bool in = yy >= 0 && yy < H && xx >= 0 && xx < W;
                                const double g = GAUSS[dy + HALO] * GAUSS[dx + HALO] * (in ? a[yy * W + xx] : 1.0);
                                const double X = in ? r[c * n + yy * W + xx] : 0.0;
                                const double Y = in ? t[c * n + yy * W + xx] : 0.0;
                                M += g;
                                sx += g * X;
                                sy += g * Y;
                                sxx += g * X * X;
                                syy += g * Y * Y;
                                sxy += g * X * Y;
                            }
                        const double mu1 = sx / M, mu2 = sy / M;
                        const double s11 = sxx / M - mu1 * mu1, s22 = syy / M - mu2 * mu2, s12 = sxy / M - mu1 * mu2;
                        s[c * n + y * W + x] = (2 * mu1 * mu2 + C1) * (2 * s12 + C2) /
                                               ((mu1 * mu1 + mu2 * mu2 + C1) * (s11 + s22 + C2));
                    }
            return s;
        }

        // sum_p lw_p sum_c [(1 - l)|r - t| + l (1 - S_cp)] / D, S from window weights a; pixels with lw = 0 skipped.
        double weighted_loss(const std::vector<double>& r, const std::vector<double>& t, const std::vector<double>& lw,
                             const std::vector<double>& a, const int H, const int W, const double D) {
            const int n = H * W;
            const auto S = ssim_maps(r, t, a, H, W);
            double sum = 0.0;
            for (int p = 0; p < n; ++p) {
                if (lw[p] == 0.0)
                    continue;
                double term = 0.0;
                for (int c = 0; c < 3; ++c)
                    term += (1.0 - LAMBDA) * std::abs(r[c * n + p] - t[c * n + p]) + LAMBDA * (1.0 - S[c * n + p]);
                sum += lw[p] * term;
            }
            return sum / D;
        }

        double bilinear(const std::vector<float>& src, const int sw, const int sh, const int x, const int y, const int W,
                        const int H) {
            if (sw == W && sh == H)
                return src[y * W + x];
            const double sx = std::clamp((x + 0.5) * static_cast<double>(sw) / W - 0.5, 0.0, sw - 1.0);
            const double sy = std::clamp((y + 0.5) * static_cast<double>(sh) / H - 0.5, 0.0, sh - 1.0);
            const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            const int x1 = std::min(x0 + 1, sw - 1), y1 = std::min(y0 + 1, sh - 1);
            const double fx = sx - x0, fy = sy - y0;
            const double top = (1 - fx) * src[y0 * sw + x0] + fx * src[y0 * sw + x1];
            const double bottom = (1 - fx) * src[y1 * sw + x0] + fx * src[y1 * sw + x1];
            return (1 - fy) * top + fy * bottom;
        }

        struct GpuOut {
            double loss;
            std::vector<float> grad;
            std::vector<float> combined;
            float denominator;
        };

        // The trainer's weighted path (Trainer::compute_photometric_loss_with_mask without thin structures).
        GpuOut gpu_weighted(const Tensor& r, const Tensor& t, const Tensor& base, const Tensor& weight) {
            const int H = static_cast<int>(r.shape()[1]), W = static_cast<int>(r.shape()[2]);
            const bool valid_padding = !base.is_valid();
            Tensor combined;
            tk::combine_loss_weight(weight, base, valid_padding ? 5 : 0, combined);
            const float D = tk::structure_base_denominator(base, H, W, valid_padding);
            tk::MaskedFusedL1SSIMWorkspace ws;
            auto [loss, ctx] = tk::masked_fused_l1_ssim_forward(r, t, combined, static_cast<float>(LAMBDA), ws, D);
            Tensor grad = tk::masked_fused_l1_ssim_backward(ctx, ws);
            GpuOut out;
            out.loss = loss.to_vector()[0];
            out.grad = grad.contiguous().to_vector();
            out.combined = combined.to_vector();
            out.denominator = D;
            return out;
        }

        int run_check() {
            std::mt19937 rng(4321);
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            std::vector<WeightCase> cases;
            cases.push_back({"no mask (valid padding), weights from a 9x7 grid", 41, 53, MaskKind::None, 9, 7, 300});
            cases.push_back({"UInt8 mask with holes, weights from a 6x5 grid", 44, 50, MaskKind::UInt8, 6, 5, 300});
            cases.push_back({"soft Float32 mask, full-size weight map", 38, 47, MaskKind::Float, 47, 38, 300});
            cases.push_back({"UInt8 mask, weights 0.5-4 from a 3x3 grid (steep)", 36, 44, MaskKind::UInt8, 3, 3, 300});

            int failures = 0;
            constexpr double H_STEP = 1e-6;
            std::println("loss weight check: lambda_dssim {}; GPU = the trainer's weighted path", LAMBDA);
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                const auto& c = cases[ci];
                const int H = c.height, W = c.width, n = H * W, N = 3 * n;
                // Smooth structure plus noise; render and target differ.
                std::vector<float> rf(N), tf(N);
                const float fx = 0.3f + 0.2f * u(rng), fy = 0.25f + 0.2f * u(rng);
                for (int ch = 0; ch < 3; ++ch)
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            const float base = 0.45f + 0.2f * std::sin(fx * x + 0.6f * ch) * std::cos(fy * y);
                            rf[ch * n + y * W + x] = std::clamp(base + 0.15f * (u(rng) - 0.5f), 0.0f, 1.0f);
                            tf[ch * n + y * W + x] = std::clamp(base + 0.06f * std::sin(0.8f * x + 1.1f * y) +
                                                                    0.2f * (u(rng) - 0.5f),
                                                                0.0f, 1.0f);
                        }
                // Mask: a rectangle hole plus scattered zeros (UInt8) / a soft ramp with a hole (Float).
                std::vector<float> mf;
                if (c.mask != MaskKind::None) {
                    mf.assign(n, 1.0f);
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            float v = 1.0f;
                            if (x > W / 3 && x < W / 3 + 8 && y > H / 4 && y < H / 4 + 10)
                                v = 0.0f;
                            if (c.mask == MaskKind::UInt8 && u(rng) < 0.05f)
                                v = 0.0f;
                            if (c.mask == MaskKind::Float && v > 0.0f)
                                v = 0.3f + 0.7f * static_cast<float>(x) / (W - 1);
                            mf[y * W + x] = v;
                        }
                }
                // Weight map source: smooth positive values in [0.5, 4].
                std::vector<float> src(c.grid_w * c.grid_h);
                for (auto& v : src)
                    v = 0.5f + 3.5f * u(rng) * u(rng);

                const Tensor r = Tensor::from_vector(rf, {3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
                const Tensor t = Tensor::from_vector(tf, {3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
                Tensor base;
                if (c.mask != MaskKind::None) {
                    base = Tensor::from_vector(mf, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
                    if (c.mask == MaskKind::UInt8)
                        base = base.to(DataType::UInt8);
                }
                const Tensor src_dev =
                    Tensor::from_vector(src, {static_cast<size_t>(c.grid_h), static_cast<size_t>(c.grid_w)}, Device::CUDA);
                Tensor weight;
                tk::resample_loss_weight_map(src_dev.ptr<float>(), c.grid_w, c.grid_h, weight, W, H);
                const auto wv = weight.to_vector();

                // Resample against the CPU bilinear reference.
                double resample_err = 0.0;
                std::vector<double> w(n);
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        w[y * W + x] = bilinear(src, c.grid_w, c.grid_h, x, y, W, H);
                        resample_err = std::max(resample_err, std::abs(w[y * W + x] - wv[y * W + x]));
                    }
                for (int p = 0; p < n; ++p)
                    w[p] = wv[p]; // the reference uses the GPU's (float) weights from here on

                // Reference weights: b = mask as the loss reads it (UInt8: != 0), or the valid-padding indicator.
                std::vector<double> b(n, 1.0);
                double b_sum = 0.0;
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        const int p = y * W + x;
                        if (c.mask == MaskKind::None)
                            b[p] = (x >= 5 && x < W - 5 && y >= 5 && y < H - 5) ? 1.0 : 0.0;
                        else
                            b[p] = c.mask == MaskKind::UInt8 ? (mf[p] != 0.0f ? 1.0 : 0.0) : mf[p];
                    }
                // Denominator as the trainer computes it: 3 x (mask sum or valid pixel count) + 1e-8.
                if (c.mask == MaskKind::None)
                    b_sum = static_cast<double>((H - 10) * (W - 10));
                else
                    for (int p = 0; p < n; ++p)
                        b_sum += b[p];
                const double D = 3.0 * b_sum + 1e-8;
                std::vector<double> bw(n);
                for (int p = 0; p < n; ++p)
                    bw[p] = b[p] * w[p];
                // Unweighted loss's window weights: the mask (without a mask the unmasked loss uses every pixel).
                std::vector<double> a_default(n);
                for (int p = 0; p < n; ++p)
                    a_default[p] = c.mask == MaskKind::None ? 1.0 : b[p];

                std::vector<double> rd(rf.begin(), rf.end());
                const std::vector<double> td(tf.begin(), tf.end());
                const GpuOut gpu = gpu_weighted(r, t, base, weight);
                double combine_err = 0.0;
                for (int p = 0; p < n; ++p)
                    combine_err = std::max(combine_err, std::abs(gpu.combined[p] - bw[p]));
                const double ref = weighted_loss(rd, td, bw, bw, H, W, D);
                const double fwd_rel = std::abs(gpu.loss - ref) / std::abs(ref);
                const double denom_rel = std::abs(gpu.denominator - D) / D;
                const double hand = weighted_loss(rd, td, bw, a_default, H, W, D);
                const double hand_rel = std::abs(gpu.loss - hand) / std::abs(hand);

                // Gradient: central differences of the reference.
                std::uniform_int_distribution<int> pick(0, N - 1);
                double num = 0, den = 0, max_fd = 0;
                std::vector<std::pair<double, double>> pairs;
                for (int k = 0; k < c.fd_coordinates; ++k) {
                    const int i = pick(rng);
                    const double keep = rd[i];
                    rd[i] = keep + H_STEP;
                    const double lp = weighted_loss(rd, td, bw, bw, H, W, D);
                    rd[i] = keep - H_STEP;
                    const double lm = weighted_loss(rd, td, bw, bw, H, W, D);
                    rd[i] = keep;
                    const double fd = (lp - lm) / (2 * H_STEP);
                    const double an = gpu.grad[i];
                    num += (an - fd) * (an - fd);
                    den += fd * fd;
                    max_fd = std::max(max_fd, std::abs(fd));
                    pairs.emplace_back(an, fd);
                }
                std::vector<double> rel;
                for (const auto& [an, fd] : pairs)
                    if (std::abs(fd) > 1e-2 * max_fd)
                        rel.push_back(std::abs(an - fd) / std::abs(fd));
                std::ranges::sort(rel);
                const double global_rel = std::sqrt(num / std::max(den, 1e-300));
                std::normal_distribution<double> gauss(0.0, 1.0);
                double dir_worst = 0.0;
                for (int k = 0; k < 3; ++k) {
                    std::vector<double> v(N), rp(rd), rm(rd);
                    double dot = 0.0;
                    for (int i = 0; i < N; ++i) {
                        v[i] = gauss(rng);
                        dot += v[i] * gpu.grad[i];
                        rp[i] += H_STEP * v[i];
                        rm[i] -= H_STEP * v[i];
                    }
                    const double fd = (weighted_loss(rp, td, bw, bw, H, W, D) - weighted_loss(rm, td, bw, bw, H, W, D)) /
                                      (2 * H_STEP);
                    dir_worst = std::max(dir_worst, std::abs(dot - fd) / std::abs(fd));
                }

                // Weights all 1 against the default loss.
                const Tensor ones = Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
                const GpuOut unit = gpu_weighted(r, t, base, ones);
                double def_loss = 0.0;
                std::vector<float> def_grad;
                if (c.mask == MaskKind::None) {
                    tk::FusedL1SSIMWorkspace fws;
                    auto [l, ctx] = tk::fused_l1_ssim_forward(r, t, static_cast<float>(LAMBDA), fws, true);
                    def_loss = l.to_vector()[0];
                    def_grad = tk::fused_l1_ssim_backward(ctx, fws).contiguous().to_vector();
                } else {
                    tk::MaskedFusedL1SSIMWorkspace mws;
                    auto [l, ctx] = tk::masked_fused_l1_ssim_forward(r, t, base, static_cast<float>(LAMBDA), mws, 0.0f);
                    def_loss = l.to_vector()[0];
                    def_grad = tk::masked_fused_l1_ssim_backward(ctx, mws).contiguous().to_vector();
                }
                const double unit_rel = std::abs(unit.loss - def_loss) / std::abs(def_loss);
                double gnum = 0, gden = 0, gmax = 0;
                for (int i = 0; i < N; ++i) {
                    const double d = unit.grad[i] - def_grad[i];
                    gnum += d * d;
                    gden += static_cast<double>(def_grad[i]) * def_grad[i];
                    gmax = std::max(gmax, std::abs(d));
                }
                const double unit_grad_rel = std::sqrt(gnum / std::max(gden, 1e-300));

                // Without a mask the unmasked loss's valid padding is emulated by zero weights on the 5 px border, so
                // SSIM windows near it no longer see the border pixels: a border-band difference, large on these tiny
                // images and checked at a training size below; with a mask the kernels and inputs are the default's.
                const bool unit_ok = c.mask == MaskKind::None ? true : (unit_rel < 1e-6 && unit_grad_rel < 1e-5);
                const bool ok = resample_err < 1e-5 && combine_err < 1e-6 && denom_rel < 1e-6 && fwd_rel < 1e-5 &&
                                global_rel < 1e-2 && dir_worst < 1e-2 && unit_ok;
                if (!ok)
                    ++failures;
                std::println("case {} [{}] {}x{}:", ci + 1, c.name, W, H);
                std::println("    resample max abs err {:.1e}; combined weight max abs err {:.1e}; denominator rel err {:.1e}",
                             resample_err, combine_err, denom_rel);
                std::println("    weighted loss {:.7g}, reference {:.7g}: rel err {:.2e}", gpu.loss, ref, fwd_rel);
                std::println("    gradient ({} coords): global rel err {:.2e}, median/max rel err {:.2e} / {:.2e} (|fd| > 1% of max); "
                             "3 random directions worst rel err {:.2e}",
                             pairs.size(), global_rel, rel.empty() ? 0.0 : rel[rel.size() / 2], rel.empty() ? 0.0 : rel.back(),
                             dir_worst);
                std::println("    hand-weighted loss (default SSIM map x weights) {:.7g}: differs by {:.2e} relative "
                             "(the weighted path also weights the SSIM window statistics)",
                             hand, hand_rel);
                std::println("    weights all 1: {:.7g} vs default {} loss {:.7g}: rel diff {:.2e}; gradient rel diff {:.2e} (max abs {:.1e})  {}",
                             unit.loss, c.mask == MaskKind::None ? "unmasked" : "masked", def_loss, unit_rel, unit_grad_rel, gmax,
                             ok ? "OK" : "FAIL");
            }
            // Weights all 1 without a mask at training sizes: the border-band difference from the unmasked loss.
            for (const auto [W, H] : {std::pair{1458, 866}, std::pair{2916, 1733}}) {
                const size_t n = static_cast<size_t>(W) * H;
                std::vector<float> rf(3 * n), tf(3 * n);
                for (int ch = 0; ch < 3; ++ch)
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            const float base = 0.45f + 0.2f * std::sin(0.05f * x + 0.6f * ch) * std::cos(0.04f * y);
                            const size_t i = ch * n + static_cast<size_t>(y) * W + x;
                            rf[i] = std::clamp(base + 0.15f * (u(rng) - 0.5f), 0.0f, 1.0f);
                            tf[i] = std::clamp(base + 0.2f * (u(rng) - 0.5f), 0.0f, 1.0f);
                        }
                const Tensor r = Tensor::from_vector(rf, {3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
                const Tensor t = Tensor::from_vector(tf, {3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
                const Tensor ones = Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
                const GpuOut unit = gpu_weighted(r, t, Tensor{}, ones);
                tk::FusedL1SSIMWorkspace fws;
                auto [l, ctx] = tk::fused_l1_ssim_forward(r, t, static_cast<float>(LAMBDA), fws, true);
                const double def_loss = l.to_vector()[0];
                const auto def_grad = tk::fused_l1_ssim_backward(ctx, fws).contiguous().to_vector();
                double gnum = 0, gden = 0, inner_num = 0, inner_den = 0;
                for (int ch = 0; ch < 3; ++ch)
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            const size_t i = ch * n + static_cast<size_t>(y) * W + x;
                            const double d = unit.grad[i] - def_grad[i];
                            gnum += d * d;
                            gden += static_cast<double>(def_grad[i]) * def_grad[i];
                            if (x >= 15 && x < W - 15 && y >= 15 && y < H - 15) {
                                inner_num += d * d;
                                inner_den += static_cast<double>(def_grad[i]) * def_grad[i];
                            }
                        }
                const double rel = std::abs(unit.loss - def_loss) / def_loss;
                const double inner = std::sqrt(inner_num / std::max(inner_den, 1e-300));
                const bool ok = rel < 1e-3 && inner < 1e-3;
                if (!ok)
                    ++failures;
                std::println("no mask, weights all 1, {}x{}: loss {:.7g} vs unmasked {:.7g}: rel diff {:.2e}; gradient rel diff "
                             "{:.2e} overall, {:.2e} more than 15 px from the edge  {}",
                             W, H, unit.loss, def_loss, rel, std::sqrt(gnum / std::max(gden, 1e-300)), inner, ok ? "OK" : "FAIL");
            }
            if (failures == 0)
                std::println("LOSSWEIGHT CHECK OK");
            else
                std::println("LOSSWEIGHT CHECK: {} of {} cases failed", failures, cases.size());
            return failures == 0 ? 0 : 1;
        }
        // Kernel time of the photometric loss (forward + backward) at a full training resolution: the default masked
        // loss (Float32 mask, mostly 1) against the weighted one (resample of a 729x433 map + combine + denominator +
        // the same kernels, whose SSIM windows then take the weighted path). CUDA events, best of `reps` x 20.
        int run_bench(const int W, const int H) {
            std::mt19937 rng(99);
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            const size_t n = static_cast<size_t>(W) * H;
            std::vector<float> rf(3 * n), tf(3 * n), mf(n, 1.0f), src(729 * 433);
            for (size_t i = 0; i < 3 * n; ++i) {
                rf[i] = u(rng);
                tf[i] = u(rng);
            }
            for (size_t i = 0; i < n / 50; ++i)
                mf[static_cast<size_t>(u(rng) * (n - 1))] = 0.0f;
            for (auto& v : src)
                v = 0.5f + 3.5f * u(rng) * u(rng);
            const Tensor r = Tensor::from_vector(rf, {3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
            const Tensor t = Tensor::from_vector(tf, {3, static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
            const Tensor mask = Tensor::from_vector(mf, {static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
            const Tensor src_dev = Tensor::from_vector(src, {433, 729}, Device::CUDA);
            tk::MaskedFusedL1SSIMWorkspace ws;
            Tensor weight, combined;
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const auto time = [&](const bool weighted) {
                float best = 1e30f;
                for (int rep = 0; rep < 5; ++rep) {
                    cudaDeviceSynchronize();
                    cudaEventRecord(e0, lfs::core::getCurrentCUDAStream());
                    for (int k = 0; k < 20; ++k) {
                        Tensor m = mask;
                        float D = 0.0f;
                        if (weighted) {
                            tk::resample_loss_weight_map(src_dev.ptr<float>(), 729, 433, weight, W, H);
                            tk::combine_loss_weight(weight, mask, 0, combined);
                            D = tk::structure_base_denominator(mask, H, W, false);
                            m = combined;
                        }
                        auto [loss, ctx] = tk::masked_fused_l1_ssim_forward(r, t, m, static_cast<float>(LAMBDA), ws, D);
                        (void)tk::masked_fused_l1_ssim_backward(ctx, ws);
                    }
                    cudaEventRecord(e1, lfs::core::getCurrentCUDAStream());
                    cudaEventSynchronize(e1);
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, e0, e1);
                    best = std::min(best, ms / 20.0f);
                }
                return best;
            };
            time(false);
            const float base_ms = time(false);
            const float weighted_ms = time(true);
            std::println("loss kernels at {}x{} (forward + backward, best of 5 x 20): masked {:.3f} ms, weighted {:.3f} ms "
                         "(+{:.3f} ms)",
                         W, H, base_ms, weighted_ms, weighted_ms - base_ms);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return 0;
        }
    } // namespace

    int run_lfe_lossweight_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "check")
            return run_check();
        if (argc >= 3 && std::string(argv[2]) == "bench") {
            const int W = argc >= 5 ? std::stoi(argv[3]) : 5832;
            const int H = argc >= 5 ? std::stoi(argv[4]) : 3466;
            return run_bench(W, H);
        }
        std::println(stderr, "usage: LichtFeld-Studio lfe-lossweight check | bench [W H]");
        return 2;
    }

} // namespace lfs::app
