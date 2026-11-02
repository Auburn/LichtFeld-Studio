/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Developer check of the generated-view loss (--generated-prefix; training/kernels/generated_view_loss.hpp).
//   LichtFeld-Studio lfe-genloss gradcheck
//     Small random images, masked and unmasked, Float32 and UInt8 targets and masks, several filter sizes and
//     weights. The GPU loss is compared with a double-precision reference that evaluates the definition directly
//     (2-D windows, no separable filtering), and the GPU's analytic gradient w.r.t. the render with central finite
//     differences of that reference: per coordinate on a random sample, and along random directions.

#include "core/tensor.hpp"
#include "training/kernels/generated_view_loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <print>
#include <random>
#include <string>
#include <vector>

namespace lfs::app {

    namespace {
        namespace gk = lfs::training::kernels;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        enum class MaskKind { None, Float, UInt8 };

        struct GenCase {
            std::string name;
            int height;
            int width;
            MaskKind mask;
            bool uint8_target;
            bool flat_patch; // a constant render block: exercises the variance clamp
            gk::GeneratedViewLossParams params;
            int fd_coordinates; // 0 = all
        };

        struct Inputs {
            std::vector<float> render;  // [3,H,W]
            std::vector<float> target;  // [3,H,W], as the GPU sees it (UInt8 targets already /255)
            std::vector<float> target_raw; // what is uploaded (0..255 integers for UInt8)
            std::vector<float> mask;    // [H,W] weights as the GPU sees them (UInt8: nonzero = 1); empty = none
            std::vector<float> mask_raw;
        };

        Inputs make_inputs(const GenCase& c, std::mt19937& rng) {
            const int H = c.height, W = c.width, n = H * W;
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            Inputs in;
            in.render.resize(3 * n);
            in.target.resize(3 * n);
            in.target_raw.resize(3 * n);
            const float fx = 0.37f + 0.2f * u(rng), fy = 0.29f + 0.2f * u(rng);
            for (int ch = 0; ch < 3; ++ch) {
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        const int i = ch * n + y * W + x;
                        // Smooth structure plus fine "texture" of different amounts in render and target.
                        const float base = 0.45f + 0.15f * std::sin(fx * x + 0.7f * ch) * std::cos(fy * y);
                        in.render[i] = std::clamp(base + 0.12f * (u(rng) - 0.5f), 0.0f, 1.0f);
                        float t = base + 0.05f * std::sin(0.9f * x + 1.3f * y) + 0.25f * (u(rng) - 0.5f);
                        t = std::clamp(t, 0.0f, 1.0f);
                        if (c.uint8_target) {
                            const float q = std::round(t * 255.0f);
                            in.target_raw[i] = q;
                            in.target[i] = q * (1.0f / 255.0f);
                        } else {
                            in.target_raw[i] = t;
                            in.target[i] = t;
                        }
                    }
                }
            }
            if (c.flat_patch) {
                const int y0 = H / 4, x0 = W / 4;
                for (int ch = 0; ch < 3; ++ch)
                    for (int y = y0; y < std::min(H, y0 + 11); ++y)
                        for (int x = x0; x < std::min(W, x0 + 11); ++x)
                            in.render[ch * n + y * W + x] = 0.4f + 0.05f * ch;
            }
            if (c.mask != MaskKind::None) {
                in.mask.resize(n);
                in.mask_raw.resize(n);
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        const int i = y * W + x;
                        float m = 1.0f;
                        if (x >= W / 2 && x < W / 2 + W / 5 && y >= H / 3 && y < H / 3 + H / 4)
                            m = 0.0f; // a hole
                        if (u(rng) < 0.1f)
                            m = 0.0f; // scattered ignored pixels
                        if (c.mask == MaskKind::Float && y < 3 && m > 0.0f)
                            m = 0.3f + 0.4f * u(rng); // fractional weights (ROI-like) along the top
                        in.mask_raw[i] = m;
                        in.mask[i] = c.mask == MaskKind::UInt8 ? (m != 0.0f ? 1.0f : 0.0f) : m;
                    }
                }
            }
            return in;
        }

        // The definition, in double, with direct 2-D windows.
        struct RefTerms {
            double loss, low, tex, pix, std_r, std_t;
        };

        RefTerms reference(const std::vector<double>& r, const std::vector<double>& t, const std::vector<double>& m,
                           const int H, const int W, const gk::GeneratedViewLossParams& p) {
            const int n = H * W;
            int rg = 0;
            std::vector<double> wg(1, 1.0);
            if (p.lowpass_sigma > 0.0f) {
                rg = std::min(gk::GENERATED_VIEW_MAX_RADIUS, static_cast<int>(std::ceil(3.0f * p.lowpass_sigma)));
                wg.resize(rg + 1);
                for (int d = 0; d <= rg; ++d)
                    wg[d] = std::exp(-0.5 * d * d / (static_cast<double>(p.lowpass_sigma) * p.lowpass_sigma));
            }
            const int rb = p.texture_radius;
            const auto mw = [&](const int i) { return m.empty() ? 1.0 : m[i]; };
            const auto luma = [&](const std::vector<double>& img, const int i) {
                return gk::GENERATED_VIEW_LUMA[0] * img[i] + gk::GENERATED_VIEW_LUMA[1] * img[n + i] +
                       gk::GENERATED_VIEW_LUMA[2] * img[2 * n + i];
            };
            double S = 0.0, low = 0.0, tex = 0.0, pix = 0.0, sr_sum = 0.0, st_sum = 0.0;
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    const int i = y * W + x;
                    const double mp = mw(i);
                    S += mp;
                    for (int c = 0; c < 3; ++c)
                        pix += mp * std::abs(r[c * n + i] - t[c * n + i]);
                    if (mp == 0.0)
                        continue;
                    double Mg = 0.0, gr[3] = {0, 0, 0}, gt[3] = {0, 0, 0};
                    for (int dy = -rg; dy <= rg; ++dy) {
                        const int yy = y + dy;
                        if (yy < 0 || yy >= H)
                            continue;
                        for (int dx = -rg; dx <= rg; ++dx) {
                            const int xx = x + dx;
                            if (xx < 0 || xx >= W)
                                continue;
                            const int q = yy * W + xx;
                            const double k = wg[std::abs(dy)] * wg[std::abs(dx)] * mw(q);
                            Mg += k;
                            for (int c = 0; c < 3; ++c) {
                                gr[c] += k * r[c * n + q];
                                gt[c] += k * t[c * n + q];
                            }
                        }
                    }
                    for (int c = 0; c < 3; ++c)
                        low += mp * std::abs((gr[c] - gt[c]) / Mg);
                    double Mb = 0.0, e1r = 0.0, e2r = 0.0, e1t = 0.0, e2t = 0.0;
                    for (int yy = std::max(0, y - rb); yy <= std::min(H - 1, y + rb); ++yy) {
                        for (int xx = std::max(0, x - rb); xx <= std::min(W - 1, x + rb); ++xx) {
                            const int q = yy * W + xx;
                            const double k = mw(q);
                            const double yr = luma(r, q), yt = luma(t, q);
                            Mb += k;
                            e1r += k * yr;
                            e2r += k * yr * yr;
                            e1t += k * yt;
                            e2t += k * yt * yt;
                        }
                    }
                    e1r /= Mb, e2r /= Mb, e1t /= Mb, e2t /= Mb;
                    const double sr = std::sqrt(std::max(e2r - e1r * e1r, static_cast<double>(gk::GENERATED_VIEW_VAR_EPS)));
                    const double st = std::sqrt(std::max(e2t - e1t * e1t, static_cast<double>(gk::GENERATED_VIEW_VAR_EPS)));
                    tex += mp * std::abs(sr - st);
                    sr_sum += mp * sr;
                    st_sum += mp * st;
                }
            }
            const double inv = S > 0.0 ? 1.0 / S : 0.0;
            RefTerms out{};
            out.low = low * inv / 3.0;
            out.tex = tex * inv;
            out.pix = pix * inv / 3.0;
            out.std_r = sr_sum * inv;
            out.std_t = st_sum * inv;
            out.loss = p.loss_weight * (p.lowpass_weight * out.low + p.texture_weight * out.tex + p.pixel_weight * out.pix);
            return out;
        }

        struct GpuResult {
            double loss;
            std::vector<float> grad;
            std::vector<float> stats;
        };

        GpuResult run_gpu(const GenCase& c, const Inputs& in, gk::GeneratedViewLossWorkspace& ws, const bool batched) {
            const size_t H = c.height, W = c.width;
            Tensor render = Tensor::from_vector(in.render, {3, H, W}, Device::CUDA);
            Tensor target = Tensor::from_vector(in.target_raw, {3, H, W}, Device::CUDA);
            if (c.uint8_target)
                target = target.to(DataType::UInt8);
            Tensor mask;
            if (c.mask != MaskKind::None) {
                mask = Tensor::from_vector(in.mask_raw, {H, W}, Device::CUDA);
                if (c.mask == MaskKind::UInt8)
                    mask = mask.to(DataType::UInt8);
            }
            if (batched) {
                render = render.unsqueeze(0);
                target = target.unsqueeze(0);
            }
            Tensor stats = Tensor::zeros({static_cast<size_t>(gk::GENERATED_VIEW_STAT_COUNT)}, Device::CUDA);
            const auto result = gk::generated_view_loss(render, target, mask, c.params, ws, &stats);
            GpuResult out;
            out.loss = result.loss.to_vector()[0];
            out.grad = result.grad.contiguous().to_vector();
            out.stats = stats.to_vector();
            return out;
        }

        int run_genloss_gradcheck() {
            std::mt19937 rng(1234);
            const gk::GeneratedViewLossParams defaults{};
            std::vector<GenCase> cases;
            cases.push_back({"unmasked, Float32 target, defaults", 30, 38, MaskKind::None, false, false, defaults, 600});
            cases.push_back({"Float32 mask (holes, fractional), UInt8 target, defaults", 31, 40, MaskKind::Float, true, false,
                             defaults, 600});
            cases.push_back({"UInt8 mask, Float32 target, sigma 1.3, radius 2, weights 0.7/2/0.3, loss weight 1.5", 28, 37,
                             MaskKind::UInt8, false, false,
                             {.lowpass_sigma = 1.3f, .texture_radius = 2, .lowpass_weight = 0.7f, .texture_weight = 2.0f,
                              .pixel_weight = 0.3f, .loss_weight = 1.5f},
                             600});
            cases.push_back({"unmasked, sigma 0 (no blur), radius 1, pixel weight 0.2", 24, 29, MaskKind::None, true, false,
                             {.lowpass_sigma = 0.0f, .texture_radius = 1, .lowpass_weight = 1.0f, .texture_weight = 1.0f,
                              .pixel_weight = 0.2f, .loss_weight = 1.0f},
                             600});
            cases.push_back({"Float32 mask, flat render patch (variance clamp), sigma 3, radius 4", 34, 36, MaskKind::Float,
                             false, true,
                             {.lowpass_sigma = 3.0f, .texture_radius = 4, .lowpass_weight = 1.0f, .texture_weight = 1.0f,
                              .pixel_weight = 0.0f, .loss_weight = 1.0f},
                             600});
            cases.push_back({"small, every coordinate, Float32 mask, defaults", 14, 17, MaskKind::Float, false, false, defaults,
                             0});

            gk::GeneratedViewLossWorkspace ws;
            int failures = 0;
            constexpr double H_STEP = 1e-6;
            std::println("generated-view loss gradcheck: GPU (float) against a double-precision direct reference");
            std::println("  gradient error = |analytic - central difference| over the checked coordinates (h = {})", H_STEP);
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                const auto& c = cases[ci];
                const Inputs in = make_inputs(c, rng);
                const int H = c.height, W = c.width, n = H * W, N = 3 * n;
                std::vector<double> r(in.render.begin(), in.render.end());
                const std::vector<double> t(in.target.begin(), in.target.end());
                const std::vector<double> m(in.mask.begin(), in.mask.end());

                const GpuResult gpu = run_gpu(c, in, ws, false);
                const RefTerms ref = reference(r, t, m, H, W, c.params);
                const double fwd_rel = std::abs(gpu.loss - ref.loss) / std::max(1e-30, std::abs(ref.loss));
                const auto stat_rel = [&](const int k, const double v) {
                    return std::abs(gpu.stats[k] - v) / std::max(1e-30, std::abs(v));
                };
                const double stats_rel = std::max({stat_rel(gk::GENERATED_VIEW_STAT_LOWPASS, ref.low),
                                                   stat_rel(gk::GENERATED_VIEW_STAT_TEXTURE, ref.tex),
                                                   stat_rel(gk::GENERATED_VIEW_STAT_PIXEL, ref.pix),
                                                   stat_rel(gk::GENERATED_VIEW_STAT_STD_RENDER, ref.std_r),
                                                   stat_rel(gk::GENERATED_VIEW_STAT_STD_TARGET, ref.std_t),
                                                   stat_rel(gk::GENERATED_VIEW_STAT_TOTAL, ref.loss)});

                // Coordinates to difference.
                std::vector<int> coords;
                if (c.fd_coordinates == 0 || c.fd_coordinates >= N) {
                    coords.resize(N);
                    for (int i = 0; i < N; ++i)
                        coords[i] = i;
                } else {
                    std::uniform_int_distribution<int> pick(0, N - 1);
                    for (int k = 0; k < c.fd_coordinates; ++k)
                        coords.push_back(pick(rng));
                }
                double num = 0.0, den = 0.0, max_abs_err = 0.0, max_fd = 0.0;
                int zero_both = 0;
                std::vector<double> rel_big;
                std::vector<std::pair<double, double>> pairs;
                for (const int i : coords) {
                    const double keep = r[i];
                    r[i] = keep + H_STEP;
                    const double lp = reference(r, t, m, H, W, c.params).loss;
                    r[i] = keep - H_STEP;
                    const double lm = reference(r, t, m, H, W, c.params).loss;
                    r[i] = keep;
                    const double fd = (lp - lm) / (2.0 * H_STEP);
                    const double an = gpu.grad[i];
                    num += (an - fd) * (an - fd);
                    den += fd * fd;
                    max_abs_err = std::max(max_abs_err, std::abs(an - fd));
                    max_fd = std::max(max_fd, std::abs(fd));
                    if (fd == 0.0 && an == 0.0)
                        ++zero_both;
                    pairs.emplace_back(an, fd);
                }
                for (const auto& [an, fd] : pairs)
                    if (std::abs(fd) > 1e-2 * max_fd)
                        rel_big.push_back(std::abs(an - fd) / std::abs(fd));
                std::ranges::sort(rel_big);
                const double global_rel = std::sqrt(num / std::max(den, 1e-300));
                const double median_rel = rel_big.empty() ? 0.0 : rel_big[rel_big.size() / 2];
                const double max_rel = rel_big.empty() ? 0.0 : rel_big.back();

                // Directional derivatives along random directions (the whole gradient at once).
                std::normal_distribution<double> gauss(0.0, 1.0);
                double dir_worst = 0.0;
                for (int k = 0; k < 3; ++k) {
                    std::vector<double> v(N);
                    double dot = 0.0;
                    for (int i = 0; i < N; ++i) {
                        v[i] = gauss(rng);
                        dot += v[i] * gpu.grad[i];
                    }
                    std::vector<double> rp(r), rm(r);
                    for (int i = 0; i < N; ++i) {
                        rp[i] += H_STEP * v[i];
                        rm[i] -= H_STEP * v[i];
                    }
                    const double fd = (reference(rp, t, m, H, W, c.params).loss -
                                       reference(rm, t, m, H, W, c.params).loss) /
                                      (2.0 * H_STEP);
                    dir_worst = std::max(dir_worst, std::abs(dot - fd) / std::max(1e-30, std::abs(fd)));
                }

                // The batched [1,3,H,W] path must agree with [3,H,W].
                const GpuResult batched = run_gpu(c, in, ws, true);
                double batched_diff = std::abs(batched.loss - gpu.loss);
                for (int i = 0; i < N; ++i)
                    batched_diff = std::max(batched_diff, static_cast<double>(std::abs(batched.grad[i] - gpu.grad[i])));

                const bool ok = fwd_rel < 1e-4 && stats_rel < 1e-3 && global_rel < 1e-3 && dir_worst < 1e-3 &&
                                batched_diff < 1e-6 && gpu.stats[gk::GENERATED_VIEW_STAT_VIEWS] == 1.0f;
                if (!ok)
                    ++failures;
                std::println("case {} [{}] {}x{}: loss {:.6g} (ref {:.6g}; low {:.4g} tex {:.4g} pix {:.4g}; std render {:.4g} target {:.4g})",
                             ci + 1, c.name, W, H, gpu.loss, ref.loss, ref.low, ref.tex, ref.pix, ref.std_r, ref.std_t);
                std::println("    forward rel err {:.2e}, logged terms rel err {:.2e}", fwd_rel, stats_rel);
                std::println("    gradient over {} coords ({} zero in both): global rel err {:.2e}, median/max rel err {:.2e} / {:.2e} "
                             "(|fd| > 1% of max), max abs err {:.2e} (max |fd| {:.2e})",
                             coords.size(), zero_both, global_rel, median_rel, max_rel, max_abs_err, max_fd);
                std::println("    directional derivative (3 random directions): worst rel err {:.2e}; [1,3,H,W] vs [3,H,W] max diff {:.1e}  {}",
                             dir_worst, batched_diff, ok ? "OK" : "FAIL");
            }
            if (failures == 0)
                std::println("GENLOSS GRADCHECK OK");
            else
                std::println("GENLOSS GRADCHECK: {} of {} cases failed", failures, cases.size());
            return failures == 0 ? 0 : 1;
        }
    } // namespace

    int run_lfe_genloss_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "gradcheck")
            return run_genloss_gradcheck();
        std::println(stderr, "usage: LichtFeld-Studio lfe-genloss gradcheck");
        return 2;
    }

} // namespace lfs::app
