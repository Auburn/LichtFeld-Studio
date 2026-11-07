/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Developer checks of the per-splat kernel shape, --kernel-shape (LFQPLCYK; FastGS kernel_shape.cuh):
//   lfe-shape refcheck OUTDIR  a small pinhole scene of splats with beta 1, 1.2, 2, 3.5, 6 and 8 placed across tile
//                              boundaries: the FastGS forward's image and alpha + the scene (OUTDIR/scene.txt, image.f32,
//                              alpha.f32) for a float64 reference (D:/splat_work/lfrqplcyk/tests/shape_ref.py); and the
//                              same scene with every beta = 2 (raw 0) against the shape pointer unbound: forward bits and
//                              backward gradients ("BETA2 IDENTITY OK" when bit-identical).
//   lfe-shape gradcheck        central finite differences of a weighted image sum on raw r, means, log-scales, rotations
//                              and raw opacity, with beta != 2 splats overlapping: "SHAPE GRADCHECK OK" or FAILED.
//   lfe-shape cutcheck         --max-std-dev-distance (LFQPLCYKD): coverage of one splat at beta 1, 1.5, 2, 4, 8 against
//                              min(alpha contour, C) (and the exponent cut of plain --max-std-dev for contrast), no gradient
//                              from pixels beyond C, beta = 2 distance mode == plain --max-std-dev: "CUTCHECK OK" or FAILED.
// Env (refcheck / gradcheck): LFE_SHAPE_CUT=C runs them under the splat cut C, LFE_SHAPE_CUT_MODE=dist (distance, the
// default) or exp (plain --max-std-dev); refcheck then writes OUTDIR/cut.txt (C, min_alpha, mode) for shape_cut_ref.py, and
// gradcheck weights 0.75 min(contour, C) and zeroes the weight within 0.1 (in d) of every splat's edge.
#include "core/kernel_shape_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <random>
#include <rasterization_api.h>
#include <cstdlib>
#include <string>
#include <vector>

namespace lfs::app {

    namespace {
        namespace fr = fast_lfs::rasterization;
        namespace ks = lfs::core::kshape;

        struct DevBuf {
            float* p = nullptr;
            size_t n = 0;
            explicit DevBuf(size_t count) : n(count) {
                cudaMalloc(&p, sizeof(float) * std::max<size_t>(count, 1));
                cudaMemset(p, 0, sizeof(float) * std::max<size_t>(count, 1));
            }
            ~DevBuf() { cudaFree(p); }
            DevBuf(const DevBuf&) = delete;
            DevBuf& operator=(const DevBuf&) = delete;
            void upload(const std::vector<float>& v) {
                if (!v.empty())
                    cudaMemcpy(p, v.data(), sizeof(float) * v.size(), cudaMemcpyHostToDevice);
            }
            std::vector<float> download() const {
                std::vector<float> v(n);
                cudaMemcpy(v.data(), p, sizeof(float) * n, cudaMemcpyDeviceToHost);
                return v;
            }
        };

        struct Scene {
            int n = 0, w = 96, h = 64;
            float fx = 80.0f, fy = 80.0f, cx = 48.0f, cy = 32.0f;
            std::vector<float> means, scales, rots, opac, sh0, shape; // shape: raw r
            std::vector<float> w_img;
        };

        struct Run {
            double loss = 0;
            std::vector<float> img, alpha, g_means, g_scales, g_rots, g_opac, g_sh0, g_beta;
            int n_visible = 0;
        };

        // shape_mode: 0 = the shape pointer unbound (the Gaussian kernels), 1 = bound (the --kernel-shape kernels)
        Run render(const Scene& s, const bool backward, const int shape_mode) {
            const int N = s.n;
            const size_t HW = static_cast<size_t>(s.w) * s.h;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_w2c(16), d_cam(3), d_shape(N), d_gbeta(N);
            DevBuf d_img(3 * HW), d_alpha(HW), d_depth(HW), d_wimg(3 * HW), d_wdepth(HW);
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_shape.upload(s.shape);
            d_w2c.upload({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
            d_cam.upload({0, 0, 0});
            d_wimg.upload(s.w_img);
            fr::FastGSKernelShape shape{};
            if (shape_mode == 1) {
                shape.raw = d_shape.p;
                shape.grad = d_gbeta.p;
            }
            Run run;
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, nullptr, d_w2c.p, d_cam.p, d_img.p, d_alpha.p,
                                       d_depth.p, nullptr, nullptr, nullptr, N, 1, 1, s.w, s.h, s.fx, s.fy, s.cx, s.cy, 0.01f, 1e10f,
                                       false, nullptr, nullptr, 0, 0, nullptr, 1.0f, fr::FastGSLens{}, fr::FastGSDistortion{}, 0,
                                       fr::FastGSRollingShutter{}, shape);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            run.n_visible = ctx.n_visible;
            run.img = d_img.download();
            run.alpha = d_alpha.download();
            double loss = 0;
            for (size_t i = 0; i < run.img.size(); ++i)
                loss += static_cast<double>(run.img[i]) * s.w_img[i];
            run.loss = loss;
            if (!backward) {
                fr::release_forward_context(ctx);
                cudaDeviceSynchronize();
                return run;
            }
            DevBuf g_means(3 * N), g_scales(3 * N), g_rots(4 * N), g_opac(N), g_sh0(3 * N);
            fr::FusedAdamSettings fa;
            fa.enabled = true;
            auto capture = [N](fr::FusedAdamParam& prm, DevBuf& grad, DevBuf& value, int attrs) {
                prm.gradient_out = grad.p;
                prm.param = value.p;
                prm.n_primitives = N;
                prm.n_elements = N * attrs;
                prm.n_attributes = attrs;
            };
            capture(fa.means, g_means, d_means, 3);
            capture(fa.scaling, g_scales, d_scales, 3);
            capture(fa.rotation, g_rots, d_rots, 4);
            capture(fa.opacity, g_opac, d_opac, 1);
            capture(fa.sh0, g_sh0, d_sh0, 3);
            const auto res = fr::backward_raw(nullptr, nullptr, d_wimg.p, fr::BackgroundAlphaGradient{}, nullptr, nullptr, d_img.p,
                                              d_alpha.p, d_means.p, d_scales.p, d_rots.p, d_opac.p, nullptr, d_w2c.p, d_cam.p, ctx,
                                              nullptr, N, ctx.n_visible, 1, 1, s.w, s.h, s.fx, s.fy, s.cx, s.cy, false,
                                              DensificationType::None, &fa);
            cudaDeviceSynchronize();
            if (!res.success) {
                std::println(stderr, "backward failed: {}", res.error_message ? res.error_message : "?");
                std::exit(1);
            }
            run.g_means = g_means.download();
            run.g_scales = g_scales.download();
            run.g_rots = g_rots.download();
            run.g_opac = g_opac.download();
            run.g_sh0 = g_sh0.download();
            run.g_beta = d_gbeta.download();
            return run;
        }

        struct SplatSpec {
            float u, v, z;     // pixel position and depth
            float s1, s2, ang; // 2D sigma (px) along the two in-plane axes, rotation about the view axis (deg)
            float o, beta;
            float r, g, b;
        };

        // Splats with beta 1.2 and 6 straddling tile boundaries (tiles are 16 px), plus 1 (the longest tails), 2, 3.5, 8
        Scene make_scene(const std::vector<SplatSpec>& specs, unsigned seed, const int w = 96, const int h = 64) {
            Scene s;
            s.w = w;
            s.h = h;
            s.cx = 0.5f * static_cast<float>(w);
            s.cy = 0.5f * static_cast<float>(h);
            s.n = static_cast<int>(specs.size());
            for (const auto& p : specs) {
                const float x = (p.u - s.cx) * p.z / s.fx, y = (p.v - s.cy) * p.z / s.fy;
                s.means.insert(s.means.end(), {x, y, p.z});
                s.scales.insert(s.scales.end(), {std::log(p.s1 * p.z / s.fx), std::log(p.s2 * p.z / s.fx), std::log(0.02f * p.z / s.fx)});
                const float a = p.ang * 3.14159265358979f / 180.0f;
                s.rots.insert(s.rots.end(), {std::cos(0.5f * a), 0.0f, 0.0f, std::sin(0.5f * a)});
                s.opac.push_back(std::log(p.o / (1.0f - p.o)));
                constexpr float C0 = 0.28209479177387814f;
                s.sh0.insert(s.sh0.end(), {(p.r - 0.5f) / C0, (p.g - 0.5f) / C0, (p.b - 0.5f) / C0});
                s.shape.push_back(ks::raw_from_beta(p.beta));
            }
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> U(-1.0f, 1.0f);
            s.w_img.resize(3 * static_cast<size_t>(s.w) * s.h);
            for (auto& v : s.w_img)
                v = U(rng);
            return s;
        }

        const std::vector<SplatSpec> kRefSplats = {
            {16.0f, 16.0f, 2.0f, 3.0f, 2.0f, 20.0f, 0.80f, 1.2f, 0.9f, 0.3f, 0.2f}, // beta 1.2 on a tile corner
            {48.0f, 32.0f, 2.2f, 5.0f, 2.5f, 30.0f, 0.90f, 6.0f, 0.2f, 0.8f, 0.3f}, // beta 6 on a tile corner
            {72.0f, 40.0f, 2.4f, 4.0f, 3.0f, -15.0f, 0.70f, 2.0f, 0.3f, 0.3f, 0.9f},
            {60.0f, 20.0f, 2.6f, 3.0f, 3.0f, 0.0f, 0.60f, 3.5f, 0.8f, 0.8f, 0.2f},
            {20.0f, 48.0f, 2.8f, 2.0f, 1.5f, 45.0f, 0.95f, 1.0f, 0.6f, 0.2f, 0.7f}, // beta 1: ~11 sigma of tails
            {80.0f, 52.0f, 3.0f, 4.0f, 2.0f, 60.0f, 0.85f, 8.0f, 0.9f, 0.6f, 0.1f},
            {32.0f, 33.0f, 3.2f, 6.0f, 1.5f, 80.0f, 0.50f, 1.2f, 0.5f, 0.9f, 0.9f}, // beta 1.2 across the x = 32 edge
            {63.5f, 47.5f, 3.4f, 3.5f, 3.5f, 0.0f, 0.75f, 6.0f, 0.7f, 0.4f, 0.6f}, // beta 6 centred near a corner
        };

        bool write_f32(const std::filesystem::path& p, const std::vector<float>& v) {
            std::ofstream f(p, std::ios::binary);
            f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
            return static_cast<bool>(f);
        }

        // LFQPLCYKD: the cut of this run (LFE_SHAPE_CUT / LFE_SHAPE_CUT_MODE); C = 0 off
        float g_cut = 0.0f;
        bool g_cut_dist = true;
        float min_alpha_now() { return g_cut > 0.0f ? fr::kGameMinAlpha : 1.0f / 255.0f; }
        // the d at which the cut splat ends: min(contour, C) under the distance cut, the exponent cut's
        // min((2 min(P, C^2/2))^(1/beta), C) under plain --max-std-dev, the contour without a cut
        double edge_d(const double o, const double beta) {
            const double P = std::log(o / min_alpha_now());
            if (!(P > 0))
                return 0.0;
            const double contour = std::pow(2.0 * P, 1.0 / beta);
            if (!(g_cut > 0.0f))
                return contour;
            const double C = g_cut;
            if (g_cut_dist)
                return std::min(contour, C);
            return std::min(std::pow(2.0 * std::min(P, 0.5 * C * C), 1.0 / beta), C);
        }

        int run_refcheck(const std::filesystem::path& out) {
            std::filesystem::create_directories(out);
            const Scene s = make_scene(kRefSplats, 7u);
            const Run a = render(s, false, 1);
            if (g_cut > 0.0f) {
                std::ofstream f(out / "cut.txt");
                f << std::format("{:.9g} {:.9g} {}\n", g_cut, fr::kGameMinAlpha, g_cut_dist ? "dist" : "exp");
            } else
                std::filesystem::remove(out / "cut.txt");
            {
                std::ofstream f(out / "scene.txt");
                f << std::format("# lfe-shape refcheck scene: w h fx fy cx cy, then per splat: x y z ls0 ls1 ls2 qw qx qy qz "
                                 "raw_opacity sh0_r sh0_g sh0_b raw_shape beta_device\n");
                f << std::format("{} {} {:.9g} {:.9g} {:.9g} {:.9g}\n", s.w, s.h, s.fx, s.fy, s.cx, s.cy);
                for (int i = 0; i < s.n; ++i) {
                    f << std::format("{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} "
                                     "{:.9g} {:.9g}\n",
                                     s.means[3 * i], s.means[3 * i + 1], s.means[3 * i + 2], s.scales[3 * i], s.scales[3 * i + 1],
                                     s.scales[3 * i + 2], s.rots[4 * i], s.rots[4 * i + 1], s.rots[4 * i + 2], s.rots[4 * i + 3],
                                     s.opac[i], s.sh0[3 * i], s.sh0[3 * i + 1], s.sh0[3 * i + 2], s.shape[i],
                                     ks::beta_from_raw(s.shape[i]));
                }
            }
            if (!write_f32(out / "image.f32", a.img) || !write_f32(out / "alpha.f32", a.alpha)) {
                std::println(stderr, "cannot write {}", out.string());
                return 1;
            }
            std::println("refcheck scene: {} splats, {} visible, {}x{}; written to {}", s.n, a.n_visible, s.w, s.h, out.string());

            // beta = 2 everywhere (raw 0) through the --kernel-shape kernels vs the shape pointer unbound
            Scene s2 = s;
            std::fill(s2.shape.begin(), s2.shape.end(), 0.0f);
            const Run off = render(s2, true, 0);
            const Run off2 = render(s2, true, 0); // the reference spread: the unbound kernels against themselves
            const Run on = render(s2, true, 1);
            auto same_bits = [](const std::vector<float>& x, const std::vector<float>& y) {
                return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
            };
            auto max_rel = [](const std::vector<float>& x, const std::vector<float>& y) {
                double m = 0, scale = 1e-30;
                for (size_t i = 0; i < x.size(); ++i) {
                    m = std::max(m, std::abs(static_cast<double>(x[i]) - y[i]));
                    scale = std::max(scale, std::abs(static_cast<double>(x[i])));
                }
                return m / scale;
            };
            const bool fwd = same_bits(off.img, on.img) && same_bits(off.alpha, on.alpha);
            const bool bwd = same_bits(off.g_means, on.g_means) && same_bits(off.g_scales, on.g_scales) &&
                             same_bits(off.g_rots, on.g_rots) && same_bits(off.g_opac, on.g_opac) && same_bits(off.g_sh0, on.g_sh0);
            std::println("beta = 2 through the shape kernels vs unbound: forward image+alpha {}; backward means/scales/rots/opacity/sh0 "
                         "{} (max rel diff means {:.3g}, scales {:.3g}, rots {:.3g}, opacity {:.3g}, sh0 {:.3g})",
                         fwd ? "BIT-IDENTICAL" : "DIFFER", bwd ? "BIT-IDENTICAL" : "differ", max_rel(off.g_means, on.g_means),
                         max_rel(off.g_scales, on.g_scales), max_rel(off.g_rots, on.g_rots), max_rel(off.g_opac, on.g_opac),
                         max_rel(off.g_sh0, on.g_sh0));
            const bool self = same_bits(off.g_means, off2.g_means) && same_bits(off.g_scales, off2.g_scales) &&
                              same_bits(off.g_rots, off2.g_rots) && same_bits(off.g_opac, off2.g_opac) && same_bits(off.g_sh0, off2.g_sh0);
            std::println("  reference: the unbound backward twice: {} (max rel diff means {:.3g}, scales {:.3g}, rots {:.3g}, opacity {:.3g})",
                         self ? "BIT-IDENTICAL" : "differ (float atomics)", max_rel(off.g_means, off2.g_means),
                         max_rel(off.g_scales, off2.g_scales), max_rel(off.g_rots, off2.g_rots), max_rel(off.g_opac, off2.g_opac));
            double gb = 0;
            for (const float g : on.g_beta)
                gb = std::max(gb, std::abs(static_cast<double>(g)));
            std::println("  dL/dbeta at beta = 2 (bound, not applied): max |g| {:.4g}", gb);
            if (!fwd) {
                std::println("BETA2 IDENTITY FAILED");
                return 1;
            }
            std::println("{}", bwd ? "BETA2 IDENTITY OK" : "BETA2 IDENTITY OK (forward); backward differs (float atomics order)");
            return 0;
        }

        // The host's 2D covariance of a splat (pinhole, identity camera, as FastGS: J R S^2 R^T J^T + 0.3 I) and its
        // Mahalanobis d^2 at a pixel centre.
        struct Conic2 {
            double mx, my, a, b, c; // mean, inverse covariance
        };
        Conic2 conic_of(const Scene& s, const int i) {
            const double X = s.means[3 * i], Y = s.means[3 * i + 1], Z = s.means[3 * i + 2];
            const double qw = s.rots[4 * i], qx = s.rots[4 * i + 1], qy = s.rots[4 * i + 2], qz = s.rots[4 * i + 3];
            const double qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            const double w = qw / qn, x = qx / qn, y = qy / qn, z = qz / qn;
            const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                                 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                                 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
            double cov[3][3] = {};
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    for (int k = 0; k < 3; ++k)
                        cov[a][b] += R[a * 3 + k] * std::exp(2.0 * s.scales[3 * i + k]) * R[b * 3 + k];
            const double J[2][3] = {{s.fx / Z, 0, -s.fx * X / (Z * Z)}, {0, s.fy / Z, -s.fy * Y / (Z * Z)}};
            double C[2][2] = {};
            for (int r = 0; r < 2; ++r)
                for (int q = 0; q < 2; ++q)
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            C[r][q] += J[r][a] * cov[a][b] * J[q][b];
            C[0][0] += 0.3;
            C[1][1] += 0.3;
            const double det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            return {s.fx * X / Z + s.cx, s.fy * Y / Z + s.cy, C[1][1] / det, -C[0][1] / det, C[0][0] / det};
        }

        int run_gradcheck() {
            // Groups far apart; each weighted only inside 0.75 x the alpha = 1/255 contour of its weighted splat
            // (d_max = (2 ln(255 o))^(1/beta)), so no splat's contour (where alpha jumps to 0, which no analytic gradient
            // follows) lies where the loss looks. Group 1 = a beta 5 splat in front of a beta 1.3 one (the back splat's
            // gradients through the front's transmittance; the back's contour is far outside the weighted region).
            const std::vector<SplatSpec> specs = {
                {42.0f, 38.0f, 2.0f, 7.0f, 4.0f, 25.0f, 0.70f, 5.0f, 0.9f, 0.3f, 0.2f},  // 0 front (weighted)
                {38.0f, 42.0f, 3.0f, 5.0f, 4.0f, -30.0f, 0.60f, 1.3f, 0.2f, 0.8f, 0.3f}, // 1 back
                {122.0f, 30.0f, 2.5f, 3.0f, 2.5f, 10.0f, 0.50f, 1.05f, 0.6f, 0.2f, 0.7f}, // 2 long tails (weighted)
                {110.0f, 95.0f, 2.8f, 5.0f, 3.0f, 70.0f, 0.70f, 2.7f, 0.8f, 0.8f, 0.2f},  // 3 (weighted)
                {35.0f, 100.0f, 3.2f, 6.0f, 4.0f, -50.0f, 0.80f, 7.9f, 0.9f, 0.6f, 0.1f}, // 4 nearly flat-topped (weighted)
                {70.0f, 70.0f, 2.2f, 4.0f, 4.0f, 0.0f, 0.65f, 2.0f, 0.3f, 0.3f, 0.9f},    // 5 the Gaussian through the shape kernels (weighted)
            };
            Scene s0 = make_scene(specs, 11u, 160, 128);
            std::fill(s0.w_img.begin(), s0.w_img.end(), 0.0f);
            const int weighted[] = {0, 2, 3, 4, 5};
            for (const int i : weighted) {
                const Conic2 q = conic_of(s0, i);
                const double beta = ks::beta_from_raw(s0.shape[i]);
                const double o = 1.0 / (1.0 + std::exp(-static_cast<double>(s0.opac[i])));
                const double dmax = g_cut > 0.0f ? edge_d(o, beta) : std::pow(2.0 * std::log(255.0 * o), 1.0 / beta);
                const double r2 = (0.75 * dmax) * (0.75 * dmax);
                for (int y = 0; y < s0.h; ++y)
                    for (int x = 0; x < s0.w; ++x) {
                        const double dx = x + 0.5 - q.mx, dy = y + 0.5 - q.my;
                        const double d2 = q.a * dx * dx + 2 * q.b * dx * dy + q.c * dy * dy;
                        if (d2 >= r2)
                            continue;
                        const double bump = (1 - d2 / r2) * (1 - d2 / r2);
                        for (int c = 0; c < 3; ++c)
                            s0.w_img[(static_cast<size_t>(c) * s0.h + y) * s0.w + x] += static_cast<float>(
                                1e-3 * bump * (1.0 + 0.6 * dx / 5.0 * (c == 1 ? -1.0 : 1.0) - 0.4 * dy / 5.0 + 0.3 * c));
                    }
            }
            if (g_cut > 0.0f) { // no weight within 0.1 (in d) of any splat's edge, where alpha jumps under the cut
                int zeroed = 0;
                for (int j = 0; j < s0.n; ++j) {
                    const Conic2 q = conic_of(s0, j);
                    const double o = 1.0 / (1.0 + std::exp(-static_cast<double>(s0.opac[j])));
                    const double R = edge_d(o, ks::beta_from_raw(s0.shape[j]));
                    for (int y = 0; y < s0.h; ++y)
                        for (int x = 0; x < s0.w; ++x) {
                            const double dx = x + 0.5 - q.mx, dy = y + 0.5 - q.my;
                            const double d = std::sqrt(std::max(q.a * dx * dx + 2 * q.b * dx * dy + q.c * dy * dy, 0.0));
                            if (std::abs(d - R) < 0.1)
                                for (int c = 0; c < 3; ++c) {
                                    float& wv = s0.w_img[(static_cast<size_t>(c) * s0.h + y) * s0.w + x];
                                    zeroed += wv != 0.0f;
                                    wv = 0.0f;
                                }
                        }
                }
                std::println("gradcheck under the cut C = {} ({}): {} weights zeroed near the splats' edges", g_cut,
                             g_cut_dist ? "distance" : "exponent", zeroed);
            }
            const Run base = render(s0, true, 1);
            std::println("gradcheck scene: {} splats ({} visible), {}x{}; each block = one splat's parameter vector, relative error "
                         "|fd - analytic| / |fd| (central differences of a weighted image sum), 1% threshold", s0.n, base.n_visible, s0.w, s0.h);
            int bad = 0, blocks = 0;
            std::vector<double> rels;
            auto block = [&](const char* name, std::vector<float> Scene::*member, const std::vector<float>& analytic, const int per,
                             const int i, const double h, const double chain) {
                double num = 0, den = 0, dot = 0, na = 0;
                for (int k = 0; k < per; ++k) {
                    Scene sp = s0, sm = s0;
                    (sp.*member)[per * i + k] += static_cast<float>(h);
                    (sm.*member)[per * i + k] -= static_cast<float>(h);
                    const double h2 = static_cast<double>((sp.*member)[per * i + k]) - (sm.*member)[per * i + k];
                    const double fd = (render(sp, false, 1).loss - render(sm, false, 1).loss) / h2;
                    const double an = static_cast<double>(analytic[per * i + k]) * chain;
                    num += (fd - an) * (fd - an);
                    den += fd * fd;
                    dot += fd * an;
                    na += an * an;
                }
                const double rel = std::sqrt(num / std::max(den, 1e-300));
                const double cosv = dot / std::max(std::sqrt(den * na), 1e-300);
                const bool sig = std::sqrt(den) > 1e-7;
                ++blocks;
                if (sig) {
                    rels.push_back(rel);
                    if (rel > 0.01)
                        ++bad;
                }
                std::println("  splat {} (beta {:.3f}) {:<12} |fd| {:9.3e}  cosine {:9.6f}  rel err {:8.2e}{}", i,
                             ks::beta_from_raw(s0.shape[i]), name, std::sqrt(den), cosv, rel,
                             !sig ? "  (no signal)" : rel > 0.01 ? "  OVER 1%" : "");
            };
            for (int i = 0; i < s0.n; ++i) {
                const double dz = s0.means[3 * i + 2];
                // the step in r that moves beta by ~1e-3 (d beta / d r is 0.1 near 7.9 and 6/7 at 2)
                const double db = ks::dbeta_draw(ks::beta_from_raw(s0.shape[i]));
                block("beta (raw r)", &Scene::shape, base.g_beta, 1, i, 1e-3 / std::max(db, 1e-2), db);
                block("means", &Scene::means, base.g_means, 3, i, 5e-4 * dz, 1.0);
                block("log scales", &Scene::scales, base.g_scales, 3, i, 1e-3, 1.0);
                block("rotations", &Scene::rots, base.g_rots, 4, i, 1e-3, 1.0);
                block("raw opacity", &Scene::opac, base.g_opac, 1, i, 1e-3, 1.0);
            }
            std::sort(rels.begin(), rels.end());
            std::println("  {} blocks ({} with signal): median rel err {:.2e}, max {:.2e}; {} over 1%", blocks, rels.size(),
                         rels.empty() ? 0.0 : rels[rels.size() / 2], rels.empty() ? 0.0 : rels.back(), bad);
            std::println("{}", bad == 0 && !rels.empty() ? "SHAPE GRADCHECK OK" : "SHAPE GRADCHECK FAILED");
            return bad == 0 && !rels.empty() ? 0 : 1;
        }
        // ------------------------------------------------------------------ cutcheck (LFQPLCYKD)
        void set_cut(const float C, const bool dist) {
            g_cut = C;
            g_cut_dist = dist;
            fr::set_splat_cutoff(C, dist);
        }
        void clear_cut() {
            g_cut = 0.0f;
            fr::set_splat_cutoff(0.0f);
        }
        double d_at(const Conic2& q, const int x, const int y) {
            const double dx = x + 0.5 - q.mx, dy = y + 0.5 - q.my;
            return std::sqrt(std::max(q.a * dx * dx + 2 * q.b * dx * dy + q.c * dy * dy, 0.0));
        }

        int run_cutcheck() {
            int bad = 0;
            const float cuts[] = {2.2360680f, 3.0f};
            const float betas[] = {1.0f, 1.5f, 2.0f, 4.0f, 8.0f};
            const float opacities[] = {0.9f, 0.03f};
            // (1) coverage: one circular splat (sigma 7 px), centre off the pixel grid
            std::println("(1) coverage: the lit pixels' largest d and the unlit pixels' smallest d bracket the expected edge "
                         "(d = Mahalanobis distance in the dilated conic, float64 on the host); gated: the distance mode");
            for (const float C : cuts)
                for (const float o : opacities)
                    for (const float beta : betas)
                        for (int mode = 0; mode < 3; ++mode) { // 0 distance cut, 1 plain --max-std-dev (exponent), 2 no cut
                            if (mode == 2 && C != cuts[0])
                                continue;
                            if (mode == 2)
                                clear_cut();
                            else
                                set_cut(C, mode == 0);
                            const std::vector<SplatSpec> one = {{80.3f, 79.6f, 2.0f, 7.0f, 7.0f, 0.0f, o, beta, 0.7f, 0.5f, 0.3f}};
                            const Scene s = make_scene(one, 3u, 160, 160);
                            const Run r = render(s, false, 1);
                            const Conic2 q = conic_of(s, 0);
                            const double ob = 1.0 / (1.0 + std::exp(-static_cast<double>(s.opac[0])));
                            const double be = ks::beta_from_raw(s.shape[0]);
                            const double R = edge_d(ob, be);
                            const double contour = std::pow(2.0 * std::log(ob / min_alpha_now()), 1.0 / be);
                            double lit_max = 0, unlit_min = 1e30;
                            int lit = 0;
                            for (int y = 0; y < s.h; ++y)
                                for (int x = 0; x < s.w; ++x) {
                                    const double d = d_at(q, x, y);
                                    if (r.alpha[static_cast<size_t>(y) * s.w + x] > 0.0f) {
                                        lit_max = std::max(lit_max, d);
                                        ++lit;
                                    } else
                                        unlit_min = std::min(unlit_min, d);
                                }
                            const double tol = 2e-4 * std::max(R, 1.0);
                            const bool ok = lit_max <= R + tol && unlit_min >= R - tol;
                            if (mode == 0)
                                bad += !ok;
                            std::println("  {:<9} C {:.4f} o {:.2f} beta {:.4f}: contour {:7.4f}  expected edge {:7.4f}  lit {:5d} px, max d "
                                         "{:7.4f}, unlit min d {:7.4f}  {}",
                                         mode == 0 ? "distance" : mode == 1 ? "exponent" : "no cut", mode == 2 ? 0.0f : C, o, be, contour,
                                         R, lit, lit_max, unlit_min, ok ? "ok" : (mode == 0 ? "FAIL" : "(mismatch)"));
                        }
            // (2) no gradient from beyond C: weight only pixels with C + 0.3 < d < contour - 0.3 of a beta 1 / 1.5 splat
            std::println("(2) leak: loss weighted only where C + 0.3 < d < contour - 0.3; every analytic gradient must be exactly 0 "
                         "under the distance cut (and is not without the cut)");
            for (const float C : cuts)
                for (const float beta : {1.0f, 1.5f}) {
                    const std::vector<SplatSpec> one = {{80.3f, 79.6f, 2.0f, 6.0f, 4.0f, 30.0f, 0.9f, beta, 0.7f, 0.5f, 0.3f}};
                    Scene s = make_scene(one, 5u, 160, 160);
                    set_cut(C, true);
                    const Conic2 q = conic_of(s, 0);
                    const double ob = 1.0 / (1.0 + std::exp(-static_cast<double>(s.opac[0])));
                    const double contour = std::pow(2.0 * std::log(ob / min_alpha_now()), 1.0 / ks::beta_from_raw(s.shape[0]));
                    int nw = 0;
                    for (int y = 0; y < s.h; ++y)
                        for (int x = 0; x < s.w; ++x) {
                            const double d = d_at(q, x, y);
                            const bool in = d > C + 0.3 && d < contour - 0.3;
                            nw += in;
                            for (int c = 0; c < 3; ++c)
                                if (!in)
                                    s.w_img[(static_cast<size_t>(c) * s.h + y) * s.w + x] = 0.0f;
                        }
                    auto maxabs = [](const Run& r) {
                        double m = 0;
                        for (const auto* v : {&r.g_means, &r.g_scales, &r.g_rots, &r.g_opac, &r.g_sh0, &r.g_beta})
                            for (const float g : *v)
                                m = std::max(m, std::abs(static_cast<double>(g)));
                        return m;
                    };
                    const Run on = render(s, true, 1);
                    clear_cut();
                    const Run off = render(s, true, 1);
                    const double mon = maxabs(on), moff = maxabs(off);
                    const bool ok = mon == 0.0 && moff > 0.0 && nw > 0;
                    bad += !ok;
                    std::println("  C {:.4f} beta {:.1f}: {} weighted px (contour {:.3f}); max |grad| distance cut {:.3g}, no cut {:.3g}  {}", C,
                                 beta, nw, contour, mon, moff, ok ? "ok" : "FAIL");
                }
            // (3) beta = 2: the distance cut == plain --max-std-dev, unbound and bound (the shape kernels at raw 0)
            std::println("(3) beta = 2: distance mode vs plain --max-std-dev on the refcheck scene with every beta = 2");
            Scene s2 = make_scene(kRefSplats, 7u);
            std::fill(s2.shape.begin(), s2.shape.end(), 0.0f);
            auto same_bits = [](const std::vector<float>& x, const std::vector<float>& y) {
                return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
            };
            auto bw = [&](const Run& a, const Run& b) {
                return same_bits(a.g_means, b.g_means) && same_bits(a.g_scales, b.g_scales) && same_bits(a.g_rots, b.g_rots) &&
                       same_bits(a.g_opac, b.g_opac) && same_bits(a.g_sh0, b.g_sh0);
            };
            for (const float C : cuts) {
                set_cut(C, false);
                const Run e0 = render(s2, true, 0), e0b = render(s2, true, 0), e1 = render(s2, true, 1);
                set_cut(C, true);
                const Run d0 = render(s2, true, 0), d1 = render(s2, true, 1);
                const bool fwd = same_bits(e0.img, d0.img) && same_bits(e0.alpha, d0.alpha) && same_bits(e1.img, d1.img) &&
                                 same_bits(e1.alpha, d1.alpha) && same_bits(e0.img, d1.img);
                bad += !fwd;
                std::println("  C {:.4f}: forward image+alpha {} (distance vs exponent, unbound and bound); backward distance vs exponent: "
                             "unbound {}, bound {} (exponent vs itself: {})",
                             C, fwd ? "BIT-IDENTICAL" : "DIFFER", bw(e0, d0) ? "bit-identical" : "differs",
                             bw(e1, d1) ? "bit-identical" : "differs", bw(e0, e0b) ? "bit-identical" : "differs (float atomics)");
            }
            clear_cut();
            std::println("{}", bad == 0 ? std::string("CUTCHECK OK") : std::format("CUTCHECK FAILED ({} checks)", bad));
            return bad == 0 ? 0 : 1;
        }
    } // namespace

    int run_lfe_shape_tool(int argc, const char* const* argv) {
        if (const char* c = std::getenv("LFE_SHAPE_CUT"); c != nullptr && std::atof(c) > 0.0) {
            const char* m = std::getenv("LFE_SHAPE_CUT_MODE");
            const bool dist = m == nullptr || std::string(m) != "exp";
            set_cut(static_cast<float>(std::atof(c)), dist);
            std::println("lfe-shape under the splat cut C = {} ({} mode, min alpha {:.6f})", g_cut, dist ? "distance" : "exponent",
                         fr::kGameMinAlpha);
        }
        if (argc >= 3 && std::string(argv[2]) == "cutcheck")
            return run_cutcheck();
        if (argc >= 4 && std::string(argv[2]) == "refcheck")
            return run_refcheck(argv[3]);
        if (argc >= 3 && std::string(argv[2]) == "gradcheck")
            return run_gradcheck();
        std::println(stderr, "usage: LichtFeld-Studio lfe-shape refcheck <outdir> | gradcheck | cutcheck");
        return 2;
    }

} // namespace lfs::app
