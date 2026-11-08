/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Developer check of the footprint loss, --loss-footprint (LFQPLCYKMO; lfs/training/game_footprint.cuh, fused into the FastGS
// preprocess backward):
//   lfe-footprint gradcheck   a pinhole scene of splats with beta 1.3 / 2 / 3 / 6 / 7.9 and Gaussians (shape unbound), opacities
//                             0.95 / 0.3 / 0.03 / 0.008, small ones and ones at the game's 512 px-equivalent cap (one or both axes),
//                             no image loss (zero image weights): the only gradient is the footprint's.
//     1. the device loss and mean footprint per pixel against a float64 host reference of the definition (fpv-render's quad of
//        the dilated 2D covariance: 4 min(M, R sqrt(l1)) min(M, R sqrt(l2)), M = 0.41539 fx, R = sqrt 5 for beta 2 / unbound,
//        min(sqrt 5, (2 ln(510 o))^(1/beta)) otherwise);
//     2. the device gradient w.r.t. the log-scales (and, with the opacity option, the raw opacity) against central finite
//        differences of that host reference, per splat block; and against central differences of the DEVICE loss itself;
//     3. the other parameters get nothing from it (means, rotations, beta: exactly 0; opacity 0 without the option);
//     4. footprint off vs only the mean-footprint scalar bound (weight 0): every gradient bit-identical (measuring is free).
//   "FOOTPRINT GRADCHECK OK" or FAILED (1% relative error per block).
#include "core/kernel_shape_map.hpp"
#include "lfs/training/game_footprint.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <print>
#include <rasterization_api.h>
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
            int n = 0, w = 160, h = 128;
            float fx = 80.0f, fy = 80.0f, cx = 80.0f, cy = 64.0f;
            std::vector<float> means, scales, rots, opac, sh0, shape; // shape: raw r
        };

        struct Run {
            float loss = 0, area = 0;
            std::vector<float> g_means, g_scales, g_rots, g_opac, g_sh0, g_beta;
            int n_visible = 0;
        };

        // shape_bound: the --kernel-shape pointer bound (forward and footprint); weight: lambda (0 = off); bind_stats: the scalars
        Run render(const Scene& s, const bool shape_bound, const float weight, const bool opacity_grad, const bool bind_stats) {
            const int N = s.n;
            const size_t HW = static_cast<size_t>(s.w) * s.h;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_w2c(16), d_cam(3), d_shape(N), d_gbeta(N);
            DevBuf d_img(3 * HW), d_alpha(HW), d_depth(HW), d_wimg(3 * HW), d_stats(2);
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_shape.upload(s.shape);
            d_w2c.upload({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
            d_cam.upload({0, 0, 0});
            fr::FastGSKernelShape shape{};
            if (shape_bound) {
                shape.raw = d_shape.p;
                shape.grad = d_gbeta.p;
            }
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, nullptr, d_w2c.p, d_cam.p, d_img.p, d_alpha.p,
                                       d_depth.p, nullptr, nullptr, nullptr, N, 1, 1, s.w, s.h, s.fx, s.fy, s.cx, s.cy, 0.01f, 1e10f,
                                       false, nullptr, nullptr, 0, 0, nullptr, 1.0f, fr::FastGSLens{}, fr::FastGSDistortion{}, 0,
                                       fr::FastGSRollingShutter{}, shape);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            Run run;
            run.n_visible = ctx.n_visible;
            DevBuf g_means(3 * N), g_scales(3 * N), g_rots(4 * N), g_opac(N), g_sh0(3 * N);
            fr::FusedAdamSettings fa;
            fa.enabled = true;
            // as MRNF training sets it (scale_reg_log): the full preprocess-backward kernels in every run here, footprint on or
            // off, so "off vs measuring only" compares one kernel (scale_reg_weight stays 0: no scale regularisation)
            fa.scale_reg_log = true;
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
            fa.footprint_weight = weight;
            fa.footprint_opacity = opacity_grad;
            fa.footprint_max_px_per_fx = lfs::training::kGameMaxPixelRadiusPerFx;
            fa.footprint_shape_raw = shape_bound ? d_shape.p : nullptr;
            if (bind_stats) {
                fa.footprint_loss_out = d_stats.p;
                fa.footprint_area_out = d_stats.p + 1;
            }
            const auto res = fr::backward_raw(nullptr, nullptr, d_wimg.p, fr::BackgroundAlphaGradient{}, nullptr, nullptr, d_img.p,
                                              d_alpha.p, d_means.p, d_scales.p, d_rots.p, d_opac.p, nullptr, d_w2c.p, d_cam.p, ctx,
                                              nullptr, N, ctx.n_visible, 1, 1, s.w, s.h, s.fx, s.fy, s.cx, s.cy, false,
                                              DensificationType::None, &fa);
            cudaDeviceSynchronize();
            if (!res.success) {
                std::println(stderr, "backward failed: {}", res.error_message ? res.error_message : "?");
                std::exit(1);
            }
            const auto st = d_stats.download();
            run.loss = st[0];
            run.area = st[1];
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
            float tilt;        // rotation about the x axis (deg): the third axis then shows in the 2D covariance
        };

        Scene make_scene(const std::vector<SplatSpec>& specs) {
            Scene s;
            s.n = static_cast<int>(specs.size());
            for (const auto& p : specs) {
                const float x = (p.u - s.cx) * p.z / s.fx, y = (p.v - s.cy) * p.z / s.fy;
                s.means.insert(s.means.end(), {x, y, p.z});
                s.scales.insert(s.scales.end(), {std::log(p.s1 * p.z / s.fx), std::log(p.s2 * p.z / s.fx), std::log(1.5f * p.z / s.fx)});
                const float a = p.ang * 3.14159265358979f / 180.0f, t = p.tilt * 3.14159265358979f / 180.0f;
                // q = q_z(a) * q_x(t)
                const float cz = std::cos(0.5f * a), sz = std::sin(0.5f * a), cxh = std::cos(0.5f * t), sxh = std::sin(0.5f * t);
                s.rots.insert(s.rots.end(), {cz * cxh, cz * sxh, sz * sxh, sz * cxh});
                s.opac.push_back(std::log(p.o / (1.0f - p.o)));
                s.sh0.insert(s.sh0.end(), {0.5f, 0.2f, -0.3f});
                s.shape.push_back(ks::raw_from_beta(p.beta));
            }
            return s;
        }

        // float64 host reference: the summed game quad area / (W H) of the scene (pinhole J at the mean, J W Sigma W^T J^T + 0.3)
        double host_area(const Scene& s, const bool shape_bound) {
            const double C = std::sqrt(5.0), M = static_cast<double>(lfs::training::kGameMaxPixelRadiusPerFx) * s.fx;
            double sum = 0;
            for (int i = 0; i < s.n; ++i) {
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
                            cov[a][b] += R[a * 3 + k] * std::exp(2.0 * static_cast<double>(s.scales[3 * i + k])) * R[b * 3 + k];
                const double J[2][3] = {{s.fx / Z, 0, -s.fx * X / (Z * Z)}, {0, s.fy / Z, -s.fy * Y / (Z * Z)}};
                double Cv[2][2] = {};
                for (int r = 0; r < 2; ++r)
                    for (int q = 0; q < 2; ++q)
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                Cv[r][q] += J[r][a] * cov[a][b] * J[q][b];
                const double A = Cv[0][0] + 0.3, B = Cv[0][1], Cc = Cv[1][1] + 0.3;
                const double o = 1.0 / (1.0 + std::exp(-static_cast<double>(s.opac[i])));
                const double beta = shape_bound ? static_cast<double>(ks::beta_from_raw(s.shape[i])) : 2.0;
                double reach = 0;
                if (o * 510.0 > 1.0)
                    reach = beta == 2.0 ? C : std::min(C, std::pow(2.0 * std::log(510.0 * o), 1.0 / beta));
                const double avg = 0.5 * (A + Cc), det = A * Cc - B * B, dl = std::sqrt(std::max(avg * avg - det, 0.0));
                const double s1 = std::min(M, reach * std::sqrt(avg + dl)), s2 = std::min(M, reach * std::sqrt(std::max(avg - dl, 0.0)));
                sum += 4.0 * s1 * s2;
            }
            return sum / (static_cast<double>(s.w) * s.h);
        }

        int run_gradcheck() {
            // LFQPLCYKMOD: LFE_FOOTPRINT_CUT=C runs the check under the game's distance cut (--max-std-dev C --max-std-dev-distance);
            // the footprint's value and gradient do not depend on it (they are the game's quad), the forward's visibility could
            if (const char* c = std::getenv("LFE_FOOTPRINT_CUT"); c != nullptr && std::atof(c) > 0.0) {
                const auto cut = fr::set_splat_cutoff(static_cast<float>(std::atof(c)), true);
                std::println("splat cut on distance: C = {} (d^2/2 <= {}, min alpha {:.6f})", c, cut.half_z2_max, cut.min_alpha);
            }
            //            u      v      z     s1     s2   ang    o      beta  tilt
            const std::vector<SplatSpec> specs = {
                {30.0f, 30.0f, 2.0f, 3.0f, 2.0f, 20.0f, 0.95f, 1.3f, 25.0f},   // 0 long tails, opaque (reach = the cut)
                {70.0f, 40.0f, 2.2f, 5.0f, 2.5f, 30.0f, 0.30f, 6.0f, -40.0f},  // 1 flat top (reach inside the cut)
                {120.0f, 30.0f, 2.4f, 4.0f, 3.0f, -15.0f, 0.03f, 2.0f, 10.0f}, // 2 beta exactly 2: a Gaussian (the cut)
                {40.0f, 90.0f, 2.6f, 3.0f, 3.5f, 0.0f, 0.008f, 1.3f, 60.0f},   // 3 faint long tails (contour inside the cut)
                {100.0f, 80.0f, 2.8f, 21.0f, 4.0f, 45.0f, 0.90f, 3.0f, 15.0f}, // 4 one axis over the 33 px cap
                {80.0f, 64.0f, 3.0f, 30.0f, 25.0f, 60.0f, 0.85f, 1.8f, 5.0f},  // 5 both axes over the cap
                {135.0f, 105.0f, 3.2f, 6.0f, 1.5f, 80.0f, 0.50f, 7.9f, -20.0f},// 6 nearly box (flat top)
                {15.0f, 110.0f, 3.4f, 3.5f, 3.5f, 0.0f, 0.03f, 3.0f, 0.0f},    // 7 faint flat top, isotropic in 2D
            };
            const Scene s0 = make_scene(specs);
            int bad = 0, blocks = 0;
            std::vector<double> rels;
            bool ok_all = true;
            for (const bool bound : {true, false}) {
                for (const bool opac : {false, true}) {
                    const float lam = 1.0f;
                    const Run base = render(s0, bound, lam, opac, true);
                    const double ref = host_area(s0, bound);
                    const double rel_v = std::abs(base.area - ref) / ref, rel_l = std::abs(base.loss - lam * ref) / (lam * ref);
                    std::println("{} shapes, opacity gradient {}: {} splats ({} visible); mean footprint {:.6f} splats per pixel "
                                 "(host float64 {:.6f}, rel {:.1e}); loss {:.6f} (rel {:.1e})",
                                 bound ? "bound" : "unbound (all Gaussians)", opac ? "on" : "off", s0.n, base.n_visible, base.area, ref,
                                 rel_v, base.loss, rel_l);
                    if (rel_v > 1e-4 || rel_l > 1e-4) {
                        std::println("  VALUE MISMATCH");
                        ok_all = false;
                    }
                    // gradients that must be exactly zero
                    double gz = 0;
                    for (const float g : base.g_means)
                        gz = std::max(gz, static_cast<double>(std::abs(g)));
                    for (const float g : base.g_rots)
                        gz = std::max(gz, static_cast<double>(std::abs(g)));
                    for (const float g : base.g_beta)
                        gz = std::max(gz, static_cast<double>(std::abs(g)));
                    double go = 0;
                    for (const float g : base.g_opac)
                        go = std::max(go, static_cast<double>(std::abs(g)));
                    std::println("  means / rotations / beta: max |g| {:.3g} (must be 0); opacity max |g| {:.3g}{}", gz, go,
                                 opac ? "" : " (must be 0 without the option)");
                    if (gz != 0.0 || (!opac && go != 0.0))
                        ok_all = false;
                    auto block = [&](const char* name, std::vector<float> Scene::*member, const std::vector<float>& analytic,
                                     const int per, const int i, const double hstep) {
                        // numd: vs central differences of the DEVICE (float32) loss; its resolution limits it (noise below)
                        double num = 0, den = 0, numd = 0, dend = 0, noise2 = 0;
                        for (int k = 0; k < per; ++k) {
                            Scene sp = s0, sm = s0;
                            (sp.*member)[per * i + k] += static_cast<float>(hstep);
                            (sm.*member)[per * i + k] -= static_cast<float>(hstep);
                            const double h2 = static_cast<double>((sp.*member)[per * i + k]) - (sm.*member)[per * i + k];
                            const double fd = lam * (host_area(sp, bound) - host_area(sm, bound)) / h2;
                            const double fdd = (static_cast<double>(render(sp, bound, lam, opac, true).loss) -
                                                render(sm, bound, lam, opac, true).loss) / h2;
                            const double an = analytic[per * i + k];
                            num += (fd - an) * (fd - an);
                            den += fd * fd;
                            numd += (fdd - an) * (fdd - an);
                            dend += fdd * fdd;
                            // float32 loss: a few ulps of |loss| (block-reduced, then one atomicAdd) over the step
                            const double nz = 4.0 * 1.19e-7 * std::abs(static_cast<double>(base.loss)) / h2;
                            noise2 += nz * nz;
                        }
                        const double rel = std::sqrt(num / std::max(den, 1e-300)), reld = std::sqrt(numd / std::max(dend, 1e-300));
                        const bool sig = std::sqrt(den) > 1e-9;
                        // the device-loss comparison fails only beyond its own float32 noise floor
                        const bool dev_bad = reld > 0.02 && std::sqrt(numd) > 3.0 * std::sqrt(noise2);
                        ++blocks;
                        if (sig) {
                            rels.push_back(rel);
                            if (rel > 0.01 || dev_bad)
                                ++bad;
                        }
                        std::println("    splat {} (beta {:.2f}, o {:.3f}) {:<12} |fd| {:9.3e}  rel err vs host fd {:8.2e}, vs device-loss fd {:8.2e} "
                                     "(its float32 floor {:8.2e}){}",
                                     i, bound ? ks::beta_from_raw(s0.shape[i]) : 2.0f, 1.0f / (1.0f + std::exp(-s0.opac[i])), name,
                                     std::sqrt(den), rel, reld, std::sqrt(noise2 / std::max(dend, 1e-300)),
                                     !sig ? "  (no signal)" : (rel > 0.01 || dev_bad) ? "  OVER" : "");
                    };
                    for (int i = 0; i < s0.n; ++i) {
                        block("log scales", &Scene::scales, base.g_scales, 3, i, 1e-3);
                        if (opac)
                            block("raw opacity", &Scene::opac, base.g_opac, 1, i, 1e-3);
                    }
                    // 4. measuring only (weight 0, scalars bound) vs off: identical gradients
                    if (!opac) {
                        const Run off = render(s0, bound, 0.0f, false, false), meas = render(s0, bound, 0.0f, false, true);
                        auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
                            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
                        };
                        const bool id = same(off.g_means, meas.g_means) && same(off.g_scales, meas.g_scales) && same(off.g_rots, meas.g_rots) &&
                                        same(off.g_opac, meas.g_opac) && same(off.g_sh0, meas.g_sh0) && same(off.g_beta, meas.g_beta);
                        std::println("  off vs measuring only (weight 0, scalars bound): gradients {}; measured {:.6f} splats per pixel",
                                     id ? "BIT-IDENTICAL" : "DIFFER", meas.area);
                        if (!id)
                            ok_all = false;
                    }
                }
            }
            std::sort(rels.begin(), rels.end());
            std::println("{} blocks ({} with signal): median rel err {:.2e}, max {:.2e}; {} over the threshold", blocks, rels.size(),
                         rels.empty() ? 0.0 : rels[rels.size() / 2], rels.empty() ? 0.0 : rels.back(), bad);
            const bool ok = ok_all && bad == 0 && !rels.empty();
            std::println("{}", ok ? "FOOTPRINT GRADCHECK OK" : "FOOTPRINT GRADCHECK FAILED");
            return ok ? 0 : 1;
        }
    } // namespace

    int run_lfe_footprint_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "gradcheck")
            return run_gradcheck();
        std::println(stderr, "usage: LichtFeld-Studio lfe-footprint gradcheck");
        return 2;
    }

} // namespace lfs::app
