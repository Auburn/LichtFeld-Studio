/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Developer check of the near-camera gradient scaling, --grad-scale-near D0 (LFQPLCYKMGS; after Philip & Deschaintre, "Floaters
// No More", EGSR 2023; fused into the FastGS preprocess backward, fused_adam_types.h):
//   lfe-gradnear check [lens_radial.txt]
// Three scenes, a camera at C = (0.3, -0.2, 0.1), turned; splats at distances 0.3 .. 3.5 D0 (D0 = 0.52, the 1 m arm), on and off
// the axis (camera z down to 0.34 of the distance), some behind one another; SH degree 1, kernel shapes (beta 1.4 .. 4), image +
// depth weights, the depth distortion term, the footprint loss and the scale / opacity / erank / dc regularisers all on:
//   pinhole       the plain FastGS kernels
//   fisheye       native fisheye (the woodland k-polynomial) + the pose gradient (--pose-opt)
//   recipe        native fisheye through the radial lens table (the argument; else the k-polynomial) + the dynamic rolling shutter
//                 with its gradient (--rs-learn), the motion blur with its exposure-scale gradient, the pose gradient: the
//                 recipe's kernels (FISHEYE RS POSE RADIAL BLUR SHAPE)
// Captured gradients (gradient_out): A = option off, B = on, R = the regularisers alone (no image / depth / distortion /
// footprint), R' = the same with the option on. With s = min(1, |mean - C|^2 / D0^2) from the host (float64):
//   1. d < D0: B - R = s (A - R) for means, log-scales, rotation, raw opacity, sh0, shN and beta (error < 1e-5 of the block);
//   2. d >= D0: B bit-identical to A (an untouched splat), when A repeats bit for bit (else within 4x A's run-to-run spread);
//   3. R' bit-identical to R (the regularisers are not scaled);
//   4. the camera side (pose [6], rolling-shutter motion + readout [13], exposure scale) of B = that of A (same rule as 2);
//   5. the scalars: visible splats, splats with s < 1 and the sum of their s = the host's;
//   6. D0 below every splat's distance: every gradient of B = A (same rule as 2).
// "GRADNEAR CHECK OK" or "GRADNEAR CHECK FAILED".
#include "core/kernel_shape_map.hpp"
#include "lfs/training/game_footprint.cuh"
#include "training/rasterization/radial_lens_table.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <numbers>
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
                    cudaMemcpy(p, v.data(), sizeof(float) * std::min(v.size(), std::max<size_t>(n, 1)), cudaMemcpyHostToDevice);
            }
            std::vector<float> download() const {
                std::vector<float> v(n);
                cudaMemcpy(v.data(), p, sizeof(float) * n, cudaMemcpyDeviceToHost);
                return v;
            }
        };

        constexpr float kD0 = 0.52f;  // the 1 m arm (scene units)
        constexpr int kShBases = 4;   // SH degree 1
        constexpr int kShSlots = 3;   // shSlotsForBases(4)
        constexpr int kShReorder = 32; // config::sh_reorder_size

        struct Scene {
            std::string name;
            int n = 0, w = 192, h = 144;
            float fx = 100, fy = 100, cx = 96, cy = 72;
            fr::FastGSLens lens{};
            bool pose = false, rs = false;
            std::vector<float> means, scales, rots, opac, sh0, shN, shape, w2c, cam, w_img, w_depth, motion;
            std::vector<double> dist;  // |mean - C|
            std::vector<double> zcam;  // camera z
            std::vector<double> theta; // off-axis angle, degrees
        };

        struct Opts {
            bool image = true;   // image + depth weights, distortion, footprint (off = the regularisers alone)
            float d0 = 0.0f;     // --grad-scale-near (0 = off)
            bool stats = false;  // bind the [3] scalars
        };

        struct Run {
            std::vector<float> g_means, g_scales, g_rots, g_opac, g_sh0, g_shN, g_beta, g_pose, g_rs, g_exp, stats;
            int n_visible = 0;
        };

        Run render(const Scene& s, const Opts& o) {
            const int N = s.n;
            const size_t HW = static_cast<size_t>(s.w) * s.h;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_shN(s.shN.size()), d_w2c(16), d_cam(3),
                d_shape(N), d_gbeta(N);
            DevBuf d_img(3 * HW), d_alpha(HW), d_depth(HW), d_wimg(3 * HW), d_wdepth(HW);
            DevBuf d_mom(3 * HW), d_dloss(1), d_ddepths(N), d_dgrad(N);
            DevBuf d_motion(12), d_readout(1), d_rsg(13), d_exp(1), d_escale(1), d_gexp(1), d_pg(6), d_stats(3), d_times(N);
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_shN.upload(s.shN);
            d_shape.upload(s.shape);
            d_w2c.upload(s.w2c);
            d_cam.upload(s.cam);
            if (o.image) {
                d_wimg.upload(s.w_img);
                d_wdepth.upload(s.w_depth);
            }
            fr::FastGSKernelShape shape{};
            shape.raw = d_shape.p;
            shape.grad = d_gbeta.p;
            fr::FastGSDistortion dist{};
            if (o.image) {
                dist.moments = d_mom.p;
                dist.loss_out = d_dloss.p;
                dist.depths = d_ddepths.p;
                dist.grad_depths = d_dgrad.p;
                dist.loss_scale = 0.5f / static_cast<float>(HW);
                dist.grad_scale = dist.loss_scale;
                dist.near_n = 0.05f;
                dist.far_n = 1000.0f;
                dist.depth_mode = fr::kDistortionDepthDistance;
                dist.plane_sigma = 2.0f;
                dist.min_dist = 0.0f;
            }
            fr::FastGSRollingShutter rs{};
            if (s.rs) {
                d_motion.upload(s.motion);
                d_readout.upload({0.031f});
                d_exp.upload({0.004f});
                d_escale.upload({1.0f});
                rs.motion = d_motion.p;
                rs.readout = d_readout.p;
                rs.grad = d_rsg.p;
                rs.times = d_times.p; // the forward writes each drawn splat's row time (as lfe-pose / lfe-blur bind it)
                rs.exposure = d_exp.p;
                rs.exposure_scale = d_escale.p;
                rs.grad_exposure = d_gexp.p;
            }
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, d_shN.p, d_w2c.p, d_cam.p, d_img.p, d_alpha.p,
                                       d_depth.p, nullptr, nullptr, nullptr, N, kShBases, kShBases, s.w, s.h, s.fx, s.fy, s.cx, s.cy,
                                       0.01f, 1e10f, false, nullptr, nullptr, 0, 0, nullptr, 1.0f, s.lens, dist, 0, rs, shape);
            if (!ctx.success) {
                std::println(stderr, "{}: forward failed: {}", s.name, ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            if (s.pose)
                ctx.pose_grad = d_pg.p;
            DevBuf g_means(3 * N), g_scales(3 * N), g_rots(4 * N), g_opac(N), g_sh0(3 * N), g_shN(s.shN.size());
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
            capture(fa.shN, g_shN, d_shN, 9);
            // the per-splat regularisers (as MRNF trains: the log-scale form), always on
            fa.scale_reg_weight = 0.05f;
            fa.scale_reg_log = true;
            fa.scale_reg_normalizer = 1.0f;
            fa.opacity_reg_weight = 0.02f;
            fa.erank_reg_weight = 0.01f;
            fa.dc_reg_weight = 0.03f;
            if (o.image) { // the footprint loss: a per-view term (scaled with the option)
                fa.footprint_weight = 0.3f;
                fa.footprint_max_px_per_fx = lfs::training::kGameMaxPixelRadiusPerFx;
                fa.footprint_shape_raw = d_shape.p;
            }
            if (o.d0 > 0.0f) { // as fast_rasterizer.cpp's make_fastgs_fused_adam_settings sets it
                fa.grad_scale_near_inv_d0_sq = 1.0f / (o.d0 * o.d0);
                if (o.stats)
                    fa.grad_scale_near_stats_out = d_stats.p;
            }
            const auto res = fr::backward_raw(nullptr, nullptr, d_wimg.p, fr::BackgroundAlphaGradient{}, d_wdepth.p, nullptr, d_img.p,
                                              d_alpha.p, d_means.p, d_scales.p, d_rots.p, d_opac.p, d_shN.p, d_w2c.p, d_cam.p, ctx, nullptr,
                                              N, ctx.n_visible, kShBases, kShBases, s.w, s.h, s.fx, s.fy, s.cx, s.cy, false,
                                              DensificationType::None, &fa);
            cudaDeviceSynchronize();
            if (!res.success) {
                std::println(stderr, "{}: backward failed: {}", s.name, res.error_message ? res.error_message : "?");
                std::exit(1);
            }
            Run r;
            r.n_visible = ctx.n_visible;
            r.g_means = g_means.download();
            r.g_scales = g_scales.download();
            r.g_rots = g_rots.download();
            r.g_opac = g_opac.download();
            r.g_sh0 = g_sh0.download();
            r.g_shN = g_shN.download();
            r.g_beta = d_gbeta.download();
            r.g_pose = d_pg.download();
            r.g_rs = d_rsg.download();
            r.g_exp = d_gexp.download();
            r.stats = d_stats.download();
            return r;
        }

        // the splat's values of one parameter block (shN: its 3 float4 slots of the swizzled layout, 12 floats)
        std::vector<float> block(const Run& r, const int k, const int i) {
            auto take = [i](const std::vector<float>& v, int per) { return std::vector<float>(v.begin() + per * i, v.begin() + per * (i + 1)); };
            switch (k) {
            case 0: return take(r.g_means, 3);
            case 1: return take(r.g_scales, 3);
            case 2: return take(r.g_rots, 4);
            case 3: return take(r.g_opac, 1);
            case 4: return take(r.g_sh0, 3);
            case 5: {
                std::vector<float> out;
                for (int slot = 0; slot < kShSlots; ++slot)
                    for (int c = 0; c < 4; ++c)
                        out.push_back(r.g_shN[4 * static_cast<size_t>((i / kShReorder) * kShSlots * kShReorder + slot * kShReorder + i % kShReorder) + c]);
                return out;
            }
            default: return take(r.g_beta, 1);
            }
        }
        constexpr int kBlocks = 7;
        const char* const kBlockName[kBlocks] = {"means", "log-scales", "rotation", "raw opacity", "sh0", "shN", "beta"};

        bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        }
        double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
            double m = 0;
            for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
                m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
            return m;
        }
        std::vector<float> all(const Run& r) {
            std::vector<float> v;
            for (const auto* x : {&r.g_means, &r.g_scales, &r.g_rots, &r.g_opac, &r.g_sh0, &r.g_shN, &r.g_beta})
                v.insert(v.end(), x->begin(), x->end());
            return v;
        }

        struct SplatSpec {
            double theta, phi, f, beta; // off-axis angle and azimuth (deg), distance / D0, kernel shape
        };

        // rotation of the camera (world -> camera rows) from a fixed unit quaternion
        std::array<double, 9> cam_rotation() {
            double qw = 0.93, qx = 0.12, qy = -0.27, qz = 0.21;
            const double n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qw /= n, qx /= n, qy /= n, qz /= n;
            return {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
                    2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                    2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)};
        }

        Scene make_scene(const std::string& name, const int kind, const lfs::training::RadialLensTable* table) {
            Scene s;
            s.name = name;
            double max_theta = 30.0, f_px = 100.0;
            if (kind >= 1) {
                s.lens.model = fr::kLensFisheye;
                s.lens.k1 = 0.39127139f; // the woodland k-polynomial (lfe-pose's)
                s.lens.k2 = -0.03901020f;
                s.lens.k3 = 0.00243290f;
                s.lens.k4 = -0.00266145f;
                const double tmax = std::tan(88.0 * std::numbers::pi / 180.0);
                s.lens.max_w = static_cast<float>(tmax * tmax);
                s.fx = s.fy = 30.0f;
                f_px = 30.0 * 1.4;
                max_theta = 72.0;
                s.pose = true;
            }
            if (kind == 2) {
                s.rs = true;
                s.motion = {0.9f, 1.6f, 0.4f, 3.0f, -2.0f, 1.0f, 2.5f, 0.8f, 1.5f, 5.0f, -3.0f, 2.0f}; // lfe-blur's
                max_theta = 60.0;
                if (table != nullptr) { // the table's camera at quarter resolution, drawn through the table (lfe-pose's)
                    s.w = table->width / 4;
                    s.h = table->height / 4;
                    s.fx = static_cast<float>(table->fx / 4);
                    s.fy = static_cast<float>(table->fy / 4);
                    s.cx = static_cast<float>(table->cx / 4);
                    s.cy = static_cast<float>(table->cy / 4);
                    s.lens.k1 = static_cast<float>(table->k[0]);
                    s.lens.k2 = static_cast<float>(table->k[1]);
                    s.lens.k3 = static_cast<float>(table->k[2]);
                    s.lens.k4 = static_cast<float>(table->k[3]);
                    s.lens.radial = table->device_view();
                    f_px = s.fx * 1.4;
                }
            }
            const auto R = cam_rotation();
            const std::array<double, 3> C = {0.3, -0.2, 0.1};
            s.w2c.assign(16, 0.0f);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    s.w2c[i * 4 + j] = static_cast<float>(R[i * 3 + j]);
                s.w2c[i * 4 + 3] = static_cast<float>(-(R[i * 3] * C[0] + R[i * 3 + 1] * C[1] + R[i * 3 + 2] * C[2]));
            }
            s.w2c[15] = 1.0f;
            s.cam = {static_cast<float>(C[0]), static_cast<float>(C[1]), static_cast<float>(C[2])};
            // (theta as a share of the scene's widest angle, azimuth, d / D0, beta); pairs on one ray: the near one in front
            const std::vector<SplatSpec> specs = {
                {0.00, 0, 0.30, 2.0}, {0.35, 40, 0.50, 1.4}, {0.35, 40, 2.00, 3.0}, {0.70, 120, 0.70, 2.6}, {1.00, 200, 0.85, 1.8},
                {1.00, 200, 1.30, 2.0}, {0.55, 280, 0.97, 4.0}, {0.90, 330, 1.03, 2.2}, {0.20, 160, 1.50, 1.6}, {0.80, 70, 2.50, 3.3},
                {0.45, 240, 3.50, 2.0}, {0.95, 15, 0.60, 2.9}};
            for (const auto& p : specs) {
                const double th = p.theta * max_theta * std::numbers::pi / 180.0, ph = p.phi * std::numbers::pi / 180.0;
                const double d = p.f * kD0;
                const std::array<double, 3> pc = {d * std::sin(th) * std::cos(ph), d * std::sin(th) * std::sin(ph), d * std::cos(th)};
                std::array<double, 3> m{};
                for (int j = 0; j < 3; ++j) // world = R^T pc + C
                    m[j] = R[j] * pc[0] + R[3 + j] * pc[1] + R[6 + j] * pc[2] + C[j];
                for (int j = 0; j < 3; ++j)
                    s.means.push_back(static_cast<float>(m[j]));
                const double dx = s.means[s.means.size() - 3] - static_cast<double>(s.cam[0]),
                             dy = s.means[s.means.size() - 2] - static_cast<double>(s.cam[1]),
                             dz = s.means[s.means.size() - 1] - static_cast<double>(s.cam[2]);
                s.dist.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
                s.zcam.push_back(pc[2]);
                s.theta.push_back(th * 180.0 / std::numbers::pi);
                const double px = 2.5 + 0.4 * static_cast<double>(s.n % 3); // screen sigma, pixels
                s.scales.insert(s.scales.end(), {static_cast<float>(std::log(px * d / f_px)), static_cast<float>(std::log(0.6 * px * d / f_px)),
                                                 static_cast<float>(std::log(0.3 * px * d / f_px))});
                const double a = 0.7 * s.n + 0.3;
                s.rots.insert(s.rots.end(), {static_cast<float>(std::cos(a)), static_cast<float>(0.4 * std::sin(a)), static_cast<float>(0.3),
                                             static_cast<float>(-0.5 * std::sin(1.3 * a))});
                const double o = 0.35 + 0.05 * (s.n % 9);
                s.opac.push_back(static_cast<float>(std::log(o / (1 - o))));
                // sh0: two of them past the dc regulariser's limit (0.5 / C0 = 1.77)
                s.sh0.insert(s.sh0.end(), {static_cast<float>(s.n % 5 == 0 ? 2.2 : 0.4 + 0.1 * (s.n % 4)), -0.3f, static_cast<float>(0.2 * (s.n % 3))});
                s.shape.push_back(ks::raw_from_beta(static_cast<float>(p.beta)));
                ++s.n;
            }
            s.shN.assign(static_cast<size_t>((s.n + kShReorder - 1) / kShReorder) * kShSlots * kShReorder * 4, 0.0f);
            for (size_t i = 0; i < s.shN.size(); ++i)
                s.shN[i] = static_cast<float>(0.25 * std::sin(0.37 * static_cast<double>(i) + 0.5));
            s.w_img.resize(3ull * s.w * s.h);
            s.w_depth.resize(static_cast<size_t>(s.w) * s.h);
            for (int y = 0; y < s.h; ++y)
                for (int x = 0; x < s.w; ++x) {
                    for (int c = 0; c < 3; ++c)
                        s.w_img[(static_cast<size_t>(c) * s.h + y) * s.w + x] =
                            static_cast<float>(std::cos(0.05 * x + 0.9 * c) * std::sin(0.07 * y + 0.3) + 0.3);
                    s.w_depth[static_cast<size_t>(y) * s.w + x] = static_cast<float>(0.2 * std::sin(0.03 * x + 0.05 * y));
                }
            return s;
        }

        int check_scene(const Scene& s, bool& ok) {
            auto gate = [&ok](const bool pass, const std::string& what) {
                std::println("  {} {}", pass ? "ok  " : "FAIL", what);
                if (!pass)
                    ok = false;
            };
            const Run A = render(s, {true, 0.0f, false}), A2 = render(s, {true, 0.0f, false});
            const Run B = render(s, {true, kD0, true});
            const Run Rr = render(s, {false, 0.0f, false}), Rn = render(s, {false, kD0, false});
            const Run F = render(s, {true, 0.05f, false}); // D0 below every splat's distance (the nearest is at 0.156)
            gate(A.n_visible == s.n && B.n_visible == s.n && Rr.n_visible == s.n,
                 std::format("{}: all {} splats visible ({} / {} / {})", s.name, s.n, A.n_visible, B.n_visible, Rr.n_visible));
            const bool repeat = same_bits(all(A), all(A2)) && same_bits(A.g_pose, A2.g_pose) && same_bits(A.g_rs, A2.g_rs) &&
                                same_bits(A.g_exp, A2.g_exp);
            const double spread = std::max(max_abs_diff(all(A), all(A2)),
                                           std::max(max_abs_diff(A.g_pose, A2.g_pose), max_abs_diff(A.g_rs, A2.g_rs)));
            std::println("  {}: option off, run to run: {} (max |diff| {:.3e})", s.name, repeat ? "bit-identical" : "differs", spread);
            auto untouched = [&](const std::vector<float>& a, const std::vector<float>& b) {
                return repeat ? same_bits(a, b) : max_abs_diff(a, b) <= 4.0 * spread + 1e-30;
            };
            const char* const rule = repeat ? "bit-identical" : "within 4x the run-to-run spread";
            std::println("    splat   d/D0   theta    z/D0   s (host)   s (measured, means)   worst block error   d >= D0: B = A");
            double worst = 0;
            int n_near = 0;
            double s_sum = 0;
            bool far_ok = true;
            for (int i = 0; i < s.n; ++i) {
                const double d = s.dist[i], sh = std::min(1.0, d * d / (static_cast<double>(kD0) * kD0));
                double block_worst = 0, num = 0, den = 0;
                bool same_i = true;
                for (int k = 0; k < kBlocks; ++k) {
                    const auto a = block(A, k, i), b = block(B, k, i), r = block(Rr, k, i);
                    if (sh < 1.0) {
                        double e = 0, sc = 0;
                        for (size_t j = 0; j < a.size(); ++j) {
                            const double pa = static_cast<double>(a[j]) - r[j], pb = static_cast<double>(b[j]) - r[j];
                            e = std::max(e, std::abs(pb - sh * pa));
                            sc = std::max(sc, std::abs(pa) + std::abs(static_cast<double>(r[j])));
                            if (k == 0) {
                                num += pb * pa;
                                den += pa * pa;
                            }
                        }
                        if (sc > 0)
                            block_worst = std::max(block_worst, e / sc);
                    } else if (!untouched(a, b)) {
                        same_i = false;
                    }
                }
                if (sh < 1.0) {
                    ++n_near;
                    s_sum += sh;
                    worst = std::max(worst, block_worst);
                } else if (!same_i) {
                    far_ok = false;
                }
                std::println("    {:5d}  {:5.2f}  {:5.1f}  {:6.3f}   {:8.5f}   {:>19}   {:>17}   {}", i, d / kD0, s.theta[i], s.zcam[i] / kD0, sh,
                             sh < 1.0 ? std::format("{:.5f}", den > 0 ? num / den : 0.0) : "-",
                             sh < 1.0 ? std::format("{:.2e}", block_worst) : "-", sh < 1.0 ? "-" : (same_i ? "yes" : "NO"));
            }
            gate(worst < 1e-5, std::format("{}: d < D0 ({} splats): B - R = s (A - R) in every block ({}), worst error {:.2e} of the block (< 1e-5)",
                                           s.name, n_near, "means, log-scales, rotation, raw opacity, sh0, shN, beta", worst));
            gate(far_ok, std::format("{}: d >= D0 ({} splats): every block of B {} to A (untouched)", s.name, s.n - n_near, rule));
            gate(same_bits(all(Rr), all(Rn)), std::format("{}: the regularisers alone, option on vs off: bit-identical (not scaled)", s.name));
            if (s.pose)
                gate(untouched(A.g_pose, B.g_pose),
                     std::format("{}: pose gradient [6] of B {} to A (max |diff| {:.2e}; |g| {:.3e})", s.name, rule, max_abs_diff(A.g_pose, B.g_pose),
                                 std::abs(static_cast<double>(A.g_pose[0])) + std::abs(static_cast<double>(A.g_pose[3]))));
            if (s.rs) {
                gate(untouched(A.g_rs, B.g_rs), std::format("{}: rolling-shutter motion + readout gradient [13] of B {} to A (max |diff| {:.2e})",
                                                            s.name, rule, max_abs_diff(A.g_rs, B.g_rs)));
                gate(untouched(A.g_exp, B.g_exp), std::format("{}: exposure-scale gradient of B {} to A ({:.6e} vs {:.6e})", s.name, rule,
                                                             A.g_exp[0], B.g_exp[0]));
            }
            gate(static_cast<int>(B.stats[0]) == s.n && static_cast<int>(B.stats[1]) == n_near && std::abs(B.stats[2] - s_sum) < 1e-4,
                 std::format("{}: the scalars: {} visible, {} with s < 1, sum of s {:.5f} (host {} / {} / {:.5f})", s.name, B.stats[0], B.stats[1],
                             B.stats[2], s.n, n_near, s_sum));
            gate(untouched(all(A), all(F)) && (!s.pose || untouched(A.g_pose, F.g_pose)),
                 std::format("{}: D0 0.05 (below every splat's distance): every gradient {} to the option off", s.name, rule));
            return 0;
        }

        int run_check(const char* table_path) {
            if (table_path != nullptr) {
                lfs::training::activate_radial_lens(lfs::training::load_radial_lens_table(table_path));
                std::println("lens of the recipe scene: the radial lens table {}", lfs::training::active_radial_lens()->describe());
            } else {
                std::println("lens of the recipe scene: the woodland k-polynomial (no table given)");
            }
            std::println("near-camera gradient scaling check: D0 = {} scene units, s = min(1, |mean - C|^2 / D0^2); camera centre (0.3, -0.2, 0.1)", kD0);
            bool ok = true;
            check_scene(make_scene("pinhole", 0, nullptr), ok);
            check_scene(make_scene("fisheye", 1, nullptr), ok);
            check_scene(make_scene("recipe", 2, lfs::training::active_radial_lens()), ok);
            std::println("{}", ok ? "GRADNEAR CHECK OK" : "GRADNEAR CHECK FAILED");
            return ok ? 0 : 1;
        }
    } // namespace

    int run_lfe_gradnear_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "check")
            return run_check(argc >= 4 ? argv[3] : nullptr);
        std::println(stderr, "usage: LichtFeld-Studio lfe-gradnear check [lens_radial.txt]");
        return 2;
    }

} // namespace lfs::app
