/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Developer check of the anti-needle (anisotropy) loss, --aniso-max-ratio r --aniso-weight lambda (LFQPLCYKMAN; after
// PhysGaussian's anisotropy regulariser; fused into the FastGS preprocess backward like erank, fused_adam_types.h):
//   lfe-aniso check [lens_radial.txt]
// The lfe-gradnear scenes (pinhole, fisheye + pose, the recipe's kernels through the radial table), with the splats' log-scales
// replaced by a set of shapes: round, flat discs, a near-tie, ratios just under / over r, needles along each axis, extreme needles, plus
// splats BEHIND the camera (not drawn in the view: the loss covers every splat). Image, depth, distortion, footprint and the
// scale / opacity / erank / dc regularisers on. Captured gradients (gradient_out): A = option off, B = on, and the loss / counts.
//   1. B - A on the log-scales (the regularisers-only path, which repeats bit for bit) = the host's analytic gradient (lambda / N) (rho e_max - rho e_mid) for rho > r, 0 otherwise
//      (error < 1e-5 relative to the block), for drawn and undrawn splats;
//   2. every other block (means, rotation, raw opacity, sh0, shN, beta) of B bit-identical to A on the regulariser path (with the
//      image terms: reported against the off path's run-to-run spread); the camera side of B = A (bit-identical when A
//      repeats bit for bit, else within 4x the run-to-run spread);
//   3. central finite differences of the device loss (loss_out; h = 1e-3 in the log-scales) = the analytic gradient (relative
//      error < 2e-2 on every component with |g| > 1e-3 of the largest);
//   4. loss_out and the counts [splats, splats over r] = the host's;
//   5. weight 0 (the count-only binding before --aniso-start): every gradient bit-identical to the option off (regulariser path).
// "ANISO CHECK OK" or "ANISO CHECK FAILED".
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

        constexpr float kD0 = 0.52f;  // the lfe-gradnear scene scale (scene units)
        constexpr float kR = 5.0f;    // --aniso-max-ratio of the check
        constexpr float kLambda = 2.0f; // --aniso-weight of the check
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
            float r = 0.0f;      // --aniso-max-ratio (0 = off)
            float lambda = kLambda;
            bool stats = false;  // bind loss + counts
            const std::vector<float>* scales = nullptr; // override the scene's log-scales (finite differences)
        };

        struct Run {
            std::vector<float> g_means, g_scales, g_rots, g_opac, g_sh0, g_shN, g_beta, g_pose, g_rs, g_exp, stats, loss;
            int n_visible = 0;
        };

        Run render(const Scene& s, const Opts& o) {
            const int N = s.n;
            const size_t HW = static_cast<size_t>(s.w) * s.h;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_shN(s.shN.size()), d_w2c(16), d_cam(3),
                d_shape(N), d_gbeta(N);
            DevBuf d_img(3 * HW), d_alpha(HW), d_depth(HW), d_wimg(3 * HW), d_wdepth(HW);
            DevBuf d_mom(3 * HW), d_dloss(1), d_ddepths(N), d_dgrad(N);
            DevBuf d_motion(12), d_readout(1), d_rsg(13), d_exp(1), d_escale(1), d_gexp(1), d_pg(6), d_stats(2), d_loss(1), d_times(N);
            d_means.upload(s.means);
            d_scales.upload(o.scales ? *o.scales : s.scales);
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
            if (o.r > 0.0f) { // as fast_rasterizer.cpp sets it
                fa.aniso_max_ratio = o.r;
                fa.aniso_weight = o.lambda;
                if (o.stats) {
                    fa.aniso_stats_out = d_stats.p;
                    fa.aniso_loss_out = d_loss.p;
                }
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
            r.loss = d_loss.download();
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
            // LFQPLCYKMAN: the shapes (log-scales relative to the splat's own middle size) and 4 splats behind the camera
            const std::vector<std::array<double, 3>> shapes = {
                {0, 0, 0}, {0, 0, -3}, {0.7, 0.7, -2.5}, {std::log(3.0), 0, -0.5}, {std::log(4.9), 0, 0}, {0, std::log(5.1), -0.2},
                {-0.4, 0, std::log(8.0)}, {std::log(12.0), 0.1, 0}, {0, -1, std::log(40.0)}, {std::log(5.0), 0, 0.3}, {0.2, std::log(200.0), 0.25},
                {std::log(15.0), std::log(2.0), 0}};
            for (int i = 0; i < s.n; ++i) {
                const double l0 = s.scales[3 * i + 1];
                for (int j = 0; j < 3; ++j)
                    s.scales[3 * i + j] = static_cast<float>(l0 + shapes[i % shapes.size()][j]);
            }
            for (int k = 0; k < 4; ++k) { // behind the camera (camera z < 0): not drawn in this view
                const std::array<double, 3> pc = {0.1 * k, -0.05 * k, -0.6 - 0.3 * k};
                for (int j = 0; j < 3; ++j)
                    s.means.push_back(static_cast<float>(R[j] * pc[0] + R[3 + j] * pc[1] + R[6 + j] * pc[2] + C[j]));
                const double l0 = std::log(0.01);
                const std::array<double, 3> sh[4] = {{std::log(9.0), 0, -1}, {0, 0, 0}, {0, -1, std::log(6.0)}, {std::log(4.0), 0, 0}};
                for (int j = 0; j < 3; ++j)
                    s.scales.push_back(static_cast<float>(l0 + sh[k][j]));
                s.rots.insert(s.rots.end(), {1.0f, 0.1f * static_cast<float>(k), 0.0f, 0.0f});
                s.opac.push_back(0.3f);
                s.sh0.insert(s.sh0.end(), {0.3f, 0.2f, 0.1f});
                s.shape.push_back(ks::raw_from_beta(2.0f));
                s.dist.push_back(0);
                s.zcam.push_back(pc[2]);
                s.theta.push_back(180);
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

        // host float64: the loss and its gradient on the log-scales (the kernel's argmax / tie rule)
        double host_aniso(const std::vector<float>& l, const int n, const double r, const double lambda, std::vector<double>* g, int* over) {
            double loss = 0;
            if (g)
                g->assign(3 * static_cast<size_t>(n), 0.0);
            if (over)
                *over = 0;
            for (int i = 0; i < n; ++i) {
                const float* v = &l[3 * static_cast<size_t>(i)];
                int imax = 0;
                if (v[1] > v[imax])
                    imax = 1;
                if (v[2] > v[imax])
                    imax = 2;
                const int a1 = (imax + 1) % 3, a2 = (imax + 2) % 3;
                const int imid = v[a1] >= v[a2] ? a1 : a2;
                const double rho = std::exp(static_cast<double>(v[imax]) - v[imid]);
                if (rho > r) {
                    const double c = lambda / n;
                    loss += c * (rho - r);
                    if (g) {
                        (*g)[3 * i + imax] += c * rho;
                        (*g)[3 * i + imid] -= c * rho;
                    }
                    if (over)
                        ++*over;
                }
            }
            return loss;
        }

        int check_scene(const Scene& s, bool& ok) {
            auto gate = [&ok](const bool pass, const std::string& what) {
                std::println("  {} {}", pass ? "ok  " : "FAIL", what);
                if (!pass)
                    ok = false;
            };
            const Run A = render(s, {true, 0.0f}), A2 = render(s, {true, 0.0f});
            const Run B = render(s, {true, kR, kLambda, true});
            const Run Z = render(s, {false, kR, 0.0f, true}); // weight 0: the count-only binding (regulariser path: bit for bit)
            const int n_drawn = s.n - 4;
            gate(A.n_visible == n_drawn && B.n_visible == n_drawn,
                 std::format("{}: {} of {} splats drawn (4 behind the camera) ({} / {})", s.name, n_drawn, s.n, A.n_visible, B.n_visible));
            const bool repeat = same_bits(all(A), all(A2)) && same_bits(A.g_pose, A2.g_pose) && same_bits(A.g_rs, A2.g_rs) &&
                                same_bits(A.g_exp, A2.g_exp);
            const double spread = std::max(max_abs_diff(all(A), all(A2)), std::max(max_abs_diff(A.g_pose, A2.g_pose), max_abs_diff(A.g_rs, A2.g_rs)));
            std::println("  {}: option off, run to run: {} (max |diff| {:.3e})", s.name, repeat ? "bit-identical" : "differs", spread);
            auto untouched = [&](const std::vector<float>& a, const std::vector<float>& b) {
                return repeat ? same_bits(a, b) : max_abs_diff(a, b) <= 4.0 * spread + 1e-30;
            };
            const char* const rule = repeat ? "bit-identical" : "within 4x the run-to-run spread";
            // the log-scale check on the regularisers-only path (no image / depth / distortion / footprint): those backward sums use
            // float atomics and do not repeat bit for bit, the regulariser path does
            const Run AR = render(s, {false, 0.0f}), BR = render(s, {false, kR, kLambda, true});
            std::vector<double> g;
            int over = 0;
            const double hl = host_aniso(s.scales, s.n, kR, kLambda, &g, &over);
            double worst = 0, gmax = 0;
            std::println("    splat   drawn   rho        over r   analytic g (max, mid)        B - A (max, mid)");
            for (int i = 0; i < s.n; ++i) {
                const auto a = block(AR, 1, i), b = block(BR, 1, i);
                double e = 0, sc = 1e-30;
                for (int j = 0; j < 3; ++j) {
                    const double dd = static_cast<double>(b[j]) - a[j];
                    e = std::max(e, std::abs(dd - g[3 * i + j]));
                    sc = std::max(sc, std::abs(static_cast<double>(a[j])) + std::abs(g[3 * i + j]));
                    gmax = std::max(gmax, std::abs(g[3 * i + j]));
                }
                worst = std::max(worst, e / sc);
                const float* v = &s.scales[3 * static_cast<size_t>(i)];
                std::array<int, 3> o = {0, 1, 2};
                std::stable_sort(o.begin(), o.end(), [v](int x, int y) { return v[x] > v[y]; });
                const double rho = std::exp(static_cast<double>(v[o[0]]) - v[o[1]]);
                std::println("    {:5d}   {:5}   {:8.3f}   {:6}   ({:+.4e}, {:+.4e})   ({:+.4e}, {:+.4e})", i, i < n_drawn ? "yes" : "no", rho,
                             rho > kR ? "yes" : "no", g[3 * i + o[0]], g[3 * i + o[1]], static_cast<double>(b[o[0]]) - a[o[0]],
                             static_cast<double>(b[o[1]]) - a[o[1]]);
            }
            gate(worst < 1e-5, std::format("{}: log-scale gradient B - A = (lambda / N) rho (e_max - e_mid) over r, 0 under (incl. the 4 undrawn; regulariser path); "
                                           "worst error {:.2e} of the block (< 1e-5)",
                                           s.name, worst));
            bool others = true;
            double img_other = 0;
            for (int i = 0; i < s.n; ++i)
                for (int k : {0, 2, 3, 4, 5, 6}) {
                    if (!same_bits(block(AR, k, i), block(BR, k, i)))
                        others = false;
                    img_other = std::max(img_other, max_abs_diff(block(A, k, i), block(B, k, i)));
                }
            gate(others, std::format("{}: means, rotation, raw opacity, sh0, shN, beta of B bit-identical to A (regulariser path)", s.name));
            std::println("  info {}: with the image terms, those blocks of B vs A: max |diff| {:.3e} (the off path's own run-to-run: {:.3e})", s.name,
                         img_other, spread);
            if (s.pose)
                gate(untouched(A.g_pose, B.g_pose), std::format("{}: pose gradient of B {} to A", s.name, rule));
            if (s.rs)
                gate(untouched(A.g_rs, B.g_rs) && untouched(A.g_exp, B.g_exp),
                     std::format("{}: rolling-shutter + exposure-scale gradients of B {} to A", s.name, rule));
            double fd_worst = 0;
            int fd_n = 0;
            for (int i = 0; i < s.n; ++i)
                for (int j = 0; j < 3; ++j) {
                    if (std::abs(g[3 * i + j]) < 1e-3 * gmax)
                        continue;
                    auto lp = s.scales, lm = s.scales;
                    lp[3 * i + j] += 1e-3f;
                    lm[3 * i + j] -= 1e-3f;
                    const double step = static_cast<double>(lp[3 * i + j]) - lm[3 * i + j];
                    const Run P = render(s, {false, kR, kLambda, true, &lp}), M = render(s, {false, kR, kLambda, true, &lm});
                    const double fd = (static_cast<double>(P.loss[0]) - M.loss[0]) / step;
                    fd_worst = std::max(fd_worst, std::abs(fd - g[3 * i + j]) / std::abs(g[3 * i + j]));
                    ++fd_n;
                }
            gate(fd_n > 0 && fd_worst < 2e-2,
                 std::format("{}: central finite differences of the device loss vs analytic: {} components, worst relative error {:.2e} (< 2e-2)",
                             s.name, fd_n, fd_worst));
            gate(std::abs(B.loss[0] - hl) <= 1e-5 * std::max(1.0, hl) && static_cast<int>(B.stats[0]) == s.n && static_cast<int>(B.stats[1]) == over,
                 std::format("{}: loss {:.7f} (host {:.7f}); counts {} splats, {} over r (host {} / {})", s.name, B.loss[0], hl, B.stats[0],
                             B.stats[1], s.n, over));
            gate(same_bits(all(AR), all(Z)) && static_cast<int>(Z.stats[1]) == over && Z.loss[0] == 0.0f,
                 std::format("{}: weight 0 (count only): every gradient {} to the option off; {} over r, loss {}", s.name, "bit-identical (regulariser path)", Z.stats[1], Z.loss[0]));
            return 0;
        }

        int run_check(const char* table_path) {
            if (table_path != nullptr) {
                lfs::training::activate_radial_lens(lfs::training::load_radial_lens_table(table_path));
                std::println("lens of the recipe scene: the radial lens table {}", lfs::training::active_radial_lens()->describe());
            } else {
                std::println("lens of the recipe scene: the woodland k-polynomial (no table given)");
            }
            std::println("anisotropy loss check: r = {}, lambda = {}; loss = lambda / N sum max(0, s_max / s_mid - r)", kR, kLambda);
            bool ok = true;
            check_scene(make_scene("pinhole", 0, nullptr), ok);
            check_scene(make_scene("fisheye", 1, nullptr), ok);
            check_scene(make_scene("recipe", 2, lfs::training::active_radial_lens()), ok);
            std::println("{}", ok ? "ANISO CHECK OK" : "ANISO CHECK FAILED");
            return ok ? 0 : 1;
        }
    } // namespace

    int run_lfe_aniso_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "check")
            return run_check(argc >= 4 ? argv[3] : nullptr);
        std::println(stderr, "usage: LichtFeld-Studio lfe-aniso check [lens_radial.txt]");
        return 2;
    }

} // namespace lfs::app
