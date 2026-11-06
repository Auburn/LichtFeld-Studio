/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Developer checks of the per-camera pose refinement (--pose-opt; training/pose_refinement.hpp, FastGSPose).
//   LichtFeld-Studio lfe-pose gradcheck   (LFE_POSE_VERBOSE=1: every component)
//     (A) the pose path at a zero offset: the device compose reproduces the camera's transform bit for bit; with the pose
//         gradient bound (the POSE backward kernels) the image is identical and the splat gradients equal the plain
//         kernels' within their own run-to-run spread (the blend backward accumulates with atomics);
//     (B) exact identities, no finite differences: moving the camera centre by dd equals moving every splat by
//         -W^T dd (SH colours included: the view direction is unchanged), and turning the camera by dr equals turning
//         every splat (centre and orientation) about the camera centre by -W^T dr (SH degree 0, where colours don't see
//         the turn); so the pose gradient must equal the same backward's splat mean / rotation gradients combined that
//         way. Per loss: image term, depth distortion (the recipe's floor loss: distance depth, plane-crossing exclusion),
//         rendered depth, all three; global shutter and the dynamic rolling shutter (capture times held, as both
//         backward paths do);
//     (C) dL/d(pose offset) (rotation r [rad], centre move d [units], start camera axes) from the backward + the exact
//         chain at a nonzero offset, against central finite differences of the full forward. Separated splats with the
//         image / depth weights faded out before each splat's alpha cut-off (the rasterizer drops alpha < 1/255; a
//         weight there adds a boundary term no analytic gradient carries); then the clustered scene ungated, all three
//         losses, beside the same finite-difference check of a global splat translation (the rasterizer's own gap).
//   LichtFeld-Studio lfe-pose gradcheck <lens_radial.txt>   (LFQPL: --pose-opt with --lens-radial)
//     The same (A), (B), (C) with every scene drawn through the radial lens table (the table's camera at quarter
//     resolution; the POSE x RADIAL kernel instantiations), so the pose gradient is checked through the table's
//     projection, Jacobian, Jacobian derivative and rolling-shutter row solve; plus (R): the table really is in use (the
//     same scene through the table's tag k-polynomial renders differently).

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include "training/kernels/pose_refinement_math.hpp"
#include "training/kernels/pose_refinement_update.hpp"
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
#include <random>
#include <rasterization_api.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace lfs::app {

    namespace {
        namespace fr = fast_lfs::rasterization;
        namespace pm = lfs::training::pose_math;
        using V6 = std::array<double, 6>;

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

        std::array<double, 9> quat_to_rotmat_d(double qw, double qx, double qy, double qz) {
            const double n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qw /= n;
            qx /= n;
            qy /= n;
            qz /= n;
            return {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
                    2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                    2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)};
        }

        struct Scene {
            int n = 0, w = 1024, h = 768;
            double fx = 292.43, fy = 296.08, cx = 512, cy = 384;
            fr::FastGSLens lens{};
            int sh_bases = 4; // 1 (degree 0) or 4 (degree 1)
            std::vector<float> means, scales, rots, opac, sh0, shN;
            std::array<double, 16> w2c0{};
            std::array<double, 3> c0{};
            V6 off{}; // the pose offset (r, d)
            std::vector<float> w_img, w_depth;
            float distortion_weight = 0.0f;
            bool rs = false;
            std::vector<float> motion; // 12
            float tau = 0.0f;
            std::vector<float> times; // held capture times (empty = solve)
        };

        struct Run {
            double loss = 0;
            std::vector<float> img, alpha, g_means, g_scales, g_rots, g_opac, g_sh0, times;
            V6 g_pose{}; // dL/d(left perturbation) from the backward
            int n_visible = 0;
        };

        // LFQPL: the radial lens table every scene is drawn through (gradcheck <lens_radial.txt>), or nullptr (the woodland
        // k-polynomial, as before)
        const lfs::training::RadialLensTable* g_radial = nullptr;

        void composed(const Scene& s, std::array<double, 16>& w2c, std::array<double, 3>& c) {
            pm::compose<double>(s.w2c0.data(), s.c0.data(), s.off.data(), w2c.data(), c.data());
        }

        Run render(const Scene& s, bool backward, bool bind_pose_grad) {
            const int N = s.n;
            const size_t HW = static_cast<size_t>(s.w) * s.h;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_shN(std::max<size_t>(s.shN.size(), 1)), d_w2c(16),
                d_cam(3);
            DevBuf d_img(3 * HW), d_alpha(HW), d_depth(HW), d_wimg(3 * HW), d_wdepth(HW);
            DevBuf d_motion(12), d_readout(1), d_times(std::max(N, 1)), d_pg(6);
            DevBuf d_mom(3 * HW), d_dloss(1), d_ddepths(std::max(N, 1)), d_dgrad(std::max(N, 1));
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_shN.upload(s.shN);
            // the refined transform, composed in double then rounded (the device compose rounds its float result)
            std::array<double, 16> w2c{};
            std::array<double, 3> c{};
            composed(s, w2c, c);
            std::vector<float> w2cf(16), cf(3);
            for (int k = 0; k < 16; ++k)
                w2cf[k] = static_cast<float>(w2c[k]);
            for (int k = 0; k < 3; ++k)
                cf[k] = static_cast<float>(c[k]);
            d_w2c.upload(w2cf);
            d_cam.upload(cf);
            d_wimg.upload(s.w_img);
            d_wdepth.upload(s.w_depth);
            fr::FastGSRollingShutter rs{};
            if (s.rs) {
                d_motion.upload(s.motion);
                d_readout.upload({s.tau});
                rs.motion = d_motion.p;
                rs.readout = d_readout.p;
                rs.times = d_times.p;
                if (!s.times.empty()) {
                    d_times.upload(s.times);
                    rs.times_fixed = 1;
                }
            }
            fr::FastGSDistortion dist{};
            if (s.distortion_weight > 0.0f) {
                dist.moments = d_mom.p;
                dist.loss_out = d_dloss.p;
                dist.depths = d_ddepths.p;
                dist.grad_depths = d_dgrad.p;
                dist.loss_scale = s.distortion_weight / static_cast<float>(HW);
                dist.grad_scale = dist.loss_scale;
                dist.near_n = 0.2f;
                dist.far_n = 1000.0f;
                dist.depth_mode = fr::kDistortionDepthDistance;
                dist.plane_sigma = 2.0f;
                dist.min_dist = 0.0f;
            }
            const float* shN = s.sh_bases > 1 ? d_shN.p : nullptr;
            Run run;
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, shN, d_w2c.p, d_cam.p, d_img.p, d_alpha.p,
                                       d_depth.p, nullptr, nullptr, nullptr, N, s.sh_bases, s.sh_bases, s.w, s.h,
                                       static_cast<float>(s.fx), static_cast<float>(s.fy), static_cast<float>(s.cx),
                                       static_cast<float>(s.cy), 0.01f, 1e10f, false, nullptr, nullptr, 0, 0, nullptr, 1.0f, s.lens,
                                       dist, 0, rs);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            run.n_visible = ctx.n_visible;
            run.img = d_img.download();
            run.alpha = d_alpha.download();
            const auto dep = d_depth.download();
            if (s.rs)
                run.times = d_times.download();
            double loss = 0;
            for (size_t i = 0; i < run.img.size(); ++i)
                loss += static_cast<double>(run.img[i]) * s.w_img[i];
            for (size_t i = 0; i < dep.size(); ++i)
                loss += static_cast<double>(dep[i]) * s.w_depth[i];
            if (s.distortion_weight > 0.0f)
                loss += d_dloss.download()[0];
            run.loss = loss;
            if (!backward) {
                fr::release_forward_context(ctx);
                cudaDeviceSynchronize();
                return run;
            }
            if (bind_pose_grad)
                ctx.pose_grad = d_pg.p;
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
            const auto res = fr::backward_raw(nullptr, nullptr, d_wimg.p, fr::BackgroundAlphaGradient{}, d_wdepth.p, nullptr, d_img.p,
                                              d_alpha.p, d_means.p, d_scales.p, d_rots.p, d_opac.p, shN, d_w2c.p, d_cam.p, ctx, nullptr,
                                              N, ctx.n_visible, s.sh_bases, s.sh_bases, s.w, s.h, static_cast<float>(s.fx),
                                              static_cast<float>(s.fy), static_cast<float>(s.cx), static_cast<float>(s.cy), false,
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
            const auto pg = d_pg.download();
            for (int k = 0; k < 6; ++k)
                run.g_pose[k] = pg[k];
            return run;
        }

        // Lines of sight over the woodland lens (quarter res), `per_line` splats each at about 0.8, 1.6 and 3 units (so
        // the distortion term has overlapping splats along the rays), a random camera; SH degree 1 colours (sh_bases 4)
        // or degree 0 (1).
        Scene make_scene(unsigned seed, int per_line, int sh_bases) {
            Scene s;
            s.sh_bases = sh_bases;
            s.lens.model = fr::kLensFisheye;
            s.lens.k1 = 0.39127139f;
            s.lens.k2 = -0.03901020f;
            s.lens.k3 = 0.00243290f;
            s.lens.k4 = -0.00266145f;
            const double tmax = std::tan(88.0 * std::numbers::pi / 180.0);
            s.lens.max_w = static_cast<float>(tmax * tmax);
            if (g_radial != nullptr) { // LFQPL: the table's camera at quarter resolution, drawn through the table
                const auto& t = *g_radial;
                s.w = t.width / 4;
                s.h = t.height / 4;
                s.fx = t.fx / 4;
                s.fy = t.fy / 4;
                s.cx = t.cx / 4;
                s.cy = t.cy / 4;
                s.lens.k1 = static_cast<float>(t.k[0]); // the tag (unused by the RADIAL kernels)
                s.lens.k2 = static_cast<float>(t.k[1]);
                s.lens.k3 = static_cast<float>(t.k[2]);
                s.lens.k4 = static_cast<float>(t.k[3]);
                s.lens.radial = t.device_view();
            }
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            const double a1 = 2 * std::numbers::pi * uni(rng), a2 = std::acos(2 * uni(rng) - 1), a3 = 2 * std::numbers::pi * uni(rng);
            const auto R = quat_to_rotmat_d(std::cos(a1 / 2), std::sin(a1 / 2) * std::sin(a2) * std::cos(a3),
                                            std::sin(a1 / 2) * std::sin(a2) * std::sin(a3), std::sin(a1 / 2) * std::cos(a2));
            const std::array<double, 3> C = {uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    s.w2c0[i * 4 + j] = R[i * 3 + j];
                s.w2c0[i * 4 + 3] = -(R[i * 3] * C[0] + R[i * 3 + 1] * C[1] + R[i * 3 + 2] * C[2]);
            }
            s.w2c0[15] = 1;
            s.c0 = C;
            const double thetas[12] = {0, 8, 15, 22, 30, 38, 45, 52, 58, 64, 70, 75};
            const double dists[3] = {0.8, 1.6, 3.0};
            for (int li = 0; li < 12; ++li) {
                const double az = 2 * std::numbers::pi * li / 12.0 + 0.4 * uni(rng);
                for (int k = 0; k < per_line; ++k) {
                    const double dist = dists[k];
                    const double th = (thetas[li] + (uni(rng) - 0.5) * 1.5) * std::numbers::pi / 180.0;
                    const double a = az + (uni(rng) - 0.5) * 0.03;
                    const double dd = dist * (0.95 + 0.1 * uni(rng));
                    const std::array<double, 3> pc = {dd * std::sin(th) * std::cos(a), dd * std::sin(th) * std::sin(a), dd * std::cos(th)};
                    for (int j = 0; j < 3; ++j) // world = R^T (pc - t)
                        s.means.push_back(static_cast<float>(s.w2c0[j] * (pc[0] - s.w2c0[3]) + s.w2c0[4 + j] * (pc[1] - s.w2c0[7]) +
                                                             s.w2c0[8 + j] * (pc[2] - s.w2c0[11])));
                    for (int j = 0; j < 3; ++j) // 2-5 degrees per axis
                        s.scales.push_back(static_cast<float>(std::log(dd * (0.035 + 0.05 * uni(rng)))));
                    for (int j = 0; j < 4; ++j)
                        s.rots.push_back(static_cast<float>(uni(rng) * 2 - 1));
                    s.opac.push_back(static_cast<float>(0.0 + 1.5 * uni(rng)));
                    for (int j = 0; j < 3; ++j)
                        s.sh0.push_back(static_cast<float>((uni(rng) - 0.5) * 1.5 + 0.8));
                }
            }
            s.n = static_cast<int>(s.opac.size());
            if (sh_bases > 1) {
                // shN: the swizzled float4 layout [ceil(N/32), slots, 32]; any values are valid coefficients
                const int slots = 3; // shSlotsForBases(4)
                s.shN.resize(static_cast<size_t>((s.n + 31) / 32) * slots * 32 * 4);
                for (auto& v : s.shN)
                    v = static_cast<float>((uni(rng) - 0.5) * 0.6);
            }
            const size_t HW = static_cast<size_t>(s.w) * s.h;
            s.w_img.assign(3 * HW, 0.0f);
            s.w_depth.assign(HW, 0.0f);
            return s;
        }

        // smooth weights over the image (times gate(alpha) when given: 0 below alpha 0.05, 1 above 0.2, smooth between, so
        // the weights vanish well before each splat's 1/255 cut-off)
        double gate(float a) {
            const double t = std::clamp((a - 0.05) / 0.15, 0.0, 1.0);
            return t * t * (3 - 2 * t);
        }
        void set_weights(Scene& s, double img_scale, double depth_scale, const std::vector<float>* alpha) {
            for (int y = 0; y < s.h; ++y)
                for (int x = 0; x < s.w; ++x) {
                    const size_t p = static_cast<size_t>(y) * s.w + x;
                    const double g = alpha ? gate((*alpha)[p]) : 1.0;
                    for (int c = 0; c < 3; ++c)
                        s.w_img[c * static_cast<size_t>(s.h) * s.w + p] = static_cast<float>(
                            g * img_scale * (std::sin(2 * std::numbers::pi * x / (37.0 + 11 * c)) * std::cos(2 * std::numbers::pi * y / (29.0 + 7 * c)) + 0.25));
                    s.w_depth[p] = static_cast<float>(g * depth_scale * (std::cos(2 * std::numbers::pi * x / 53.0) * std::sin(2 * std::numbers::pi * y / 41.0) + 0.3));
                }
        }

        double rel(const V6& a, const V6& b, int b0, int b1) {
            double nd = 0, nf = 0;
            for (int i = b0; i < b1; ++i) {
                nd += (a[i] - b[i]) * (a[i] - b[i]);
                nf += a[i] * a[i];
            }
            return std::sqrt(nd) / std::max(std::sqrt(nf), 1e-30);
        }

        V6 fd_pose(Scene s, double h) {
            V6 out{};
            const auto off0 = s.off;
            for (int k = 0; k < 6; ++k) {
                s.off = off0;
                s.off[k] += h;
                const double lp = render(s, false, false).loss;
                s.off = off0;
                s.off[k] -= h;
                const double lm = render(s, false, false).loss;
                out[k] = (lp - lm) / (2 * h);
            }
            return out;
        }

        // the pose gradient predicted from the same backward's splat gradients (left perturbation, camera axes):
        //   dd: every splat moves by -W^T dd                         -> dL/ddd = -W sum_i g_x,i
        //   dr: every splat turns about c by exp(-[w]x), w = W^T dr   -> dL/dw = sum_i g_x,i x (x_i - c) - 1/2 sum_i Q_i,
        //       Q = -g_s v + s g_v + v x g_v for its quaternion (s, v) and gradient (g_s, g_v);  dL/ddr = W dL/dw
        V6 from_splats(const Scene& s, const Run& r) {
            std::array<double, 16> W{};
            std::array<double, 3> c{};
            composed(s, W, c);
            double G[3] = {0, 0, 0}, gw[3] = {0, 0, 0};
            for (int i = 0; i < s.n; ++i) {
                const double g[3] = {r.g_means[i * 3], r.g_means[i * 3 + 1], r.g_means[i * 3 + 2]};
                const double x[3] = {s.means[i * 3] - c[0], s.means[i * 3 + 1] - c[1], s.means[i * 3 + 2] - c[2]};
                for (int j = 0; j < 3; ++j)
                    G[j] += g[j];
                gw[0] += g[1] * x[2] - g[2] * x[1];
                gw[1] += g[2] * x[0] - g[0] * x[2];
                gw[2] += g[0] * x[1] - g[1] * x[0];
                const double qs = s.rots[i * 4], v[3] = {s.rots[i * 4 + 1], s.rots[i * 4 + 2], s.rots[i * 4 + 3]};
                const double gs = r.g_rots[i * 4], gv[3] = {r.g_rots[i * 4 + 1], r.g_rots[i * 4 + 2], r.g_rots[i * 4 + 3]};
                const double vxg[3] = {v[1] * gv[2] - v[2] * gv[1], v[2] * gv[0] - v[0] * gv[2], v[0] * gv[1] - v[1] * gv[0]};
                for (int j = 0; j < 3; ++j)
                    gw[j] -= 0.5 * (-gs * v[j] + qs * gv[j] + vxg[j]);
            }
            V6 out{};
            for (int i = 0; i < 3; ++i) {
                out[i] = W[i * 4] * gw[0] + W[i * 4 + 1] * gw[1] + W[i * 4 + 2] * gw[2];
                out[3 + i] = -(W[i * 4] * G[0] + W[i * 4 + 1] * G[1] + W[i * 4 + 2] * G[2]);
            }
            return out;
        }

        // central differences of a global splat translation (camera axes, the same directions as dd): the rasterizer's own
        // finite-difference gap, against -W sum_i g_x,i
        std::array<double, 3> fd_global_translation(Scene s, double h) {
            std::array<double, 16> W{};
            std::array<double, 3> c{};
            composed(s, W, c);
            std::array<double, 3> out{};
            const auto m0 = s.means;
            for (int k = 0; k < 3; ++k) {
                double lpm[2];
                for (int sg = 0; sg < 2; ++sg) {
                    s.means = m0;
                    const double step = (sg == 0 ? -h : h); // the splats move by -W^T e_k step (as the camera by +step)
                    for (int i = 0; i < s.n; ++i)
                        for (int j = 0; j < 3; ++j)
                            s.means[i * 3 + j] = static_cast<float>(m0[i * 3 + j] - W[k * 4 + j] * step);
                    lpm[sg] = render(s, false, false).loss;
                }
                out[k] = (lpm[1] - lpm[0]) / (2 * h);
            }
            return out;
        }

        void set_rs(Scene& s) {
            s.rs = true;
            // the RS tool's motion: omega (1.5,-2,0.8) rad/s, alpha (40,-25,30), vel (1.2,-0.8,2.5) u/s, acc (8,-5,6); 20 ms
            s.motion = {1.5f, -2.0f, 0.8f, 40.0f, -25.0f, 30.0f, 1.2f, -0.8f, 2.5f, 8.0f, -5.0f, 6.0f};
            s.tau = 0.020f;
        }

        int run_gradcheck() {
            int bad = 0;
            const bool verbose = std::getenv("LFE_POSE_VERBOSE") != nullptr;
            const V6 off = {0.010, -0.007, 0.012, 0.012, -0.008, 0.010};
            if (g_radial != nullptr) {
                // ---------------------------------------------------------------- (R) LFQPL: the table is what is drawn
                std::println("lens: the radial lens table {}", g_radial->describe());
                Scene s = make_scene(11, 3, 4);
                set_weights(s, 1e-3, 1e-4, nullptr);
                s.off = off;
                Scene p = s;
                p.lens.radial = {}; // the same camera through its tag k-polynomial
                double worst = 0;
                for (bool rs : {false, true}) {
                    Scene a = s, b = p;
                    if (rs) {
                        set_rs(a);
                        set_rs(b);
                    }
                    const auto ra = render(a, true, true), rb = render(b, true, true);
                    double md = 0, mx = 0;
                    for (size_t i = 0; i < ra.img.size(); ++i) {
                        md = std::max(md, std::abs(static_cast<double>(ra.img[i]) - rb.img[i]));
                        mx = std::max(mx, static_cast<double>(std::abs(ra.img[i])));
                    }
                    std::println("(R) {}: table vs its tag polynomial, same scene: image max |diff| {:.3e} (of max {:.3f}); "
                                 "loss {:.9g} vs {:.9g}; |dL/ddr| {:.4e} vs {:.4e}",
                                 rs ? "rolling shutter" : "global shutter ", md, mx, ra.loss, rb.loss,
                                 std::hypot(ra.g_pose[0], ra.g_pose[1], ra.g_pose[2]), std::hypot(rb.g_pose[0], rb.g_pose[1], rb.g_pose[2]));
                    worst = std::max(worst, md);
                    if (!(md > 1e-4)) // the table must change the render (it is up to ~12 px from the tag lens at 75 deg)
                        ++bad;
                }
                std::println("    the table is {}", worst > 1e-4 ? "in use" : "NOT IN USE (the renders match the polynomial)");
            } else {
                std::println("lens: the woodland quarter-res k-polynomial (no radial lens table)");
            }
            // ---------------------------------------------------------------- (A)
            {
                Scene s = make_scene(11, 3, 4);
                set_weights(s, 1e-3, 0, nullptr);
                {
                    std::vector<float> w0(16), c0(3), z(6, 0.0f);
                    for (int k = 0; k < 16; ++k)
                        w0[k] = static_cast<float>(s.w2c0[k]);
                    for (int k = 0; k < 3; ++k)
                        c0[k] = static_cast<float>(s.c0[k]);
                    DevBuf d_off(6), d_w0(16), d_c0(3), d_w(16), d_c(3);
                    d_off.upload(z);
                    d_w0.upload(w0);
                    d_c0.upload(c0);
                    lfs::training::kernels::pose_compose(d_off.p, d_w0.p, d_c0.p, 0, d_w.p, d_c.p, nullptr);
                    cudaDeviceSynchronize();
                    const auto w = d_w.download(), c = d_c.download();
                    const bool same = std::memcmp(w.data(), w0.data(), 64) == 0 && std::memcmp(c.data(), c0.data(), 12) == 0;
                    std::println("(A) device compose at a zero offset reproduces the camera's transform and centre bit for bit: {}",
                                 same ? "yes" : "NO");
                    if (!same)
                        ++bad;
                }
                for (bool rs : {false, true}) {
                    Scene t = s;
                    if (rs)
                        set_rs(t);
                    const auto a = render(t, true, false), a2 = render(t, true, false), b = render(t, true, true);
                    auto maxrel = [](const std::vector<float>& x, const std::vector<float>& y, size_t& ndiff) {
                        double mx = 0, md = 0;
                        ndiff = 0;
                        for (float v : x)
                            mx = std::max(mx, static_cast<double>(std::abs(v)));
                        for (size_t i = 0; i < x.size(); ++i) {
                            ndiff += std::memcmp(&x[i], &y[i], sizeof(float)) != 0;
                            md = std::max(md, std::abs(static_cast<double>(x[i]) - y[i]));
                        }
                        return md / std::max(mx, 1e-30);
                    };
                    size_t ni = 0, n1 = 0, n2 = 0;
                    maxrel(a.img, b.img, ni);
                    double rep = 0, pose = 0;
                    for (auto [x, y, z] : {std::tuple{&a.g_means, &a2.g_means, &b.g_means}, std::tuple{&a.g_scales, &a2.g_scales, &b.g_scales},
                                           std::tuple{&a.g_rots, &a2.g_rots, &b.g_rots}, std::tuple{&a.g_opac, &a2.g_opac, &b.g_opac},
                                           std::tuple{&a.g_sh0, &a2.g_sh0, &b.g_sh0}}) {
                        rep = std::max(rep, maxrel(*x, *y, n1));
                        pose = std::max(pose, maxrel(*x, *z, n2));
                    }
                    std::println("    {}: {} splats ({} visible): image values differing with the pose gradient bound: {}; splat "
                                 "gradients, max |diff| / max |grad|: plain vs plain (repeat) {:.1e}, plain vs pose-bound {:.1e}",
                                 rs ? "rolling shutter" : "global shutter ", t.n, a.n_visible, ni, rep, pose);
                    if (ni != 0 || pose > std::max(rep * 4.0, 2e-6))
                        ++bad;
                }
            }
            // ---------------------------------------------------------------- (B)
            struct Case {
                const char* name;
                double img, depth, dist;
            };
            const Case cases[] = {
                {"image term", 1e-3, 0, 0},
                {"depth distortion only", 0, 0, 2000},
                {"rendered depth only", 0, 1e-4, 0},
                {"all three", 1e-3, 1e-4, 2000},
            };
            std::println("(B) pose gradient vs the same backward's splat gradients moved/turned with the camera (exact identity; "
                         "offset r = (0.010, -0.007, 0.012) rad, d = (0.012, -0.008, 0.010) units; relative error per block)");
            std::println("  {:<24} {:<16} {:>7} | {:>11} {:>11} | {:>9} {:>9} | {:>9}", "loss", "shutter", "visible", "|dL/ddr|", "|dL/ddd|",
                         "turn SH0", "move SH0", "move SH1");
            double worst_id = 0;
            for (const auto& cs : cases)
                for (bool rs : {false, true}) {
                    double e[3] = {0, 0, 0};
                    V6 an{};
                    int vis = 0;
                    for (int sh : {1, 4}) {
                        Scene s = make_scene(23, 3, sh);
                        set_weights(s, cs.img, cs.depth, nullptr);
                        s.distortion_weight = static_cast<float>(cs.dist);
                        s.off = off;
                        if (rs)
                            set_rs(s);
                        Run r = render(s, true, true);
                        if (rs) { // hold the capture times (both backward paths hold them)
                            s.times = r.times;
                            r = render(s, true, true);
                        }
                        const auto pred = from_splats(s, r);
                        if (sh == 1) {
                            e[0] = rel(r.g_pose, pred, 0, 3);
                            e[1] = rel(r.g_pose, pred, 3, 6);
                            an = r.g_pose;
                            vis = r.n_visible;
                        } else {
                            e[2] = rel(r.g_pose, pred, 3, 6);
                        }
                        if (verbose)
                            for (int k = 0; k < 6; ++k)
                                std::println("      SH{} [{}] pose {:14.7g}  from splats {:14.7g}", sh == 1 ? 0 : 1, k, r.g_pose[k], pred[k]);
                    }
                    std::println("  {:<24} {:<16} {:>7} | {:11.4e} {:11.4e} | {:9.2e} {:9.2e} | {:9.2e}", cs.name, rs ? "rolling (held)" : "global",
                                 vis, std::hypot(an[0], an[1], an[2]), std::hypot(an[3], an[4], an[5]), e[0], e[1], e[2]);
                    worst_id = std::max({worst_id, e[0], e[1], e[2]});
                }
            std::println("  worst {:.2e} (pass < 1e-4: float rounding of the sums)", worst_id);
            if (worst_id > 1e-4)
                ++bad;
            // ---------------------------------------------------------------- (C)
            std::println("(C) dL/d(pose offset) (backward + exact chain) vs central differences of the full forward, same offset; "
                         "relative error per block (h, 2.5 h: two finite-difference steps)");
            std::println("  {:<44} {:<16} | {:>9} {:>9} | {:>9} {:>9}", "scene / loss", "shutter", "turn h", "move h", "turn 2.5h", "move 2.5h");
            double worst_fd = 0;
            for (bool rs : {false, true}) {
                // separated splats (nearest of each line of sight), image + depth weights gated by the base alpha
                Scene s = make_scene(31, 1, 4);
                s.off = off;
                if (rs)
                    set_rs(s);
                const auto a0 = render(s, false, false);
                set_weights(s, 1e-3, 1e-4, &a0.alpha);
                Run r = render(s, true, true);
                if (rs) {
                    s.times = r.times;
                    r = render(s, true, true);
                }
                V6 an{};
                pm::chain<double>(s.off.data(), r.g_pose.data(), an.data());
                const auto f1 = fd_pose(s, 2e-4), f2 = fd_pose(s, 5e-4);
                const double e[4] = {rel(f1, an, 0, 3), rel(f1, an, 3, 6), rel(f2, an, 0, 3), rel(f2, an, 3, 6)};
                std::println("  {:<44} {:<16} | {:9.2e} {:9.2e} | {:9.2e} {:9.2e}", "12 separated splats, image + depth, gated",
                             rs ? "rolling (held)" : "global", e[0], e[1], e[2], e[3]);
                if (verbose)
                    for (int k = 0; k < 6; ++k)
                        std::println("      [{}] fd {:14.7g}  fd(2.5h) {:14.7g}  analytic {:14.7g}", k, f1[k], f2[k], an[k]);
                worst_fd = std::max({worst_fd, std::min(e[0], e[2]), std::min(e[1], e[3])});
            }
            for (bool rs : {false, true}) {
                // clustered splats, all three losses, ungated; with the global-translation finite differences beside
                Scene s = make_scene(23, 3, 4);
                set_weights(s, 1e-3, 1e-4, nullptr);
                s.distortion_weight = 2000.0f;
                s.off = off;
                if (rs)
                    set_rs(s);
                Run r = render(s, true, true);
                if (rs) {
                    s.times = r.times;
                    r = render(s, true, true);
                }
                V6 an{};
                pm::chain<double>(s.off.data(), r.g_pose.data(), an.data());
                const auto f1 = fd_pose(s, 2e-4), f2 = fd_pose(s, 5e-4);
                // the same gap for a global splat translation, at the offset (left perturbation directions)
                const auto ft = fd_global_translation(s, 2e-4);
                const auto pred = from_splats(s, r);
                const V6 ftv = {0, 0, 0, ft[0], ft[1], ft[2]};
                std::println("  {:<44} {:<16} | {:9.2e} {:9.2e} | {:9.2e} {:9.2e}   (global splat translation vs its analytic "
                             "gradient: {:.2e})",
                             "36 clustered splats, all three, ungated", rs ? "rolling (held)" : "global", rel(f1, an, 0, 3), rel(f1, an, 3, 6),
                             rel(f2, an, 0, 3), rel(f2, an, 3, 6), rel(ftv, pred, 3, 6));
            }
            std::println("  worst gated {:.2e} (pass < 1e-2)", worst_fd);
            if (worst_fd > 1e-2)
                ++bad;
            const char* const what = g_radial != nullptr ? "POSE GRADCHECK (radial lens table)" : "POSE GRADCHECK";
            if (bad == 0)
                std::println("{} OK", what);
            else
                std::println("{}: {} failed checks", what, bad);
            return bad == 0 ? 0 : 1;
        }
    } // namespace

    int run_lfe_pose_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "gradcheck") {
            if (argc >= 4) { // LFQPL: through a radial lens table (as lfe-fisheye radialcheck loads it)
                try {
                    lfs::training::activate_radial_lens(lfs::training::load_radial_lens_table(argv[3]));
                } catch (const std::exception& e) {
                    std::println(stderr, "{}", e.what());
                    return 1;
                }
                g_radial = lfs::training::active_radial_lens();
            }
            return run_gradcheck();
        }
        std::println(stderr, "usage: LichtFeld-Studio lfe-pose gradcheck [<lens_radial.txt>]");
        return 2;
    }

} // namespace lfs::app
