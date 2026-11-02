/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Developer checks for the dynamic rolling shutter of the standard rasterizer (FastGSRollingShutter, native fisheye).
//   LichtFeld-Studio lfe-rs render <model.ply> <cameras.txt> <out_dir>
//     cameras.txt, one camera per line (lfe-fisheye render's columns, then optionally the rolling shutter):
//       name width height model fx fy cx cy k1 k2 k3 k4 qw qx qy qz tx ty tz [tau_ms wx wy wz ax ay az vx vy vz Ax Ay Az]
//     model FISHEYE / OPENCV_FISHEYE (or PINHOLE, global shutter only); without the rolling-shutter columns the render
//     is global shutter (the unchanged kernels). Background LFE_RS_BG="r,g,b" (0-255, default black). Writes
//     <out_dir>/<name>.f32 (float32 RGB, H x W x 3) through fast_rasterize_forward (the training path).
//   LichtFeld-Studio lfe-rs gradcheck
//     (A) zero motion: image and every gradient bit-identical to the global-shutter kernels;
//     (B) forward: each splat's capture time and projected centre (rendered blob centroid) against a double reference;
//     (C) dL/d(omega, alpha, vel, acc, tau) and dL/d(means, scales, rotations, opacities) against central finite
//         differences with each splat's capture time held (as the analytic backward does);
//     (D) the same with the capture time free (the full forward): the gap is the time's own dependence, O(image speed
//         x tau / H), reported with that ratio.

#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/loader.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <numbers>
#include <print>
#include <random>
#include <rasterization_api.h>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::app {

    namespace {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        namespace fr = fast_lfs::rasterization;
        using V3 = std::array<double, 3>;

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
            void upload(const std::vector<float>& v) { cudaMemcpy(p, v.data(), sizeof(float) * v.size(), cudaMemcpyHostToDevice); }
            std::vector<float> download() const {
                std::vector<float> v(n);
                cudaMemcpy(v.data(), p, sizeof(float) * n, cudaMemcpyDeviceToHost);
                return v;
            }
        };

        // ---------------------------------------------------------------- render
        int run_render(const std::filesystem::path& ply, const std::filesystem::path& spec, const std::filesystem::path& out_dir) {
            auto loader = lfs::io::Loader::create();
            auto load_result = loader->load(ply);
            if (!load_result) {
                std::println(stderr, "load failed: {}", load_result.error().message);
                return 1;
            }
            auto* const splat = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&load_result->data);
            if (!splat || !*splat || (*splat)->size() == 0) {
                std::println(stderr, "not a splat scene");
                return 1;
            }
            auto& model = **splat;
            std::println("{} splats, SH degree {}", model.size(), model.get_active_sh_degree());
            std::filesystem::create_directories(out_dir);
            std::vector<float> bgv = {0.0f, 0.0f, 0.0f};
            if (const char* e = std::getenv("LFE_RS_BG"); e && *e) {
                std::istringstream bs(e);
                std::string tok;
                for (int c = 0; c < 3 && std::getline(bs, tok, ','); ++c)
                    bgv[c] = static_cast<float>(std::atof(tok.c_str()) / 255.0);
            }
            auto background = Tensor::from_vector(bgv, {3}, Device::CPU).to(Device::CUDA);
            DevBuf motion(12), readout(1);

            std::ifstream in(spec);
            std::string line;
            int uid = 0;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#')
                    continue;
                std::istringstream ss(line);
                std::string name, model_name;
                int w = 0, h = 0;
                double fx, fy, cx, cy, k1, k2, k3, k4, qw, qx, qy, qz, tx, ty, tz;
                ss >> name >> w >> h >> model_name >> fx >> fy >> cx >> cy >> k1 >> k2 >> k3 >> k4 >> qw >> qx >> qy >> qz >> tx >> ty >> tz;
                if (!ss) {
                    std::println(stderr, "bad line: {}", line);
                    return 1;
                }
                std::vector<double> rs_cols;
                double v;
                while (ss >> v)
                    rs_cols.push_back(v);
                if (!rs_cols.empty() && rs_cols.size() != 13) {
                    std::println(stderr, "rolling-shutter columns: 13 expected (tau_ms + 12 motion), got {}: {}", rs_cols.size(), line);
                    return 1;
                }
                const bool fisheye = model_name == "FISHEYE" || model_name == "OPENCV_FISHEYE";
                const auto r = quat_to_rotmat_d(qw, qx, qy, qz);
                std::vector<float> rf(9);
                for (int i = 0; i < 9; ++i)
                    rf[i] = static_cast<float>(r[i]);
                auto R = Tensor::from_vector(rf, {3, 3}, Device::CPU);
                auto T = Tensor::from_vector(std::vector<float>{static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz)}, {3}, Device::CPU);
                auto radial = fisheye ? Tensor::from_vector(std::vector<float>{static_cast<float>(k1), static_cast<float>(k2), static_cast<float>(k3), static_cast<float>(k4)}, {4}, Device::CPU)
                                      : Tensor::empty({0}, Device::CPU);
                Camera camera(R, T, static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx), static_cast<float>(cy),
                              radial, Tensor::empty({0}, Device::CPU),
                              fisheye ? CameraModelType::FISHEYE : CameraModelType::PINHOLE,
                              name, {}, {}, w, h, uid++);
                fr::FastGSRollingShutter rs{};
                if (!rs_cols.empty()) {
                    if (!fisheye) {
                        std::println(stderr, "{}: the rolling shutter needs a fisheye camera", name);
                        return 1;
                    }
                    std::vector<float> m(12);
                    for (int i = 0; i < 12; ++i)
                        m[i] = static_cast<float>(rs_cols[1 + i]);
                    motion.upload(m);
                    readout.upload({static_cast<float>(rs_cols[0] / 1000.0)});
                    rs.motion = motion.p;
                    rs.readout = readout.p;
                    if (const char* it = std::getenv("LFE_RS_ITERS"); it && *it)
                        rs.iters = std::max(1, std::atoi(it));
                }
                auto result = lfs::training::fast_rasterize_forward(camera, model, background, 0, 0, 0, 0, false, {}, false, true,
                                                                    1.0f, false, {}, 0, rs);
                if (!result) {
                    std::println(stderr, "{}: render failed: {}", name, result.error().detail());
                    return 1;
                }
                auto out = std::move(result->first);
                result->second.release_forward_context();
                auto hwc = out.image.permute({1, 2, 0}).contiguous().cpu();
                const auto path = out_dir / (name + ".f32");
                std::ofstream f(path, std::ios::binary);
                f.write(reinterpret_cast<const char*>(hwc.ptr<float>()), static_cast<std::streamsize>(sizeof(float) * 3ull * w * h));
                std::println("{} {}x{} {}{} -> {}", name, w, h, fisheye ? "fisheye" : "pinhole",
                             rs_cols.empty() ? "" : std::format(", rolling shutter {} ms", rs_cols[0]), path.string());
            }
            return 0;
        }

        // ---------------------------------------------------------------- double reference
        struct RefLens {
            double fx, fy, cx, cy, k[4];
        };

        double theta_d(const RefLens& L, double t) {
            const double t2 = t * t;
            return t * (1 + t2 * (L.k[0] + t2 * (L.k[1] + t2 * (L.k[2] + t2 * L.k[3]))));
        }

        std::array<double, 2> ref_project(const RefLens& L, const V3& p) {
            const double rho = std::hypot(p[0], p[1]);
            if (rho < 1e-300)
                return {L.cx, L.cy};
            const double th = std::atan2(rho, p[2]);
            const double s = theta_d(L, th) / rho;
            return {L.fx * s * p[0] + L.cx, L.fy * s * p[1] + L.cy};
        }

        struct RefMotion {
            V3 w{}, al{}, v{}, a{};
            double tau = 0;
        };

        // E = exp([phi]x) (3x3 row-major), double
        std::array<double, 9> ref_exp(const V3& phi) {
            const double th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
            const double th = std::sqrt(th2);
            const double A = th2 < 1e-12 ? 1 - th2 / 6 : std::sin(th) / th;
            const double B = th2 < 1e-12 ? 0.5 - th2 / 24 : (1 - std::cos(th)) / th2;
            const double x = phi[0], y = phi[1], z = phi[2];
            return {1 + B * (x * x - th2), B * x * y - A * z, B * x * z + A * y,
                    B * x * y + A * z, 1 + B * (y * y - th2), B * y * z - A * x,
                    B * x * z - A * y, B * y * z + A * x, 1 + B * (z * z - th2)};
        }

        // world-to-camera rows (3x4 row-major) of the pose at time t
        std::array<double, 12> ref_rows_at(const std::array<double, 12>& mid, const RefMotion& m, double t) {
            const double h = 0.5 * t * t;
            V3 phi, d;
            for (int i = 0; i < 3; ++i) {
                phi[i] = -(m.w[i] * t + m.al[i] * h);
                d[i] = m.v[i] * t + m.a[i] * h;
            }
            const auto E = ref_exp(phi);
            std::array<double, 12> out{};
            for (int i = 0; i < 3; ++i) {
                for (int c = 0; c < 3; ++c)
                    out[i * 4 + c] = E[i * 3] * mid[c] + E[i * 3 + 1] * mid[4 + c] + E[i * 3 + 2] * mid[8 + c];
                out[i * 4 + 3] = E[i * 3] * (mid[3] - d[0]) + E[i * 3 + 1] * (mid[7] - d[1]) + E[i * 3 + 2] * (mid[11] - d[2]);
            }
            return out;
        }

        V3 apply_rows(const std::array<double, 12>& rows, const V3& x) {
            V3 p;
            for (int i = 0; i < 3; ++i)
                p[i] = rows[i * 4] * x[0] + rows[i * 4 + 1] * x[1] + rows[i * 4 + 2] * x[2] + rows[i * 4 + 3];
            return p;
        }

        // the splat's row fraction s (fixed point, `iters` re-projections) and its rows
        double ref_solve(const RefLens& L, double H, const std::array<double, 12>& mid, const RefMotion& m, const V3& x, int iters,
                         std::array<double, 12>& rows) {
            double s = 0;
            V3 p = apply_rows(mid, x);
            for (int k = 0; k < iters; ++k) {
                if (!(p[2] > 0))
                    break;
                const auto uv = ref_project(L, p);
                s = std::clamp(uv[1] / H - 0.5, -0.5, 0.5);
                if (k + 1 < iters)
                    p = apply_rows(ref_rows_at(mid, m, s * m.tau), x);
            }
            rows = ref_rows_at(mid, m, s * m.tau);
            return s;
        }

        // ---------------------------------------------------------------- scenes
        struct Scene {
            int n = 0, w = 0, h = 0;
            RefLens L{};
            fr::FastGSLens lens{};
            std::vector<float> means, scales, rots, opac, sh0, w2c, campos, weights;
            RefMotion motion{};
            bool rs = false;
            std::vector<float> times; // fixed row fractions (empty = solve)
        };

        struct GradRun {
            double loss = 0;
            std::vector<float> img, g_means, g_scales, g_rots, g_opac, g_sh0, g_rs, times;
            int n_visible = 0;
        };

        GradRun render_and_grad(const Scene& s, bool with_backward) {
            const int N = s.n;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_w2c(16), d_cam(3);
            DevBuf d_img(3ull * s.w * s.h), d_alpha(1ull * s.w * s.h), d_depth(1ull * s.w * s.h), d_weights(3ull * s.w * s.h);
            DevBuf d_motion(12), d_readout(1), d_grad(13), d_times(std::max(N, 1));
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_w2c.upload(s.w2c);
            d_cam.upload(s.campos);
            d_weights.upload(s.weights);
            fr::FastGSRollingShutter rs{};
            if (s.rs) {
                std::vector<float> m(12);
                for (int i = 0; i < 3; ++i) {
                    m[i] = static_cast<float>(s.motion.w[i]);
                    m[3 + i] = static_cast<float>(s.motion.al[i]);
                    m[6 + i] = static_cast<float>(s.motion.v[i]);
                    m[9 + i] = static_cast<float>(s.motion.a[i]);
                }
                d_motion.upload(m);
                d_readout.upload({static_cast<float>(s.motion.tau)});
                rs.motion = d_motion.p;
                rs.readout = d_readout.p;
                rs.grad = d_grad.p;
                rs.times = d_times.p;
                if (!s.times.empty()) {
                    d_times.upload(s.times);
                    rs.times_fixed = 1;
                }
            }
            GradRun run;
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, nullptr, d_w2c.p, d_cam.p,
                                       d_img.p, d_alpha.p, d_depth.p, nullptr, nullptr, nullptr, N, 1, 1, s.w, s.h,
                                       static_cast<float>(s.L.fx), static_cast<float>(s.L.fy), static_cast<float>(s.L.cx),
                                       static_cast<float>(s.L.cy), 0.01f, 1e10f, false, nullptr, nullptr, 0, 0, nullptr, 1.0f,
                                       s.lens, {}, 0, rs);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            run.n_visible = ctx.n_visible;
            run.img = d_img.download();
            if (s.rs)
                run.times = d_times.download();
            double loss = 0;
            for (size_t i = 0; i < run.img.size(); ++i)
                loss += static_cast<double>(run.img[i]) * s.weights[i];
            run.loss = loss;
            if (!with_backward) {
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
            const auto res = fr::backward_raw(nullptr, nullptr, d_weights.p, fr::BackgroundAlphaGradient{}, nullptr, nullptr,
                                              d_img.p, d_alpha.p, d_means.p, d_scales.p, d_rots.p, d_opac.p, nullptr,
                                              d_w2c.p, d_cam.p, ctx, nullptr, N, ctx.n_visible, 1, 1, s.w, s.h,
                                              static_cast<float>(s.L.fx), static_cast<float>(s.L.fy), static_cast<float>(s.L.cx),
                                              static_cast<float>(s.L.cy), false, DensificationType::None, &fa);
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
            if (s.rs)
                run.g_rs = d_grad.download();
            return run;
        }

        std::array<double, 12> mid_rows(const Scene& s) {
            std::array<double, 12> r{};
            for (int i = 0; i < 12; ++i)
                r[i] = s.w2c[i];
            return r;
        }

        // Splats spread over the woodland lens (quarter res), at the given off-axis angles and distances, a random camera.
        // local: dL/dimage is a smooth bump inside each splat's 2.5-sigma ellipse at its rolling-shutter pose (double
        // reference), else zero; otherwise smooth weights over the whole image.
        Scene make_scene(const std::vector<double>& thetas_deg, const std::vector<double>& dists, unsigned seed, bool local,
                         const RefMotion& motion, bool rs, int only = -1) {
            Scene s;
            s.w = 1024;
            s.h = 768;
            s.L = {292.43, 296.08, 512, 384, {0.39127139, -0.03901020, 0.00243290, -0.00266145}};
            s.lens.model = fr::kLensFisheye;
            s.lens.k1 = static_cast<float>(s.L.k[0]);
            s.lens.k2 = static_cast<float>(s.L.k[1]);
            s.lens.k3 = static_cast<float>(s.L.k[2]);
            s.lens.k4 = static_cast<float>(s.L.k[3]);
            const double tmax = std::tan(88.0 * std::numbers::pi / 180.0);
            s.lens.max_w = static_cast<float>(tmax * tmax);
            s.motion = motion;
            s.rs = rs;
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            const double a1 = 2 * std::numbers::pi * uni(rng), a2 = std::acos(2 * uni(rng) - 1), a3 = 2 * std::numbers::pi * uni(rng);
            const auto R = quat_to_rotmat_d(std::cos(a1 / 2), std::sin(a1 / 2) * std::sin(a2) * std::cos(a3),
                                            std::sin(a1 / 2) * std::sin(a2) * std::sin(a3), std::sin(a1 / 2) * std::cos(a2));
            const V3 C = {uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1};
            V3 t{};
            for (int i = 0; i < 3; ++i)
                t[i] = -(R[i * 3] * C[0] + R[i * 3 + 1] * C[1] + R[i * 3 + 2] * C[2]);
            s.w2c = {static_cast<float>(R[0]), static_cast<float>(R[1]), static_cast<float>(R[2]), static_cast<float>(t[0]),
                     static_cast<float>(R[3]), static_cast<float>(R[4]), static_cast<float>(R[5]), static_cast<float>(t[1]),
                     static_cast<float>(R[6]), static_cast<float>(R[7]), static_cast<float>(R[8]), static_cast<float>(t[2]), 0, 0, 0, 1};
            s.campos = {static_cast<float>(C[0]), static_cast<float>(C[1]), static_cast<float>(C[2])};
            const auto mid = mid_rows(s);
            for (size_t i = 0; i < thetas_deg.size(); ++i) {
                const double th = thetas_deg[i] * std::numbers::pi / 180.0;
                const double dist = dists[i % dists.size()];
                double az = 2 * std::numbers::pi * uni(rng);
                V3 pc{};
                for (int tries = 0; tries < 4000; ++tries) {
                    pc = {dist * std::sin(th) * std::cos(az), dist * std::sin(th) * std::sin(az), dist * std::cos(th)};
                    const auto m = ref_project(s.L, pc);
                    if (m[0] > 90 && m[0] < s.w - 90 && m[1] > 90 && m[1] < s.h - 90)
                        break;
                    az = 2 * std::numbers::pi * uni(rng);
                }
                V3 pw{};
                for (int j = 0; j < 3; ++j)
                    pw[j] = mid[0 * 4 + j] * (pc[0] - mid[3]) + mid[1 * 4 + j] * (pc[1] - mid[7]) + mid[2 * 4 + j] * (pc[2] - mid[11]);
                for (double v : pw)
                    s.means.push_back(static_cast<float>(v));
                for (int j = 0; j < 3; ++j) // 1.5-3 degrees per axis
                    s.scales.push_back(static_cast<float>(std::log(dist * (0.026 + 0.026 * uni(rng)))));
                for (int j = 0; j < 4; ++j)
                    s.rots.push_back(static_cast<float>(uni(rng) * 2 - 1));
                s.opac.push_back(static_cast<float>(-0.5 + 1.5 * uni(rng)));
                for (int j = 0; j < 3; ++j)
                    s.sh0.push_back(static_cast<float>((uni(rng) - 0.5) * 1.5 + 0.8));
            }
            s.n = static_cast<int>(thetas_deg.size());
            if (only >= 0) {
                auto keep = [only](std::vector<float>& v, int k) { v = std::vector<float>(v.begin() + only * k, v.begin() + (only + 1) * k); };
                keep(s.means, 3);
                keep(s.scales, 3);
                keep(s.rots, 4);
                keep(s.opac, 1);
                keep(s.sh0, 3);
                s.n = 1;
            }
            s.weights.assign(3ull * s.w * s.h, 0.0f);
            if (!local) {
                for (int c = 0; c < 3; ++c)
                    for (int y = 0; y < s.h; ++y)
                        for (int x = 0; x < s.w; ++x)
                            s.weights[(static_cast<size_t>(c) * s.h + y) * s.w + x] = static_cast<float>(
                                1e-3 * (std::sin(2 * std::numbers::pi * x / (37.0 + 11 * c)) * std::cos(2 * std::numbers::pi * y / (29.0 + 7 * c)) + 0.25));
                return s;
            }
            for (int i = 0; i < s.n; ++i) {
                const V3 xw = {s.means[i * 3], s.means[i * 3 + 1], s.means[i * 3 + 2]};
                std::array<double, 12> rows = mid;
                if (rs)
                    ref_solve(s.L, s.h, mid, s.motion, xw, 2, rows);
                const V3 pc = apply_rows(rows, xw);
                const auto m = ref_project(s.L, pc);
                // footprint: numerical Jacobian of the reference projection, covariance through it
                const auto Rq = quat_to_rotmat_d(s.rots[i * 4], s.rots[i * 4 + 1], s.rots[i * 4 + 2], s.rots[i * 4 + 3]);
                double cov[3][3] = {};
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        for (int k = 0; k < 3; ++k)
                            cov[a][b] += Rq[a * 3 + k] * std::exp(2.0 * s.scales[i * 3 + k]) * Rq[b * 3 + k];
                double J[2][3];
                for (int k = 0; k < 3; ++k) {
                    V3 pp = pc, pm = pc;
                    const double hstep = 1e-6 * std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
                    pp[k] += hstep;
                    pm[k] -= hstep;
                    const auto a = ref_project(s.L, pp), b = ref_project(s.L, pm);
                    J[0][k] = (a[0] - b[0]) / (2 * hstep);
                    J[1][k] = (a[1] - b[1]) / (2 * hstep);
                }
                double JW[2][3] = {};
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 3; ++c)
                        for (int k = 0; k < 3; ++k)
                            JW[r][c] += J[r][k] * rows[k * 4 + c];
                double Cm[2][2] = {};
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 2; ++c)
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                Cm[r][c] += JW[r][a] * cov[a][b] * JW[c][b];
                Cm[0][0] += 0.3;
                Cm[1][1] += 0.3;
                const double det = Cm[0][0] * Cm[1][1] - Cm[0][1] * Cm[1][0];
                const double ia = Cm[1][1] / det, ib = -Cm[0][1] / det, ic = Cm[0][0] / det;
                const double sx = std::sqrt(Cm[0][0]), sy = std::sqrt(Cm[1][1]);
                const int x0 = std::max(0, static_cast<int>(m[0] - 3 * sx)), x1 = std::min(s.w - 1, static_cast<int>(m[0] + 3 * sx) + 1);
                const int y0 = std::max(0, static_cast<int>(m[1] - 3 * sy)), y1 = std::min(s.h - 1, static_cast<int>(m[1] + 3 * sy) + 1);
                for (int y = y0; y <= y1; ++y)
                    for (int x = x0; x <= x1; ++x) {
                        const double dx = x + 0.5 - m[0], dy = y + 0.5 - m[1];
                        const double d2 = ia * dx * dx + 2 * ib * dx * dy + ic * dy * dy;
                        if (d2 >= 6.25)
                            continue;
                        const double bump = (1 - d2 / 6.25) * (1 - d2 / 6.25);
                        for (int c = 0; c < 3; ++c)
                            s.weights[(static_cast<size_t>(c) * s.h + y) * s.w + x] += static_cast<float>(
                                1e-3 * bump * (1.0 + 0.6 * dx / sx * (c == 1 ? -1.0 : 1.0) - 0.4 * dy / sy + 0.3 * c));
                    }
            }
            return s;
        }

        double rel(const std::vector<double>& fd, const std::vector<double>& an) {
            double nd = 0, nf = 0;
            for (size_t i = 0; i < fd.size(); ++i) {
                nd += (fd[i] - an[i]) * (fd[i] - an[i]);
                nf += fd[i] * fd[i];
            }
            return std::sqrt(nd) / std::max(std::sqrt(nf), 1e-30);
        }

        // central differences of the loss in one scalar (setter writes v into the scene)
        template <class F>
        double fd(Scene s, F set, double v0, double step) {
            set(s, v0 + step);
            const double lp = render_and_grad(s, false).loss;
            set(s, v0 - step);
            const double lm = render_and_grad(s, false).loss;
            return (lp - lm) / (2 * step);
        }

        // ---------------------------------------------------------------- the check
        int run_gradcheck() {
            int bad = 0;
            const RefMotion mo = {{1.5, -2.0, 0.8}, {40.0, -25.0, 30.0}, {1.2, -0.8, 2.5}, {8.0, -5.0, 6.0}, 0.020};
            RefMotion zero{};
            zero.tau = 0.020;
            const std::vector<double> thetas = {0.0, 5, 17.5, 30, 45, 55, 62, 68, 72, 75};
            const std::vector<double> dists = {0.6, 1.0, 1.7, 2.5, 3.5};

            // (A) zero motion: bit-identical to the global-shutter kernels (whole scene, smooth weights)
            {
                const Scene g = make_scene(thetas, dists, 7, false, zero, false);
                Scene z = g;
                z.rs = true;
                // the global-shutter kernels twice (their own run-to-run spread: the blend backward accumulates with
                // atomics), then the rolling-shutter kernels at zero motion twice
                const auto a = render_and_grad(g, true), a2 = render_and_grad(g, true);
                const auto b = render_and_grad(z, true), b2 = render_and_grad(z, true);
                struct D {
                    size_t n = 0;
                    double rel = 0;
                };
                auto diff = [](const std::vector<float>& x, const std::vector<float>& y) {
                    D d;
                    double mx = 0;
                    for (float v : x)
                        mx = std::max(mx, static_cast<double>(std::abs(v)));
                    for (size_t i = 0; i < x.size(); ++i) {
                        d.n += std::memcmp(&x[i], &y[i], sizeof(float)) != 0;
                        d.rel = std::max(d.rel, std::abs(static_cast<double>(x[i]) - y[i]) / std::max(mx, 1e-30));
                    }
                    return d;
                };
                auto line = [&](const char* what, const GradRun& x, const GradRun& y) {
                    const D di = diff(x.img, y.img), dm = diff(x.g_means, y.g_means), ds = diff(x.g_scales, y.g_scales),
                            dr = diff(x.g_rots, y.g_rots), dop = diff(x.g_opac, y.g_opac), dsh = diff(x.g_sh0, y.g_sh0);
                    std::println("    {}: differing image values {}; gradient values differing (max |diff| / max |grad|): means {} ({:.1e}), "
                                 "scales {} ({:.1e}), rotations {} ({:.1e}), opacities {} ({:.1e}), colours {} ({:.1e})",
                                 what, di.n, dm.n, dm.rel, ds.n, ds.rel, dr.n, dr.rel, dop.n, dop.rel, dsh.n, dsh.rel);
                    return std::array<double, 2>{static_cast<double>(di.n), std::max({dm.rel, ds.rel, dr.rel, dop.rel, dsh.rel})};
                };
                std::println("(A) zero motion through the rolling-shutter kernels vs the global-shutter kernels ({} splats, {} visible):",
                             g.n, a.n_visible);
                const auto r_gg = line("global vs global (repeat)  ", a, a2);
                const auto r_rr = line("RS-zero vs RS-zero (repeat)", b, b2);
                const auto r_gr = line("global vs RS-zero          ", a, b);
                (void)r_rr;
                // pass: identical images; gradients within the global kernels' own run-to-run spread (or 1e-6 if none)
                if (r_gr[0] != 0 || r_gr[1] > std::max(r_gg[1] * 2.0, 1e-6))
                    ++bad;
            }

            // (B) forward vs double reference, (C) gradients with the times held, (D) times free
            const char* gname[5] = {"omega", "alpha", "vel", "acc", "tau"};
            const double steps[13] = {0.05, 0.05, 0.05, 10, 10, 10, 0.05, 0.05, 0.05, 10, 10, 10, 5e-4};
            double worst_s = 0, worst_c = 0, worst_rs_held = 0, worst_sp_held = 0, worst_rs_free = 0, worst_kappa = 0;
            std::println("(B-D) one splat per render, dL/dimage a bump around its rolling-shutter footprint; motion omega (1.5,-2,0.8) rad/s, "
                         "alpha (40,-25,30) rad/s^2, vel (1.2,-0.8,2.5) u/s, acc (8,-5,6) u/s^2, tau 20 ms (2x the real readout)");
            std::println("  (s = the splat's row fraction, t = s tau; ds px = GPU vs double-reference capture row; ctr RS / ctr GS = blob");
            std::println("   centroid vs the reference projection, rolling shutter / the same splat global shutter; then relative errors");
            std::println("   analytic vs central differences with the time held; RS free = with the time free; kappa = d(row)/dt tau / H)");
            std::println("  {:>6} {:>5} {:>6} | {:>8} {:>6} {:>6} | {:>9} {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9} {:>9} | {:>9} {:>8}",
                         "theta", "dist", "s", "ds px", "ctr RS", "ctr GS", "omega", "alpha", "vel", "acc", "tau", "means", "scales", "rots", "opac",
                         "RS free", "kappa");
            for (size_t i = 0; i < thetas.size(); ++i) {
                Scene s = make_scene(thetas, dists, 21, true, mo, true, static_cast<int>(i));
                const auto base = render_and_grad(s, true);
                if (base.n_visible == 0) {
                    std::println("  {:6.1f}: not visible", thetas[i]);
                    continue;
                }
                // (B) the capture time and the blob centroid
                std::array<double, 12> rows;
                const V3 xw = {s.means[0], s.means[1], s.means[2]};
                const double s_ref = ref_solve(s.L, s.h, mid_rows(s), s.motion, xw, 2, rows);
                const double ds_px = std::abs(base.times[0] - s_ref) * s.h;
                const auto m = ref_project(s.L, apply_rows(rows, xw));
                double sw = 0, sx = 0, sy = 0;
                for (int y = 0; y < s.h; ++y)
                    for (int x = 0; x < s.w; ++x) {
                        double v = 0;
                        for (int c = 0; c < 3; ++c)
                            v += base.img[(static_cast<size_t>(c) * s.h + y) * s.w + x];
                        sw += v;
                        sx += v * (x + 0.5);
                        sy += v * (y + 0.5);
                    }
                const double ctr_rs = std::hypot(sx / sw - m[0], sy / sw - m[1]);
                // the same splat drawn global shutter against the global-shutter reference projection: the centroid's own
                // error (frame-edge clipping, pixel sampling), which the rolling shutter must not add to
                double ctr_gs;
                {
                    Scene gsc = s;
                    gsc.rs = false;
                    const auto gr = render_and_grad(gsc, false);
                    const auto mg = ref_project(s.L, apply_rows(mid_rows(s), xw));
                    double w0 = 0, x0 = 0, y0 = 0;
                    for (int y = 0; y < s.h; ++y)
                        for (int x = 0; x < s.w; ++x) {
                            double v = 0;
                            for (int c = 0; c < 3; ++c)
                                v += gr.img[(static_cast<size_t>(c) * s.h + y) * s.w + x];
                            w0 += v;
                            x0 += v * (x + 0.5);
                            y0 += v * (y + 0.5);
                        }
                    ctr_gs = std::hypot(x0 / w0 - mg[0], y0 / w0 - mg[1]);
                }
                const double ctr = std::max(ctr_rs - ctr_gs, 0.0);
                // kappa: d(row)/dt * tau / H of this splat (how much its time depends on its own row)
                double kappa;
                {
                    const double hstep = 1e-5;
                    const auto pa = ref_project(s.L, apply_rows(ref_rows_at(mid_rows(s), s.motion, s_ref * s.motion.tau + hstep), xw));
                    const auto pb = ref_project(s.L, apply_rows(ref_rows_at(mid_rows(s), s.motion, s_ref * s.motion.tau - hstep), xw));
                    kappa = std::abs((pa[1] - pb[1]) / (2 * hstep)) * s.motion.tau / s.h;
                }
                // (C) held times: RS parameters
                Scene held = s;
                held.times = base.times;
                const auto bh = render_and_grad(held, true);
                std::vector<double> fd_rs(13), an_rs(13), fd_free(13);
                auto motion_set = [](int k) {
                    return [k](Scene& sc, double v) {
                        double* f[13] = {&sc.motion.w[0], &sc.motion.w[1], &sc.motion.w[2], &sc.motion.al[0], &sc.motion.al[1], &sc.motion.al[2],
                                         &sc.motion.v[0], &sc.motion.v[1], &sc.motion.v[2], &sc.motion.a[0], &sc.motion.a[1], &sc.motion.a[2], &sc.motion.tau};
                        *f[k] = v;
                    };
                };
                auto motion_get = [&](int k) {
                    const double f[13] = {mo.w[0], mo.w[1], mo.w[2], mo.al[0], mo.al[1], mo.al[2], mo.v[0], mo.v[1], mo.v[2], mo.a[0], mo.a[1], mo.a[2], mo.tau};
                    return f[k];
                };
                for (int k = 0; k < 13; ++k) {
                    fd_rs[k] = fd(held, motion_set(k), motion_get(k), steps[k]);
                    fd_free[k] = fd(s, motion_set(k), motion_get(k), steps[k]);
                    an_rs[k] = bh.g_rs[k];
                }
                // a splat read within 3% of the centre row (|t| < 0.03 tau) barely depends on the motion: its finite
                // differences are float noise around ~0, so its motion rows are not scored (shown as -1)
                const bool centre_row = std::abs(base.times[0]) < 0.03;
                double e_group[5];
                for (int gi = 0; gi < 5; ++gi) {
                    const int b0 = gi * 3, b1 = gi < 4 ? b0 + 3 : 13;
                    e_group[gi] = centre_row ? -1.0
                                             : rel(std::vector<double>(fd_rs.begin() + b0, fd_rs.begin() + b1),
                                                   std::vector<double>(an_rs.begin() + b0, an_rs.begin() + b1));
                }
                // (C) held times: splat parameters
                auto splat_rel = [&](std::vector<float> Scene::* member, const std::vector<float>& grad, int attrs, double step_abs) {
                    std::vector<double> f(attrs), a(attrs);
                    for (int k = 0; k < attrs; ++k) {
                        const double v0 = (held.*member)[k];
                        f[k] = fd(held, [member, k](Scene& sc, double v) { (sc.*member)[k] = static_cast<float>(v); }, v0, step_abs);
                        a[k] = grad[k];
                    }
                    return rel(f, a);
                };
                const double dist = std::sqrt(std::pow(s.means[0] - s.campos[0], 2) + std::pow(s.means[1] - s.campos[1], 2) + std::pow(s.means[2] - s.campos[2], 2));
                const double e_means = splat_rel(&Scene::means, bh.g_means, 3, 5e-4 * dist);
                const double e_scales = splat_rel(&Scene::scales, bh.g_scales, 3, 2e-3);
                const double e_rots = splat_rel(&Scene::rots, bh.g_rots, 4, 2e-3);
                const double e_opac = splat_rel(&Scene::opac, bh.g_opac, 1, 2e-3);
                const double e_free = centre_row ? -1.0 : rel(fd_free, an_rs);
                std::println("  {:6.1f} {:5.2f} {:+6.3f} | {:8.2e} {:6.3f} {:6.3f} | {:9.2e} {:9.2e} {:9.2e} {:9.2e} {:9.2e} | {:9.2e} {:9.2e} {:9.2e} {:9.2e} | {:9.2e} {:8.2e}",
                             thetas[i], dists[i % dists.size()], base.times[0], ds_px, ctr_rs, ctr_gs, e_group[0], e_group[1], e_group[2], e_group[3], e_group[4],
                             e_means, e_scales, e_rots, e_opac, e_free, kappa);
                worst_s = std::max(worst_s, ds_px);
                worst_c = std::max(worst_c, ctr);
                for (double e : e_group)
                    worst_rs_held = std::max(worst_rs_held, e);
                worst_sp_held = std::max({worst_sp_held, e_means, e_scales, e_rots, e_opac});
                worst_rs_free = std::max(worst_rs_free, e_free);
                worst_kappa = std::max(worst_kappa, kappa);
                if (std::getenv("LFE_RS_VERBOSE")) {
                    for (int k = 0; k < 13; ++k)
                        std::println("      {:>5}[{}] fd {:12.6g} (free {:12.6g})  analytic {:12.6g}", gname[std::min(k / 3, 4)], k % 3, fd_rs[k], fd_free[k], an_rs[k]);
                }
            }
            std::println("worst: capture time {:.2e} px, centroid {:.4f} px; held-time relative errors: motion {:.2e}, splat {:.2e}; "
                         "time free: {:.2e} (largest kappa {:.2e})",
                         worst_s, worst_c, worst_rs_held, worst_sp_held, worst_rs_free, worst_kappa);
            if (worst_s > 0.01 || worst_c > 0.05)
                ++bad;
            if (worst_rs_held > 0.01 || worst_sp_held > 0.01)
                ++bad;
            if (bad == 0)
                std::println("RS GRADCHECK OK");
            else
                std::println("RS GRADCHECK: {} failed checks", bad);
            return bad == 0 ? 0 : 1;
        }
    } // namespace

    int run_lfe_rs_tool(int argc, const char* const* argv) {
        if (argc >= 6 && std::string(argv[2]) == "render")
            return run_render(argv[3], argv[4], argv[5]);
        if (argc >= 3 && std::string(argv[2]) == "gradcheck")
            return run_gradcheck();
        std::println(stderr, "usage: LichtFeld-Studio lfe-rs render <model.ply> <cameras.txt> <out_dir> | gradcheck");
        return 2;
    }

} // namespace lfs::app
