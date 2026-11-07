/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Developer checks of the motion blur (--motion-blur; FastGSRollingShutter::exposure, native fisheye, with the rolling shutter).
//   LichtFeld-Studio lfe-blur render <model.ply> <cameras.txt> <out_dir>
//     cameras.txt, one camera per line: lfe-rs render's columns with the rolling shutter, then the exposure in ms:
//       name width height model fx fy cx cy k1 k2 k3 k4 qw qx qy qz tx ty tz tau_ms wx wy wz ax ay az vx vy vz Ax Ay Az exposure_ms
//     For each mode of LFE_BLUR_MODES (comma list, default sharp,a,b): sharp = the rolling shutter alone; a = the screen-space
//     blur (the trainer's); b = the reference: the average of LFE_BLUR_K (default 32) sub-exposure renders at times
//     t + T ((k + 0.5) / K - 1/2) (FastGSRollingShutter::time_offset), exact up to the rolling shutter's own model.
//     Writes <out_dir>/<name>_<mode>.f32 (float32 RGB, H x W x 3) and <name>_<mode>_alpha.f32 (H x W).
//     LFE_BLUR_NEAR=d: only splats within distance d of the camera centre (the others' opacity off; edge-profile checks).
//     LFE_BLUR_SMEAR=1: also <name>_smear{,_rot,_trav}.f32 (float32 per model splat: |dmu| px of the drawn splats, -1 if not
//     drawn; full motion, rotation only, travel only) and <name>_dist.f32 (distance from the camera centre).
//     Background LFE_RS_BG="r,g,b" (0-255, default black).
//   LichtFeld-Studio lfe-blur gradcheck <out_dir>
//     (A) exposure bound at 0 ms: image bit-identical to no blur, gradients within the backward's own run-to-run spread;
//     (B) dL/d(means, scales, rotations, opacities), dL/d(motion[12], tau), dL/d(exposure scale) and dL/d(pose) against
//         central finite differences (splat times held, as the backward does);
//     (C) coverage: one splat's summed alpha with and without the blur against the prediction (kept up to the 1/255 floor);
//     (D) the scene, its sharp / (a) / (b, K = 16) CUDA renders for blur_ref.py (D:/splat_work/lfrqplcym), and the (a) vs (b) error.
//   LFQPLCYKM: LFE_BLUR_SHAPE=mix (betas 1.05 .. 7.9 cycled over the splats) or =<beta> gives the gradcheck scene a
//     --kernel-shape beta per splat: (B) adds dL/dbeta (FD on the raw r) and (C) runs the coverage check at beta 1.2 / 2 / 3.3 / 6
//     (4 px splats); (D) also writes shape.f32 (beta per splat) for blur_shape_ref.py (D:/splat_work/lfrqplcykm). The render
//     mode draws a PLY's own beta (its `shape` property) as every FastGS render does.
#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/kernel_shape_map.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/loader.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <chrono>
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

        std::array<double, 9> quat_rot(double qw, double qx, double qy, double qz) {
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

        void write_f32(const std::filesystem::path& p, const float* d, size_t n) {
            std::ofstream f(p, std::ios::binary);
            f.write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(sizeof(float) * n));
        }

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
            const size_t N = model.size();
            std::println("{} splats, SH degree {}", N, model.get_active_sh_degree());
            std::filesystem::create_directories(out_dir);
            std::vector<float> bgv = {0.0f, 0.0f, 0.0f};
            if (const char* e = std::getenv("LFE_RS_BG"); e && *e) {
                std::istringstream bs(e);
                std::string tok;
                for (int c = 0; c < 3 && std::getline(bs, tok, ','); ++c)
                    bgv[c] = static_cast<float>(std::atof(tok.c_str()) / 255.0);
            }
            auto background = Tensor::from_vector(bgv, {3}, Device::CPU).to(Device::CUDA);
            std::vector<std::string> modes;
            {
                const char* e = std::getenv("LFE_BLUR_MODES");
                std::istringstream ms(e && *e ? e : "sharp,a,b");
                std::string tok;
                while (std::getline(ms, tok, ','))
                    if (!tok.empty())
                        modes.push_back(tok);
            }
            const int K = std::max(1, std::getenv("LFE_BLUR_K") ? std::atoi(std::getenv("LFE_BLUR_K")) : 32);
            const double near_d = std::getenv("LFE_BLUR_NEAR") ? std::atof(std::getenv("LFE_BLUR_NEAR")) : 0.0;
            const bool smear_dump = std::getenv("LFE_BLUR_SMEAR") && std::string(std::getenv("LFE_BLUR_SMEAR")) == "1";
            const auto means_host = model.means().cpu();
            const float* mh = means_host.ptr<float>();
            const Tensor opacity_full = model.opacity_raw().clone();
            DevBuf motion(12), readout(1), exposure(1), smear(N);

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
                std::vector<double> cols;
                double v;
                while (ss >> v)
                    cols.push_back(v);
                if (cols.size() != 14 || !(model_name == "FISHEYE" || model_name == "OPENCV_FISHEYE")) {
                    std::println(stderr, "fisheye camera + 14 columns (tau_ms, 12 motion, exposure_ms) expected: {}", line);
                    return 1;
                }
                const auto r = quat_rot(qw, qx, qy, qz);
                std::vector<float> rf(9);
                for (int i = 0; i < 9; ++i)
                    rf[i] = static_cast<float>(r[i]);
                auto R = Tensor::from_vector(rf, {3, 3}, Device::CPU);
                auto T = Tensor::from_vector(std::vector<float>{static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz)}, {3}, Device::CPU);
                auto radial = Tensor::from_vector(std::vector<float>{static_cast<float>(k1), static_cast<float>(k2), static_cast<float>(k3), static_cast<float>(k4)}, {4}, Device::CPU);
                Camera camera(R, T, static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx), static_cast<float>(cy),
                              radial, Tensor::empty({0}, Device::CPU), CameraModelType::FISHEYE, name, {}, {}, w, h, uid++);
                // camera centre c = -R^T t
                const double C[3] = {-(r[0] * tx + r[3] * ty + r[6] * tz), -(r[1] * tx + r[4] * ty + r[7] * tz), -(r[2] * tx + r[5] * ty + r[8] * tz)};
                std::vector<float> dist(N);
                for (size_t i = 0; i < N; ++i) {
                    const double dx = mh[3 * i] - C[0], dy = mh[3 * i + 1] - C[1], dz = mh[3 * i + 2] - C[2];
                    dist[i] = static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz));
                }
                if (near_d > 0.0) {
                    auto op = opacity_full.cpu();
                    float* o = op.ptr<float>();
                    size_t kept = 0;
                    for (size_t i = 0; i < N; ++i) {
                        if (dist[i] > near_d)
                            o[i] = -30.0f;
                        else
                            ++kept;
                    }
                    model.opacity_raw() = op.to(Device::CUDA);
                    std::println("{}: {} splats within {} units", name, kept, near_d);
                }
                std::vector<float> m(12);
                for (int i = 0; i < 12; ++i)
                    m[i] = static_cast<float>(cols[1 + i]);
                readout.upload({static_cast<float>(cols[0] / 1000.0)});
                const double exp_s = cols[13] / 1000.0;
                exposure.upload({static_cast<float>(exp_s)});
                const size_t npx = static_cast<size_t>(w) * h;
                auto render = [&](fr::FastGSRollingShutter rs, std::vector<float>* img, std::vector<float>* alpha) {
                    auto result = lfs::training::fast_rasterize_forward(camera, model, background, 0, 0, 0, 0, false, {}, false, true,
                                                                        1.0f, false, {}, 0, rs);
                    if (!result) {
                        std::println(stderr, "{}: render failed: {}", name, result.error().detail());
                        std::exit(1);
                    }
                    auto out = std::move(result->first);
                    result->second.release_forward_context();
                    if (img) {
                        auto hwc = out.image.permute({1, 2, 0}).contiguous().cpu();
                        img->assign(hwc.ptr<float>(), hwc.ptr<float>() + 3 * npx);
                    }
                    if (alpha) {
                        auto a = out.alpha.contiguous().cpu();
                        alpha->assign(a.ptr<float>(), a.ptr<float>() + npx);
                    }
                };
                auto base = [&](const std::vector<float>& mot) {
                    motion.upload(mot);
                    fr::FastGSRollingShutter rs{};
                    rs.motion = motion.p;
                    rs.readout = readout.p;
                    return rs;
                };
                if (const char* bn = std::getenv("LFE_BLUR_BENCH"); bn && std::atoi(bn) > 0) {
                    const int nb = std::atoi(bn);
                    std::vector<double> t_off, t_on;
                    for (int k = 0; k < 2 * nb + 2; ++k) {
                        auto rs = base(m);
                        if (k % 2 == 1)
                            rs.exposure = exposure.p;
                        cudaDeviceSynchronize();
                        const auto t0 = std::chrono::steady_clock::now();
                        auto result = lfs::training::fast_rasterize_forward(camera, model, background, 0, 0, 0, 0, false, {}, false, true,
                                                                            1.0f, false, {}, 0, rs);
                        cudaDeviceSynchronize();
                        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                        if (result)
                            result->second.release_forward_context();
                        if (k >= 2)
                            (k % 2 == 1 ? t_on : t_off).push_back(ms);
                    }
                    std::sort(t_off.begin(), t_off.end());
                    std::sort(t_on.begin(), t_on.end());
                    std::println("{} bench: forward median {:.2f} ms without the blur, {:.2f} ms with it (x{:.3f}; {} interleaved pairs)", name,
                                 t_off[t_off.size() / 2], t_on[t_on.size() / 2], t_on[t_on.size() / 2] / t_off[t_off.size() / 2], nb);
                }
                for (const auto& mode : modes) {
                    std::vector<float> img, alpha;
                    if (mode == "sharp") {
                        render(base(m), &img, &alpha);
                    } else if (mode == "a") {
                        auto rs = base(m);
                        rs.exposure = exposure.p;
                        render(rs, &img, &alpha);
                    } else if (mode == "b") {
                        std::vector<double> ai(3 * npx, 0.0), aa(npx, 0.0);
                        for (int k = 0; k < K; ++k) {
                            auto rs = base(m);
                            rs.time_offset = static_cast<float>(exp_s * ((k + 0.5) / K - 0.5));
                            render(rs, &img, &alpha);
                            for (size_t i = 0; i < 3 * npx; ++i)
                                ai[i] += img[i];
                            for (size_t i = 0; i < npx; ++i)
                                aa[i] += alpha[i];
                        }
                        for (size_t i = 0; i < 3 * npx; ++i)
                            img[i] = static_cast<float>(ai[i] / K);
                        for (size_t i = 0; i < npx; ++i)
                            alpha[i] = static_cast<float>(aa[i] / K);
                    } else {
                        std::println(stderr, "unknown mode '{}' (sharp, a, b)", mode);
                        return 1;
                    }
                    write_f32(out_dir / (name + "_" + mode + ".f32"), img.data(), img.size());
                    write_f32(out_dir / (name + "_" + mode + "_alpha.f32"), alpha.data(), alpha.size());
                    std::println("{} {}x{} {} (exposure {} ms{})", name, w, h, mode, cols[13], mode == "b" ? std::format(", K {}", K) : "");
                }
                if (smear_dump) {
                    const char* tags[3] = {"", "_rot", "_trav"};
                    for (int variant = 0; variant < 3; ++variant) {
                        std::vector<float> mv = m;
                        for (int i = 0; i < 6; ++i) {
                            if (variant == 1)
                                mv[6 + i] = 0.0f; // rotation only
                            if (variant == 2)
                                mv[i] = 0.0f; // travel only
                        }
                        std::vector<float> minus(N, -1.0f);
                        smear.upload(minus);
                        auto rs = base(mv);
                        rs.exposure = exposure.p;
                        rs.smear = smear.p;
                        render(rs, nullptr, nullptr);
                        cudaDeviceSynchronize();
                        const auto sm = smear.download();
                        write_f32(out_dir / (name + "_smear" + tags[variant] + ".f32"), sm.data(), N);
                    }
                    write_f32(out_dir / (name + "_dist.f32"), dist.data(), N);
                }
                if (near_d > 0.0)
                    model.opacity_raw() = opacity_full.clone();
            }
            return 0;
        }

        // ---------------------------------------------------------------- gradcheck scene
        struct Scene {
            int n = 0, w = 320, h = 240;
            double fx = 140, fy = 140, cx = 160, cy = 120, k[4] = {0.39127139, -0.03901020, 0.00243290, -0.00266145};
            fr::FastGSLens lens{};
            std::vector<float> means, scales, rots, opac, sh0, w2c, campos, weights;
            std::vector<float> motion; // 12
            float tau = 0.031f, exposure = 0.012f, scale = 1.0f;
            bool blur = true;
            float time_offset = 0.0f;
            std::vector<float> times; // fixed row fractions (empty = solve)
            bool bind_scale = true, pose = true;
            std::vector<float> shape; // --kernel-shape (LFQPLCYKM): raw r per splat; empty = the Gaussian (no shape bound)
        };

        struct Run {
            double loss = 0;
            std::vector<float> img, alpha, g_means, g_scales, g_rots, g_opac, g_rs, times, g_pose;
            std::vector<float> g_shape; // dL/dbeta (LFQPLCYKM)
            float g_exp = 0;
        };

        Run render_and_grad(const Scene& s, bool with_backward) {
            const int N = s.n;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_w2c(16), d_cam(3);
            DevBuf d_img(3ull * s.w * s.h), d_alpha(1ull * s.w * s.h), d_depth(1ull * s.w * s.h), d_weights(3ull * s.w * s.h);
            DevBuf d_motion(12), d_readout(1), d_grad(13), d_times(std::max(N, 1)), d_exp(1), d_scale(1), d_gexp(1), d_pg(6);
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_w2c.upload(s.w2c);
            d_cam.upload(s.campos);
            d_weights.upload(s.weights);
            d_motion.upload(s.motion);
            d_readout.upload({s.tau});
            d_exp.upload({s.exposure});
            d_scale.upload({s.scale});
            DevBuf d_shape(std::max(N, 1)), d_gshape(std::max(N, 1));
            fr::FastGSKernelShape kshape{};
            if (!s.shape.empty()) { // --kernel-shape (LFQPLCYKM)
                d_shape.upload(s.shape);
                kshape.raw = d_shape.p;
                kshape.grad = d_gshape.p;
            }
            fr::FastGSRollingShutter rs{};
            rs.motion = d_motion.p;
            rs.readout = d_readout.p;
            rs.grad = d_grad.p;
            rs.times = d_times.p;
            rs.time_offset = s.time_offset;
            if (!s.times.empty()) {
                d_times.upload(s.times);
                rs.times_fixed = 1;
            }
            if (s.blur) {
                rs.exposure = d_exp.p;
                if (s.bind_scale) {
                    rs.exposure_scale = d_scale.p;
                    rs.grad_exposure = d_gexp.p;
                }
            }
            Run run;
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, nullptr, d_w2c.p, d_cam.p,
                                       d_img.p, d_alpha.p, d_depth.p, nullptr, nullptr, nullptr, N, 1, 1, s.w, s.h,
                                       static_cast<float>(s.fx), static_cast<float>(s.fy), static_cast<float>(s.cx),
                                       static_cast<float>(s.cy), 0.01f, 1e10f, false, nullptr, nullptr, 0, 0, nullptr, 1.0f,
                                       s.lens, {}, 0, rs, kshape);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            run.img = d_img.download();
            run.alpha = d_alpha.download();
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
            if (s.pose)
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
            const auto res = fr::backward_raw(nullptr, nullptr, d_weights.p, fr::BackgroundAlphaGradient{}, nullptr, nullptr,
                                              d_img.p, d_alpha.p, d_means.p, d_scales.p, d_rots.p, d_opac.p, nullptr,
                                              d_w2c.p, d_cam.p, ctx, nullptr, N, ctx.n_visible, 1, 1, s.w, s.h,
                                              static_cast<float>(s.fx), static_cast<float>(s.fy), static_cast<float>(s.cx),
                                              static_cast<float>(s.cy), false, DensificationType::None, &fa);
            cudaDeviceSynchronize();
            if (!res.success) {
                std::println(stderr, "backward failed: {}", res.error_message ? res.error_message : "?");
                std::exit(1);
            }
            run.g_means = g_means.download();
            run.g_scales = g_scales.download();
            run.g_rots = g_rots.download();
            run.g_opac = g_opac.download();
            run.g_rs = d_grad.download();
            run.g_exp = d_gexp.download()[0];
            run.g_pose = d_pg.download();
            if (!s.shape.empty())
                run.g_shape = d_gshape.download();
            return run;
        }

        // LFE_BLUR_SHAPE (LFQPLCYKM): unset = no kernel shape; "mix" = betas cycled over the splats; a number = that beta
        std::vector<float> scene_shape(int n) {
            const char* e = std::getenv("LFE_BLUR_SHAPE");
            if (e == nullptr || *e == 0)
                return {};
            static const float mix[] = {1.05f, 3.3f, 1.3f, 6.0f, 1.7f, 4.5f, 2.6f, 7.9f, 2.0f};
            std::vector<float> r;
            for (int i = 0; i < n; ++i)
                r.push_back(lfs::core::kshape::raw_from_beta(std::string(e) == "mix" ? mix[i % 9] : static_cast<float>(std::atof(e))));
            return r;
        }

        // splats in front of an identity camera at distances 0.6-4 units, 1-4 px, the RS tool's fisheye lens
        Scene make_scene(unsigned seed, int n) {
            Scene s;
            s.n = n;
            s.lens.model = fr::kLensFisheye;
            s.lens.k1 = static_cast<float>(s.k[0]);
            s.lens.k2 = static_cast<float>(s.k[1]);
            s.lens.k3 = static_cast<float>(s.k[2]);
            s.lens.k4 = static_cast<float>(s.k[3]);
            const double tmax = std::tan(88.0 * std::numbers::pi / 180.0);
            s.lens.max_w = static_cast<float>(tmax * tmax);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            // a slightly rotated camera at (0.1, -0.05, 0.2)
            const auto R = quat_rot(0.995, 0.03, -0.05, 0.08);
            const double C[3] = {0.1, -0.05, 0.2};
            s.w2c.assign(16, 0.0f);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    s.w2c[i * 4 + j] = static_cast<float>(R[i * 3 + j]);
                s.w2c[i * 4 + 3] = static_cast<float>(-(R[i * 3] * C[0] + R[i * 3 + 1] * C[1] + R[i * 3 + 2] * C[2]));
            }
            s.w2c[15] = 1.0f;
            s.campos = {static_cast<float>(C[0]), static_cast<float>(C[1]), static_cast<float>(C[2])};
            for (int i = 0; i < n; ++i) {
                const double d = 0.6 + 3.4 * uni(rng) * uni(rng), th = 0.9 * uni(rng), ph = 2 * std::numbers::pi * uni(rng);
                const double pc[3] = {d * std::sin(th) * std::cos(ph), d * std::sin(th) * std::sin(ph), d * std::cos(th)};
                for (int j = 0; j < 3; ++j) // world = R^T pc + C
                    s.means.push_back(static_cast<float>(R[j] * pc[0] + R[3 + j] * pc[1] + R[6 + j] * pc[2] + C[j]));
                const double px = 1.0 + 3.0 * uni(rng);
                for (int j = 0; j < 3; ++j)
                    s.scales.push_back(static_cast<float>(std::log(px * d / s.fx * (0.6 + 0.8 * uni(rng)))));
                double q[4] = {uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1};
                for (double x : q)
                    s.rots.push_back(static_cast<float>(x));
                const double o = 0.3 + 0.65 * uni(rng);
                s.opac.push_back(static_cast<float>(std::log(o / (1 - o))));
                for (int j = 0; j < 3; ++j)
                    s.sh0.push_back(static_cast<float>(uni(rng) * 3 - 1.5));
            }
            s.weights.resize(3ull * s.w * s.h);
            for (int y = 0; y < s.h; ++y)
                for (int x = 0; x < s.w; ++x)
                    for (int c = 0; c < 3; ++c)
                        s.weights[(static_cast<size_t>(c) * s.h + y) * s.w + x] =
                            static_cast<float>(std::cos(0.05 * x + 0.9 * c) * std::sin(0.07 * y + 0.3) + 0.3);
            s.motion = {0.9f, 1.6f, 0.4f, 3.0f, -2.0f, 1.0f, 2.5f, 0.8f, 1.5f, 5.0f, -3.0f, 2.0f};
            s.shape = scene_shape(n);
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

        template <class F>
        double fd(Scene s, F set, double step) {
            set(s, step);
            const double lp = render_and_grad(s, false).loss;
            set(s, -2 * step);
            const double lm = render_and_grad(s, false).loss;
            return (lp - lm) / (2 * step);
        }

        // the pose perturbation of FastGSPose: W <- exp(-[dr]x) W, b <- exp(-[dr]x) (b - dd), c <- c + W^T dd (double)
        void perturb_pose(Scene& s, int k, double e) {
            double W[9], b[3], dr[3] = {0, 0, 0}, dd[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    W[i * 3 + j] = s.w2c[i * 4 + j];
                b[i] = s.w2c[i * 4 + 3];
            }
            (k < 3 ? dr[k] : dd[k - 3]) = e;
            // E = exp(-[dr]x) (Rodrigues)
            const double th = std::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
            double E[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            if (th > 0) {
                const double x = -dr[0] / th, y = -dr[1] / th, z = -dr[2] / th, c = std::cos(th), sn = std::sin(th), t = 1 - c;
                const double M[9] = {c + x * x * t, x * y * t - z * sn, x * z * t + y * sn, x * y * t + z * sn, c + y * y * t, y * z * t - x * sn,
                                     x * z * t - y * sn, y * z * t + x * sn, c + z * z * t};
                std::copy(M, M + 9, E);
            }
            for (int i = 0; i < 3; ++i)
                s.campos[i] = static_cast<float>(s.campos[i] + W[0 * 3 + i] * dd[0] + W[1 * 3 + i] * dd[1] + W[2 * 3 + i] * dd[2]);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    s.w2c[i * 4 + j] = static_cast<float>(E[i * 3] * W[j] + E[i * 3 + 1] * W[3 + j] + E[i * 3 + 2] * W[6 + j]);
                s.w2c[i * 4 + 3] = static_cast<float>(E[i * 3] * (b[0] - dd[0]) + E[i * 3 + 1] * (b[1] - dd[1]) + E[i * 3 + 2] * (b[2] - dd[2]));
            }
        }

        int run_gradcheck(const std::filesystem::path& out_dir) {
            int bad = 0;
            auto gate = [&bad](bool ok, const std::string& what) {
                std::println("  GATE {}: {}", ok ? "PASS" : "FAIL", what);
                if (!ok)
                    ++bad;
            };
            std::filesystem::create_directories(out_dir);
            const Scene base = make_scene(11, 24);

            // (A) exposure bound at 0 ms vs no blur
            {
                Scene off = base;
                off.blur = false;
                Scene z = base;
                z.exposure = 0.0f;
                const auto a = render_and_grad(off, true), a2 = render_and_grad(off, true), b = render_and_grad(z, true);
                const bool img_same = std::memcmp(a.img.data(), b.img.data(), sizeof(float) * a.img.size()) == 0 &&
                                      std::memcmp(a.alpha.data(), b.alpha.data(), sizeof(float) * a.alpha.size()) == 0;
                const bool self_same = std::memcmp(a.img.data(), a2.img.data(), sizeof(float) * a.img.size()) == 0;
                auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
                    double m = 0;
                    for (size_t i = 0; i < x.size(); ++i)
                        m = std::max(m, static_cast<double>(std::fabs(x[i] - y[i])));
                    return m;
                };
                const double spread = std::max({maxd(a.g_means, a2.g_means), maxd(a.g_scales, a2.g_scales), maxd(a.g_rots, a2.g_rots),
                                                maxd(a.g_opac, a2.g_opac), maxd(a.g_rs, a2.g_rs), maxd(a.g_pose, a2.g_pose)});
                const double diff = std::max({maxd(a.g_means, b.g_means), maxd(a.g_scales, b.g_scales), maxd(a.g_rots, b.g_rots),
                                              maxd(a.g_opac, b.g_opac), maxd(a.g_rs, b.g_rs), maxd(a.g_pose, b.g_pose)});
                double dimg = 0, dself = 0;
                for (size_t i = 0; i < a.img.size(); ++i) {
                    dimg = std::max(dimg, static_cast<double>(std::fabs(a.img[i] - b.img[i])));
                    dself = std::max(dself, static_cast<double>(std::fabs(a.img[i] - a2.img[i])));
                }
                std::println("(A) no-blur forward run to run: {} (max {:.3e}); exposure 0 bound vs no blur: max |image diff| {:.3e}", self_same ? "bit-identical" : "differs", dself, dimg);
                std::println("(A) exposure 0 ms bound vs no blur: image + alpha {}; gradients max |diff| {:.3e} (no-blur run-to-run spread {:.3e}); "
                             "dL/d(scale) at 0 ms {:.3e}", img_same ? "bit-identical" : "DIFFER", diff, spread, b.g_exp);
                gate(img_same || dimg <= 1e-6, std::format("(A) exposure 0 bound (the BLUR kernels): image within 1e-6 of no blur ({:.2e})", dimg));
                const double gmax = std::max({maxd(a.g_means, std::vector<float>(a.g_means.size(), 0.0f)),
                                              maxd(a.g_scales, std::vector<float>(a.g_scales.size(), 0.0f)),
                                              maxd(a.g_opac, std::vector<float>(a.g_opac.size(), 0.0f))});
                gate(diff <= std::max(4.0 * spread, 1e-6 * gmax),
                     std::format("(A) exposure 0: gradients within 4x the run-to-run spread or 1e-6 of the largest ({:.3e})", gmax));
            }

            // (B) finite differences, times held
            {
                Scene s = base;
                const auto r0 = render_and_grad(s, true);
                s.times = r0.times;
                // dL/dimage local to the splats' cores (smoothstep of the base render's alpha from 0.1 to 0.3), so no pixel near a
                // splat's 1/255 alpha floor (a step the analytic backward does not differentiate) carries weight
                for (int c = 0; c < 3; ++c)
                    for (size_t i = 0; i < r0.alpha.size(); ++i) {
                        const double u = std::clamp((r0.alpha[i] - 0.1) / 0.2, 0.0, 1.0);
                        s.weights[c * r0.alpha.size() + i] *= static_cast<float>(u * u * (3 - 2 * u));
                    }
                const auto r = render_and_grad(s, true);
                Scene sb = s;
                sb.blur = false;
                const auto rnb = render_and_grad(sb, true);
                std::println("(B) loss {:.6f} (no blur {:.6f}); finite differences with the splat times held", r.loss, rnb.loss);
                const std::vector<int> pick = {0, 3, 5, 8, 13, 17, 21};
                struct G {
                    const char* name;
                    std::vector<double> fd, an, nb;
                };
                G gm{"means"}, gs{"scales"}, gr{"rotations"}, go{"opacities"}, gmo{"motion[12]+tau"}, ge{"exposure scale"}, gp{"pose[6]"};
                G bm{"means (no blur)"}, bmo{"motion (no blur)"}, bp{"pose (no blur)"};
                G gb{"shape (beta)"}, bsh{"beta (no blur)"}; // --kernel-shape (LFQPLCYKM): FD on the raw r / (dbeta/dr), every splat
                if (!s.shape.empty()) {
                    for (int i = 0; i < s.n; ++i) {
                        const double db = lfs::core::kshape::dbeta_draw(lfs::core::kshape::beta_from_raw(s.shape[i]));
                        gb.fd.push_back(fd(s, [i](Scene& q, double e) { q.shape[i] += static_cast<float>(e); }, 2e-3) / db);
                        gb.an.push_back(r.g_shape[i]);
                        gb.nb.push_back(rnb.g_shape[i]);
                        bsh.fd.push_back(fd(sb, [i](Scene& q, double e) { q.shape[i] += static_cast<float>(e); }, 2e-3) / db);
                        bsh.an.push_back(rnb.g_shape[i]);
                        bsh.nb.push_back(rnb.g_shape[i]);
                    }
                }
                for (int i : pick) {
                    for (int j = 0; j < 3; ++j) {
                        gm.fd.push_back(fd(s, [i, j](Scene& q, double e) { q.means[3 * i + j] += static_cast<float>(e); }, 1e-3));
                        gm.an.push_back(r.g_means[3 * i + j]);
                        gm.nb.push_back(rnb.g_means[3 * i + j]);
                        gs.fd.push_back(fd(s, [i, j](Scene& q, double e) { q.scales[3 * i + j] += static_cast<float>(e); }, 1e-3));
                        gs.an.push_back(r.g_scales[3 * i + j]);
                        gs.nb.push_back(rnb.g_scales[3 * i + j]);
                    }
                    for (int j = 0; j < 4; ++j) {
                        gr.fd.push_back(fd(s, [i, j](Scene& q, double e) { q.rots[4 * i + j] += static_cast<float>(e); }, 1e-3));
                        gr.an.push_back(r.g_rots[4 * i + j]);
                        gr.nb.push_back(rnb.g_rots[4 * i + j]);
                    }
                    for (int j = 0; j < 3; ++j) { // the same check without the blur: the finite differences' own floor
                        bm.fd.push_back(fd(sb, [i, j](Scene& q, double e) { q.means[3 * i + j] += static_cast<float>(e); }, 1e-3));
                        bm.an.push_back(rnb.g_means[3 * i + j]);
                        bm.nb.push_back(rnb.g_means[3 * i + j]);
                    }
                    go.fd.push_back(fd(s, [i](Scene& q, double e) { q.opac[i] += static_cast<float>(e); }, 1e-3));
                    go.an.push_back(r.g_opac[i]);
                    go.nb.push_back(rnb.g_opac[i]);
                }
                const double msteps[12] = {5e-2, 5e-2, 5e-2, 5.0, 5.0, 5.0, 5e-2, 5e-2, 5e-2, 5.0, 5.0, 5.0};
                for (int k = 0; k < 12; ++k) {
                    gmo.fd.push_back(fd(s, [k](Scene& q, double e) { q.motion[k] += static_cast<float>(e); }, msteps[k]));
                    gmo.an.push_back(r.g_rs[k]);
                    gmo.nb.push_back(rnb.g_rs[k]);
                    bmo.fd.push_back(fd(sb, [k](Scene& q, double e) { q.motion[k] += static_cast<float>(e); }, msteps[k]));
                    bmo.an.push_back(rnb.g_rs[k]);
                    bmo.nb.push_back(rnb.g_rs[k]);
                }
                gmo.fd.push_back(fd(s, [](Scene& q, double e) { q.tau += static_cast<float>(e); }, 5e-4));
                gmo.an.push_back(r.g_rs[12]);
                gmo.nb.push_back(rnb.g_rs[12]);
                ge.fd.push_back(fd(s, [](Scene& q, double e) { q.scale += static_cast<float>(e); }, 5e-2));
                ge.an.push_back(r.g_exp);
                ge.nb.push_back(0.0);
                for (int k = 0; k < 6; ++k) {
                    Scene sp = s, sm = s;
                    perturb_pose(sp, k, 1e-3);
                    perturb_pose(sm, k, -1e-3);
                    gp.fd.push_back((render_and_grad(sp, false).loss - render_and_grad(sm, false).loss) / 2e-3);
                    gp.an.push_back(r.g_pose[k]);
                    gp.nb.push_back(rnb.g_pose[k]);
                    Scene bp_p = sb, bp_m = sb;
                    perturb_pose(bp_p, k, 1e-3);
                    perturb_pose(bp_m, k, -1e-3);
                    bp.fd.push_back((render_and_grad(bp_p, false).loss - render_and_grad(bp_m, false).loss) / 2e-3);
                    bp.an.push_back(rnb.g_pose[k]);
                    bp.nb.push_back(rnb.g_pose[k]);
                }
                std::vector<const G*> refs = {&bm, &bmo, &bp}, checks = {&gm, &gs, &gr, &go, &gmo, &ge, &gp};
                if (!s.shape.empty()) {
                    refs.push_back(&bsh);
                    checks.push_back(&gb);
                }
                for (const G* g : refs) {
                    const double e = rel(g->fd, g->an);
                    std::println("  {:>16}: |FD - analytic| / |FD| = {:.4f}   (reference: the kernels without the blur)", g->name, e);
                }
                // components where the no-blur kernels match their own finite differences to 1% (FD-reliable points)
                auto reliable = [](const G& ref, const G& g) {
                    G o{g.name};
                    for (size_t k = 0; k < g.fd.size() && k < ref.fd.size(); ++k)
                        if (std::fabs(ref.fd[k] - ref.an[k]) <= 0.01 * std::max(std::fabs(ref.fd[k]), 1e-6)) {
                            o.fd.push_back(g.fd[k]);
                            o.an.push_back(g.an[k]);
                        }
                    return o;
                };
                for (const G* g : checks) {
                    const double e = rel(g->fd, g->an), enb = rel(g->fd, g->nb);
                    std::println("  {:>15}: |FD - analytic| / |FD| = {:.4f}   (the no-blur gradient would be {:.4f} off: the blur terms' size)",
                                 g->name, e, enb);
                    const G* ref = g == &gm ? &bm : g == &gmo ? &bmo : g == &gp ? &bp : g == &gb ? &bsh : nullptr;
                    if (ref != nullptr) {
                        const G sub = reliable(*ref, *g);
                        const double es = rel(sub.fd, sub.an);
                        std::println("  {:>15}  on the {} of {} components where the no-blur kernels match their FD to 1%: {:.4f}", "", sub.fd.size(),
                                     g->fd.size(), es);
                        gate(es < 0.03 && sub.fd.size() * 2 >= g->fd.size(),
                             std::format("(B) {} gradient within 3% of central differences (FD-reliable components)", g->name));
                    } else {
                        gate(e < 0.03, std::format("(B) {} gradient within 3% of central differences", g->name));
                    }
                }
                if (std::getenv("LFE_BLUR_VERBOSE"))
                    for (size_t k = 0; k < gb.fd.size(); ++k)
                        std::println("    beta[{}] {:.3f} fd {:+.5e} an {:+.5e} (no blur: fd {:+.5e} an {:+.5e})", k,
                                     lfs::core::kshape::beta_from_raw(s.shape[k]), gb.fd[k], gb.an[k], bsh.fd[k], bsh.an[k]);
                if (std::getenv("LFE_BLUR_VERBOSE"))
                    for (size_t k = 0; k < gm.fd.size(); ++k)
                        std::println("    mean[{}] fd {:+.5e} an {:+.5e} (no blur: fd {:+.5e} an {:+.5e})", k, gm.fd[k], gm.an[k], bm.fd[k], bm.an[k]);
                std::println("    exposure scale fd {:.6e} an {:.6e}; pose fd {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}", ge.fd[0], ge.an[0],
                             gp.fd[0], gp.fd[1], gp.fd[2], gp.fd[3], gp.fd[4], gp.fd[5]);
            }

            // (C) coverage: one splat, summed alpha with / without the blur vs the prediction (Gaussian mass above the floor)
            {
                std::println("(C) coverage of one splat (summed alpha; predicted: the same mass, less the part under the 1/255 alpha floor)");
                // LFQPLCYKM (LFE_BLUR_SHAPE set): the same at beta 1.2 / 2 / 3.3 / 6 with 4 px splats; the beta profile keeps
                // q P(2/beta, ln(q / floor)) of its mass above the floor (P: the regularized lower incomplete gamma)
                const bool with_shape = !base.shape.empty();
                const std::vector<double> betas = with_shape ? std::vector<double>{1.2, 2.0, 3.3, 6.0} : std::vector<double>{2.0};
                auto reg_lower_gamma = [](double a, double x) {
                    double term = 1.0 / a, sum = term;
                    for (int n = 1; n < 400 && term > 1e-17 * sum; ++n) {
                        term *= x / (a + n);
                        sum += term;
                    }
                    return std::exp(a * std::log(x) - x - std::lgamma(a)) * sum;
                };
                for (const double beta : betas)
                for (const double o : {0.5, 0.95, 0.99995}) {
                    for (const double te : {0.004, 0.012, 0.03}) {
                        Scene s = make_scene(3, 1);
                        if (with_shape)
                            s.shape = {lfs::core::kshape::raw_from_beta(static_cast<float>(beta))};
                        // a 1.5-px round splat 1 unit ahead, 20 deg off axis; travel sideways
                        const double pc[3] = {0.0, std::sin(0.35), std::cos(0.35)};
                        const auto R = quat_rot(0.995, 0.03, -0.05, 0.08);
                        for (int j = 0; j < 3; ++j)
                            s.means[j] = static_cast<float>(R[j] * pc[0] + R[3 + j] * pc[1] + R[6 + j] * pc[2] + s.campos[j]);
                        for (int j = 0; j < 3; ++j)
                            s.scales[j] = static_cast<float>(std::log((with_shape ? 4.0 : 1.5) / s.fx));
                        s.opac[0] = static_cast<float>(std::log(o / (1 - o)));
                        s.motion = {0.0f, 0.3f, 0.0f, 0, 0, 0, 3.0f, 0.0f, 0.0f, 0, 0, 0};
                        s.exposure = static_cast<float>(te);
                        s.pose = false;
                        Scene sh = s;
                        sh.blur = false;
                        const auto rb = render_and_grad(s, false), rs_ = render_and_grad(sh, false);
                        double mb = 0, ms = 0;
                        for (float x : rb.alpha)
                            mb += x;
                        for (float x : rs_.alpha)
                            ms += x;
                        // prediction: a 2D Gaussian of peak q keeps 2 pi sqrt(det) q (1 - e^-hf (1 + hf)) above the floor,
                        // hf = ln(q / floor); the peak is min(q, 0.999) -> ignored (reported)
                        auto kept = [&](double q) { // 2D: the floor cuts e^-hf = floor / q of it (beta 2)
                            if (!with_shape)
                                return std::max(q - 1.0 / 255.0, 0.0);
                            return q > 1.0 / 255.0 ? q * reg_lower_gamma(2.0 / beta, std::log(q * 255.0)) : 0.0;
                        };
                        // kappa from the blurred / sharp peak ratio (pixel-sampled maxima; the prediction depends on it only
                        // through the floor's log term)
                        double pb = 0, ps = 0;
                        for (float x : rb.alpha)
                            pb = std::max(pb, static_cast<double>(x));
                        for (float x : rs_.alpha)
                            ps = std::max(ps, static_cast<double>(x));
                        const double k_est = pb / ps;
                        const double pred = (kept(o * k_est) / k_est) / kept(o);
                        std::println("    {}opacity {:.5f}, exposure {:>4.0f} ms: mass blurred / sharp {:.5f} (predicted from the floor {:.5f}); "
                                     "peak ratio {:.3f} (without the opacity factor the mass would be x{:.3f})",
                                     with_shape ? std::format("beta {:.1f}, ", beta) : std::string(), o, te * 1e3, mb / ms, pred, k_est, 1.0 / k_est);
                        if (o < 0.999)
                            gate(std::fabs(mb / ms - pred) < 0.003, std::format("(C) coverage kept ({}opacity {}, {} ms): within 0.3% of the prediction",
                                                                                 with_shape ? std::format("beta {}, ", beta) : std::string(), o, te * 1e3));
                    }
                }
            }

            // (D) the scene and its renders for blur_ref.py; (a) vs (b)
            {
                Scene s = base;
                s.pose = false;
                const int K = 16;
                Scene sh = s;
                sh.blur = false;
                const auto r_sharp = render_and_grad(sh, false), r_a = render_and_grad(s, false);
                std::vector<double> bi(r_a.img.size(), 0.0), ba(r_a.alpha.size(), 0.0);
                for (int k = 0; k < K; ++k) {
                    Scene sk = sh;
                    sk.time_offset = static_cast<float>(s.exposure * s.scale * ((k + 0.5) / K - 0.5));
                    const auto rk = render_and_grad(sk, false);
                    for (size_t i = 0; i < bi.size(); ++i)
                        bi[i] += rk.img[i] / K;
                    for (size_t i = 0; i < ba.size(); ++i)
                        ba[i] += rk.alpha[i] / K;
                }
                std::vector<float> b_img(bi.begin(), bi.end()), b_alpha(ba.begin(), ba.end());
                write_f32(out_dir / "img_sharp.f32", r_sharp.img.data(), r_sharp.img.size());
                write_f32(out_dir / "alpha_sharp.f32", r_sharp.alpha.data(), r_sharp.alpha.size());
                write_f32(out_dir / "img_a.f32", r_a.img.data(), r_a.img.size());
                write_f32(out_dir / "alpha_a.f32", r_a.alpha.data(), r_a.alpha.size());
                write_f32(out_dir / "img_b.f32", b_img.data(), b_img.size());
                write_f32(out_dir / "alpha_b.f32", b_alpha.data(), b_alpha.size());
                write_f32(out_dir / "means.f32", s.means.data(), s.means.size());
                write_f32(out_dir / "scales.f32", s.scales.data(), s.scales.size());
                write_f32(out_dir / "rots.f32", s.rots.data(), s.rots.size());
                write_f32(out_dir / "opac.f32", s.opac.data(), s.opac.size());
                write_f32(out_dir / "sh0.f32", s.sh0.data(), s.sh0.size());
                write_f32(out_dir / "w2c.f32", s.w2c.data(), s.w2c.size());
                write_f32(out_dir / "motion.f32", s.motion.data(), s.motion.size());
                if (!s.shape.empty()) { // LFQPLCYKM: beta per splat (blur_shape_ref.py)
                    std::vector<float> betas_out;
                    for (float r : s.shape)
                        betas_out.push_back(lfs::core::kshape::beta_from_raw(r));
                    write_f32(out_dir / "shape.f32", betas_out.data(), betas_out.size());
                } else {
                    std::filesystem::remove(out_dir / "shape.f32");
                }
                std::ofstream f(out_dir / "scene.txt");
                f << std::format("W {}\nH {}\nN {}\nfx {:.9g}\nfy {:.9g}\ncx {:.9g}\ncy {:.9g}\nk {:.9g} {:.9g} {:.9g} {:.9g}\ntau {:.9g}\n"
                                 "exposure {:.9g}\nK {}\niters 2\n",
                                 s.w, s.h, s.n, s.fx, s.fy, s.cx, s.cy, s.k[0], s.k[1], s.k[2], s.k[3], s.tau, s.exposure * s.scale, K);
                double e_ab = 0, e_sb = 0;
                for (size_t i = 0; i < bi.size(); ++i) {
                    e_ab = std::max(e_ab, std::fabs(r_a.img[i] - bi[i]));
                    e_sb = std::max(e_sb, std::fabs(r_sharp.img[i] - bi[i]));
                }
                std::println("(D) wrote the scene and renders to {}: max |(a) - (b)| {:.4f}, max |sharp - (b)| {:.4f} (exposure {} ms)",
                             out_dir.string(), e_ab, e_sb, s.exposure * 1e3);
            }
            std::println("BLUR GRADCHECK: {}", bad == 0 ? "ALL PASS" : std::format("{} FAILED", bad));
            return bad == 0 ? 0 : 1;
        }
    } // namespace

    int run_lfe_blur_tool(int argc, const char* const* argv) {
        if (argc >= 6 && std::string(argv[2]) == "render")
            return run_render(argv[3], argv[4], argv[5]);
        if (argc >= 4 && std::string(argv[2]) == "gradcheck")
            return run_gradcheck(argv[3]);
        std::println(stderr, "usage: LichtFeld-Studio lfe-blur render <model.ply> <cameras.txt> <out_dir> | gradcheck <out_dir>");
        return 2;
    }

} // namespace lfs::app
