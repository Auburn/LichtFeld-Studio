/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Developer checks for the standard rasterizer's native fisheye projection.
//   LichtFeld-Studio lfe-fisheye render <model.ply> <cameras.txt> <out_dir>
//     cameras.txt, one camera per line:
//       name width height model fx fy cx cy k1 k2 k3 k4 qw qx qy qz tx ty tz
//     model PINHOLE or FISHEYE (OPENCV_FISHEYE; k = 0 is the equidistant lens); the pose is
//     COLMAP's world-to-camera quaternion and translation. Writes <out_dir>/<name>.f32: float32
//     RGB, row-major H x W x 3, black background, through fast_rasterize (the training path).
//   LichtFeld-Studio lfe-fisheye projcheck
//     The projection maths on the GPU against a double-precision reference.
//   LichtFeld-Studio lfe-fisheye gradcheck
//     Analytic gradients (means, scales, rotations, opacities) of a tiny scene through the full
//     forward/backward against central finite differences of the forward.
//   LichtFeld-Studio lfe-fisheye radialcheck <lens_radial.txt> [--vs-poly]   (LFRL, --lens-radial)
//     The radial lens table on the GPU (projection, Jacobian, Jacobian backward) against the same table in double, at
//     the table camera's full and quarter resolution; --vs-poly: also against the k-polynomial of the table's camera
//     (for a table made from an 8-value lens's own curve: the table machinery reproduces the polynomial); then the
//     analytic gradients through the full forward/backward with the table against central differences (gradcheck).
//   LichtFeld-Studio lfe-fisheye cuberaycheck <lens_radial.txt>   (LFQPLC, --bg-mode cubemap with --lens-radial)
//     The background cubemap's pixel rays (the render kernel's own) projected back through the rasterizer's projection
//     land on their pixel centres, through the table (and miss through the tag polynomial: negative control).
//   LFS_LENS_RADIAL=<lens_radial.txt> LichtFeld-Studio lfe-fisheye render ...: render through the table (the cameras
//     must be the table's camera, as in training with --lens-radial).
//   LichtFeld-Studio lfe-distortion gradcheck
//     The depth distortion loss alone (pinhole and both fisheye lenses): the forward's loss against a
//     double-precision reference, and its analytic gradients against central finite differences. Both
//     distortion depths (distance from the camera, camera z), the plane-crossing exclusion and the min-dist
//     exclusion (round 2), including splats beside the camera at large angles.

#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/loader.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/radial_lens_table.hpp"
#include "training/kernels/bg_cubemap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

        std::array<float, 9> quat_to_rotmat(double qw, double qx, double qy, double qz) {
            const double n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qw /= n;
            qx /= n;
            qy /= n;
            qz /= n;
            return {static_cast<float>(1 - 2 * (qy * qy + qz * qz)), static_cast<float>(2 * (qx * qy - qw * qz)), static_cast<float>(2 * (qx * qz + qw * qy)),
                    static_cast<float>(2 * (qx * qy + qw * qz)), static_cast<float>(1 - 2 * (qx * qx + qz * qz)), static_cast<float>(2 * (qy * qz - qw * qx)),
                    static_cast<float>(2 * (qx * qz - qw * qy)), static_cast<float>(2 * (qy * qz + qw * qx)), static_cast<float>(1 - 2 * (qx * qx + qy * qy))};
        }

        int run_render(const std::filesystem::path& ply, const std::filesystem::path& spec, const std::filesystem::path& out_dir) {
            if (const char* rl = std::getenv("LFS_LENS_RADIAL"); rl != nullptr && *rl) { // --lens-radial (LFRL)
                try {
                    lfs::training::activate_radial_lens(lfs::training::load_radial_lens_table(rl));
                } catch (const std::exception& e) {
                    std::println(stderr, "{}", e.what());
                    return 1;
                }
                std::println("radial lens table: {}", lfs::training::active_radial_lens()->describe());
            }
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
            auto background = Tensor::zeros({3}, Device::CUDA);

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
                const bool fisheye = model_name == "FISHEYE" || model_name == "OPENCV_FISHEYE";
                const auto r = quat_to_rotmat(qw, qx, qy, qz);
                auto R = Tensor::from_vector(std::vector<float>(r.begin(), r.end()), {3, 3}, Device::CPU);
                auto T = Tensor::from_vector(std::vector<float>{static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz)}, {3}, Device::CPU);
                auto radial = fisheye ? Tensor::from_vector(std::vector<float>{static_cast<float>(k1), static_cast<float>(k2), static_cast<float>(k3), static_cast<float>(k4)}, {4}, Device::CPU)
                                      : Tensor::empty({0}, Device::CPU);
                Camera camera(R, T, static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx), static_cast<float>(cy),
                              radial, Tensor::empty({0}, Device::CPU),
                              fisheye ? CameraModelType::FISHEYE : CameraModelType::PINHOLE,
                              name, {}, {}, w, h, uid++);
                lfs::training::RenderOutput out;
                try {
                    out = lfs::training::fast_rasterize(camera, model, background);
                } catch (const std::exception& e) {
                    std::println(stderr, "render {} failed: {}", name, e.what());
                    return 1;
                }
                auto hwc = out.image.permute({1, 2, 0}).contiguous().cpu();
                const auto path = out_dir / (name + ".f32");
                std::ofstream f(path, std::ios::binary);
                f.write(reinterpret_cast<const char*>(hwc.ptr<float>()), static_cast<std::streamsize>(sizeof(float) * 3ull * w * h));
                std::println("{} {}x{} {} -> {}", name, w, h, fisheye ? "fisheye" : "pinhole", path.string());
            }
            return 0;
        }

        // ---------------------------------------------------------------- projection check
        struct RefLens {
            double fx, fy, cx, cy, k[4];
            bool radial = false; // --lens-radial (LFRL): the active radial lens table instead of k (radialcheck)
        };

        // the active radial lens table in double: the same segments (the file's double values) and index rule as core/radial_lens.h
        void table_P(double u, double& P, double& P1) {
            const auto& t = *lfs::training::active_radial_lens();
            const double th = std::sqrt(std::max(u, 0.0));
            if (th < t.th_end) {
                const int i = std::clamp(static_cast<int>(th / t.dth), 0, t.n - 2);
                const double* s = t.segd.data() + 8 * i;
                const double x = (u - s[0]) * s[1];
                P = s[2] + x * (s[3] + x * (s[4] + x * (s[5] + x * (s[6] + x * s[7]))));
                P1 = (s[3] + x * (2.0 * s[4] + x * (3.0 * s[5] + x * (4.0 * s[6] + x * 5.0 * s[7])))) * s[1];
            } else {
                P = t.A / th + t.B;
                P1 = -t.A / (2.0 * th * th * th);
            }
        }

        double theta_d(const RefLens& L, double t) {
            if (L.radial) {
                double P, P1;
                table_P(t * t, P, P1);
                return t * P;
            }
            const double t2 = t * t;
            return t * (1 + t2 * (L.k[0] + t2 * (L.k[1] + t2 * (L.k[2] + t2 * L.k[3]))));
        }
        double theta_d1(const RefLens& L, double t) {
            if (L.radial) {
                double P, P1;
                table_P(t * t, P, P1);
                return P + 2.0 * t * t * P1;
            }
            const double t2 = t * t;
            return 1 + t2 * (3 * L.k[0] + t2 * (5 * L.k[1] + t2 * (7 * L.k[2] + t2 * 9 * L.k[3])));
        }

        // Reference projection, straight from the definition.
        std::array<double, 2> ref_project(const RefLens& L, const std::array<double, 3>& p) {
            const double rho = std::hypot(p[0], p[1]);
            if (rho < 1e-300)
                return {L.cx, L.cy};
            const double th = std::atan2(rho, p[2]);
            const double s = theta_d(L, th) / rho;
            return {L.fx * s * p[0] + L.cx, L.fy * s * p[1] + L.cy};
        }

        // Reference Jacobian in double: the closed form away from the axis, its limit on it.
        std::array<double, 6> ref_jacobian(const RefLens& L, const std::array<double, 3>& p) {
            const double x = p[0], y = p[1], z = p[2];
            const double rho = std::hypot(x, y);
            const double r2 = rho * rho + z * z;
            if (rho < 1e-7 * std::sqrt(r2)) {
                // on the axis: theta_d / rho -> theta_d'(0)/z (1/z for OPENCV_FISHEYE; P(0)/z for a radial lens table),
                // its rho-derivative vanishes, d/dz = -theta_d'(0)/z^2 (times x, y = 0)
                const double s0 = theta_d1(L, 0) / z;
                return {L.fx * s0, 0, L.fx * x * (-theta_d1(L, 0) / r2), 0, L.fy * s0, L.fy * y * (-theta_d1(L, 0) / r2)};
            }
            const double th = std::atan2(rho, z);
            const double D = theta_d(L, th), D1 = theta_d1(L, th);
            const double s = D / rho;
            const double s_rho = (D1 * z / r2 - s) / rho; // d s / d rho
            const double s_z = -D1 / r2;
            return {L.fx * (s + s_rho * x * x / rho), L.fx * s_rho * x * y / rho, L.fx * x * s_z,
                    L.fy * s_rho * x * y / rho, L.fy * (s + s_rho * y * y / rho), L.fy * y * s_z};
        }

        struct ProjCase {
            std::string name;
            RefLens lens;     // the reference (double)
            RefLens gpu = {}; // the GPU's lens if gpu_set (radialcheck: the table against the polynomial)
            bool gpu_set = false;
        };
        int projcheck_cases(const std::vector<ProjCase>& cases, const char* ok_word);

        int run_projcheck() {
            const std::vector<ProjCase> cases = {
                {"game equidistant (fx 1232.6, k 0)", {1232.6, 1232.6, 1280, 720, {0, 0, 0, 0}}},
                {"woodland quarter-res (k 0.391 -0.039 0.0024 -0.0027)", {292.43, 296.08, 512, 384, {0.39127139, -0.03901020, 0.00243290, -0.00266145}}},
            };
            return projcheck_cases(cases, "PROJCHECK");
        }

        int projcheck_cases(const std::vector<ProjCase>& cases, const char* ok_word) {
            std::mt19937 rng(7);
            std::uniform_real_distribution<double> uni(-1.0, 1.0);
            const std::vector<double> thetas_deg = {0.0, 1e-7, 1e-4, 0.01, 0.5, 5, 15, 17.4, 17.6, 25, 35, 45, 55, 65, 70, 75, 80, 83, 85, 87, 88};
            int failures = 0;
            for (const auto& c : cases) {
                fr::FastGSLens lens;
                const RefLens& gl = c.gpu_set ? c.gpu : c.lens;
                lens.model = fr::kLensFisheye;
                lens.k1 = static_cast<float>(gl.k[0]);
                lens.k2 = static_cast<float>(gl.k[1]);
                lens.k3 = static_cast<float>(gl.k[2]);
                lens.k4 = static_cast<float>(gl.k[3]);
                if (gl.radial)
                    lens.radial = lfs::training::active_radial_lens()->device_view();
                std::vector<float> in;
                std::vector<std::array<double, 3>> pts;
                std::vector<std::array<double, 6>> gs;
                std::vector<double> th_of;
                for (const double tdeg : thetas_deg) {
                    for (int rep = 0; rep < 8; ++rep) {
                        const double th = tdeg * std::numbers::pi / 180.0;
                        const double az = 2 * std::numbers::pi * (rep + 0.37) / 8.0;
                        const double dist = std::pow(10.0, uni(rng)); // 0.1 .. 10
                        // the exact axis for the first repeat of theta 0
                        std::array<double, 3> p = {dist * std::sin(th) * std::cos(az), dist * std::sin(th) * std::sin(az), dist * std::cos(th)};
                        // round to float first: the GPU sees the float point, so the reference must too
                        for (auto& v : p)
                            v = static_cast<float>(v);
                        std::array<double, 6> g;
                        for (auto& v : g)
                            v = uni(rng);
                        pts.push_back(p);
                        gs.push_back(g);
                        th_of.push_back(tdeg);
                        for (int i = 0; i < 3; ++i)
                            in.push_back(static_cast<float>(p[i]));
                        for (int i = 0; i < 6; ++i)
                            in.push_back(static_cast<float>(g[i]));
                    }
                }
                const int n = static_cast<int>(pts.size());
                std::vector<float> out(17ull * n);
                fr::fisheye_projection_selftest(lens, static_cast<float>(c.lens.fx), static_cast<float>(c.lens.fy),
                                                static_cast<float>(c.lens.cx), static_cast<float>(c.lens.cy), in.data(), n, out.data());
                std::println("== {}", c.name);
                std::println("  theta_deg  max|du| px   max relerr J   max relerr dJ-backward");
                double worst_mean = 0, worst_j = 0, worst_jb = 0;
                for (size_t ti = 0; ti < thetas_deg.size(); ++ti) {
                    double em = 0, ej = 0, ejb = 0;
                    for (int rep = 0; rep < 8; ++rep) {
                        const int i = static_cast<int>(ti) * 8 + rep;
                        const auto& p = pts[i];
                        const auto& g = gs[i];
                        const float* o = out.data() + 17ull * i;
                        const auto m = ref_project(c.lens, p);
                        em = std::max({em, std::abs(o[0] - m[0]), std::abs(o[1] - m[1])});
                        const auto J = ref_jacobian(c.lens, p);
                        double jn = 0;
                        for (double v : J)
                            jn = std::max(jn, std::abs(v));
                        for (int k = 0; k < 6; ++k)
                            ej = std::max(ej, std::abs(o[2 + k] - J[k]) / jn);
                        // d/dp of S(p) = G1 . J1(p) + G2 . J2(p), central differences of the reference Jacobian
                        const double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                        const double hstep = 1e-5 * r;
                        std::array<double, 3> ref_jb{};
                        for (int a = 0; a < 3; ++a) {
                            auto pp = p, pm = p;
                            pp[a] += hstep;
                            pm[a] -= hstep;
                            const auto Jp = ref_jacobian(c.lens, pp), Jm = ref_jacobian(c.lens, pm);
                            double sp = 0, sm = 0;
                            for (int k = 0; k < 6; ++k) {
                                sp += g[k] * Jp[k];
                                sm += g[k] * Jm[k];
                            }
                            ref_jb[a] = (sp - sm) / (2 * hstep);
                        }
                        // scale: |G| |J| / r, the natural size of this term
                        const double scale = jn / r;
                        for (int a = 0; a < 3; ++a)
                            ejb = std::max(ejb, std::abs(o[8 + a] - ref_jb[a]) / scale);
                        for (int k = 0; k < 17; ++k)
                            if (!std::isfinite(o[k]))
                                ++failures;
                    }
                    std::println("  {:9.4g}  {:10.3g}   {:12.3g}   {:12.3g}", thetas_deg[ti], em, ej, ejb);
                    worst_mean = std::max(worst_mean, em);
                    worst_j = std::max(worst_j, ej);
                    worst_jb = std::max(worst_jb, ejb);
                }
                std::println("  worst: centre {:.3g} px, Jacobian {:.3g} (relative), Jacobian backward {:.3g} (relative to |J|/r)",
                             worst_mean, worst_j, worst_jb);
                if (worst_mean > 0.05 || worst_j > 1e-4 || worst_jb > 1e-3)
                    ++failures;
            }
            if (failures == 0)
                std::println("{} OK", ok_word);
            else
                std::println("{} FAILED ({} problems)", ok_word, failures);
            return failures == 0 ? 0 : 1;
        }

        // ---------------------------------------------------------------- gradient check
        struct DevBuf {
            float* p = nullptr;
            size_t n = 0;
            explicit DevBuf(size_t count) : n(count) {
                cudaMalloc(&p, sizeof(float) * std::max<size_t>(count, 1));
                cudaMemset(p, 0, sizeof(float) * std::max<size_t>(count, 1));
            }
            ~DevBuf() { cudaFree(p); }
            void upload(const std::vector<float>& v) { cudaMemcpy(p, v.data(), sizeof(float) * v.size(), cudaMemcpyHostToDevice); }
            std::vector<float> download() const {
                std::vector<float> v(n);
                cudaMemcpy(v.data(), p, sizeof(float) * n, cudaMemcpyDeviceToHost);
                return v;
            }
            void zero() { cudaMemset(p, 0, sizeof(float) * n); }
        };

        struct Scene {
            int n = 0, w = 0, h = 0;
            float fx, fy, cx, cy;
            fr::FastGSLens lens;
            std::vector<float> means, scales, rots, opac, sh0, w2c, campos, weights;
            std::vector<double> theta_deg; // per splat, for the report
        };

        struct GradRun {
            double loss = 0;
            std::vector<float> g_means, g_scales, g_rots, g_opac;
            int n_visible = 0;
            int n_left_out = 0;  // distortion check: visible splats the GPU left out of the term (both reasons)
            int n_left_near = 0; // of which nearer than min_dist
        };

        GradRun render_and_grad(const Scene& s, bool with_backward) {
            const int N = s.n;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_w2c(16), d_cam(3);
            DevBuf d_img(3ull * s.w * s.h), d_alpha(1ull * s.w * s.h), d_depth(1ull * s.w * s.h), d_weights(3ull * s.w * s.h);
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_w2c.upload(s.w2c);
            d_cam.upload(s.campos);
            d_weights.upload(s.weights);
            GradRun run;
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, nullptr, d_w2c.p, d_cam.p,
                                       d_img.p, d_alpha.p, d_depth.p, nullptr, nullptr, nullptr, N, 1, 1, s.w, s.h,
                                       s.fx, s.fy, s.cx, s.cy, 0.01f, 1e10f, false, nullptr, nullptr, 0, 0, nullptr, 1.0f, s.lens);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            run.n_visible = ctx.n_visible;
            const auto img = d_img.download();
            double loss = 0;
            for (size_t i = 0; i < img.size(); ++i)
                loss += static_cast<double>(img[i]) * s.weights[i];
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
                                              s.fx, s.fy, s.cx, s.cy, false, DensificationType::None, &fa);
            cudaDeviceSynchronize();
            if (!res.success) {
                std::println(stderr, "backward failed: {}", res.error_message ? res.error_message : "?");
                std::exit(1);
            }
            run.g_means = g_means.download();
            run.g_scales = g_scales.download();
            run.g_rots = g_rots.download();
            run.g_opac = g_opac.download();
            return run;
        }

        // local = true: dL/dimage is a smooth bump inside each splat's 2.5-sigma ellipse (Mahalanobis, from the reference
        // projection and Jacobian), with a slope across it, and zero elsewhere. The rasterizer cuts each splat off where
        // its alpha falls under 1/255 (>= 3 sigma here); the analytic gradient has no term for that edge moving, but a
        // finite difference sees it. Keeping the weights inside the cut-off isolates the projection maths.
        // local = false: smooth weights over the whole image (edge terms included, the rasterizer's own noise floor).
        Scene make_scene(bool fisheye, const RefLens& L, int w, int h, const std::vector<double>& thetas_deg, unsigned seed,
                         bool local, int only = -1) {
            Scene s;
            s.w = w;
            s.h = h;
            s.fx = static_cast<float>(L.fx);
            s.fy = static_cast<float>(L.fy);
            s.cx = static_cast<float>(L.cx);
            s.cy = static_cast<float>(L.cy);
            if (fisheye) {
                s.lens.model = fr::kLensFisheye;
                s.lens.k1 = static_cast<float>(L.k[0]);
                s.lens.k2 = static_cast<float>(L.k[1]);
                s.lens.k3 = static_cast<float>(L.k[2]);
                s.lens.k4 = static_cast<float>(L.k[3]);
                const double tmax = std::tan(88.0 * std::numbers::pi / 180.0);
                s.lens.max_w = static_cast<float>(tmax * tmax);
                if (L.radial)
                    s.lens.radial = lfs::training::active_radial_lens()->device_view();
            }
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            // camera: a random rotation and position, so W and its transpose take part
            const double a1 = 2 * std::numbers::pi * uni(rng), a2 = std::acos(2 * uni(rng) - 1), a3 = 2 * std::numbers::pi * uni(rng);
            const auto R = quat_to_rotmat(std::cos(a1 / 2), std::sin(a1 / 2) * std::sin(a2) * std::cos(a3),
                                          std::sin(a1 / 2) * std::sin(a2) * std::sin(a3), std::sin(a1 / 2) * std::cos(a2));
            const std::array<double, 3> C = {uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1}; // camera centre
            // w2c = [R | -R C]
            std::array<double, 3> t{};
            for (int i = 0; i < 3; ++i)
                t[i] = -(R[i * 3] * C[0] + R[i * 3 + 1] * C[1] + R[i * 3 + 2] * C[2]);
            s.w2c = {R[0], R[1], R[2], static_cast<float>(t[0]), R[3], R[4], R[5], static_cast<float>(t[1]),
                     R[6], R[7], R[8], static_cast<float>(t[2]), 0, 0, 0, 1};
            s.campos = {static_cast<float>(C[0]), static_cast<float>(C[1]), static_cast<float>(C[2])};
            for (size_t i = 0; i < thetas_deg.size(); ++i) {
                const double th = thetas_deg[i] * std::numbers::pi / 180.0;
                // an azimuth that keeps the projected centre inside the frame
                double az = 2 * std::numbers::pi * uni(rng);
                const double dist = 2.0 + 2.0 * uni(rng);
                std::array<double, 3> pc = {dist * std::sin(th) * std::cos(az), dist * std::sin(th) * std::sin(az), dist * std::cos(th)};
                for (int tries = 0; tries < 4000; ++tries) {
                    const auto m = fisheye ? ref_project(L, pc) : std::array<double, 2>{L.fx * pc[0] / pc[2] + L.cx, L.fy * pc[1] / pc[2] + L.cy};
                    if (m[0] > 10 && m[0] < w - 10 && m[1] > 10 && m[1] < h - 10)
                        break;
                    if (tries == 3999)
                        std::println("  (theta {} deg: no azimuth puts the centre inside the frame)", thetas_deg[i]);
                    az = 2 * std::numbers::pi * uni(rng);
                    pc = {dist * std::sin(th) * std::cos(az), dist * std::sin(th) * std::sin(az), dist * std::cos(th)};
                }
                if (thetas_deg[i] == 0.0)
                    pc = {0.0, 0.0, dist}; // exactly on the axis
                // world = R^T (pc - t)
                std::array<double, 3> pw{};
                for (int j = 0; j < 3; ++j)
                    pw[j] = R[0 * 3 + j] * (pc[0] - t[0]) + R[1 * 3 + j] * (pc[1] - t[1]) + R[2 * 3 + j] * (pc[2] - t[2]);
                for (double v : pw)
                    s.means.push_back(static_cast<float>(v));
                // angular size 2-5 degrees per axis, anisotropic
                for (int j = 0; j < 3; ++j)
                    s.scales.push_back(static_cast<float>(std::log(dist * (0.035 + 0.05 * uni(rng)))));
                std::array<double, 4> q = {uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1};
                for (double v : q)
                    s.rots.push_back(static_cast<float>(v));
                s.opac.push_back(static_cast<float>(-0.5 + 1.5 * uni(rng)));          // sigmoid 0.38 .. 0.73
                for (int j = 0; j < 3; ++j)
                    s.sh0.push_back(static_cast<float>((uni(rng) - 0.5) * 1.5)); // colour 0.29 .. 0.71
                s.theta_deg.push_back(thetas_deg[i]);
            }
            s.n = static_cast<int>(thetas_deg.size());
            if (only >= 0) { // this splat alone (the same random draw as the full scene)
                auto keep = [only](std::vector<float>& v, int k) { v = std::vector<float>(v.begin() + only * k, v.begin() + (only + 1) * k); };
                keep(s.means, 3);
                keep(s.scales, 3);
                keep(s.rots, 4);
                keep(s.opac, 1);
                keep(s.sh0, 3);
                s.theta_deg = {s.theta_deg[only]};
                s.n = 1;
            }
            s.weights.assign(3ull * w * h, 0.0f);
            if (!local) {
                for (int c = 0; c < 3; ++c)
                    for (int y = 0; y < h; ++y)
                        for (int x = 0; x < w; ++x)
                            s.weights[(static_cast<size_t>(c) * h + y) * w + x] = static_cast<float>(
                                std::sin(2 * std::numbers::pi * x / (37.0 + 11 * c)) * std::cos(2 * std::numbers::pi * y / (29.0 + 7 * c)) + 0.25);
                for (auto& v : s.weights)
                    v *= 1e-3f;
                return s;
            }
            for (int i = 0; i < s.n; ++i) {
                // camera-space centre and 3D covariance
                std::array<double, 3> pw = {s.means[i * 3], s.means[i * 3 + 1], s.means[i * 3 + 2]};
                std::array<double, 3> pc{};
                for (int r = 0; r < 3; ++r)
                    pc[r] = s.w2c[r * 4] * pw[0] + s.w2c[r * 4 + 1] * pw[1] + s.w2c[r * 4 + 2] * pw[2] + s.w2c[r * 4 + 3];
                const auto Rq = quat_to_rotmat(s.rots[i * 4], s.rots[i * 4 + 1], s.rots[i * 4 + 2], s.rots[i * 4 + 3]);
                double cov[3][3] = {};
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        for (int k = 0; k < 3; ++k)
                            cov[a][b] += Rq[a * 3 + k] * std::exp(2.0 * s.scales[i * 3 + k]) * Rq[b * 3 + k];
                // JW (2x3): pinhole or fisheye Jacobian times the camera rotation
                std::array<double, 6> J;
                std::array<double, 2> m;
                if (fisheye) {
                    J = ref_jacobian(L, pc);
                    m = ref_project(L, pc);
                } else {
                    J = {L.fx / pc[2], 0, -L.fx * pc[0] / (pc[2] * pc[2]), 0, L.fy / pc[2], -L.fy * pc[1] / (pc[2] * pc[2])};
                    m = {L.fx * pc[0] / pc[2] + L.cx, L.fy * pc[1] / pc[2] + L.cy};
                }
                double JW[2][3] = {};
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 3; ++c)
                        for (int k = 0; k < 3; ++k)
                            JW[r][c] += J[r * 3 + k] * s.w2c[k * 4 + c];
                double C[2][2] = {};
                for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 2; ++c)
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                C[r][c] += JW[r][a] * cov[a][b] * JW[c][b];
                C[0][0] += 0.3;
                C[1][1] += 0.3;
                const double det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
                const double ia = C[1][1] / det, ib = -C[0][1] / det, ic = C[0][0] / det;
                const double sx = std::sqrt(C[0][0]), sy = std::sqrt(C[1][1]);
                const int x0 = std::max(0, static_cast<int>(m[0] - 3 * sx)), x1 = std::min(w - 1, static_cast<int>(m[0] + 3 * sx) + 1);
                const int y0 = std::max(0, static_cast<int>(m[1] - 3 * sy)), y1 = std::min(h - 1, static_cast<int>(m[1] + 3 * sy) + 1);
                for (int y = y0; y <= y1; ++y)
                    for (int x = x0; x <= x1; ++x) {
                        const double dx = x + 0.5 - m[0], dy = y + 0.5 - m[1];
                        const double d2 = ia * dx * dx + 2 * ib * dx * dy + ic * dy * dy;
                        if (d2 >= 6.25)
                            continue;
                        const double bump = (1 - d2 / 6.25) * (1 - d2 / 6.25);
                        for (int c = 0; c < 3; ++c)
                            s.weights[(static_cast<size_t>(c) * h + y) * w + x] += static_cast<float>(
                                bump * (1.0 + 0.6 * dx / sx * (c == 1 ? -1.0 : 1.0) - 0.4 * dy / sy + 0.3 * c));
                    }
            }
            for (auto& v : s.weights) // dL/dimage at the scale of a mean loss: the backward clamps per-warp partials at 1e4
                v *= 1e-3f;
            return s;
        }

        struct GradRow {
            int group;
            double theta, cosine, rel, norm, fd_norm;
        };

        void gradcheck_scene(Scene s, std::vector<GradRow>& rows, int& visible) {
            const auto base = render_and_grad(s, true);
            visible += base.n_visible;
            struct Group {
                std::vector<float>* param;
                const std::vector<float>* grad;
                int attrs;
                float step;
            };
            Group groups[] = {
                {&s.means, &base.g_means, 3, 0.0f},
                {&s.scales, &base.g_scales, 3, 2e-3f},
                {&s.rots, &base.g_rots, 4, 2e-3f},
                {&s.opac, &base.g_opac, 1, 2e-3f},
            };
            for (int gi = 0; gi < 4; ++gi) {
                auto& g = groups[gi];
                for (int i = 0; i < s.n; ++i) {
                    std::vector<double> fd(g.attrs), an(g.attrs);
                    for (int a = 0; a < g.attrs; ++a) {
                        const size_t idx = static_cast<size_t>(i) * g.attrs + a;
                        static const float step_mult = std::getenv("LFE_GC_STEP") ? static_cast<float>(std::atof(std::getenv("LFE_GC_STEP"))) : 1.0f;
                        float step = g.step * step_mult;
                        if (step == 0.0f) {
                            // means: a fixed fraction of the distance to the camera
                            const double dx = s.means[i * 3] - s.campos[0], dy = s.means[i * 3 + 1] - s.campos[1], dz = s.means[i * 3 + 2] - s.campos[2];
                            step = static_cast<float>(5e-4 * step_mult * std::sqrt(dx * dx + dy * dy + dz * dz));
                        }
                        const float v0 = (*g.param)[idx];
                        (*g.param)[idx] = v0 + step;
                        const double lp = render_and_grad(s, false).loss;
                        (*g.param)[idx] = v0 - step;
                        const double lm = render_and_grad(s, false).loss;
                        (*g.param)[idx] = v0;
                        const double h2 = static_cast<double>(v0 + step) - static_cast<double>(v0 - step);
                        fd[a] = (lp - lm) / h2;
                        an[a] = (*g.grad)[idx];
                    }
                    double dot = 0, nf = 0, na = 0, nd = 0;
                    for (int a = 0; a < g.attrs; ++a) {
                        dot += fd[a] * an[a];
                        nf += fd[a] * fd[a];
                        na += an[a] * an[a];
                        nd += (fd[a] - an[a]) * (fd[a] - an[a]);
                    }
                    if (std::getenv("LFE_GC_VERBOSE") && std::sqrt(nd) > 0.01 * std::sqrt(nf)) {
                        std::string line;
                        for (int a = 0; a < g.attrs; ++a)
                            line += std::format(" [fd {:.6g} an {:.6g}]", fd[a], an[a]);
                        std::println("    group {} theta {}:{}", gi, s.theta_deg[i], line);
                    }
                    rows.push_back({gi, s.theta_deg[i], dot / std::sqrt(std::max(nf * na, 1e-300)),
                                    std::sqrt(nd) / std::max(std::sqrt(nf), 1e-12), std::sqrt(na), std::sqrt(nf)});
                }
            }
        }

        // threshold: relative error (|fd - analytic| / |fd|) above which a parameter block counts as bad
        int run_gradcheck_case(const char* title, const std::vector<Scene>& scenes, double threshold) {
            std::vector<GradRow> rows;
            int visible = 0, total = 0;
            for (const auto& sc : scenes) {
                gradcheck_scene(sc, rows, visible);
                total += sc.n;
            }
            std::println("== {}: {} splats in {} render(s), {} visible, {}x{}", title, total, scenes.size(), visible,
                         scenes[0].w, scenes[0].h);
            const char* names[] = {"means", "scales", "rotations", "opacities"};
            std::println("  {:>10} {:>8} {:>10} {:>10} {:>10}", "param", "theta", "cosine", "rel err", "|grad|");
            int bad = 0;
            std::vector<double> all_rel;
            for (int gi = 0; gi < 4; ++gi) {
                std::vector<double> group_rel;
                for (const auto& r : rows) {
                    if (r.group != gi)
                        continue;
                    std::println("  {:>10} {:8.3g} {:10.6f} {:10.2e} {:10.3g}", names[gi], r.theta, r.cosine, r.rel, r.norm);
                    group_rel.push_back(r.rel);
                    all_rel.push_back(r.rel);
                    if (r.fd_norm > 1e-6 && r.rel > threshold)
                        ++bad;
                }
                std::sort(group_rel.begin(), group_rel.end());
                std::println("  {:>10}: median rel err {:.2e}, max {:.2e}", names[gi], group_rel[group_rel.size() / 2], group_rel.back());
            }
            std::sort(all_rel.begin(), all_rel.end());
            std::println("  all: median rel err {:.2e}, 90th pct {:.2e}, max {:.2e}; {} over {:g}%",
                         all_rel[all_rel.size() / 2], all_rel[all_rel.size() * 9 / 10], all_rel.back(), bad, 100 * threshold);
            return bad;
        }

        int run_gradcheck() {
            int bad = 0;
            const RefLens game_q = {308.15, 308.15, 320, 180, {0, 0, 0, 0}}; // the game's lens at 640x360 (1232.6 / 4)
            const RefLens wood_q = {292.43, 296.08, 512, 384, {0.39127139, -0.03901020, 0.00243290, -0.00266145}};
            const RefLens pin = {300.0, 300.0, 320, 240, {0, 0, 0, 0}};
            const std::vector<double> fish_thetas = {0.0, 0.002, 5, 15, 17.5, 25, 35, 45, 55, 60, 64};
            const std::vector<double> wood_thetas = {0.0, 0.002, 5, 17.5, 30, 45, 55, 62, 68, 72, 75, 77};
            const std::vector<double> pin_thetas = {0.0, 0.002, 5, 15, 17.5, 25, 30, 35, 40, 42, 44};
            struct Setup {
                const char* title;
                bool fisheye;
                RefLens lens;
                int w, h;
                const std::vector<double>* thetas;
                unsigned seed;
            };
            const Setup setups[] = {
                {"control: pinhole 640x480 (unchanged path)", false, pin, 640, 480, &pin_thetas, 11},
                {"fisheye: the game's equidistant lens at 640x360", true, game_q, 640, 360, &fish_thetas, 12},
                {"fisheye: woodland OPENCV_FISHEYE lens at 1024x768", true, wood_q, 1024, 768, &wood_thetas, 13},
            };
            std::println("######## each splat rendered alone, weights inside its 2.5-sigma ellipse: isolates the projection maths");
            for (const auto& st : setups) {
                std::vector<Scene> scenes;
                for (int i = 0; i < static_cast<int>(st.thetas->size()); ++i)
                    scenes.push_back(make_scene(st.fisheye, st.lens, st.w, st.h, *st.thetas, st.seed, true, i));
                bad += run_gradcheck_case(st.title, scenes, 0.01);
            }
            std::println("######## all splats together, smooth weights over the whole image: includes overlaps and the alpha");
            std::println("######## cut-off edge (no analytic gradient has a term for it), i.e. the rasterizer's own noise floor");
            for (const auto& st : setups)
                (void)run_gradcheck_case(st.title, {make_scene(st.fisheye, st.lens, st.w, st.h, *st.thetas, st.seed, false)}, 0.1);
            if (bad == 0)
                std::println("GRADCHECK OK");
            else
                std::println("GRADCHECK: {} parameter blocks over 1% relative error (local weights)", bad);
            return bad == 0 ? 0 : 1;
        }

        // ---------------------------------------------------------------- radial lens table check (LFRL)
        int run_radialcheck(const std::filesystem::path& file, const bool vs_poly) {
            try {
                lfs::training::activate_radial_lens(lfs::training::load_radial_lens_table(file));
            } catch (const std::exception& e) {
                std::println(stderr, "{}", e.what());
                return 1;
            }
            const auto& t = *lfs::training::active_radial_lens();
            std::println("radial lens table: {}", t.describe());
            const RefLens full = {t.fx, t.fy, t.cx, t.cy, {t.k[0], t.k[1], t.k[2], t.k[3]}, true};
            const RefLens quarter = {t.fx / 4, t.fy / 4, t.cx / 4, t.cy / 4, {t.k[0], t.k[1], t.k[2], t.k[3]}, true};
            std::vector<ProjCase> cases = {
                {std::format("table on the GPU vs the table in double, {}x{}", t.width, t.height), full},
                {std::format("table on the GPU vs the table in double, {}x{}", t.width / 4, t.height / 4), quarter},
            };
            int bad = projcheck_cases(cases, "RADIAL PROJCHECK (table vs table)");
            if (vs_poly) {
                RefLens poly = full, polyq = quarter;
                poly.radial = polyq.radial = false;
                const std::vector<ProjCase> pc = {
                    {std::format("table on the GPU vs the k-polynomial in double, {}x{}", t.width, t.height), poly, full, true},
                    {std::format("table on the GPU vs the k-polynomial in double, {}x{}", t.width / 4, t.height / 4), polyq, quarter, true},
                };
                bad += projcheck_cases(pc, "RADIAL PROJCHECK (table vs polynomial)");
            }
            const std::vector<double> thetas = {0.0, 0.002, 0.2, 5, 17.5, 30, 45, 55, 62, 68, 72, 75, 77};
            std::println("######## gradients through the full forward/backward with the table (each splat alone, weights inside its 2.5-sigma ellipse)");
            std::vector<Scene> scenes;
            for (int i = 0; i < static_cast<int>(thetas.size()); ++i)
                scenes.push_back(make_scene(true, quarter, t.width / 4, t.height / 4, thetas, 21, true, i));
            const int gbad = run_gradcheck_case("fisheye: the radial lens table at quarter res", scenes, 0.01);
            (void)run_gradcheck_case("fisheye: the radial lens table, all splats, smooth weights (noise floor)",
                                     {make_scene(true, quarter, t.width / 4, t.height / 4, thetas, 21, false)}, 0.1);
            if (gbad == 0)
                std::println("RADIAL GRADCHECK OK");
            else
                std::println("RADIAL GRADCHECK: {} parameter blocks over 1% relative error", gbad);
            return bad == 0 && gbad == 0 ? 0 : 1;
        }

        // ---------------------------------------------------------------- depth distortion check
        // The depth distortion loss alone (no image loss): clusters of overlapping splats at different
        // depths, the forward's loss against a double-precision reference, and the backward's gradients
        // (means, scales, rotations, opacities) against central differences of the forward's loss.
        struct Footprint {
            double mx = 0, my = 0;         // projected centre (px)
            double ia = 0, ib = 0, ic = 0; // inverse 2D covariance, 0.3 px dilation included
            double depth = 0;              // the rasterizer's per-splat (sort) depth: camera z, or distance (fisheye)
            double distortion_depth = 0;   // the distortion term's depth; < 0 = left out (plane-crossing or min-dist)
            double plane_ratio = 0;        // z / (plane_sigma sigma_z): < 1 = crosses the camera plane
            double near_ratio = 1e30;      // distortion depth / min_dist: < 1 = nearer than min_dist (1e30 = min-dist off)
            int left_out = 0;              // 0 = taking part, 1 = plane-crossing, 2 = nearer than min_dist
            double opacity = 0;
        };

        // How the distortion term picks its depth (FastGSDistortion): distance from the camera or the sort
        // depth (camera z on the pinhole path), the plane-crossing exclusion (0 = off) and the min-dist
        // exclusion (0 = off).
        struct DistortionMode {
            bool depth_distance = true;
            float plane_sigma = 2.0f;
            float min_dist = 0.0f;
        };

        Footprint splat_footprint(const Scene& s, const RefLens& L, bool fisheye, int i, const DistortionMode& dm = {}) {
            Footprint f;
            const std::array<double, 3> pw = {s.means[i * 3], s.means[i * 3 + 1], s.means[i * 3 + 2]};
            std::array<double, 3> pc{};
            for (int r = 0; r < 3; ++r)
                pc[r] = s.w2c[r * 4] * pw[0] + s.w2c[r * 4 + 1] * pw[1] + s.w2c[r * 4 + 2] * pw[2] + s.w2c[r * 4 + 3];
            const auto Rq = quat_to_rotmat(s.rots[i * 4], s.rots[i * 4 + 1], s.rots[i * 4 + 2], s.rots[i * 4 + 3]);
            double cov[3][3] = {};
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    for (int k = 0; k < 3; ++k)
                        cov[a][b] += Rq[a * 3 + k] * std::exp(2.0 * s.scales[i * 3 + k]) * Rq[b * 3 + k];
            std::array<double, 6> J;
            std::array<double, 2> m;
            if (fisheye) {
                J = ref_jacobian(L, pc);
                m = ref_project(L, pc);
            } else {
                J = {L.fx / pc[2], 0, -L.fx * pc[0] / (pc[2] * pc[2]), 0, L.fy / pc[2], -L.fy * pc[1] / (pc[2] * pc[2])};
                m = {L.fx * pc[0] / pc[2] + L.cx, L.fy * pc[1] / pc[2] + L.cy};
            }
            double JW[2][3] = {};
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 3; ++c)
                    for (int k = 0; k < 3; ++k)
                        JW[r][c] += J[r * 3 + k] * s.w2c[k * 4 + c];
            double C[2][2] = {};
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c)
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            C[r][c] += JW[r][a] * cov[a][b] * JW[c][b];
            C[0][0] += 0.3;
            C[1][1] += 0.3;
            const double det = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            f.mx = m[0];
            f.my = m[1];
            f.ia = C[1][1] / det;
            f.ib = -C[0][1] / det;
            f.ic = C[0][0] / det;
            const double dist = std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
            f.depth = fisheye ? dist : pc[2];
            f.distortion_depth = (fisheye || dm.depth_distance) ? dist : pc[2];
            // variance along the camera axis a = w2c row 3: a^T Sigma a
            double var_z = 0;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    var_z += s.w2c[8 + a] * cov[a][b] * s.w2c[8 + b];
            f.plane_ratio = dm.plane_sigma > 0 ? pc[2] / (dm.plane_sigma * std::sqrt(var_z)) : 1e30;
            f.near_ratio = dm.min_dist > 0 ? f.distortion_depth / dm.min_dist : 1e30;
            if (f.plane_ratio < 1.0) {
                f.distortion_depth = -1.0;
                f.left_out = 1;
            } else if (f.near_ratio < 1.0) {
                f.distortion_depth = -1.0;
                f.left_out = 2;
            }
            f.opacity = 1.0 / (1.0 + std::exp(-static_cast<double>(s.opac[i])));
            return f;
        }

        double footprint_d2(const Footprint& f, double px, double py) {
            const double dx = px - f.mx, dy = py - f.my;
            return f.ia * dx * dx + 2 * f.ib * dx * dy + f.ic * dy * dy;
        }

        // Clusters of 4 splats along a few view directions, at the given distances (default 1.2 to 4 units),
        // overlapping on screen.
        Scene make_distortion_scene(bool fisheye, const RefLens& L, int w, int h, unsigned seed,
                                    const std::vector<std::array<double, 2>>& dirs_deg, double ang_sigma,
                                    const std::array<double, 4>& cluster_depths = {1.2, 1.9, 2.8, 4.0}) {
            Scene s;
            s.w = w;
            s.h = h;
            s.fx = static_cast<float>(L.fx);
            s.fy = static_cast<float>(L.fy);
            s.cx = static_cast<float>(L.cx);
            s.cy = static_cast<float>(L.cy);
            if (fisheye) {
                s.lens.model = fr::kLensFisheye;
                s.lens.k1 = static_cast<float>(L.k[0]);
                s.lens.k2 = static_cast<float>(L.k[1]);
                s.lens.k3 = static_cast<float>(L.k[2]);
                s.lens.k4 = static_cast<float>(L.k[3]);
                const double tmax = std::tan(88.0 * std::numbers::pi / 180.0);
                s.lens.max_w = static_cast<float>(tmax * tmax);
            }
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            const double a1 = 2 * std::numbers::pi * uni(rng), a2 = std::acos(2 * uni(rng) - 1), a3 = 2 * std::numbers::pi * uni(rng);
            const auto R = quat_to_rotmat(std::cos(a1 / 2), std::sin(a1 / 2) * std::sin(a2) * std::cos(a3),
                                          std::sin(a1 / 2) * std::sin(a2) * std::sin(a3), std::sin(a1 / 2) * std::cos(a2));
            const std::array<double, 3> Cc = {uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1};
            std::array<double, 3> t{};
            for (int i = 0; i < 3; ++i)
                t[i] = -(R[i * 3] * Cc[0] + R[i * 3 + 1] * Cc[1] + R[i * 3 + 2] * Cc[2]);
            s.w2c = {R[0], R[1], R[2], static_cast<float>(t[0]), R[3], R[4], R[5], static_cast<float>(t[1]),
                     R[6], R[7], R[8], static_cast<float>(t[2]), 0, 0, 0, 1};
            s.campos = {static_cast<float>(Cc[0]), static_cast<float>(Cc[1]), static_cast<float>(Cc[2])};
            const auto& depths = cluster_depths;
            for (const auto& dir : dirs_deg) {
                for (int k = 0; k < 4; ++k) {
                    // a small jitter of direction and depth per splat, so the overlaps are not symmetric
                    const double th = (dir[0] + (uni(rng) - 0.5) * 1.5) * std::numbers::pi / 180.0;
                    const double az = (dir[1] + (uni(rng) - 0.5) * 20.0) * std::numbers::pi / 180.0;
                    const double dist = depths[k] * (0.92 + 0.16 * uni(rng));
                    const std::array<double, 3> pc = {dist * std::sin(th) * std::cos(az), dist * std::sin(th) * std::sin(az), dist * std::cos(th)};
                    std::array<double, 3> pw{};
                    for (int j = 0; j < 3; ++j)
                        pw[j] = R[0 * 3 + j] * (pc[0] - t[0]) + R[1 * 3 + j] * (pc[1] - t[1]) + R[2 * 3 + j] * (pc[2] - t[2]);
                    for (double v : pw)
                        s.means.push_back(static_cast<float>(v));
                    for (int j = 0; j < 3; ++j)
                        s.scales.push_back(static_cast<float>(std::log(dist * ang_sigma * (0.75 + 0.5 * uni(rng)))));
                    std::array<double, 4> q = {uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1, uni(rng) * 2 - 1};
                    for (double v : q)
                        s.rots.push_back(static_cast<float>(v));
                    s.opac.push_back(static_cast<float>(-0.5 + 1.5 * uni(rng))); // sigmoid 0.38 .. 0.73
                    for (int j = 0; j < 3; ++j)
                        s.sh0.push_back(static_cast<float>((uni(rng) - 0.5) * 1.5));
                    s.theta_deg.push_back(dir[0]);
                }
            }
            s.n = static_cast<int>(s.opac.size());
            s.weights.assign(3ull * w * h, 0.0f); // no image loss: the distortion loss alone
            return s;
        }

        // 1 where every splat is either well inside its 2.5-sigma ellipse or well outside its alpha
        // cut-off (and at least one is inside), else 0: keeps the 1/255 alpha cut-off edge (which no
        // analytic gradient has a term for) out of the check, like the local image weights above.
        std::vector<float> distortion_local_weights(const Scene& s, const std::vector<Footprint>& fp) {
            std::vector<float> pw(static_cast<size_t>(s.w) * s.h, 0.0f);
            for (int y = 0; y < s.h; ++y)
                for (int x = 0; x < s.w; ++x) {
                    bool ok = true, inside = false;
                    for (const auto& f : fp) {
                        const double d2 = footprint_d2(f, x + 0.5, y + 0.5);
                        if (d2 <= 6.25)
                            inside = true;
                        else if (d2 < 16.0)
                            ok = false;
                    }
                    pw[static_cast<size_t>(y) * s.w + x] = ok && inside ? 1.0f : 0.0f;
                }
            return pw;
        }

        // The loss straight from the definition in double: per pixel, the splats front to back by their sort
        // depth (alpha = o exp(-d2/2), cut at 1/255, clamped at 0.999), weights w = alpha T, and
        // D = sum_i sum_{j<i} w_i w_j (m_i - m_j)^2 as an explicit double sum over the splats taking part
        // (m from the distortion depth; splats left out still occlude through T).
        double distortion_reference(const Scene& s, const std::vector<Footprint>& fp, const std::vector<float>& pixel_weight,
                                    double weight, double near_n, double far_n) {
            std::vector<int> order(fp.size());
            for (size_t i = 0; i < order.size(); ++i)
                order[i] = static_cast<int>(i);
            std::sort(order.begin(), order.end(), [&fp](int a, int b) { return fp[a].depth < fp[b].depth; });
            auto mapped = [&](double d) {
                const double dc = std::clamp(d, near_n, far_n);
                return far_n / (far_n - near_n) * (1.0 - near_n / dc);
            };
            double total = 0;
            std::vector<double> ws, ms;
            for (int y = 0; y < s.h; ++y)
                for (int x = 0; x < s.w; ++x) {
                    double T = 1.0;
                    ws.clear();
                    ms.clear();
                    for (int i : order) {
                        const double alpha = std::min(fp[i].opacity * std::exp(-0.5 * footprint_d2(fp[i], x + 0.5, y + 0.5)), 0.999);
                        if (alpha < 1.0 / 255.0)
                            continue;
                        if (fp[i].distortion_depth >= 0.0) { // a splat left out only occludes
                            ws.push_back(alpha * T);
                            ms.push_back(mapped(fp[i].distortion_depth));
                        }
                        T *= 1.0 - alpha;
                        if (T < 1e-4)
                            break;
                    }
                    double D = 0;
                    for (size_t i = 0; i < ws.size(); ++i)
                        for (size_t j = 0; j < i; ++j)
                            D += ws[i] * ws[j] * (ms[i] - ms[j]) * (ms[i] - ms[j]);
                    total += (pixel_weight.empty() ? 1.0 : pixel_weight[static_cast<size_t>(y) * s.w + x]) * D;
                }
            return weight * total / (static_cast<double>(s.w) * s.h);
        }

        struct DistortionSetup {
            float weight = 1.0f;
            float near_n = 0.2f;
            float far_n = 1000.0f;
            DistortionMode mode;
            std::vector<float> pixel_weight; // empty = every pixel 1
        };

        GradRun distortion_render_and_grad(const Scene& s, const DistortionSetup& ds, bool with_backward) {
            const int N = s.n;
            const size_t n_px = static_cast<size_t>(s.w) * s.h;
            DevBuf d_means(3 * N), d_scales(3 * N), d_rots(4 * N), d_opac(N), d_sh0(3 * N), d_w2c(16), d_cam(3);
            DevBuf d_img(3 * n_px), d_alpha(n_px), d_depth(n_px), d_grad_img(3 * n_px);
            DevBuf d_moments(3 * n_px), d_loss(1), d_pw(std::max<size_t>(ds.pixel_weight.size(), 1));
            DevBuf d_ddepth(N), d_gddepth(N), d_left_out(2);
            d_means.upload(s.means);
            d_scales.upload(s.scales);
            d_rots.upload(s.rots);
            d_opac.upload(s.opac);
            d_sh0.upload(s.sh0);
            d_w2c.upload(s.w2c);
            d_cam.upload(s.campos);
            if (!ds.pixel_weight.empty())
                d_pw.upload(ds.pixel_weight);
            fr::FastGSDistortion dist;
            dist.moments = d_moments.p;
            dist.loss_out = d_loss.p;
            dist.pixel_weight = ds.pixel_weight.empty() ? nullptr : d_pw.p;
            dist.loss_scale = ds.weight / static_cast<float>(n_px);
            dist.grad_scale = dist.loss_scale;
            dist.near_n = ds.near_n;
            dist.far_n = ds.far_n;
            dist.depths = d_ddepth.p;
            dist.grad_depths = d_gddepth.p;
            dist.n_left_out = d_left_out.p;
            dist.depth_mode = ds.mode.depth_distance ? fr::kDistortionDepthDistance : fr::kDistortionDepthZ;
            dist.plane_sigma = ds.mode.plane_sigma;
            dist.min_dist = ds.mode.min_dist;
            GradRun run;
            auto ctx = fr::forward_raw(d_means.p, d_scales.p, d_rots.p, d_opac.p, d_sh0.p, nullptr, d_w2c.p, d_cam.p,
                                       d_img.p, d_alpha.p, d_depth.p, nullptr, nullptr, nullptr, N, 1, 1, s.w, s.h,
                                       s.fx, s.fy, s.cx, s.cy, 0.01f, 1e10f, false, nullptr, nullptr, 0, 0, nullptr, 1.0f, s.lens,
                                       dist);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            run.n_visible = ctx.n_visible;
            run.loss = d_loss.download()[0];
            {
                const auto lo = d_left_out.download(); // [plane-crossing, nearer than min_dist]
                run.n_left_out = static_cast<int>(lo[0] + lo[1]);
                run.n_left_near = static_cast<int>(lo[1]);
            }
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
            const auto res = fr::backward_raw(nullptr, nullptr, d_grad_img.p, fr::BackgroundAlphaGradient{}, nullptr, nullptr,
                                              d_img.p, d_alpha.p, d_means.p, d_scales.p, d_rots.p, d_opac.p, nullptr,
                                              d_w2c.p, d_cam.p, ctx, nullptr, N, ctx.n_visible, 1, 1, s.w, s.h,
                                              s.fx, s.fy, s.cx, s.cy, false, DensificationType::None, &fa);
            cudaDeviceSynchronize();
            if (!res.success) {
                std::println(stderr, "backward failed: {}", res.error_message ? res.error_message : "?");
                std::exit(1);
            }
            run.g_means = g_means.download();
            run.g_scales = g_scales.download();
            run.g_rots = g_rots.download();
            run.g_opac = g_opac.download();
            return run;
        }

        struct DistortionCaseResult {
            int bad = 0;
            double median[4] = {0, 0, 0, 0};
            double max[4] = {0, 0, 0, 0};
            double group[4] = {0, 0, 0, 0};     // |fd - analytic| / |fd| over the whole group, double-reference fd
            double group_gpu[4] = {0, 0, 0, 0}; // the same against differences of the GPU forward
            double forward_rel = 0;
            int left_out_ref = 0, left_out_gpu = 0;   // splats left out of the term (plane-crossing or min-dist)
            int left_near_ref = 0, left_near_gpu = 0; // of which nearer than min_dist
        };

        // threshold: relative error (|fd - analytic| / |fd|) above which a splat's parameter block counts as bad
        DistortionCaseResult run_distortion_case(const char* title, bool fisheye, const RefLens& L, int w, int h, unsigned seed,
                                                 const std::vector<std::array<double, 2>>& dirs, double ang_sigma,
                                                 bool local, double threshold, const DistortionMode& mode,
                                                 const std::array<double, 4>& cluster_depths) {
            DistortionCaseResult out;
            Scene s = make_distortion_scene(fisheye, L, w, h, seed, dirs, ang_sigma, cluster_depths);
            // Keep every splat clear of the min-dist boundary (distortion depth = min_dist), so that no
            // finite-difference step of its mean flips it in or out of the term: within 6% of it, move the splat
            // along its ray from the camera to 12% inside / outside.
            if (mode.min_dist > 0.0f) {
                for (int i = 0; i < s.n; ++i) {
                    const double r = splat_footprint(s, L, fisheye, i, mode).near_ratio;
                    if (r > 0.94 && r < 1.06) {
                        const double k = (r < 1.0 ? 0.88 : 1.12) / r;
                        for (int j = 0; j < 3; ++j)
                            s.means[i * 3 + j] = static_cast<float>(s.campos[j] + (s.means[i * 3 + j] - s.campos[j]) * k);
                    }
                }
            }
            // Keep every splat clear of the plane-crossing boundary (z = k sigma_z), so that no finite-difference
            // step flips it in or out of the term: within 7% of it, grow or shrink the splat by 15%.
            if (mode.plane_sigma > 0.0f) {
                for (int i = 0; i < s.n; ++i) {
                    const double r = splat_footprint(s, L, fisheye, i, mode).plane_ratio;
                    if (r > 0.93 && r < 1.07)
                        for (int j = 0; j < 3; ++j)
                            s.scales[i * 3 + j] += static_cast<float>(std::log(r < 1.0 ? 1.15 : 1.0 / 1.15));
                }
            }
            std::vector<Footprint> fp;
            for (int i = 0; i < s.n; ++i)
                fp.push_back(splat_footprint(s, L, fisheye, i, mode));
            DistortionSetup ds;
            ds.mode = mode;
            if (local)
                ds.pixel_weight = distortion_local_weights(s, fp);
            const auto base = distortion_render_and_grad(s, ds, true);
            const double ref = distortion_reference(s, fp, ds.pixel_weight, ds.weight, ds.near_n, ds.far_n);
            out.forward_rel = std::abs(base.loss - ref) / std::max(std::abs(ref), 1e-30);
            for (const auto& f : fp) {
                out.left_out_ref += f.distortion_depth < 0.0 ? 1 : 0;
                out.left_near_ref += f.left_out == 2 ? 1 : 0;
            }
            out.left_out_gpu = base.n_left_out;
            out.left_near_gpu = base.n_left_near;
            size_t n_weighted = 0;
            for (float v : ds.pixel_weight)
                n_weighted += v > 0 ? 1 : 0;
            std::println("== {}: {} splats, {} visible, {}x{}, {}", title, s.n, base.n_visible, w, h,
                         local ? std::format("{} pixels weighted (away from every alpha cut-off)", n_weighted)
                               : std::string("every pixel weighted"));
            std::println("  distortion depth: {}{}; plane-crossing exclusion: {}; min-dist exclusion: {}; left out: {} (reference) / {} (GPU), "
                         "of which nearer than min-dist {} / {}",
                         (fisheye || mode.depth_distance) ? "distance from the camera" : "camera z (the old pinhole behaviour)",
                         fisheye ? " (fisheye)" : "",
                         mode.plane_sigma > 0 ? std::format("z - {} sigma_z < 0", mode.plane_sigma) : std::string("off"),
                         mode.min_dist > 0 ? std::format("d < {}", mode.min_dist) : std::string("off"),
                         out.left_out_ref, out.left_out_gpu, out.left_near_ref, out.left_near_gpu);
            for (int i = 0; i < s.n; i += 4) {
                std::println("  cluster at theta {:5.1f} deg: centre ({:6.1f}, {:6.1f}) px", s.theta_deg[i], fp[i].mx, fp[i].my);
                for (int k = i; k < i + 4 && k < s.n; ++k)
                    std::println("    splat {:2}: sort depth {:.3f}, distortion depth {}, z / (k sigma_z) {}", k, fp[k].depth,
                                 fp[k].left_out == 1   ? std::string("left out (plane-crossing)")
                                 : fp[k].left_out == 2 ? std::format("left out (nearer than min-dist: {:.3f})", fp[k].near_ratio * mode.min_dist)
                                                       : std::format("{:.3f}", fp[k].distortion_depth),
                                 mode.plane_sigma > 0 ? std::format("{:.2f}", fp[k].plane_ratio) : std::string("-"));
            }
            std::println("  forward: loss {:.6e}, double reference {:.6e}, relative difference {:.2e}", base.loss, ref, out.forward_rel);

            struct Group {
                std::vector<float>* param;
                const std::vector<float>* grad;
                int attrs;
                float step;
            };
            Group groups[] = {
                {&s.means, &base.g_means, 3, 0.0f},
                {&s.scales, &base.g_scales, 3, 2e-3f},
                {&s.rots, &base.g_rots, 4, 2e-3f},
                {&s.opac, &base.g_opac, 1, 2e-3f},
            };
            // Two numerical gradients per parameter: central differences of the GPU forward's loss (float,
            // so its noise floor is ~1e-9 absolute here), and of the double-precision reference of the same
            // loss (no float noise). The analytic (backward) gradient is scored against the second.
            const char* names[] = {"means", "scales", "rotations", "opacities"};
            auto reference_loss = [&](const Scene& sc) {
                std::vector<Footprint> f;
                for (int i = 0; i < sc.n; ++i)
                    f.push_back(splat_footprint(sc, L, fisheye, i, mode));
                return distortion_reference(sc, f, ds.pixel_weight, ds.weight, ds.near_n, ds.far_n);
            };
            std::println("  {:>10} {:>6} {:>8} {:>10} {:>12} {:>12} {:>10}", "param", "splat", "theta", "cosine", "rel err ref", "rel err gpu", "|grad|");
            for (int gi = 0; gi < 4; ++gi) {
                auto& g = groups[gi];
                std::vector<double> rel(s.n), rel_gpu(s.n), fdn(s.n), cosv(s.n), an_n(s.n);
                double sum_nd = 0, sum_nf = 0, sum_nd_gpu = 0, sum_nf_gpu = 0;
                for (int i = 0; i < s.n; ++i) {
                    std::vector<double> fd(g.attrs), fd_gpu(g.attrs), an(g.attrs);
                    for (int a = 0; a < g.attrs; ++a) {
                        const size_t idx = static_cast<size_t>(i) * g.attrs + a;
                        float step = g.step;
                        if (step == 0.0f) {
                            const double dx = s.means[i * 3] - s.campos[0], dy = s.means[i * 3 + 1] - s.campos[1], dz = s.means[i * 3 + 2] - s.campos[2];
                            step = static_cast<float>(5e-4 * std::sqrt(dx * dx + dy * dy + dz * dz));
                        }
                        const float v0 = (*g.param)[idx];
                        (*g.param)[idx] = v0 + step;
                        const double lp = distortion_render_and_grad(s, ds, false).loss;
                        const double lp_ref = reference_loss(s);
                        (*g.param)[idx] = v0 - step;
                        const double lm = distortion_render_and_grad(s, ds, false).loss;
                        const double lm_ref = reference_loss(s);
                        (*g.param)[idx] = v0;
                        const double h2 = static_cast<double>(v0 + step) - static_cast<double>(v0 - step);
                        fd_gpu[a] = (lp - lm) / h2;
                        fd[a] = (lp_ref - lm_ref) / h2;
                        an[a] = (*g.grad)[idx];
                    }
                    double dot = 0, nf = 0, na = 0, nd = 0, nf_gpu = 0, nd_gpu = 0;
                    for (int a = 0; a < g.attrs; ++a) {
                        dot += fd[a] * an[a];
                        nf += fd[a] * fd[a];
                        na += an[a] * an[a];
                        nd += (fd[a] - an[a]) * (fd[a] - an[a]);
                        nf_gpu += fd_gpu[a] * fd_gpu[a];
                        nd_gpu += (fd_gpu[a] - an[a]) * (fd_gpu[a] - an[a]);
                    }
                    sum_nd += nd;
                    sum_nf += nf;
                    sum_nd_gpu += nd_gpu;
                    sum_nf_gpu += nf_gpu;
                    rel[i] = std::sqrt(nd) / std::max(std::sqrt(nf), 1e-30);
                    rel_gpu[i] = std::sqrt(nd_gpu) / std::max(std::sqrt(nf_gpu), 1e-30);
                    fdn[i] = std::sqrt(nf);
                    an_n[i] = std::sqrt(na);
                    cosv[i] = dot / std::sqrt(std::max(nf * na, 1e-300));
                }
                const double fd_max = *std::max_element(fdn.begin(), fdn.end());
                std::vector<double> counted;
                for (int i = 0; i < s.n; ++i) {
                    // splats whose distortion gradient is negligible (no overlap left) are listed, not scored
                    const bool scored = fdn[i] > 1e-3 * fd_max;
                    std::println("  {:>10} {:6} {:8.3g} {:10.6f} {:12.2e} {:12.2e} {:10.3g}{}", names[gi], i, s.theta_deg[i], cosv[i], rel[i],
                                 rel_gpu[i], an_n[i], scored ? "" : "  (negligible, not scored)");
                    if (!scored)
                        continue;
                    counted.push_back(rel[i]);
                    if (rel[i] > threshold)
                        ++out.bad;
                }
                std::sort(counted.begin(), counted.end());
                out.median[gi] = counted.empty() ? 0.0 : counted[counted.size() / 2];
                out.max[gi] = counted.empty() ? 0.0 : counted.back();
                out.group[gi] = std::sqrt(sum_nd) / std::max(std::sqrt(sum_nf), 1e-30);
                out.group_gpu[gi] = std::sqrt(sum_nd_gpu) / std::max(std::sqrt(sum_nf_gpu), 1e-30);
                std::println("  {:>10}: vs double-reference differences: median rel err {:.2e}, max {:.2e} ({} splats scored), "
                             "whole group {:.2e}; vs GPU-forward differences: whole group {:.2e}",
                             names[gi], out.median[gi], out.max[gi], counted.size(), out.group[gi], out.group_gpu[gi]);
            }
            return out;
        }

        int run_distortion_gradcheck() {
            const RefLens pin = {110.0, 110.0, 100, 60, {0, 0, 0, 0}};
            // a wide pinhole (half-angles 70 / 59 deg, like the 136-deg training images), for splats beside the camera
            const RefLens pin_wide = {36.0, 36.0, 100, 60, {0, 0, 0, 0}};
            const RefLens game_fish = {77.0, 77.0, 100, 60, {0, 0, 0, 0}}; // equidistant (k = 0), the game lens model
            const RefLens wood_fish = {48.74, 49.35, 100, 60, {0.39127139, -0.03901020, 0.00243290, -0.00266145}}; // woodland lens at 1/6 res
            const std::vector<std::array<double, 2>> pin_dirs = {{0.0, 0.0}, {14.0, 40.0}, {24.0, 180.0}};
            const std::vector<std::array<double, 2>> fish_dirs = {{0.0, 0.0}, {30.0, 40.0}, {55.0, 180.0}};
            const std::vector<std::array<double, 2>> wood_dirs = {{0.0, 0.0}, {35.0, 40.0}, {62.0, 180.0}};
            // beside the camera: clusters 0.45-1.5 units away at 62 and 68 deg off the axis (z = 0.17-0.7), one at 20 deg
            const std::vector<std::array<double, 2>> beside_pin_dirs = {{20.0, 90.0}, {62.0, 0.0}, {68.0, 180.0}};
            const std::vector<std::array<double, 2>> beside_fish_dirs = {{20.0, 90.0}, {60.0, 0.0}, {70.0, 180.0}};
            const std::array<double, 4> normal_depths = {1.2, 1.9, 2.8, 4.0};
            const std::array<double, 4> beside_depths = {0.45, 0.7, 1.0, 1.5};
            const DistortionMode dflt{true, 2.0f};      // the defaults: distance, 2-sigma plane-crossing exclusion
            const DistortionMode dist_only{true, 0.0f}; // distance, nothing left out
            const DistortionMode old_z{false, 0.0f};    // the old pinhole behaviour: camera z, nothing left out
            // min-dist (round 2): splats nearer than min_dist left out too, on top of the 2-sigma exclusion
            const DistortionMode md08{true, 2.0f, 0.8f};
            const DistortionMode md15{true, 2.0f, 1.5f};
            const DistortionMode md23{true, 2.0f, 2.3f};
            const DistortionMode md_only{true, 0.0f, 1.0f}; // min-dist 1 alone
            struct Setup {
                const char* title;
                bool fisheye;
                RefLens lens;
                const std::vector<std::array<double, 2>>* dirs;
                double ang_sigma;
                unsigned seed;
                DistortionMode mode;
                const std::array<double, 4>* depths;
            };
            const Setup setups[] = {
                {"pinhole 200x120, distance, 2-sigma exclusion", false, pin, &pin_dirs, 0.06, 21, dflt, &normal_depths},
                {"pinhole 200x120, camera z (old)", false, pin, &pin_dirs, 0.06, 21, old_z, &normal_depths},
                {"fisheye: game equidistant lens 200x120, 2-sigma exclusion", true, game_fish, &fish_dirs, 0.08, 22, dflt, &normal_depths},
                {"fisheye: woodland OPENCV_FISHEYE 200x120, 2-sigma exclusion", true, wood_fish, &wood_dirs, 0.10, 23, dflt, &normal_depths},
                {"wide pinhole, beside the camera, distance, 2-sigma exclusion", false, pin_wide, &beside_pin_dirs, 0.20, 24, dflt, &beside_depths},
                {"wide pinhole, beside the camera, distance, no exclusion", false, pin_wide, &beside_pin_dirs, 0.20, 24, dist_only, &beside_depths},
                {"wide pinhole, beside the camera, camera z (old)", false, pin_wide, &beside_pin_dirs, 0.20, 24, old_z, &beside_depths},
                {"fisheye: game lens, beside the camera, 2-sigma exclusion", true, game_fish, &beside_fish_dirs, 0.20, 25, dflt, &beside_depths},
                {"pinhole 200x120, 2-sigma + min-dist 2.3", false, pin, &pin_dirs, 0.06, 21, md23, &normal_depths},
                {"fisheye: game lens 200x120, 2-sigma + min-dist 1.5", true, game_fish, &fish_dirs, 0.08, 22, md15, &normal_depths},
                {"fisheye: woodland lens, beside the camera, 2-sigma + min-dist 0.8", true, wood_fish, &beside_fish_dirs, 0.20, 26, md08, &beside_depths},
                {"wide pinhole, beside the camera, min-dist 1 only", false, pin_wide, &beside_pin_dirs, 0.20, 24, md_only, &beside_depths},
            };
            int bad = 0;
            int forward_bad = 0;
            int left_out_mismatch = 0;
            std::vector<std::pair<std::string, DistortionCaseResult>> summary;
            std::println("######## distortion loss alone; pixels weighted only away from every splat's alpha cut-off");
            for (const auto& st : setups) {
                const auto r = run_distortion_case(st.title, st.fisheye, st.lens, 200, 120, st.seed, *st.dirs, st.ang_sigma, true, 0.01,
                                                   st.mode, *st.depths);
                bad += r.bad;
                if (r.forward_rel > 5e-3)
                    ++forward_bad;
                if (r.left_out_ref != r.left_out_gpu || r.left_near_ref != r.left_near_gpu)
                    ++left_out_mismatch;
                summary.emplace_back(std::string("local  ") + st.title, r);
            }
            std::println("######## every pixel weighted: includes the alpha cut-off edges, which no analytic gradient has");
            std::println("######## a term for, i.e. the rasterizer's own noise floor (not scored)");
            for (const auto& st : setups) {
                const auto r = run_distortion_case(st.title, st.fisheye, st.lens, 200, 120, st.seed, *st.dirs, st.ang_sigma, false, 0.1,
                                                   st.mode, *st.depths);
                if (r.forward_rel > 5e-3)
                    ++forward_bad;
                if (r.left_out_ref != r.left_out_gpu || r.left_near_ref != r.left_near_gpu)
                    ++left_out_mismatch;
                summary.emplace_back(std::string("global ") + st.title, r);
            }
            std::println("######## summary: relative error of the analytic gradient per parameter group against double-reference");
            std::println("######## differences, as median/max over splats and [whole group]; then the whole group against GPU-forward");
            std::println("######## differences; the forward loss against the double reference; splats left out (reference/GPU; of which min-dist)");
            for (const auto& [name, r] : summary)
                std::println("  {:<70} means {:.1e}/{:.1e} [{:.1e}]  scales {:.1e}/{:.1e} [{:.1e}]  rotations {:.1e}/{:.1e} [{:.1e}]  "
                             "opacities {:.1e}/{:.1e} [{:.1e}]  | gpu-fd groups {:.1e} {:.1e} {:.1e} {:.1e}  | forward {:.1e} | left out {}/{} (min-dist {}/{})",
                             name, r.median[0], r.max[0], r.group[0], r.median[1], r.max[1], r.group[1], r.median[2], r.max[2], r.group[2],
                             r.median[3], r.max[3], r.group[3], r.group_gpu[0], r.group_gpu[1], r.group_gpu[2], r.group_gpu[3],
                             r.forward_rel, r.left_out_ref, r.left_out_gpu, r.left_near_ref, r.left_near_gpu);
            if (bad == 0 && forward_bad == 0 && left_out_mismatch == 0)
                std::println("DISTORTION GRADCHECK OK");
            else
                std::println("DISTORTION GRADCHECK: {} splat parameter blocks over 1% relative error (local weights), {} forward mismatches, "
                             "{} left-out count mismatches",
                             bad, forward_bad, left_out_mismatch);
            return bad == 0 && forward_bad == 0 && left_out_mismatch == 0 ? 0 : 1;
        }
        // ---------------------------------------------------------------- cubemap rays vs the splats' projection (LFQPLC)
        // The background cubemap's pixel rays (kernels::cubemap_pixel_dirs: exactly the render / backward kernels' ray, here
        // with an identity pose and no rolling shutter) projected back through the rasterizer's own projection
        // (fisheye_projection_selftest: FastGS's fisheye::terms_lens + project, the forward preprocess path) must land on
        // the pixel centres. Cases: the table's rays through the table (the --lens-radial path); the table's rays through
        // the tag k-polynomial (negative control: must miss); the polynomial's rays through the polynomial (the original path).
        int run_cuberaycheck(const std::filesystem::path& file) {
            try {
                lfs::training::activate_radial_lens(lfs::training::load_radial_lens_table(file));
            } catch (const std::exception& e) {
                std::println(stderr, "{}", e.what());
                return 1;
            }
            const auto& t = *lfs::training::active_radial_lens();
            std::println("radial lens table: {}", t.describe());
            const int W = t.width / 4, H = t.height / 4;
            const float fx = static_cast<float>(t.fx / 4), fy = static_cast<float>(t.fy / 4), cx = static_cast<float>(t.cx / 4),
                        cy = static_cast<float>(t.cy / 4);
            const double tmax = 88.0 * std::numbers::pi / 180.0;
            float* d_w2c = nullptr;
            float* d_dirs = nullptr;
            const float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            cudaMalloc(&d_w2c, sizeof(ident));
            cudaMemcpy(d_w2c, ident, sizeof(ident), cudaMemcpyHostToDevice);
            cudaMalloc(&d_dirs, sizeof(float) * 3 * W * H);
            const double valid_deg = t.valid * 180.0 / std::numbers::pi;
            int bad = 0;
            struct Case {
                const char* name;
                bool ray_table, proj_table, expect_hit;
            };
            for (const Case c : {Case{"cube rays through the table -> splat projection through the table (--lens-radial)", true, true, true},
                                 Case{"cube rays through the table -> projection through the tag k-polynomial (negative control)", true, false, false},
                                 Case{"cube rays through the k-polynomial -> projection through the k-polynomial (original path)", false, false, true}}) {
                lfs::training::kernels::CubemapView v{};
                v.width = W;
                v.height = H;
                v.fx = fx;
                v.fy = fy;
                v.cx = cx;
                v.cy = cy;
                v.fisheye = 1;
                v.k1 = static_cast<float>(t.k[0]);
                v.k2 = static_cast<float>(t.k[1]);
                v.k3 = static_cast<float>(t.k[2]);
                v.k4 = static_cast<float>(t.k[3]);
                v.theta_max = static_cast<float>(tmax);
                if (c.ray_table)
                    v.radial = t.device_view();
                v.w2c = d_w2c;
                lfs::training::kernels::cubemap_pixel_dirs(v, d_dirs, nullptr);
                std::vector<float> dirs(3ull * W * H);
                cudaMemcpy(dirs.data(), d_dirs, sizeof(float) * dirs.size(), cudaMemcpyDeviceToHost);
                fr::FastGSLens lens;
                lens.model = fr::kLensFisheye;
                lens.k1 = v.k1;
                lens.k2 = v.k2;
                lens.k3 = v.k3;
                lens.k4 = v.k4;
                lens.max_w = static_cast<float>(std::tan(tmax) * std::tan(tmax));
                if (c.proj_table)
                    lens.radial = t.device_view();
                const int n = W * H;
                std::vector<float> in(9ull * n, 0.0f), out(17ull * n);
                for (int i = 0; i < n; ++i)
                    for (int k = 0; k < 3; ++k)
                        in[9ull * i + k] = 3.0f * dirs[3ull * i + k]; // a point 3 units along the ray
                fr::fisheye_projection_selftest(lens, fx, fy, cx, cy, in.data(), n, out.data());
                // error per angle band (the ray's own angle off the axis)
                const double bands[] = {0, 1, 10, 30, 50, 65, 72, valid_deg, 90};
                const int nb = static_cast<int>(std::size(bands)) - 1;
                std::vector<double> bmax(nb, 0.0), bsum(nb, 0.0);
                std::vector<long> bn(nb, 0);
                std::vector<double> errs_valid;
                double emax_valid = 0, emax_all = 0;
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x) {
                        const size_t i = static_cast<size_t>(y) * W + x;
                        const float* o = out.data() + 17 * i;
                        const double e = std::hypot(o[0] - (x + 0.5), o[1] - (y + 0.5));
                        const double th = std::acos(std::clamp(static_cast<double>(dirs[3 * i + 2]), -1.0, 1.0)) * 180.0 / std::numbers::pi;
                        int b = 0;
                        while (b + 1 < nb && th >= bands[b + 1])
                            ++b;
                        bmax[b] = std::max(bmax[b], e);
                        bsum[b] += e;
                        ++bn[b];
                        emax_all = std::max(emax_all, e);
                        if (th <= valid_deg) {
                            emax_valid = std::max(emax_valid, e);
                            errs_valid.push_back(e);
                        }
                    }
                std::sort(errs_valid.begin(), errs_valid.end());
                const auto q = [&](double f) { return errs_valid.empty() ? 0.0 : errs_valid[static_cast<size_t>(f * (errs_valid.size() - 1))]; };
                std::println("== {} ({}x{} = {} pixels)", c.name, W, H, n);
                std::println("   inside the table's valid {:.2f} deg: |projected - pixel centre| median {:.3g} px, p99.9 {:.3g}, max {:.3g}; "
                             "all pixels max {:.3g} px",
                             valid_deg, q(0.5), q(0.999), emax_valid, emax_all);
                for (int b = 0; b < nb; ++b)
                    if (bn[b])
                        std::println("   theta {:6.2f}-{:6.2f} deg: {:7d} px  mean {:.3g}  max {:.3g} px", bands[b], bands[b + 1], bn[b],
                                     bsum[b] / bn[b], bmax[b]);
                const bool hit = emax_all < 5e-3;
                if (hit != c.expect_hit) {
                    ++bad;
                    std::println("   FAILED: expected {}", c.expect_hit ? "every ray to land within 0.005 px" : "a miss (> 0.005 px)");
                } else {
                    std::println("   ok ({})", c.expect_hit ? "every pixel's ray lands on its pixel centre within 0.005 px" : "misses, as it must");
                }
            }
            cudaFree(d_w2c);
            cudaFree(d_dirs);
            std::println("{}", bad == 0 ? "CUBERAYCHECK OK" : "CUBERAYCHECK FAILED");
            return bad == 0 ? 0 : 1;
        }

    } // namespace

    int run_lfe_fisheye_tool(int argc, const char* const* argv) {
        if (argc >= 6 && std::string(argv[2]) == "render")
            return run_render(argv[3], argv[4], argv[5]);
        if (argc >= 3 && std::string(argv[2]) == "projcheck")
            return run_projcheck();
        if (argc >= 3 && std::string(argv[2]) == "gradcheck")
            return run_gradcheck();
        if (argc >= 4 && std::string(argv[2]) == "radialcheck")
            return run_radialcheck(argv[3], argc >= 5 && std::string(argv[4]) == "--vs-poly");
        if (argc >= 4 && std::string(argv[2]) == "cuberaycheck")
            return run_cuberaycheck(argv[3]);
        std::println(stderr, "usage: LichtFeld-Studio lfe-fisheye render <model.ply> <cameras.txt> <out_dir> | projcheck | gradcheck | "
                             "radialcheck <lens_radial.txt> [--vs-poly] | cuberaycheck <lens_radial.txt>");
        return 2;
    }

    int run_lfe_distortion_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "gradcheck")
            return run_distortion_gradcheck();
        std::println(stderr, "usage: LichtFeld-Studio lfe-distortion gradcheck");
        return 2;
    }

} // namespace lfs::app
