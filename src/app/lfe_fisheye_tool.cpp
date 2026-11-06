/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Developer checks for the standard rasterizer's native fisheye projection.
//   LichtFeld-Studio lfe-fisheye render <model.ply> <cameras.txt> <out_dir>
//     cameras.txt, one camera per line:
//       name width height model fx fy cx cy k1 k2 k3 k4 qw qx qy qz tx ty tz
//     model PINHOLE or FISHEYE (OPENCV_FISHEYE; k = 0 is the equidistant lens); the pose is
//     COLMAP's world-to-camera quaternion and translation. Writes <out_dir>/<name>.f32: float32
//     RGB, row-major H x W x 3, black background, through fast_rasterize (the training path; with the model's
//     per-splat kernel shape when its PLY has a `shape` property, LFQPLCYK). LFS_RENDER_ALPHA_DIR / LFS_RENDER_DEPTH_DIR:
//     also the alpha / depth planes there (<name>.f32, float32 H x W, + <name>.json).
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
#include "training/kernels/ycc_loss.hpp"

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
            if (model._shape.is_valid()) // LFQPLCYK: the PLY's `shape` property, drawn with each splat's beta
                std::println("per-splat kernel shape (PLY property shape): {} values", model._shape.numel());
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
                // LFQPLCYK: opt-in extra outputs for the edge-width measure (the .f32 RGB set above is unchanged):
                // LFS_RENDER_ALPHA_DIR / LFS_RENDER_DEPTH_DIR -> <dir>/<name>.f32 (float32 H x W) + <name>.json {"w", "h"}
                for (const auto& [env, tensor] : {std::pair<const char*, const lfs::core::Tensor*>{"LFS_RENDER_ALPHA_DIR", &out.alpha},
                                                  std::pair<const char*, const lfs::core::Tensor*>{"LFS_RENDER_DEPTH_DIR", &out.depth}}) {
                    const char* dir = std::getenv(env);
                    if (dir == nullptr || *dir == 0 || !tensor->is_valid() || tensor->numel() != static_cast<size_t>(w) * h)
                        continue;
                    std::filesystem::create_directories(dir);
                    const auto plane = tensor->contiguous().cpu();
                    std::ofstream fa(std::filesystem::path(dir) / (name + ".f32"), std::ios::binary);
                    fa.write(reinterpret_cast<const char*>(plane.ptr<float>()), static_cast<std::streamsize>(sizeof(float) * w * h));
                    std::ofstream fj(std::filesystem::path(dir) / (name + ".json"));
                    fj << std::format("{{\"w\": {}, \"h\": {}}}", w, h) << '\n';
                }
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

        // ---------------------------------------------------------------- --loss-ycc kernels (LFQPLCY)
        // The chroma term and the luma conversion of --loss-ycc (kernels/ycc_loss.hpp) on synthetic images:
        //  (1) a chroma change finer than 2x2 (a checkerboard in Cb only, Y and Cr held) -> zero chroma loss, and no Y change
        //      beyond float rounding;
        //  (2) a 2x2-constant Cb step of d -> chroma loss = scale x d / 2 exactly;
        //  (3) a random render / target / mask (some pixels 0) -> the inputs and every output written to OUT_DIR (float32
        //      .f32 + ycc_case.txt) for the Python reference (D:/splat_work/lfrqplcy/ycc_ref.py), and the chroma gradient
        //      against central differences of the chroma loss.
        struct YccDev {
            std::vector<float*> ptrs;
            float* alloc(const std::vector<float>& h) {
                float* d = nullptr;
                cudaMalloc(&d, sizeof(float) * std::max<size_t>(h.size(), 1));
                cudaMemcpy(d, h.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice);
                ptrs.push_back(d);
                return d;
            }
            float* zeros(size_t n) { return alloc(std::vector<float>(n, 0.0f)); }
            static std::vector<float> get(const float* d, size_t n) {
                std::vector<float> h(n);
                cudaMemcpy(h.data(), d, sizeof(float) * n, cudaMemcpyDeviceToHost);
                return h;
            }
            ~YccDev() {
                for (float* p : ptrs)
                    cudaFree(p);
            }
        };

        // chroma loss (and coefficient) of pred vs tgt [3, H, W] with weights w [H, W] (empty: none, border 0)
        float ycc_chroma(YccDev& dv, const std::vector<float>& pred, const std::vector<float>& tgt, const std::vector<float>& w,
                         int H, int W, float scale, std::vector<float>* grad, std::vector<float>* sums_out) {
            namespace yk = lfs::training::kernels;
            float* dp = dv.alloc(pred);
            float* dt = dv.alloc(tgt);
            float* dw = w.empty() ? nullptr : dv.alloc(w);
            float* ds = dv.zeros(4);
            yk::ycc_chroma_sums(dp, dt, false, dw, w.empty() ? yk::kYccWeightNone : yk::kYccWeightFloat, H, W, 0, ds, nullptr);
            yk::ycc_chroma_finalize(ds, scale, ds + 2, ds + 3, nullptr);
            if (grad) {
                float* dg = dv.zeros(3ull * H * W);
                yk::ycc_chroma_backward(dp, dt, false, dw, w.empty() ? yk::kYccWeightNone : yk::kYccWeightFloat, H, W, 0, ds + 3, dg,
                                        nullptr);
                cudaDeviceSynchronize(); // the kernels run on the current (non-blocking) stream: finish before the plain copy
                *grad = YccDev::get(dg, 3ull * H * W);
            }
            cudaDeviceSynchronize();
            const auto s = YccDev::get(ds, 4);
            if (sums_out)
                *sums_out = s;
            return s[2];
        }

        std::vector<float> ycc_luma_host(YccDev& dv, const std::vector<float>& rgb, int H, int W) {
            float* d = dv.alloc(rgb);
            float* y = dv.zeros(static_cast<size_t>(H) * W);
            lfs::training::kernels::ycc_luma(d, false, H, W, y, nullptr);
            cudaDeviceSynchronize();
            return YccDev::get(y, static_cast<size_t>(H) * W);
        }

        int run_ycccheck(const std::filesystem::path& out_dir) {
            namespace yk = lfs::training::kernels;
            std::filesystem::create_directories(out_dir);
            const int H = 96, W = 128;
            const size_t HW = static_cast<size_t>(H) * W;
            std::mt19937 rng(5);
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            int bad = 0;
            YccDev dv;
            // a smooth target in [0.2, 0.8]
            std::vector<float> tgt(3 * HW);
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        tgt[c * HW + y * W + x] = 0.5f + 0.25f * std::sin(0.07f * x + 1.3f * c) * std::cos(0.05f * y + 0.7f * c);
            const float scale = 0.8f;
            // (1) Cb-only checkerboard: dB = 1.8556 d, dG = -0.0722 x 1.8556 d / 0.7152, dR = 0 (Y and Cr unchanged)
            const float d = 0.04f;
            std::vector<float> chk = tgt;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const float s = ((x + y) & 1) ? -d : d;
                    chk[2 * HW + y * W + x] += yk::kYccCbDen * s;
                    chk[HW + y * W + x] += -yk::kYccKb * yk::kYccCbDen / yk::kYccKg * s;
                }
            const float l_chk = ycc_chroma(dv, chk, tgt, {}, H, W, scale, nullptr, nullptr);
            const auto y_chk = ycc_luma_host(dv, chk, H, W), y_tgt = ycc_luma_host(dv, tgt, H, W);
            double dy = 0;
            for (size_t i = 0; i < HW; ++i)
                dy = std::max(dy, static_cast<double>(std::abs(y_chk[i] - y_tgt[i])));
            // (2) the same amplitude constant over each 2x2 block (alternating per block): chroma loss = scale d / 2
            std::vector<float> blk = tgt;
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const float s = (((x >> 1) + (y >> 1)) & 1) ? -d : d;
                    blk[2 * HW + y * W + x] += yk::kYccCbDen * s;
                    blk[HW + y * W + x] += -yk::kYccKb * yk::kYccCbDen / yk::kYccKg * s;
                }
            const float l_blk = ycc_chroma(dv, blk, tgt, {}, H, W, scale, nullptr, nullptr);
            std::println("(1) Cb-only checkerboard finer than 2x2 (amplitude {:g} in Cb): chroma loss {:.3e} (vs {:.6f} for the same "
                         "amplitude constant over 2x2 blocks); max |dY| {:.2e}",
                         d, l_chk, l_blk, dy);
            std::println("(2) 2x2-constant Cb step: chroma loss {:.7f}, expected scale x d / 2 = {:.7f}", l_blk, scale * d / 2);
            if (!(l_chk < 1e-6f * l_blk + 1e-9f) || dy > 1e-6)
                ++bad;
            if (std::abs(l_blk - scale * d / 2) > 1e-6f)
                ++bad;
            // (3) random case + mask, dumped for the Python reference; central differences of the chroma loss
            std::vector<float> pred(3 * HW), w(HW);
            for (size_t i = 0; i < 3 * HW; ++i)
                pred[i] = std::clamp(tgt[i] + 0.15f * (uni(rng) - 0.5f), 0.0f, 1.0f);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    w[y * W + x] = (std::hypot(x - 40.0, y - 50.0) < 18.0 || (x > 100 && y < 20) || ((x * 7 + y * 3) % 23 == 0)) ? 0.0f
                                   : (x < 64 ? 1.0f : 1.5f);
            std::vector<float> grad, sums;
            const float loss = ycc_chroma(dv, pred, tgt, w, H, W, scale, &grad, &sums);
            const auto y_pred = ycc_luma_host(dv, pred, H, W);
            std::vector<float> gy(HW), grgb;
            for (auto& v : gy)
                v = uni(rng) - 0.5f;
            {
                float* dgy = dv.alloc(gy);
                float* dg = dv.zeros(3 * HW);
                yk::ycc_luma_grad_to_rgb(dgy, 1.7f, static_cast<int>(HW), dg, false, nullptr);
                cudaDeviceSynchronize();
                grgb = YccDev::get(dg, 3 * HW);
            }
            const auto put = [&](const char* n, const std::vector<float>& v) {
                std::ofstream f(out_dir / std::format("{}.f32", n), std::ios::binary);
                f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
            };
            put("pred", pred);
            put("tgt", tgt);
            put("w", w);
            put("y_pred", y_pred);
            put("y_tgt", y_tgt);
            put("grad", grad);
            put("gy", gy);
            put("grgb", grgb);
            put("chk", chk);
            {
                std::ofstream info(out_dir / "ycc_case.txt");
                info << std::format("H {}\nW {}\nscale {:.9g}\nloss {:.9g}\nabs_sum {:.9g}\nweight_sum {:.9g}\ncoef {:.9g}\n"
                                    "luma_grad_scale 1.7\nchk_loss {:.9g}\nblk_loss {:.9g}\nd {:.9g}\n",
                                    H, W, scale, loss, sums[0], sums[1], sums[3], l_chk, l_blk, d);
            }
            // central differences on 40 (pixel, channel) entries. The loss is a mean over the kept blocks: with ~2800 of them
            // one entry moves it by ~1e-8, the size of the float32 atomic-sum noise. So the differences use the same images with
            // only a 16x16-pixel patch weighted (one entry then moves the loss ~40x more), against that case's own analytic
            // gradient, and skip entries whose block is within 4 steps of an L1 kink (|dCb| or |dCr| of the block mean).
            std::vector<float> wp(HW, 0.0f);
            for (int y = 32; y < 48; ++y)
                for (int x = 80; x < 96; ++x)
                    wp[y * W + x] = w[y * W + x];
            std::vector<float> gp;
            ycc_chroma(dv, pred, tgt, wp, H, W, scale, &gp, nullptr);
            const auto blk_d = [&](int y, int x, int which) { // block-mean dCb (0) or dCr (1) of pred - tgt, in double
                double s = 0;
                for (int yy = y & ~1; yy < (y & ~1) + 2; ++yy)
                    for (int xx = x & ~1; xx < (x & ~1) + 2; ++xx) {
                        const size_t q = static_cast<size_t>(yy) * W + xx;
                        double yp = 0, yt = 0;
                        const double kk[3] = {yk::kYccKr, yk::kYccKg, yk::kYccKb};
                        for (int c = 0; c < 3; ++c) {
                            yp += kk[c] * pred[c * HW + q];
                            yt += kk[c] * tgt[c * HW + q];
                        }
                        s += which == 0 ? ((pred[2 * HW + q] - yp) - (tgt[2 * HW + q] - yt)) / yk::kYccCbDen
                                        : ((pred[q] - yp) - (tgt[q] - yt)) / yk::kYccCrDen;
                    }
                return s / 4.0;
            };
            double worst = 0;
            int checked = 0;
            const float h = 1e-3f;
            for (int k = 0; k < 4000 && checked < 40; ++k) {
                const int y = 32 + static_cast<int>(uni(rng) * 16) % 16, x = 80 + static_cast<int>(uni(rng) * 16) % 16;
                const int c = static_cast<int>(uni(rng) * 3) % 3;
                const size_t i = c * HW + static_cast<size_t>(y) * W + x;
                if (wp[static_cast<size_t>(y) * W + x] == 0.0f || std::min(std::abs(blk_d(y, x, 0)), std::abs(blk_d(y, x, 1))) < 4.0 * 0.25 * h)
                    continue;
                auto pp = pred, pm = pred;
                pp[i] += h;
                pm[i] -= h;
                const double hreal = static_cast<double>(pp[i]) - pm[i];
                const double fd = (static_cast<double>(ycc_chroma(dv, pp, tgt, wp, H, W, scale, nullptr, nullptr)) -
                                   ycc_chroma(dv, pm, tgt, wp, H, W, scale, nullptr, nullptr)) / hreal;
                worst = std::max(worst, std::abs(fd - gp[i]) / std::max(1e-6, std::abs(static_cast<double>(gp[i]))));
                ++checked;
            }
            std::println("(3) random render / target / mask ({}x{}; {} of {} pixels weighted, weights 1 and 1.5): chroma loss {:.7f}, "
                         "kept 2x2-block weight sum {:.2f}; gradient vs central differences on {} entries: worst relative "
                         "{:.2e}; inputs and outputs -> {}",
                         W, H, std::count_if(w.begin(), w.end(), [](float v) { return v > 0; }), HW, loss, sums[1], checked, worst,
                         out_dir.string());
            if (worst > 2e-2)
                ++bad;
            std::println("{}", bad == 0 ? "YCCCHECK OK" : "YCCCHECK FAILED");
            return bad == 0 ? 0 : 1;
        }

        // ---------------------------------------------------------------- --max-std-dev (LFQPLCY)
        // The game's splat cut in the FastGS forward and backward (fr::set_splat_cutoff), on a pinhole scene of separated
        // splats (some with opacities near 0.5/255 x e^(X^2/2), where the alpha floor binds instead of the cut):
        //  renders with the cut off and at X, written to OUT_DIR with the scene for the Python reference
        //  (D:/splat_work/lfrqplcy/cutcheck_ref.py: every pixel, edge pixels included);
        //  analytic gradients (means, scales, rotations, opacities, SH0) with the cut against central differences of the full
        //  forward, the image weights smooth everywhere except thin bands around each splat's cut ellipse and alpha contours
        //  (where a step would move a pixel across the boundary: a jump no gradient carries), so pixels just inside the cut
        //  and the ring the cut removes (sqrt(5)..3.3 sigma, nonzero weights there) are both checked;
        //  control: the analytic gradient with the cut differs from the uncut backward's.
        struct CutScene {
            int W = 192, H = 144, n = 0;
            float fx = 170.0f, fy = 170.0f, cx = 96.0f, cy = 72.0f;
            std::vector<float> means, scales, rots, opac, sh0;
            float bg[3] = {0.2f, 0.3f, 0.45f};
            bool fish = false;     // native fisheye through the active radial lens table (cutcheck ... TABLE)
            fr::FastGSLens lens{}; // pinhole unless fish
        };
        struct CutRun {
            std::vector<float> img, alpha, g[5]; // means, scales, rotations, opacities, sh0
        };

        CutScene make_cut_scene(unsigned seed, const bool fish = false) {
            CutScene s;
            if (fish) { // a wide field through the table: the corner splats ~64 deg off the axis
                s.fish = true;
                s.fx = s.fy = 60.0f;
                s.lens.model = fr::kLensFisheye;
                s.lens.radial = lfs::training::active_radial_lens()->device_view();
                const double tm = std::tan(75.0 * std::numbers::pi / 180.0);
                s.lens.max_w = static_cast<float>(tm * tm);
            }
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> uni(0.0f, 1.0f);
            const float opacities[] = {0.9f, 0.6f, 0.35f, 0.12f, 0.05f, 0.03f, 0.024f, 0.02f, 0.012f, 0.008f};
            int k = 0;
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 6; ++i, ++k) {
                    const float u = (i + 0.5f) * s.W / 6.0f + (uni(rng) - 0.5f) * 4.0f, v = (j + 0.5f) * s.H / 4.0f + (uni(rng) - 0.5f) * 4.0f;
                    const float z = 2.5f + 1.5f * uni(rng);
                    s.means.insert(s.means.end(), {(u - s.cx) / s.fx * z, (v - s.cy) / s.fy * z, z});
                    for (int a = 0; a < 3; ++a) // 2D sigma ~1.5..4.5 px, anisotropic (fisheye: / 1.4, its edges magnify)
                        s.scales.push_back(std::log((1.5f + 3.0f * uni(rng)) * z / s.fx / (fish ? 1.4f : 1.0f)));
                    for (int a = 0; a < 4; ++a)
                        s.rots.push_back(uni(rng) * 2.0f - 1.0f);
                    const float o = opacities[k % 10];
                    s.opac.push_back(std::log(o / (1.0f - o)));
                    for (int a = 0; a < 3; ++a)
                        s.sh0.push_back((uni(rng) - 0.5f) * 2.0f);
                }
            s.n = k;
            if (fish) { // each centre at its grid point's distance from the image centre through the device's own projection
                std::vector<double> lo(k, 0.0), hi(k, 75.0 * std::numbers::pi / 180.0), az(k), rt(k), dist(k);
                for (int i = 0; i < k; ++i) {
                    const double z = s.means[3 * i + 2], du = s.means[3 * i] / z * s.fx, dv = s.means[3 * i + 1] / z * s.fy;
                    az[i] = std::atan2(dv, du);
                    rt[i] = std::hypot(du, dv);
                    dist[i] = z;
                }
                std::vector<float> in(9ull * k, 0.0f), out(17ull * k);
                for (int it = 0; it < 50; ++it) { // bisection on theta (the projected radius grows with it)
                    for (int i = 0; i < k; ++i) {
                        const double th = 0.5 * (lo[i] + hi[i]);
                        in[9ull * i] = static_cast<float>(std::sin(th) * std::cos(az[i]));
                        in[9ull * i + 1] = static_cast<float>(std::sin(th) * std::sin(az[i]));
                        in[9ull * i + 2] = static_cast<float>(std::cos(th));
                    }
                    fr::fisheye_projection_selftest(s.lens, s.fx, s.fy, s.cx, s.cy, in.data(), k, out.data());
                    for (int i = 0; i < k; ++i) {
                        const double mid = 0.5 * (lo[i] + hi[i]);
                        (std::hypot(out[17ull * i] - s.cx, out[17ull * i + 1] - s.cy) < rt[i] ? lo[i] : hi[i]) = mid;
                    }
                }
                for (int i = 0; i < k; ++i) {
                    const double th = 0.5 * (lo[i] + hi[i]);
                    s.means[3 * i] = static_cast<float>(dist[i] * std::sin(th) * std::cos(az[i]));
                    s.means[3 * i + 1] = static_cast<float>(dist[i] * std::sin(th) * std::sin(az[i]));
                    s.means[3 * i + 2] = static_cast<float>(dist[i] * std::cos(th));
                }
            }
            return s;
        }

        // the fisheye scene's per-splat mean2d (2), Jacobian rows (3 + 3) from the device's own projection
        // (fisheye_projection_selftest = the preprocess's fisheye::terms_lens + project; w2c is the identity)
        std::vector<float> cut_proj(const CutScene& s) {
            std::vector<float> in(9ull * s.n, 0.0f), out(17ull * s.n), pj(8ull * s.n);
            for (int i = 0; i < s.n; ++i)
                for (int a = 0; a < 3; ++a)
                    in[9ull * i + a] = s.means[3ull * i + a];
            fr::fisheye_projection_selftest(s.lens, s.fx, s.fy, s.cx, s.cy, in.data(), s.n, out.data());
            for (int i = 0; i < s.n; ++i)
                for (int a = 0; a < 8; ++a)
                    pj[8ull * i + a] = out[17ull * i + a];
            return pj;
        }

        CutRun cut_render(const CutScene& s, const std::vector<float>* w_img) {
            const int N = s.n;
            const size_t HW = static_cast<size_t>(s.W) * s.H;
            YccDev dv;
            float* dm = dv.alloc(s.means);
            float* ds = dv.alloc(s.scales);
            float* dr = dv.alloc(s.rots);
            float* dop = dv.alloc(s.opac);
            float* dsh = dv.alloc(s.sh0);
            const std::vector<float> w2c = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            float* dw2c = dv.alloc(w2c);
            float* dcam = dv.zeros(3);
            float* dimg = dv.zeros(3 * HW);
            float* dal = dv.zeros(HW);
            float* ddep = dv.zeros(HW);
            float* dbg = dv.alloc({s.bg[0], s.bg[1], s.bg[2]});
            auto ctx = fr::forward_raw(dm, ds, dr, dop, dsh, nullptr, dw2c, dcam, dimg, dal, ddep, nullptr, dbg, nullptr, N, 1, 1, s.W,
                                       s.H, s.fx, s.fy, s.cx, s.cy, 0.01f, 1e10f, false, nullptr, nullptr, 0, 0, nullptr, 1.0f, s.lens);
            if (!ctx.success) {
                std::println(stderr, "forward failed: {}", ctx.error_message ? ctx.error_message : "?");
                std::exit(1);
            }
            cudaDeviceSynchronize();
            CutRun r;
            r.img = YccDev::get(dimg, 3 * HW);
            r.alpha = YccDev::get(dal, HW);
            if (!w_img) {
                fr::release_forward_context(ctx);
                cudaDeviceSynchronize();
                return r;
            }
            float* dgi = dv.alloc(*w_img);
            float* g[5] = {dv.zeros(3ull * N), dv.zeros(3ull * N), dv.zeros(4ull * N), dv.zeros(N), dv.zeros(3ull * N)};
            float* vals[5] = {dm, ds, dr, dop, dsh};
            const int attrs[5] = {3, 3, 4, 1, 3};
            fr::FusedAdamSettings fa;
            fa.enabled = true;
            fr::FusedAdamParam* prm[5] = {&fa.means, &fa.scaling, &fa.rotation, &fa.opacity, &fa.sh0};
            for (int b = 0; b < 5; ++b) {
                prm[b]->gradient_out = g[b];
                prm[b]->param = vals[b];
                prm[b]->n_primitives = N;
                prm[b]->n_elements = N * attrs[b];
                prm[b]->n_attributes = attrs[b];
            }
            fr::BackgroundAlphaGradient bgg; // the scene renders over bg: its -bg dT term, as the trainer passes it
            bgg.bg_color = dbg;
            const auto res = fr::backward_raw(nullptr, nullptr, dgi, bgg, nullptr, nullptr, dimg, dal, dm, ds, dr, dop,
                                              nullptr, dw2c, dcam, ctx, nullptr, N, ctx.n_visible, 1, 1, s.W, s.H, s.fx, s.fy, s.cx, s.cy,
                                              false, DensificationType::None, &fa);
            cudaDeviceSynchronize();
            if (!res.success) {
                std::println(stderr, "backward failed: {}", res.error_message ? res.error_message : "?");
                std::exit(1);
            }
            for (int b = 0; b < 5; ++b)
                r.g[b] = YccDev::get(g[b], static_cast<size_t>(attrs[b]) * N);
            return r;
        }

        // the scene's 2D means and conics (pinhole EWA + 0.3 px^2, as the preprocess) and opacities, in double
        void cut_conics(const CutScene& s, std::vector<std::array<double, 6>>& out) {
            out.clear();
            const std::vector<float> pj = s.fish ? cut_proj(s) : std::vector<float>{}; // fisheye: the device's mean2d and J
            for (int i = 0; i < s.n; ++i) {
                const double X = s.means[3 * i], Y = s.means[3 * i + 1], Z = s.means[3 * i + 2];
                double q[4] = {s.rots[4 * i], s.rots[4 * i + 1], s.rots[4 * i + 2], s.rots[4 * i + 3]};
                const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
                for (double& v : q)
                    v /= qn;
                const double r = q[0], x = q[1], y = q[2], z = q[3];
                const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y), 2 * (x * y + r * z), 1 - 2 * (x * x + z * z),
                                     2 * (y * z - r * x), 2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)};
                double var[3];
                for (int a = 0; a < 3; ++a)
                    var[a] = std::exp(2.0 * s.scales[3 * i + a]);
                double S[9];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        S[3 * a + b] = R[3 * a] * var[0] * R[3 * b] + R[3 * a + 1] * var[1] * R[3 * b + 1] + R[3 * a + 2] * var[2] * R[3 * b + 2];
                double J[6] = {s.fx / Z, 0, -s.fx * X / (Z * Z), 0, s.fy / Z, -s.fy * Y / (Z * Z)};
                if (s.fish)
                    for (int a = 0; a < 6; ++a)
                        J[a] = pj[8ull * i + 2 + a];
                double C[3] = {0, 0, 0};
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        C[0] += J[a] * S[3 * a + b] * J[b];
                        C[1] += J[a] * S[3 * a + b] * J[3 + b];
                        C[2] += J[3 + a] * S[3 * a + b] * J[3 + b];
                    }
                C[0] += 0.3;
                C[2] += 0.3;
                const double det = C[0] * C[2] - C[1] * C[1];
                out.push_back({s.fish ? pj[8ull * i] : s.fx * X / Z + s.cx, s.fish ? pj[8ull * i + 1] : s.fy * Y / Z + s.cy, C[2] / det,
                               -C[1] / det, C[0] / det,
                               1.0 / (1.0 + std::exp(-static_cast<double>(s.opac[i])))});
            }
        }

        int run_cutcheck(const std::filesystem::path& out_dir, const float X, const std::string& table) {
            std::filesystem::create_directories(out_dir);
            if (!table.empty()) { // through the native fisheye and this radial lens table
                try {
                    lfs::training::activate_radial_lens(lfs::training::load_radial_lens_table(table));
                } catch (const std::exception& e) {
                    std::println(stderr, "{}", e.what());
                    return 1;
                }
                std::println("lens: native fisheye through {}", lfs::training::active_radial_lens()->describe());
            }
            const CutScene s = make_cut_scene(3, !table.empty());
            const size_t HW = static_cast<size_t>(s.W) * s.H;
            const auto put = [&](const std::string& n, const std::vector<float>& v) {
                std::ofstream f(out_dir / (n + ".f32"), std::ios::binary);
                f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
            };
            {
                std::ofstream f(out_dir / "cut_scene.txt");
                f << std::format("W {}\nH {}\nfx {:.9g}\nfy {:.9g}\ncx {:.9g}\ncy {:.9g}\nbg {:.9g} {:.9g} {:.9g}\nN {}\nX {:.9g}\n", s.W, s.H, s.fx,
                                 s.fy, s.cx, s.cy, s.bg[0], s.bg[1], s.bg[2], s.n, X);
                f << "lens " << (s.fish ? "fisheye" : "pinhole") << "\n";
            }
            if (s.fish) {
                std::ofstream f(out_dir / "proj.f32", std::ios::binary);
                const auto pj = cut_proj(s);
                f.write(reinterpret_cast<const char*>(pj.data()), static_cast<std::streamsize>(pj.size() * sizeof(float)));
            }
            put("means", s.means);
            put("scales", s.scales);
            put("rots", s.rots);
            put("opac", s.opac);
            put("sh0", s.sh0);
            int bad = 0;
            fr::set_splat_cutoff(0.0f);
            const auto off = cut_render(s, nullptr);
            const auto cut = fr::set_splat_cutoff(X);
            const auto on = cut_render(s, nullptr);
            put("img_off", off.img);
            put("alpha_off", off.alpha);
            put("img_cut", on.img);
            put("alpha_cut", on.alpha);
            double dmax = 0;
            for (size_t i = 0; i < 3 * HW; ++i)
                dmax = std::max(dmax, static_cast<double>(std::abs(on.img[i] - off.img[i])));
            std::println("cut {:.7f} sigma: half z^2 max {:.7f}, min alpha {:.7f}; renders written to {} (the cut changes the image by up to {:.4f})",
                         X, cut.half_z2_max, cut.min_alpha, out_dir.string(), dmax);
            if (!(dmax > 1e-3))
                ++bad;
            // weights: smooth, zero in bands around each splat's cut ellipse and alpha contours
            std::vector<std::array<double, 6>> cs;
            cut_conics(s, cs);
            std::vector<float> w(3 * HW);
            size_t banded = 0, ring = 0;
            const double hz = 0.5 * X * X, band = 0.3;
            for (int y = 0; y < s.H; ++y)
                for (int x = 0; x < s.W; ++x) {
                    bool keep = true;
                    for (const auto& c : cs) {
                        const double dx = c[0] - (x + 0.5), dy = c[1] - (y + 0.5);
                        const double h2 = 0.5 * (c[2] * dx * dx + c[4] * dy * dy) + c[3] * dx * dy; // z^2 / 2
                        const double ha = std::log(c[5] / cut.min_alpha), h255 = std::log(c[5] * 255.0);
                        if (std::abs(h2 - hz) < band || std::abs(h2 - ha) < band || std::abs(h2 - h255) < band)
                            keep = false;
                        if (h2 > std::min(hz, ha) + band && h2 < h255 - band)
                            ++ring;
                    }
                    banded += !keep;
                    for (int ch = 0; ch < 3; ++ch)
                        w[ch * HW + y * s.W + x] = keep ? static_cast<float>(0.3 + std::sin(0.11 * x + 1.7 * ch) * std::cos(0.09 * y + 0.5 * ch)) : 0.0f;
                }
            put("w_img", w);
            const auto an = cut_render(s, &w);
            const char* names[5] = {"means", "scales", "rotations", "opacities", "sh0"};
            const double steps[5] = {5e-4, 5e-4, 1e-3, 1e-3, 1e-2};
            std::println("gradient check with the cut ({} of {} pixels banded out; {} pixel-splat pairs in the ring the cut removes carry weight): "
                         "relative error per block (|fd - analytic| / |analytic|)",
                         banded, HW, ring);
            double worst = 0;
            for (int b = 0; b < 5; ++b) {
                double nd = 0, na = 0;
                const int attrs = b == 2 ? 4 : (b == 3 ? 1 : 3);
                for (int i = 0; i < s.n; ++i)
                    for (int a = 0; a < attrs; ++a) {
                        auto sp = s, sm = s;
                        std::vector<float>* vp[5] = {&sp.means, &sp.scales, &sp.rots, &sp.opac, &sp.sh0};
                        std::vector<float>* vm[5] = {&sm.means, &sm.scales, &sm.rots, &sm.opac, &sm.sh0};
                        (*vp[b])[attrs * i + a] += static_cast<float>(steps[b]);
                        (*vm[b])[attrs * i + a] -= static_cast<float>(steps[b]);
                        const double hreal = static_cast<double>((*vp[b])[attrs * i + a]) - (*vm[b])[attrs * i + a];
                        const auto rp = cut_render(sp, nullptr), rm = cut_render(sm, nullptr);
                        double lp = 0, lm = 0;
                        for (size_t p = 0; p < 3 * HW; ++p) {
                            lp += static_cast<double>(rp.img[p]) * w[p];
                            lm += static_cast<double>(rm.img[p]) * w[p];
                        }
                        const double fd = (lp - lm) / hreal, ag = an.g[b][attrs * i + a];
                        nd += (fd - ag) * (fd - ag);
                        na += ag * ag;
                    }
                const double e = std::sqrt(nd) / std::max(std::sqrt(na), 1e-30);
                worst = std::max(worst, e);
                std::println("   {:<10} {:.3e}  (|analytic| {:.4e})", names[b], e, std::sqrt(na));
            }
            if (worst > 2e-2)
                ++bad;
            fr::set_splat_cutoff(0.0f);
            const auto an_off = cut_render(s, &w);
            fr::set_splat_cutoff(X);
            double nd = 0, na = 0;
            for (int b = 0; b < 5; ++b)
                for (size_t i = 0; i < an.g[b].size(); ++i) {
                    const double d = static_cast<double>(an.g[b][i]) - an_off.g[b][i];
                    nd += d * d;
                    na += static_cast<double>(an.g[b][i]) * an.g[b][i];
                }
            const double ctl = std::sqrt(nd / std::max(na, 1e-30));
            std::println("control: the gradient with the cut differs from the uncut backward's by {:.3e} (relative): the backward cuts too", ctl);
            if (!(ctl > 1e-3))
                ++bad;
            // AT the cut boundary: weights only on pixels just inside (z^2/2 in [hz - 0.25, hz - 0.02]) or just outside
            // ([hz + 0.02, hz + 0.25]) one splat's cut, a splat whose alpha floor is well past the cut (log(o / min_alpha) >
            // hz + 0.3), with no other splat reaching the pixel (z^2/2 > log(o / min_alpha) + band for every other one). Steps
            // small enough that no weighted pixel crosses the cut (a step moves z^2/2 there by < 0.015). Inside: the analytic
            // gradient = the central differences; outside: with the cut the analytic gradient and the differences are exactly
            // 0, without it the analytic gradient is not (the uncut backward reaches the ring the cut removes).
            {
                std::vector<float> w_in(3 * HW, 0.0f), w_out(3 * HW, 0.0f);
                size_t n_in = 0, n_out = 0;
                for (int y = 0; y < s.H; ++y)
                    for (int x = 0; x < s.W; ++x) {
                        int owner = -1, others = 0;
                        double ho = 0;
                        for (int i = 0; i < s.n; ++i) {
                            const auto& c = cs[i];
                            const double dx = c[0] - (x + 0.5), dy = c[1] - (y + 0.5);
                            const double h2 = 0.5 * (c[2] * dx * dx + c[4] * dy * dy) + c[3] * dx * dy;
                            const double ha = std::log(c[5] / cut.min_alpha), off = std::abs(h2 - hz);
                            if (owner < 0 && off > 0.02 && off < 0.25 && ha > hz + 0.3) {
                                owner = i;
                                ho = h2;
                            } else if (!(h2 > ha + band))
                                ++others;
                        }
                        if (owner < 0 || others > 0)
                            continue;
                        auto& wt = ho < hz ? w_in : w_out;
                        ++(ho < hz ? n_in : n_out);
                        for (int ch = 0; ch < 3; ++ch)
                            wt[ch * HW + y * s.W + x] = static_cast<float>(0.8 + 0.4 * std::sin(0.11 * x + 1.7 * ch) * std::cos(0.09 * y + 0.5 * ch));
                    }
                const double qsteps[5] = {1.25e-4, 5e-4, 5e-5, 1e-3, 1e-2};
                // -> worst relative block error |fd - analytic| / |analytic| (0 when both vanish); |fd| and max |analytic|
                const auto fd_vs = [&](const std::vector<float>& wt, const CutRun& an, double& fd_norm, double& an_max) {
                    double worst_b = 0;
                    fd_norm = 0;
                    an_max = 0;
                    for (int b = 0; b < 5; ++b) {
                        double nd = 0, na = 0;
                        const int attrs = b == 2 ? 4 : (b == 3 ? 1 : 3);
                        for (int i = 0; i < s.n; ++i)
                            for (int a = 0; a < attrs; ++a) {
                                auto sp = s, sm = s;
                                std::vector<float>* vp[5] = {&sp.means, &sp.scales, &sp.rots, &sp.opac, &sp.sh0};
                                std::vector<float>* vm[5] = {&sm.means, &sm.scales, &sm.rots, &sm.opac, &sm.sh0};
                                (*vp[b])[attrs * i + a] += static_cast<float>(qsteps[b]);
                                (*vm[b])[attrs * i + a] -= static_cast<float>(qsteps[b]);
                                const double hreal = static_cast<double>((*vp[b])[attrs * i + a]) - (*vm[b])[attrs * i + a];
                                const auto rp = cut_render(sp, nullptr), rm = cut_render(sm, nullptr);
                                double lp = 0, lm = 0;
                                for (size_t p = 0; p < 3 * HW; ++p) {
                                    lp += static_cast<double>(rp.img[p]) * wt[p];
                                    lm += static_cast<double>(rm.img[p]) * wt[p];
                                }
                                const double fd = (lp - lm) / hreal, ag = an.g[b][attrs * i + a];
                                nd += (fd - ag) * (fd - ag);
                                na += ag * ag;
                                fd_norm += fd * fd;
                                an_max = std::max(an_max, std::abs(ag));
                            }
                        if (nd > 0)
                            worst_b = std::max(worst_b, std::sqrt(nd) / std::max(std::sqrt(na), 1e-30));
                    }
                    fd_norm = std::sqrt(fd_norm);
                    return worst_b;
                };
                const auto an_in = cut_render(s, &w_in), an_out = cut_render(s, &w_out);
                double fd_in = 0, g_in = 0, fd_out = 0, g_out = 0;
                const double e_in = fd_vs(w_in, an_in, fd_in, g_in);
                fd_vs(w_out, an_out, fd_out, g_out);
                fr::set_splat_cutoff(0.0f);
                const auto an_out_off = cut_render(s, &w_out);
                fr::set_splat_cutoff(X);
                double g_out_off = 0;
                for (int b = 0; b < 5; ++b)
                    for (const float v : an_out_off.g[b])
                        g_out_off = std::max(g_out_off, static_cast<double>(std::abs(v)));
                std::println("at the cut boundary ({} pixels just inside, {} just outside one splat's cut: z^2/2 0.02..0.25 from {:.4f}): "
                             "inside: central differences vs analytic, worst relative block error {:.3e} (|fd| {:.3e}, max |analytic| "
                             "{:.3e}); outside, with the cut: max |analytic| {:.3e}, |fd| {:.3e} (both must be 0); without the cut max "
                             "|analytic| there {:.3e} (must be > 0)",
                             n_in, n_out, hz, e_in, fd_in, g_in, g_out, fd_out, g_out_off);
                if (n_in < 20 || n_out < 20 || !(e_in <= 2e-2) || !(g_in > 0.0) || g_out != 0.0 || fd_out != 0.0 || !(g_out_off > 0.0))
                    ++bad;
            }
            fr::set_splat_cutoff(0.0f);
            std::println("{}", bad == 0 ? "CUTCHECK OK" : "CUTCHECK FAILED");
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
        if (argc >= 4 && std::string(argv[2]) == "ycccheck")
            return run_ycccheck(argv[3]);
        if (argc >= 4 && std::string(argv[2]) == "cutcheck")
            return run_cutcheck(argv[3], argc >= 5 ? std::stof(argv[4]) : 2.2360680f, argc >= 6 ? std::string(argv[5]) : std::string());
        std::println(stderr, "usage: LichtFeld-Studio lfe-fisheye render <model.ply> <cameras.txt> <out_dir> | projcheck | gradcheck | "
                             "radialcheck <lens_radial.txt> [--vs-poly] | cuberaycheck <lens_radial.txt> | ycccheck <out_dir> | "
                             "cutcheck <out_dir> [max_std_dev [lens_radial.txt]]");
        return 2;
    }

    int run_lfe_distortion_tool(int argc, const char* const* argv) {
        if (argc >= 3 && std::string(argv[2]) == "gradcheck")
            return run_distortion_gradcheck();
        std::println(stderr, "usage: LichtFeld-Studio lfe-distortion gradcheck");
        return 2;
    }

} // namespace lfs::app
