/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bg_cubemap.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include <rasterization_config.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr const char* kFaceNames[6] = {"px", "nx", "py", "ny", "pz", "nz"};

        bool check(cudaError_t e, const char* what, std::string& error) {
            if (e == cudaSuccess)
                return true;
            error = std::format("bg cubemap: {} failed: {}", what, cudaGetErrorString(e));
            return false;
        }

        // NumPy .npy v1.0, little-endian float32, C order
        bool write_npy(const std::filesystem::path& p, const float* data, const std::string& shape, size_t count,
                       std::string& error) {
            std::string header = std::format("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}", shape);
            const size_t total = 10 + header.size() + 1;
            header.append((64 - total % 64) % 64, ' ');
            header.push_back('\n');
            std::ofstream os(p, std::ios::binary | std::ios::trunc);
            if (!os) {
                error = std::format("bg cubemap: cannot write '{}'", lfs::core::path_to_utf8(p));
                return false;
            }
            const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
            os.write(magic, 8);
            const uint16_t hl = static_cast<uint16_t>(header.size());
            const char hlb[2] = {static_cast<char>(hl & 0xff), static_cast<char>(hl >> 8)};
            os.write(hlb, 2);
            os.write(header.data(), static_cast<std::streamsize>(header.size()));
            os.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
            if (!os) {
                error = std::format("bg cubemap: write failed '{}'", lfs::core::path_to_utf8(p));
                return false;
            }
            return true;
        }

        // NumPy .npy (v1/v2/v3), little-endian float32 or float64, C order -> float32 values + shape
        bool read_npy(const std::filesystem::path& p, std::vector<float>& out, std::vector<size_t>& shape, std::string& error) {
            std::ifstream is(p, std::ios::binary);
            if (!is) {
                error = std::format("cannot open '{}'", lfs::core::path_to_utf8(p));
                return false;
            }
            char magic[8] = {};
            is.read(magic, 8);
            if (!is || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
                error = std::format("'{}' is not a .npy file", lfs::core::path_to_utf8(p));
                return false;
            }
            size_t hl = 0;
            if (magic[6] == 1) {
                unsigned char b[2] = {};
                is.read(reinterpret_cast<char*>(b), 2);
                hl = b[0] | (static_cast<size_t>(b[1]) << 8);
            } else {
                unsigned char b[4] = {};
                is.read(reinterpret_cast<char*>(b), 4);
                hl = b[0] | (static_cast<size_t>(b[1]) << 8) | (static_cast<size_t>(b[2]) << 16) | (static_cast<size_t>(b[3]) << 24);
            }
            std::string header(hl, '\0');
            is.read(header.data(), static_cast<std::streamsize>(hl));
            if (!is) {
                error = "truncated .npy header";
                return false;
            }
            const bool f4 = header.find("'<f4'") != std::string::npos, f8 = header.find("'<f8'") != std::string::npos;
            if (!f4 && !f8) {
                error = std::format("'{}': dtype must be little-endian float32 or float64 (header {})", lfs::core::path_to_utf8(p), header);
                return false;
            }
            if (header.find("'fortran_order': False") == std::string::npos) {
                error = "fortran-ordered .npy is not supported";
                return false;
            }
            const auto a = header.find('('), b = header.find(')', a == std::string::npos ? 0 : a);
            if (a == std::string::npos || b == std::string::npos) {
                error = "no shape in the .npy header";
                return false;
            }
            shape.clear();
            size_t count = 1;
            for (size_t i = a + 1; i < b;) {
                while (i < b && (header[i] == ' ' || header[i] == ','))
                    ++i;
                if (i >= b)
                    break;
                size_t v = 0;
                while (i < b && header[i] >= '0' && header[i] <= '9')
                    v = v * 10 + static_cast<size_t>(header[i++] - '0');
                shape.push_back(v);
                count *= v;
                while (i < b && header[i] != ',')
                    ++i;
            }
            out.resize(count);
            if (f4) {
                is.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
            } else {
                std::vector<double> d(count);
                is.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(count * sizeof(double)));
                for (size_t i = 0; i < count; ++i)
                    out[i] = static_cast<float>(d[i]);
            }
            if (!is) {
                error = std::format("'{}': truncated data ({} values expected)", lfs::core::path_to_utf8(p), count);
                return false;
            }
            return true;
        }

        // Host versions of the kernel's face table / seam rule (kernels/bg_cubemap.cu), for resampling an init cubemap.
        void dir_to_face_h(const double x, const double y, const double z, int& f, double& s, double& t) {
            const double ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
            double ma, sc, tc;
            if (ax >= ay && ax >= az) {
                ma = ax;
                f = x > 0 ? 0 : 1;
                sc = x > 0 ? -z : z;
                tc = -y;
            } else if (ay >= az) {
                ma = ay;
                f = y > 0 ? 2 : 3;
                sc = x;
                tc = y > 0 ? z : -z;
            } else {
                ma = az;
                f = z > 0 ? 4 : 5;
                sc = z > 0 ? x : -x;
                tc = -y;
            }
            ma = std::max(ma, 1e-30);
            s = std::clamp(sc / ma, -1.0, 1.0);
            t = std::clamp(tc / ma, -1.0, 1.0);
        }

        void face_to_dir_h(const int f, const double s, const double t, double d[3]) {
            switch (f) {
            case 0: d[0] = 1; d[1] = -t; d[2] = -s; break;
            case 1: d[0] = -1; d[1] = -t; d[2] = s; break;
            case 2: d[0] = s; d[1] = 1; d[2] = t; break;
            case 3: d[0] = s; d[1] = -1; d[2] = -t; break;
            case 4: d[0] = s; d[1] = -t; d[2] = 1; break;
            default: d[0] = -s; d[1] = -t; d[2] = -1; break;
            }
        }

        size_t texel_index_h(const int f, const int i, const int j, const int R) {
            if (i >= 0 && i < R && j >= 0 && j < R)
                return (static_cast<size_t>(f) * R + j) * R + i;
            double d[3];
            face_to_dir_h(f, (2.0 * i + 1.0) / R - 1.0, (2.0 * j + 1.0) / R - 1.0, d);
            int f2;
            double s2, t2;
            dir_to_face_h(d[0], d[1], d[2], f2, s2, t2);
            const int i2 = std::clamp(static_cast<int>(std::floor((s2 + 1.0) * 0.5 * R)), 0, R - 1);
            const int j2 = std::clamp(static_cast<int>(std::floor((t2 + 1.0) * 0.5 * R)), 0, R - 1);
            return (static_cast<size_t>(f2) * R + j2) * R + i2;
        }

        // Every texel of a res x res cubemap = the source cubemap (R0) sampled bilinearly at that texel's centre direction.
        std::vector<float> resample_cubemap(const std::vector<float>& src, const int R0, const int R) {
            std::vector<float> out(static_cast<size_t>(6) * R * R * 3);
            for (int f = 0; f < 6; ++f)
                for (int j = 0; j < R; ++j)
                    for (int i = 0; i < R; ++i) {
                        double d[3];
                        face_to_dir_h(f, (2.0 * i + 1.0) / R - 1.0, (2.0 * j + 1.0) / R - 1.0, d);
                        int g;
                        double s, t;
                        dir_to_face_h(d[0], d[1], d[2], g, s, t);
                        const double x = (s + 1.0) * 0.5 * R0 - 0.5, y = (t + 1.0) * 0.5 * R0 - 0.5;
                        const double x0 = std::floor(x), y0 = std::floor(y), wx = x - x0, wy = y - y0;
                        const int i0 = static_cast<int>(x0), j0 = static_cast<int>(y0);
                        const size_t k[4] = {texel_index_h(g, i0, j0, R0), texel_index_h(g, i0 + 1, j0, R0),
                                             texel_index_h(g, i0, j0 + 1, R0), texel_index_h(g, i0 + 1, j0 + 1, R0)};
                        const double w[4] = {(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy};
                        for (int c = 0; c < 3; ++c) {
                            double v = 0;
                            for (int q = 0; q < 4; ++q)
                                v += w[q] * src[k[q] * 3 + c];
                            out[((static_cast<size_t>(f) * R + j) * R + i) * 3 + c] = static_cast<float>(v);
                        }
                    }
            return out;
        }
    } // namespace

    bool BackgroundCubemap::dump_last(const std::filesystem::path& path, std::string& error, cudaStream_t stream) {
        if (!bg_.is_valid()) {
            error = "bg cubemap dump: nothing rendered yet";
            return false;
        }
        const size_t H = bg_.shape()[1], W = bg_.shape()[2];
        std::vector<float> bg(3 * H * W), tex(n_);
        if (!check(cudaMemcpyAsync(bg.data(), bg_.ptr<float>(), bg.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "dump readback", error) ||
            !check(cudaMemcpyAsync(tex.data(), texels_, n_ * sizeof(float), cudaMemcpyDeviceToHost, stream), "dump readback",
                   error) ||
            !check(cudaStreamSynchronize(stream), "dump sync", error))
            return false;
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        return write_npy(path, bg.data(), std::format("3, {}, {}", H, W), bg.size(), error) &&
               write_npy(std::filesystem::path(path).concat(".faces.npy"), tex.data(),
                         std::format("6, {}, {}, 3", settings_.res, settings_.res), n_, error);
    }

    std::unique_ptr<BackgroundCubemap> BackgroundCubemap::create(const BackgroundCubemapSettings& settings, std::string& error) {
        if (settings.res < 4 || settings.res > 4096) {
            error = std::format("--bg-cubemap-res must be within [4, 4096] (got {})", settings.res);
            return nullptr;
        }
        std::unique_ptr<BackgroundCubemap> c(new BackgroundCubemap(settings));
        c->n_ = static_cast<size_t>(6) * settings.res * settings.res * 3;
        const size_t bytes = c->n_ * sizeof(float);
        if (!check(cudaMalloc(reinterpret_cast<void**>(&c->texels_), bytes), "cudaMalloc texels", error) ||
            !check(cudaMalloc(reinterpret_cast<void**>(&c->grad_), bytes), "cudaMalloc grad", error) ||
            !check(cudaMalloc(reinterpret_cast<void**>(&c->adam_m_), bytes), "cudaMalloc adam m", error) ||
            !check(cudaMalloc(reinterpret_cast<void**>(&c->adam_v_), bytes), "cudaMalloc adam v", error) ||
            !check(cudaMalloc(reinterpret_cast<void**>(&c->stats_), 8 * sizeof(float)), "cudaMalloc stats", error))
            return nullptr;
        if (!check(cudaMemset(c->grad_, 0, bytes), "memset", error) || !check(cudaMemset(c->adam_m_, 0, bytes), "memset", error) ||
            !check(cudaMemset(c->adam_v_, 0, bytes), "memset", error) ||
            !check(cudaMemset(c->stats_, 0, 8 * sizeof(float)), "memset", error))
            return nullptr;
        if (settings.init_path.empty()) {
            kernels::cubemap_fill(c->texels_, settings.res, settings.init[0], settings.init[1], settings.init[2], nullptr);
            c->init_host_.resize(c->n_);
            for (size_t k = 0; k < c->n_; ++k)
                c->init_host_[k] = settings.init[k % 3];
            c->init_note_ = std::format("({:.4f}, {:.4f}, {:.4f})", settings.init[0], settings.init[1], settings.init[2]);
        } else {
            // --bg-cubemap-init: a faces.npy as written by save() ([6, R0, R0, 3] float, faces +X -X +Y -Y +Z -Z, the photos'
            // encoding), resampled bilinearly (seam rule as the kernel) when R0 != res; values clamped to [0, 1] as in training.
            std::vector<float> src;
            std::vector<size_t> shape;
            std::string npy_error;
            const auto path = lfs::core::utf8_to_path(settings.init_path);
            if (!read_npy(path, src, shape, npy_error)) {
                error = "--bg-cubemap-init: " + npy_error;
                return nullptr;
            }
            if (shape.size() != 4 || shape[0] != 6 || shape[1] != shape[2] || shape[3] != 3 || shape[1] < 2) {
                std::string sh;
                for (const size_t v : shape)
                    sh += std::format("{}{}", sh.empty() ? "" : ", ", v);
                error = std::format("--bg-cubemap-init: '{}' must hold [6, R, R, 3] (got [{}])", settings.init_path, sh);
                return nullptr;
            }
            const int R0 = static_cast<int>(shape[1]);
            size_t nonfinite = 0;
            for (float& v : src) {
                if (!std::isfinite(v)) {
                    v = 0.0f;
                    ++nonfinite;
                }
                v = std::clamp(v, 0.0f, 1.0f);
            }
            if (nonfinite) {
                error = std::format("--bg-cubemap-init: '{}' holds {} non-finite values", settings.init_path, nonfinite);
                return nullptr;
            }
            c->init_host_ = R0 == settings.res ? src : resample_cubemap(src, R0, settings.res);
            const std::vector<float>& init = c->init_host_;
            if (!check(cudaMemcpy(c->texels_, init.data(), bytes, cudaMemcpyHostToDevice), "upload init", error))
                return nullptr;
            double mean[3] = {0, 0, 0};
            for (size_t k = 0; k < init.size(); ++k)
                mean[k % 3] += init[k];
            const double inv = 3.0 / static_cast<double>(init.size());
            c->init_note_ = std::format("'{}' ({} x {} per face{}; mean ({:.4f}, {:.4f}, {:.4f}))", settings.init_path, R0, R0,
                                        R0 == settings.res ? "" : std::format(", resampled to {}", settings.res),
                                        mean[0] * inv, mean[1] * inv, mean[2] * inv);
        }
        if (!check(cudaDeviceSynchronize(), "init", error))
            return nullptr;
        return c;
    }

    BackgroundCubemap::~BackgroundCubemap() {
        for (float* p : {texels_, grad_, adam_m_, adam_v_, stats_})
            if (p)
                cudaFree(p);
    }

    std::string BackgroundCubemap::description() const {
        return std::format("Learned background cubemap (--bg-mode cubemap): 6 x {} x {} RGB texels initialised to {}; Adam lr {:g} "
                           "decaying to {:g} at the last iteration; smoothness weight {:g}; drawn through each "
                           "camera's lens and rolling-shutter row pose; exported to <output>/bg_cubemap/",
                           settings_.res, settings_.res, init_note_, settings_.lr, settings_.lr * settings_.lr_final_factor,
                           settings_.smooth);
    }

    const lfs::core::Tensor& BackgroundCubemap::render(const lfs::core::Camera& camera,
                                                       const fast_lfs::rasterization::FastGSRollingShutter& rs,
                                                       cudaStream_t stream, const float* w2c_override) {
        const int W = camera.image_width();
        const int H = camera.image_height();
        const auto [fx, fy, cx, cy] = camera.get_intrinsics();
        kernels::CubemapView v{};
        v.width = W;
        v.height = H;
        v.fx = fx;
        v.fy = fy;
        v.cx = cx;
        v.cy = cy;
        const auto lens = fastgs_lens_for_camera(camera); // the rasterizer's own lens (pinhole when undistorted)
        if (lens.model == fast_lfs::rasterization::kLensFisheye) {
            v.fisheye = 1;
            v.k1 = lens.k1;
            v.k2 = lens.k2;
            v.k3 = lens.k3;
            v.k4 = lens.k4;
            v.theta_max = static_cast<float>(std::atan(std::sqrt(static_cast<double>(lens.max_w))));
            // LFQPLC (--lens-radial): the rasterizer's radial lens table (its device view; seg == nullptr without the
            // table). fastgs_lens_for_camera refuses any camera that is not the table's camera while a table is active,
            // and computed theta_max (the culling bound) on the table's slope, so the pixel rays invert exactly the lens
            // the splats are projected through.
            v.radial = lens.radial;
            radial_used_ = radial_used_ || lens.radial.seg != nullptr;
        }
        camera.world_view_transform().sync_to_stream(stream);
        // LFQPLC (--pose-opt): the refined world-to-camera transform the rasterizer draws this camera with (device [16],
        // composed by CameraPoseRefinement::options_for on this stream); null = the camera's own transform
        v.w2c = w2c_override != nullptr ? w2c_override : camera.world_view_transform_ptr();
        refined_pose_used_ = refined_pose_used_ || w2c_override != nullptr;
        if (rs.motion != nullptr && rs.readout != nullptr) {
            v.motion = rs.motion;
            v.readout = rs.readout;
            v.row_offset = rs.row_offset;
            v.sensor_height = rs.sensor_height;
            v.time_map = rs.time_map; // --rs-time-map (LFQPLCYKMR); neither form = the row rule
            v.time_map_nx = rs.time_map_nx;
            v.time_map_ny = rs.time_map_ny;
            v.time_radial = rs.time_radial;
            v.tr_k = rs.tr_k;
            v.tr_a = rs.tr_a;
            v.tr_b = rs.tr_b;
            v.tr_c = rs.tr_c;
            v.tr_cx = rs.tr_cx;
            v.tr_cy = rs.tr_cy;
            v.tr_aspect = rs.tr_aspect;
            v.tr_s0 = rs.tr_s0;
        }
        const lfs::core::TensorShape shape{3, static_cast<size_t>(H), static_cast<size_t>(W)};
        if (!bg_.is_valid() || bg_.shape() != shape)
            bg_ = lfs::core::Tensor::empty(shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        if (bg_.stream() != stream)
            bg_.set_stream(stream);
        kernels::cubemap_render_background(texels_, settings_.res, v, bg_.ptr<float>(), stream);
        view_ = v;
        have_view_ = true;
        return bg_;
    }

    void BackgroundCubemap::backward(const lfs::core::Tensor& grad_image, const lfs::core::Tensor& alpha, cudaStream_t stream) {
        if (!have_view_)
            return;
        const size_t HW = static_cast<size_t>(view_.width) * static_cast<size_t>(view_.height);
        if (!grad_image.is_valid() || grad_image.numel() != 3 * HW || !alpha.is_valid() || alpha.numel() != HW) {
            LOG_ERROR("bg cubemap backward: shape mismatch (grad {} values, alpha {} values, view {}x{}); step skipped",
                      grad_image.is_valid() ? grad_image.numel() : 0, alpha.is_valid() ? alpha.numel() : 0, view_.width,
                      view_.height);
            return;
        }
        const auto g = grad_image.contiguous();
        const auto a = alpha.contiguous();
        g.sync_to_stream(stream);
        a.sync_to_stream(stream);
        kernels::cubemap_backward(g.ptr<float>(), a.ptr<float>(), settings_.res, view_, grad_, stream);
        ++backward_calls_;
        have_view_ = false;
    }

    void BackgroundCubemap::step(const int iter, const int total_iters, const bool log, cudaStream_t stream) {
        if (log)
            cudaMemsetAsync(stats_, 0, 4 * sizeof(float), stream); // [0..2] grad stats, [3] smoothness loss
        if (settings_.smooth > 0.0f)
            kernels::cubemap_smooth_grad(texels_, settings_.res, settings_.smooth, grad_, log ? stats_ + 3 : nullptr, stream);
        if (log)
            kernels::cubemap_grad_stats(grad_, static_cast<int>(n_), stats_, stream);
        ++adam_steps_;
        const double frac = total_iters > 0 ? std::clamp(static_cast<double>(iter) / total_iters, 0.0, 1.0) : 0.0;
        const float lr = static_cast<float>(settings_.lr * std::pow(static_cast<double>(settings_.lr_final_factor), frac));
        constexpr float b1 = 0.9f, b2 = 0.999f, eps = 1e-15f;
        const float bias1 = static_cast<float>(1.0 - std::pow(0.9, static_cast<double>(adam_steps_)));
        const float bias2 = static_cast<float>(1.0 - std::pow(0.999, static_cast<double>(adam_steps_)));
        kernels::cubemap_adam_step(texels_, grad_, adam_m_, adam_v_, static_cast<int>(n_), lr, b1, b2, eps, bias1, bias2, 0.0f,
                                   1.0f, stats_ + 4, stream);
        last_lr_ = lr;
        if (log) {
            float host[5] = {};
            cudaMemcpyAsync(host, stats_, sizeof(host), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            last_grad_sum_ = host[0];
            std::memcpy(&last_grad_max_, &host[1], sizeof(float));
            last_nonfinite_stats_ += host[2];
            last_smooth_loss_ = host[3];
            last_log_iter_ = iter;
        }
    }

    std::string BackgroundCubemap::summary(cudaStream_t stream) {
        std::vector<float> h(n_);
        float nonfinite_adam = 0.0f;
        cudaMemcpyAsync(h.data(), texels_, n_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&nonfinite_adam, stats_ + 4, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        const size_t texels = n_ / 3;
        double sum[3] = {0, 0, 0};
        float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
        size_t moved = 0, nonfinite = 0;
        double moved_sum[3] = {0, 0, 0};
        for (size_t t = 0; t < texels; ++t) {
            bool m = false;
            for (int c = 0; c < 3; ++c) {
                const float x = h[t * 3 + c];
                if (!std::isfinite(x)) {
                    ++nonfinite;
                    continue;
                }
                sum[c] += x;
                mn[c] = std::min(mn[c], x);
                mx[c] = std::max(mx[c], x);
                if (std::fabs(x - init_host_[t * 3 + c]) > 0.5f / 255.0f)
                    m = true;
            }
            if (m) {
                ++moved;
                for (int c = 0; c < 3; ++c)
                    moved_sum[c] += h[t * 3 + c];
            }
        }
        const double inv = 1.0 / static_cast<double>(texels);
        const double minv = moved ? 1.0 / static_cast<double>(moved) : 0.0;
        return std::format("bg cubemap: mean RGB ({:.4f}, {:.4f}, {:.4f}), min ({:.3f}, {:.3f}, {:.3f}), max ({:.3f}, {:.3f}, {:.3f}); "
                           "{:.1f}% of texels moved > 0.5/255 from the init (their mean ({:.4f}, {:.4f}, {:.4f})); "
                           "non-finite texels {}, non-finite gradients seen {} (Adam) {} (logged); at iter {}: gradient sum|g| {:.4g} "
                           "max|g| {:.4g}, smoothness loss {:.4g}, lr {:.4g}; {} backward passes, {} Adam steps; pixel rays "
                           "through {}, camera pose {}",
                           sum[0] * inv, sum[1] * inv, sum[2] * inv, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
                           100.0 * static_cast<double>(moved) * inv, moved_sum[0] * minv, moved_sum[1] * minv,
                           moved_sum[2] * minv, nonfinite, nonfinite_adam, last_nonfinite_stats_, last_log_iter_,
                           last_grad_sum_, last_grad_max_, last_smooth_loss_, last_lr_, backward_calls_, adam_steps_,
                           radial_used_ ? "the radial lens table (--lens-radial)" : "the cameras' own lens",
                           refined_pose_used_ ? "refined (--pose-opt) in renders so far" : "the dataset's");
    }

    bool BackgroundCubemap::write_tensor_npy(const std::filesystem::path& path, const lfs::core::Tensor& t, std::string& error,
                                             cudaStream_t stream) {
        if (!t.is_valid() || t.numel() == 0) {
            error = std::format("bg cubemap dump: no tensor for '{}'", lfs::core::path_to_utf8(path));
            return false;
        }
        auto c = t.contiguous();
        if (c.dtype() != lfs::core::DataType::Float32)
            c = c.to(lfs::core::DataType::Float32);
        std::string shape;
        for (size_t i = 0; i < c.ndim(); ++i)
            shape += std::format("{}{}", i ? ", " : "", c.shape()[i]);
        if (c.ndim() == 1)
            shape += ",";
        std::vector<float> h(c.numel());
        if (c.device() == lfs::core::Device::CUDA) {
            c.sync_to_stream(stream);
            if (!check(cudaMemcpyAsync(h.data(), c.ptr<float>(), h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                       "dump readback", error) ||
                !check(cudaStreamSynchronize(stream), "dump sync", error))
                return false;
        } else {
            std::memcpy(h.data(), c.ptr<float>(), h.size() * sizeof(float));
        }
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        return write_npy(path, h.data(), shape, h.size(), error);
    }

    bool BackgroundCubemap::save(const std::filesystem::path& dir, const int iter, const std::string& note,
                                 std::string& error, cudaStream_t stream) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            error = std::format("bg cubemap: cannot create '{}': {}", lfs::core::path_to_utf8(dir), ec.message());
            return false;
        }
        std::vector<float> h(n_);
        if (!check(cudaMemcpyAsync(h.data(), texels_, n_ * sizeof(float), cudaMemcpyDeviceToHost, stream), "readback", error) ||
            !check(cudaStreamSynchronize(stream), "readback sync", error))
            return false;
        const int R = settings_.res;
        const size_t face = static_cast<size_t>(R) * R * 3;
        std::vector<uint8_t> u8(face);
        for (int f = 0; f < 6; ++f) {
            for (size_t k = 0; k < face; ++k) {
                const float x = h[f * face + k];
                const float c = std::isfinite(x) ? std::clamp(x, 0.0f, 1.0f) : 0.0f;
                u8[k] = static_cast<uint8_t>(std::lround(c * 255.0f));
            }
            const auto p = dir / (std::string(kFaceNames[f]) + ".png");
            if (!lfs::core::save_png(p, u8.data(), R, R, 3, 8, 6)) {
                error = std::format("bg cubemap: cannot write '{}'", lfs::core::path_to_utf8(p));
                return false;
            }
        }
        if (!write_npy(dir / "faces.npy", h.data(), std::format("6, {}, {}, 3", R, R), n_, error))
            return false;
        {
            const auto p = dir / "convention.txt";
            std::ofstream os(p, std::ios::trunc);
            os << "LichtFeld LFC learned background cubemap (--bg-mode cubemap), iteration " << iter << "\n"
               << note << "\n"
               << "Resolution: 6 faces x " << R << " x " << R << " texels, RGB.\n"
               << "Files: px.png nx.png py.png ny.png pz.png nz.png = faces +X -X +Y -Y +Z -Z (8-bit, the photos' own sRGB\n"
               << "  encoding: texel value * 255, clamped to [0, 1] during training); faces.npy = float32 array (6, R, R, 3) in\n"
               << "  the same face order, [face][row][column][rgb], values in [0, 1] (same encoding, not linear light).\n"
               << "Coordinates: the dataset's WORLD axes (the COLMAP reconstruction the cameras and the exported PLY live in;\n"
               << "  COLMAP / OpenCV camera axes: x right, y down, z forward). A world direction d = (x, y, z) is looked up on\n"
               << "  the face of its largest |component|, with s, t in [-1, 1]:\n"
               << "    +X (px): s = -z/|x|  t = -y/|x|       -X (nx): s =  z/|x|  t = -y/|x|\n"
               << "    +Y (py): s =  x/|y|  t =  z/|y|       -Y (ny): s =  x/|y|  t = -z/|y|\n"
               << "    +Z (pz): s =  x/|z|  t = -y/|z|       -Z (nz): s = -x/|z|  t = -y/|z|\n"
               << "  column = (s + 1) / 2 * R, row = (t + 1) / 2 * R, row 0 at the TOP of each PNG, texel centres at half-integers.\n"
               << "  This is the standard cube-map face table (OpenGL / Vulkan / D3D / wgpu), so the six images load as a wgpu\n"
               << "  cube texture (layers +X -X +Y -Y +Z -Z) and are sampled with the world direction expressed in these\n"
               << "  dataset axes: if the game maps dataset points p to game points M p (rotation part Rm), sample the cube\n"
               << "  with Rm^T * (game view direction).\n"
               << "Training: per pixel, image = splats + (1 - alpha) * cube(direction of that pixel's ray through the camera's\n"
               << "  lens and the rolling-shutter pose of its row), bilinear across face edges; the composited image goes\n"
               << "  through the per-photo appearance model (exposure correction) and the loss like the splats, so the cubemap\n"
               << "  is the scene's sky in the same space as the splat colours (the PLY).\n";
            if (!os) {
                error = std::format("bg cubemap: cannot write '{}'", lfs::core::path_to_utf8(p));
                return false;
            }
        }
        return true;
    }

} // namespace lfs::training
