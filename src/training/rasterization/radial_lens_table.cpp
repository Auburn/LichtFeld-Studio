/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "radial_lens_table.hpp"

#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <format>
#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace lfs::training {

    namespace {
        std::shared_ptr<RadialLensTable> g_active;

        double num(const std::string& tok, const std::filesystem::path& file, int line_no) {
            char* end = nullptr;
            const double v = std::strtod(tok.c_str(), &end);
            if (end == tok.c_str() || *end != '\0' || !std::isfinite(v))
                throw std::runtime_error(std::format("--lens-radial {}: line {}: '{}' is not a number", file.string(), line_no, tok));
            return v;
        }

        bool close_rel(double a, double b, double rel) { return std::abs(a - b) <= 1e-9 + rel * std::abs(b); }
    } // namespace

    core::RadialLensView RadialLensTable::host_view() const {
        return {seg.data(), n, inv_dth_f, th_end_f, A_f, B_f};
    }

    core::RadialLensView RadialLensTable::device_view() const {
        return {d_seg, n, inv_dth_f, th_end_f, A_f, B_f};
    }

    std::string RadialLensTable::describe() const {
        return std::format("{} ({} nodes to {:.3f} deg, valid to {:.3f} deg; camera {}x{} fx {:.5f} fy {:.5f} cx {:.5f} cy {:.5f}, "
                           "k {:.8g} {:.8g} {:.8g} {:.8g})",
                           path.string(), n, th_end * 180.0 / std::numbers::pi, valid * 180.0 / std::numbers::pi, width, height,
                           fx, fy, cx, cy, k[0], k[1], k[2], k[3]);
    }

    std::shared_ptr<RadialLensTable> load_radial_lens_table(const std::filesystem::path& file) {
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error(std::format("--lens-radial {}: cannot open the file", file.string()));
        auto t = std::make_shared<RadialLensTable>();
        t->path = file;
        bool magic = false, have_cam = false, have_nodes = false, have_valid = false, have_extrap = false;
        std::vector<double> segd;
        int line_no = 0, n_seg = 0;
        std::string line;
        while (std::getline(in, line)) {
            ++line_no;
            std::istringstream ss(line);
            std::vector<std::string> tok;
            for (std::string s; ss >> s;)
                tok.push_back(s);
            if (tok.empty() || tok[0][0] == '#')
                continue;
            const auto need = [&](size_t count) {
                if (tok.size() != count)
                    throw std::runtime_error(std::format("--lens-radial {}: line {}: '{}' needs {} values, has {}", file.string(),
                                                         line_no, tok[0], count - 1, tok.size() - 1));
            };
            if (tok[0] == "LENS_RADIAL_TABLE") {
                need(2);
                if (tok[1] != "1")
                    throw std::runtime_error(std::format("--lens-radial {}: format version {} (this build reads 1)", file.string(), tok[1]));
                magic = true;
            } else if (tok[0] == "camera") {
                need(11);
                t->width = static_cast<int>(num(tok[1], file, line_no));
                t->height = static_cast<int>(num(tok[2], file, line_no));
                t->fx = num(tok[3], file, line_no);
                t->fy = num(tok[4], file, line_no);
                t->cx = num(tok[5], file, line_no);
                t->cy = num(tok[6], file, line_no);
                for (int i = 0; i < 4; ++i)
                    t->k[i] = num(tok[7 + i], file, line_no);
                have_cam = true;
            } else if (tok[0] == "nodes") {
                need(4);
                t->n = static_cast<int>(num(tok[1], file, line_no));
                t->dth = num(tok[2], file, line_no);
                t->th_end = num(tok[3], file, line_no);
                have_nodes = true;
            } else if (tok[0] == "valid") {
                need(2);
                t->valid = num(tok[1], file, line_no);
                have_valid = true;
            } else if (tok[0] == "extrap") {
                need(3);
                t->A = num(tok[1], file, line_no);
                t->B = num(tok[2], file, line_no);
                have_extrap = true;
            } else if (tok[0] == "centre") {
                need(2);
                t->centre = num(tok[1], file, line_no);
            } else if (tok[0] == "seg") {
                need(10);
                if (static_cast<int>(num(tok[1], file, line_no)) != n_seg)
                    throw std::runtime_error(std::format("--lens-radial {}: line {}: segment {} out of order", file.string(), line_no, tok[1]));
                for (int i = 2; i < 10; ++i)
                    segd.push_back(num(tok[i], file, line_no));
                ++n_seg;
            } else if (tok[0] == "node" || tok[0] == "source") {
                // reference values / provenance: not used by the lens
            } else {
                throw std::runtime_error(std::format("--lens-radial {}: line {}: unknown key '{}'", file.string(), line_no, tok[0]));
            }
        }
        if (!magic || !have_cam || !have_nodes || !have_valid || !have_extrap)
            throw std::runtime_error(std::format("--lens-radial {}: not a complete 'LENS_RADIAL_TABLE 1' file (header lines missing)", file.string()));
        if (t->n < 2 || n_seg != t->n - 1)
            throw std::runtime_error(std::format("--lens-radial {}: {} segments for {} nodes", file.string(), n_seg, t->n));
        if (!(t->dth > 0) || std::abs((t->n - 1) * t->dth - t->th_end) > 1e-9 * t->th_end || !(t->th_end < 0.5 * std::numbers::pi))
            throw std::runtime_error(std::format("--lens-radial {}: nodes {} x dtheta {} != theta_end {} (or theta_end >= 90 deg)",
                                                 file.string(), t->n, t->dth, t->th_end));
        if (t->width <= 0 || t->height <= 0 || !(t->fx > 0) || !(t->fy > 0))
            throw std::runtime_error(std::format("--lens-radial {}: bad camera line", file.string()));
        t->segd = segd;
        t->seg.resize(segd.size());
        for (size_t i = 0; i < segd.size(); ++i)
            t->seg[i] = static_cast<float>(segd[i]);
        t->inv_dth_f = static_cast<float>(1.0 / t->dth);
        t->th_end_f = static_cast<float>(t->th_end);
        t->A_f = static_cast<float>(t->A);
        t->B_f = static_cast<float>(t->B);
        // the profile must start at the axis with theta_d / theta near 1 and increase over the table and beyond
        const auto hv = t->host_view();
        float P0, P1, P2;
        core::radial_lens_P(hv, 0.0f, P0, P1, P2);
        if (!(P0 > 0.5f && P0 < 2.0f))
            throw std::runtime_error(std::format("--lens-radial {}: theta_d / theta at the axis is {} (expected ~1)", file.string(), P0));
        for (int i = 0; i <= 4 * (t->n - 1); ++i) {
            const float th = static_cast<float>(i * 0.25 * t->dth);
            const float s = core::radial_lens_slope(hv, th);
            if (!(s > 0.0f) || !std::isfinite(s))
                throw std::runtime_error(std::format("--lens-radial {}: theta_d is not increasing at {:.4f} deg (slope {})",
                                                     file.string(), th * 180.0 / std::numbers::pi, s));
        }
        if (!(t->B > 0))
            throw std::runtime_error(std::format("--lens-radial {}: the extrapolation slope {} is not positive", file.string(), t->B));
        return t;
    }

    void activate_radial_lens(std::shared_ptr<RadialLensTable> table) {
        if (table && table->d_seg == nullptr) {
            const size_t bytes = sizeof(float) * table->seg.size();
            float* d = nullptr;
            if (cudaMalloc(&d, bytes) != cudaSuccess || cudaMemcpy(d, table->seg.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                if (d)
                    cudaFree(d);
                throw std::runtime_error(std::format("--lens-radial {}: could not upload the table to the GPU", table->path.string()));
            }
            table->d_seg = d; // lives as long as the process (a few tens of KB)
        }
        g_active = std::move(table);
    }

    const RadialLensTable* active_radial_lens() noexcept { return g_active.get(); }

    std::string radial_lens_camera_mismatch(const RadialLensTable& t, const core::Camera& camera) {
        if (camera.camera_model_type() != core::CameraModelType::FISHEYE)
            return "not a fisheye camera";
        if (camera.is_undistort_prepared())
            return "the camera is undistort-prepared (pinhole)";
        if (camera.tangential_distortion().is_valid() && camera.tangential_distortion().numel() != 0)
            return "the camera has tangential coefficients";
        float k[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (const auto radial = camera.radial_distortion(); radial.is_valid() && radial.numel() > 0) {
            const auto host = (radial.device() == core::Device::CPU ? radial : radial.cpu()).contiguous();
            const auto host_f32 = host.dtype() == core::DataType::Float32 ? host : host.to(core::DataType::Float32);
            if (host_f32.numel() != 4)
                return std::format("{} radial coefficients (OPENCV_FISHEYE has 4)", host_f32.numel());
            for (int i = 0; i < 4; ++i)
                k[i] = host_f32.ptr<float>()[i];
        }
        for (int i = 0; i < 4; ++i)
            if (!close_rel(k[i], t.k[i], 1e-6))
                return std::format("k{} {:.9g} is not the table's {:.9g}", i + 1, k[i], t.k[i]);
        const double cw = camera.camera_width(), ch = camera.camera_height();
        if (!(cw > 0 && ch > 0))
            return "the camera has no size";
        const struct {
            const char* name;
            double cam, tab, rel;
        } q[] = {{"fx", camera.focal_x() / cw, t.fx / t.width, 1e-4},
                 {"fy", camera.focal_y() / ch, t.fy / t.height, 1e-4},
                 {"cx", camera.center_x() / cw, t.cx / t.width, 1e-3},
                 {"cy", camera.center_y() / ch, t.cy / t.height, 1e-3}};
        for (const auto& c : q)
            if (!close_rel(c.cam, c.tab, c.rel))
                return std::format("{} / size {:.7g} is not the table's {:.7g}", c.name, c.cam, c.tab);
        return {};
    }

} // namespace lfs::training
