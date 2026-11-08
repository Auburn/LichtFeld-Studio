/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// LFQPLCYR: headless scoring through the trainer's own renderer, so that the scorers (abq/*.py) and the trainer draw a
// model with ONE implementation (FastGS forward, native fisheye / radial lens table, dynamic rolling shutter, learned
// background cubemap, PPISP, --max-std-dev cut), instead of a third renderer (gsplat + our fused fisheye projection).
//
//   LichtFeld-Studio lfe-render --ply MODEL.ply [model options] (--serve | --views REQUESTS.jsonl | --colmap SPARSE_DIR)
//     model options (once per process; the model is loaded once):
//       --sh-degree D          render SH degree (default: the PLY's)
//       --lens-radial FILE     the radial lens table (= the trainer's --lens-radial); cameras must be the table's camera,
//                              except an equidistant camera (OPENCV_FISHEYE with k1..k4 = 0, e.g. the game lens), which
//                              is drawn through the polynomial path with the table switched off for that view
//       --max-std-dev X        the game's splat cut (= the trainer's --max-std-dev; 0 = off)
//       --max-std-dev-distance with --max-std-dev: the cut on distance (d <= X for any beta; = the trainer's flag)
//       --bg R,G,B             solid background in [0, 1] (default 0,0,0); composited by the rasterizer
//       --bg-cubemap DIR|NPY   a learned background cube (DIR/faces.npy, float32 [6, R, R, 3]); composited by the
//                              rasterizer (splats + (1 - alpha) cube) through the same lens / rolling shutter as the
//                              splats, as in training (overrides --bg)
//       --ppisp FILE           a PPISP state dump (ppisp_final.txt) for the post modes below
//     request defaults (each request may override them): --out DIR, --factor F, --readout-ms MS, --rotation-only,
//       --post none|curve|curve_vig|frame, --save image,post,alpha,depth, --dtype f32|f16, --motion FILE (--colmap only),
//       --names FILE (--colmap only: these image names)
//     --serve: one JSON request per stdin line; each answered on stdout by "@@LFS ok <id> <ms>" or "@@LFS err <id> <why>"
//       ("quit" or EOF ends). Other stdout lines are logging. --views: the same requests from a file. --colmap: one
//       request per image of a COLMAP text model (cameras.txt + images.txt), output stem = the image name without its
//       extension, motion from --motion by image name.
//     request (JSON object): id (output stem), out, camera {model OPENCV_FISHEYE|PINHOLE|SIMPLE_PINHOLE, width, height,
//       params (COLMAP order, full resolution)}, camera_id (PPISP camera; default the dump's first), factor (render at
//       width/F x height/F with the intrinsics scaled by the same rule as the trainer's image resize), pose: qvec + tvec
//       (COLMAP world-to-camera, parsed as float like the trainer's COLMAP loader) or w2c (12 or 16 numbers row-major),
//       motion [12] (wx wy wz ax ay az vx vy vz Ax Ay Az: the trainer's rolling-shutter motion row, mid-readout camera
//       axes), readout_ms, rotation_only, bg [3], cube (bool, default on when loaded), post, post_ev (EV for curve
//       modes, default 0), frame (image name of the PPISP frame for post "frame"; default id), sh_degree, sphere
//       [x, y, z, r] (draw only the splats whose centre is within r of (x, y, z)), save, dtype.
//     post modes: none; curve = the PPISP camera's response curve only (vignette off, 0 EV, identity colour:
//       abq/ppisp_post.PPISPPost(st, vignette=False)); curve_vig = + the camera's vignette (PPISPPost(st)); frame = the
//       frame's own exposure + colour + vignette + curve (the trainer's appearance stage for a training camera).
//     outputs per request, <out>/<id>_<what>.npy, little-endian float32 (or float16 with dtype f16):
//       image [H, W, 3]  the rasterizer's output (splats + (1 - alpha) background), before any post; not clamped
//       post  [H, W, 3]  after the PPISP post
//       alpha [H, W]     1 - final transmittance (splats only)
//       depth [H, W]     sum_i w_i d_i, NOT normalised by alpha; d = distance from the camera centre (fisheye / the
//                        game lens) or camera z (pinhole)
//
//   LichtFeld-Studio lfe-register DS MODEL.ply OUT [STEPS [POSE_LR]] [--max-std-dev X] [--sh-degree D]
//     Registers DS's cameras against the FROZEN model (the splats never move): the algorithm of abq/pose_opt_fish_fast.py
//     as abq/register_test.py runs it (FC_FAST=1, FC_COLOR=affine, one camera per step, dense Adam over all cameras),
//     with LichtFeld's forward and pose gradient (FastGSPose) in place of gsplat + the fused projection. The same
//     environment and the same output files (OUT/pose_deltas.npz, pose_deltas_<n>.npz, model/sparse/0, settle.tsv,
//     percam.tsv; the same log lines), so the downstream scripts are unchanged. Env: FC_CACHE (half-res frames from
//     `pose_opt_fish_fast.py --prep`), FC_FACTOR (2), FC_BG, FC_RS + FC_RS_MS, FC_LENS_RADIAL, FC_NO_DRIFT,
//     POSE_SNAP_EVERY, POSE_DECAY_AFTER, POSE_HALF_LIFE, POSE_SETTLE_TURN, POSE_SETTLE_MOVE, POSE_T_SCALE (required:
//     abq/register_test.py computes it as the Python refiner does), FC_ORDER (the draw order, one camera index per
//     line, from abq/register_test.py = numpy default_rng(0) shuffled epochs; without it a mt19937 shuffle).

#include "core/camera.hpp"
#include "core/camera_types.h"
#include "core/image_io.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/loader.hpp"
#include "lfe_register_kernels.hpp"
#include "training/bg_cubemap.hpp"
#include "training/components/ppisp.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/radial_lens_table.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <numbers>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <rasterization_api.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lfs::app {

    namespace {
        namespace fs = std::filesystem;
        namespace fr = fast_lfs::rasterization;
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using json = nlohmann::json;
        using M3 = std::array<double, 9>;
        using M4 = std::array<double, 16>;

        void cuda_ok(cudaError_t e, const char* what) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::format("{}: {}", what, cudaGetErrorString(e)));
        }

        // ------------------------------------------------------------------ small utilities
        uint16_t float_to_half(float f) {
            uint32_t x;
            std::memcpy(&x, &f, 4);
            const uint32_t sign = (x >> 16) & 0x8000u;
            const uint32_t fexp = (x >> 23) & 0xffu;
            uint32_t mant = x & 0x7fffffu;
            if (fexp == 0xffu)
                return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
            const int exp = static_cast<int>(fexp) - 127 + 15;
            if (exp >= 31)
                return static_cast<uint16_t>(sign | 0x7c00u);
            if (exp <= 0) {
                if (exp < -10)
                    return static_cast<uint16_t>(sign);
                mant |= 0x800000u;
                const int shift = 14 - exp;
                uint32_t h = mant >> shift;
                const uint32_t rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
                if (rem > half || (rem == half && (h & 1u)))
                    ++h;
                return static_cast<uint16_t>(sign | h);
            }
            uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
            const uint32_t rem = mant & 0x1fffu;
            if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
                ++h;
            return static_cast<uint16_t>(h);
        }

        std::string npy_bytes(const void* data, size_t bytes, const std::string& descr, const std::vector<size_t>& shape) {
            std::string shp = "(";
            for (size_t i = 0; i < shape.size(); ++i)
                shp += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
            shp += ")";
            std::string hdr = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shp + ", }";
            const size_t total = 10 + hdr.size() + 1;
            hdr += std::string((64 - total % 64) % 64, ' ') + "\n";
            std::string out = std::string("\x93NUMPY\x01\x00", 8);
            const uint16_t n = static_cast<uint16_t>(hdr.size());
            out.push_back(static_cast<char>(n & 0xff));
            out.push_back(static_cast<char>(n >> 8));
            out += hdr;
            out.append(static_cast<const char*>(data), bytes);
            return out;
        }

        void write_file(const fs::path& p, const std::string& s) {
            const fs::path tmp = fs::path(p).concat(".part");
            {
                std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
                if (!f)
                    throw std::runtime_error("cannot write " + p.string());
                f.write(s.data(), static_cast<std::streamsize>(s.size()));
                if (!f)
                    throw std::runtime_error("write failed: " + p.string());
            }
            std::error_code ec;
            fs::rename(tmp, p, ec);
            if (ec)
                throw std::runtime_error("rename failed: " + p.string() + ": " + ec.message());
        }

        void write_npy_float(const fs::path& p, const std::vector<float>& v, const std::vector<size_t>& shape, bool f16) {
            if (!f16) {
                write_file(p, npy_bytes(v.data(), v.size() * 4, "<f4", shape));
                return;
            }
            std::vector<uint16_t> h(v.size());
            for (size_t i = 0; i < v.size(); ++i)
                h[i] = float_to_half(v[i]);
            write_file(p, npy_bytes(h.data(), h.size() * 2, "<f2", shape));
        }

        uint32_t crc32(const std::string& s) {
            static uint32_t table[256];
            static bool init = false;
            if (!init) {
                for (uint32_t i = 0; i < 256; ++i) {
                    uint32_t c = i;
                    for (int k = 0; k < 8; ++k)
                        c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                    table[i] = c;
                }
                init = true;
            }
            uint32_t c = 0xffffffffu;
            for (const unsigned char ch : s)
                c = table[(c ^ ch) & 0xff] ^ (c >> 8);
            return c ^ 0xffffffffu;
        }

        // an uncompressed .npz (zip, stored) of named .npy members, as numpy.savez writes
        void write_npz(const fs::path& p, const std::vector<std::pair<std::string, std::string>>& members) {
            std::string zip, central;
            auto u16 = [](std::string& s, uint16_t v) { s.push_back(static_cast<char>(v & 0xff)); s.push_back(static_cast<char>(v >> 8)); };
            auto u32 = [](std::string& s, uint32_t v) { for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((v >> (8 * k)) & 0xff)); };
            for (const auto& [name0, data] : members) {
                const std::string name = name0 + ".npy";
                const uint32_t crc = crc32(data), size = static_cast<uint32_t>(data.size());
                const uint32_t offset = static_cast<uint32_t>(zip.size());
                u32(zip, 0x04034b50u); u16(zip, 20); u16(zip, 0); u16(zip, 0); u16(zip, 0); u16(zip, 0x21);
                u32(zip, crc); u32(zip, size); u32(zip, size); u16(zip, static_cast<uint16_t>(name.size())); u16(zip, 0);
                zip += name;
                zip += data;
                u32(central, 0x02014b50u); u16(central, 20); u16(central, 20); u16(central, 0); u16(central, 0);
                u16(central, 0); u16(central, 0x21); u32(central, crc); u32(central, size); u32(central, size);
                u16(central, static_cast<uint16_t>(name.size())); u16(central, 0); u16(central, 0); u16(central, 0);
                u16(central, 0); u32(central, 0); u32(central, offset);
                central += name;
            }
            const uint32_t cd_offset = static_cast<uint32_t>(zip.size()), cd_size = static_cast<uint32_t>(central.size());
            zip += central;
            u32(zip, 0x06054b50u); u16(zip, 0); u16(zip, 0); u16(zip, static_cast<uint16_t>(members.size()));
            u16(zip, static_cast<uint16_t>(members.size())); u32(zip, cd_size); u32(zip, cd_offset); u16(zip, 0);
            write_file(p, zip);
        }

        std::string npy_unicode(const std::vector<std::string>& names) {
            size_t L = 1;
            for (const auto& n : names)
                L = std::max(L, n.size());
            std::vector<uint32_t> buf(names.size() * L, 0u);
            for (size_t i = 0; i < names.size(); ++i)
                for (size_t k = 0; k < names[i].size(); ++k)
                    buf[i * L + k] = static_cast<unsigned char>(names[i][k]);
            return npy_bytes(buf.data(), buf.size() * 4, "<U" + std::to_string(L), {names.size()});
        }

        std::string npy_f64(const std::vector<double>& v, const std::vector<size_t>& shape) {
            return npy_bytes(v.data(), v.size() * 8, "<f8", shape);
        }

        std::vector<std::string> split_ws(const std::string& line) {
            std::istringstream ss(line);
            std::vector<std::string> t;
            for (std::string s; ss >> s;)
                t.push_back(s);
            return t;
        }

        std::vector<float> parse_floats(const std::string& s, char sep = ',') {
            std::vector<float> v;
            std::stringstream ss(s);
            for (std::string item; std::getline(ss, item, sep);)
                if (!item.empty())
                    v.push_back(std::stof(item));
            return v;
        }

        // ------------------------------------------------------------------ rotations (double)
        M3 quat_to_rot_d(double w, double x, double y, double z) { // abq/pose_opt_fish.quat_to_rot (normalised)
            const double n = std::sqrt(w * w + x * x + y * y + z * z);
            w /= n; x /= n; y /= n; z /= n;
            return {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
        }

        // the trainer's COLMAP loader (io/formats/colmap.cpp qvec2rotmat_values): float, normalised in float
        std::array<float, 9> qvec2rotmat_f(const std::array<float, 4>& q) {
            const float len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            const float w = q[0] / len, x = q[1] / len, y = q[2] / len, z = q[3] / len;
            return {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w),
                    2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w),
                    2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y)};
        }

        M4 mul4(const M4& a, const M4& b) {
            M4 c{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    double s = 0;
                    for (int k = 0; k < 4; ++k)
                        s += a[i * 4 + k] * b[k * 4 + j];
                    c[i * 4 + j] = s;
                }
            return c;
        }

        M4 rigid_inverse(const M4& m) {
            M4 r{};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    r[i * 4 + j] = m[j * 4 + i];
                r[i * 4 + 3] = -(m[0 * 4 + i] * m[3] + m[1 * 4 + i] * m[7] + m[2 * 4 + i] * m[11]);
            }
            r[15] = 1.0;
            return r;
        }

        M3 rot_of(const M4& m) { return {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]}; }
        std::array<double, 3> pos_of(const M4& m) { return {m[3], m[7], m[11]}; }
        M3 mul3(const M3& a, const M3& b) {
            M3 c{};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
            return c;
        }
        M3 transpose3(const M3& a) { return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]}; }
        std::array<double, 3> mv3(const M3& a, const std::array<double, 3>& v) {
            return {a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
                    a[6] * v[0] + a[7] * v[1] + a[8] * v[2]};
        }
        double det3(const M3& a) {
            return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
        }

        // scipy Rotation.from_matrix(...).as_quat() -> (w, x, y, z), sign as scipy leaves it
        std::array<double, 4> rot_to_quat_wxyz(const M3& m) {
            const double dec[4] = {m[0], m[4], m[8], m[0] + m[4] + m[8]};
            int choice = 0;
            for (int k = 1; k < 4; ++k)
                if (dec[k] > dec[choice])
                    choice = k;
            double q[4]; // x y z w
            if (choice != 3) {
                const int i = choice, j = (i + 1) % 3, k = (j + 1) % 3;
                q[i] = 1 - dec[3] + 2 * m[i * 3 + i];
                q[j] = m[j * 3 + i] + m[i * 3 + j];
                q[k] = m[k * 3 + i] + m[i * 3 + k];
                q[3] = m[k * 3 + j] - m[j * 3 + k];
            } else {
                q[0] = m[7] - m[5];
                q[1] = m[2] - m[6];
                q[2] = m[3] - m[1];
                q[3] = 1 + dec[3];
            }
            const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            return {q[3] / n, q[0] / n, q[1] / n, q[2] / n};
        }

        std::array<double, 3> rotvec_of(const M3& m) { // scipy as_rotvec
            auto q = rot_to_quat_wxyz(m);
            if (q[0] < 0)
                for (auto& x : q)
                    x = -x;
            const double vn = std::sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            const double angle = 2 * std::atan2(vn, q[0]);
            const double scale = angle <= 1e-3 ? 2 + angle * angle / 12 + 7 * angle * angle * angle * angle / 2880
                                               : angle / std::sin(angle / 2);
            return {scale * q[1], scale * q[2], scale * q[3]};
        }

        double rot_angle_deg(const M3& a, const M3& b) { // abq/pose_opt_fish_fast.rot_angle
            double d = 0;
            for (int k = 0; k < 9; ++k)
                d += (b[k] - a[k]) * (b[k] - a[k]);
            d = std::sqrt(d) / (2 * std::sqrt(2.0));
            return 2 * std::asin(std::clamp(d, 0.0, 1.0)) * 180.0 / std::numbers::pi;
        }

        double median(std::vector<double> v) {
            if (v.empty())
                return 0;
            std::sort(v.begin(), v.end());
            const size_t n = v.size();
            return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        }
        double percentile(std::vector<double> v, double p) { // numpy linear
            if (v.empty())
                return 0;
            std::sort(v.begin(), v.end());
            const double pos = p / 100.0 * static_cast<double>(v.size() - 1);
            const size_t lo = static_cast<size_t>(std::floor(pos));
            const size_t hi = std::min(lo + 1, v.size() - 1);
            return v[lo] + (pos - static_cast<double>(lo)) * (v[hi] - v[lo]);
        }

        // symmetric 3x3 eigen decomposition (Jacobi), eigenvalues descending, eigenvectors in columns of V
        void eig_sym3(M3 a, std::array<double, 3>& w, M3& V) {
            V = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            for (int sweep = 0; sweep < 60; ++sweep) {
                const double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
                if (off < 1e-300)
                    break;
                for (int p = 0; p < 2; ++p)
                    for (int q = p + 1; q < 3; ++q) {
                        const double apq = a[p * 3 + q];
                        if (std::abs(apq) < 1e-300)
                            continue;
                        const double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2 * apq);
                        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                        for (int k = 0; k < 3; ++k) { // A = J^T A J
                            const double akp = a[k * 3 + p], akq = a[k * 3 + q];
                            a[k * 3 + p] = c * akp - s * akq;
                            a[k * 3 + q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 3; ++k) {
                            const double apk = a[p * 3 + k], aqk = a[q * 3 + k];
                            a[p * 3 + k] = c * apk - s * aqk;
                            a[q * 3 + k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 3; ++k) {
                            const double vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                            V[k * 3 + p] = c * vkp - s * vkq;
                            V[k * 3 + q] = s * vkp + c * vkq;
                        }
                    }
            }
            std::array<int, 3> idx{0, 1, 2};
            std::sort(idx.begin(), idx.end(), [&](int x, int y) { return a[x * 4] > a[y * 4]; });
            M3 Vs{};
            for (int c = 0; c < 3; ++c) {
                w[c] = a[idx[c] * 4];
                for (int r = 0; r < 3; ++r)
                    Vs[r * 3 + c] = V[r * 3 + idx[c]];
            }
            V = Vs;
        }

        // M = U diag(S) V^T (numpy convention: S >= 0, descending)
        void svd3(const M3& M, M3& U, std::array<double, 3>& S, M3& V) {
            std::array<double, 3> w{};
            eig_sym3(mul3(transpose3(M), M), w, V);
            for (int c = 0; c < 3; ++c)
                S[c] = std::sqrt(std::max(w[c], 0.0));
            for (int c = 0; c < 3; ++c) {
                const std::array<double, 3> v{V[c], V[3 + c], V[6 + c]};
                auto u = mv3(M, v);
                if (S[c] > 1e-12 * std::max(S[0], 1e-300)) {
                    for (auto& x : u)
                        x /= S[c];
                } else if (c == 2) {
                    u = {U[3] * U[7] - U[6] * U[4], U[6] * U[1] - U[0] * U[7], U[0] * U[4] - U[3] * U[1]};
                }
                for (int r = 0; r < 3; ++r)
                    U[r * 3 + c] = u[r];
            }
        }

        struct DriftFix {
            std::vector<M4> fix;
            double sg = 1.0, rg_deg = 0.0;
        };

        DriftFix remove_drift(const std::vector<M4>& old_, const std::vector<M4>& new_) { // pose_opt_fish_fast.remove_drift
            const size_t n = new_.size();
            std::array<double, 3> ma{}, mb{};
            for (size_t i = 0; i < n; ++i)
                for (int k = 0; k < 3; ++k) {
                    ma[k] += new_[i][k * 4 + 3] / static_cast<double>(n);
                    mb[k] += old_[i][k * 4 + 3] / static_cast<double>(n);
                }
            M3 C{};
            double ssa = 0;
            for (size_t i = 0; i < n; ++i) {
                std::array<double, 3> a{}, b{};
                for (int k = 0; k < 3; ++k) {
                    a[k] = new_[i][k * 4 + 3] - ma[k];
                    b[k] = old_[i][k * 4 + 3] - mb[k];
                    ssa += a[k] * a[k];
                }
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        C[r * 3 + c] += b[r] * a[c];
            }
            M3 U{}, V{};
            std::array<double, 3> S{};
            svd3(C, U, S, V);
            const M3 Vt = transpose3(V);
            const double sgn = det3(mul3(U, Vt)) < 0 ? -1.0 : 1.0;
            const M3 D{1, 0, 0, 0, 1, 0, 0, 0, sgn};
            const M3 Rg = mul3(mul3(U, D), Vt);
            DriftFix out;
            out.sg = (S[0] + S[1] + S[2]) / ssa;
            const auto rma = mv3(Rg, ma);
            const std::array<double, 3> tg{mb[0] - out.sg * rma[0], mb[1] - out.sg * rma[1], mb[2] - out.sg * rma[2]};
            out.rg_deg = rot_angle_deg({1, 0, 0, 0, 1, 0, 0, 0, 1}, Rg);
            out.fix.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const M3 R = mul3(Rg, rot_of(new_[i]));
                const auto c = mv3(Rg, pos_of(new_[i]));
                M4 f{};
                for (int r = 0; r < 3; ++r) {
                    for (int k = 0; k < 3; ++k)
                        f[r * 4 + k] = R[r * 3 + k];
                    f[r * 4 + 3] = out.sg * c[r] + tg[r];
                }
                f[15] = 1;
                out.fix[i] = f;
            }
            return out;
        }

        // the 6D rotation of abq/pose_opt_fish (rot6d_to_matrix): a1 = d6[0:3] + e_x, a2 = d6[3:6] + e_y, Gram-Schmidt,
        // R = rows [b1; b2; b3]; with forward-mode derivatives dR/dd6_k
        struct Dual {
            double v = 0, d = 0;
        };
        inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
        inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
        inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
        inline Dual operator/(Dual a, Dual b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
        inline Dual dsqrt(Dual a) {
            const double s = std::sqrt(a.v);
            return {s, s > 0 ? a.d / (2 * s) : 0.0};
        }

        void rot6d(const float* d6, int dir, Dual R[9]) {
            Dual a1[3], a2[3];
            for (int k = 0; k < 3; ++k) {
                a1[k] = {static_cast<double>(d6[k]) + (k == 0 ? 1.0 : 0.0), dir == k ? 1.0 : 0.0};
                a2[k] = {static_cast<double>(d6[3 + k]) + (k == 1 ? 1.0 : 0.0), dir == 3 + k ? 1.0 : 0.0};
            }
            auto normalize = [](Dual v[3]) {
                Dual n = dsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                if (n.v < 1e-12)
                    n = {1e-12, 0.0};
                for (int k = 0; k < 3; ++k)
                    v[k] = v[k] / n;
            };
            Dual b1[3] = {a1[0], a1[1], a1[2]};
            normalize(b1);
            const Dual dp = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2];
            Dual b2[3];
            for (int k = 0; k < 3; ++k)
                b2[k] = a2[k] - dp * b1[k];
            normalize(b2);
            const Dual b3[3] = {b1[1] * b2[2] - b1[2] * b2[1], b1[2] * b2[0] - b1[0] * b2[2], b1[0] * b2[1] - b1[1] * b2[0]};
            for (int k = 0; k < 3; ++k) {
                R[k] = b1[k];
                R[3 + k] = b2[k];
                R[6 + k] = b3[k];
            }
        }

        M4 delta_matrix(const float* e, double t_scale) {
            Dual R[9];
            rot6d(e + 3, -1, R);
            M4 D{};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c)
                    D[r * 4 + c] = R[r * 3 + c].v;
                D[r * 4 + 3] = static_cast<double>(e[r]) * t_scale;
            }
            D[15] = 1;
            return D;
        }

        // ------------------------------------------------------------------ model / renderer state
        struct LoadedModel {
            std::shared_ptr<lfs::core::SplatData> splat;
            std::shared_ptr<lfs::training::RadialLensTable> table;
            int max_sh = 0;
            std::vector<float> means_host;        // lazily, for sphere requests
            std::vector<float> opacity_host;      // raw
            Tensor opacity_backup;                // device clone of the raw opacity
        };

        // --max-std-dev-distance (LFQPLCYKD): the trainer's distance cut (d <= C for any beta); set by both mains' flag
        bool g_cut_distance = false;

        void load_model(LoadedModel& m, const fs::path& ply, int sh_degree, const std::string& lens_radial, float max_std_dev) {
            if (!lens_radial.empty()) {
                m.table = lfs::training::load_radial_lens_table(lens_radial);
                lfs::training::activate_radial_lens(m.table);
                std::println("radial lens table: {}", lfs::training::active_radial_lens()->describe());
            }
            if (max_std_dev > 0.0f) {
                if (g_cut_distance) {
                    fr::set_splat_cutoff(max_std_dev, true);
                    std::println("splat cut (--max-std-dev {} --max-std-dev-distance): distance cap ON, d <= {} for any beta, bounds min(contour, C)",
                                 max_std_dev, max_std_dev);
                } else {
                    fr::set_splat_cutoff(max_std_dev);
                    std::println("splat cut (--max-std-dev): {} sigma", max_std_dev);
                }
            }
            auto loader = lfs::io::Loader::create();
            auto r = loader->load(ply);
            if (!r)
                throw std::runtime_error("load failed: " + std::string(r.error().message));
            auto* sp = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&r->data);
            if (!sp || !*sp || (*sp)->size() == 0)
                throw std::runtime_error("not a splat scene: " + ply.string());
            m.splat = *sp;
            if (m.splat->shN_value_quantized() || m.splat->shN_ieee_f16())
                throw std::runtime_error("quantized / f16 SH storage is not supported by lfe-render (load as float32)");
            m.max_sh = m.splat->get_max_sh_degree();
            if (sh_degree >= 0)
                m.splat->set_active_sh_degree(sh_degree);
            std::println("model {}: {} splats, SH degree {} (max {})", ply.string(), m.splat->size(),
                         m.splat->get_active_sh_degree(), m.max_sh);
            if (m.splat->_shape.is_valid()) { // LFQPLCYK: the PLY's `shape` property; every view / step draws each splat's beta
                if (max_std_dev > 0.0f && !g_cut_distance)
                    throw std::runtime_error("--max-std-dev cannot be used with a per-splat kernel shape (the PLY's `shape` "
                                             "property): the cut is in standard deviations of a Gaussian");
                std::println("per-splat kernel shape (PLY property shape): {} values", m.splat->_shape.numel());
            }
        }

        struct CameraSpec {
            std::string model = "OPENCV_FISHEYE";
            int width = 0, height = 0;
            std::vector<double> params;
        };

        CameraSpec camera_from_json(const json& j) {
            CameraSpec c;
            c.model = j.value("model", std::string("OPENCV_FISHEYE"));
            c.width = j.at("width").get<int>();
            c.height = j.at("height").get<int>();
            c.params = j.at("params").get<std::vector<double>>();
            return c;
        }

        // builds the camera at full resolution, then the trainer's image resize rule (Camera::get_intrinsics scales by
        // image / camera size) for the factor
        std::unique_ptr<Camera> make_camera(const CameraSpec& cs, const std::array<float, 9>& R, const std::array<float, 3>& T,
                                            int factor, const std::string& name, int uid, int camera_id) {
            float fx, fy, cx, cy;
            std::vector<float> k;
            CameraModelType type;
            const auto& p = cs.params;
            if (cs.model == "OPENCV_FISHEYE") {
                if (p.size() != 8)
                    throw std::runtime_error("OPENCV_FISHEYE needs 8 params");
                fx = static_cast<float>(p[0]); fy = static_cast<float>(p[1]); cx = static_cast<float>(p[2]); cy = static_cast<float>(p[3]);
                k = {static_cast<float>(p[4]), static_cast<float>(p[5]), static_cast<float>(p[6]), static_cast<float>(p[7])};
                type = CameraModelType::FISHEYE;
            } else if (cs.model == "PINHOLE") {
                if (p.size() != 4)
                    throw std::runtime_error("PINHOLE needs 4 params");
                fx = static_cast<float>(p[0]); fy = static_cast<float>(p[1]); cx = static_cast<float>(p[2]); cy = static_cast<float>(p[3]);
                type = CameraModelType::PINHOLE;
            } else if (cs.model == "SIMPLE_PINHOLE") {
                if (p.size() != 3)
                    throw std::runtime_error("SIMPLE_PINHOLE needs 3 params");
                fx = fy = static_cast<float>(p[0]); cx = static_cast<float>(p[1]); cy = static_cast<float>(p[2]);
                type = CameraModelType::PINHOLE;
            } else {
                throw std::runtime_error("unsupported camera model " + cs.model);
            }
            auto Rt = Tensor::from_vector(std::vector<float>(R.begin(), R.end()), {3, 3}, Device::CPU);
            auto Tt = Tensor::from_vector(std::vector<float>(T.begin(), T.end()), {3}, Device::CPU);
            auto radial = k.empty() ? Tensor::empty({0}, Device::CPU) : Tensor::from_vector(k, {4}, Device::CPU);
            auto cam = std::make_unique<Camera>(Rt, Tt, fx, fy, cx, cy, radial, Tensor::empty({0}, Device::CPU), type, name,
                                                fs::path{}, fs::path{}, cs.width, cs.height, uid, camera_id);
            if (factor > 1)
                cam->set_image_dimensions(cs.width / factor, cs.height / factor);
            return cam;
        }

        // ------------------------------------------------------------------ PPISP from a ppisp_final.txt dump
        struct PPISPState {
            std::unique_ptr<lfs::training::PPISP> ppisp;
            int first_camera_id = 1;
            std::unordered_map<std::string, int> uid_of_name;
            std::map<int, int> camera_of_uid;
            bool lock_colour = false;
        };

        void set_tensor(const Tensor& t0, const std::vector<float>& v, const char* what) {
            Tensor& t = const_cast<Tensor&>(t0);
            if (t.numel() != v.size())
                throw std::runtime_error(std::format("PPISP dump: {} has {} values, the PPISP tensor {}", what, v.size(), t.numel()));
            if (t.device() == Device::CUDA)
                cuda_ok(cudaMemcpy(t.ptr<float>(), v.data(), v.size() * 4, cudaMemcpyHostToDevice), "PPISP upload");
            else
                std::memcpy(t.ptr<float>(), v.data(), v.size() * 4);
        }

        void load_ppisp(PPISPState& st, const fs::path& path) {
            std::ifstream in(path);
            if (!in)
                throw std::runtime_error("cannot open PPISP dump " + path.string());
            struct Cam { int id, idx; std::vector<float> vig, crf; };
            struct Frm { int idx, uid, cam; std::string name; float ev; std::vector<float> col; };
            std::vector<Cam> cams;
            std::vector<Frm> frames;
            for (std::string line; std::getline(in, line);) {
                if (line.empty() || line[0] == '#')
                    continue;
                const auto t = split_ws(line);
                if (t.empty())
                    continue;
                if (t[0] == "lock_colour" && t.size() > 1)
                    st.lock_colour = t[1] == "1";
                if (t[0] == "camera") {
                    if (t.size() != 1 + 2 + 1 + 15 + 1 + 12 || t[3] != "vig" || t[19] != "crf")
                        throw std::runtime_error("PPISP dump: bad camera line");
                    Cam c{std::stoi(t[1]), std::stoi(t[2]), {}, {}};
                    for (int k = 0; k < 15; ++k)
                        c.vig.push_back(std::stof(t[4 + k]));
                    for (int k = 0; k < 12; ++k)
                        c.crf.push_back(std::stof(t[20 + k]));
                    cams.push_back(std::move(c));
                } else if (t[0] == "frame") {
                    if (t.size() != 6 + 8)
                        throw std::runtime_error("PPISP dump: bad frame line");
                    Frm f{std::stoi(t[1]), std::stoi(t[2]), std::stoi(t[3]), t[4], std::stof(t[5]), {}};
                    for (int k = 0; k < 8; ++k)
                        f.col.push_back(std::stof(t[6 + k]));
                    frames.push_back(std::move(f));
                }
            }
            if (cams.empty() || frames.empty())
                throw std::runtime_error("PPISP dump without cameras / frames: " + path.string());
            std::sort(cams.begin(), cams.end(), [](const Cam& a, const Cam& b) { return a.idx < b.idx; });
            std::sort(frames.begin(), frames.end(), [](const Frm& a, const Frm& b) { return a.idx < b.idx; });
            // camera indices are assigned in order of first appearance at registration: check they agree with the dump
            std::map<int, int> first_seen;
            for (const auto& f : frames)
                if (!first_seen.contains(f.cam))
                    first_seen.emplace(f.cam, static_cast<int>(first_seen.size()));
            for (const auto& c : cams)
                if (!first_seen.contains(c.id) || first_seen[c.id] != c.idx)
                    throw std::runtime_error("PPISP dump: camera index order differs from the frames' first appearance");
            st.ppisp = std::make_unique<lfs::training::PPISP>(1);
            for (const auto& f : frames) {
                st.ppisp->register_frame(f.uid, f.cam);
                st.uid_of_name[f.name] = f.uid;
                st.camera_of_uid[f.uid] = f.cam;
            }
            st.ppisp->finalize();
            std::vector<float> ev, col, vig, crf;
            for (const auto& f : frames) {
                ev.push_back(f.ev);
                col.insert(col.end(), f.col.begin(), f.col.end());
            }
            for (const auto& c : cams) {
                vig.insert(vig.end(), c.vig.begin(), c.vig.end());
                crf.insert(crf.end(), c.crf.begin(), c.crf.end());
            }
            set_tensor(st.ppisp->exposure_params(), ev, "exposure");
            set_tensor(st.ppisp->color_params(), col, "colour");
            set_tensor(st.ppisp->vignetting_params(), vig, "vignetting");
            set_tensor(st.ppisp->crf_params(), crf, "crf");
            cuda_ok(cudaDeviceSynchronize(), "PPISP load");
            st.first_camera_id = cams[0].id;
            std::println("PPISP state {}: {} cameras, {} frames (lock_colour {})", path.string(), cams.size(), frames.size(),
                         st.lock_colour ? 1 : 0);
        }

        // ------------------------------------------------------------------ the render server
        struct Defaults {
            fs::path out = ".";
            int factor = 1;
            float readout_ms = 10.0f;
            bool rotation_only = false;
            std::string post = "none";
            std::vector<std::string> save{"image", "alpha"};
            bool f16 = false;
        };

        struct Renderer {
            LoadedModel model;
            std::array<float, 3> bg{0, 0, 0};
            std::unique_ptr<lfs::training::BackgroundCubemap> cube;
            PPISPState ppisp;
            bool have_ppisp = false;
            Defaults def;
            float* d_motion = nullptr;
            float* d_readout = nullptr;
            int uid_counter = 100000;
            bool warned_equidistant = false;

            ~Renderer() {
                cudaFree(d_motion);
                cudaFree(d_readout);
            }

            void apply_sphere(const std::vector<double>& s) {
                auto& sp = *model.splat;
                const size_t n = sp.size();
                if (model.means_host.empty()) {
                    model.means_host.resize(n * 3);
                    model.opacity_host.resize(n);
                    cuda_ok(cudaDeviceSynchronize(), "sync");
                    cuda_ok(cudaMemcpy(model.means_host.data(), sp.means().ptr<float>(), n * 12, cudaMemcpyDeviceToHost), "means");
                    cuda_ok(cudaMemcpy(model.opacity_host.data(), sp.opacity_raw().ptr<float>(), n * 4, cudaMemcpyDeviceToHost), "opacity");
                    model.opacity_backup = sp.opacity_raw().clone();
                }
                std::vector<float> op(model.opacity_host);
                const double r2 = s[3] * s[3];
                size_t kept = 0;
                for (size_t i = 0; i < n; ++i) {
                    const double dx = model.means_host[3 * i] - s[0], dy = model.means_host[3 * i + 1] - s[1],
                                 dz = model.means_host[3 * i + 2] - s[2];
                    if (!(dx * dx + dy * dy + dz * dz < r2))
                        op[i] = -1e4f; // sigmoid -> 0: culled (opacity < 1/255)
                    else
                        ++kept;
                }
                cuda_ok(cudaMemcpy(sp.opacity_raw().ptr<float>(), op.data(), n * 4, cudaMemcpyHostToDevice), "opacity");
                std::println("sphere ({:.4f}, {:.4f}, {:.4f}) r {:.4f}: {} of {} splats", s[0], s[1], s[2], s[3], kept, n);
            }

            void restore_opacity() {
                auto& sp = *model.splat;
                cuda_ok(cudaMemcpy(sp.opacity_raw().ptr<float>(), model.opacity_backup.ptr<float>(), sp.size() * 4,
                                   cudaMemcpyDeviceToDevice),
                        "opacity restore");
            }

            std::vector<float> download(const Tensor& t) {
                std::vector<float> v(t.numel());
                cuda_ok(cudaMemcpy(v.data(), t.ptr<float>(), v.size() * 4, cudaMemcpyDeviceToHost), "download");
                return v;
            }

            static std::vector<float> chw_to_hwc(const std::vector<float>& v, int H, int W) {
                std::vector<float> o(v.size());
                const size_t HW = static_cast<size_t>(H) * W;
                for (size_t i = 0; i < HW; ++i)
                    for (int c = 0; c < 3; ++c)
                        o[i * 3 + c] = v[c * HW + i];
                return o;
            }

            void render(const json& rq) {
                const std::string id = rq.at("id").get<std::string>();
                const fs::path out = rq.contains("out") ? fs::path(rq["out"].get<std::string>()) : def.out;
                const int factor = rq.value("factor", def.factor);
                const CameraSpec cs = camera_from_json(rq.at("camera"));
                std::array<float, 9> R{};
                std::array<float, 3> T{};
                if (rq.contains("qvec")) {
                    std::array<float, 4> q{};
                    for (int k = 0; k < 4; ++k) // as the trainer's COLMAP text loader: parsed as float
                        q[k] = std::stof(rq["qvec"][k].is_string() ? rq["qvec"][k].get<std::string>() : rq["qvec"][k].dump());
                    R = qvec2rotmat_f(q);
                    for (int k = 0; k < 3; ++k)
                        T[k] = std::stof(rq["tvec"][k].is_string() ? rq["tvec"][k].get<std::string>() : rq["tvec"][k].dump());
                } else {
                    const auto w = rq.at("w2c").get<std::vector<double>>();
                    if (w.size() != 12 && w.size() != 16)
                        throw std::runtime_error("w2c needs 12 or 16 numbers");
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c)
                            R[r * 3 + c] = static_cast<float>(w[r * 4 + c]);
                        T[r] = static_cast<float>(w[r * 4 + 3]);
                    }
                }
                const int camera_id = rq.value("camera_id", have_ppisp ? ppisp.first_camera_id : 1);
                const std::string frame = rq.value("frame", id);
                int uid = uid_counter++;
                if (have_ppisp && ppisp.uid_of_name.contains(frame))
                    uid = ppisp.uid_of_name[frame];
                auto cam = make_camera(cs, R, T, factor, frame, uid, camera_id);
                const int W = cam->image_width(), H = cam->image_height();

                // the lens table: the table's camera through the table; an equidistant camera (k = 0, the game lens)
                // through the polynomial path with the table off for this view; anything else is refused (the trainer's rule)
                bool table_off = false;
                if (model.table && cam->camera_model_type() == CameraModelType::FISHEYE) {
                    const std::string why = lfs::training::radial_lens_camera_mismatch(*model.table, *cam);
                    if (!why.empty()) {
                        const bool equidistant = cs.params.size() == 8 && cs.params[4] == 0 && cs.params[5] == 0 &&
                                                 cs.params[6] == 0 && cs.params[7] == 0;
                        if (!equidistant)
                            throw std::runtime_error("--lens-radial: camera of '" + id + "' is not the table's camera: " + why);
                        table_off = true;
                        if (!warned_equidistant) {
                            std::println("equidistant camera (k = 0, the game lens): drawn with the radial table off");
                            warned_equidistant = true;
                        }
                    }
                }
                if (table_off)
                    lfs::training::activate_radial_lens(nullptr);
                struct TableGuard {
                    Renderer* r; bool on;
                    ~TableGuard() { if (on) lfs::training::activate_radial_lens(r->model.table); }
                } table_guard{this, table_off};

                // rolling shutter
                fr::FastGSRollingShutter rs{};
                if (rq.contains("motion") && !rq["motion"].is_null()) {
                    auto mo = rq["motion"].get<std::vector<float>>();
                    if (mo.size() != 12)
                        throw std::runtime_error("motion needs 12 numbers");
                    if (rq.value("rotation_only", def.rotation_only))
                        for (int k = 6; k < 12; ++k)
                            mo[k] = 0.0f;
                    // seconds exactly as the trainer forms them (rolling_shutter_motion.cpp: readout_ms * 1e-3f)
                    const float readout_s = rq.value("readout_ms", def.readout_ms) * 1e-3f;
                    if (!d_motion) {
                        cuda_ok(cudaMalloc(&d_motion, 12 * 4), "alloc");
                        cuda_ok(cudaMalloc(&d_readout, 4), "alloc");
                    }
                    cuda_ok(cudaMemcpy(d_motion, mo.data(), 48, cudaMemcpyHostToDevice), "motion");
                    cuda_ok(cudaMemcpy(d_readout, &readout_s, 4, cudaMemcpyHostToDevice), "readout");
                    rs.motion = d_motion;
                    rs.readout = d_readout;
                    rs.iters = 2;
                }

                // splat subset, SH degree
                const bool sphere = rq.contains("sphere") && !rq["sphere"].is_null();
                if (sphere)
                    apply_sphere(rq["sphere"].get<std::vector<double>>());
                const int sh_prev = model.splat->get_active_sh_degree();
                if (rq.contains("sh_degree"))
                    model.splat->set_active_sh_degree(rq["sh_degree"].get<int>());
                struct Restore {
                    Renderer* r; bool sphere; int sh;
                    ~Restore() {
                        r->model.splat->set_active_sh_degree(sh);
                        if (sphere) r->restore_opacity();
                    }
                } restore{this, sphere, sh_prev};

                // background
                std::array<float, 3> bgv = bg;
                if (rq.contains("bg"))
                    bgv = rq["bg"].get<std::array<float, 3>>();
                Tensor bg_t = Tensor::from_vector(std::vector<float>(bgv.begin(), bgv.end()), {3}, Device::CPU).to(Device::CUDA);
                Tensor bg_image;
                const bool use_cube = cube && rq.value("cube", true);
                if (use_cube)
                    bg_image = cube->render(*cam, rs, nullptr, nullptr);

                std::vector<std::string> save = rq.contains("save") ? rq["save"].get<std::vector<std::string>>() : def.save;
                auto wants = [&](const char* w) { return std::find(save.begin(), save.end(), w) != save.end(); };
                const bool want_depth = wants("depth");
                const bool f16 = rq.contains("dtype") ? rq["dtype"].get<std::string>() == "f16" : def.f16;

                auto result = lfs::training::fast_rasterize_forward(
                    *cam, *model.splat, bg_t, 0, 0, 0, 0, /*mip_filter=*/false, bg_image, /*render_normal=*/false,
                    /*render_depth=*/want_depth, 1.0f, /*update_screen_share=*/false, {}, 0, rs, {});
                if (!result)
                    throw std::runtime_error("render failed: " + std::string(result.error().detail()));
                lfs::training::RenderOutput o = std::move(result->first);
                result->second.release_forward_context();
                cuda_ok(cudaDeviceSynchronize(), "render");

                fs::create_directories(out);
                if (wants("image"))
                    write_npy_float(out / (id + "_image.npy"), chw_to_hwc(download(o.image), H, W), {size_t(H), size_t(W), 3}, f16);
                if (wants("alpha"))
                    write_npy_float(out / (id + "_alpha.npy"), download(o.alpha), {size_t(H), size_t(W)}, f16);
                if (want_depth)
                    write_npy_float(out / (id + "_depth.npy"), download(o.depth), {size_t(H), size_t(W)}, f16);
                const std::string post = rq.value("post", def.post);
                if (wants("post") || post != "none") {
                    if (post == "none")
                        throw std::runtime_error("save post needs a post mode");
                    if (!have_ppisp)
                        throw std::runtime_error("post " + post + " needs --ppisp");
                    Tensor img = o.image.contiguous();
                    Tensor pi;
                    const float ev = rq.value("post_ev", 0.0f);
                    if (post == "curve") {
                        lfs::training::PPISPRenderOverrides ov;
                        ov.vignette_enabled = false;
                        pi = ppisp.ppisp->apply_with_exposure_and_overrides(img, camera_id, ev, ov);
                    } else if (post == "curve_vig") {
                        pi = ppisp.ppisp->apply_with_exposure(img, camera_id, ev);
                    } else if (post == "frame") {
                        if (!ppisp.uid_of_name.contains(frame))
                            throw std::runtime_error("post frame: '" + frame + "' is not a frame of the PPISP dump");
                        pi = ppisp.ppisp->apply(img, ppisp.camera_of_uid[uid], uid);
                    } else {
                        throw std::runtime_error("unknown post mode " + post);
                    }
                    cuda_ok(cudaDeviceSynchronize(), "post");
                    write_npy_float(out / (id + "_post.npy"), chw_to_hwc(download(pi), H, W), {size_t(H), size_t(W), 3}, f16);
                }
            }
        };

        std::map<std::string, std::vector<float>> read_motion_file(const fs::path& p) {
            std::map<std::string, std::vector<float>> m;
            std::ifstream in(p);
            if (!in)
                throw std::runtime_error("cannot open motion file " + p.string());
            for (std::string line; std::getline(in, line);) {
                if (line.empty() || line[0] == '#')
                    continue;
                const auto t = split_ws(line);
                if (t.size() < 13)
                    continue;
                std::vector<float> v;
                for (int k = 1; k <= 12; ++k)
                    v.push_back(std::stof(t[k]));
                m[t[0]] = v;
            }
            return m;
        }

        const std::vector<float>* motion_lookup(const std::map<std::string, std::vector<float>>& m, const std::string& name) {
            for (const std::string& k : {name, name + ".png", name.size() > 4 && name.ends_with(".png") ? name.substr(0, name.size() - 4) : name})
                if (auto it = m.find(k); it != m.end())
                    return &it->second;
            return nullptr;
        }

        struct ColmapCam {
            CameraSpec spec;
        };

        std::map<int, CameraSpec> read_cameras_txt(const fs::path& p) {
            std::map<int, CameraSpec> cams;
            std::ifstream in(p);
            if (!in)
                throw std::runtime_error("cannot open " + p.string());
            for (std::string line; std::getline(in, line);) {
                if (line.empty() || line[0] == '#')
                    continue;
                const auto t = split_ws(line);
                if (t.size() < 5)
                    continue;
                CameraSpec c;
                c.model = t[1];
                c.width = std::stoi(t[2]);
                c.height = std::stoi(t[3]);
                for (size_t k = 4; k < t.size(); ++k)
                    c.params.push_back(std::stod(t[k]));
                cams[std::stoi(t[0])] = c;
            }
            return cams;
        }

        // images.txt: the non-comment lines, every other one (as abq/pose_opt_fish read_model: L[0::2])
        std::vector<std::vector<std::string>> read_images_txt(const fs::path& p) {
            std::ifstream in(p);
            if (!in)
                throw std::runtime_error("cannot open " + p.string());
            std::vector<std::string> L;
            for (std::string line; std::getline(in, line);) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (!line.empty() && line[0] == '#')
                    continue;
                L.push_back(line);
            }
            std::vector<std::vector<std::string>> rows;
            for (size_t i = 0; i < L.size(); i += 2) {
                auto t = split_ws(L[i]);
                if (t.size() >= 10)
                    rows.push_back(std::move(t));
            }
            return rows;
        }

        std::string arg_value(int& i, int argc, const char* const* argv) {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value after ") + argv[i]);
            return argv[++i];
        }

        int render_main(int argc, const char* const* argv) {
            Renderer r;
            fs::path ply, cube_path, ppisp_path, views, colmap, motion_path, names_path;
            std::string lens_radial;
            int sh_degree = -1;
            float max_std_dev = 0.0f;
            bool serve = false;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--ply") ply = arg_value(i, argc, argv);
                else if (a == "--sh-degree") sh_degree = std::stoi(arg_value(i, argc, argv));
                else if (a == "--lens-radial") lens_radial = arg_value(i, argc, argv);
                else if (a == "--max-std-dev") max_std_dev = std::stof(arg_value(i, argc, argv));
                else if (a == "--max-std-dev-distance") g_cut_distance = true;
                else if (a == "--bg") {
                    const auto v = parse_floats(arg_value(i, argc, argv));
                    if (v.size() != 3) throw std::runtime_error("--bg R,G,B");
                    r.bg = {v[0], v[1], v[2]};
                } else if (a == "--bg-cubemap") cube_path = arg_value(i, argc, argv);
                else if (a == "--ppisp") ppisp_path = arg_value(i, argc, argv);
                else if (a == "--serve") serve = true;
                else if (a == "--views") views = arg_value(i, argc, argv);
                else if (a == "--colmap") colmap = arg_value(i, argc, argv);
                else if (a == "--out") r.def.out = arg_value(i, argc, argv);
                else if (a == "--factor") r.def.factor = std::stoi(arg_value(i, argc, argv));
                else if (a == "--readout-ms") r.def.readout_ms = std::stof(arg_value(i, argc, argv));
                else if (a == "--rotation-only") r.def.rotation_only = true;
                else if (a == "--post") r.def.post = arg_value(i, argc, argv);
                else if (a == "--save") {
                    r.def.save.clear();
                    std::stringstream ss(arg_value(i, argc, argv));
                    for (std::string s; std::getline(ss, s, ',');)
                        if (!s.empty()) r.def.save.push_back(s);
                } else if (a == "--dtype") r.def.f16 = arg_value(i, argc, argv) == "f16";
                else if (a == "--motion") motion_path = arg_value(i, argc, argv);
                else if (a == "--names") names_path = arg_value(i, argc, argv);
                else throw std::runtime_error("lfe-render: unknown option " + a);
            }
            if (ply.empty() || (!serve && views.empty() && colmap.empty()))
                throw std::runtime_error("usage: lfe-render --ply MODEL.ply [options] (--serve | --views FILE | --colmap DIR) (see the header of src/app/lfe_render_tool.cpp)");
            load_model(r.model, ply, sh_degree, lens_radial, max_std_dev);
            if (!cube_path.empty()) {
                const fs::path faces = fs::is_directory(cube_path) ? cube_path / "faces.npy" : cube_path;
                // the face resolution from the npy header ([6, R, R, 3]) so the cube is used as stored (no resampling)
                std::ifstream f(faces, std::ios::binary);
                std::string head(256, '\0');
                f.read(head.data(), 256);
                const size_t s = head.find("'shape': (");
                if (s == std::string::npos)
                    throw std::runtime_error("cannot read the shape of " + faces.string());
                int six = 0, res = 0;
                if (std::sscanf(head.c_str() + s + 10, "%d, %d", &six, &res) != 2 || six != 6)
                    throw std::runtime_error("unexpected cube shape in " + faces.string());
                lfs::training::BackgroundCubemapSettings cset;
                cset.res = res;
                cset.init_path = faces.string();
                std::string err;
                r.cube = lfs::training::BackgroundCubemap::create(cset, err);
                if (!r.cube)
                    throw std::runtime_error("background cubemap: " + err);
                std::println("background = the learned cubemap {} ({}), composited by the rasterizer", faces.string(),
                             r.cube->description());
            }
            if (!ppisp_path.empty()) {
                load_ppisp(r.ppisp, ppisp_path);
                r.have_ppisp = true;
            }
            auto run_one = [&](const json& rq, bool protocol) {
                const auto t0 = std::chrono::steady_clock::now();
                const std::string id = rq.value("id", std::string("?"));
                try {
                    r.render(rq);
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    if (protocol)
                        std::println("@@LFS ok {} {:.1f}", id, ms);
                    else
                        std::println("rendered {} ({:.0f} ms)", id, ms);
                    std::fflush(stdout);
                    return true;
                } catch (const std::exception& e) {
                    std::string why = e.what();
                    std::replace(why.begin(), why.end(), '\n', ' ');
                    if (protocol)
                        std::println("@@LFS err {} {}", id, why);
                    else
                        std::println(stderr, "render {} failed: {}", id, why);
                    std::fflush(stdout);
                    return false;
                }
            };
            if (serve) {
                std::println("@@LFS ready");
                std::fflush(stdout);
                for (std::string line; std::getline(std::cin, line);) {
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();
                    if (line.empty())
                        continue;
                    if (line == "quit")
                        break;
                    json rq;
                    try {
                        rq = json::parse(line);
                    } catch (const std::exception& e) {
                        std::println("@@LFS err ? bad json: {}", e.what());
                        std::fflush(stdout);
                        continue;
                    }
                    run_one(rq, true);
                }
                return 0;
            }
            int failed = 0, n = 0;
            if (!views.empty()) {
                std::ifstream in(views);
                if (!in)
                    throw std::runtime_error("cannot open " + views.string());
                for (std::string line; std::getline(in, line);) {
                    if (line.empty() || line[0] == '#')
                        continue;
                    ++n;
                    failed += run_one(json::parse(line), false) ? 0 : 1;
                }
            }
            if (!colmap.empty()) {
                const auto cams = read_cameras_txt(colmap / "cameras.txt");
                const auto rows = read_images_txt(colmap / "images.txt");
                std::map<std::string, std::vector<float>> motion;
                if (!motion_path.empty())
                    motion = read_motion_file(motion_path);
                std::vector<std::string> only;
                if (!names_path.empty()) {
                    std::ifstream in(names_path);
                    for (std::string s; std::getline(in, s);)
                        if (!s.empty()) only.push_back(s);
                }
                for (const auto& t : rows) {
                    const std::string name = t[9];
                    const std::string stem = fs::path(name).stem().string();
                    if (!only.empty() && std::find(only.begin(), only.end(), name) == only.end() &&
                        std::find(only.begin(), only.end(), stem) == only.end())
                        continue;
                    const int cid = std::stoi(t[8]);
                    if (!cams.contains(cid))
                        throw std::runtime_error("images.txt names camera " + t[8] + " which cameras.txt lacks");
                    const auto& c = cams.at(cid);
                    json rq = {{"id", stem}, {"frame", name}, {"camera_id", cid},
                               {"camera", {{"model", c.model}, {"width", c.width}, {"height", c.height}, {"params", c.params}}},
                               {"qvec", {t[1], t[2], t[3], t[4]}}, {"tvec", {t[5], t[6], t[7]}}};
                    if (!motion.empty()) {
                        const auto* mo = motion_lookup(motion, name);
                        if (mo)
                            rq["motion"] = *mo;
                    }
                    ++n;
                    failed += run_one(rq, false) ? 0 : 1;
                }
            }
            std::println("lfe-render: {} of {} requests rendered", n - failed, n);
            return failed ? 1 : 0;
        }

        // ------------------------------------------------------------------ registration (lfe-register)
        double env_d(const char* k, double d) {
            const char* v = std::getenv(k);
            return v && *v ? std::stod(v) : d;
        }
        std::string env_s(const char* k, const std::string& d = "") {
            const char* v = std::getenv(k);
            return v && *v ? std::string(v) : d;
        }

        struct Frame {
            std::vector<uint8_t> rgb;  // [h, w, 3]
            std::vector<uint8_t> mask; // [h, w] 0/1
            double n_kept = 0;
        };

        void load_frame(const fs::path& img, const fs::path& msk, int W, int H, Frame& f) {
            auto [data, w, h, c] = lfs::core::load_image(img);
            if (!data)
                throw std::runtime_error("cannot read " + img.string());
            if (w != W || h != H) {
                lfs::core::free_image(data);
                throw std::runtime_error(std::format("{}: {}x{}, expected {}x{}", img.string(), w, h, W, H));
            }
            f.rgb.resize(static_cast<size_t>(W) * H * 3);
            for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i)
                for (int k = 0; k < 3; ++k)
                    f.rgb[i * 3 + k] = data[i * c + std::min(k, c - 1)];
            lfs::core::free_image(data);
            f.mask.assign(static_cast<size_t>(W) * H, 1);
            if (fs::exists(msk)) {
                auto [md, mw, mh, mc] = lfs::core::load_image(msk);
                if (!md || mw != W || mh != H)
                    throw std::runtime_error("bad mask " + msk.string());
                for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i)
                    f.mask[i] = md[i * mc] > 127 ? 1 : 0;
                lfs::core::free_image(md);
            }
            f.n_kept = 0;
            for (const auto m : f.mask)
                f.n_kept += m;
        }

        void solve4x3(const double G0[16], const double Y[12], double sol[12]) {
            double A[4][7];
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c)
                    A[r][c] = G0[r * 4 + c] + (r == c ? 1e-6 : 0.0);
                for (int c = 0; c < 3; ++c)
                    A[r][4 + c] = Y[r * 3 + c];
            }
            for (int col = 0; col < 4; ++col) {
                int piv = col;
                for (int r = col + 1; r < 4; ++r)
                    if (std::abs(A[r][col]) > std::abs(A[piv][col]))
                        piv = r;
                if (piv != col)
                    for (int c = 0; c < 7; ++c)
                        std::swap(A[col][c], A[piv][c]);
                for (int r = 0; r < 4; ++r) {
                    if (r == col)
                        continue;
                    const double f = A[r][col] / A[col][col];
                    for (int c = col; c < 7; ++c)
                        A[r][c] -= f * A[col][c];
                }
            }
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 3; ++c)
                    sol[r * 3 + c] = A[r][4 + c] / A[r][r];
        }

        // minimal .npy reader (little-endian float32 / uint8 / bool, C order) for the loss check
        std::vector<float> read_npy(const fs::path& p, std::vector<size_t>& shape) {
            std::ifstream f(p, std::ios::binary);
            if (!f)
                throw std::runtime_error("cannot open " + p.string());
            char magic[8];
            f.read(magic, 8);
            uint16_t hl = 0;
            f.read(reinterpret_cast<char*>(&hl), 2);
            std::string hdr(hl, '\0');
            f.read(hdr.data(), hl);
            const bool f4 = hdr.find("'<f4'") != std::string::npos;
            const bool u1 = hdr.find("'|u1'") != std::string::npos || hdr.find("'|b1'") != std::string::npos;
            if (!f4 && !u1)
                throw std::runtime_error("npy dtype must be <f4, |u1 or |b1: " + p.string());
            if (hdr.find("'fortran_order': False") == std::string::npos)
                throw std::runtime_error("npy must be C order: " + p.string());
            const size_t s0 = hdr.find('(') + 1, s1 = hdr.find(')');
            shape.clear();
            std::stringstream ss(hdr.substr(s0, s1 - s0));
            for (std::string item; std::getline(ss, item, ',');)
                if (item.find_first_not_of(' ') != std::string::npos)
                    shape.push_back(std::stoull(item));
            size_t n = 1;
            for (const auto d : shape)
                n *= d;
            std::vector<float> v(n);
            if (f4) {
                f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
            } else {
                std::vector<uint8_t> b(n);
                f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(n));
                for (size_t i = 0; i < n; ++i)
                    v[i] = b[i];
            }
            if (!f)
                throw std::runtime_error("short npy " + p.string());
            return v;
        }

        // lfe-register --losscheck R.npy P.npy M.npy OUT: the loss + its image gradient for one render (test of
        // lfe_register_kernels.cu against abq/rs_render_fast.py's colour_fit_ns / _loss_one under torch autograd)
        int losscheck_main(const fs::path& rp, const fs::path& pp, const fs::path& mp, const fs::path& out) {
            std::vector<size_t> sr, sp, sm;
            const auto r = read_npy(rp, sr), p = read_npy(pp, sp), m = read_npy(mp, sm);
            if (sr.size() != 3 || sr[2] != 3 || sp != sr || sm.size() != 2 || sm[0] != sr[0] || sm[1] != sr[1])
                throw std::runtime_error("losscheck: R, P [H, W, 3] float32, M [H, W]");
            const int H = static_cast<int>(sr[0]), W = static_cast<int>(sr[1]);
            const size_t HW = static_cast<size_t>(H) * W;
            std::vector<float> rc(3 * HW), pc(3 * HW), grad(3 * HW);
            std::vector<uint8_t> mk(HW);
            double nk = 0;
            for (size_t i = 0; i < HW; ++i) {
                for (int c = 0; c < 3; ++c) {
                    rc[c * HW + i] = r[i * 3 + c];
                    pc[c * HW + i] = p[i * 3 + c];
                }
                mk[i] = m[i] > 0.5f ? 1 : 0;
                nk += mk[i];
            }
            float *d_r = nullptr, *d_p = nullptr, *d_g = nullptr;
            uint8_t* d_m = nullptr;
            double* d_acc = nullptr;
            cuda_ok(cudaMalloc(&d_r, HW * 12), "alloc");
            cuda_ok(cudaMalloc(&d_p, HW * 12), "alloc");
            cuda_ok(cudaMalloc(&d_g, HW * 12), "alloc");
            cuda_ok(cudaMalloc(&d_m, HW), "alloc");
            cuda_ok(cudaMalloc(&d_acc, 28 * 8), "alloc");
            cuda_ok(cudaMemcpy(d_r, rc.data(), HW * 12, cudaMemcpyHostToDevice), "up");
            cuda_ok(cudaMemcpy(d_p, pc.data(), HW * 12, cudaMemcpyHostToDevice), "up");
            cuda_ok(cudaMemcpy(d_m, mk.data(), HW, cudaMemcpyHostToDevice), "up");
            cuda_ok(cudaMemset(d_acc, 0, 28 * 8), "zero");
            regk::affine_normal_equations(d_r, d_p, d_m, H, W, 4, d_acc, nullptr);
            double acc[28], sol_d[12], sums[2];
            cuda_ok(cudaMemcpy(acc, d_acc, sizeof(acc), cudaMemcpyDeviceToHost), "acc");
            solve4x3(acc, acc + 16, sol_d);
            float sol[12];
            for (int k = 0; k < 12; ++k)
                sol[k] = static_cast<float>(sol_d[k]);
            {
                regk::LossWorkspace ws;
                regk::loss_and_grad(d_r, d_p, d_m, H, W, sol, nk, ws, d_g, nullptr);
                cuda_ok(cudaDeviceSynchronize(), "loss");
                cuda_ok(cudaMemcpy(sums, ws.sums, sizeof(sums), cudaMemcpyDeviceToHost), "sums");
            }
            cuda_ok(cudaMemcpy(grad.data(), d_g, HW * 12, cudaMemcpyDeviceToHost), "grad");
            const double n = std::max(nk, 1.0);
            fs::create_directories(out);
            {
                std::ofstream t(out / "losscheck.txt");
                t << std::format("n_kept {:.0f}\nl1 {:.9g}\nssim {:.9g}\nloss {:.9g}\nsol", nk, sums[0] / n, sums[1] / n,
                                 0.8 * sums[0] / n + 0.2 * (1.0 - sums[1] / n));
                for (const double s : sol_d)
                    t << std::format(" {:.9g}", s);
                t << "\n";
            }
            std::vector<float> g_hwc(3 * HW);
            for (size_t i = 0; i < HW; ++i)
                for (int c = 0; c < 3; ++c)
                    g_hwc[i * 3 + c] = grad[c * HW + i];
            write_npy_float(out / "grad.npy", g_hwc, {size_t(H), size_t(W), 3}, false);
            std::println("losscheck: loss {:.9g} (n {:.0f}) -> {}", 0.8 * sums[0] / n + 0.2 * (1.0 - sums[1] / n), nk, out.string());
            cudaFree(d_r);
            cudaFree(d_p);
            cudaFree(d_g);
            cudaFree(d_m);
            cudaFree(d_acc);
            return 0;
        }

        // a checksum of every splat parameter buffer on the device (the frozen-model check of lfe-register)
        uint64_t model_checksum(lfs::core::SplatData& sp) {
            cuda_ok(cudaDeviceSynchronize(), "sync");
            uint64_t h = 1469598103934665603ull;
            for (const Tensor* t : {&sp.means(), &sp.scaling_raw(), &sp.rotation_raw(), &sp.opacity_raw(), &sp.sh0(), &sp.shN()}) {
                if (!t->is_valid() || t->numel() == 0)
                    continue;
                std::vector<uint32_t> v(t->numel());
                cuda_ok(cudaMemcpy(v.data(), t->ptr<float>(), v.size() * 4, cudaMemcpyDeviceToHost), "checksum");
                for (const uint32_t x : v)
                    h = (h ^ x) * 1099511628211ull;
            }
            return h;
        }

        int register_main(int argc, const char* const* argv) {
            if (argc >= 7 && std::string(argv[2]) == "--losscheck")
                return losscheck_main(argv[3], argv[4], argv[5], argv[6]);
            if (argc >= 5 && std::string(argv[2]) == "--decode") { // the frame decoder lfe-register uses (lfs::core::load_image) -> .npy
                auto [data, w, h, c] = lfs::core::load_image(argv[3]);
                if (!data)
                    throw std::runtime_error(std::string("cannot read ") + argv[3]);
                const std::string bytes = npy_bytes(data, static_cast<size_t>(w) * h * c, "|u1", {size_t(h), size_t(w), size_t(c)});
                lfs::core::free_image(data);
                write_file(argv[4], bytes);
                std::println("decoded {} ({}x{}x{}) -> {}", argv[3], w, h, c, argv[4]);
                return 0;
            }
            std::vector<std::string> pos;
            float max_std_dev = 0.0f;
            int sh_degree = -1;
            for (int i = 2; i < argc; ++i) {
                const std::string a = argv[i];
                if (a == "--max-std-dev") max_std_dev = std::stof(arg_value(i, argc, argv));
                else if (a == "--max-std-dev-distance") g_cut_distance = true;
                else if (a == "--sh-degree") sh_degree = std::stoi(arg_value(i, argc, argv));
                else pos.push_back(a);
            }
            if (pos.size() < 3)
                throw std::runtime_error("usage: lfe-register DS MODEL.ply OUT [STEPS [POSE_LR]] [--max-std-dev X] [--sh-degree D]");
            const fs::path ds = pos[0], ply = pos[1], out = pos[2];
            const int steps = pos.size() > 3 ? std::stoi(pos[3]) : 150000;
            const double lr0 = pos.size() > 4 ? std::stod(pos[4]) : 3e-5;
            const int decay_after = static_cast<int>(env_d("POSE_DECAY_AFTER", 20000));
            const double half_life = env_d("POSE_HALF_LIFE", 10000);
            const int snap = static_cast<int>(env_d("POSE_SNAP_EVERY", 10000));
            const double settle_turn = env_d("POSE_SETTLE_TURN", 0.005), settle_move = env_d("POSE_SETTLE_MOVE", 0.0005);
            const int factor = static_cast<int>(env_d("FC_FACTOR", 2));
            const std::string cache = env_s("FC_CACHE", std::format("D:/splat_work/fishcam/half{}", factor));
            const bool no_drift = env_s("FC_NO_DRIFT") == "1";
            const std::string tsc = env_s("POSE_T_SCALE");
            if (tsc.empty())
                throw std::runtime_error("lfe-register needs POSE_T_SCALE (abq/register_test.py computes it like the Python refiner)");
            const double t_scale = std::stod(tsc);
            for (const char* unsupported : {"FC_LENS", "FC_CAMS", "FC_PERTURB", "FC_LENS_PERTURB"})
                if (env_s(unsupported) != "" && !(std::string(unsupported) == "FC_LENS" && env_s("FC_LENS") == "0"))
                    throw std::runtime_error(std::format("lfe-register: {} is not supported (use abq/pose_opt_fish_fast.py)", unsupported));
            if (env_s("FC_COLOR", "affine") != "affine")
                throw std::runtime_error("lfe-register: only FC_COLOR=affine is supported");

            // the dataset: one OPENCV_FISHEYE camera, the images' start poses (c2w, COLMAP world)
            const auto cams = read_cameras_txt(ds / "sparse/0/cameras.txt");
            if (cams.size() != 1 || cams.begin()->second.model != "OPENCV_FISHEYE")
                throw std::runtime_error("lfe-register: DS needs exactly one OPENCV_FISHEYE camera");
            const CameraSpec cs = cams.begin()->second;
            const auto rows = read_images_txt(ds / "sparse/0/images.txt");
            const size_t N = rows.size();
            if (N == 0)
                throw std::runtime_error("no images in " + ds.string());
            std::vector<std::string> names(N);
            std::vector<M4> C0(N);
            for (size_t i = 0; i < N; ++i) {
                const auto& t = rows[i];
                names[i] = t[9];
                const M3 Rw = quat_to_rot_d(std::stod(t[1]), std::stod(t[2]), std::stod(t[3]), std::stod(t[4]));
                const std::array<double, 3> tv{std::stod(t[5]), std::stod(t[6]), std::stod(t[7])};
                const M3 Rt = transpose3(Rw);
                const auto c = mv3(Rt, tv);
                M4 m{};
                for (int r = 0; r < 3; ++r) {
                    for (int k = 0; k < 3; ++k)
                        m[r * 4 + k] = Rt[r * 3 + k];
                    m[r * 4 + 3] = -c[r];
                }
                m[15] = 1;
                C0[i] = m;
            }
            const int W = cs.width / factor, H = cs.height / factor;

            LoadedModel model;
            load_model(model, ply, sh_degree, env_s("FC_LENS_RADIAL"), max_std_dev);
            if (model.table)
                std::println("FC_LENS_RADIAL in use: radial lens table {}", env_s("FC_LENS_RADIAL"));

            // rolling shutter
            std::map<std::string, std::vector<float>> motion;
            const std::string rs_path = env_s("FC_RS");
            const float readout_s = static_cast<float>(env_d("FC_RS_MS", 10.0)) * 1e-3f; // as the trainer: ms * 1e-3f
            std::vector<std::vector<float>> mot(N);
            if (!rs_path.empty()) {
                motion = read_motion_file(rs_path);
                for (size_t i = 0; i < N; ++i) {
                    const auto* m = motion_lookup(motion, names[i]);
                    if (!m)
                        throw std::runtime_error("FC_RS: no motion for " + names[i]);
                    mot[i] = *m;
                }
                std::println("dynamic rolling shutter (FC_RS): {} cameras, readout {:g} ms (LichtFeld's rolling-shutter kernels)",
                             N, readout_s * 1000.0f);
            }

            // background
            std::array<float, 3> bgv{0, 0, 0};
            if (const auto b = env_s("FC_BG"); !b.empty()) {
                const auto v = parse_floats(b);
                if (v.size() != 3)
                    throw std::runtime_error("FC_BG r,g,b");
                for (int k = 0; k < 3; ++k)
                    bgv[k] = v[k] / 255.0f;
            }
            Tensor bg_t = Tensor::from_vector(std::vector<float>(bgv.begin(), bgv.end()), {3}, Device::CPU).to(Device::CUDA);

            // cameras (one per frame; their own transforms are replaced by the refined pose each step)
            std::vector<std::unique_ptr<Camera>> camera(N);
            for (size_t i = 0; i < N; ++i) {
                const M4 w2c = rigid_inverse(C0[i]);
                std::array<float, 9> R{};
                std::array<float, 3> T{};
                for (int r = 0; r < 3; ++r) {
                    for (int k = 0; k < 3; ++k)
                        R[r * 3 + k] = static_cast<float>(w2c[r * 4 + k]);
                    T[r] = static_cast<float>(w2c[r * 4 + 3]);
                }
                camera[i] = make_camera(cs, R, T, factor, names[i], static_cast<int>(i), cams.begin()->first);
                if (camera[i]->image_width() != W || camera[i]->image_height() != H)
                    throw std::runtime_error("camera size");
                if (model.table) {
                    const auto why = lfs::training::radial_lens_camera_mismatch(*model.table, *camera[i]);
                    if (!why.empty())
                        throw std::runtime_error("FC_LENS_RADIAL: the camera is not the table's camera: " + why);
                }
            }

            // the frames (the Python refiner's half-res cache), loaded in parallel
            std::vector<Frame> frames(N);
            {
                std::vector<std::thread> pool;
                std::vector<std::string> errors(8);
                const unsigned nt = 8;
                for (unsigned t = 0; t < nt; ++t)
                    pool.emplace_back([&, t] {
                        try {
                            for (size_t i = t; i < N; i += nt)
                                load_frame(fs::path(cache) / "images" / names[i], fs::path(cache) / "masks" / (names[i] + ".png"), W, H, frames[i]);
                        } catch (const std::exception& e) {
                            errors[t] = e.what();
                        }
                    });
                for (auto& th : pool)
                    th.join();
                for (const auto& e : errors)
                    if (!e.empty())
                        throw std::runtime_error(e);
            }
            std::println("{} frames {}x{} from {} (render {}x{}, factor {})", N, W, H, cache, W, H, factor);

            // the draw order
            std::vector<int> order_file;
            if (const auto of = env_s("FC_ORDER"); !of.empty()) {
                std::ifstream in(of);
                for (std::string s; std::getline(in, s);)
                    if (!s.empty())
                        order_file.push_back(std::stoi(s));
                std::println("draw order: {} ({} draws)", of, order_file.size());
            }
            std::mt19937 rng(0);
            std::vector<int> order;
            size_t order_pos = 0;
            auto draw = [&]() -> int {
                if (order_pos < order_file.size()) {
                    const int j = order_file[order_pos++];
                    if (j < 0 || static_cast<size_t>(j) >= N)
                        throw std::runtime_error("FC_ORDER: index out of range");
                    return j;
                }
                if (order.empty()) {
                    order.resize(N);
                    std::iota(order.begin(), order.end(), 0);
                    std::shuffle(order.begin(), order.end(), rng);
                }
                const int j = order.back();
                order.pop_back();
                return j;
            };

            // device buffers
            uint8_t *d_rgb = nullptr, *d_mask = nullptr;
            float *d_photo = nullptr, *d_grad = nullptr, *d_w2c = nullptr, *d_cam = nullptr, *d_pg = nullptr,
                  *d_motion = nullptr, *d_readout = nullptr;
            double* d_acc = nullptr;
            const size_t HW = static_cast<size_t>(W) * H;
            cuda_ok(cudaMalloc(&d_rgb, HW * 3), "alloc");
            cuda_ok(cudaMalloc(&d_mask, HW), "alloc");
            cuda_ok(cudaMalloc(&d_photo, HW * 12), "alloc");
            cuda_ok(cudaMalloc(&d_grad, HW * 12), "alloc");
            cuda_ok(cudaMalloc(&d_w2c, 64), "alloc");
            cuda_ok(cudaMalloc(&d_cam, 12), "alloc");
            cuda_ok(cudaMalloc(&d_pg, 24), "alloc");
            cuda_ok(cudaMalloc(&d_motion, 48), "alloc");
            cuda_ok(cudaMalloc(&d_readout, 4), "alloc");
            cuda_ok(cudaMalloc(&d_acc, 28 * 8), "alloc");
            cuda_ok(cudaMemcpy(d_readout, &readout_s, 4, cudaMemcpyHostToDevice), "readout");
            regk::LossWorkspace ws;

            std::println("warm start: translation steps x {:.3f} (the cold runs' normalised units)", t_scale);
            std::println("lfe-register (LFQPLCYR): {} cameras against the frozen model, LichtFeld forward + pose gradient; "
                         "snapshot every {}, lr {:.2e} held to step {}, then halving every {:g}; settle < {:g} deg and < {:g} units; cap {}",
                         N, snap, lr0, decay_after, half_life, settle_turn, settle_move, steps);
            std::fflush(stdout);

            // optimiser state: emb [N, 9] (t 3, d6 6), torch Adam (coupled weight decay 1e-6), one global step counter
            std::vector<float> emb(N * 9, 0.0f), am(N * 9, 0.0f), av(N * 9, 0.0f), grad(N * 9, 0.0f);
            const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 1e-6f;
            auto lr_of = [&](int s) { return s < decay_after ? lr0 : lr0 * std::pow(0.5, (s - decay_after) / half_life); };
            auto current = [&]() {
                std::vector<M4> cur(N);
                for (size_t i = 0; i < N; ++i)
                    cur[i] = mul4(C0[i], delta_matrix(&emb[i * 9], t_scale));
                return cur;
            };
            std::vector<double> first_loss(N, std::nan("")), last_loss(N, std::nan(""));
            std::vector<int> visits(N, 0);
            std::vector<double> win;
            std::optional<std::vector<M4>> prev;
            const auto t_start = std::chrono::steady_clock::now();
            auto minutes = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() / 60.0; };
            fs::create_directories(out);
            std::ofstream settle(out / "settle.tsv");
            settle << "step\tvisits_per_cam\tminutes\tlr\tmean_loss\tmove_since\tturn_since\tdrift_scale\tdrift_rot_deg\t"
                      "lens_edge_shift_since_px\tfx\tfy\tcx\tcy\tk1\tk2\tk3\tk4\n";
            const std::vector<double> lens(cs.params.begin(), cs.params.end());
            auto npz_poses = [&](const fs::path& p, const std::vector<M4>& newp) {
                std::vector<double> a, b;
                for (size_t i = 0; i < N; ++i) {
                    a.insert(a.end(), C0[i].begin(), C0[i].end());
                    b.insert(b.end(), newp[i].begin(), newp[i].end());
                }
                write_npz(p, {{"names", npy_unicode(names)}, {"c2w_old", npy_f64(a, {N, 4, 4})}, {"c2w_new", npy_f64(b, {N, 4, 4})},
                              {"lens", npy_f64(lens, {lens.size()})}});
            };
            const size_t n_splats = model.splat->size();
            const uint64_t checksum_start = model_checksum(*model.splat);
            std::string stop_reason;
            int n_done = 0;
            for (int step = 0; step < steps; ++step) {
                const int j = draw();
                const M4 c2w = mul4(C0[j], delta_matrix(&emb[j * 9], t_scale));
                const M4 w2c = rigid_inverse(c2w);
                float w2c_f[16], cam_f[3];
                for (int k = 0; k < 16; ++k)
                    w2c_f[k] = static_cast<float>(w2c[k]);
                for (int k = 0; k < 3; ++k)
                    cam_f[k] = static_cast<float>(c2w[k * 4 + 3]);
                cuda_ok(cudaMemcpy(d_w2c, w2c_f, 64, cudaMemcpyHostToDevice), "w2c");
                cuda_ok(cudaMemcpy(d_cam, cam_f, 12, cudaMemcpyHostToDevice), "cam");
                cuda_ok(cudaMemset(d_pg, 0, 24), "pg");
                fr::FastGSRollingShutter rs{};
                if (!mot[j].empty()) {
                    cuda_ok(cudaMemcpy(d_motion, mot[j].data(), 48, cudaMemcpyHostToDevice), "motion");
                    rs.motion = d_motion;
                    rs.readout = d_readout;
                    rs.iters = 2;
                }
                fr::FastGSPose pose{};
                pose.w2c = d_w2c;
                pose.cam_position = d_cam;
                pose.grad = d_pg;
                cuda_ok(cudaMemcpy(d_rgb, frames[j].rgb.data(), HW * 3, cudaMemcpyHostToDevice), "frame");
                cuda_ok(cudaMemcpy(d_mask, frames[j].mask.data(), HW, cudaMemcpyHostToDevice), "mask");
                regk::photo_to_chw(d_rgb, d_photo, H, W, nullptr);

                auto result = lfs::training::fast_rasterize_forward(*camera[j], *model.splat, bg_t, 0, 0, 0, 0, false, {}, false,
                                                                    false, 1.0f, false, {}, 0, rs, pose);
                if (!result)
                    throw std::runtime_error("forward failed: " + std::string(result.error().detail()));
                lfs::training::RenderOutput o = std::move(result->first);
                auto& ctx = result->second;
                cuda_ok(cudaDeviceSynchronize(), "forward");

                // colour affine (detached), then the loss and its image gradient
                cuda_ok(cudaMemset(d_acc, 0, 28 * 8), "acc");
                regk::affine_normal_equations(o.image.ptr<float>(), d_photo, d_mask, H, W, 4, d_acc, nullptr);
                double acc[28];
                cuda_ok(cudaMemcpy(acc, d_acc, sizeof(acc), cudaMemcpyDeviceToHost), "acc");
                double sol_d[12];
                solve4x3(acc, acc + 16, sol_d);
                float sol[12];
                for (int k = 0; k < 12; ++k)
                    sol[k] = static_cast<float>(sol_d[k]);
                regk::loss_and_grad(o.image.ptr<float>(), d_photo, d_mask, H, W, sol, frames[j].n_kept, ws, d_grad, nullptr);
                double sums[2];
                cuda_ok(cudaMemcpy(sums, ws.sums, sizeof(sums), cudaMemcpyDeviceToHost), "sums");
                const double n = std::max(frames[j].n_kept, 1.0);
                const double loss = 0.8 * sums[0] / n + 0.2 * (1.0 - sums[1] / n);

                // the rasterizer backward with the pose gradient bound and NOTHING captured: the splats never move
                fr::BackgroundAlphaGradient bgg{};
                bgg.bg_color = ctx.bg_color.ptr<float>();
                fr::FusedAdamSettings fa{};
                fa.enabled = true;
                auto bres = fr::backward_raw(nullptr, nullptr, d_grad, bgg, nullptr, nullptr, ctx.image.ptr<float>(),
                                             ctx.alpha.ptr<float>(), ctx.means.ptr<float>(), ctx.raw_scales.ptr<float>(),
                                             ctx.raw_rotations.ptr<float>(), ctx.raw_opacities.ptr<float>(),
                                             ctx.shN.is_valid() ? ctx.shN.ptr<float>() : nullptr, ctx.w2c_ptr,
                                             ctx.cam_position_ptr, ctx.forward_ctx, nullptr, static_cast<int>(n_splats),
                                             ctx.forward_ctx.n_visible, ctx.active_sh_bases, ctx.forward_ctx.sh_layout_bases,
                                             ctx.width, ctx.height, ctx.focal_x, ctx.focal_y, ctx.center_x, ctx.center_y,
                                             ctx.mip_filter, DensificationType::None, &fa);
                ctx.mark_forward_context_released();
                if (!bres.success)
                    throw std::runtime_error(std::string("backward failed: ") + bres.error_message);
                float g6[6];
                cuda_ok(cudaMemcpy(g6, d_pg, 24, cudaMemcpyDeviceToHost), "pose grad");

                // chain dL/d(dr, dd) (right perturbation of c2w in its own axes) into the 9 embedding numbers
                Dual R[9];
                rot6d(&emb[j * 9 + 3], -1, R);
                const M3 Rv{R[0].v, R[1].v, R[2].v, R[3].v, R[4].v, R[5].v, R[6].v, R[7].v, R[8].v};
                const auto Rg = mv3(Rv, {g6[3], g6[4], g6[5]});
                std::fill(grad.begin(), grad.end(), 0.0f);
                for (int k = 0; k < 3; ++k)
                    grad[j * 9 + k] = static_cast<float>(t_scale * Rg[k]);
                for (int k = 0; k < 6; ++k) {
                    Dual Rd[9];
                    rot6d(&emb[j * 9 + 3], k, Rd);
                    M3 dR{};
                    for (int q = 0; q < 9; ++q)
                        dR[q] = Rd[q].d;
                    const M3 S = mul3(transpose3(Rv), dR); // skew: [omega]x
                    const double om[3] = {0.5 * (S[7] - S[5]), 0.5 * (S[2] - S[6]), 0.5 * (S[3] - S[1])};
                    grad[j * 9 + 3 + k] = static_cast<float>(g6[0] * om[0] + g6[1] * om[1] + g6[2] * om[2]);
                }

                // torch.optim.Adam (weight_decay = L2 into the gradient), dense over every row
                const int t = step + 1;
                const float lr = static_cast<float>(lr_of(step));
                const double bc1 = 1.0 - std::pow(0.9, t), bc2 = 1.0 - std::pow(0.999, t);
                const float step_size = static_cast<float>(lr / bc1);
                const float bc2_sqrt = static_cast<float>(std::sqrt(bc2));
                for (size_t q = 0; q < N * 9; ++q) {
                    const float g = grad[q] + wd * emb[q];
                    am[q] = am[q] + (g - am[q]) * (1.0f - beta1);
                    av[q] = av[q] * beta2 + (1.0f - beta2) * g * g;
                    const float denom = std::sqrt(av[q]) / bc2_sqrt + eps;
                    emb[q] = emb[q] - step_size * am[q] / denom;
                }
                visits[j] += 1;
                if (std::isnan(first_loss[j]))
                    first_loss[j] = loss;
                last_loss[j] = loss;
                win.push_back(loss);
                n_done = step + 1;

                if (n_done % snap == 0 || n_done == steps) {
                    const auto cur = current();
                    const auto drift = remove_drift(C0, cur);
                    npz_poses(out / std::format("pose_deltas_{}.npz", n_done), drift.fix);
                    std::optional<std::pair<double, double>> since;
                    if (prev) {
                        std::vector<double> mv(N), tn(N);
                        for (size_t i = 0; i < N; ++i) {
                            const auto a = pos_of((*prev)[i]), b = pos_of(drift.fix[i]);
                            mv[i] = std::sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]));
                            tn[i] = rot_angle_deg(rot_of((*prev)[i]), rot_of(drift.fix[i]));
                        }
                        since = std::pair{median(mv), median(tn)};
                    }
                    prev = drift.fix;
                    const double lr_now = lr_of(n_done - 1);
                    const double mean_loss = win.empty() ? 0.0 : std::accumulate(win.begin(), win.end(), 0.0) / static_cast<double>(win.size());
                    win.clear();
                    settle << std::format("{}\t{:.1f}\t{:.1f}\t{:.3e}\t{:.5f}\t", n_done, static_cast<double>(n_done) / N, minutes(), lr_now, mean_loss)
                           << (since ? std::format("{:.6f}\t{:.5f}", since->first, since->second) : std::string("-\t-"))
                           << std::format("\t{:.6f}\t{:.5f}\t0.0000", drift.sg, drift.rg_deg);
                    for (const double v : lens)
                        settle << std::format("\t{:.8g}", v);
                    settle << "\n";
                    settle.flush();
                    std::println("  step {} ({:.1f} per camera, {:.1f} min): loss {:.5f}, lr {:.2e}{}", n_done,
                                 static_cast<double>(n_done) / N, minutes(), mean_loss, lr_now,
                                 since ? std::format(", since last snapshot: move median {:.6f} turn median {:.5f} deg", since->first, since->second) : std::string());
                    if (n_done == decay_after)
                        std::println("  pose lr now decays smoothly from {:.2e}, halving every {:.0f} steps", lr_of(n_done), half_life);
                    std::fflush(stdout);
                    if (n_done >= decay_after) {
                        const bool still = since && since->second < settle_turn && since->first < settle_move;
                        if (n_done > decay_after && (still || lr_now < 0.01 * lr0)) {
                            stop_reason = still ? "cameras still" : "lr at 1% of its start";
                            std::println("settled at step {} ({:.1f} updates per camera, {:.0f} min, pose lr {:.2e}; {})", n_done,
                                         static_cast<double>(n_done) / N, minutes(), lr_now, stop_reason);
                            break;
                        }
                    }
                }
            }
            if (stop_reason.empty())
                std::println("step cap {} reached before the cameras settled", steps);

            // the result
            const auto cur = current();
            std::vector<M4> fin = cur;
            if (no_drift)
                std::println("no drift removal (FC_NO_DRIFT=1)");
            else
                fin = remove_drift(C0, cur).fix;
            npz_poses(out / "pose_deltas.npz", fin);
            {
                const fs::path md = out / "model/sparse/0";
                fs::create_directories(md);
                fs::copy_file(ds / "sparse/0/cameras.txt", md / "cameras.txt", fs::copy_options::overwrite_existing);
                fs::copy_file(ds / "sparse/0/points3D.txt", md / "points3D.txt", fs::copy_options::overwrite_existing);
                std::ofstream im(md / "images.txt");
                im << std::format("# Image list with two lines of data per image (refined by LichtFeld-Studio lfe-register (LFQPLCYR) from {}/sparse/0)\n",
                                  ds.generic_string());
                for (size_t i = 0; i < N; ++i) {
                    const M4 w = rigid_inverse(fin[i]);
                    const auto q = rot_to_quat_wxyz(rot_of(w));
                    auto t = rows[i];
                    const double v[7] = {q[0], q[1], q[2], q[3], w[3], w[7], w[11]};
                    for (int k = 0; k < 7; ++k)
                        t[1 + k] = std::format("{:.12f}", v[k]);
                    std::string line;
                    for (size_t k = 0; k < t.size(); ++k)
                        line += (k ? " " : "") + t[k];
                    im << line << "\n\n";
                }
            }
            std::vector<double> mv(N), tn(N);
            std::array<double, 3> rv_mean{}, loc_mean{};
            std::ofstream pc(out / "percam.tsv");
            pc << "name\tvisits\tloss_first\tloss_last\tmove\tturn_deg\trx\try\trz\tcx\tcy\tcz\n";
            int better = 0, seen = 0;
            double lf = 0, ll = 0;
            for (size_t i = 0; i < N; ++i) {
                const M3 Ro = rot_of(C0[i]), Rn = rot_of(fin[i]);
                const auto co = pos_of(C0[i]), cn = pos_of(fin[i]);
                const std::array<double, 3> dc{cn[0] - co[0], cn[1] - co[1], cn[2] - co[2]};
                mv[i] = std::sqrt(dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]);
                tn[i] = rot_angle_deg(Ro, Rn);
                auto rv = rotvec_of(mul3(transpose3(Ro), Rn));
                for (auto& x : rv)
                    x *= 180.0 / std::numbers::pi;
                const auto loc = mv3(transpose3(Ro), dc);
                for (int k = 0; k < 3; ++k) {
                    rv_mean[k] += rv[k] / static_cast<double>(N);
                    loc_mean[k] += loc[k] / static_cast<double>(N);
                }
                pc << std::format("{}\t{}\t{:.5f}\t{:.5f}\t{:.6f}\t{:.5f}\t{:+.5f}\t{:+.5f}\t{:+.5f}\t{:+.6f}\t{:+.6f}\t{:+.6f}\n", names[i],
                                  visits[i], first_loss[i], last_loss[i], mv[i], tn[i], rv[0], rv[1], rv[2], loc[0], loc[1], loc[2]);
                if (!std::isnan(first_loss[i])) {
                    ++seen;
                    lf += first_loss[i];
                    ll += last_loss[i];
                    better += last_loss[i] < first_loss[i] ? 1 : 0;
                }
            }
            pc.close();
            std::println("per-camera corrections ({} cameras): move median {:.5f} p90 {:.5f} max {:.5f} units; turn median {:.4f} p90 {:.4f} max {:.4f} deg",
                         N, median(mv), percentile(mv, 90), *std::max_element(mv.begin(), mv.end()), median(tn), percentile(tn, 90),
                         *std::max_element(tn.begin(), tn.end()));
            const double rvn = std::sqrt(rv_mean[0] * rv_mean[0] + rv_mean[1] * rv_mean[1] + rv_mean[2] * rv_mean[2]);
            std::println("common part (mean over cameras, camera axes x right, y down, z forward): rotation ({:+.4f}, {:+.4f}, {:+.4f}) deg = {:.4f} deg (vs median turn {:.4f}); translation ({:+.5f}, {:+.5f}, {:+.5f}) units",
                         rv_mean[0], rv_mean[1], rv_mean[2], rvn, median(tn), loc_mean[0], loc_mean[1], loc_mean[2]);
            int moved = 0;
            for (size_t i = 0; i < N; ++i)
                moved += (mv[i] > 0 || tn[i] > 0) ? 1 : 0;
            std::println("refined model: {} ({} of {} cameras refined, lens unchanged)", (out / "model/sparse/0").generic_string(), moved, N);
            if (seen)
                std::println("loss per camera, first visit -> last visit: mean {:.4f} -> {:.4f}; better at the end for {:.0f}% of cameras",
                             lf / seen, ll / seen, 100.0 * better / seen);
            const uint64_t checksum_end = model_checksum(*model.splat);
            std::println("frozen model: splat parameters {} (checksum {:016x} -> {:016x})",
                         checksum_end == checksum_start ? "unchanged" : "CHANGED", checksum_start, checksum_end);
            std::fflush(stdout);
            cudaFree(d_rgb); cudaFree(d_mask); cudaFree(d_photo); cudaFree(d_grad); cudaFree(d_w2c); cudaFree(d_cam);
            cudaFree(d_pg); cudaFree(d_motion); cudaFree(d_readout); cudaFree(d_acc);
            return 0;
        }

        int guarded(int (*fn)(int, const char* const*), int argc, const char* const* argv) {
            try {
                return fn(argc, argv);
            } catch (const std::exception& e) {
                std::println(stderr, "{}: {}", argv[1], e.what());
                std::println("@@LFS fatal {}", e.what());
                std::fflush(stdout);
                return 1;
            }
        }
    } // namespace

    int run_lfe_render_tool(int argc, const char* const* argv) { return guarded(render_main, argc, argv); }
    int run_lfe_register_tool(int argc, const char* const* argv) { return guarded(register_main, argc, argv); }

} // namespace lfs::app
