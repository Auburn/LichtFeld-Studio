/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --rs-time-map FILE (LFQPLCYKMR, 11 Nov 2026): a per-pixel rolling-shutter readout time in place of the row rule. The
// camera remaps its sensor image radially after capture, so the lines of equal readout time are curved (measured with a
// 1 kHz LED: D:/splat_work/rs_led/build_rs_map.py). File (text; '#' comments anywhere):
//   LFS_RS_TIME_MAP 1
//   size W H                      the image the form is defined on, output pixels (e.g. 4096 3072); any camera with this
//                                 aspect uses it (scaled)
//   radial K A B C CX CY          the radial form (ms): x = (u - W/2) / (W/2) - CX, y = (v - H/2) / (W/2) - CY (both in half
//                                 image WIDTHS), r^2 = x^2 + y^2, t = K y (1 + A r^2 + B r^4 + C r^6)
//   or
//   grid NX NY + NY rows of NX values (ms): nodes at u = i W / (NX - 1), v = j H / (NY - 1), bilinear between them
//   readout_ms R                  optional: the form's linear-equivalent readout; default = computed (below)
// (u, v) are pixel-corner coordinates (pixel (c, r) centre = (c + 0.5, r + 0.5)). The rasterizer reads a point imaged at
// (u, v) at t = s tau, s = (t(u, v) - t(W/2, H/2)) / R, tau = --rs-readout-ms (or a request's readout_ms): R = the
// least-squares slope of t against the row over the whole image (t ~ a + b v, R = b H), so tau keeps its meaning (a linear
// form, e.g. radial K 0 0 0 0 0, gives s = v / H - 1/2: the row rule at any tau; with tau = R the form's own times), and
// t = 0 at the image centre as with the row rule. Header-only (the trainer and the lfe-render / lfe-register tools).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace lfs::training {

    struct RsTimeMap {
        std::string path;
        int width = 0, height = 0; // the image the form is defined on
        bool radial = false;
        double K = 0, A = 0, B = 0, C = 0, CX = 0, CY = 0; // radial form
        int nx = 0, ny = 0;                                // grid form
        std::vector<float> t_ms;                           // [ny * nx] grid values as read (ms)
        double readout_ms = 0.0;                           // R (linear-equivalent readout)
        bool readout_from_file = false;
        double t_centre = 0.0;                             // t(W/2, H/2) as read
        std::vector<float> s;                              // grid form: [ny * nx] (t - t_centre) / R (what the kernels read)

        // t (ms, as the file defines it) at the normalised position (fu, fv), clamped to [0, 1]^2 as the kernels do
        [[nodiscard]] double t_raw(double fu, double fv) const {
            fu = std::fmin(std::fmax(fu, 0.0), 1.0);
            fv = std::fmin(std::fmax(fv, 0.0), 1.0);
            if (radial) {
                const double x = 2.0 * fu - 1.0 - CX, y = (2.0 * fv - 1.0) * height / width - CY, r2 = x * x + y * y;
                return K * y * (1.0 + r2 * (A + r2 * (B + r2 * C)));
            }
            fu *= (nx - 1);
            fv *= (ny - 1);
            const int i0 = std::min(static_cast<int>(fu), nx - 2), j0 = std::min(static_cast<int>(fv), ny - 2);
            const double ax = fu - i0, ay = fv - j0;
            const auto T = [&](int i, int j) { return static_cast<double>(t_ms[static_cast<size_t>(j) * nx + i]); };
            const double a = T(i0, j0) + ax * (T(i0 + 1, j0) - T(i0, j0));
            const double b = T(i0, j0 + 1) + ax * (T(i0 + 1, j0 + 1) - T(i0, j0 + 1));
            return a + ay * (b - a);
        }
        // the time the rasterizer uses at readout tau_ms: (t - t_centre) / R x tau
        [[nodiscard]] double t_used(double fu, double fv, double tau_ms) const { return (t_raw(fu, fv) - t_centre) / readout_ms * tau_ms; }

        // fills a FastGSRollingShutter's time fields (d_grid = the device copy of s for the grid form; unused for radial)
        template <class RS>
        void fill(RS& rs, const float* d_grid) const {
            if (radial) {
                rs.time_radial = 1;
                rs.tr_k = static_cast<float>(K / readout_ms);
                rs.tr_a = static_cast<float>(A);
                rs.tr_b = static_cast<float>(B);
                rs.tr_c = static_cast<float>(C);
                rs.tr_cx = static_cast<float>(CX);
                rs.tr_cy = static_cast<float>(CY);
                rs.tr_aspect = static_cast<float>(static_cast<double>(height) / width);
                rs.tr_s0 = static_cast<float>(t_centre / readout_ms);
            } else {
                rs.time_map = d_grid;
                rs.time_map_nx = nx;
                rs.time_map_ny = ny;
            }
        }

        // one line for the logs
        [[nodiscard]] std::string describe(const double tau_ms) const {
            double lo = 1e30, hi = -1e30;
            for (int j = 0; j <= 96; ++j)
                for (int i = 0; i <= 128; ++i) {
                    const double t = t_used(i / 128.0, j / 96.0, tau_ms);
                    lo = std::min(lo, t);
                    hi = std::max(hi, t);
                }
            const double top = t_used(0.5, 0.0, tau_ms), bot = t_used(0.5, 1.0, tau_ms);
            const double tl = t_used(0.0, 0.0, tau_ms) - top, tr = t_used(1.0, 0.0, tau_ms) - top;
            const double bl = t_used(0.0, 1.0, tau_ms) - bot, br = t_used(1.0, 1.0, tau_ms) - bot;
            const std::string form = radial ? std::format("radial form K {:g} A {:g} B {:g} C {:g} CX {:g} CY {:g} on {} x {}", K, A, B, C, CX, CY, width, height)
                                            : std::format("grid {} x {} nodes over {} x {}", nx, ny, width, height);
            return std::format(
                "'{}': {}; linear-equivalent readout R {:.4f} ms ({}); s = (t - t(centre) {:+.4f}) / R replaces the row fraction (v / H - 1/2), "
                "t = s x the readout {:g} ms (the form x {:.5f}): t {:+.3f} .. {:+.3f} ms, centre column top to bottom {:.3f} ms, row ends vs the "
                "row's centre: top {:+.3f} / {:+.3f}, bottom {:+.3f} / {:+.3f} ms",
                path, form, readout_ms, readout_from_file ? "from the file" : "least squares over the image", t_centre, tau_ms,
                tau_ms / readout_ms, lo, hi, bot - top, tl, tr, bl, br);
        }

        // the render / camera size must have the form's aspect (any resolution); "" = fine
        [[nodiscard]] std::string aspect_error(const int w, const int h) const {
            if (w <= 0 || h <= 0)
                return std::format("--rs-time-map: image size {} x {}", w, h);
            const double r = (static_cast<double>(w) * height) / (static_cast<double>(h) * width);
            if (std::fabs(r - 1.0) > 2e-3)
                return std::format("--rs-time-map: a {} x {} image does not have the map's aspect ({} x {})", w, h, width, height);
            return {};
        }
    };

    inline std::expected<RsTimeMap, std::string> load_rs_time_map(const std::string& path) {
        RsTimeMap m;
        m.path = path;
        std::ifstream in{std::filesystem::path(std::u8string(path.begin(), path.end()))};
        if (!in)
            return std::unexpected(std::format("--rs-time-map: cannot open '{}'", path));
        std::string line;
        bool magic = false, have_grid = false;
        std::vector<float> vals;
        int line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            if (const auto c = line.find('#'); c != std::string::npos)
                line.resize(c);
            std::istringstream ls(line);
            std::string key;
            if (!(ls >> key))
                continue;
            if (!magic) {
                int ver = 0;
                if (key != "LFS_RS_TIME_MAP" || !(ls >> ver) || ver != 1)
                    return std::unexpected(std::format("--rs-time-map: '{}' line {}: 'LFS_RS_TIME_MAP 1' expected", path, line_no));
                magic = true;
            } else if (key == "size") {
                if (!(ls >> m.width >> m.height) || m.width <= 0 || m.height <= 0)
                    return std::unexpected(std::format("--rs-time-map: '{}' line {}: size W H", path, line_no));
            } else if (key == "readout_ms") {
                if (!(ls >> m.readout_ms) || !std::isfinite(m.readout_ms) || m.readout_ms <= 0.0 || m.readout_ms > 100.0)
                    return std::unexpected(std::format("--rs-time-map: '{}' line {}: readout_ms R (0 < R <= 100)", path, line_no));
                m.readout_from_file = true;
            } else if (key == "radial") {
                if (!(ls >> m.K >> m.A >> m.B >> m.C >> m.CX >> m.CY))
                    return std::unexpected(std::format("--rs-time-map: '{}' line {}: radial K A B C CX CY", path, line_no));
                for (double v : {m.K, m.A, m.B, m.C, m.CX, m.CY})
                    if (!std::isfinite(v))
                        return std::unexpected(std::format("--rs-time-map: '{}' line {}: non-finite radial value", path, line_no));
                m.radial = true;
            } else if (key == "grid") {
                if (!(ls >> m.nx >> m.ny) || m.nx < 2 || m.ny < 2 || m.nx > 4097 || m.ny > 4097)
                    return std::unexpected(std::format("--rs-time-map: '{}' line {}: grid NX NY (2 .. 4097)", path, line_no));
                have_grid = true;
            } else {
                // a row of grid values
                std::istringstream vs(line);
                std::string tok;
                while (vs >> tok) {
                    char* end = nullptr;
                    const float v = std::strtof(tok.c_str(), &end);
                    if (end == tok.c_str() || *end != '\0' || !std::isfinite(v))
                        return std::unexpected(std::format("--rs-time-map: '{}' line {}: '{}' is not a number", path, line_no, tok));
                    vals.push_back(v);
                }
            }
        }
        if (!magic)
            return std::unexpected(std::format("--rs-time-map: '{}' is empty", path));
        if (m.width <= 0)
            return std::unexpected(std::format("--rs-time-map: '{}' needs a size line", path));
        if (m.radial == have_grid)
            return std::unexpected(std::format("--rs-time-map: '{}' needs exactly one of a radial line or a grid", path));
        if (m.radial && !vals.empty())
            return std::unexpected(std::format("--rs-time-map: '{}': values after a radial form", path));
        if (have_grid) {
            if (vals.size() != static_cast<size_t>(m.nx) * static_cast<size_t>(m.ny))
                return std::unexpected(std::format("--rs-time-map: '{}' has {} values, grid {} x {} needs {}", path, vals.size(), m.nx, m.ny,
                                                   static_cast<size_t>(m.nx) * m.ny));
            m.t_ms = std::move(vals);
        }
        m.t_centre = m.t_raw(0.5, 0.5);
        if (!m.readout_from_file) {
            // least squares t = a + b (fv - 1/2) over a uniform 257 x 193 sample of the image: R = b
            double sy = 0, syy = 0, st = 0, syt = 0, n = 0;
            for (int j = 0; j <= 192; ++j)
                for (int i = 0; i <= 256; ++i) {
                    const double fv = j / 192.0 - 0.5, t = m.t_raw(i / 256.0, j / 192.0);
                    sy += fv;
                    syy += fv * fv;
                    st += t;
                    syt += fv * t;
                    n += 1;
                }
            m.readout_ms = (syt - sy * st / n) / (syy - sy * sy / n);
            if (!std::isfinite(m.readout_ms) || m.readout_ms <= 0.0 || m.readout_ms > 100.0)
                return std::unexpected(std::format("--rs-time-map: '{}': linear-equivalent readout {} ms (not in (0, 100])", path, m.readout_ms));
        }
        if (have_grid) {
            m.s.resize(m.t_ms.size());
            for (size_t k = 0; k < m.t_ms.size(); ++k)
                m.s[k] = static_cast<float>((static_cast<double>(m.t_ms[k]) - m.t_centre) / m.readout_ms);
        }
        // sanity: top row first down the middle, spanning about one readout (catches a unit or sign mistake)
        const double top = m.t_raw(0.5, 0.0), bot = m.t_raw(0.5, 1.0);
        if (!(bot - top > 0.5 * m.readout_ms && bot - top < 2.0 * m.readout_ms))
            return std::unexpected(std::format("--rs-time-map: '{}': the centre column runs {:.3f} ms top to bottom, not about the readout {:.3f} ms "
                                               "(top row first expected)",
                                               path, bot - top, m.readout_ms));
        return m;
    }

} // namespace lfs::training
