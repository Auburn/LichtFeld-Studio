/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --lens-radial FILE (LFRL fork, 6 Nov 2026): the process-wide radial lens table. Once activated, every native-fisheye
// camera that is the table's own camera (same k1..k4, fx/fy/cx/cy relative to the image size) is projected through the
// table (fastgs_lens_for_camera, MRNF blob seeding) instead of its k-polynomial; any other fisheye camera is an error,
// so a table can never silently pair with the wrong lens. Not activated: nothing changes. See core/radial_lens.h for the
// lens and abq/lens_radial.py (Videos/splat/work) for the file format.

#include "core/radial_lens.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {
    class Camera;
}

namespace lfs::training {

    struct RadialLensTable {
        std::filesystem::path path;
        int width = 0, height = 0;           // the paired camera (full resolution)
        double fx = 0, fy = 0, cx = 0, cy = 0;
        double k[4] = {0, 0, 0, 0};
        int n = 0;                           // nodes
        double dth = 0, th_end = 0, valid = 0, A = 0, B = 0, centre = 0;
        std::vector<float> seg;              // [(n - 1) * 8] u0, inv_h, c0..c5 (float32, as every implementation uses them)
        std::vector<double> segd;            // the same in double, as written in the file (double-precision references only)
        float inv_dth_f = 0, th_end_f = 0, A_f = 0, B_f = 0;
        float* d_seg = nullptr;              // device copy (activate_radial_lens)

        [[nodiscard]] core::RadialLensView host_view() const;
        [[nodiscard]] core::RadialLensView device_view() const;
        [[nodiscard]] std::string describe() const;
    };

    // Parses and checks FILE (the 'LENS_RADIAL_TABLE 1' text format); throws std::runtime_error with the reason.
    [[nodiscard]] std::shared_ptr<RadialLensTable> load_radial_lens_table(const std::filesystem::path& file);

    // Makes the table the process's radial lens and uploads it to the current CUDA device (throws on failure).
    void activate_radial_lens(std::shared_ptr<RadialLensTable> table);

    // The active table, or nullptr (the default: no radial lens anywhere).
    [[nodiscard]] const RadialLensTable* active_radial_lens() noexcept;

    // Empty if the camera is the table's camera (FISHEYE with the table's k's and intrinsics), else the reason.
    [[nodiscard]] std::string radial_lens_camera_mismatch(const RadialLensTable& table, const core::Camera& camera);

} // namespace lfs::training
