/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define LFS_BLOB_HD __host__ __device__
#else
#define LFS_BLOB_HD
#endif

namespace lfs::training::kernels::blob_seeding {

    enum PeakField : int { PeakX = 0,
                           PeakY,
                           PeakView,
                           PeakPolarity,
                           PeakR,
                           PeakG,
                           PeakB,
                           PeakFieldCount };

    // SweepAccepted is 1 when enough covisible views support SweepDepth, else 0.
    enum SweepField : int { SweepDepth = 0,
                            SweepAccepted,
                            SweepFieldCount };

    struct ViewPeaks {
        lfs::core::Tensor peaks;  // [count, PeakFieldCount] float32
        lfs::core::Tensor bitmap; // ceil(H*W/16) uint32 words, 2 bits per pixel, dilated by one pixel
        std::array<float, 2> density{};
    };

    struct SweepView {
        float R[9]; // world to camera, row major
        float t[3];
        float fx, fy, cx, cy;
        int width, height;
        const uint32_t* bitmap;
        float density[2];
        float z_min, z_max;
        // OPENCV_FISHEYE lens of a natively trained fisheye camera (--native-fisheye); 0 = pinhole
        int fisheye = 0;
        float k[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    // A view's pixel ray as (dx, dy, 1) in camera space from (m_x, m_y) = ((u - cx) / fx, (v - cy) / fy);
    // false where a fisheye ray is not in front of the camera (theta past ~89 degrees).
    LFS_BLOB_HD inline bool sweep_unproject(const SweepView& v, const float mx, const float my, float& dx, float& dy) {
        if (!v.fisheye) {
            dx = mx;
            dy = my;
            return true;
        }
        const float td = sqrtf(mx * mx + my * my);
        if (td < 1e-7f) {
            dx = mx;
            dy = my;
            return true;
        }
        // theta_d(theta) is increasing on the bracket the lens is used in: bisect it
        float lo = 0.0f, hi = 1.56f;
        for (int i = 0; i < 40; ++i) {
            const float th = 0.5f * (lo + hi);
            const float t2 = th * th;
            const float d = th * (1.0f + t2 * (v.k[0] + t2 * (v.k[1] + t2 * (v.k[2] + t2 * v.k[3]))));
            (d < td ? lo : hi) = th;
        }
        const float th = 0.5f * (lo + hi);
        if (!(th < 1.555f))
            return false;
        const float s = tanf(th) / td;
        dx = mx * s;
        dy = my * s;
        return true;
    }

    // A camera-space point (Z > 0) to the view's pixel coordinates (pixel centres at +0.5).
    LFS_BLOB_HD inline void sweep_project(const SweepView& v, const float X, const float Y, const float Z, float& u, float& w) {
        if (!v.fisheye) {
            u = v.fx * X / Z + v.cx;
            w = v.fy * Y / Z + v.cy;
            return;
        }
        const float rho = sqrtf(X * X + Y * Y);
        const float th = atan2f(rho, Z);
        const float t2 = th * th;
        const float td = th * (1.0f + t2 * (v.k[0] + t2 * (v.k[1] + t2 * (v.k[2] + t2 * v.k[3]))));
        const float s = rho > 1e-7f * Z ? td / rho : 1.0f / Z;
        u = v.fx * s * X + v.cx;
        w = v.fy * s * Y + v.cy;
    }

    // Detects compact bright and dark blobs (local contrast maxima over a 5x5 window, contrast =
    // 0-255 luminance against its 7x7 grey opening or closing) in a CUDA float32 [C, H, W] image
    // with C >= 3 and values in [0, 1].
    // Scratch buffers reused across detect_peaks calls; growing them per call fragments the memory pool
    // because every call also allocates the peaks and bitmap it returns.
    struct DetectionWorkspace {
        lfs::core::Tensor rgb, rows, a, b, c, mask, peaks, counters, bitmap;
    };

    // The returned peaks and bitmap are views into the workspace, valid until its next use.
    LFS_CUDA_API ViewPeaks detect_peaks(const lfs::core::Tensor& image, int view, DetectionWorkspace& workspace);

    // Averages factor x factor pixel blocks of a CUDA [C, H, W] uint8 or float32 image (C >= 3) into a
    // float32 [3, H / factor, W / factor] image in [0, 1], stored in workspace.rgb.
    LFS_CUDA_API lfs::core::Tensor downsample_rgb(const lfs::core::Tensor& image, int factor,
                                                  DetectionWorkspace& workspace);

    // For every peak, sweeps depth along its pixel ray and counts the neighbour views whose
    // peak bitmap of the same polarity is hit, minus the hits expected by chance. views and
    // neighbors ([V, K] int32, -1 padded) are device arrays. Returns [P, SweepFieldCount].
    LFS_CUDA_API lfs::core::Tensor sweep_peaks(const lfs::core::Tensor& peaks,
                                               const SweepView* views,
                                               const lfs::core::Tensor& neighbors);

} // namespace lfs::training::kernels::blob_seeding
