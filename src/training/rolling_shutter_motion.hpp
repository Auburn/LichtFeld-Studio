/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --rolling-shutter FILE: the dynamic rolling shutter's per-camera motion (and the shared readout) for training through
// the native-fisheye standard rasterizer (fast_lfs::rasterization::FastGSRollingShutter). FILE: one line per image,
//   name wx wy wz ax ay az vx vy vz Ax Ay Az
// (angular velocity, angular acceleration, linear velocity, linear acceleration in the image's mid-readout camera axes;
// rad/s, rad/s^2, scene units/s, units/s^2; '#' comments; abq/rs_motion.py makes it from the camera path). Images
// without a line train with a global shutter. The readout (top row to bottom row) is --rs-readout-ms.
// --rs-learn omega,vel,alpha,acc,readout (any of them): those are refined during training from --rs-learn-start on,
// by a per-camera Adam step after each backward (kernels/rolling_shutter_update.hpp), as corrections to the file's
// motion in "pixels at the readout ends", with --rs-prior * |correction|^2 and --rs-smooth * |correction - its time
// neighbour's|^2; the readout by its own Adam (--rs-readout-lr ms per step). The learned motion and readout are written
// to <output>/rs_motion_learned.txt at the end (same format; the readout in a '# readout_ms' line). --rs-motion-start
// stages it: the readout first (motion fixed), then the motion (readout fixed).

#include "core/camera.hpp"
#include <cuda_runtime.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fast_lfs::rasterization {
    struct FastGSRollingShutter; // rasterization_config.h
}

namespace lfs::training {

    struct RollingShutterSettings {
        std::string path;
        float readout_ms = 10.0f;
        std::string learn;        // comma list of omega, alpha, vel, acc, readout ("" = nothing learned)
        float lr = 0.05f;         // q units (pixels at the readout ends) per camera update
        float readout_lr_ms = 0.002f;
        float prior = 1e-4f;
        float smooth = 0.0f;
        int learn_start = 3000;
        // Staged learning (avoids the readout / velocity trade-off: a longer readout with slower motion draws the same
        // frame): with both the readout and motion components learned and motion_start > learn_start, the readout is
        // learned from learn_start until motion_start (motion fixed), then the motion from motion_start on (readout
        // fixed). 0 = both from learn_start.
        int motion_start = 0;
        int iters = 2;
    };

    class RollingShutterMotion {
    public:
        static std::expected<std::unique_ptr<RollingShutterMotion>, std::string> create(
            const RollingShutterSettings& settings, const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);
        ~RollingShutterMotion();
        RollingShutterMotion(const RollingShutterMotion&) = delete;
        RollingShutterMotion& operator=(const RollingShutterMotion&) = delete;

        // The rasterizer options for this camera at this iteration (motion = nullptr: global shutter). When learning is
        // active it binds and zeros the gradient buffer on `stream`.
        fast_lfs::rasterization::FastGSRollingShutter options_for(const lfs::core::Camera& camera, int iter, cudaStream_t stream);
        // After the backward of the same camera: the Adam step (device; no-op unless learning is active).
        void step(const lfs::core::Camera& camera, int iter, cudaStream_t stream);
        // A one-line summary (copies the state to the host: call rarely).
        std::string summary(cudaStream_t stream);
        // Writes the current motion and readout (same format as the input).
        bool save(const std::filesystem::path& file, cudaStream_t stream, std::string& error);

        [[nodiscard]] size_t n_cameras() const { return names_.size(); }
        [[nodiscard]] bool learning() const { return learn_mask_ != 0u || learn_readout_; }
        // which parts learn at this iteration
        [[nodiscard]] bool motion_learning(int iter) const;
        [[nodiscard]] bool readout_learning(int iter) const;
        [[nodiscard]] const std::string& description() const { return description_; }

    private:
        RollingShutterMotion() = default;
        [[nodiscard]] int row_for(const lfs::core::Camera& camera) const;

        RollingShutterSettings settings_;
        std::unordered_map<int, int> row_of_uid_;
        std::vector<std::string> names_;
        std::vector<float> init_host_;
        unsigned learn_mask_ = 0u;
        bool learn_readout_ = false;
        float scale_[12] = {};
        std::string description_;
        // device
        float* motion_ = nullptr;   // [N,12]
        float* init_ = nullptr;     // [N,12]
        float* adam_m_ = nullptr;   // [N,12]
        float* adam_v_ = nullptr;   // [N,12]
        float* steps_ = nullptr;    // [N]
        float* readout_ = nullptr;  // [1] seconds
        float* readout_state_ = nullptr; // [3]
        float* grad_ = nullptr;     // [13]
        int* nb_ = nullptr;         // [N,2]
    };

} // namespace lfs::training
