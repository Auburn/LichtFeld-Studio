/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --pose-opt: per-camera pose refinement during training (native-fisheye standard rasterizer; with or without
// --rolling-shutter; not --gut). Every training camera gets an offset (r, d) from its start pose (rotation r in radians
// and the camera centre's move d in scene units, both in the start camera's axes; kernels/pose_refinement_math.hpp):
//   W = exp(-[r]x) W0,  b = exp(-[r]x) (b0 - d),  centre c = c0 + W0^T d.
// Each step the rendered camera's refined transform is composed on the device and handed to the rasterizer in place of
// the camera's own (FastGSPose), so culling, projection, covariance, depth, the distortion loss's depth, normals, colours'
// view direction and the dynamic rolling shutter (which acts in the refined camera's axes) all use it. The backward
// returns dL/d(left perturbation) of that transform from every splat (photometric loss, depth distortion, depth and normal
// terms - whatever reaches the splats' camera-space rows), chained exactly to the offset, and one Adam step updates only
// that camera's offset (device; no host sync), like --rs-learn's motion.
// Learning rates: --pose-lr-deg (degrees) and --pose-lr-units (scene units) per camera update, i.e. the size of one Adam
// step; schedule: 0 before --pose-opt-start, held to --pose-opt-hold, then halving smoothly every --pose-opt-half-life
// steps (0 = auto: 1/128 of the rate by the last step). Optional --pose-prior w |q|^2 (to the start) and --pose-smooth
// w |q - q_n|^2 (to the time neighbours: adjacent training images at most 12 video frames apart, compared in the world
// frame), with q = focal_px * (r, d) (pixels at the image centre / one scene unit away).
// Output (end of training): <output>/cameras_refined/ (COLMAP text: the dataset's cameras.txt, images.txt with the refined
// poses in the frame of the exported PLY, points3D.txt copied) and <output>/pose_refinement.tsv (per camera: move, turn).

#include "core/camera.hpp"
#include <cuda_runtime.h>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fast_lfs::rasterization {
    struct FastGSPose; // rasterization_config.h
}

namespace lfs::training {

    struct PoseRefinementSettings {
        float lr_deg = 0.01f;    // rotation: degrees per camera update (one Adam step)
        float lr_units = 5e-4f;  // translation: scene units per camera update
        int start = 6000;        // iterations (after --steps-scaler)
        int hold_until = 30000;  // full rate until here, then the smooth halving
        float half_life = 0.0f;  // iterations; 0 = auto ((iterations - hold_until) / 7: 1/128 at the end)
        int iterations = 60000;  // the run's last iteration
        float prior = 0.0f;
        float smooth = 0.0f;
        int log_every = 2000;
        // developer option (env LFS_POSE_SNAPSHOTS): every summary also writes the offsets to <snapshot_dir>/iter_NNNNNN.tsv
        std::filesystem::path snapshot_dir;
    };

    class CameraPoseRefinement {
    public:
        static std::expected<std::unique_ptr<CameraPoseRefinement>, std::string> create(
            const PoseRefinementSettings& settings, const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);
        ~CameraPoseRefinement();
        CameraPoseRefinement(const CameraPoseRefinement&) = delete;
        CameraPoseRefinement& operator=(const CameraPoseRefinement&) = delete;

        // The rasterizer's pose for this camera at this iteration: before the start, nothing (the camera's own pose, the
        // unchanged path); from the start on, the refined transform composed on `stream`, and while the rate is above
        // zero the gradient buffer (zeroed on `stream`).
        fast_lfs::rasterization::FastGSPose options_for(const lfs::core::Camera& camera, int iter, cudaStream_t stream);
        // After the backward of the same camera: its Adam step (device).
        void step(const lfs::core::Camera& camera, int iter, cudaStream_t stream);
        [[nodiscard]] float lr_factor(int iter) const;
        [[nodiscard]] bool active(int iter) const { return iter >= settings_.start; }
        // One log line: the rate, cameras updated, per-camera move / turn median, p90, max, the mean move (copies the
        // state to the host: call rarely).
        std::string summary(int iter, cudaStream_t stream);
        // Writes <out_dir>/cameras_refined/ and <out_dir>/pose_refinement.tsv; `report` gets the log lines.
        bool export_results(const std::filesystem::path& out_dir, const std::filesystem::path& dataset_dir, cudaStream_t stream,
                            std::vector<std::string>& report, std::string& error);

        [[nodiscard]] const std::string& description() const { return description_; }
        [[nodiscard]] size_t n_cameras() const { return names_.size(); }
        [[nodiscard]] const PoseRefinementSettings& settings() const { return settings_; }

    private:
        CameraPoseRefinement() = default;
        [[nodiscard]] int row_for(const lfs::core::Camera& camera) const;
        void download(std::vector<float>& pose, std::vector<float>& steps, cudaStream_t stream) const;

        PoseRefinementSettings settings_;
        float half_life_ = 0.0f;
        float focal_px_ = 1.0f;
        std::unordered_map<int, int> row_of_uid_;
        std::vector<std::string> names_;
        std::vector<int> frames_;
        std::vector<int> camera_ids_;
        std::vector<double> w2c0_host_; // [N, 16] start transforms as the trainer had them (float values, widened)
        std::string description_;
        // device
        float* pose_ = nullptr;   // [N, 6] r (rad), d (units)
        float* w2c0_ = nullptr;   // [N, 16]
        float* c0_ = nullptr;     // [N, 3]
        float* adam_m_ = nullptr; // [N, 6]
        float* adam_v_ = nullptr; // [N, 6]
        float* steps_ = nullptr;  // [N]
        float* grad_ = nullptr;   // [6]
        float* w2c_out_ = nullptr; // [16]
        float* cam_out_ = nullptr; // [3]
        int* nb_ = nullptr;       // [N, 2]
    };

} // namespace lfs::training
