/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// --bg-mode cubemap (LFC): a learned environment cubemap as the training background.
// Each pixel's background is the cubemap in the world direction of the pixel's ray (the camera's own lens, the
// rolling-shutter pose of its row); the rasterizer composites image = splats + (1 - alpha) * background, and every
// appearance stage (exposure correction / PPISP, bilateral grid) and the loss see that composited image, exactly as
// with a solid clear colour. The texels get dL/d(texel) = sum (1 - alpha) dL/d(image) (bilinear) and their own Adam
// step, plus an optional smoothness (squared-difference total variation) term. Exported at the end of training to
// <output>/bg_cubemap/ (faces px nx py ny pz nz as 8-bit PNG in the photos' encoding, faces.npy float32 [6, R, R, 3],
// convention.txt). See kernels/bg_cubemap.hpp for the face layout.

#include "core/camera.hpp"
#include "kernels/bg_cubemap.hpp"

#include <array>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fast_lfs::rasterization {
    struct FastGSRollingShutter; // rasterization_config.h (private to lfs_training, as in rolling_shutter_motion.hpp)
}

namespace lfs::training {

    struct BackgroundCubemapSettings {
        int res = 256;                       // texels per face edge
        std::array<float, 3> init{};         // initial colour of every texel ([0, 1], the --bg-color)
        float lr = 2e-3f;                    // Adam learning rate at iteration 0
        float lr_final_factor = 0.1f;        // exponential decay to lr * factor at the last iteration
        float smooth = 100.0f;               // smoothness weight (0 = off)
        std::string init_path;               // LFC2: start from this faces.npy (float32 [6, R0, R0, 3]; resampled to res)
    };

    class BackgroundCubemap {
    public:
        static std::unique_ptr<BackgroundCubemap> create(const BackgroundCubemapSettings& settings, std::string& error);
        ~BackgroundCubemap();
        BackgroundCubemap(const BackgroundCubemap&) = delete;
        BackgroundCubemap& operator=(const BackgroundCubemap&) = delete;

        // This camera's background [3, H, W] at its render size (H, W = the camera's image size), through its lens and
        // the rolling-shutter motion of this step (rs.motion == nullptr: global shutter). Remembers the view for backward().
        // LFQPLC: the lens is the rasterizer's own (fastgs_lens_for_camera: the radial lens table under --lens-radial);
        // w2c_override = the refined world-to-camera transform the rasterizer draws with under --pose-opt (device [16],
        // CameraPoseRefinement::options_for on the same stream), null = the camera's own. The cubemap does NOT add a
        // pose gradient: the camera poses are refined from the splats' gradient only (the background is held for it).
        const lfs::core::Tensor& render(const lfs::core::Camera& camera,
                                        const fast_lfs::rasterization::FastGSRollingShutter& rs, cudaStream_t stream,
                                        const float* w2c_override = nullptr);
        // Adds (1 - alpha) * grad_image, scattered to the texels the last render() sampled. grad_image [3, H, W] =
        // dL/d(composited render), alpha [1, H, W] or [H, W] = the render's alpha.
        void backward(const lfs::core::Tensor& grad_image, const lfs::core::Tensor& alpha, cudaStream_t stream);
        // Smoothness gradient, Adam step (lr decayed by iteration), clamp to [0, 1], zero the gradient.
        // log = also measure the gradient (photometric + smoothness) for summary(); synchronous only when log is set.
        void step(int iter, int total_iters, bool log, cudaStream_t stream);
        // A one-line state summary (synchronous readback).
        std::string summary(cudaStream_t stream);
        // Writes <dir>/{px,nx,py,ny,pz,nz}.png, faces.npy and convention.txt.
        bool save(const std::filesystem::path& dir, int iter, const std::string& note, std::string& error, cudaStream_t stream);
        // Test hook (LFS_CUBEMAP_DUMP=<iteration>): the last render()'s background [3, H, W] float32 as .npy, plus the
        // faces it was sampled from (<path>.faces.npy).
        bool dump_last(const std::filesystem::path& path, std::string& error, cudaStream_t stream);
        // Test hook (LFQPLC): any float tensor as .npy (its own shape), e.g. the composite / alpha / post-PPISP image.
        static bool write_tensor_npy(const std::filesystem::path& path, const lfs::core::Tensor& t, std::string& error,
                                     cudaStream_t stream);

        [[nodiscard]] int res() const { return settings_.res; }
        [[nodiscard]] std::string description() const;
        [[nodiscard]] const lfs::core::Tensor& last_background() const { return bg_; }

    private:
        explicit BackgroundCubemap(const BackgroundCubemapSettings& settings) : settings_(settings) {}

        BackgroundCubemapSettings settings_;
        std::string init_note_;      // how the texels were initialised (description())
        std::vector<float> init_host_; // the initial texels (summary: how far each moved)
        size_t n_ = 0;               // floats: 6 R^2 3
        float* texels_ = nullptr;    // device
        float* grad_ = nullptr;
        float* adam_m_ = nullptr;
        float* adam_v_ = nullptr;
        float* stats_ = nullptr;     // device [8]: grad sum|g|, max|g|, non-finite (stats), smoothness loss, non-finite (Adam)
        long long adam_steps_ = 0;
        kernels::CubemapView view_{};
        bool have_view_ = false;
        lfs::core::Tensor bg_;
        // last logged measurements (host)
        float last_grad_sum_ = 0.0f, last_grad_max_ = 0.0f, last_smooth_loss_ = 0.0f, last_lr_ = 0.0f;
        float last_nonfinite_stats_ = 0.0f;
        int backward_calls_ = 0;
        int last_log_iter_ = 0;
        bool radial_used_ = false;       // LFQPLC: some render() drew through the radial lens table (summary)
        bool refined_pose_used_ = false; // LFQPLC: some render() used a refined pose (--pose-opt) (summary)
    };

} // namespace lfs::training
