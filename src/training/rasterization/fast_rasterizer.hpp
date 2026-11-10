/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/render_output.hpp"
#include <expected>
#include <rasterization_api.h>
#include <string>
#include <utility>

namespace lfs::training {
    // Depth distortion loss (2DGS / Mip-NeRF 360) for fast_rasterize_forward: weight * mean over the
    // image pixels of sum_{i>j} w_i w_j (m_i - m_j)^2, m = far/(far - near) (1 - near/d). weight 0 = off
    // (the unchanged kernels run). See fast_lfs::rasterization::FastGSDistortion.
    //   depth_distance: d = distance from the camera centre (default); false = the sort depth (camera z on the
    //     pinhole path: the pre-2026-11-01 behaviour, which rewards opaque splats beside the camera).
    //   plane_sigma > 0: splats whose plane_sigma-sigma extent along the camera axis crosses the camera plane
    //     are left out of the term for that view (0 = none).
    //   min_dist > 0: splats whose distortion depth d is below min_dist are left out too (0 = none).
    //   count_left_out: also count the visible splats left out (FastRasterizeContext::distortion_left_out:
    //     [0] plane-crossing, [1] nearer than min_dist).
    struct FastGSDistortionOptions {
        float weight = 0.0f;
        float near_n = 0.2f;
        float far_n = 1000.0f;
        bool depth_distance = true;
        float plane_sigma = 2.0f;
        float min_dist = 0.0f;
        bool count_left_out = false;
    };

    // Forward pass context - holds intermediate buffers needed for backward
    struct FastRasterizeContext {
        FastRasterizeContext() = default;
        ~FastRasterizeContext() {
            release_forward_context();
        }

        FastRasterizeContext(const FastRasterizeContext&) = delete;
        FastRasterizeContext& operator=(const FastRasterizeContext&) = delete;

        FastRasterizeContext(FastRasterizeContext&& other) noexcept {
            move_from(std::move(other));
        }

        FastRasterizeContext& operator=(FastRasterizeContext&& other) noexcept {
            if (this != &other) {
                release_forward_context();
                move_from(std::move(other));
            }
            return *this;
        }

        lfs::core::Tensor image;
        lfs::core::Tensor alpha;
        lfs::core::Tensor depth;
        lfs::core::Tensor normal;   // [3, H, W] camera-space accumulated normals, empty unless rendered
        lfs::core::Tensor bg_color; // Saved for alpha gradient computation

        // Gaussian parameters (saved to avoid re-fetching in backward)
        lfs::core::Tensor means;
        lfs::core::Tensor raw_scales;
        lfs::core::Tensor raw_rotations;
        lfs::core::Tensor raw_opacities;
        lfs::core::Tensor shN;

        const float* w2c_ptr = nullptr;
        const float* cam_position_ptr = nullptr;

        // Forward context (contains buffer pointers, frame_id, etc.)
        fast_lfs::rasterization::ForwardContext forward_ctx = {};

        int active_sh_bases = 0;
        int width = 0;
        int height = 0;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        float near_plane = 0.0f;
        float far_plane = 0.0f;
        bool mip_filter = false;

        // Tile information (for tile-based training)
        int tile_x_offset = 0; // Horizontal offset of this tile
        int tile_y_offset = 0; // Vertical offset of this tile
        int tile_width = 0;    // Width of this tile (0 = full image)
        int tile_height = 0;   // Height of this tile (0 = full image)

        // Background image for per-pixel blending (optional, empty = use bg_color)
        lfs::core::Tensor bg_image;

        // Depth distortion (only when requested): per-pixel moments [3, H, W] the backward reads,
        // the weighted loss [1] the forward accumulated, the per-splat distortion depths [N] and their
        // gradient [N], and (with count_left_out) the number of visible splats left out [2]
        // ([0] plane-crossing, [1] nearer than min_dist).
        lfs::core::Tensor distortion_moments;
        lfs::core::Tensor distortion_loss;
        lfs::core::Tensor distortion_depths;
        lfs::core::Tensor distortion_grad_depths;
        lfs::core::Tensor distortion_left_out;

        void set_forward_context(fast_lfs::rasterization::ForwardContext ctx) noexcept {
            release_forward_context();
            forward_ctx = ctx;
            owns_forward_context_ = ctx.success;
        }

        void release_forward_context() noexcept {
            if (!owns_forward_context_) {
                return;
            }
            owns_forward_context_ = false;
            fast_lfs::rasterization::release_forward_context(forward_ctx);
            forward_ctx = {};
        }

        void mark_forward_context_released() noexcept {
            owns_forward_context_ = false;
            forward_ctx = {};
        }

    private:
        bool owns_forward_context_ = false;

        void move_from(FastRasterizeContext&& other) noexcept {
            image = std::move(other.image);
            alpha = std::move(other.alpha);
            depth = std::move(other.depth);
            normal = std::move(other.normal);
            bg_color = std::move(other.bg_color);
            means = std::move(other.means);
            raw_scales = std::move(other.raw_scales);
            raw_rotations = std::move(other.raw_rotations);
            raw_opacities = std::move(other.raw_opacities);
            shN = std::move(other.shN);
            w2c_ptr = std::exchange(other.w2c_ptr, nullptr);
            cam_position_ptr = std::exchange(other.cam_position_ptr, nullptr);
            forward_ctx = std::exchange(other.forward_ctx, {});
            active_sh_bases = std::exchange(other.active_sh_bases, 0);
            width = std::exchange(other.width, 0);
            height = std::exchange(other.height, 0);
            focal_x = std::exchange(other.focal_x, 0.0f);
            focal_y = std::exchange(other.focal_y, 0.0f);
            center_x = std::exchange(other.center_x, 0.0f);
            center_y = std::exchange(other.center_y, 0.0f);
            near_plane = std::exchange(other.near_plane, 0.0f);
            far_plane = std::exchange(other.far_plane, 0.0f);
            mip_filter = std::exchange(other.mip_filter, false);
            tile_x_offset = std::exchange(other.tile_x_offset, 0);
            tile_y_offset = std::exchange(other.tile_y_offset, 0);
            tile_width = std::exchange(other.tile_width, 0);
            tile_height = std::exchange(other.tile_height, 0);
            bg_image = std::move(other.bg_image);
            distortion_moments = std::move(other.distortion_moments);
            distortion_loss = std::move(other.distortion_loss);
            distortion_depths = std::move(other.distortion_depths);
            distortion_grad_depths = std::move(other.distortion_grad_depths);
            distortion_left_out = std::move(other.distortion_left_out);
            owns_forward_context_ = std::exchange(other.owns_forward_context_, false);
        }
    };

    struct FastGSFusedExtraGradients {
        float scale_reg_weight = 0.0f;
        bool scale_reg_log = false;
        float scale_reg_normalizer = 1.0f;
        float erank_reg_weight = 0.0f;
        float dc_reg_weight = 0.0f;
        float sh_rest_reg_weight = 0.0f;
        float flatten_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        // weight * mean(o (1 - o)) over all splats; 0 = off.
        float opacity_binarize_weight = 0.0f;
        float* rendered_count = nullptr;
        // Optional persistent device scalars (caller zeros each step). Accumulated in
        // preprocess_backward so loss-only reg kernels can be skipped on the FastGS path.
        float* scale_reg_loss_out = nullptr;
        float* opacity_reg_loss_out = nullptr;
        float* opacity_binarize_loss_out = nullptr;
        float* erank_reg_loss_out = nullptr;
        float* sh_rest_reg_loss_out = nullptr;
        float* dc_reg_loss_out = nullptr;
        // --loss-footprint (LFQPLCYKMO): see FusedAdamSettings (fused_adam_types.h). weight 0 = off.
        float footprint_weight = 0.0f;
        bool footprint_opacity = false;
        float footprint_max_px_per_fx = 0.0f;
        const float* footprint_shape_raw = nullptr;
        float* footprint_loss_out = nullptr;
        float* footprint_area_out = nullptr;
        // --grad-scale-near D0 (LFQPLCYKMGS; FusedAdamSettings::grad_scale_near_*): scene units, 0 = off. stats_out: optional
        // device [3] (caller zeros): visible splats, splats with s < 1, the sum of their s
        float grad_scale_near = 0.0f;
        float* grad_scale_near_stats_out = nullptr;
        // --aniso-max-ratio r / --aniso-weight lambda (LFQPLCYKMAN; FusedAdamSettings::aniso_*): r = 0 = off.
        // loss_out: optional device scalar (caller zeros) += lambda x mean excess; stats_out: optional [2] += splats, += splats over r
        float aniso_max_ratio = 0.0f;
        float aniso_weight = 0.0f;
        float* aniso_loss_out = nullptr;
        float* aniso_stats_out = nullptr;
        // Optional edge guidance folded into the main blend backward. The map
        // is a median-normalized row-major float32 [H,W], and the destination
        // is a zeroed float32 [N] per-view scratch vector.
        const float* edge_weight_map = nullptr;
        float* edge_score_out = nullptr;
        float* impact_sum_out = nullptr;
        float* impact_peak_out = nullptr;
    };

    // Accumulates FP64 removal energy into aligned CUDA UInt8 [N, sizeof(double)] storage.
    // The caller zeros these bytes once; each call adds a view without clearing them.
    // The projection fast_rasterize_forward uses for a camera: OPENCV_FISHEYE (and its RADIAL /
    // SIMPLE_RADIAL variants) natively unless the camera was undistort-prepared, else pinhole.
    // LFS_FISHEYE_MAX_DEG (default 88) sets how far off the axis splat centres are still drawn.
    [[nodiscard]] fast_lfs::rasterization::FastGSLens fastgs_lens_for_camera(const lfs::core::Camera& camera);

    lfs::Status fast_accumulate_pop_scores(const FastRasterizeContext& ctx, lfs::core::Tensor& scores);

    [[nodiscard]] fast_lfs::rasterization::FusedAdamSettings make_fastgs_fused_adam_settings(
        const FastGSFusedAdamState& optimizer_fused,
        const FastGSFusedExtraGradients& fused_extra_gradients = {});

    // Explicit forward pass - returns render output and context for backward
    // Optional tile parameters for memory-efficient training (tile_width/height=0 means full image)
    // bg_image is optional - if provided, uses per-pixel background blending instead of solid color
    std::expected<std::pair<RenderOutput, FastRasterizeContext>, lfs::Error> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        int tile_x_offset = 0,
        int tile_y_offset = 0,
        int tile_width = 0,
        int tile_height = 0,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {},
        bool render_normal = false,
        bool render_depth = true,
        float dilation_scale = 1.0f,
        bool update_screen_share = true,
        const FastGSDistortionOptions& distortion = {},
        // Radius the collected screen share measures (lfs/training/screen_share.cuh): kScreenShareAxisMax (0,
        // the longest axis) or kScreenShareAxisArea (1, area of the two longest). Must match the optimizer's
        // hinge (publish_screen_share_cap); training callers pass screen_share_axis_code(params).
        int screen_share_axis = 0,
        // Dynamic rolling shutter (native fisheye; FastGSRollingShutter): device pointers to this camera's motion and the
        // readout (training/rolling_shutter_motion.hpp makes them). Default: off, the unchanged kernels.
        const fast_lfs::rasterization::FastGSRollingShutter& rolling_shutter = {},
        // --pose-opt (native fisheye; FastGSPose): a refined world-to-camera transform and centre drawn in place of the
        // camera's own, and the buffer the backward adds the pose gradient to (training/pose_refinement.hpp makes them).
        // Default: the camera's own transform, no pose gradient (the unchanged kernels).
        const fast_lfs::rasterization::FastGSPose& pose = {},
        // --kernel-shape (LFQPLCYK): a model carrying a shape state (SplatData::_shape) is always drawn with its beta;
        // this device [N] buffer, when given, receives dL/dbeta from the backward (training/kernel_shape.hpp).
        float* kernel_shape_grad = nullptr);

    // Backward pass with optional extra alpha gradient for masked training
    void fast_rasterize_backward(
        FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& grad_alpha_extra = {},
        const lfs::core::Tensor& pixel_error_map = {},
        DensificationType densification_type = DensificationType::None,
        int iteration = 0,
        const FastGSFusedExtraGradients& fused_extra_gradients = {},
        const lfs::core::Tensor& grad_depth = {},
        const lfs::core::Tensor& grad_normal = {},
        bool defer_optimizer_step = false);

    // Release per-thread renderer caches before the owning CUDA stream is torn down.
    bool release_fast_rasterizer_thread_local_caches() noexcept;

    // Compatibility no-op for callers of the removed FastGS sort TLS cache.
    void release_fastgs_sort_workspace_buffers() noexcept;

    // Convenience wrapper for inference (no backward needed)
    inline RenderOutput fast_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {},
        bool render_normal = false,
        float dilation_scale = 1.0f) {
        auto result = fast_rasterize_forward(
            viewpoint_camera, gaussian_model, bg_color, 0, 0, 0, 0, mip_filter,
            bg_image, render_normal, /*render_depth=*/true, dilation_scale, /*update_screen_share=*/false);
        if (!result) {
            throw lfs::Exception(std::move(result.error()));
        }
        RenderOutput output = std::move(result->first);
        result->second.release_forward_context();
        return output;
    }

    inline RenderOutput fast_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        bool mip_filter,
        bool render_normal) {
        return fast_rasterize(
            viewpoint_camera, gaussian_model, bg_color, mip_filter, {}, render_normal);
    }

    // Inference-only rasterization does not mutate the camera; this overload avoids
    // forcing callers with const camera handles to cast away constness at the call site.
    inline RenderOutput fast_rasterize(
        const lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {},
        bool render_normal = false,
        float dilation_scale = 1.0f) {
        return fast_rasterize(
            const_cast<lfs::core::Camera&>(viewpoint_camera),
            gaussian_model,
            bg_color,
            mip_filter,
            bg_image,
            render_normal,
            dilation_scale);
    }

    inline RenderOutput fast_rasterize(
        const lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        bool mip_filter,
        bool render_normal) {
        return fast_rasterize(
            const_cast<lfs::core::Camera&>(viewpoint_camera),
            gaussian_model,
            bg_color,
            mip_filter,
            render_normal);
    }
} // namespace lfs::training
