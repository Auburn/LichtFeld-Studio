/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/optimization_properties.hpp"

#include "core/parameters.hpp"
#include "core/property_registry.hpp"

#include <limits>
#include <mutex>

namespace lfs::core::param {

    using namespace lfs::core::prop;

    void register_optimization_properties() {
        // Registry defaults match the default strategy (MRNF), not bare struct {}.
        const OptimizationParameters d = OptimizationParameters::mrnf_defaults();
        PropertyGroupBuilder<OptimizationParameters>("optimization", "Optimization")
            // Training control
            .size_prop(&OptimizationParameters::iterations,
                       "iterations", "Max Iterations", d.iterations, 1, 1000000,
                       "Maximum number of training iterations")
            .json_required()
            .locale("training_params.iterations")
            .tooltip("training.tooltip.iterations")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .int_prop(&OptimizationParameters::sh_degree,
                      "sh_degree", "SH Degree", d.sh_degree, 0, 3,
                      "Spherical harmonics degree (0-3)")
            .json_required()
            .all_strategies()
            .size_prop(&OptimizationParameters::sh_degree_interval,
                       "sh_degree_interval", "SH Interval", d.sh_degree_interval, 100, 10000,
                       "Iterations between SH degree increases")
            .locale("training.refinement.sh_upgrade_every")
            .tooltip("training.tooltip.sh_upgrade_every")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .int_prop(&OptimizationParameters::max_cap,
                      "max_cap", "Max Gaussians", d.max_cap, 1000, 200000000,
                      "Maximum number of gaussians")
            .locale("training_params.max_gaussians")
            .tooltip("training.tooltip.max_gaussians")
            .precision(0)
            .ui_step(100000)

            // Learning rates
            .all_strategies()
            .float_prop(&OptimizationParameters::means_lr,
                        "means_lr", "Position LR", d.means_lr, 0.0f, 0.001f,
                        "Learning rate for gaussian positions")
            .json_required()
            .locale("training.opt.lr.position")
            .tooltip("training.tooltip.lr_position")
            .precision(6)
            .ui_step(1e-6)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::means_lr_end,
                        "means_lr_end", "Position LR End", d.means_lr_end, 0.0f, 0.001f,
                        "Target end learning rate for gaussian positions")
            .locale("training.advanced.means_lr_end")
            .tooltip("training.tooltip.means_lr_end")
            .precision(8)
            .ui_step(1e-8)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::shs_lr,
                        "shs_lr", "SH LR", d.shs_lr, 0.0f, 0.1f,
                        "Learning rate for spherical harmonics")
            .json_required()
            .locale("training.opt.lr.sh_coeff")
            .tooltip("training.tooltip.lr_sh_coeff")
            .precision(4)
            .ui_step(1e-4)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::opacity_lr,
                        "opacity_lr", "Opacity LR", d.opacity_lr, 0.0f, 1.0f,
                        "Learning rate for opacity")
            .json_required()
            .locale("training.opt.lr.opacity")
            .tooltip("training.tooltip.lr_opacity")
            .precision(4)
            .ui_step(0.001)
            .all_strategies()
            .float_prop(&OptimizationParameters::scaling_lr,
                        "scaling_lr", "Scale LR", d.scaling_lr, 0.0f, 0.1f,
                        "Learning rate for gaussian scales")
            .json_required()
            .locale("training.opt.lr.scaling")
            .tooltip("training.tooltip.lr_scaling")
            .precision(4)
            .ui_step(1e-4)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::scaling_lr_end,
                        "scaling_lr_end", "Scale LR End", d.scaling_lr_end, 0.0f, 0.1f,
                        "Target end learning rate for gaussian scales")
            .locale("training.advanced.scaling_lr_end")
            .tooltip("training.tooltip.scaling_lr_end")
            .precision(6)
            .ui_step(1e-5)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .strategies({"mrnf"})
            .float_prop(&OptimizationParameters::late_lr_anneal,
                        "late_lr_anneal", "Late Learning Rate Fade", d.late_lr_anneal, 0.01f, 1.0f,
                        "After growth ends, color and opacity learning slows to this fraction of normal speed by the last iteration, so the model does not memorize single photos; 1 turns it off")
            .locale("training.advanced.late_lr_anneal")
            .tooltip("training.tooltip.late_lr_anneal")
            .precision(2)
            .ui_step(0.05)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::rotation_lr,
                        "rotation_lr", "Rotation LR", d.rotation_lr, 0.0f, 0.1f,
                        "Learning rate for rotations")
            .json_required()
            .locale("training.opt.lr.rotation")
            .tooltip("training.tooltip.lr_rotation")
            .precision(4)
            .ui_step(1e-4)
            .flags(PROP_LIVE_UPDATE)
            .all_strategies()
            .float_prop(&OptimizationParameters::cropbox_lr_scale,
                        "cropbox_lr_scale", "Rejected splat LR scale", d.cropbox_lr_scale, 0.0f, 1.0f,
                        "Scales Adam steps and refinement signals for rejected splats; strategy noise, decay, and resets remain active")
            .locale("training.advanced.cropbox_lr_scale")
            .tooltip("training.tooltip.cropbox_lr_scale")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::cropbox_loss_weight,
                        "cropbox_loss_weight", "Outside ROI loss weight", d.cropbox_loss_weight, 0.0f, 1.0f,
                        "Scales pixel losses for camera rays outside the active crop box")
            .locale("training.advanced.cropbox_loss_weight")
            .tooltip("training.tooltip.cropbox_loss_weight")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::cropbox_growth_inside,
                        "cropbox_growth_inside", "ROI growth inside", d.cropbox_growth_inside, 0.0f, 1.0f,
                        "Share of new splats placed inside the active crop box")
            .locale("training.advanced.cropbox_growth_inside")
            .tooltip("training.tooltip.cropbox_growth_inside")
            .precision(2)
            .ui_step(0.05)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)

            // Loss parameters
            .all_strategies()
            .float_prop(&OptimizationParameters::lambda_dssim,
                        "lambda_dssim", "DSSIM Weight", d.lambda_dssim, 0.0f, 1.0f,
                        "Weight for structural similarity loss")
            .json_required()
            .all_strategies()
            .float_prop(&OptimizationParameters::opacity_reg,
                        "opacity_reg", "Opacity Reg", d.opacity_reg, 0.0f, 1.0f,
                        "Opacity regularization weight")
            .locale("training.losses.opacity_reg")
            .tooltip("training.tooltip.opacity_reg")
            .precision(4)
            .ui_step(0.001)
            .all_strategies()
            .float_prop(&OptimizationParameters::opacity_binarize_weight,
                        "opacity_binarize_weight", "Opacity Binarize", d.opacity_binarize_weight, 0.0f, 1.0f,
                        "Weight of the loss mean(o (1 - o)) over all splats, which pushes every opacity toward 0 or 1; 0 turns it off")
            .precision(4)
            .ui_step(0.001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::opacity_binarize_start,
                       "opacity_binarize_start", "Opacity Binarize Start", d.opacity_binarize_start, 0, 1000000,
                       "Iteration from which the opacity binarization loss is active")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::distortion_weight,
                        "distortion_weight", "Depth Distortion", d.distortion_weight, 0.0f, 100000.0f,
                        "Weight of the per-pixel depth distortion loss (2DGS), which pulls the splats seen through a pixel onto one depth; standard rasterizer only, 0 turns it off")
            .precision(2)
            .ui_step(10.0)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::distortion_start,
                       "distortion_start", "Depth Distortion Start", d.distortion_start, 0, 1000000,
                       "Iteration from which the depth distortion loss is active")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::distortion_near,
                        "distortion_near", "Depth Distortion Near", d.distortion_near, 0.001f, 100.0f,
                        "Near depth of the distortion loss depth mapping m = far/(far - near) (1 - near/d)")
            .precision(3)
            .ui_step(0.05)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::distortion_far,
                        "distortion_far", "Depth Distortion Far", d.distortion_far, 1.0f, 1000000.0f,
                        "Far depth of the distortion loss depth mapping")
            .precision(1)
            .ui_step(10.0)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::distortion_depth,
                         "distortion_depth", "Depth Distortion Depth", d.distortion_depth,
                         "Depth used by the distortion loss: distance (from the camera centre, default) or z (camera z on the pinhole path, the old behaviour, which rewards opaque splats beside the camera)")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::distortion_skip_plane_crossing,
                        "distortion_skip_plane_crossing", "Depth Distortion Plane Skip", d.distortion_skip_plane_crossing, 0.0f, 10.0f,
                        "Leaves out of the distortion loss, per view, splats whose k-sigma extent along the camera axis crosses the camera plane (k = this value; 0 = none)")
            .precision(2)
            .ui_step(0.5)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::distortion_min_dist,
                        "distortion_min_dist", "Depth Distortion Min Distance", d.distortion_min_dist, 0.0f, 100.0f,
                        "Leaves out of the distortion loss, per view, splats nearer to the camera than this (scene units; 0 = none). Stops very close objects (whose edges dominate the loss) from being widened into opaque walls")
            .precision(2)
            .ui_step(0.25)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::generated_prefix,
                         "generated_prefix", "Generated View Prefix", d.generated_prefix,
                         "Training images whose file name starts with this prefix are generated views: they use the alignment-tolerant low-pass + texture loss instead of L1 + SSIM; empty turns it off")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::generated_lowpass_sigma,
                        "generated_lowpass_sigma", "Generated Low-pass Sigma", d.generated_lowpass_sigma, 0.0f, 10.0f,
                        "Generated views: standard deviation in pixels of the Gaussian blur before the low-pass L1 (0 = no blur)")
            .precision(2)
            .ui_step(0.25)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .int_prop(&OptimizationParameters::generated_texture_radius,
                      "generated_texture_radius", "Generated Texture Radius", d.generated_texture_radius, 1, 32,
                      "Generated views: radius in pixels of the box window of the local luma standard deviation")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::generated_lowpass_weight,
                        "generated_lowpass_weight", "Generated Low-pass Weight", d.generated_lowpass_weight, 0.0f, 100.0f,
                        "Generated views: weight of the L1 between the blurred render and blurred target")
            .precision(3)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::generated_texture_weight,
                        "generated_texture_weight", "Generated Texture Weight", d.generated_texture_weight, 0.0f, 100.0f,
                        "Generated views: weight of the mean absolute difference of the local luma standard deviations")
            .precision(3)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::generated_pixel_weight,
                        "generated_pixel_weight", "Generated Pixel Weight", d.generated_pixel_weight, 0.0f, 100.0f,
                        "Generated views: weight of the plain per-pixel L1")
            .precision(3)
            .ui_step(0.05)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::generated_loss_weight,
                        "generated_loss_weight", "Generated Loss Weight", d.generated_loss_weight, 0.0f, 100.0f,
                        "Generated views: scale of the whole generated-view loss relative to a photo's photometric loss")
            .precision(3)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::loss_weight_dir,
                         "loss_weight_dir", "Loss Weight Folder", d.loss_weight_dir,
                         "Folder of per-pixel photometric loss weight maps, <image name>.png per training image (16-bit PNG, weight = value / 10000, or float EXR/TIFF); they multiply the per-pixel L1 + SSIM and the densification error map; a missing map is weight 1; empty turns it off")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::lens_radial,
                         "lens_radial", "Radial Lens Table", d.lens_radial,
                         "Radial lens table (--native-fisheye; abq/lens_radial.py format): a tabulated radial profile theta -> theta_d that replaces the OPENCV_FISHEYE k-polynomial of the table's own camera in the projection, its Jacobian, the rolling-shutter row solve, the culling bound and blob seeding; any other fisheye camera is an error; empty turns it off")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::rolling_shutter,
                         "rolling_shutter", "Rolling Shutter Motion File", d.rolling_shutter,
                         "Dynamic rolling shutter (--native-fisheye): file of per-image motion during the readout (name, angular velocity, angular acceleration, linear velocity, linear acceleration in the mid-readout camera axes); every splat is drawn through the pose of the time its sensor row was read; empty turns it off")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::rs_readout_ms,
                        "rs_readout_ms", "Rolling Shutter Readout", d.rs_readout_ms, 0.0f, 100.0f,
                        "Rolling shutter readout time from the top row to the bottom row (ms)")
            .precision(2)
            .ui_step(1.0)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::rs_learn,
                         "rs_learn", "Rolling Shutter Learn", d.rs_learn,
                         "Rolling shutter quantities refined during training: comma list of omega, alpha, vel, acc, readout (empty = the motion file and readout are used as they are)")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::rs_lr,
                        "rs_lr", "Rolling Shutter Motion LR", d.rs_lr, 0.0f, 10.0f,
                        "Adam step of the per-camera rolling-shutter motion corrections, in pixels at the readout ends per camera update")
            .precision(4)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::rs_readout_lr,
                        "rs_readout_lr", "Rolling Shutter Readout LR", d.rs_readout_lr, 0.0f, 1.0f,
                        "Adam step of the shared rolling-shutter readout time (ms per iteration)")
            .precision(4)
            .ui_step(0.001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::rs_prior,
                        "rs_prior", "Rolling Shutter Motion Prior", d.rs_prior, 0.0f, 1.0f,
                        "Weight of the prior pulling each camera's rolling-shutter motion correction (pixels at the readout ends) to zero")
            .precision(6)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::rs_smooth,
                        "rs_smooth", "Rolling Shutter Smoothness", d.rs_smooth, 0.0f, 1.0f,
                        "Weight tying each camera's rolling-shutter motion correction to its time neighbours' (adjacent training images at most 12 video frames apart)")
            .precision(6)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::rs_learn_start,
                       "rs_learn_start", "Rolling Shutter Learn Start", d.rs_learn_start, 0, 1000000,
                       "Iteration from which the rolling-shutter motion / readout are refined")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::rs_motion_start,
                       "rs_motion_start", "Rolling Shutter Motion Start", d.rs_motion_start, 0, 1000000,
                       "With the readout and the motion both learned: the readout is learned until this iteration and the per-camera motion from it on (0 = both from the learn start)")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::motion_blur,
                       "motion_blur", "Motion Blur", d.motion_blur,
                       "With --rolling-shutter: model the blur of each frame's exposure (--exposure-ms) on top of the row times: each splat's 2D covariance is widened along its exposure smear (camera rotation and travel at the splat) and its opacity scaled to keep its coverage")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::exposure_ms,
                        "exposure_ms", "Exposure (ms)", d.exposure_ms, 0.0f, 50.0f,
                        "--motion-blur: the exposure time in milliseconds (1/500 s = 2); 0 = no blur (identical to without --motion-blur)")
            .precision(3)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::motion_blur_learn_exposure,
                       "motion_blur_learn_exposure", "Learn Exposure Scale", d.motion_blur_learn_exposure,
                       "--motion-blur: learn a per-camera scale of the exposure (start 1, clamped to [0.5, 2]) from --rs-learn-start; written to motion_blur_exposure.txt")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::motion_blur_exposure_lr,
                        "motion_blur_exposure_lr", "Exposure Scale LR", d.motion_blur_exposure_lr, 0.0f, 1.0f,
                        "--motion-blur-learn-exposure: Adam step of the scale per camera update")
            .precision(4)
            .ui_step(0.001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::loss_ycc,
                       "loss_ycc", "YCbCr Loss", d.loss_ycc,
                       "Photometric loss in BT.709 full-range Y'CbCr of the image the RGB loss sees (after the cubemap composite and PPISP) and of the target: L1 + D-SSIM on luma at full resolution instead of the RGB L1 + D-SSIM, plus an L1 on Cb and Cr after a 2x2 box downsample (half resolution: 4:2:0 video carries no finer colour); masks keep only fully kept 2x2 blocks for the chroma term")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::loss_ycc_chroma_weight,
                        "loss_ycc_chroma_weight", "YCbCr Chroma Weight", d.loss_ycc_chroma_weight, 0.0f, 100.0f,
                        "--loss-ycc: weight of the half-resolution chroma L1 (mean of Cb and Cr) relative to the luma L1, inside the L1 share (1 - lambda_dssim)")
            .precision(4)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::loss_ycc_luma_scale,
                        "loss_ycc_luma_scale", "YCbCr Luma L1 Scale", d.loss_ycc_luma_scale, 0.0f, 100.0f,
                        "--loss-ycc: scale of the luma L1 (the D-SSIM share stays lambda_dssim), to match the RGB L1's magnitude")
            .precision(4)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::max_std_dev,
                        "max_std_dev", "Splat Cutoff (std devs)", d.max_std_dev, 0.0f, 10.0f,
                        "Cut every splat as the game renderer (fpv-render) does: a pixel takes part only within this many standard deviations (Mahalanobis, in the 0.3-px-dilated 2D covariance) and above alpha 0.5/255, in the forward and the backward; the tile radius shrinks to match; the game draws at sqrt(5) = 2.2360680; 0 = off (the trainer's own extent, alpha >= 1/255)")
            .precision(7)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::max_std_dev_distance,
                       "max_std_dev_distance", "Splat Cutoff on Distance", d.max_std_dev_distance,
                       "With --max-std-dev and --kernel-shape: cut every splat at Mahalanobis distance d <= max_std_dev (and alpha >= 0.5/255), the bounds at min(alpha contour, max_std_dev), as a game drawing per-splat radius min(contour, C) would; identical to --max-std-dev for the Gaussian")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::kernel_shape,
                       "kernel_shape", "Kernel Shape (per splat)", d.kernel_shape,
                       "Per-splat trainable kernel shape (a generalized Gaussian): alpha = opacity x exp(-1/2 d^beta), d the Mahalanobis distance in the 2D conic, beta = 1 + 7 sigmoid(r - ln 6) in [1, 8] learned per splat from 2 (the Gaussian); beta > 2 flat top and sharp edge, beta < 2 a peak with long tails; the tile extent follows (2 ln(255 opacity))^(1/beta); exported as the PLY vertex property 'shape'; standard rasterizer, MRNF; not with --max-std-dev, --enable-mip or --enable-sparsity")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::kernel_shape_lr,
                        "kernel_shape_lr", "Kernel Shape Learning Rate", d.kernel_shape_lr, 0.0f, 1.0f,
                        "Adam learning rate of the raw shape r (beta = 1 + 7 sigmoid(r - ln 6); d beta / d r = 6/7 at beta 2); 0 freezes beta at 2")
            .precision(5)
            .ui_step(0.001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::kernel_shape_start,
                       "kernel_shape_start", "Kernel Shape Start", d.kernel_shape_start, 0, 1000000,
                       "Iteration from which the kernel shape is learned (before it: beta stays where it is)")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::loss_footprint,
                        "loss_footprint", "Footprint Loss (game draw cost)", d.loss_footprint, 0.0f, 10.0f,
                        "LFQPLCYKMO: lambda x the mean number of splats the game draws per pixel of the view (each splat's fpv-render quad, "
                        "4 min(512 px, R sqrt(l1)) min(512 px, R sqrt(l2)) of the dilated 2D covariance, R = sqrt5 for a Gaussian, "
                        "min(sqrt5, (2 ln(510 o))^(1/beta)) for a shaped splat; summed over the view, over W x H); gradient into the "
                        "log-scales (beta, means and rotations get none); standard rasterizer; 0 = off")
            .precision(6)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::loss_footprint_start,
                       "loss_footprint_start", "Footprint Loss Start", d.loss_footprint_start, 0, 1000000,
                       "Iteration from which the footprint loss acts (scaled by --steps-scaler, as --distortion-start)")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::loss_footprint_ramp,
                       "loss_footprint_ramp", "Footprint Loss Ramp", d.loss_footprint_ramp, 0, 1000000,
                       "Iterations over which the footprint loss weight ramps linearly from 0 to lambda (scaled by --steps-scaler; 0 = at once)")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::loss_footprint_fraction,
                        "loss_footprint_fraction", "Footprint Loss Fraction (auto lambda)", d.loss_footprint_fraction, 0.0f, 10.0f,
                        "Instead of a fixed lambda: set lambda at the footprint loss start so the term starts at this fraction of the image "
                        "loss (fraction x mean image loss / mean footprint over the log-cadence samples of the 2000 iterations before the "
                        "start, where the footprint is measured without a gradient); 0 = use --loss-footprint")
            .precision(4)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::grad_scale_near,
                        "grad_scale_near", "Near-Camera Gradient Scaling Distance", d.grad_scale_near, 0.0f, 1000.0f,
                        "Floaters No More (Philip & Deschaintre 2023): each training view's gradients of a splat's own parameters "
                        "(means, scales, rotation, opacity, SH colour, kernel shape) x min(1, (d / D0)^2), d = the splat centre's "
                        "distance from that view's camera centre; D0 in scene units; 0 = off")
            .precision(3)
            .ui_step(0.05)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::aniso_max_ratio,
                        "aniso_max_ratio", "Anisotropy Max Ratio", d.aniso_max_ratio, 0.0f, 1000.0f,
                        "Anti-needle loss (PhysGaussian): per splat max(0, s_max / s_mid - r) of its activated scales, averaged over "
                        "splats, x --aniso-weight; r = the allowed largest-to-middle scale ratio; 0 = off")
            .precision(2)
            .ui_step(0.5)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::aniso_weight,
                        "aniso_weight", "Anisotropy Loss Weight", d.aniso_weight, 0.0f, 1000000.0f,
                        "lambda of the anti-needle loss (--aniso-max-ratio): loss = lambda x mean over splats of max(0, s_max / s_mid - r)")
            .precision(3)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::aniso_start,
                       "aniso_start", "Anisotropy Loss Start", d.aniso_start, 0, 1000000,
                       "Iteration from which the anti-needle loss (--aniso-max-ratio) is active")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::thick_max_ratio,
                        "thick_max_ratio", "Thickness Floor Max Ratio", d.thick_max_ratio, 0.0f, 1000000.0f,
                        "Thickness floor (against paper-thin discs that draw as hairlines edge-on): per splat max(0, ln(s_mid / s_min) - ln R) "
                        "of its activated scales, averaged over splats, x --thick-weight; R = the allowed middle-to-smallest scale ratio; 0 = off")
            .precision(2)
            .ui_step(1.0)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::thick_weight,
                        "thick_weight", "Thickness Floor Weight", d.thick_weight, 0.0f, 1000000.0f,
                        "lambda of the thickness floor (--thick-max-ratio): loss = lambda x mean over splats of max(0, ln(s_mid / s_min) - ln R)")
            .precision(3)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::thick_start,
                       "thick_start", "Thickness Floor Start", d.thick_start, 0, 1000000,
                       "Iteration from which the thickness floor (--thick-max-ratio) is active")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::thick_longest,
                       "thick_longest", "Thickness Floor vs Longest Axis", d.thick_longest,
                       "Measure the thickness floor (--thick-max-ratio) against the longest axis, ln(s_max / s_min), instead of the middle one: "
                       "bounds every splat's projected aspect by about R")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::loss_footprint_opacity,
                       "loss_footprint_opacity", "Footprint Loss Opacity Gradient", d.loss_footprint_opacity,
                       "Also send the footprint loss gradient into the opacity, through the quad reach (shaped splats whose alpha-floor "
                       "contour lies inside the sqrt5 cut: it pushes faint big splats to fade out); default off")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::pose_opt,
                       "pose_opt", "Camera Pose Refinement", d.pose_opt,
                       "Refine every training camera's pose during training (--native-fisheye standard rasterizer, with or without the rolling shutter; not --gut): a per-camera Adam step of its rotation and camera-centre offset after each of its renders; writes <output>/cameras_refined (COLMAP text, the splat's frame) and pose_refinement.tsv")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::pose_lr_deg,
                        "pose_lr_deg", "Pose Rotation LR", d.pose_lr_deg, 0.0f, 10.0f,
                        "Camera pose refinement: Adam step of each camera's rotation, in degrees per camera update (one update each time the camera is rendered)")
            .precision(5)
            .ui_step(0.001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::pose_lr_units,
                        "pose_lr_units", "Pose Translation LR", d.pose_lr_units, 0.0f, 10.0f,
                        "Camera pose refinement: Adam step of each camera centre's move, in scene units per camera update")
            .precision(6)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::pose_opt_start,
                       "pose_opt_start", "Pose Refinement Start", d.pose_opt_start, 0, 1000000,
                       "Camera pose refinement: first iteration that moves the cameras")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .size_prop(&OptimizationParameters::pose_opt_hold,
                       "pose_opt_hold", "Pose Refinement Hold", d.pose_opt_hold, 0, 1000000,
                       "Camera pose refinement: the learning rates stay at full size until this iteration, then halve smoothly every half-life")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::pose_opt_half_life,
                        "pose_opt_half_life", "Pose Refinement Half-life", d.pose_opt_half_life, 0.0f, 1000000.0f,
                        "Camera pose refinement: iterations per halving of the learning rates after the hold (0 = auto: (last iteration - hold) / 7, i.e. 1/128 of the rate at the end)")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::pose_prior,
                        "pose_prior", "Pose Prior", d.pose_prior, 0.0f, 1.0f,
                        "Camera pose refinement: weight of the prior pulling each camera's offset (in pixels: focal x radians / x units) back to its start pose")
            .precision(6)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::pose_smooth,
                        "pose_smooth", "Pose Smoothness", d.pose_smooth, 0.0f, 1.0f,
                        "Camera pose refinement: weight tying each camera's offset (pixels, compared in the world frame) to its time neighbours' (adjacent training images at most 12 video frames apart)")
            .precision(6)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::scale_reg,
                        "scale_reg", "Scale Reg", d.scale_reg, 0.0f, 1.0f,
                        "Scale regularization weight")
            .locale("training.losses.scale_reg")
            .tooltip("training.tooltip.scale_reg")
            .precision(4)
            .ui_step(0.001)
            .all_strategies()
            .float_prop(&OptimizationParameters::scale_reg_decay_power,
                        "scale_reg_decay_power", "Scale Penalty Fade", d.scale_reg_decay_power, -1.0f, 4.0f,
                        "How fast the scale penalty fades to zero over training; higher starts stronger and fades faster, negative keeps it constant")
            .strategies({"mrnf"})
            .locale("training.losses.scale_reg_decay_power")
            .tooltip("training.tooltip.scale_reg_decay_power")
            .flags(PROP_ADVANCED)
            .precision(2)
            .ui_step(0.1)
            .float_prop(&OptimizationParameters::erank_reg,
                        "erank_reg", "Spike Shape Penalty", d.erank_reg, 0.0f, 1.0f,
                        "Discourages needle-shaped splats that show up as streaks from new angles; 0 turns it off")
            .strategies({"mrnf"})
            .locale("training.losses.erank_reg")
            .tooltip("training.tooltip.erank_reg")
            .flags(PROP_ADVANCED)
            .precision(4)
            .ui_step(0.001)
            .float_prop(&OptimizationParameters::dc_reg,
                        "dc_reg", "Color Range Penalty", d.dc_reg, 0.0f, 1.0f,
                        "Keeps base colors between black and white so splats cannot cancel each other out; 0 turns it off")
            .strategies({"mrnf"})
            .locale("training.losses.dc_reg")
            .tooltip("training.tooltip.dc_reg")
            .flags(PROP_ADVANCED)
            .precision(4)
            .ui_step(0.001)
            .float_prop(&OptimizationParameters::sh_rest_reg,
                        "sh_rest_reg", "View-Dependent Color Penalty", d.sh_rest_reg, 0.0f, 1.0f,
                        "Keeps the angle-dependent part of each color small so splats cannot fake detail by changing color with the view; 0 turns it off")
            .strategies({"mrnf"})
            .locale("training.losses.sh_rest_reg")
            .tooltip("training.tooltip.sh_rest_reg")
            .precision(4)
            .ui_step(0.001)
            .flags(PROP_LIVE_UPDATE | PROP_ADVANCED)

            .all_strategies()
            .float_prop(&OptimizationParameters::thin_structure_weight,
                        "thin_structure_weight", "Thin Structure Emphasis", d.thin_structure_weight, 0.0f, 4.0f,
                        "Makes errors on thin lines such as cables and branches count more so they are not blurred away; 0 turns it off, 0.5 to 1 works well")
            .locale("training.losses.thin_structure_weight")
            .tooltip("training.tooltip.thin_structure_weight")
            .flags(PROP_ADVANCED)
            .precision(2)
            .ui_step(0.1)

            .all_strategies()
            .float_prop(&OptimizationParameters::gradient_loss_weight,
                        "gradient_loss_weight", "Edge Sharpness Loss", d.gradient_loss_weight, 0.0f, 8.0f,
                        "From iteration 2000, also matches edges and fine texture of the photos to keep the result sharp; 0 turns it off")
            .locale("training.losses.gradient_loss_weight")
            .tooltip("training.tooltip.gradient_loss_weight")
            .flags(PROP_ADVANCED)
            .precision(2)
            .ui_step(0.1)

            .all_strategies()
            .bool_prop(&OptimizationParameters::opacity_decay_rendered_only,
                       "opacity_decay_rendered_only", "Fade Only Visible Splats", d.opacity_decay_rendered_only,
                       "Fades the opacity only of splats drawn since the last refinement, so splats out of view are not lost; no effect with GUT")
            .locale("training.losses.opacity_decay_rendered_only")
            .tooltip("training.tooltip.opacity_decay_rendered_only")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::densify_structure_weight,
                        "densify_structure_weight", "Thin Structure Growth", d.densify_structure_weight, 0.0f, 4.0f,
                        "Gives thin lines such as cables and branches priority when new splats are added; 0 turns it off")
            .locale("training.losses.densify_structure_weight")
            .tooltip("training.tooltip.densify_structure_weight")
            .flags(PROP_ADVANCED)
            .precision(2)
            .ui_step(0.1)

            // Refinement
            .all_strategies()
            .size_prop(&OptimizationParameters::refine_every,
                       "refine_every", "Refine Every", d.refine_every, 1, 1000,
                       "Interval for adaptive density control")
            .json_required()
            .locale("training.refinement.refine_every")
            .tooltip("training.tooltip.refine_every")
            .precision(0)
            .ui_step(10)
            .all_strategies()
            .size_prop(&OptimizationParameters::start_refine,
                       "start_refine", "Start Refine", d.start_refine, 0, 10000,
                       "Iteration to start refinement")
            .json_required()
            .locale("training.refinement.start_refine")
            .tooltip("training.tooltip.start_refine")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .size_prop(&OptimizationParameters::stop_refine,
                       "stop_refine", "Stop Refine", d.stop_refine, 0, 100000,
                       "Iteration to stop refinement")
            .json_required()
            .locale("training.refinement.stop_refine")
            .tooltip("training.tooltip.stop_refine")
            .precision(0)
            .ui_step(1000)
            .all_strategies()
            .size_prop(&OptimizationParameters::morton_reorder_interval,
                       "morton_reorder_interval", "Morton Reorder", d.morton_reorder_interval, 0, 30000,
                       "Reorder Gaussians by 3D Morton code every N iterations (0 disables)")
            .locale("training.refinement.morton_reorder_interval")
            .tooltip("training.tooltip.morton_reorder_interval")
            .precision(0)
            .ui_step(1000)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::min_opacity,
                        "min_opacity", "Min Opacity", d.min_opacity, 0.0f, std::numeric_limits<float>::infinity(),
                        "Minimum opacity for pruning")
            .json_required()
            .locale("training.thresholds.min_opacity")
            .tooltip("training.tooltip.min_opacity")
            .precision(4)
            .ui_step(0.001)
            .flags(PROP_ADVANCED)
            .strategies({"mcmc"})
            .all_strategies()
            .float_prop(&OptimizationParameters::init_opacity,
                        "init_opacity", "Init Opacity", d.init_opacity, 0.0f, 1.0f,
                        "Initial opacity for new gaussians")
            .all_strategies()
            .float_prop(&OptimizationParameters::init_scaling,
                        "init_scaling", "Init Scale", d.init_scaling, 0.0f, 1.0f,
                        "Initial scale for new gaussians")
            .locale("training.init.init_scaling")
            .tooltip("training.tooltip.init_scaling")
            .precision(3)
            .ui_step(0.01)

            // Mask parameters
            .all_strategies()
            .enum_prop(&OptimizationParameters::mask_mode,
                       "mask_mode", "Mask Mode", d.mask_mode,
                       {{"None", MaskMode::None, "training.options.mask.none", "none"},
                        {"Segment", MaskMode::Segment, "training.options.mask.segment", "segment"},
                        {"Ignore", MaskMode::Ignore, "training.options.mask.ignore", "ignore"},
                        {"SegmentAndIgnore", MaskMode::SegmentAndIgnore, "training.options.mask.segment_and_ignore", "segment_and_ignore"},
                        {"AlphaConsistent", MaskMode::AlphaConsistent, "training.options.mask.alpha_consistent", "alpha_consistent"}},
                       "Attention mask behavior during training")
            .locale("training_params.mask_mode")
            .tooltip("training.tooltip.mask_mode")
            .all_strategies()
            .bool_prop(&OptimizationParameters::invert_masks,
                       "invert_masks", "Invert Masks", d.invert_masks,
                       "Swap object and background in masks")
            .locale("training_params.invert_masks")
            .tooltip("training.tooltip.invert_masks")
            .all_strategies()
            .float_prop(&OptimizationParameters::mask_threshold,
                        "mask_threshold", "Mask Threshold", d.mask_threshold, 0.0f, 1.0f,
                        "Threshold for mask binarization")
            .locale("training.masking.threshold")
            .tooltip("training.tooltip.mask_threshold")
            .precision(3)
            .ui_step(0.05)
            .all_strategies()
            .float_prop(&OptimizationParameters::mask_opacity_penalty_weight,
                        "mask_opacity_penalty_weight", "Penalty Weight", d.mask_opacity_penalty_weight, 0.0f, 10.0f,
                        "Opacity penalty weight for segment mode")
            .locale("training.masking.penalty_weight")
            .tooltip("training.tooltip.penalty_weight")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .float_prop(&OptimizationParameters::mask_opacity_penalty_power,
                        "mask_opacity_penalty_power", "Penalty Power", d.mask_opacity_penalty_power, 0.5f, 4.0f,
                        "Power for opacity penalty in segment mode")
            .locale("training.masking.penalty_power")
            .tooltip("training.tooltip.penalty_power")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_alpha_as_mask,
                       "use_alpha_as_mask", "Use Alpha as Mask", d.use_alpha_as_mask,
                       "Use alpha channel from RGBA images as mask source")
            .locale("training_params.use_alpha_as_mask")
            .tooltip("training.tooltip.use_alpha_as_mask")
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_depth_loss,
                       "use_depth_loss", "Use Depth Loss", d.use_depth_loss,
                       "Use dataset depth maps for depth supervision")
            .locale("training_params.use_depth_loss")
            .tooltip("training.tooltip.use_depth_loss")
            .all_strategies()
            .float_prop(&OptimizationParameters::depth_loss_weight,
                        "depth_loss_weight", "Depth Loss Weight", d.depth_loss_weight, 0.0f, 100.0f,
                        "Weight for depth supervision")
            .locale("training_params.depth_loss_weight")
            .tooltip("training.tooltip.depth_loss_weight")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .string_prop(&OptimizationParameters::depth_loss_mode,
                         "depth_loss_mode", "Depth Loss Mode", d.depth_loss_mode,
                         "Depth prior convention: ssi (auto-detect), ssi-disparity, or ssi-depth")
            .all_strategies()
            .bool_prop(&OptimizationParameters::depth_auto_generate,
                       "depth_auto_generate", "Auto-generate Depth", d.depth_auto_generate,
                       "Generate missing or size-mismatched depth maps with MoGe-2 from the full-resolution images/ folder")
            .locale("training_params.depth_auto_generate")
            .tooltip("training.tooltip.depth_auto_generate")
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_normal_loss,
                       "use_normal_loss", "Use Normal Loss", d.use_normal_loss,
                       "Use dataset normal maps for normal supervision")
            .locale("training_params.use_normal_loss")
            .tooltip("training.tooltip.use_normal_loss")
            .all_strategies()
            .bool_prop(&OptimizationParameters::normal_auto_generate,
                       "normal_auto_generate", "Auto-generate Normals", d.normal_auto_generate,
                       "Generate missing or size-mismatched maps with MoGe-2 from the full-resolution images/ folder so they work at every training resolution")
            .locale("training_params.normal_auto_generate")
            .tooltip("training.tooltip.normal_auto_generate")
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_loss_weight,
                        "normal_loss_weight", "Normal Loss Weight", d.normal_loss_weight, 0.0f, 100.0f,
                        "Weight for prior normal supervision")
            .locale("training_params.normal_loss_weight")
            .tooltip("training.tooltip.normal_loss_weight")
            .precision(3)
            .ui_step(0.01)
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_consistency_weight,
                        "normal_consistency_weight", "Normal Consistency Weight", d.normal_consistency_weight, 0.0f, 100.0f,
                        "Weight for depth-normal consistency")
            .locale("training_params.normal_consistency_weight")
            .tooltip("training.tooltip.normal_consistency_weight")
            .precision(3)
            .ui_step(0.01)
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_flatten_weight,
                        "normal_flatten_weight", "Normal Flatten Weight", d.normal_flatten_weight, 0.0f, 1000.0f,
                        "Min-axis scale flattening weight while normal supervision is active")
            .locale("training_params.normal_flatten_weight")
            .tooltip("training.tooltip.normal_flatten_weight")
            .precision(3)
            .ui_step(0.1)
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_start_fraction,
                        "normal_start_fraction", "Normal Start Fraction", d.normal_start_fraction, 0.0f, 1.0f,
                        "Fraction of total iterations at which normal supervision starts")
            .locale("training_params.normal_start_fraction")
            .tooltip("training.tooltip.normal_start_fraction")
            .precision(3)
            .ui_step(0.01)
            .all_strategies()
            .float_prop(&OptimizationParameters::normal_end_fraction,
                        "normal_end_fraction", "Normal End Fraction", d.normal_end_fraction, 0.0f, 1.0f,
                        "Fraction of total iterations at which normal supervision stops; 1.0 keeps it on until the end")
            .locale("training_params.normal_end_fraction")
            .tooltip("training.tooltip.normal_end_fraction")
            .precision(3)
            .ui_step(0.01)
            .all_strategies()
            .enum_prop(&OptimizationParameters::normal_loss_space,
                       "normal_loss_space", "Normal Loss Space", d.normal_loss_space,
                       {{"Auto", NormalLossSpace::Auto, "training.options.normal_loss_space.auto", "auto"},
                        {"Camera (OpenCV)", NormalLossSpace::CameraOpenCV,
                         "training.options.normal_loss_space.camera_opencv", "camera-opencv"},
                        {"Camera (OpenGL)", NormalLossSpace::CameraOpenGL,
                         "training.options.normal_loss_space.camera_opengl", "camera-opengl"},
                        {"World", NormalLossSpace::World, "training.options.normal_loss_space.world", "world"}},
                       "Normal prior coordinate space: auto, camera-opencv, camera-opengl, or world")
            .locale("training.advanced.normal_loss_space")
            .tooltip("training.tooltip.normal_loss_space")
            .flags(PROP_ADVANCED)
            .all_strategies()

            // Bilateral grid
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_exposure_correction,
                       "use_exposure_correction", "Exposure correction", d.use_exposure_correction,
                       "Combine a per-photo exposure/white-balance model with a local residual grid")
            .locale("training_params.exposure_correction")
            .tooltip("training.tooltip.exposure_correction")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .int_prop(&OptimizationParameters::exposure_correction_grid_start_iter,
                      "exposure_correction_grid_start_iter", "Grid start iteration",
                      d.exposure_correction_grid_start_iter, 0, 100000,
                      "Iteration at which the local residual grid starts training")
            .locale("training_params.exposure_correction_grid_start")
            .tooltip("training.tooltip.exposure_correction_grid_start")
            .precision(0)
            .ui_step(100)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_bilateral_grid,
                       "use_bilateral_grid", "Bilateral Grid", d.use_bilateral_grid,
                       "Enable bilateral grid color correction")
            .locale("training_params.bilateral_grid")
            .tooltip("training.tooltip.bilateral_grid")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .int_prop(&OptimizationParameters::bilateral_grid_X,
                      "bilateral_grid_x", "Grid X", d.bilateral_grid_X, 4, 64,
                      "Bilateral grid X resolution")
            .json_key("bilateral_grid_X")
            .locale("training.bilateral.grid_x")
            .tooltip("training.tooltip.bilateral_grid_x")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .int_prop(&OptimizationParameters::bilateral_grid_Y,
                      "bilateral_grid_y", "Grid Y", d.bilateral_grid_Y, 4, 64,
                      "Bilateral grid Y resolution")
            .json_key("bilateral_grid_Y")
            .locale("training.bilateral.grid_y")
            .tooltip("training.tooltip.bilateral_grid_y")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .int_prop(&OptimizationParameters::bilateral_grid_W,
                      "bilateral_grid_w", "Grid W", d.bilateral_grid_W, 2, 32,
                      "Bilateral grid intensity bins")
            .json_key("bilateral_grid_W")
            .locale("training.bilateral.grid_w")
            .tooltip("training.tooltip.bilateral_grid_w")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .float_prop(&OptimizationParameters::bilateral_grid_lr,
                        "bilateral_grid_lr", "Grid LR", d.bilateral_grid_lr, 0.0f, 0.1f,
                        "Bilateral grid learning rate")
            .locale("training.bilateral.learning_rate")
            .tooltip("training.tooltip.bilateral_grid_lr")
            .precision(6)
            .ui_step(0.00001)
            .all_strategies()
            .float_prop(&OptimizationParameters::tv_loss_weight,
                        "tv_loss_weight", "TV Loss Weight", d.tv_loss_weight, 0.0f, 100.0f,
                        "Total variation loss weight")
            .locale("training.losses.tv_loss_weight")
            .tooltip("training.tooltip.tv_loss_weight")
            .precision(1)
            .ui_step(0.5)

            // Strategy
            .all_strategies()
            .string_prop(&OptimizationParameters::strategy,
                         "strategy", "Strategy", d.strategy,
                         "Optimization strategy: mcmc, mrnf, or igs+")
            .flags(PROP_NEEDS_RESTART)

            // Shared densification parameters
            .all_strategies()
            .float_prop(&OptimizationParameters::prune_opacity,
                        "prune_opacity", "Prune Opacity", d.prune_opacity, 0.0f, std::numeric_limits<float>::infinity(),
                        "Opacity threshold for pruning")
            .locale("training.thresholds.prune_opacity")
            .tooltip("training.tooltip.prune_opacity")
            .precision(4)
            .ui_step(0.001)
            .strategies({"igs+"})
            .all_strategies()
            .size_prop(&OptimizationParameters::reset_every,
                       "reset_every", "Reset Every", d.reset_every, 100, 10000,
                       "Iteration interval for opacity reset")
            .locale("training.refinement.reset_every")
            .tooltip("training.tooltip.reset_every")
            .precision(0)
            .ui_step(100)
            .strategies({"igs+"})
            // MRNF strategy parameters
            .all_strategies()
            .float_prop(&OptimizationParameters::growth_grad_threshold,
                        "growth_grad_threshold", "Growth Grad Threshold", d.growth_grad_threshold, 0.0f, 1.0f,
                        "Min refine weight for growth candidacy (MRNF)")
            .locale("training.advanced.growth_grad_threshold")
            .tooltip("training.tooltip.growth_grad_threshold")
            .precision(4)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::grow_fraction,
                        "grow_fraction", "Grow Fraction", d.grow_fraction, 0.0f, 1.0f,
                        "Fraction of above-threshold splats to grow (MRNF)")
            .locale("training.advanced.grow_fraction")
            .tooltip("training.tooltip.grow_fraction")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .size_prop(&OptimizationParameters::grow_until_iter,
                       "grow_until_iter", "Grow Until Iter", d.grow_until_iter, 0, 100000,
                       "Stop MRNF growth after this iteration")
            .locale("training.refinement.grow_until_iter")
            .tooltip("training.tooltip.grow_until_iter")
            .precision(0)
            .ui_step(1000)
            .strategies({"mrnf"})
            .all_strategies()
            .bool_prop(&OptimizationParameters::growth_exchange,
                       "growth_exchange", "Growth Exchange", d.growth_exchange,
                       "Keeps error-guided growth working at the splat cap by freeing the splats whose removal changes the image least (MRNF)")
            .locale("training.advanced.growth_exchange")
            .tooltip("training.tooltip.growth_exchange")
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .bool_prop(&OptimizationParameters::growth_visit_normalize,
                       "growth_visit_normalize", "Visit-normalized Growth", d.growth_visit_normalize,
                       "Ranks growth candidates by their mean error signal per visit (steps of the refine window that gave them densification signal) instead of the window maximum, so rarely seen splats compete on equal terms (MRNF)")
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .string_prop(&OptimizationParameters::growth_visit_stat,
                         "growth_visit_stat", "Visit-normalized Growth Statistic", d.growth_visit_stat,
                         "Statistic of visit-normalized growth: mean (sum of the per-visit signals / visits) or weighted (sum of their squares / their sum, led by the closest views) (MRNF)")
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::opacity_decay,
                        "opacity_decay", "Opacity Decay", d.opacity_decay, 0.0f, 0.1f,
                        "Opacity decay rate per refine (MRNF)")
            .locale("training.advanced.opacity_decay")
            .tooltip("training.tooltip.opacity_decay")
            .precision(4)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::scale_decay,
                        "scale_decay", "Scale Decay", d.scale_decay, 0.0f, 0.1f,
                        "Scale decay rate per refine (MRNF)")
            .locale("training.advanced.scale_decay")
            .tooltip("training.tooltip.scale_decay")
            .precision(4)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::means_noise_weight,
                        "means_noise_weight", "Means Noise Weight", d.means_noise_weight, 0.0f, 200.0f,
                        "Exploration noise multiplier for means updates (MRNF)")
            .locale("training.advanced.means_noise_weight")
            .tooltip("training.tooltip.means_noise_weight")
            .precision(2)
            .ui_step(1.0)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .float_prop(&OptimizationParameters::bounds_percentile,
                        "bounds_percentile", "Bounds Percentile", d.bounds_percentile, 0.5f, 1.0f,
                        "Percentile for bounds computation (MRNF)")
            .locale("training.advanced.bounds_percentile")
            .tooltip("training.tooltip.bounds_percentile")
            .precision(3)
            .ui_step(0.01)
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_error_map,
                       "use_error_map", "Error Map", d.use_error_map,
                       "Weight MRNF refine signal by per-pixel SSIM error map")
            .locale("training.advanced.use_error_map")
            .tooltip("training.tooltip.use_error_map")
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})
            .all_strategies()
            .enum_prop(&OptimizationParameters::densify_error_map,
                       "densify_error_map", "Densify Error Map", d.densify_error_map,
                       {{"Ssim", DensifyErrorMap::Ssim, "training.options.densify_error_map.ssim", "ssim"},
                        {"SsimCs", DensifyErrorMap::SsimCs, "training.options.densify_error_map.ssim_cs", "ssim_cs"}},
                       "Densification error map: full SSIM or contrast-structure only")
            .locale("training.advanced.densify_error_map")
            .tooltip("training.tooltip.densify_error_map")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::max_screen_share,
                        "max_screen_share", "Max Screen Share", d.max_screen_share, 0.0f, 1.0f,
                        "Shrink splats that cover more than this share of the view; 0 or 1 disables")
            .locale("training.advanced.max_screen_share")
            .tooltip("training.tooltip.max_screen_share")
            .precision(3)
            .ui_step(0.05)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::screen_share_penalty,
                        "screen_share_penalty", "Screen Share Penalty", d.screen_share_penalty,
                        0.0f, std::numeric_limits<float>::infinity(),
                        "Soft hinge weight on log-scale for splats over the screen-share cap")
            .locale("training.advanced.screen_share_penalty")
            .tooltip("training.tooltip.screen_share_penalty")
            .precision(2)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::screen_share_axis,
                         "screen_share_axis", "Screen Share Axis", d.screen_share_axis,
                         "Splat radius the screen-share cap measures: max (its longest axis, default) or area (the area-equivalent radius of its two longest axes, so long thin splats such as straps and ropes are not shrunk; the shortest axis is never pushed). Standard rasterizer only")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_edge_map,
                       "use_edge_map", "Edge Map", d.use_edge_map,
                       "Weight MRNF refine signal by Canny edge map on GT images")
            .locale("training.advanced.use_edge_map")
            .tooltip("training.tooltip.use_edge_map")
            .flags(PROP_ADVANCED)
            .strategies({"mrnf"})

            // Flags
            .all_strategies()
            .bool_prop(&OptimizationParameters::mip_filter,
                       "mip_filter", "Mip Filter", d.mip_filter,
                       "Enable mip filtering (anti-aliasing)")
            .locale("training_params.mip_filter")
            .tooltip("training.tooltip.mip_filter")
            .all_strategies()
            .bool_prop(&OptimizationParameters::use_ppisp,
                       "ppisp", "PPISP", d.use_ppisp,
                       "Enable per-camera physically plausible image signal processing")
            .json_key("use_ppisp")
            .locale("training_params.ppisp")
            .tooltip("training.tooltip.ppisp")
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_exposure_from_exif,
                       "ppisp_exposure_from_exif", "EXIF Exposure", d.ppisp_exposure_from_exif,
                       "Seed per-frame PPISP exposure from image EXIF")
            .locale("training_params.ppisp_exposure_from_exif")
            .tooltip("training.tooltip.ppisp_exposure_from_exif")
            .all_strategies()
            .float_prop(&OptimizationParameters::ppisp_lr,
                        "ppisp_lr", "PPISP Learning Rate", d.ppisp_lr, 0.0f, 0.1f,
                        "Learning rate for PPISP parameters")
            .locale("training_params.ppisp_lr")
            .tooltip("training.tooltip.ppisp_lr")
            .precision(5)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::ppisp_reg_weight,
                        "ppisp_reg_weight", "PPISP Regularization", d.ppisp_reg_weight, 0.0f, 2.0f,
                        "Regularization weight for PPISP parameters")
            .locale("training_params.ppisp_reg")
            .tooltip("training.tooltip.ppisp_reg")
            .precision(5)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .int_prop(&OptimizationParameters::ppisp_warmup_steps,
                      "ppisp_warmup_steps", "PPISP Warmup Steps", d.ppisp_warmup_steps, 0, 100000,
                      "Steps before PPISP training starts")
            .locale("training_params.ppisp_warmup")
            .tooltip("training.tooltip.ppisp_warmup")
            .precision(0)
            .ui_step(100)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .enum_prop(&OptimizationParameters::ppisp_holdout_appearance,
                       "ppisp_holdout_appearance", "Held-Out Photo Appearance", d.ppisp_holdout_appearance,
                       {{"Mean", PPISPHoldoutAppearance::Mean, "training.options.ppisp_holdout_appearance.mean", "mean"},
                        {"Nearest", PPISPHoldoutAppearance::Nearest, "training.options.ppisp_holdout_appearance.nearest", "nearest"}},
                       "Exposure and white balance used to evaluate held-out photos: nearest blends the training photos shot just before and after, mean uses one average setting")
            .locale("training_params.ppisp_holdout_appearance")
            .tooltip("training.tooltip.ppisp_holdout_appearance")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_use_controller,
                       "ppisp_use_controller", "Controller", d.ppisp_use_controller,
                       "Enable PPISP controller for novel view synthesis")
            .locale("training_params.ppisp_controller")
            .tooltip("training.tooltip.ppisp_controller")
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_freeze_from_sidecar,
                       "ppisp_freeze_from_sidecar", "Freeze From Sidecar", d.ppisp_freeze_from_sidecar,
                       "Load PPISP weights from a sidecar and freeze PPISP learning during training")
            .locale("training_params.ppisp_freeze_from_sidecar")
            .tooltip("training.tooltip.ppisp_freeze_from_sidecar")
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_lock_frames,
                       "ppisp_lock_frames", "Lock Per-Frame Appearance", d.ppisp_lock_frames,
                       "Keep PPISP per-frame exposure and colour at their initial values (identity unless EXIF-seeded): "
                       "no optimizer step, no mean projection; per-camera vignetting and CRF still train. Requires --ppisp")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_lock_colour,
                       "ppisp_lock_colour", "Lock Per-Frame Colour", d.ppisp_lock_colour,
                       "Keep every PPISP per-frame colour latent (white point and primaries: the per-frame white balance) "
                       "at identity, i.e. the footage as shot: no optimizer step. Per-frame exposure, per-camera vignetting "
                       "and CRF still train. Requires --ppisp")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .int_prop(&OptimizationParameters::ppisp_controller_activation_step,
                      "ppisp_controller_activation_step", "Controller Step", d.ppisp_controller_activation_step, -1, 100000,
                      "Iteration to start controller distillation (negative = final 5000 planned steps)")
            .all_strategies()
            .float_prop(&OptimizationParameters::ppisp_controller_lr,
                        "ppisp_controller_lr", "Controller LR", d.ppisp_controller_lr, 1e-5f, 1e-1f,
                        "Learning rate for PPISP controller")
            .locale("training_params.ppisp_controller_lr")
            .tooltip("training.tooltip.ppisp_controller_lr")
            .precision(5)
            .ui_step(0.0001)
            .all_strategies()
            .bool_prop(&OptimizationParameters::ppisp_freeze_gaussians_on_distill,
                       "ppisp_freeze_gaussians", "Freeze Gaussians", d.ppisp_freeze_gaussians_on_distill,
                       "Freeze Gaussians during controller distillation")
            .json_key("ppisp_freeze_gaussians_on_distill")
            .locale("training_params.ppisp_freeze_gaussians")
            .tooltip("training.tooltip.ppisp_freeze_gaussians")
            .all_strategies()
            .bool_prop(&OptimizationParameters::bg_modulation,
                       "bg_modulation", "BG Modulation", d.bg_modulation,
                       "Enable sinusoidal background modulation")
            .all_strategies()
            .bool_prop(&OptimizationParameters::headless,
                       "headless", "Headless", d.headless,
                       "Run without visualization")
            .flags(PROP_READONLY)
            .all_strategies()
            .bool_prop(&OptimizationParameters::enable_eval,
                       "enable_eval", "Enable Eval", d.enable_eval,
                       "Run evaluation at specified steps")
            .locale("training_params.enable_eval")
            .tooltip("training.tooltip.enable_eval")
            .all_strategies()
            .bool_prop(&OptimizationParameters::eval_all,
                       "eval_all", "Eval All", d.eval_all,
                       "Train on every image and evaluate all of them; no image is held out")
            .locale("training_params.eval_all")
            .tooltip("training.tooltip.eval_all")
            .all_strategies()
            .bool_prop(&OptimizationParameters::eval_flip,
                       "eval_flip", "Eval FLIP", d.eval_flip,
                       "Also compute FLIP per evaluated image and save its error map next to the evaluation images")
            .locale("training_params.eval_flip")
            .tooltip("training.tooltip.eval_flip")
            .all_strategies()
            .enum_prop(&OptimizationParameters::eval_space,
                       "eval_space", "Eval Space", d.eval_space,
                       {{"Distorted", EvalSpace::Distorted, "training.options.eval_space.distorted", "distorted"},
                        {"Undistorted", EvalSpace::Undistorted, "training.options.eval_space.undistorted", "undistorted"}},
                       "Reference images for evaluation with --undistort: distorted = the original "
                       "images, with the render warped into the original lens; undistorted = the "
                       "undistorted training images")
            .locale("training_params.eval_space")
            .tooltip("training.tooltip.eval_space")
            .all_strategies()
            .enum_prop(&OptimizationParameters::eval_bit_depth,
                       "eval_bit_depth", "Eval Bit Depth", d.eval_bit_depth,
                       {{"Auto", EvalBitDepth::Auto, "training.options.eval_bit_depth.auto", "auto"},
                        {"8-bit", EvalBitDepth::Eight, "training.options.eval_bit_depth.eight", "8"},
                        {"16-bit", EvalBitDepth::Sixteen, "training.options.eval_bit_depth.sixteen", "16"},
                        {"Float", EvalBitDepth::Float, "training.options.eval_bit_depth.float", "float"}},
                       "Grid the render is quantized to before evaluation metrics: auto = each reference "
                       "image's own encoding (8-bit, 16-bit or float)")
            .locale("training_params.eval_bit_depth")
            .tooltip("training.tooltip.eval_bit_depth")
            .all_strategies()
            .string_prop(&OptimizationParameters::eval_mask,
                         "eval_mask", "Evaluation Mask", d.eval_mask,
                         "Scores only part of each evaluated image (a mesh, a box, the crop box, a mask folder, a depth range, points or a splat); training is not affected")
            .locale("training_params.eval_mask")
            .tooltip("training.tooltip.eval_mask")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .string_prop(&OptimizationParameters::eval_mask_report,
                         "eval_mask_report", "Headline Set", d.eval_mask_report,
                         "Pixel sets scored with the evaluation mask, comma list of mask, unmasked and inverted; the "
                         "first is the headline set, whose scores go to metrics.csv, the log and this panel")
            .locale("training_params.eval_mask_report")
            .tooltip("training.tooltip.eval_mask_report")
            .all_strategies()
            .float_prop(&OptimizationParameters::eval_mask_opacity,
                        "eval_mask_opacity", "Splat Mask Opacity", d.eval_mask_opacity, 0.01f, 1.0f,
                        "Rendered opacity a pixel needs to count as covered by a splat mask; lower widens the mask past the outline, higher pulls it in")
            .locale("training_params.eval_mask_opacity")
            .tooltip("training.tooltip.eval_mask_opacity")
            .precision(2)
            .ui_step(0.05)
            .flags(PROP_NEEDS_RESTART)

            // Random initialization
            .all_strategies()
            .bool_prop(&OptimizationParameters::random,
                       "random", "Random Init", d.random,
                       "Use random initialization instead of SfM")
            .locale("training.init.random_init")
            .tooltip("training.tooltip.random_init")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .int_prop(&OptimizationParameters::init_num_pts,
                      "init_num_pts", "Init Points", d.init_num_pts, 1000, 1000000,
                      "Number of random points to initialize")
            .locale("training.init.num_points")
            .tooltip("training.tooltip.num_points")
            .precision(0)
            .ui_step(10000)
            .all_strategies()
            .float_prop(&OptimizationParameters::init_extent,
                        "init_extent", "Init Extent", d.init_extent, 0.1f, 10.0f,
                        "Extent of random point cloud")
            .locale("training.init.extent")
            .tooltip("training.tooltip.extent")
            .precision(1)
            .ui_step(0.5)
            .all_strategies()
            .int_prop(&OptimizationParameters::init_mesh_resolution,
                      "init_mesh_resolution", "Mesh Init Resolution", d.init_mesh_resolution,
                      Mesh2SplatOptions::kMinResolution, kMaxInitMeshResolution,
                      "Mesh2Splat raster resolution when the init file is a mesh; lowered automatically to fit the splat budget")
            .locale("training.init.mesh_resolution")
            .tooltip("training.tooltip.mesh_resolution")
            .precision(0)
            .ui_step(128)

            // Sparsity
            .all_strategies()
            .int_prop(&OptimizationParameters::popspa_target_count,
                      "popspa_target_count", "Target Gaussians", d.popspa_target_count, 1, 2147483647,
                      "Exact final Gaussian count.")
            .locale("training_params.popspa_target_count")
            .tooltip("training.tooltip.popspa_target_count")
            .precision(0)
            .ui_step(1000)
            .all_strategies()
            .int_prop(&OptimizationParameters::popspa_first_prune_count,
                      "popspa_first_prune_count", "First Prune Count", d.popspa_first_prune_count, 0, 2147483647,
                      "Intermediate Gaussian count; zero chooses the geometric-mean budget automatically.")
            .locale("training_params.popspa_first_prune_count")
            .tooltip("training.tooltip.popspa_first_prune_count")
            .precision(0)
            .ui_step(1000)
            .all_strategies()
            .int_prop(&OptimizationParameters::popspa_sparsify_steps,
                      "popspa_sparsify_steps", "Sparsify Steps", d.popspa_sparsify_steps, 0, 2147483647,
                      "GaussianSpa and shape optimization steps after the first prune.")
            .locale("training_params.popspa_sparsify_steps")
            .tooltip("training.tooltip.popspa_sparsify_steps")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .int_prop(&OptimizationParameters::popspa_refine_steps,
                      "popspa_refine_steps", "Recovery Steps", d.popspa_refine_steps, 0, 2147483647,
                      "Photometric recovery steps after the final prune.")
            .locale("training_params.popspa_refine_steps")
            .tooltip("training.tooltip.popspa_refine_steps")
            .precision(0)
            .ui_step(100)
            .all_strategies()
            .int_prop(&OptimizationParameters::popspa_projection_interval,
                      "popspa_projection_interval", "Projection Interval", d.popspa_projection_interval, 1, 2147483647,
                      "Number of sparse steps between projection and dual updates.")
            .locale("training_params.popspa_projection_interval")
            .tooltip("training.tooltip.popspa_projection_interval")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .int_prop(&OptimizationParameters::popspa_seed,
                      "popspa_seed", "Seed", d.popspa_seed, 0, 2147483647,
                      "Random seed for POPSpa optimization.")
            .locale("training_params.popspa_seed")
            .tooltip("training.tooltip.popspa_seed")
            .precision(0)
            .ui_step(1)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_rho,
                        "popspa_rho", "ADMM Penalty", d.popspa_rho, 1e-12f, 1.0f,
                        "GaussianSpa penalty coefficient.")
            .locale("training_params.popspa_rho")
            .tooltip("training.tooltip.popspa_rho")
            .precision(8)
            .ui_step(0.00005)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_erank_weight,
                        "popspa_erank_weight", "Effective Rank Weight", d.popspa_erank_weight, 0.0f, 100.0f,
                        "Weight of the effective-rank shape regularizer.")
            .locale("training_params.popspa_erank_weight")
            .tooltip("training.tooltip.popspa_erank_weight")
            .precision(8)
            .ui_step(0.001)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_thin_scale_weight,
                        "popspa_thin_scale_weight", "Thin Scale Weight", d.popspa_thin_scale_weight, 0.0f, 100.0f,
                        "Weight of the thin-scale shape regularizer.")
            .locale("training_params.popspa_thin_scale_weight")
            .tooltip("training.tooltip.popspa_thin_scale_weight")
            .precision(8)
            .ui_step(0.1)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_erank_epsilon,
                        "popspa_erank_epsilon", "Effective Rank Epsilon", d.popspa_erank_epsilon, 1e-12f, 1.0f,
                        "Positive numerical epsilon for effective-rank normalization.")
            .locale("training_params.popspa_erank_epsilon")
            .tooltip("training.tooltip.popspa_erank_epsilon")
            .precision(8)
            .ui_step(1e-8)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_means_lr,
                        "popspa_means_lr", "Position Learning Rate", d.popspa_means_lr, 0.0f, 1.0f,
                        "Position learning rate for POPSpa optimization.")
            .locale("training_params.popspa_means_lr")
            .tooltip("training.tooltip.popspa_means_lr")
            .precision(8)
            .ui_step(1.6e-7)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_scales_lr,
                        "popspa_scales_lr", "Scale Learning Rate", d.popspa_scales_lr, 0.0f, 1.0f,
                        "Scale learning rate for POPSpa optimization.")
            .locale("training_params.popspa_scales_lr")
            .tooltip("training.tooltip.popspa_scales_lr")
            .precision(8)
            .ui_step(0.0005)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_opacities_lr,
                        "popspa_opacities_lr", "Opacity Learning Rate", d.popspa_opacities_lr, 0.0f, 1.0f,
                        "Opacity learning rate for POPSpa optimization.")
            .locale("training_params.popspa_opacities_lr")
            .tooltip("training.tooltip.popspa_opacities_lr")
            .precision(8)
            .ui_step(0.005)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_quaternions_lr,
                        "popspa_quaternions_lr", "Rotation Learning Rate", d.popspa_quaternions_lr, 0.0f, 1.0f,
                        "Quaternion learning rate for POPSpa optimization.")
            .locale("training_params.popspa_quaternions_lr")
            .tooltip("training.tooltip.popspa_quaternions_lr")
            .precision(8)
            .ui_step(0.0001)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_sh0_lr,
                        "popspa_sh0_lr", "Base Color Learning Rate", d.popspa_sh0_lr, 0.0f, 1.0f,
                        "Degree-zero spherical-harmonic learning rate.")
            .locale("training_params.popspa_sh0_lr")
            .tooltip("training.tooltip.popspa_sh0_lr")
            .precision(8)
            .ui_step(0.00025)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_shn_lr,
                        "popspa_shn_lr", "Higher Color Learning Rate", d.popspa_shn_lr, 0.0f, 1.0f,
                        "Higher-degree spherical-harmonic learning rate.")
            .locale("training_params.popspa_shn_lr")
            .tooltip("training.tooltip.popspa_shn_lr")
            .precision(8)
            .ui_step(0.0000125)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::popspa_ssim_weight,
                        "popspa_ssim_weight", "SSIM Weight", d.popspa_ssim_weight, 0.0f, 1.0f,
                        "SSIM fraction in the POPSpa photometric loss; the remainder uses L1.")
            .locale("training_params.popspa_ssim_weight")
            .tooltip("training.tooltip.popspa_ssim_weight")
            .precision(8)
            .ui_step(0.02)
            .all_strategies()
            .bool_prop(&OptimizationParameters::enable_sparsity,
                       "enable_sparsity", "Enable Sparsity", d.enable_sparsity,
                       "Run POPSpa sparsification after training")
            .locale("training_params.sparsity")
            .tooltip("training.tooltip.sparsity")
            .all_strategies()
            .float_prop(&OptimizationParameters::steps_scaler,
                        "steps_scaler", "Steps Scaler", d.steps_scaler, 0.0f, 10.0f,
                        "Scale training step counts")
            .locale("training_params.steps_scaler")
            .tooltip("training.tooltip.steps_scaler")
            .precision(2)
            .ui_step(0.1)
            .flags(PROP_ADVANCED)
            .all_strategies()
            .bool_prop(&OptimizationParameters::gut,
                       "gut", "GUT", d.gut,
                       "Gaussian Unscented Transform")
            .locale("training_params.gut")
            .tooltip("training.tooltip.gut")
            .all_strategies()
            .bool_prop(&OptimizationParameters::undistort,
                       "undistort", "Undistort", d.undistort,
                       "Remove lens distortion before training: each image and its mask, depth and "
                       "normal map are resampled once from full resolution into a distortion-free "
                       "pinhole camera, which training then uses. Alternative to --gut for distorted "
                       "or non-pinhole cameras")
            .locale("training_params.undistort")
            .tooltip("training.tooltip.undistort")
            .flags(PROP_NEEDS_RESTART)
            .all_strategies()
            .enum_prop(&OptimizationParameters::bg_mode,
                       "bg_mode", "Background Mode", d.bg_mode,
                       {{"SolidColor", BackgroundMode::SolidColor, "training.options.bg.color", "solid_color"},
                        {"Modulation", BackgroundMode::Modulation, "training.options.bg.modulation", "modulation"},
                        {"Image", BackgroundMode::Image, "training.options.bg.image", "image"},
                        {"Random", BackgroundMode::Random, "training.options.bg.random", "random"},
                        {"Cubemap", BackgroundMode::Cubemap, "training.options.bg.cubemap", "cubemap"}},
                       "Background mode")
            .locale("training_params.bg_mode")
            .tooltip("training.tooltip.bg_modulation")
            .all_strategies()
            .int_prop(&OptimizationParameters::bg_cubemap_res,
                      "bg_cubemap_res", "Background Cubemap Resolution", d.bg_cubemap_res, 4, 4096,
                      "--bg-mode cubemap: texels per cube face edge (6 faces x res x res RGB)")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::bg_cubemap_lr,
                        "bg_cubemap_lr", "Background Cubemap LR", d.bg_cubemap_lr, 0.0f, 1.0f,
                        "--bg-mode cubemap: Adam learning rate of the cubemap texels at iteration 0")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::bg_cubemap_lr_final,
                        "bg_cubemap_lr_final", "Background Cubemap Final LR Factor", d.bg_cubemap_lr_final, 0.0f, 1.0f,
                        "--bg-mode cubemap: the learning rate decays exponentially to lr * this factor at the last iteration")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .float_prop(&OptimizationParameters::bg_cubemap_smooth,
                        "bg_cubemap_smooth", "Background Cubemap Smoothness", d.bg_cubemap_smooth, 0.0f, 10000.0f,
                        "--bg-mode cubemap: weight of the mean squared difference of neighbouring texels (keeps the "
                        "cubemap sky-like and low-frequency; 0 = off)")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .string_prop(&OptimizationParameters::bg_cubemap_init,
                         "bg_cubemap_init", "Background Cubemap Init", d.bg_cubemap_init,
                         "--bg-mode cubemap: start the cubemap from this faces.npy (float32 [6, R, R, 3], faces +X -X +Y -Y "
                         "+Z -Z as the export writes them, resampled if R differs) instead of the clear colour")
            .flags(PROP_ADVANCED)
            .all_strategies()
            .build();
    }

    namespace {
        std::mutex optimization_registration_mutex;
    }

    void ensure_optimization_properties_registered() {
        auto& registry = PropertyRegistry::instance();
        if (registry.get_group_snapshot("optimization"))
            return;

        std::lock_guard lock(optimization_registration_mutex);
        if (!registry.get_group_snapshot("optimization"))
            register_optimization_properties();
    }

} // namespace lfs::core::param
