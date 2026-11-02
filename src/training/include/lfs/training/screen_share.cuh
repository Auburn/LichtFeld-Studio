/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace lfs::training {

    // Which radius of a splat the screen-share cap measures (--screen-share-axis, OptimizationParameters::
    // screen_share_axis). One definition for every user: the forward share collection (FastGS preprocess), the
    // optimizer's hinge on the log-scales (fused and unfused Adam) and the MCMC clip.
    //   kScreenShareAxisMax (0, default): the LONGEST axis, exp(max log-scale). The hinge pushes all three
    //     log-scales (unchanged behaviour), the MCMC clip shrinks the longest axis.
    //   kScreenShareAxisArea (1): the area-equivalent radius of the two longest axes, sqrt(s1 s2) = exp(mean of
    //     the two largest log-scales). A long thin splat (strap, rope, gate bar) no longer counts as big. Only the
    //     two axes that enter that radius are pushed (hinge) or shrunk (clip); the shortest axis is left alone.
    // The GUT path measures a projected bounding-box area instead; there the axis setting is not used (max).
    inline constexpr int kScreenShareAxisMax = 0;
    inline constexpr int kScreenShareAxisArea = 1;

    // Index of the log-scale axis that the cap leaves alone: the shortest one (the first on ties) with
    // kScreenShareAxisArea and a 3-wide row, else -1 (every axis is pushed). Read the row before updating it.
    [[nodiscard]]
#ifdef __CUDACC__
    __host__ __device__
#endif
        inline int screen_share_unpushed_axis(const int axis_mode, const float log_s0, const float log_s1,
                                              const float log_s2) {
        if (axis_mode != kScreenShareAxisArea)
            return -1;
        return (log_s0 <= log_s1 && log_s0 <= log_s2) ? 0 : (log_s1 <= log_s2 ? 1 : 2);
    }

    // Dimensionless angular / screen-share of a 3D Gaussian. Saturates toward 1
    // when the splat swallows the camera (d ≲ r). Host and device.
    // axis_mode: kScreenShareAxisMax (radius = longest axis) or kScreenShareAxisArea (radius = sqrt of the
    // product of the two longest axes, i.e. log radius = mean of the two largest log-scales).
    [[nodiscard]]
#ifdef __CUDACC__
    __host__ __device__
#endif
        inline float gaussian_screen_share(
            const float mean_x,
            const float mean_y,
            const float mean_z,
            const float cam_x,
            const float cam_y,
            const float cam_z,
            const float log_sx,
            const float log_sy,
            const float log_sz,
            const float opacity_raw,
            const int axis_mode = kScreenShareAxisMax) {
#ifdef __CUDA_ARCH__
        const float max_log = axis_mode == kScreenShareAxisArea
                                  ? 0.5f * (log_sx + log_sy + log_sz - fminf(log_sx, fminf(log_sy, log_sz)))
                                  : fmaxf(log_sx, fmaxf(log_sy, log_sz));
        const float opacity = 1.0f / (1.0f + expf(-opacity_raw));
        const float extend = sqrtf(2.0f * logf(fmaxf(255.0f * opacity, 1.0f)));
        const float r = expf(max_log) * extend;
        const float dx = mean_x - cam_x;
        const float dy = mean_y - cam_y;
        const float dz = mean_z - cam_z;
        const float d = sqrtf(dx * dx + dy * dy + dz * dz);
        const float denom = fmaxf(d, r) + sqrtf(fmaxf(d * d - r * r, 0.0f));
        if (!(denom > 0.0f) || !(r > 0.0f))
            return 0.0f;
        return fminf(fmaxf(r / denom, 0.0f), 1.0f);
#else
        const float max_log = axis_mode == kScreenShareAxisArea
                                  ? 0.5f * (log_sx + log_sy + log_sz - std::fmin(log_sx, std::fmin(log_sy, log_sz)))
                                  : std::fmax(log_sx, std::fmax(log_sy, log_sz));
        const float opacity = 1.0f / (1.0f + std::exp(-opacity_raw));
        const float extend = std::sqrt(2.0f * std::log(std::fmax(255.0f * opacity, 1.0f)));
        const float r = std::exp(max_log) * extend;
        const float dx = mean_x - cam_x;
        const float dy = mean_y - cam_y;
        const float dz = mean_z - cam_z;
        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float denom = std::fmax(d, r) + std::sqrt(std::fmax(d * d - r * r, 0.0f));
        if (!(denom > 0.0f) || !(r > 0.0f))
            return 0.0f;
        const float share = r / denom;
        return share < 0.0f ? 0.0f : (share > 1.0f ? 1.0f : share);
#endif
    }

    [[nodiscard]]
#ifdef __CUDACC__
    __host__ __device__
#endif
        inline bool screen_share_cap_active(const float max_screen_share) {
        return max_screen_share > 0.0f && max_screen_share < 1.0f;
    }

#ifdef __CUDACC__
    __device__ __forceinline__ void atomic_max_float(float* addr, const float val) {
        int* const addr_i = reinterpret_cast<int*>(addr);
        int old = *addr_i;
        while (true) {
            const float old_f = __int_as_float(old);
            if (old_f >= val)
                return;
            const int assumed = old;
            old = atomicCAS(addr_i, assumed, __float_as_int(val));
            if (old == assumed)
                return;
        }
    }

    // Extra gradient added to ONE log-scale entry of a splat over the cap: a push of constant size in log space
    // (penalty * log2(share / limit) after Adam's normalisation). The caller applies it only to the axes that
    // enter the measured radius: all three with kScreenShareAxisMax (unchanged), the two longest with
    // kScreenShareAxisArea (skip i == screen_share_unpushed_axis(...)). Pushing both of those by the same step
    // lowers the area radius at the rate the max mode lowers the longest axis.
    __device__ __forceinline__ float screen_share_hinge_extra_grad(
        const float share,
        const float limit,
        const float penalty,
        const float old_v,
        const float bias_correction2_sqrt_rcp,
        const float eps) {
        if (!(limit > 0.0f) || !(limit < 1.0f) || !(share > limit) || !(penalty > 0.0f))
            return 0.0f;
        const float hinge = penalty * log2f(share / limit);
        const float denom = sqrtf(old_v) * bias_correction2_sqrt_rcp + eps;
        return hinge * denom;
    }
#endif

} // namespace lfs::training
