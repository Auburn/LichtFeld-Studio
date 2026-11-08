/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// LFQPLCYKMO (8 Nov 2026): the quad the GAME (fpvREAL's fpv-render, D:/fpvreal/repo/rust/crates/fpv-render) draws for a splat, and
// its derivatives, for --loss-footprint (the game's draw cost per view: mean splats drawn per pixel).
//   src/shaders/project.wgsl: the dilated 2D covariance (a b; b c) (+0.3 px), its eigenvalues l1 >= l2, an eigen-aligned quad of
//     half-axes s_k = min(max_pixel_radius = 512 px, reach x sqrt(l_k)): area A = (2 s1) (2 s2) = 4 s1 s2.
//   src/shaders/kernel_shape.wgsl kernel_reach: a Gaussian (shape code 0 = beta 2) reaches the cut C = sqrt 5 (fly --max-std-dev);
//     a shaped splat reaches min(C, d_c), d_c = (2 ln(o / min alpha))^(1/beta) its alpha-floor contour, min alpha 0.5/255.
// The same contour rule as tools/LFQPLCYKMA's beta-aware screen-share cap (screen_share.cuh screen_share_beta_footprint =
// game_contour_reach(o, beta) / game_contour_reach(o, 2)).

#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define LFS_GF_HD __host__ __device__
#else
#define LFS_GF_HD
#endif

namespace lfs::training {

    inline constexpr float kGameCut = 2.2360679775f;          // sqrt 5: fpv-render's / Spark's splat cut (fly --max-std-dev)
    inline constexpr float kGameInvMinAlpha = 510.0f;         // 1 / (0.5 / 255): fpv-render's alpha floor (fly --min-alpha)
    // fpv-render's per-axis quad cap, 512 px (frame.max_pixel_radius), as an angle: 512 px in the 2560-wide 119-degree game view
    // (1280 / radians(59.5) = 1232.58 px per radian) = 0.41539 rad; times a view's fx it is the cap in that view's pixels.
    inline constexpr float kGameMaxPixelRadiusPerFx = 0.41539267f;

    // min(C, d_c(beta, o)); 0 when the splat is under the alpha floor everywhere (o <= 0.5/255: the game culls it)
    [[nodiscard]] LFS_GF_HD inline float game_contour_reach(const float opacity, const float beta) {
        const float two_p = 2.0f * logf(opacity * kGameInvMinAlpha);
        if (!(two_p > 0.0f))
            return 0.0f;
        return fminf(kGameCut, beta == 2.0f ? sqrtf(two_p) : exp2f(log2f(two_p) / beta));
    }

    // The reach of the game's quad, in standard deviations: a Gaussian (beta exactly 2: no --kernel-shape, or shape code 0) the cut
    // C whatever its opacity; a shaped splat min(C, d_c). 0 = not drawn.
    [[nodiscard]] LFS_GF_HD inline float game_quad_reach(const float opacity, const float beta) {
        if (!(opacity * kGameInvMinAlpha > 1.0f))
            return 0.0f;
        return beta == 2.0f ? kGameCut : game_contour_reach(opacity, beta);
    }

    struct GameFootprint {
        float area = 0.0f;              // 4 s1 s2, px^2 of the view
        float d_a = 0.0f, d_b = 0.0f, d_c = 0.0f; // dA / d(a, b, c) of the dilated 2D covariance (b the off-diagonal entry)
        float d_raw_opacity = 0.0f;     // dA / d(raw opacity) through the reach (shaped splats whose contour is inside the cut)
        bool capped1 = false, capped2 = false;
    };

    // a, b, c: the dilated 2D covariance in the view's pixels; max_px: the per-axis cap in those pixels.
    [[nodiscard]] LFS_GF_HD inline GameFootprint game_footprint(const float a, const float b, const float c, const float opacity,
                                                                const float beta, const float max_px, const bool want_opacity) {
        GameFootprint f{};
        const float R = game_quad_reach(opacity, beta);
        const float det = a * c - b * b;
        if (!(R > 0.0f) || !(det > 0.0f) || !(a > 0.0f) || !(c > 0.0f))
            return f;
        const float avg = 0.5f * (a + c);
        const float delta = sqrtf(fmaxf(avg * avg - det, 0.0f));
        const float l1 = avg + delta, l2 = fmaxf(avg - delta, 1e-12f);
        const float r1 = sqrtf(l1), r2 = sqrtf(l2);
        const float s1 = R * r1, s2 = R * r2;
        f.capped1 = !(s1 < max_px);
        f.capped2 = !(s2 < max_px);
        const float S1 = f.capped1 ? max_px : s1, S2 = f.capped2 ? max_px : s2;
        f.area = 4.0f * S1 * S2;
        if (!f.capped1 && !f.capped2) {
            // A = 4 R^2 sqrt(det): smooth at l1 = l2
            const float k = 2.0f * R * R / sqrtf(det);
            f.d_a = k * c;
            f.d_c = k * a;
            f.d_b = -2.0f * k * b;
        } else if (!(f.capped1 && f.capped2) && delta > 0.0f) {
            // one axis at the cap: A = 4 S_capped s_free, through the free axis's eigenvalue
            const float dA_dl1 = f.capped1 ? 0.0f : 2.0f * S2 * R / r1;
            const float dA_dl2 = f.capped2 ? 0.0f : 2.0f * S1 * R / r2;
            const float t = (a - c) / (4.0f * delta), u = b / delta;
            f.d_a = dA_dl1 * (0.5f + t) + dA_dl2 * (0.5f - t);
            f.d_c = dA_dl1 * (0.5f - t) + dA_dl2 * (0.5f + t);
            f.d_b = (dA_dl1 - dA_dl2) * u;
        }
        if (want_opacity && beta != 2.0f && R < kGameCut) {
            // R = (2 ln(510 o))^(1/beta): dR / d(raw opacity) = 2 R (1 - o) / (beta 2 ln(510 o)); dA/dR over the free axes
            const float two_p = 2.0f * logf(opacity * kGameInvMinAlpha);
            const float dA_dR = 4.0f * ((f.capped1 ? 0.0f : r1) * S2 + S1 * (f.capped2 ? 0.0f : r2));
            f.d_raw_opacity = dA_dR * 2.0f * R * (1.0f - opacity) / (beta * two_p);
        }
        return f;
    }

} // namespace lfs::training

#undef LFS_GF_HD
