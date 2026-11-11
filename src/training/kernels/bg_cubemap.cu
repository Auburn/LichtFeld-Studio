/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// --bg-mode cubemap (LFC): a learned environment cubemap behind the splats.
//
// Forward: each pixel's background is the cubemap, bilinearly sampled in the world direction of the pixel's ray,
//   bg(p) = cube(d(p)),   composited by the rasterizer as  image = splats + (1 - alpha) * bg   (the per-pixel background
//   image path the rasterizer already has; its backward already gives the splats d/d(alpha) through bg).
// The ray goes through the camera's own lens: pixel centre (x + 0.5, y + 0.5) -> m = ((u - cx) / fx, (v - cy) / fy);
//   pinhole: ray = (m, 1);  OPENCV_FISHEYE: theta from theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8) = |m| (Newton,
//   theta <= theta_max, where the polynomial is monotonic), ray = (sin theta m / |m|, cos theta).
//   LFQPLC (--lens-radial): for the radial lens table's camera theta inverts the table's theta_d = theta P(theta^2) instead
//   (the rasterizer's own lens and culling bound), in separate RADIAL kernel instantiations; R below is the refined pose
//   under --pose-opt (the transform the rasterizer draws the splats with).
// Dynamic rolling shutter (FastGS's model, rolling_shutter.cuh): the row read at s = clamp((v + row_offset) / H - 1/2,
//   -1/2, 1/2) is seen through the pose at t = s tau, camera-space p(t) = E (p_mid - d), E = exp(-[r(t)]x),
//   r(t) = omega t + alpha t^2 / 2. A direction does not see d, so the pixel's ray in the mid-readout camera is E^T ray,
//   and in world axes R^T E^T ray (R = the rows of the world-to-camera matrix).
// Backward: dL/d(texel) = sum over pixels of (1 - alpha) * dL/d(image) * bilinear weight. Warp-aggregated atomics: the
//   lanes of a warp (32 neighbouring pixels of one row) that sample the same 4 texels add their values first
//   (__match_any_sync + a peer reduction) and one lane does the atomics.

#include "bg_cubemap.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"

#include <cmath>

namespace lfs::training::kernels {
    namespace {

        __device__ __forceinline__ void dir_to_face(const float3 d, int& f, float& s, float& t) {
            const float ax = fabsf(d.x), ay = fabsf(d.y), az = fabsf(d.z);
            float ma, sc, tc;
            if (ax >= ay && ax >= az) {
                ma = ax;
                if (d.x > 0.0f) {
                    f = 0;
                    sc = -d.z;
                } else {
                    f = 1;
                    sc = d.z;
                }
                tc = -d.y;
            } else if (ay >= az) {
                ma = ay;
                if (d.y > 0.0f) {
                    f = 2;
                    tc = d.z;
                } else {
                    f = 3;
                    tc = -d.z;
                }
                sc = d.x;
            } else {
                ma = az;
                if (d.z > 0.0f) {
                    f = 4;
                    sc = d.x;
                } else {
                    f = 5;
                    sc = -d.x;
                }
                tc = -d.y;
            }
            const float inv = 1.0f / fmaxf(ma, 1e-30f);
            s = fminf(fmaxf(sc * inv, -1.0f), 1.0f);
            t = fminf(fmaxf(tc * inv, -1.0f), 1.0f);
        }

        __device__ __forceinline__ float3 face_to_dir(const int f, const float s, const float t) {
            switch (f) {
            case 0: return make_float3(1.0f, -t, -s);
            case 1: return make_float3(-1.0f, -t, s);
            case 2: return make_float3(s, 1.0f, t);
            case 3: return make_float3(s, -1.0f, -t);
            case 4: return make_float3(s, -t, 1.0f);
            default: return make_float3(-s, -t, -1.0f);
            }
        }

        // Texel (face f, column i, row j); i or j may be one outside [0, R - 1]: then the texel of the adjacent face that
        // the direction of that virtual texel centre falls in.
        __device__ __forceinline__ int texel_index(const int f, const int i, const int j, const int R) {
            if (i >= 0 && i < R && j >= 0 && j < R)
                return (f * R + j) * R + i;
            const float s = (2.0f * static_cast<float>(i) + 1.0f) / static_cast<float>(R) - 1.0f;
            const float t = (2.0f * static_cast<float>(j) + 1.0f) / static_cast<float>(R) - 1.0f;
            int f2;
            float s2, t2;
            dir_to_face(face_to_dir(f, s, t), f2, s2, t2);
            const int i2 = min(max(static_cast<int>(floorf((s2 + 1.0f) * 0.5f * static_cast<float>(R))), 0), R - 1);
            const int j2 = min(max(static_cast<int>(floorf((t2 + 1.0f) * 0.5f * static_cast<float>(R))), 0), R - 1);
            return (f2 * R + j2) * R + i2;
        }

        struct Bilinear {
            int idx[4];  // texel indices (00, 10, 01, 11)
            float w[4];  // weights
            int key;     // identifies the 4 texels (face, i0, j0)
        };

        __device__ __forceinline__ Bilinear bilinear(const float3 d, const int R) {
            int f;
            float s, t;
            dir_to_face(d, f, s, t);
            const float x = (s + 1.0f) * 0.5f * static_cast<float>(R) - 0.5f;
            const float y = (t + 1.0f) * 0.5f * static_cast<float>(R) - 0.5f;
            const float x0 = floorf(x), y0 = floorf(y);
            const float wx = x - x0, wy = y - y0;
            const int i0 = static_cast<int>(x0), j0 = static_cast<int>(y0);
            Bilinear b;
            b.idx[0] = texel_index(f, i0, j0, R);
            b.idx[1] = texel_index(f, i0 + 1, j0, R);
            b.idx[2] = texel_index(f, i0, j0 + 1, R);
            b.idx[3] = texel_index(f, i0 + 1, j0 + 1, R);
            b.w[0] = (1.0f - wx) * (1.0f - wy);
            b.w[1] = wx * (1.0f - wy);
            b.w[2] = (1.0f - wx) * wy;
            b.w[3] = wx * wy;
            b.key = (f * (R + 2) + (j0 + 1)) * (R + 2) + (i0 + 1);
            return b;
        }

        // E = exp([phi]x) as rows (rolling_shutter.cuh's exp_so3).
        __device__ __forceinline__ void exp_so3(const float3 phi, float3 E[3]) {
            const float th2 = phi.x * phi.x + phi.y * phi.y + phi.z * phi.z;
            float A, B;
            if (th2 < 1e-8f) {
                A = 1.0f - th2 * (1.0f / 6.0f);
                B = 0.5f - th2 * (1.0f / 24.0f);
            } else {
                const float th = sqrtf(th2);
                A = sinf(th) / th;
                B = (1.0f - cosf(th)) / th2;
            }
            const float x = phi.x, y = phi.y, z = phi.z;
            E[0] = make_float3(1.0f + B * (x * x - th2), B * x * y - A * z, B * x * z + A * y);
            E[1] = make_float3(B * x * y + A * z, 1.0f + B * (y * y - th2), B * y * z - A * x);
            E[2] = make_float3(B * x * z - A * y, B * y * z + A * x, 1.0f + B * (z * z - th2));
        }

        // World direction (not normalised) of the ray through the centre of pixel (px, py).
        // RADIAL (LFQPLC, --lens-radial): the fisheye angle inverts the radial lens table's theta_d = theta P(theta^2) (the
        // rasterizer's own lens, core/radial_lens.h) instead of the k-polynomial; RADIAL = false is the original code.
        template <bool RADIAL>
        __device__ __forceinline__ float3 pixel_dir(const CubemapView& v, const int px, const int py) {
            const float u = static_cast<float>(px) + 0.5f;
            const float vv = static_cast<float>(py) + 0.5f;
            const float mx = (u - v.cx) / v.fx;
            const float my = (vv - v.cy) / v.fy;
            float3 rc;
            if (v.fisheye) {
                const float rd = sqrtf(mx * mx + my * my);
                if (rd < 1e-9f) {
                    rc = make_float3(0.0f, 0.0f, 1.0f);
                } else {
                    float th = fminf(rd, v.theta_max);
                    if constexpr (RADIAL) {
#pragma unroll
                        for (int k = 0; k < 10; ++k) {
                            const float t2 = th * th;
                            float P, P1, P2;
                            lfs::core::radial_lens_P(v.radial, t2, P, P1, P2);
                            (void)P2;
                            // d(theta P(theta^2)) / d theta = P + 2 theta^2 dP/du
                            th -= (th * P - rd) / fmaxf(P + 2.0f * t2 * P1, 1e-6f);
                            th = fminf(fmaxf(th, 0.0f), v.theta_max);
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < 10; ++k) {
                            const float t2 = th * th;
                            const float P = 1.0f + t2 * (v.k1 + t2 * (v.k2 + t2 * (v.k3 + t2 * v.k4)));
                            const float dP = 1.0f + t2 * (3.0f * v.k1 + t2 * (5.0f * v.k2 + t2 * (7.0f * v.k3 + t2 * 9.0f * v.k4)));
                            th -= (th * P - rd) / fmaxf(dP, 1e-6f);
                            th = fminf(fmaxf(th, 0.0f), v.theta_max);
                        }
                    }
                    const float sn = sinf(th) / rd;
                    rc = make_float3(mx * sn, my * sn, cosf(th));
                }
            } else {
                rc = make_float3(mx, my, 1.0f);
            }
            if (v.motion != nullptr) {
                const float H = v.sensor_height > 0.0f ? v.sensor_height : static_cast<float>(v.height);
                float s;
                if (v.time_radial != 0) { // --rs-time-map (LFQPLCYKMR), radial form: this pixel's own time (FastGS's time_map_s)
                    const float x = 2.0f * fminf(fmaxf(u / static_cast<float>(v.width), 0.0f), 1.0f) - 1.0f - v.tr_cx;
                    const float y = (2.0f * fminf(fmaxf((vv + v.row_offset) / H, 0.0f), 1.0f) - 1.0f) * v.tr_aspect - v.tr_cy;
                    const float r2 = x * x + y * y;
                    s = v.tr_k * y * (1.0f + r2 * (v.tr_a + r2 * (v.tr_b + r2 * v.tr_c))) - v.tr_s0;
                } else if (v.time_map != nullptr) { // --rs-time-map, grid form: this pixel's own time (FastGS's time_map_s)
                    const int nx = v.time_map_nx, ny = v.time_map_ny;
                    const float gx = fminf(fmaxf(u / static_cast<float>(v.width), 0.0f), 1.0f) * static_cast<float>(nx - 1);
                    const float gy = fminf(fmaxf((vv + v.row_offset) / H, 0.0f), 1.0f) * static_cast<float>(ny - 1);
                    const int i0 = min(static_cast<int>(gx), nx - 2);
                    const int j0 = min(static_cast<int>(gy), ny - 2);
                    const float ax = gx - static_cast<float>(i0);
                    const float ay = gy - static_cast<float>(j0);
                    const float* r0 = v.time_map + static_cast<size_t>(j0) * static_cast<size_t>(nx) + static_cast<size_t>(i0);
                    const float* r1 = r0 + nx;
                    const float s00 = __ldg(r0), s10 = __ldg(r0 + 1), s01 = __ldg(r1), s11 = __ldg(r1 + 1);
                    const float a = s00 + ax * (s10 - s00);
                    const float b = s01 + ax * (s11 - s01);
                    s = a + ay * (b - a);
                } else {
                    s = fminf(fmaxf((vv + v.row_offset) / H - 0.5f, -0.5f), 0.5f);
                }
                const float t = s * __ldg(v.readout);
                const float h = 0.5f * t * t;
                const float3 r = make_float3(__ldg(v.motion + 0) * t + __ldg(v.motion + 3) * h,
                                             __ldg(v.motion + 1) * t + __ldg(v.motion + 4) * h,
                                             __ldg(v.motion + 2) * t + __ldg(v.motion + 5) * h);
                float3 E[3];
                exp_so3(make_float3(-r.x, -r.y, -r.z), E);
                // E^T rc
                rc = make_float3(E[0].x * rc.x + E[1].x * rc.y + E[2].x * rc.z,
                                 E[0].y * rc.x + E[1].y * rc.y + E[2].y * rc.z,
                                 E[0].z * rc.x + E[1].z * rc.y + E[2].z * rc.z);
            }
            // R^T rc, R = rows 0..2 (first three columns) of the row-major world-to-camera matrix
            const float* W = v.w2c;
            return make_float3(__ldg(W + 0) * rc.x + __ldg(W + 4) * rc.y + __ldg(W + 8) * rc.z,
                               __ldg(W + 1) * rc.x + __ldg(W + 5) * rc.y + __ldg(W + 9) * rc.z,
                               __ldg(W + 2) * rc.x + __ldg(W + 6) * rc.y + __ldg(W + 10) * rc.z);
        }

        // LFQPLC: the per-pixel world directions, for lfe-fisheye cuberaycheck (normalised)
        template <bool RADIAL>
        __global__ void dirs_kernel(const CubemapView v, float* __restrict__ dirs) {
            const int px = blockIdx.x * blockDim.x + threadIdx.x;
            const int py = blockIdx.y * blockDim.y + threadIdx.y;
            if (px >= v.width || py >= v.height)
                return;
            const float3 d = pixel_dir<RADIAL>(v, px, py);
            const float in = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
            float* o = dirs + (static_cast<size_t>(py) * static_cast<size_t>(v.width) + static_cast<size_t>(px)) * 3;
            o[0] = d.x * in;
            o[1] = d.y * in;
            o[2] = d.z * in;
        }

        template <bool RADIAL>
        __global__ void render_kernel(const float* __restrict__ texels, const int R, const CubemapView v,
                                      float* __restrict__ bg) {
            const int px = blockIdx.x * blockDim.x + threadIdx.x;
            const int py = blockIdx.y * blockDim.y + threadIdx.y;
            if (px >= v.width || py >= v.height)
                return;
            const Bilinear b = bilinear(pixel_dir<RADIAL>(v, px, py), R);
            float3 c = make_float3(0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float* t = texels + static_cast<size_t>(b.idx[k]) * 3;
                c.x += b.w[k] * __ldg(t + 0);
                c.y += b.w[k] * __ldg(t + 1);
                c.z += b.w[k] * __ldg(t + 2);
            }
            const size_t n = static_cast<size_t>(v.width) * static_cast<size_t>(v.height);
            const size_t p = static_cast<size_t>(py) * static_cast<size_t>(v.width) + static_cast<size_t>(px);
            bg[p] = c.x;
            bg[n + p] = c.y;
            bg[2 * n + p] = c.z;
        }

        // Sum of x over the lanes in `peers` (all 32 lanes must call it with their own peer mask); the lowest lane of
        // each peer group ends with the group's total (the CUDA warp-aggregated-atomics peer reduction).
        template <int K>
        __device__ __forceinline__ void reduce_peers(unsigned peers, float (&x)[K]) {
            const int lane = static_cast<int>(threadIdx.x & 31u);
            int rel_pos = __popc(peers & ((1u << lane) - 1u));
            peers &= (0xfffffffeu << lane);
            while (__any_sync(0xffffffffu, peers != 0u)) {
                const int next = __ffs(peers);
                const int src = next > 0 ? next - 1 : lane;
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const float t = __shfl_sync(0xffffffffu, x[k], src);
                    if (next)
                        x[k] += t;
                }
                const unsigned done = static_cast<unsigned>(rel_pos & 1);
                peers &= ~__ballot_sync(0xffffffffu, done != 0u);
                rel_pos >>= 1;
            }
        }

        // blockDim = (32, BY): a warp is 32 neighbouring pixels of one row.
        template <bool RADIAL>
        __global__ void backward_kernel(const float* __restrict__ grad_image, const float* __restrict__ alpha,
                                        const int R, const CubemapView v, float* __restrict__ grad_texels) {
            const int px = blockIdx.x * 32 + static_cast<int>(threadIdx.x);
            const int py = blockIdx.y * blockDim.y + threadIdx.y;
            const bool inside = px < v.width && py < v.height;
            const size_t n = static_cast<size_t>(v.width) * static_cast<size_t>(v.height);
            float3 g = make_float3(0.0f, 0.0f, 0.0f);
            if (inside) {
                const size_t p = static_cast<size_t>(py) * static_cast<size_t>(v.width) + static_cast<size_t>(px);
                const float T = 1.0f - __ldg(alpha + p);
                g = make_float3(__ldg(grad_image + p) * T, __ldg(grad_image + n + p) * T, __ldg(grad_image + 2 * n + p) * T);
                if (!(isfinite(g.x) && isfinite(g.y) && isfinite(g.z)))
                    g = make_float3(0.0f, 0.0f, 0.0f); // counted by the stats of the texel gradient instead (never here)
            }
            const bool nonzero = g.x != 0.0f || g.y != 0.0f || g.z != 0.0f;
            if (__all_sync(0xffffffffu, !nonzero))
                return; // the whole warp is uniform here
            Bilinear b;
            int key = -1;
            if (nonzero) {
                b = bilinear(pixel_dir<RADIAL>(v, px, py), R);
                key = b.key;
            } else {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    b.idx[k] = 0;
                    b.w[k] = 0.0f;
                }
            }
            float vals[12];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                vals[3 * k + 0] = b.w[k] * g.x;
                vals[3 * k + 1] = b.w[k] * g.y;
                vals[3 * k + 2] = b.w[k] * g.z;
            }
            const unsigned peers = __match_any_sync(0xffffffffu, key);
            reduce_peers<12>(peers, vals);
            const int lane = static_cast<int>(threadIdx.x & 31u);
            if (key >= 0 && lane == __ffs(peers) - 1) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float* t = grad_texels + static_cast<size_t>(b.idx[k]) * 3;
                    atomicAdd(t + 0, vals[3 * k + 0]);
                    atomicAdd(t + 1, vals[3 * k + 1]);
                    atomicAdd(t + 2, vals[3 * k + 2]);
                }
            }
        }

        __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                x += __shfl_down_sync(0xffffffffu, x, o);
            return x;
        }

        __global__ void smooth_kernel(const float* __restrict__ texels, const int R, const float weight,
                                      float* __restrict__ grad, float* __restrict__ loss_out) {
            const int N = 6 * R * R;
            const int a = blockIdx.x * blockDim.x + threadIdx.x;
            float l = 0.0f;
            if (a < N) {
                const int f = a / (R * R);
                const int j = (a / R) % R;
                const int i = a % R;
                const int nb[4] = {texel_index(f, i - 1, j, R), texel_index(f, i + 1, j, R),
                                   texel_index(f, i, j - 1, R), texel_index(f, i, j + 1, R)};
                const float* ta = texels + static_cast<size_t>(a) * 3;
                const float a0 = ta[0], a1 = ta[1], a2 = ta[2];
                float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float* tn = texels + static_cast<size_t>(nb[k]) * 3;
                    const float d0 = a0 - tn[0], d1 = a1 - tn[1], d2 = a2 - tn[2];
                    g0 += d0;
                    g1 += d1;
                    g2 += d2;
                    l += d0 * d0 + d1 * d1 + d2 * d2;
                }
                const float sc = weight / static_cast<float>(N);
                grad[static_cast<size_t>(a) * 3 + 0] += sc * g0;
                grad[static_cast<size_t>(a) * 3 + 1] += sc * g1;
                grad[static_cast<size_t>(a) * 3 + 2] += sc * g2;
            }
            if (loss_out != nullptr) {
                l = warp_sum(l);
                if ((threadIdx.x & 31u) == 0u && l != 0.0f)
                    atomicAdd(loss_out, l * weight / (4.0f * static_cast<float>(N)));
            }
        }

        __global__ void adam_kernel(float* __restrict__ p, float* __restrict__ g, float* __restrict__ m,
                                    float* __restrict__ v, const int n, const float lr, const float b1, const float b2,
                                    const float eps, const float bias1, const float bias2, const float lo, const float hi,
                                    float* __restrict__ nonfinite) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            float gi = g[i];
            if (!isfinite(gi)) {
                if (nonfinite != nullptr)
                    atomicAdd(nonfinite, 1.0f);
                gi = 0.0f;
            }
            const float mi = b1 * m[i] + (1.0f - b1) * gi;
            const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
            m[i] = mi;
            v[i] = vi;
            const float upd = lr * (mi / bias1) / (sqrtf(vi / bias2) + eps);
            p[i] = fminf(fmaxf(p[i] - upd, lo), hi);
            g[i] = 0.0f;
        }

        __global__ void fill_kernel(float* __restrict__ t, const int n, const float r, const float gg, const float b) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            t[3 * i + 0] = r;
            t[3 * i + 1] = gg;
            t[3 * i + 2] = b;
        }

        __global__ void grad_stats_kernel(const float* __restrict__ g, const int n, float* __restrict__ out) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            float a = 0.0f, mx = 0.0f, bad = 0.0f;
            if (i < n) {
                const float gi = g[i];
                if (isfinite(gi)) {
                    a = fabsf(gi);
                    mx = a;
                } else {
                    bad = 1.0f;
                }
            }
            a = warp_sum(a);
            bad = warp_sum(bad);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o));
            if ((threadIdx.x & 31u) == 0u) {
                if (a != 0.0f)
                    atomicAdd(out + 0, a);
                if (mx > 0.0f)
                    atomicMax(reinterpret_cast<int*>(out + 1), __float_as_int(mx)); // non-negative floats order as ints
                if (bad != 0.0f)
                    atomicAdd(out + 2, bad);
            }
        }

    } // namespace

    // LFQPLC: the radial lens table (--lens-radial) is its own kernel instantiation (RADIAL = true); without it the
    // original kernels run.
    static bool cubemap_radial(const CubemapView& view) { return view.fisheye && view.radial.seg != nullptr; }

    void cubemap_render_background(const float* texels, const int res, const CubemapView& view, float* bg_chw,
                                   cudaStream_t stream) {
        stream = resolve_stream(stream);
        const dim3 block(16, 16);
        const dim3 grid((view.width + 15) / 16, (view.height + 15) / 16);
        if (cubemap_radial(view))
            render_kernel<true><<<grid, block, 0, stream>>>(texels, res, view, bg_chw);
        else
            render_kernel<false><<<grid, block, 0, stream>>>(texels, res, view, bg_chw);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bg_cubemap.render");
    }

    void cubemap_backward(const float* grad_image_chw, const float* alpha_hw, const int res, const CubemapView& view,
                          float* grad_texels, cudaStream_t stream) {
        stream = resolve_stream(stream);
        const dim3 block(32, 8);
        const dim3 grid((view.width + 31) / 32, (view.height + 7) / 8);
        if (cubemap_radial(view))
            backward_kernel<true><<<grid, block, 0, stream>>>(grad_image_chw, alpha_hw, res, view, grad_texels);
        else
            backward_kernel<false><<<grid, block, 0, stream>>>(grad_image_chw, alpha_hw, res, view, grad_texels);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bg_cubemap.backward");
    }

    void cubemap_pixel_dirs(const CubemapView& view, float* dirs_hw3, cudaStream_t stream) {
        stream = resolve_stream(stream);
        const dim3 block(16, 16);
        const dim3 grid((view.width + 15) / 16, (view.height + 15) / 16);
        if (cubemap_radial(view))
            dirs_kernel<true><<<grid, block, 0, stream>>>(view, dirs_hw3);
        else
            dirs_kernel<false><<<grid, block, 0, stream>>>(view, dirs_hw3);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bg_cubemap.pixel_dirs");
    }

    void cubemap_smooth_grad(const float* texels, const int res, const float weight, float* grad_texels, float* loss_out,
                             cudaStream_t stream) {
        stream = resolve_stream(stream);
        const int N = 6 * res * res;
        smooth_kernel<<<(N + 255) / 256, 256, 0, stream>>>(texels, res, weight, grad_texels, loss_out);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bg_cubemap.smooth");
    }

    void cubemap_adam_step(float* texels, float* grad, float* m, float* v, const int n, const float lr, const float beta1,
                           const float beta2, const float eps, const float bias1, const float bias2, const float lo,
                           const float hi, float* nonfinite_count, cudaStream_t stream) {
        stream = resolve_stream(stream);
        adam_kernel<<<(n + 255) / 256, 256, 0, stream>>>(texels, grad, m, v, n, lr, beta1, beta2, eps, bias1, bias2, lo, hi,
                                                          nonfinite_count);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bg_cubemap.adam");
    }

    void cubemap_fill(float* texels, const int res, const float r, const float g, const float b, cudaStream_t stream) {
        stream = resolve_stream(stream);
        const int N = 6 * res * res;
        fill_kernel<<<(N + 255) / 256, 256, 0, stream>>>(texels, N, r, g, b);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bg_cubemap.fill");
    }

    void cubemap_grad_stats(const float* grad, const int n, float* out3, cudaStream_t stream) {
        stream = resolve_stream(stream);
        grad_stats_kernel<<<(n + 255) / 256, 256, 0, stream>>>(grad, n, out3);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.bg_cubemap.grad_stats");
    }

} // namespace lfs::training::kernels
