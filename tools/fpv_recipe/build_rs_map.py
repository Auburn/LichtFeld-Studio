#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-pixel rolling-shutter readout-time map from a video of an LED flickering at a known frequency.

Usage (numpy, scipy, opencv-python; ffmpeg and ffprobe on PATH):
    python build_rs_map.py VIDEO --out DIR [--led-khz 1.0] [--step 7] [--bin 4] [--harmonics 3] [--grid 129x97]

Recording: the camera is static, in the same video mode as the footage (resolution, frame rate, stabilisation and lens
correction settings), filming a wall lit by an LED that flickers at a steady, known frequency (e.g. 1 kHz). Light the
wall evenly and keep it below clipping; a minute or two of video is enough. The rows read at different times see the
LED in different phases, so the frame shows bands, and the band phase at each pixel is its readout time.

Method (temporal phase shifting):
  * Every pixel sees I_k = a + A cos(theta_k + phi(u, v)) (+ harmonics) over the frames k, where theta_k = 2 pi f t_k is
    the LED phase at frame k's start and phi = 2 pi f t(u, v) carries the pixel's readout time t.
  * Frames: every STEP-th frame plus every I-frame (the I-frames are also used for a codec-bias check). STEP should
    spread the LED phases evenly: at 23.976 fps and 1 kHz the LED advances 17/24 cycle (mod 1) per frame, so any STEP
    coprime with 24 does. Frames are binned BIN x BIN (Y plane, 8-bit).
  * f is searched against the frame clock on a central patch. The frame clock fixes f only modulo the frame rate, so the
    nominal --led-khz has to be right to ~1 %. Then theta_k and the per-pixel (a, A_h, phi_h) are refined alternately
    (the advanced iterative algorithm of random-step phase-shifting interferometry); harmonics 2..H are fitted so a
    non-sinusoidal LED waveform does not bias phi.
  * Per bin: the phase noise from the fit residual gives sigma_t (ms); SNR = A1 / residual per frame.
  * The phase is unwrapped (the centre row along x, then each column from the centre row), and smooth models are fitted
    with weights 1 / (sigma_t^2 + floor^2), leaving out clipped bins and bins with sigma_t > --sigma-max: 2D Legendre
    polynomials, whose degree is chosen by 4-fold spatial-block cross-validation and an extrapolation test (fit without
    the outer border, score the border), and a radial-remap model as a cross-check.
    fit_rs_radial.py then fits the trainer's radial form to the measured bins.

Outputs (in --out):
  rs_time_map_grid.txt   the degree-chosen polynomial as an LFS_RS_TIME_MAP 1 grid (--rs-time-map; grid form), plus .npy
  rs_time_map_raw.npz    per-bin measured times t_raw, sigma_t, SNR, amplitude, level, valid / clipped masks, the I-minus-P
                         codec check, the bin centres u, v, the image size W, H: the input of fit_rs_radial.py
  rs_time_map_poly.json  the polynomial (Legendre and monomial coefficients), the radial-remap fit, the CV table, summary numbers
  rs_time_map_isochrones.png, rs_time_map_residual.png, rs_time_map_sigma.png   check images on the bin grid
  rs_map_report.txt      the log
Coordinates: u, v are continuous output-pixel coordinates with the pixel-CORNER convention: pixel (col, row) covers
    [col, col+1) x [row, row+1), so its centre is (col + 0.5, row + 0.5); the image spans u in [0, W], v in [0, H].
Times: ms, relative to the time at the image centre (u, v) = (W/2, H/2); later rows are positive.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

T0 = time.time()
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:7.1f}s] {s}", flush=True)
    LOG.append(s)


def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,codec_name,bit_rate:stream_side_data",
                          "-of", "json", video], capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = (int(x) for x in st["r_frame_rate"].split("/"))
    flags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=flags",
                            "-of", "csv=p=0", video], capture_output=True, text=True, check=True).stdout.split()
    keys = [i for i, f in enumerate(flags) if "K" in f]
    rot = [sd for sd in st.get("side_data_list", []) if "rotation" in sd]
    return dict(W=int(st["width"]), H=int(st["height"]), num=num, den=den, n=len(flags), keys=keys, rot=rot,
                codec=st.get("codec_name"), bit_rate=int(st.get("bit_rate", 0) or 0))


def decode(video, idx, W, H, b, threads=16):
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    cmd = ["ffmpeg", "-v", "error", "-noautorotate", "-threads", str(threads), "-i", video, "-map", "0:v:0",
           "-vf", f"select='{sel}',extractplanes=y", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 2)
    Hb, Wb = H // b, W // b
    F = np.empty((len(idx), Hb, Wb), np.float32)
    mx = np.zeros((Hb, Wb), np.uint8)
    for k in range(len(idx)):
        buf = p.stdout.read(W * H)
        if len(buf) != W * H:
            raise SystemExit(f"decode: got {k} of {len(idx)} frames")
        im = np.frombuffer(buf, np.uint8).reshape(H, W)
        blk = im[:Hb * b, :Wb * b].reshape(Hb, b, Wb, b)
        F[k] = blk.mean((1, 3), dtype=np.float32)
        np.maximum(mx, blk.max((1, 3)), out=mx)
        if k % 40 == 0:
            log(f"  decoded {k + 1}/{len(idx)}")
    rest = p.stdout.read()
    p.wait()
    if rest:
        log(f"  WARNING: {len(rest)} extra bytes after the expected frames")
    return F, mx


def design(theta, nh):
    cols = [np.ones_like(theta)]
    for h in range(1, nh + 1):
        cols += [np.cos(h * theta), np.sin(h * theta)]
    return np.stack(cols, 1)                                  # K x (1 + 2 nh)


def fit_pixels(F2, theta, nh):
    D = design(theta, nh)
    C = np.linalg.pinv(D) @ F2                                # (1+2nh) x P
    return C, D


def legendre_basis(x, y, d):
    from numpy.polynomial import legendre as L
    terms, cols = [], []
    Vx = L.legvander(x, d)
    Vy = L.legvander(y, d)
    for tot in range(d + 1):
        for j in range(tot + 1):
            i = tot - j
            terms.append((i, j))
            cols.append(Vx[:, i] * Vy[:, j])
    return np.stack(cols, 1), terms


def wlsq(B, t, w):
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(B * sw[:, None], t * sw, rcond=None)
    return c


def wrms(r, w):
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


def leg_to_mono(coef, terms, d):
    from numpy.polynomial import legendre as L
    M = np.zeros((d + 1, d + 1))                               # M[a, b] coefficient of x^a y^b
    for c, (i, j) in zip(coef, terms):
        px = L.leg2poly(np.eye(d + 1)[i])
        py = L.leg2poly(np.eye(d + 1)[j])
        M[:len(px), :len(py)] += c * np.outer(px, py)
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--led-khz", type=float, default=1.0, help="nominal LED frequency (kHz)")
    ap.add_argument("--step", type=int, default=7, help="use every STEP-th frame (coprime with 24 at 23.976 fps / 1 kHz)")
    ap.add_argument("--bin", type=int, default=4, help="bin size in pixels (BIN x BIN)")
    ap.add_argument("--harmonics", type=int, default=3, help="LED waveform harmonics fitted per pixel")
    ap.add_argument("--sigma-max", type=float, default=0.25, help="ms; bins noisier than this are left out of the fit")
    ap.add_argument("--sigma-floor", type=float, default=0.02, help="ms; added in quadrature in the fit weights")
    ap.add_argument("--max-degree", type=int, default=12, help="largest polynomial degree tried")
    ap.add_argument("--grid", default="129x97", help="nodes of the exported grid, NXxNY")
    ap.add_argument("--threads", type=int, default=16, help="ffmpeg decoding threads")
    ap.add_argument("--ref-readout-ms", type=float, default=31.0,
                    help="a linear row-rule readout (ms) to compare the map with in the report and the isochrone image")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import cv2
    from scipy.ndimage import gaussian_filter

    info = probe(a.video)
    W, H, b = info["W"], info["H"], a.bin
    ref_ms = a.ref_readout_ms
    dt_ms = 1000.0 * info["den"] / info["num"]
    log(f"video {a.video}: {W}x{H} {info['codec']} {info['num']}/{info['den']} fps, {info['n']} frames, "
        f"{info['bit_rate'] / 1e6:.1f} Mb/s, {len(info['keys'])} keyframes, rotation side data: {info['rot'] or 'none'}")
    if info["rot"]:
        log("  NOTE: rotation side data present; decoded with -noautorotate (sensor orientation)")
    idx = sorted(set(range(0, info["n"], a.step)) | set(info["keys"]))
    is_key = np.array([i in set(info["keys"]) for i in idx])
    in_step = np.array([i % a.step == 0 for i in idx])
    log(f"frames: {len(idx)} ({in_step.sum()} every {a.step}th, {is_key.sum()} I-frames, {(in_step & is_key).sum()} both)")
    F, mx = decode(a.video, idx, W, H, b, a.threads)
    K, Hb, Wb = F.shape
    tk = np.array(idx, np.float64) * dt_ms                    # frame start times (ms)
    F2 = F.reshape(K, -1)
    mean = F2.mean(0)
    log(f"decoded {K} frames, binned {b}x{b} -> {Wb}x{Hb}; level (8-bit) median {np.median(mean):.1f}, "
        f"p1 {np.percentile(mean, 1):.1f}, p99 {np.percentile(mean, 99):.1f}")

    # ---- 1. LED frequency against the frame clock (central patch) ----
    r0, c0 = Hb // 2, Wb // 2
    patch = (slice(r0 - Hb // 6, r0 - Hb // 24), slice(c0 - Wb // 5, c0 + Wb // 5))
    pix = np.zeros((Hb, Wb), bool)
    pix[patch] = True
    pix &= mx < 250
    sub = np.flatnonzero(pix.ravel())[::3]
    Z = F2[:, sub] - F2[:, sub].mean(0)
    best = None
    for lo, hi, st in [(-600e-6, 600e-6, 0.5e-6), (-3e-6, 3e-6, 0.01e-6)]:
        center = a.led_khz if best is None else best
        fs = center * (1 + np.arange(lo, hi + st / 2, st))
        E = np.empty(len(fs))
        for s in range(0, len(fs), 200):
            ph = np.exp(-2j * np.pi * fs[s:s + 200, None] * tk[None, :])          # nf x K
            E[s:s + 200] = (np.abs(ph @ Z) ** 2).sum(1)
        best = float(fs[np.argmax(E)])
    f = best
    shift = (f * dt_ms) % 1.0
    log(f"LED frequency vs the frame clock: {f:.7f} kHz (nominal {a.led_khz}; {1e6 * (f / a.led_khz - 1):+.1f} ppm; "
        f"frame-to-frame band shift {shift:.4f} cycles). The frame clock fixes f only modulo {1 / dt_ms:.5f} kHz.")

    # ---- 2. Alternate: per-pixel coefficients <-> per-frame LED phase ----
    theta = 2 * np.pi * f * tk
    good = np.flatnonzero((mx.ravel() < 250))
    for it in range(8):
        C, D = fit_pixels(F2, theta, a.harmonics)
        A1 = np.hypot(C[1], C[2])
        sel = good[A1[good] > np.percentile(A1[good], 40)][::5]
        b1, c1 = C[1, sel], C[2, sel]
        new = theta.copy()
        for k in range(K):
            # remove the mean and the harmonics (current theta), solve r = b1 cos th + c1 sin th
            r = F2[k, sel] - C[0, sel]
            for h in range(2, a.harmonics + 1):
                r = r - C[2 * h - 1, sel] * np.cos(h * theta[k]) - C[2 * h, sel] * np.sin(h * theta[k])
            M = np.stack([b1, c1], 1)
            cs, *_ = np.linalg.lstsq(M, r, rcond=None)
            new[k] = np.arctan2(cs[1], cs[0])
        d = np.angle(np.exp(1j * (new - theta)))
        d -= np.angle(np.mean(np.exp(1j * d)))              # a common offset is free (absorbed by phi)
        theta = theta + d
        log(f"  iteration {it + 1}: theta update rms {np.degrees(np.sqrt(np.mean(d * d))):.4f} deg")
        if np.sqrt(np.mean(d * d)) < np.radians(0.002):
            break
    # the per-frame LED phases against the searched frequency: drift (a frequency correction) and scatter
    rph = np.unwrap(np.angle(np.exp(1j * (theta - 2 * np.pi * f * tk))))
    lin = np.polyfit(tk, rph, 1)
    jit = rph - np.polyval(lin, tk)
    f_ref = f + lin[0] / (2 * np.pi)
    log(f"per-frame LED phase vs the frame clock: frequency {f_ref:.7f} kHz ({1e6 * (f_ref / a.led_khz - 1):+.1f} ppm; band shift "
        f"{(f_ref * dt_ms) % 1:.4f} cycles per frame); scatter of the frame phases about a steady LED: rms {np.degrees(jit.std()):.2f} deg "
        f"({1000 * jit.std() / (2 * np.pi * f):.2f} us), max {np.degrees(np.abs(jit).max()):.2f} deg")
    hist = np.histogram((theta / (2 * np.pi)) % 1, bins=12, range=(0, 1))[0]
    log(f"LED phase coverage over the {K} frames (12 bins of 30 deg): {hist.tolist()}")

    C, D = fit_pixels(F2, theta, a.harmonics)
    res2 = np.zeros(F2.shape[1])
    for s in range(0, F2.shape[1], 1 << 17):
        R = F2[:, s:s + (1 << 17)] - D @ C[:, s:s + (1 << 17)]
        res2[s:s + (1 << 17)] = (R * R).sum(0)
    dof = K - D.shape[1]
    sig_I = np.sqrt(res2 / dof)
    A1 = np.hypot(C[1], C[2])
    phi = np.arctan2(-C[2], C[1])                              # I = a + A cos(theta + phi)
    sig_phi = sig_I / np.maximum(A1, 1e-9) / np.sqrt(K / 2)
    sig_t = sig_phi / (2 * np.pi * f)
    snr = A1 / np.maximum(sig_I, 1e-9)
    harm = [np.hypot(C[2 * h - 1], C[2 * h]) for h in range(1, a.harmonics + 1)]
    ok = mx.ravel() < 250
    hr = ", ".join(f"h{h + 1} {np.median(harm[h][ok] / np.maximum(A1[ok], 1e-9)):.3f}" for h in range(1, a.harmonics))
    log(f"waveform: median harmonic / fundamental: {hr}; band contrast A1 / mean: median {np.median(A1[ok] / mean[ok]):.3f}")

    # ---- 3. codec-bias check: I-frames only vs the rest ----
    def sub_phase(mask):
        Cs, _ = fit_pixels(F2[mask], theta[mask], min(a.harmonics, 2))
        return np.arctan2(-Cs[2], Cs[1]), np.hypot(Cs[1], Cs[2])
    phI, AI = sub_phase(is_key)
    phP, AP = sub_phase(~is_key)
    dIP = np.angle(np.exp(1j * (phI - phP))) / (2 * np.pi * f)   # ms

    # ---- 4. maps on the bin grid ----
    shp = (Hb, Wb)
    phi, sig_t, snr, A1, mean, dIP = (x.reshape(shp) for x in (phi, sig_t, snr, A1, mean, dIP))
    clip = mx >= 250
    # unwrap on a lightly smoothed phasor field, then add back the wrapped raw difference
    zs = gaussian_filter(np.cos(phi) * A1, 1.5) + 1j * gaussian_filter(np.sin(phi) * A1, 1.5)
    ps = np.angle(zs)
    U = np.empty(shp)
    U[r0] = np.unwrap(ps[r0])
    for j in range(Wb):
        up = np.unwrap(np.concatenate([[U[r0, j]], ps[r0 - 1::-1, j]]))
        dn = np.unwrap(np.concatenate([[U[r0, j]], ps[r0 + 1:, j]]))
        U[r0::-1, j] = up
        U[r0:, j] = dn
    U = U + np.angle(np.exp(1j * (phi - U)))
    t_raw = U / (2 * np.pi * f)                                # ms
    uc = (np.arange(Wb) + 0.5) * b                            # bin centres, pixel-corner convention
    vc = (np.arange(Hb) + 0.5) * b
    uu, vv = np.meshgrid(uc, vc)
    xn = (uu - W / 2) / (W / 2)
    yn = (vv - H / 2) / (H / 2)
    sign = np.sign(np.median(t_raw[-Hb // 8:, Wb // 3:2 * Wb // 3]) - np.median(t_raw[:Hb // 8, Wb // 3:2 * Wb // 3]))
    log(f"time runs {'DOWN (top row first)' if sign > 0 else 'UP (bottom row first)'} the frame (sign fixed by the frame clock)")

    valid = (~clip) & (sig_t < a.sigma_max) & np.isfinite(t_raw)
    for thr in (0.02, 0.05, 0.1, a.sigma_max):
        log(f"  bins with sigma_t < {thr:.2f} ms: {100 * np.mean((sig_t < thr) & ~clip):.1f}% of the frame")
    log(f"  clipped bins (excluded): {100 * clip.mean():.2f}%; used in the fit: {100 * valid.mean():.1f}%")

    def region(name, rs, cs):
        s = (slice(*rs), slice(*cs))
        return (f"{name:13s} level {np.median(mean[s]):6.1f}  contrast {np.median(A1[s] / np.maximum(mean[s], 1e-6)):.3f}  "
                f"SNR/frame/bin {np.median(snr[s]):6.1f}  sigma_t {1000 * np.median(sig_t[s]):6.1f} us  "
                f"I-vs-P {1000 * np.median(dIP[s]):+6.1f} us (spread {1000 * np.std(dIP[s]):5.1f})")
    q = 24
    log("regions (8-bit levels; SNR = fundamental amplitude / residual noise per frame per bin; I-vs-P = codec check):")
    for nm, rs, cs in [("centre", (r0 - q, r0 + q), (c0 - q, c0 + q)),
                       ("quarter-left", (r0 - q, r0 + q), (Wb // 4 - q, Wb // 4 + q)),
                       ("left edge", (r0 - q, r0 + q), (0, 2 * q)), ("right edge", (r0 - q, r0 + q), (Wb - 2 * q, Wb)),
                       ("top centre", (0, 2 * q), (c0 - q, c0 + q)), ("bottom centre", (Hb - 2 * q, Hb), (c0 - q, c0 + q)),
                       ("TL corner", (0, 2 * q), (0, 2 * q)), ("TR corner", (0, 2 * q), (Wb - 2 * q, Wb)),
                       ("BL corner", (Hb - 2 * q, Hb), (0, 2 * q)), ("BR corner", (Hb - 2 * q, Hb), (Wb - 2 * q, Wb))]:
        log("  " + region(nm, rs, cs))
    good_IP = valid & (sig_t < 0.05)
    log(f"codec check over good bins: I-frames-only minus P-frames-only time median {1000 * np.median(dIP[good_IP]):+.1f} us, "
        f"rms {1000 * np.std(dIP[good_IP]):.1f} us (expected from noise alone ~{1000 * np.median(sig_t[good_IP]) * np.sqrt(K / is_key.sum()):.1f} us)")

    # ---- 5. smooth models ----
    m = valid.ravel()
    X, Y, T = xn.ravel()[m], yn.ravel()[m], t_raw.ravel()[m]
    Wt = 1.0 / (sig_t.ravel()[m] ** 2 + a.sigma_floor ** 2)
    bi = np.minimum((uu.ravel()[m] / W * 16).astype(int), 15)
    bj = np.minimum((vv.ravel()[m] / H * 12).astype(int), 11)
    fold = (bi % 2) + 2 * (bj % 2)
    border = (np.abs(X) > 0.85) | (np.abs(Y) > 0.85)
    log(f"fit points: {m.sum()} bins; weights 1 / (sigma_t^2 + {a.sigma_floor}^2)")
    log(" degree  terms  fit wrms(ms)  CV wrms(ms)  CV max(ms)  border-extrap wrms / max (ms)")
    table = []
    for d in range(1, a.max_degree + 1):
        B, terms = legendre_basis(X, Y, d)
        c = wlsq(B, T, Wt)
        r = T - B @ c
        cvr = np.empty_like(T)
        for fo in range(4):
            tr = fold != fo
            cc = wlsq(B[tr], T[tr], Wt[tr])
            cvr[~tr] = T[~tr] - B[~tr] @ cc
        cc = wlsq(B[~border], T[~border], Wt[~border])
        br = T[border] - B[border] @ cc
        table.append((d, len(terms), wrms(r, Wt), wrms(cvr, Wt), np.abs(cvr).max(), wrms(br, Wt[border]), np.abs(br).max()))
        log(f"   {d:3d}   {len(terms):4d}    {table[-1][2]:.4f}       {table[-1][3]:.4f}      {table[-1][4]:.3f}       "
            f"{table[-1][5]:.4f} / {table[-1][6]:.3f}")
    cvs = np.array([t[3] for t in table])
    ext = np.array([t[5] for t in table])
    # order: the lowest degree whose CV error is within max(10%, 2 us) of the best and whose border extrapolation is within
    # 25% of the best (a higher degree that fits a few us better but extrapolates worse is not worth it)
    tol = max(0.10 * cvs.min(), 0.002)
    ok_d = [t[0] for t in table if t[3] <= cvs.min() + tol and t[5] <= 1.25 * ext.min()]
    if not ok_d:
        ok_d = [t[0] for t in table if t[3] <= cvs.min() + tol]
    dsel = min(ok_d)
    log(f"chosen polynomial degree {dsel} (lowest with CV error within max(10%, 2 us) of the best ({cvs.min():.4f} ms) and border "
        f"extrapolation within 25% of the best ({ext.min():.4f} ms))")

    # radial-remap model: the camera remaps the sensor image; sensor rows are read linearly in time.
    from scipy.optimize import least_squares
    px_u, px_v = (xn.ravel()[m] * W / 2), (yn.ravel()[m] * H / 2)

    def radial(p, U_, V_):
        t0, tau, k1, k2, k3, cu, cv, tx, ky = p
        du, dv = U_ - cu, V_ - cv
        r2 = (du * du + dv * dv) / (H / 2) ** 2
        g = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        return t0 + tau * (dv / (H / 2)) * g + tx * du / (W / 2) + ky * (dv / (H / 2)) ** 2
    sw = np.sqrt(Wt)
    p0 = [0, 15.6, 0, 0, 0, 0, 0, 0, 0]
    sol = least_squares(lambda p: (radial(p, px_u, px_v) - T) * sw, p0, method="lm")
    rr = T - radial(sol.x, px_u, px_v)
    cvr = np.empty_like(T)
    for fo in range(4):
        tr = fold != fo
        s2 = least_squares(lambda p: (radial(p, px_u[tr], px_v[tr]) - T[tr]) * sw[tr], sol.x, method="lm")
        cvr[~tr] = T[~tr] - radial(s2.x, px_u[~tr], px_v[~tr])
    s3 = least_squares(lambda p: (radial(p, px_u[~border], px_v[~border]) - T[~border]) * sw[~border], sol.x, method="lm")
    brr = T[border] - radial(s3.x, px_u[border], px_v[border])
    log(f"radial-remap model (9 params: t0, tau, k1..k3, centre, x tilt, y^2): fit wrms {wrms(rr, Wt):.4f} ms, "
        f"CV wrms {wrms(cvr, Wt):.4f} ms, border extrapolation {wrms(brr, Wt[border]):.4f} / max {np.abs(brr).max():.3f} ms; "
        f"params {np.round(sol.x, 5).tolist()}")

    B, terms = legendre_basis(X, Y, dsel)
    coef = wlsq(B, T, Wt)

    def model(u, v):
        Bm, _ = legendre_basis(((u - W / 2) / (W / 2)).ravel(), ((v - H / 2) / (H / 2)).ravel(), dsel)
        return (Bm @ coef).reshape(np.shape(u))
    tc = float(model(np.array([W / 2.0]), np.array([H / 2.0]))[0])
    coef[0] -= tc                                              # P0 = 1, so this shifts the centre to t = 0
    t_raw = t_raw - tc
    T = T - tc
    t_fit = model(uu, vv)
    res = t_raw - t_fit
    r_used = res[valid]
    w_used = 1.0 / (sig_t[valid] ** 2 + a.sigma_floor ** 2)
    hq = valid & (sig_t < 0.05)
    log(f"FINAL degree-{dsel} fit: residual over fitted bins wrms {wrms(r_used, w_used):.4f} ms, rms {r_used.std():.4f} ms, "
        f"p99 |r| {np.percentile(np.abs(r_used), 99):.3f} ms, max {np.abs(r_used).max():.3f} ms; over sigma_t < 50 us bins: "
        f"rms {res[hq].std():.4f}, max {np.abs(res[hq]).max():.3f} ms")
    rad_full = radial(sol.x, uu.ravel() - W / 2, vv.ravel() - H / 2).reshape(shp)
    rad_full -= radial(sol.x, np.array([0.0]), np.array([0.0]))[0]
    dpr = t_fit - rad_full
    log(f"polynomial vs radial model over the whole frame: rms {dpr.std():.4f} ms, max {np.abs(dpr).max():.3f} ms "
        f"(over fitted bins {dpr[valid].std():.4f} / {np.abs(dpr[valid]).max():.3f}; outside them "
        f"{(dpr[~valid].std() if (~valid).any() else 0):.4f} / {(np.abs(dpr[~valid]).max() if (~valid).any() else 0):.3f})")

    # ---- 6. summary numbers ----
    def T_at(u, v):
        return float(model(np.array([float(u)]), np.array([float(v)]))[0])
    top, bot = T_at(W / 2, 0), T_at(W / 2, H)
    log(f"centre column: t(top edge) {top:+.3f} ms, t(bottom edge) {bot:+.3f} ms -> top-to-bottom {bot - top:.3f} ms "
        f"(rows 0..H; reference linear rule {ref_ms:g})")
    Bl = np.stack([np.ones(T.size), Y * H / 2], 1)
    cl = wlsq(Bl, T, Wt)
    rl = T - Bl @ cl
    log(f"best linear-in-row model over the fitted bins: {cl[1] * H:.3f} ms over the full height; residual wrms {wrms(rl, Wt):.3f} ms, "
        f"max {np.abs(rl).max():.3f} ms")
    rref = t_fit - (vv / H - 0.5) * ref_ms
    log(f"map vs the reference linear {ref_ms:g} ms rule (centre-aligned) over the whole frame: rms {rref.std():.3f} ms, "
        f"max {np.abs(rref).max():.3f} ms")
    for nm, v in [("top", 0), ("centre", H / 2), ("bottom", H)]:
        tl, tm, tr_ = T_at(0, v), T_at(W / 2, v), T_at(W, v)
        log(f"  row v={v:6.0f}: left edge {tl - tm:+.3f} ms, right edge {tr_ - tm:+.3f} ms relative to the centre of the row "
            f"(left-right asymmetry {tl - tr_:+.3f} ms)")
    mir = model(W - uu, vv)
    asym = t_fit - mir
    log(f"left-right asymmetry t(u,v) - t(W-u,v) of the fit: rms {asym.std():.3f} ms, max {np.abs(asym).max():.3f} ms; "
        f"over fitted bins only, max {np.abs(asym[valid & valid[:, ::-1]]).max():.3f} ms")
    rowtime = np.gradient(t_fit[:, Wb // 2], vc)
    for a_, b_ in [(0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]:
        s = (vc >= a_ * H) & (vc < b_ * H)
        log(f"  centre column rows {a_ * H:5.0f}-{b_ * H:5.0f}: {1 / rowtime[s].mean():6.1f} rows per ms")
    log(f"time range over the whole frame: {t_fit.min():+.3f} .. {t_fit.max():+.3f} ms (span {t_fit.max() - t_fit.min():.3f}); "
        f"mean over the frame {t_fit.mean():+.4f} ms; mid-range {(t_fit.max() + t_fit.min()) / 2:+.4f} ms")
    for thr in (0.5, 1, 2, 3, 5):
        log(f"  bins with band SNR per frame (fundamental / residual noise, per {b}x{b} bin) >= {thr}: "
            f"{100 * np.mean(valid & (snr >= thr)):.1f}% of the frame")
    # corner robustness: the same degree fitted on the good bins only (SNR >= 2 per frame), compared where it extrapolates
    gm = (valid & (snr >= 2.0)).ravel()[m]
    if gm.sum() > 1000 and (~gm).any():
        Bg, _ = legendre_basis(X[gm], Y[gm], dsel)
        cg = wlsq(Bg, T[gm], Wt[gm])
        Bx, _ = legendre_basis(xn.ravel(), yn.ravel(), dsel)
        tg = (Bx @ cg).reshape(shp)
        tg -= float((legendre_basis(np.array([0.0]), np.array([0.0]), dsel)[0] @ cg)[0])
        dd = tg - t_fit
        ex = ~(valid & (snr >= 2.0))
        log(f"corner robustness: degree {dsel} fitted on the SNR >= 2 bins only ({100 * gm.mean():.1f}% of the fitted bins) vs the full fit: "
            f"where it extrapolates ({100 * ex.mean():.1f}% of the frame) rms {dd[ex].std():.4f} ms, max {np.abs(dd[ex]).max():.3f} ms; "
            f"elsewhere max {np.abs(dd[~ex]).max():.3f} ms")
        for nm, s_ in [("TL", (slice(0, 48), slice(0, 48))), ("TR", (slice(0, 48), slice(-48, None))),
                       ("BL", (slice(-48, None), slice(0, 48))), ("BR", (slice(-48, None), slice(-48, None)))]:
            log(f"  {nm} corner (outer {48 * b} px): good-bins fit minus full fit median {1000 * np.median(dd[s_]):+.1f} us; "
                f"radial model minus full fit median {1000 * np.median(rad_full[s_] - t_fit[s_]):+.1f} us; "
                f"measured minus full fit median {1000 * np.median(res[s_]):+.1f} us")
    cover = valid.mean()
    log(f"COVERAGE: {100 * cover:.1f}% of the frame's bins constrain the fit (sigma_t < {a.sigma_max} ms, not clipped); "
        f"{100 * np.mean(sig_t < 0.05):.1f}% with sigma_t < 50 us; the rest ({100 * (1 - cover):.1f}%) is extrapolated / interpolated by the model")

    # ---- 7. exports ----
    # the map's linear-equivalent readout: the least-squares t = a + b v over the whole image (uniform weight), b H
    Bl2 = np.stack([np.ones(t_fit.size), vv.ravel() - H / 2], 1)
    cl2, *_ = np.linalg.lstsq(Bl2, t_fit.ravel(), rcond=None)
    ro_lin = float(cl2[1] * H)
    log(f"linear-equivalent readout of the map (least-squares t = a + b v over the whole frame, uniform): {ro_lin:.4f} ms "
        f"(written to the grid as readout_ms: the trainer divides the map by it, so --rs-readout-ms R draws the map x R / {ro_lin:.3f})")
    gx, gy = (int(s) for s in a.grid.split("x"))
    gu = np.linspace(0, W, gx)
    gv = np.linspace(0, H, gy)
    GU, GV = np.meshgrid(gu, gv)
    G = model(GU, GV)
    G -= G[gy // 2, gx // 2] if (gx % 2 and gy % 2) else T_at(W / 2, H / 2)
    np.save(os.path.join(a.out, "rs_time_map_grid.npy"), G.astype(np.float32))
    with open(os.path.join(a.out, "rs_time_map_grid.txt"), "w") as fo:
        fo.write("LFS_RS_TIME_MAP 1\n")
        fo.write(f"# rolling-shutter time map from {os.path.basename(a.video)} (LED {f:.6f} kHz), build_rs_map.py, degree-{dsel} Legendre fit\n")
        fo.write("# t(u, v) in milliseconds relative to the image centre (u, v) = (W/2, H/2); later = positive\n")
        fo.write("# u, v: output-pixel coordinates, pixel-corner convention (pixel (c, r) centre = (c + 0.5, r + 0.5)), image [0, W] x [0, H]\n")
        fo.write("# node (i, j) at u = i * W / (nx - 1), v = j * H / (ny - 1); bilinear between nodes; rows below: j = 0 .. ny-1, each nx values\n")
        fo.write("# readout_ms: the map's linear-equivalent readout (least-squares t = a + b v over the whole frame: b H). The trainer uses\n")
        fo.write("#   s(u, v) = t(u, v) / readout_ms in place of the row fraction (v / H - 1/2), and t = s x --rs-readout-ms.\n")
        fo.write(f"size {W} {H}\n")
        fo.write(f"readout_ms {ro_lin:.6f}\n")
        fo.write(f"grid {gx} {gy}\n")
        for j in range(gy):
            fo.write(" ".join(f"{x:.6f}" for x in G[j]) + "\n")
    # check: bilinear grid vs the model at the bin centres
    from scipy.interpolate import RegularGridInterpolator
    rgi = RegularGridInterpolator((gv, gu), G)
    gb = rgi(np.stack([vv.ravel(), uu.ravel()], 1)).reshape(shp)
    log(f"grid {gx}x{gy}: bilinear vs the model at every bin centre: max |d| {np.abs(gb - t_fit).max() * 1000:.2f} us")
    M = leg_to_mono(coef, terms, dsel)
    json.dump(dict(
        format="rs_time_map_poly 1", video=os.path.basename(a.video), led_khz=f, degree=dsel,
        units="ms relative to the image centre (W/2, H/2)",
        coords="x = (u - W/2) / (W/2), y = (v - H/2) / (H/2); u, v output pixels, pixel-corner convention",
        W=W, H=H, legendre_terms=[[int(i), int(j), float(c)] for (i, j), c in zip(terms, coef)],
        legendre_note="t = sum c * P_i(x) * P_j(y), P = Legendre polynomials",
        monomial=[[int(i), int(j), float(M[i, j])] for i in range(dsel + 1) for j in range(dsel + 1) if abs(M[i, j]) > 0],
        monomial_note="t = sum c * x^i * y^j (same function)",
        radial_model=dict(params=dict(zip(["t0", "tau", "k1", "k2", "k3", "cu", "cv", "tx", "ky"], [float(x) for x in sol.x])),
                          formula="du = u - W/2 - cu, dv = v - H/2 - cv, r2 = (du^2 + dv^2) / (H/2)^2, "
                                  "t = t0 + tau * dv/(H/2) * (1 + k1 r2 + k2 r2^2 + k3 r2^3) + tx * du/(W/2) + ky * (dv/(H/2))^2 "
                                  "(not centre-shifted)"),
        cv_table=[dict(degree=t[0], terms=t[1], fit_wrms=t[2], cv_wrms=t[3], cv_max=t[4], border_wrms=t[5], border_max=t[6]) for t in table],
        top_to_bottom_centre_ms=bot - top, linear_fit_ms=float(cl[1] * H), linear_equivalent_readout_ms=ro_lin),
        open(os.path.join(a.out, "rs_time_map_poly.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "rs_time_map_raw.npz"), t_raw=t_raw.astype(np.float32), t_fit=t_fit.astype(np.float32),
                        sigma_t=sig_t.astype(np.float32), snr=snr.astype(np.float32), amp=A1.astype(np.float32),
                        level=mean.astype(np.float32), valid=valid, clipped=clip, i_minus_p=dIP.astype(np.float32),
                        u=uc, v=vc, theta=theta, frames=np.array(idx), led_khz=f, W=W, H=H)

    # check images (bin grid = W/b x H/b)
    lo, hi = np.floor(t_fit.min()), np.ceil(t_fit.max())
    norm = np.clip((np.where(valid, t_raw, t_fit) - lo) / (hi - lo), 0, 1)
    col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~valid] = (col[~valid] * 0.35).astype(np.uint8)
    iso = np.zeros(shp, bool)
    for lvl in np.arange(lo, hi + 1):
        s_ = np.sign(t_fit - lvl)
        iso |= (np.abs(np.diff(s_, axis=0, prepend=s_[:1])) > 0) | (np.abs(np.diff(s_, axis=1, prepend=s_[:, :1])) > 0)
    col[iso] = (255, 255, 255)
    linref = (vv / H - 0.5) * ref_ms
    for lvl in np.arange(np.floor(linref.min()), np.ceil(linref.max()) + 1, 5):
        s_ = np.sign(linref - lvl)
        e = (np.abs(np.diff(s_, axis=0, prepend=s_[:1])) > 0)
        col[e & ~iso] = (0, 0, 0)
    cv2.putText(col, f"t(u,v): 1 ms isochrones (white, fit deg {dsel}); black = linear {ref_ms:g} ms rule every 5 ms; dim = not fitted",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, "rs_time_map_isochrones.png"), col)
    rn = np.clip(res / 0.2 * 0.5 + 0.5, 0, 1)
    rc = cv2.applyColorMap((rn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rc[~valid] = 40
    cv2.putText(rc, f"measured - fit, +-0.2 ms full scale; grey = not fitted; rms {r_used.std() * 1000:.0f} us",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, "rs_time_map_residual.png"), rc)
    sn = np.clip((np.log10(np.maximum(sig_t, 1e-4)) + 3) / 3, 0, 1)    # 1 us .. 1 ms
    sc = cv2.applyColorMap((sn * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.putText(sc, f"sigma_t per {b}x{b} bin, log 1 us (dark) .. 1 ms (yellow)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, "rs_time_map_sigma.png"), sc)

    with open(os.path.join(a.out, "rs_map_report.txt"), "w") as fo:
        fo.write("\n".join(LOG) + "\n")
    log("done")


if __name__ == "__main__":
    main()
