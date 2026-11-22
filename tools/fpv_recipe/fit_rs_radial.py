#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The trainer's radial readout-time form (--rs-time-map, LFS_RS_TIME_MAP 1, 'radial'), fitted to a build_rs_map.py result.

Usage:
    python fit_rs_radial.py MAPDIR [--out FILE] [--size WxH] [--sigma-floor 0.02] [--compare OTHER_MAPDIR]
                            [--linear-test MS]

MAPDIR/rs_time_map_raw.npz (build_rs_map.py): the measured per-bin times t_raw, their noise sigma_t, the valid mask
and the smooth polynomial fit t_fit. The image size is read from the npz (W, H), else from MAPDIR/rs_time_map_poly.json,
else from --size.

Model (ms), with x and y both in half image WIDTHS:
    x = (u - W/2) / (W/2) - cx,  y = (v - H/2) / (W/2) - cy,  r^2 = x^2 + y^2
    t = t0 + k y (1 + a r^2 + b r^4 + c r^6)
fitted to every valid bin with weights 1 / (sigma_t^2 + floor^2) (the weights of build_rs_map.py's polynomial),
Levenberg-Marquardt, with a 4-fold spatial-block cross-validation.

Writes --out (default MAPDIR/rs_time_radial.txt):
    LFS_RS_TIME_MAP 1
    size W H
    radial k a b c cx cy
and the fitted parameters and residuals next to it as JSON (same name, .json). t0 is dropped from the file: the trainer
subtracts the centre's time and divides by the form's linear-equivalent readout R (the least-squares slope of t against
the row over the whole image, times H), so s = (t(u, v) - t(centre)) / R and the splat is drawn at s x --rs-readout-ms.
--linear-test MS also writes the plain row rule at MS milliseconds in both file forms, rs_time_linear<MS>_radial.txt and
rs_time_linear<MS>_grid.txt, next to --out: a check that the trainer's time-map path reproduces the row rule.
--compare OTHER_MAPDIR prints the differences to another build_rs_map.py result (e.g. a second LED video).
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares


def make_model(W, H):
    def model(p, u, v):
        t0, k, a, b, c, cx, cy = p
        x = (u - W / 2) / (W / 2) - cx
        y = (v - H / 2) / (W / 2) - cy
        r2 = x * x + y * y
        return t0 + k * y * (1 + r2 * (a + r2 * (b + r2 * c)))
    return model


def wrms(r, w):
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


def load(d):
    z = np.load(os.path.join(d, "rs_time_map_raw.npz"))
    uu, vv = np.meshgrid(z["u"], z["v"])
    return z, uu, vv


def image_size(d, z, size):
    if size:
        W, H = (int(s) for s in size.lower().split("x"))
        return W, H
    if "W" in z.files and "H" in z.files:
        return int(z["W"]), int(z["H"])
    pj = os.path.join(d, "rs_time_map_poly.json")
    if os.path.exists(pj):
        j = json.load(open(pj))
        return int(j["W"]), int(j["H"])
    raise SystemExit(f"{d}: no image size in rs_time_map_raw.npz or rs_time_map_poly.json; pass --size WxH")


def linear_equivalent(tfun, W, H):
    fu, fv = np.meshgrid(np.linspace(0, 1, 257), np.linspace(0, 1, 193))
    t = tfun(fu * W, fv * H)
    A = np.stack([np.ones(t.size), fv.ravel() - 0.5], 1)
    return float(np.linalg.lstsq(A, t.ravel(), rcond=None)[0][1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mapdir", help="a build_rs_map.py output folder")
    ap.add_argument("--out", default="", help="output map file (default MAPDIR/rs_time_radial.txt)")
    ap.add_argument("--size", default="", help="image size WxH, if the map folder does not record it")
    ap.add_argument("--sigma-floor", type=float, default=0.02, help="ms; added in quadrature in the fit weights")
    ap.add_argument("--compare", default="", help="another build_rs_map.py folder to compare with")
    ap.add_argument("--linear-test", type=float, default=None, metavar="MS",
                    help="also write the row rule at MS ms as radial and grid test maps")
    a = ap.parse_args()
    z, uu, vv = load(a.mapdir)
    W, H = image_size(a.mapdir, z, a.size)
    model = make_model(W, H)
    t, sig, valid, tfit = z["t_raw"].astype(np.float64), z["sigma_t"].astype(np.float64), z["valid"], z["t_fit"].astype(np.float64)
    m = valid
    U, V, T = uu[m], vv[m], t[m]
    w = 1.0 / (sig[m] ** 2 + a.sigma_floor ** 2)
    sw = np.sqrt(w)
    p0 = [0, 22.8, -0.22, 0.16, -0.08, 0, 0]
    sol = least_squares(lambda p: (model(p, U, V) - T) * sw, p0, method="lm", x_scale="jac")
    p = sol.x
    r = T - model(p, U, V)
    bi = np.minimum((U / W * 16).astype(int), 15)
    bj = np.minimum((V / H * 12).astype(int), 11)
    fold = (bi % 2) + 2 * (bj % 2)
    cvr = np.empty_like(T)
    for f in range(4):
        tr = fold != f
        s2 = least_squares(lambda q: (model(q, U[tr], V[tr]) - T[tr]) * sw[tr], p, method="lm", x_scale="jac")
        cvr[~tr] = T[~tr] - model(s2.x, U[~tr], V[~tr])
    rp = (t - tfit)[m]
    rp = rp - np.average(rp, weights=w)
    print(f"{a.mapdir}: radial model fitted to {m.sum()} bins of a {W}x{H} image (weights 1 / (sigma_t^2 + {a.sigma_floor}^2))")
    print(f"  t0 {p[0]:+.5f} ms, k {p[1]:.5f} ms per half-width, a {p[2]:+.5f}, b {p[3]:+.5f}, c {p[4]:+.5f}, cx {p[5]:+.5f}, cy {p[6]:+.5f} "
          f"(cx, cy = {p[5] * W / 2:+.1f}, {p[6] * W / 2:+.1f} px)")
    print(f"  residual: wrms {wrms(r, w) * 1000:.2f} us, rms {r.std() * 1000:.2f} us, p99 |r| {np.percentile(np.abs(r), 99) * 1000:.1f} us, "
          f"max {np.abs(r).max() * 1000:.1f} us; 4-fold block CV wrms {wrms(cvr, w) * 1000:.2f} us, max {np.abs(cvr).max() * 1000:.1f} us")
    print(f"  the smooth 2D polynomial (build_rs_map.py): residual wrms {wrms(rp, w) * 1000:.2f} us, max {np.abs(rp).max() * 1000:.1f} us")
    full_r = model(p, uu, vv)
    d = full_r - tfit
    d -= d[d.shape[0] // 2, d.shape[1] // 2]
    print(f"  radial minus polynomial over the whole frame (centre-aligned): rms {d.std() * 1000:.2f} us, max {np.abs(d).max() * 1000:.1f} us")
    res_full = t - full_r
    q = 48
    nv, nu = len(z["v"]), len(z["u"])
    regs = {"centre": (slice(nv // 2 - q, nv // 2 + q), slice(nu // 2 - q, nu // 2 + q)),
            "left edge": (slice(nv // 2 - q, nv // 2 + q), slice(0, q)),
            "right edge": (slice(nv // 2 - q, nv // 2 + q), slice(-q, None)),
            "top edge": (slice(0, q), slice(nu // 2 - q, nu // 2 + q)),
            "bottom edge": (slice(-q, None), slice(nu // 2 - q, nu // 2 + q)),
            "TL corner": (slice(0, q), slice(0, q)), "TR corner": (slice(0, q), slice(-q, None)),
            "BL corner": (slice(-q, None), slice(0, q)), "BR corner": (slice(-q, None), slice(-q, None))}
    print(f"  per region (outer {q} bins for edges / corners): measured minus radial (median / rms / max), radial minus polynomial (median)")
    for nm, s in regs.items():
        e = res_full[s]
        print(f"    {nm:12s} {np.median(e) * 1000:+6.1f} / {e.std() * 1000:5.1f} / {np.abs(e).max() * 1000:5.1f} us   {np.median(d[s]) * 1000:+6.1f} us")
    # the trainer's file (t0 dropped: the trainer subtracts the centre's time)
    out = a.out or os.path.join(a.mapdir, "rs_time_radial.txt")
    odir = os.path.dirname(os.path.abspath(out))
    os.makedirs(odir, exist_ok=True)
    tc = model(p, np.array([W / 2.0]), np.array([H / 2.0]))[0] - p[0]
    R = linear_equivalent(lambda u, v: model(p, u, v), W, H)
    with open(out, "w") as fo:
        fo.write("LFS_RS_TIME_MAP 1\n")
        fo.write(f"# radial readout-time form fitted to {os.path.basename(os.path.normpath(a.mapdir))} (fit_rs_radial.py; all valid bins, "
                 f"weights 1 / (sigma_t^2 + {a.sigma_floor}^2))\n")
        fo.write("# x = (u - W/2) / (W/2) - cx, y = (v - H/2) / (W/2) - cy (half image widths), r^2 = x^2 + y^2, t = k y (1 + a r^2 + b r^4 + c r^6) ms\n")
        fo.write(f"# residual wrms {wrms(r, w) * 1000:.2f} us, max {np.abs(r).max() * 1000:.1f} us; CV wrms {wrms(cvr, w) * 1000:.2f} us\n")
        fo.write(f"# the trainer: s = (t - t(centre) [{tc:+.5f} ms]) / R, R = the least-squares linear-equivalent readout over the image "
                 f"[{R:.5f} ms, computed by the loader]; t = s x --rs-readout-ms\n")
        fo.write(f"size {W} {H}\n")
        fo.write(f"radial {p[1]:.8f} {p[2]:.8f} {p[3]:.8f} {p[4]:.8f} {p[5]:.8f} {p[6]:.8f}\n")
    print(f"  wrote {out}: linear-equivalent readout {R:.4f} ms; t(centre) {tc:+.4f} ms (subtracted by the trainer); centre column top "
          f"to bottom {model(p, np.array([W / 2.0]), np.array([float(H)]))[0] - model(p, np.array([W / 2.0]), np.array([0.0]))[0]:.4f} ms")
    json.dump(dict(t0=p[0], k=p[1], a=p[2], b=p[3], c=p[4], cx=p[5], cy=p[6], wrms_us=wrms(r, w) * 1000, max_us=float(np.abs(r).max() * 1000),
                   cv_wrms_us=wrms(cvr, w) * 1000, linear_equivalent_ms=R, t_centre_ms=float(tc), W=W, H=H),
              open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
    if a.linear_test is not None:
        ms = a.linear_test
        span = 2.0 * H / W                                   # y runs over +-H/W half-widths
        with open(os.path.join(odir, f"rs_time_linear{ms:g}_radial.txt"), "w") as fo:
            fo.write(f"LFS_RS_TIME_MAP 1\n# TEST: the row rule at {ms:g} ms as a radial form (a = b = c = cx = cy = 0, k = {ms:g} / {span:g})\n")
            fo.write(f"size {W} {H}\nradial {ms / span:.9f} 0 0 0 0 0\n")
        with open(os.path.join(odir, f"rs_time_linear{ms:g}_grid.txt"), "w") as fo:
            fo.write(f"LFS_RS_TIME_MAP 1\n# TEST: the row rule at {ms:g} ms as a 129 x 97 grid\n")
            fo.write(f"size {W} {H}\nreadout_ms {ms:g}\ngrid 129 97\n")
            for j in range(97):
                fo.write(" ".join([f"{(j / 96 - 0.5) * ms:.6f}"] * 129) + "\n")
        print(f"  wrote the row-rule test maps rs_time_linear{ms:g}_radial.txt / _grid.txt in {odir}")
    if a.compare:
        z2, uu2, vv2 = load(a.compare)
        t2 = z2["t_fit"].astype(np.float64)
        dd = tfit - t2
        dd -= dd[dd.shape[0] // 2, dd.shape[1] // 2]
        print(f"\ncompare: {a.mapdir} minus {a.compare} (both smooth polynomial fits, centre-aligned):")
        print(f"  whole frame rms {dd.std() * 1000:.1f} us, max {np.abs(dd).max() * 1000:.1f} us")
        for nm, s in regs.items():
            print(f"    {nm:12s} median {np.median(dd[s]) * 1000:+7.1f} us, max |d| {np.abs(dd[s]).max() * 1000:6.1f} us")
        j1 = json.load(open(os.path.join(a.mapdir, "rs_time_map_poly.json")))
        j2 = json.load(open(os.path.join(a.compare, "rs_time_map_poly.json")))
        for key in ("top_to_bottom_centre_ms", "linear_fit_ms", "linear_equivalent_readout_ms"):
            print(f"  {key}: {j1.get(key, float('nan')):.4f} vs {j2.get(key, float('nan')):.4f} ms")
        rp2 = os.path.join(a.compare, "rs_time_radial.json")
        if os.path.exists(rp2):
            j3 = json.load(open(rp2))
            print("  radial parameters: " + ", ".join(f"{k} {p[i]:+.5f} vs {j3[k]:+.5f}" for i, k in enumerate(["t0", "k", "a", "b", "c", "cx", "cy"])))
            p2 = [j3[k] for k in ["t0", "k", "a", "b", "c", "cx", "cy"]]
            dr = model(p, uu, vv) - model(p2, uu, vv)
            dr -= dr[dr.shape[0] // 2, dr.shape[1] // 2]
            print(f"  radial forms differ over the frame: rms {dr.std() * 1000:.1f} us, max {np.abs(dr).max() * 1000:.1f} us; corners "
                  + ", ".join(f"{nm} {np.median(dr[s]) * 1000:+.1f}" for nm, s in regs.items() if "corner" in nm) + " us")


if __name__ == "__main__":
    main()
