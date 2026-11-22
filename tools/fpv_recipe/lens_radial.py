#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Radial lens table for the trainer's --lens-radial FILE: a free radial profile theta -> theta_d that replaces the
OPENCV_FISHEYE k-polynomial, for lenses that a four-term polynomial cannot follow out to the edge of the frame.

Usage:
  python lens_radial.py profile PROFILE.txt OUT_DIR [--cameras CAMERAS.txt] [--valid-deg DEG] [--nodes 2049]
        a table from a theta -> theta_d profile sampled from any calibration (format below)
  python lens_radial.py rspl RESULTS.json OUT_DIR --size WxH --valid-deg DEG [--nodes 2049] [--centre-deg 1.0] [--model rspl]
        a table from a fisheye-polynomial + B-spline calibration (the 'rspl' model, format below)
  python lens_radial.py poly CAMERAS.txt OUT_DIR [--end-deg 88] [--nodes 2049]
        a table of an OPENCV_FISHEYE lens's own polynomial: a test that the table reproduces the polynomial
  python lens_radial.py info FILE
Each builder writes OUT_DIR/lens_radial.txt and OUT_DIR/cameras.txt (the paired OPENCV_FISHEYE camera line).

THE LENS. A camera-space point at angle theta from the axis (rho = |(x, y)|, theta = atan2(rho, z)) is imaged at
  u = fx theta_d x / rho + cx,  v = fy theta_d y / rho + cy,   theta_d = theta P(theta^2)
(OPENCV_FISHEYE has P(u) = 1 + k1 u + k2 u^2 + k3 u^3 + k4 u^4). Here P is a table: nodes uniform in theta,
theta_i = i dtheta (i = 0 .. n-1, theta_{n-1} = theta_end), and on each interval [u_i, u_{i+1}] (u = theta^2) the
quintic Hermite interpolant in u of P, dP/du and d2P/du2 at the two nodes, stored as polynomial coefficients in
t = (u - u_i) / (u_{i+1} - u_i):  P = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5 (C2 in u: the trainer's backward
pass uses d2P/du2). Evaluation (as the trainer does it, in float32):
  theta = sqrt(u); if theta < theta_end: i = min(floor(theta / dtheta), n - 2), t = (u - u0_i) inv_h_i,
      P = Horner(c, t), dP/du = Horner'(c, t) inv_h_i, d2P/du2 = Horner''(c, t) inv_h_i^2
  else (beyond the table): theta_d = A + B theta (linear in theta, C1 at theta_end): P = A / theta + B,
      dP/du = -A / (2 theta^3), d2P/du2 = 3 A / (4 theta^5).
fx, fy, cx, cy are the camera's (cameras.txt); the table holds only the radial profile (angles: resolution-free).

THE FILE (text; '#' comments):
  LENS_RADIAL_TABLE 1
  camera W H fx fy cx cy k1 k2 k3 k4   the OPENCV_FISHEYE camera this table belongs to (full resolution). The trainer
                                       applies the table only to a camera with exactly these k's (relative 1e-6) and
                                       these fx/W, fy/H, cx/W, cy/H, so other lenses in the same run stay polynomial.
                                       The k's are the camera's own (the profile builder, when given) or a least-squares
                                       8-value approximation of the profile over the frame inside 'valid', so a tool
                                       that does not read the table still sees a close lens.
  nodes n dtheta theta_end             radians; theta_end = (n - 1) dtheta < 90 deg
  valid theta_valid                    the largest angle the profile is supported by data (radians); the table ends
                                       there. Pixels beyond it should be masked out of training (dataset masks).
  extrap A B                           theta_d = A + B theta beyond theta_end
  centre theta_c                       optional (rspl tables): inside theta_c the B-spline correction is replaced by an
                                       odd polynomial a1 t + a3 t^3 + a5 t^5 matching it to second order at theta_c (a
                                       uniform B-spline is not pinned at theta = 0, so the fit can leave a small offset at
                                       the axis that no smooth lens has)
  seg i u0 inv_h c0 c1 c2 c3 c4 c5     n - 1 rows, in order
  node i theta P dP/du d2P/du2 theta_d n rows (the reference values the segments were built from; not read by the trainer)
  source ...                           free text (not read by the trainer)
The trainer checks theta_d / theta ~ 1 on the axis, theta_d increasing over the whole table, and B > 0.

PROFILE INPUT (profile): text or CSV, '#' comments, values separated by spaces and/or commas, a header line of words
(e.g. 'theta_rad,theta_d_rad') is skipped:
  camera W H fx fy cx cy [k1 k2 k3 k4]    the camera the profile belongs to (or --cameras: the first OPENCV_FISHEYE
                                          camera of a COLMAP cameras.txt, whose k1..k4 are then used)
  theta_rad theta_d_rad                   one sample per line, theta_d in the same normalised units as theta_d above
                                          (image radius / focal length), increasing; dense enough to follow the lens
                                          (e.g. every 0.05-0.25 deg), from near the axis to the edge of the data.
P = theta_d / theta is interpolated as a quintic spline in u = theta^2 (samples at theta = 0 are dropped), and the
table is built from its values and derivatives at the nodes, from 0 to --valid-deg (default: the last sample). The
profile must be smooth (sampled from a calibration model, not raw measurements). If the camera line gives k1..k4, the
table is paired with exactly those k's, so a dataset whose cameras.txt already holds them needs no change; without
them the k's are fitted to the profile and OUT_DIR/cameras.txt holds the camera line to put in the dataset.

RSPL INPUT (rspl): a JSON {MODEL: {"params": {...}, "rms": px, "cv": px}} ("rms" and "cv" optional, MODEL default 'rspl')
with params fx, fy, cx, cy, k1..k4, q0..q19, for the profile
  theta_d(theta) = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) + 1e-3 sum_j q_j B_j(theta / 1.45)
where B_j is the uniform cubic B-spline basis of 20 control points over theta in [0, 1.45 rad] (17 spans):
s = clip(theta / 1.45, 0, 1) x 17, i0 = min(floor(s), 16), t = s - i0, and the four weights (1 - t)^3 / 6,
(3t^3 - 6t^2 + 4) / 6, (-3t^3 + 3t^2 + 3t + 1) / 6, t^3 / 6 on q[i0 .. i0 + 3]. --size is the calibrated image's size.
"""
import argparse
import json
import math
import os
import re

import numpy as np

NQ, QMAX = 20, 1.45            # rspl: 20 uniform cubic B-spline control points over theta in [0, 1.45 rad]
MAGIC = "LENS_RADIAL_TABLE"


# ----------------------------------------------------------------------------------------------- the rspl model (float64)
def rspl_thetad(g, th, deriv=0):
    """the rspl radial profile theta_d(theta) (fisheye polynomial + 1e-3 sum_j B_j(theta / QMAX) q_j) and its first /
    second theta-derivatives (deriv 0, 1, 2), float64, theta in [0, QMAX]."""
    th = np.asarray(th, np.float64)
    k = [g["k1"], g["k2"], g["k3"], g["k4"]]
    t2 = th * th
    if deriv == 0:
        d = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))
    elif deriv == 1:
        d = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
    else:
        d = th * (6 * k[0] + t2 * (20 * k[1] + t2 * (42 * k[2] + t2 * 72 * k[3])))
    q = np.array([g[f"q{j}"] for j in range(NQ)], np.float64)
    s = np.clip(th / QMAX, 0, 1) * (NQ - 3)
    i0 = np.minimum(np.floor(s).astype(int), NQ - 4)
    t = s - i0
    if deriv == 0:
        w = np.stack([(1 - t) ** 3, 3 * t ** 3 - 6 * t ** 2 + 4, -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3], 1) / 6
    elif deriv == 1:
        w = np.stack([-3 * (1 - t) ** 2, 9 * t ** 2 - 12 * t, -9 * t ** 2 + 6 * t + 3, 3 * t ** 2], 1) / 6 * ((NQ - 3) / QMAX)
    else:
        w = np.stack([6 * (1 - t), 18 * t - 12, -18 * t + 6, 6 * t], 1) / 6 * ((NQ - 3) / QMAX) ** 2
    idx = i0[:, None] + np.arange(4)[None]
    return d + (w * q[idx]).sum(1) * 1e-3


def _odd_centre(g, thc):
    """a1, a3, a5 of c~(t) = a1 t + a3 t^3 + a5 t^5 matching the B-spline correction c(t) = theta_d(t) - t P_poly(t^2) in
    value, slope and curvature at t = thc"""
    gp = dict(g)
    for j in range(NQ):
        gp[f"q{j}"] = 0.0
    c = [float(rspl_thetad(g, [thc], d)[0] - rspl_thetad(gp, [thc], d)[0]) for d in (0, 1, 2)]
    t = thc
    A = np.array([[t, t ** 3, t ** 5], [1, 3 * t ** 2, 5 * t ** 4], [0, 6 * t, 20 * t ** 3]], np.float64)
    return np.linalg.solve(A, np.array(c)), c


def _P_from_thetad(td, d1, d2, th):
    """P = theta_d / theta and its u-derivatives (u = theta^2) from theta_d and its theta-derivatives (theta > 0)"""
    P = td / th
    Pt = (d1 * th - td) / th ** 2
    Ptt = d2 / th - 2 * d1 / th ** 2 + 2 * td / th ** 3
    return P, Pt / (2 * th), (Ptt - Pt / th) / (4 * th ** 2)


# ----------------------------------------------------------------------------------------------- building a table
def _segments(theta, P, P1, P2):
    """quintic Hermite (in u) coefficients per interval -> rows (u0, inv_h, c0..c5), float64"""
    u = theta ** 2
    rows = []
    for i in range(len(theta) - 1):
        h = u[i + 1] - u[i]
        p0, d0, s0 = P[i], P1[i] * h, P2[i] * h * h          # value, first, second derivative in t
        p1, d1, s1 = P[i + 1], P1[i + 1] * h, P2[i + 1] * h * h
        c0, c1, c2 = p0, d0, s0 / 2
        # remaining c3..c5 from the right-end conditions
        r0 = p1 - (c0 + c1 + c2)
        r1 = d1 - (c1 + 2 * c2)
        r2 = s1 - 2 * c2
        M = np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], np.float64)
        c3, c4, c5 = np.linalg.solve(M, np.array([r0, r1, r2]))
        rows.append([u[i], 1.0 / h, c0, c1, c2, c3, c4, c5])
    return np.array(rows, np.float64)


def write_table(path, cam, theta, P, P1, P2, valid, extra, source, centre=None):
    n = len(theta)
    dth = theta[1] - theta[0]
    te = theta[-1]
    seg = _segments(theta, P, P1, P2)
    td_e = te * P[-1]
    slope_e = P[-1] + 2 * te * te * P1[-1]          # theta_d'(theta) = P + 2 u P'
    A, B = td_e - slope_e * te, slope_e
    W, H, fx, fy, cx, cy, k = cam
    with open(path, "w") as f:
        f.write("# radial lens table (lens_radial.py; format in its docstring)\n")
        for ln in extra:
            f.write(f"# {ln}\n")
        f.write(f"{MAGIC} 1\n")
        f.write("camera " + " ".join(str(v) for v in (W, H)) + " " + " ".join(repr(float(v)) for v in (fx, fy, cx, cy, *k)) + "\n")
        f.write(f"nodes {n} {float(dth)!r} {float(te)!r}\n")
        f.write(f"valid {float(valid)!r}\n")
        f.write(f"extrap {float(A)!r} {float(B)!r}\n")
        if centre is not None:
            f.write(f"centre {float(centre)!r}\n")
        f.write(f"source {source}\n")
        for i, r in enumerate(seg):
            f.write(f"seg {i} " + " ".join(repr(float(v)) for v in r) + "\n")
        for i in range(n):
            f.write(f"node {i} " + " ".join(repr(float(v)) for v in (theta[i], P[i], P1[i], P2[i], theta[i] * P[i])) + "\n")
    return path


def write_cameras(out_dir, W, H, fx, fy, cx, cy, k, note):
    cam_line = f"1 OPENCV_FISHEYE {W} {H} " + " ".join(repr(float(v)) for v in (fx, fy, cx, cy, *k)) + "\n"
    with open(f"{out_dir}/cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                f"# {note}\n"
                "# (use with --lens-radial lens_radial.txt; without the table, k1..k4 are only an approximation of the lens)\n"
                + cam_line)


def fit_k(thetad, t_hi, W, H, fx, fy, cx, cy, valid):
    """least-squares k1..k4 of theta_d ~ theta (1 + k1 theta^2 + .. + k4 theta^8) over the frame's pixels (every 8th, at
    their centres) whose ray is inside 'valid'; thetad(theta) is the profile, increasing on [0, t_hi] up to its first
    turning point. Returns the k's and the error theta_d(fit) - theta_d at the sampled pixels."""
    yy, xx = np.mgrid[0:H:8, 0:W:8] + 4.0
    rd = np.hypot((xx - cx) / fx, (yy - cy) / fy).ravel()
    tt = np.linspace(1e-6, t_hi, 200001)
    dd = thetad(tt)
    ok = np.nonzero(np.diff(dd) <= 0)[0]
    last = ok[0] if len(ok) else len(dd) - 1
    rd = rd[rd < dd[last]]                         # pixels beyond the profile's range (np.interp would clamp them)
    ths = np.interp(rd, dd[1:last + 1], tt[1:last + 1])
    ths = ths[ths <= valid]
    Am = np.stack([ths ** 3, ths ** 5, ths ** 7, ths ** 9], 1)
    kf = np.linalg.lstsq(Am, thetad(ths) - ths, rcond=None)[0]
    t2 = ths ** 2
    err = ths * (1 + t2 * (kf[0] + t2 * (kf[1] + t2 * (kf[2] + t2 * kf[3])))) - thetad(ths)
    return kf, err


def build_rspl(results, out_dir, W, H, valid_deg, nodes=2049, centre_deg=1.0, model="rspl"):
    R = json.load(open(results))[model]
    g = R["params"]
    valid = math.radians(valid_deg)
    if not 0 < valid <= QMAX:
        raise SystemExit(f"--valid-deg {valid_deg}: must be in (0, {math.degrees(QMAX):.2f}] (the B-spline's range)")
    theta = np.linspace(0.0, valid, nodes)
    dth = theta[1]
    m = max(1, int(round(math.radians(centre_deg) / dth)))
    thc = m * dth                                                                   # the centre zone ends on a node
    (a1, a3, a5), cc = _odd_centre(g, thc)
    k = [g["k1"], g["k2"], g["k3"], g["k4"]]
    P = np.empty(nodes)
    P1 = np.empty(nodes)
    P2 = np.empty(nodes)
    inner = theta < thc - 0.5 * dth                                                 # nodes 0 .. m-1: the odd-polynomial centre
    u = theta[inner] ** 2
    P[inner] = 1 + a1 + u * (k[0] + a3 + u * (k[1] + a5 + u * (k[2] + u * k[3])))
    P1[inner] = k[0] + a3 + u * (2 * (k[1] + a5) + u * (3 * k[2] + u * 4 * k[3]))
    P2[inner] = 2 * (k[1] + a5) + u * (6 * k[2] + u * 12 * k[3])
    th = theta[~inner]
    P[~inner], P1[~inner], P2[~inner] = _P_from_thetad(rspl_thetad(g, th, 0), rspl_thetad(g, th, 1), rspl_thetad(g, th, 2), th)
    # the paired OPENCV_FISHEYE camera: the fit's fx, fy, cx, cy and an 8-value approximation of the profile
    fx, fy, cx, cy = g["fx"], g["fy"], g["cx"], g["cy"]
    kf, fit = fit_k(lambda t_: rspl_thetad(g, t_), QMAX, W, H, fx, fy, cx, cy, valid)
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.basename(results)
    extra = [f"from {src} [{model}]: fit RMS {R.get('rms', float('nan')):.3f} px, held-out {R.get('cv', float('nan')):.3f} px",
             f"rspl: fx {fx:.5f} fy {fy:.5f} cx {cx:.5f} cy {cy:.5f} k {k} q {[g[f'q{j}'] for j in range(NQ)]}",
             f"valid to {valid_deg} deg (table end); centre zone {math.degrees(thc):.4f} deg: a1 {a1:.6g} a3 {a3:.6g} a5 {a5:.6g} "
             f"(B-spline correction there: c {cc[0]:.4g} c' {cc[1]:.4g} c'' {cc[2]:.4g})",
             f"camera k's = least-squares 8-value approximation inside {valid_deg} deg: |error| median "
             f"{np.median(np.abs(fit)) * fx:.3f} px, p95 {np.percentile(np.abs(fit), 95) * fx:.3f} px, max {np.abs(fit).max() * fx:.3f} px"]
    write_table(f"{out_dir}/lens_radial.txt", (W, H, fx, fy, cx, cy, list(kf)), theta, P, P1, P2, valid, extra,
                f"rspl {src}", centre=thc)
    write_cameras(out_dir, W, H, fx, fy, cx, cy, kf,
                  f"fx fy cx cy of {src} [{model}]; k1..k4 = the 8-value approximation paired with lens_radial.txt")
    return f"{out_dir}/lens_radial.txt", extra


def read_profile(path):
    """-> (camera list [W, H, fx, fy, cx, cy, (k1..k4)] or None, theta, theta_d)"""
    cam, rows = None, []
    for ln in open(path):
        s = ln.split("#", 1)[0].strip()
        if not s:
            continue
        p = [x for x in re.split(r"[\s,;]+", s) if x]
        if p[0].lower() == "camera":
            cam = [float(x) for x in p[1:]]
            if len(cam) not in (6, 10):
                raise SystemExit(f"{path}: 'camera W H fx fy cx cy [k1 k2 k3 k4]' needs 6 or 10 values, has {len(cam)}")
            continue
        try:
            v = [float(x) for x in p]
        except ValueError:
            if rows:
                raise SystemExit(f"{path}: not a 'theta_rad theta_d_rad' line: {ln.strip()!r}")
            continue                                                            # a header line before the data
        if len(v) != 2:
            raise SystemExit(f"{path}: expected 2 values (theta_rad theta_d_rad), got {len(v)}: {ln.strip()!r}")
        rows.append(v)
    if len(rows) < 8:
        raise SystemExit(f"{path}: {len(rows)} samples; a profile needs at least 8")
    a = np.array(rows, np.float64)
    a = a[np.argsort(a[:, 0])]
    return cam, a[:, 0], a[:, 1]


def read_fisheye_camera(cameras):
    for ln in open(cameras):
        p = ln.split()
        if p and not ln.startswith("#") and len(p) >= 12 and p[1] == "OPENCV_FISHEYE":
            return [int(p[2]), int(p[3])] + [float(x) for x in p[4:12]]
    raise SystemExit(f"no OPENCV_FISHEYE camera in {cameras}")


def build_profile(profile, out_dir, cameras=None, valid_deg=None, nodes=2049):
    from scipy.interpolate import make_interp_spline
    cam, th, td = read_profile(profile)
    if cameras:
        cam = read_fisheye_camera(cameras)
    if cam is None:
        raise SystemExit(f"{profile}: no 'camera W H fx fy cx cy [k1 k2 k3 k4]' line; add one or pass --cameras")
    W, H = int(round(cam[0])), int(round(cam[1]))
    fx, fy, cx, cy = cam[2:6]
    keep = th > 0
    th, td = th[keep], td[keep]
    if np.any(np.diff(th) <= 0):
        raise SystemExit(f"{profile}: repeated theta values")
    if np.any(np.diff(td) <= 0):
        raise SystemExit(f"{profile}: theta_d is not increasing")
    valid = th[-1] if valid_deg is None else math.radians(valid_deg)
    if not 0 < valid < 0.5 * math.pi:
        raise SystemExit(f"valid {math.degrees(valid):.3f} deg: must be in (0, 90) deg")
    if valid > th[-1] * (1 + 1e-12):
        raise SystemExit(f"--valid-deg {valid_deg}: beyond the profile's last sample ({math.degrees(th[-1]):.4f} deg)")
    # P = theta_d / theta as a smooth function of u = theta^2: a quintic interpolating spline (C4)
    spl = make_interp_spline(th * th, td / th, k=5)
    theta = np.linspace(0.0, valid, nodes)
    u = theta * theta
    P, P1, P2 = spl(u), spl(u, 1), spl(u, 2)

    def thetad(t_):
        t_ = np.asarray(t_, np.float64)
        return t_ * spl(t_ * t_)
    if len(cam) == 10:
        kf = list(cam[6:10])
        kdesc = "camera k's as given (the profile's camera)"
    else:
        kf, fit = fit_k(thetad, th[-1], W, H, fx, fy, cx, cy, valid)
        kf = list(kf)
        kdesc = (f"camera k's = least-squares 8-value approximation inside {math.degrees(valid):.4f} deg: |error| median "
                 f"{np.median(np.abs(fit)) * fx:.3f} px, p95 {np.percentile(np.abs(fit), 95) * fx:.3f} px, max {np.abs(fit).max() * fx:.3f} px")
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.basename(profile)
    extra = [f"from the sampled profile {src}: {len(th)} samples, {math.degrees(th[0]):.4f} .. {math.degrees(th[-1]):.4f} deg; "
             f"P = theta_d / theta as a quintic spline in theta^2",
             f"valid to {math.degrees(valid):.4f} deg (table end)", kdesc]
    path = write_table(f"{out_dir}/lens_radial.txt", (W, H, fx, fy, cx, cy, kf), theta, P, P1, P2, valid, extra,
                       f"profile {src}")
    write_cameras(out_dir, W, H, fx, fy, cx, cy, kf, f"fx fy cx cy of {src}; k1..k4 paired with lens_radial.txt")
    # check: the table against the samples it was built from
    tab = RadialLens(path)
    s = th <= valid
    e = (tab.thetad(th[s]) - td[s]) * fx
    extra.append(f"table vs the profile's samples: max |d| {np.abs(e).max():.2e} px (x fx)")
    return path, extra


def build_poly(cameras, out_dir, end_deg=88.0, nodes=2049):
    c = read_fisheye_camera(cameras)
    W, H = c[0], c[1]
    prm = c[2:10]
    k = prm[4:8]
    te = math.radians(end_deg)
    theta = np.linspace(0.0, te, nodes)
    u = theta ** 2
    P = 1 + u * (k[0] + u * (k[1] + u * (k[2] + u * k[3])))
    P1 = k[0] + u * (2 * k[1] + u * (3 * k[2] + u * 4 * k[3]))
    P2 = 2 * k[1] + u * (6 * k[2] + u * 12 * k[3])
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.basename(cameras)
    path = write_table(f"{out_dir}/lens_radial.txt", (W, H, *prm[:4], k), theta, P, P1, P2, te,
                       [f"the OPENCV_FISHEYE lens's own curve ({src}), to {end_deg} deg: a test table"], f"poly {src}")
    write_cameras(out_dir, W, H, *prm[:4], k, f"the camera of {src} (unchanged)")
    return path


# ----------------------------------------------------------------------------------------------- reading and evaluating
class RadialLens:
    """A radial lens table (see the module docstring), evaluated in numpy float64."""

    def __init__(self, path):
        self.path = path
        hdr, seg, node = {}, [], []
        for ln in open(path):
            p = ln.split()
            if not p or p[0].startswith("#"):
                continue
            if p[0] == "seg":
                seg.append([float(x) for x in p[2:10]])
            elif p[0] == "node":
                node.append([float(x) for x in p[2:7]])
            elif p[0] == "source":
                hdr["source"] = ln.split(None, 1)[1].strip()
            else:
                hdr[p[0]] = p[1:]
        if hdr.get(MAGIC) != ["1"]:
            raise ValueError(f"{path}: not a '{MAGIC} 1' file")
        for key in ("camera", "nodes", "valid", "extrap"):
            if key not in hdr:
                raise ValueError(f"{path}: no '{key}' line")
        c = hdr["camera"]
        self.W, self.H = int(c[0]), int(c[1])
        self.intr = [float(x) for x in c[2:6]]
        self.k = [float(x) for x in c[6:10]]
        self.n = int(hdr["nodes"][0])
        self.dth = float(hdr["nodes"][1])
        self.th_end = float(hdr["nodes"][2])
        self.valid = float(hdr["valid"][0])
        self.A, self.B = map(float, hdr["extrap"])
        self.centre = float(hdr["centre"][0]) if "centre" in hdr else 0.0
        self.source = hdr.get("source", "")
        self.seg = np.array(seg, np.float64)
        self.node = np.array(node, np.float64)
        if self.seg.shape != (self.n - 1, 8) or (len(node) and self.node.shape != (self.n, 5)):
            raise ValueError(f"{path}: {len(seg)} seg / {len(node)} node rows for {self.n} nodes")
        if abs((self.n - 1) * self.dth - self.th_end) > 1e-9 * self.th_end:
            raise ValueError(f"{path}: nodes {self.n} x dtheta {self.dth} != theta_end {self.th_end}")
        self.inv_dth = 1.0 / self.dth

    def __repr__(self):
        return (f"radial lens table {self.path} ({self.n} nodes to {math.degrees(self.th_end):.2f} deg, valid "
                f"{math.degrees(self.valid):.2f} deg; camera fx {self.intr[0]:.4f} k {[round(v, 8) for v in self.k]})")

    def matches(self, k, intr=None, rel=1e-6):
        """True if k (4 values) are this table's paired k's (and intr = full-res fx, fy, cx, cy, if given, its intrinsics)"""
        try:
            kv = [float(x) for x in list(k)[:4]]
        except (TypeError, ValueError):
            return False
        if len(kv) != 4 or not all(abs(a - b) <= 1e-9 + rel * abs(b) for a, b in zip(kv, self.k)):
            return False
        return intr is None or all(abs(float(a) - b) <= 1e-4 * abs(b) for a, b in zip(list(intr)[:4], self.intr))

    def P_terms(self, u):
        """P, dP/du, d2P/du2 at u = theta^2 (numpy, float64)"""
        u = np.asarray(u, np.float64)
        th = np.sqrt(np.maximum(u, 0))
        i = np.minimum((th * self.inv_dth).astype(np.int64), self.n - 2)
        r = self.seg[np.maximum(i, 0)]
        t = (u - r[..., 0]) * r[..., 1]
        c = [r[..., 2 + j] for j in range(6)]
        P = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))))
        Pt = c[1] + t * (2 * c[2] + t * (3 * c[3] + t * (4 * c[4] + t * 5 * c[5])))
        Ptt = 2 * c[2] + t * (6 * c[3] + t * (12 * c[4] + t * 20 * c[5]))
        far = th >= self.th_end
        thf = np.where(far, th, 1.0)
        P = np.where(far, self.A / thf + self.B, P)
        P1 = np.where(far, -self.A / (2 * thf ** 3), Pt * r[..., 1])
        P2 = np.where(far, 3 * self.A / (4 * thf ** 5), Ptt * r[..., 1] ** 2)
        return P, P1, P2

    def thetad(self, th):
        th = np.asarray(th, np.float64)
        return th * self.P_terms(th * th)[0]

    def dthetad(self, th):
        th = np.asarray(th, np.float64)
        P, P1, _ = self.P_terms(th * th)
        return P + 2 * th * th * P1

    def theta_of(self, td, hi=math.radians(88)):
        """inverse of theta_d (bisection, 60 halvings; theta_d is increasing), numpy"""
        td = np.asarray(td, np.float64)
        lo_ = np.zeros_like(td)
        hi_ = np.full_like(td, hi)
        for _ in range(60):
            mid = 0.5 * (lo_ + hi_)
            big = self.thetad(mid) > td
            hi_ = np.where(big, mid, hi_)
            lo_ = np.where(big, lo_, mid)
        return 0.5 * (lo_ + hi_)


def check(tab):
    """the trainer's load checks (in float64 here): P(0) ~ 1, theta_d increasing over the table, B > 0"""
    P0 = float(tab.P_terms(0.0)[0])
    th = np.arange(4 * (tab.n - 1) + 1) * 0.25 * tab.dth
    s = tab.dthetad(th)
    bad = np.nonzero(~(s > 0))[0]
    msgs = [f"theta_d / theta at the axis {P0:.6f}" + ("" if 0.5 < P0 < 2.0 else "  FAILS (expected ~1)"),
            "theta_d increasing over the table" if not len(bad) else
            f"theta_d NOT increasing at {math.degrees(th[bad[0]]):.4f} deg  FAILS",
            f"extrapolation slope B {tab.B:.6f}" + ("" if tab.B > 0 else "  FAILS (must be > 0)")]
    return msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("profile", help="table from a sampled theta -> theta_d profile")
    p.add_argument("profile")
    p.add_argument("out_dir")
    p.add_argument("--cameras", default=None, help="COLMAP cameras.txt whose first OPENCV_FISHEYE camera to use")
    p.add_argument("--valid-deg", type=float, default=None, help="table end (default: the last sample's angle)")
    p.add_argument("--nodes", type=int, default=2049)
    p = sp.add_parser("rspl", help="table from an rspl calibration JSON")
    p.add_argument("results")
    p.add_argument("out_dir")
    p.add_argument("--size", required=True, help="the calibrated image's size, WxH")
    p.add_argument("--valid-deg", type=float, required=True, help="the largest angle the calibration has data for")
    p.add_argument("--nodes", type=int, default=2049)
    p.add_argument("--centre-deg", type=float, default=1.0)
    p.add_argument("--model", default="rspl", help="the JSON's top-level key")
    p = sp.add_parser("poly", help="test table of an OPENCV_FISHEYE camera's own polynomial")
    p.add_argument("cameras")
    p.add_argument("out_dir")
    p.add_argument("--end-deg", type=float, default=88.0)
    p.add_argument("--nodes", type=int, default=2049)
    p = sp.add_parser("info", help="print a table's summary and theta_d at some angles")
    p.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "rspl":
        W, H = (int(s) for s in a.size.lower().split("x"))
        path, extra = build_rspl(a.results, a.out_dir, W, H, a.valid_deg, nodes=a.nodes, centre_deg=a.centre_deg, model=a.model)
        print(path)
        print("\n".join(extra))
    elif a.cmd == "profile":
        path, extra = build_profile(a.profile, a.out_dir, cameras=a.cameras, valid_deg=a.valid_deg, nodes=a.nodes)
        print(path)
        print("\n".join(extra))
    elif a.cmd == "poly":
        path = build_poly(a.cameras, a.out_dir, end_deg=a.end_deg, nodes=a.nodes)
        print(path)
    else:
        path = a.file
    t = RadialLens(path)
    print(t)
    print("  " + "; ".join(check(t)))
    if a.cmd == "info":
        degs = sorted(set([0.1, 1.0] + list(range(10, int(math.degrees(t.th_end)) + 1, 10)) +
                          [round(math.degrees(t.valid), 4), round(math.degrees(t.th_end), 4), round(math.degrees(t.th_end) + 2, 4)]))
        for d in degs:
            v = float(t.thetad(math.radians(d)))
            print(f"  {d:7.3f} deg: theta_d {v:.7f} ({v * t.intr[0]:9.3f} px)")


if __name__ == "__main__":
    main()
