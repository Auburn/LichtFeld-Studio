#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Per-image rolling-shutter motion for the trainer's --rolling-shutter FILE, differentiated from a camera path.

Usage:
  python rs_motion.py SPARSE PATH OUT.txt (--fps F | --frames-csv CSV) [--half 3] [--max-gap-s 0.4]
                      [--frame-regex REGEX] [--zero] [--rotation-only] [--scale K]

SPARSE  the training dataset's COLMAP text model (a folder with images.txt). One output line is written per image.
        Each image's pose is taken as its MID-READOUT pose, which is how the trainer uses it.
PATH    a COLMAP text model of the camera path the motion is differentiated from: registered video frames in the
        same world frame as SPARSE (a denser registration of the same video, or SPARSE itself). An image whose frame
        is also in PATH is left out of its own fit, and its two poses are compared (a warning names the worst mismatch).

Frame numbers come from the image names (the last match of --frame-regex, default 'frame_(\d+)': 'frame_00123.jpg'
-> 123). Images whose name does not match are skipped. A frame's time is frame / fps, with --fps, or --frames-csv: a
CSV with the columns 'frame' and 'time_s', fps = the last row's frame / its time_s.

Method. For every image f: a polynomial in time t through the +-N (--half) nearest other PATH frames within
--max-gap-s seconds, with no constant term, so it passes exactly through the image's own pose at t = 0:
  rotation  r_k = log(R_f^T R_k)    (rotation vector in the image's camera axes)
  position  p_k = c_k - c_f         (world)
fitted by least squares, degree 3 with neighbours on both sides and at least 4 in all, degree 2 with both sides and
2-3 in all, degree 1 (constant velocity) with one side only; no neighbour at all gives zero motion. Its derivatives at
t = 0 are the motion within the frame:
  omega = dr/dt,  alpha = d2r/dt2              camera axes, rad/s and rad/s^2
  vel = R_f^T dp/dt,  acc = R_f^T d2p/dt2      camera axes, scene units/s and units/s^2
The trainer draws the row read at time t (t = (v/H - 1/2) tau, top row first, tau = --rs-readout-ms) through
  c2w(t) = [R_f exp([omega t + alpha t^2 / 2]x) | c_f + R_f (vel t + acc t^2 / 2)].

Output (text, '#' comments): one line per SPARSE image, in frame order, values printed with %.9g:
  name wx wy wz ax ay az vx vy vz Ax Ay Az
Options:
  --zero           all motion 0 (a global-shutter control through the rolling-shutter code path)
  --rotation-only  vel = acc = 0
  --scale K        multiply all motion by K (synthetic tests)
"""
import argparse
import csv
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_rot(w, x, y, z):
    return R.from_quat([x, y, z, w]).as_matrix()


def frame_number(name, rx):
    m = list(rx.finditer(str(name)))
    return int(m[-1].group(1)) if m else None


def read_poses(images_txt, rx):
    """images.txt -> {name: c2w 4x4} (COLMAP world-to-camera quaternion + translation, inverted); names without a frame
    number are skipped"""
    out = {}
    lines = [ln for ln in open(images_txt) if not ln.startswith("#")]
    for m in lines[0::2]:
        p = m.split()
        if len(p) < 10 or frame_number(p[9], rx) is None:
            continue
        Rw = quat_rot(*map(float, p[1:5]))
        t = np.array(list(map(float, p[5:8])))
        c2w = np.eye(4)
        c2w[:3, :3] = Rw.T
        c2w[:3, 3] = -Rw.T @ t
        out[p[9]] = c2w
    return out


def fps_from_csv(path):
    rows = [(int(r["frame"]), float(r["time_s"])) for r in csv.DictReader(open(path))]
    return rows[-1][0] / rows[-1][1]


def motion(f, c2w, P, F, half, max_gap):
    """(omega, alpha, vel, acc) in the frame's camera axes, and the number of neighbours used on each side"""
    Rf, cf = c2w[:3, :3], c2w[:3, 3]
    ks = sorted(k for k in P if k != f)
    before = [k for k in ks if k < f and (f - k) / F <= max_gap][-half:]
    after = [k for k in ks if k > f and (k - f) / F <= max_gap][:half]
    nb = before + after
    if not nb:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0, 0
    t = np.array([(k - f) / F for k in nb])
    r = np.array([R.from_matrix(Rf.T @ P[k][:3, :3]).as_rotvec() for k in nb])
    c = np.array([P[k][:3, 3] - cf for k in nb])
    deg = 3 if (before and after and len(nb) >= 4) else (2 if (before and after and len(nb) >= 2) else 1)
    T = np.stack([t ** (j + 1) for j in range(deg)], 1)
    rc = np.linalg.lstsq(T, r, rcond=None)[0]
    pc = np.linalg.lstsq(T, c, rcond=None)[0]
    om = rc[0]
    al = 2 * rc[1] if deg >= 2 else np.zeros(3)
    v = pc[0]
    a = 2 * pc[1] if deg >= 2 else np.zeros(3)
    return om, al, Rf.T @ v, Rf.T @ a, len(before), len(after)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sparse", help="the training dataset's COLMAP text model (folder with images.txt)")
    ap.add_argument("path", help="COLMAP text model of the camera path (may be the same folder as SPARSE)")
    ap.add_argument("out", help="output motion file")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fps", type=float, help="video frame rate (frames per second)")
    g.add_argument("--frames-csv", help="CSV with columns 'frame' and 'time_s'; fps = last frame / its time_s")
    ap.add_argument("--half", type=int, default=3, help="neighbours used on each side (default 3)")
    ap.add_argument("--max-gap-s", type=float, default=0.4, help="largest time to a neighbour, seconds (default 0.4)")
    ap.add_argument("--frame-regex", default=r"frame_(\d+)", help="regex whose group 1 is the frame number")
    ap.add_argument("--zero", action="store_true", help="write zero motion for every image")
    ap.add_argument("--rotation-only", action="store_true", help="write vel = acc = 0")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all motion by K")
    a = ap.parse_args()
    rx = re.compile(a.frame_regex)
    F = a.fps if a.fps is not None else fps_from_csv(a.frames_csv)
    train = read_poses(os.path.join(a.sparse, "images.txt"), rx)
    reg = {frame_number(n, rx): p for n, p in read_poses(os.path.join(a.path, "images.txt"), rx).items()}
    if not train:
        raise SystemExit(f"{a.sparse}/images.txt: no image name matches {a.frame_regex!r}")
    lines, stats, one_side, none_, mism = [], [], 0, 0, []
    for name in sorted(train, key=lambda n: frame_number(n, rx)):
        f = frame_number(name, rx)
        c2w = train[name]
        if f in reg:  # the path's own pose of this frame should be the training pose (the same camera set)
            dR = np.degrees(np.linalg.norm(R.from_matrix(c2w[:3, :3].T @ reg[f][:3, :3]).as_rotvec()))
            dc = np.linalg.norm(c2w[:3, 3] - reg[f][:3, 3])
            if dR > 0.01 or dc > 1e-3:
                mism.append((name, dR, dc))
        P = dict(reg)
        P.pop(f, None)
        om, al, v, acc, nb_, na_ = motion(f, c2w, P, F, a.half, a.max_gap_s)
        if nb_ == 0 and na_ == 0:
            none_ += 1
        elif nb_ == 0 or na_ == 0:
            one_side += 1
        if a.zero:
            om, al, v, acc = (np.zeros(3),) * 4
        if a.rotation_only:
            v, acc = np.zeros(3), np.zeros(3)
        om, al, v, acc = (x * a.scale for x in (om, al, v, acc))
        lines.append(name + " " + " ".join(f"{x:.9g}" for x in (*om, *al, *v, *acc)))
        stats.append((np.degrees(np.linalg.norm(om)), np.degrees(np.linalg.norm(al)), np.linalg.norm(v), np.linalg.norm(acc)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fo:
        fo.write(f"# rolling-shutter motion (rs_motion.py; half {a.half}, max gap {a.max_gap_s} s, fps {F:.4f}"
                 f"{', ZERO' if a.zero else ''}{', rotation only' if a.rotation_only else ''}"
                 f"{f', scale {a.scale:g}' if a.scale != 1 else ''})\n")
        fo.write("# name wx wy wz ax ay az vx vy vz Ax Ay Az  (mid-readout camera axes: rad/s, rad/s^2, units/s, units/s^2)\n")
        fo.write("\n".join(lines) + "\n")
    s = np.array(stats)
    q = lambda c: " / ".join(f"{np.percentile(s[:, c], p):.3g}" for p in (50, 90, 99))  # noqa: E731
    print(f"{a.out}: {len(lines)} frames; turn rate deg/s (median / p90 / p99) {q(0)}; angular accel deg/s^2 {q(1)}; "
          f"speed u/s {q(2)}; accel u/s^2 {q(3)}; {one_side} one-sided fits, {none_} without neighbours")
    if mism:
        print(f"WARNING: {len(mism)} frames' training pose differs from the path's (worst "
              f"{max(mism, key=lambda x: x[1])[0]}: {max(m[1] for m in mism):.4f} deg, {max(m[2] for m in mism):.5f} units)")


if __name__ == "__main__":
    main()
