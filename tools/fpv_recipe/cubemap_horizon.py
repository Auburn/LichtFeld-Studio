#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A fixed horizon cube for the trainer's --bg-cubemap-init: a background cubemap kept below the horizon and faded to a
solid grey above it.

Usage:
  python cubemap_horizon.py SRC OUT --up UP [--sparse SPARSE] [--lo -25] [--hi 25] [--grey 249]

SRC    the cube to keep below the horizon: a folder with faces.npy (or the six face PNGs px nx py ny pz nz), or a
       faces.npy file. For example the trainer's own export <output>/bg_cubemap/faces.npy from a run with
       --bg-mode cubemap, or any cube in the same layout (e.g. one painted from renders of a trained model).
OUT    output folder: OUT/faces.npy and the six face PNGs.
UP     the scene's up direction in the dataset's (COLMAP) world frame, which sets the elevation of every texel:
         x,y,z     a vector (write --up=x,y,z when it starts with a minus sign)
         FILE      a JSON file with "up": [x, y, z]
         cams      the mean 'up' of the cameras in --sparse (each camera's -y axis in world)
       The cameras' mean up is only an estimate of the true vertical: a camera tilted up on the drone, and the drone's
       average pitch and roll, tilt it (by several degrees on FPV footage), which tilts the horizon of the blend. A
       measured vertical (e.g. from the flight controller's accelerometer, or from vertical structures such as tree
       trunks) is better.
SPARSE a COLMAP text model (folder with images.txt): needed for --up cams; otherwise only used to report the angle
       between UP and the cameras' mean up.

The grey share w rises smoothly (smoothstep) from 0 at elevation LO degrees to 1 at HI degrees:
  x = clip((el - lo) / (hi - lo), 0, 1),  w = x^2 (3 - 2x),  out = (1 - w) src + w (grey / 255)
where el = asin(d . up) is the elevation of the texel centre's direction d.

Cube layout (the trainer's --bg-cubemap-init / bg_cubemap export): float32 [6, R, R, 3], RGB in 0..1 (the photos'
encoding), faces +X -X +Y -Y +Z -Z (px nx py ny pz nz) in the dataset's world axes. A direction is looked up on the face
of its largest |component| with
  +X: s = -z/|x|, t = -y/|x|   -X: s = z/|x|, t = -y/|x|   +Y: s = x/|y|, t = z/|y|   -Y: s = x/|y|, t = -z/|y|
  +Z: s = x/|z|, t = -y/|z|    -Z: s = -x/|z|, t = -y/|z|
column (s + 1) / 2 * R, row (t + 1) / 2 * R (row 0 at the top), texel centres at half-integers.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

FACES = ("px", "nx", "py", "ny", "pz", "nz")


def load_faces(src):
    """-> float32 (6, R, R, 3) in 0..1, from faces.npy (a file, or in a folder) or the six face PNGs"""
    p = src if src.endswith(".npy") else os.path.join(src, "faces.npy")
    if os.path.exists(p):
        a = np.load(p).astype(np.float32)
        if not (a.ndim == 4 and a.shape[0] == 6 and a.shape[1] == a.shape[2] and a.shape[3] == 3):
            raise SystemExit(f"{p}: shape {a.shape}, expected [6, R, R, 3]")
        return a
    out = []
    for f in FACES:
        im = cv2.imread(os.path.join(src, f + ".png"), cv2.IMREAD_COLOR)
        if im is None:
            raise SystemExit(f"{src}: no faces.npy and no {f}.png")
        out.append(im[..., ::-1].astype(np.float32) / 255.0)
    return np.stack(out)


def face_to_dir(f, s, t):
    """direction (not normalised) of face f at face coordinates s, t in [-1, 1]"""
    one = np.ones_like(s)
    return {0: np.stack([one, -t, -s], -1), 1: np.stack([-one, -t, s], -1), 2: np.stack([s, one, t], -1),
            3: np.stack([s, -one, -t], -1), 4: np.stack([s, -t, one], -1), 5: np.stack([-s, -t, -one], -1)}[f]


def texel_dirs(R):
    """unit world directions of every texel centre, (6, R, R, 3) float64"""
    c = (np.arange(R) + 0.5) / R * 2 - 1
    t, s = np.meshgrid(c, c, indexing="ij")
    out = []
    for f in range(6):
        d = face_to_dir(f, s, t)
        out.append(d / np.linalg.norm(d, axis=-1, keepdims=True))
    return np.stack(out)


def solid_angle_weights(R):
    """per-texel solid angle (6, R, R), sums to 4 pi"""
    c = (np.arange(R) + 0.5) / R * 2 - 1
    t, s = np.meshgrid(c, c, indexing="ij")
    w = (2.0 / R) ** 2 / (1 + s * s + t * t) ** 1.5
    return np.broadcast_to(w, (6, R, R))


def quat_to_rot(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mean_up(sparse):
    """the cameras' mean 'up' (the -y camera axis in world) of a COLMAP text model, and the number of cameras"""
    ups = []
    for ln in [ln for ln in open(os.path.join(sparse, "images.txt")) if not ln.startswith("#")][0::2]:
        q = ln.split()
        if len(q) < 10:
            continue
        Rw = quat_to_rot(*map(float, q[1:5]))
        ups.append(-Rw[1])                                   # row 1 of world-to-camera = camera y in world; up = -y
    if not ups:
        raise SystemExit(f"{sparse}/images.txt: no cameras")
    u = np.mean(ups, 0)
    return u / np.linalg.norm(u), len(ups)


def scene_up(spec, sparse=None):
    """(unit up, description) from 'x,y,z', a JSON file with "up", or 'cams' (the cameras' mean up of sparse)"""
    cu, n = mean_up(sparse) if sparse else (None, 0)
    if spec == "cams":
        if cu is None:
            raise SystemExit("--up cams needs --sparse")
        return cu, f"the {n} cameras' mean up"
    if os.path.exists(spec):
        u = np.asarray(json.load(open(spec))["up"], float)
        src = os.path.basename(spec)
    else:
        try:
            u = np.asarray([float(x) for x in spec.split(",")], float)
        except ValueError:
            raise SystemExit(f"--up {spec!r}: not x,y,z, an existing JSON file or 'cams'")
        if u.shape != (3,):
            raise SystemExit(f"--up {spec!r}: needs 3 values")
        src = "given"
    u = u / np.linalg.norm(u)
    if cu is None:
        return u, src
    ang = float(np.degrees(np.arccos(np.clip(abs(u @ cu), -1, 1))))
    return u, f"{src} ({ang:.1f} deg from the cameras' mean up)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="source cube: folder with faces.npy (or the six PNGs), or a faces.npy file")
    ap.add_argument("out", help="output folder")
    ap.add_argument("--up", required=True, help="x,y,z | JSON file with \"up\" | cams")
    ap.add_argument("--sparse", default=None, help="COLMAP text model (for --up cams, and to report the angle to it)")
    ap.add_argument("--lo", type=float, default=-25.0, help="elevation (deg) below which the source is kept (default -25)")
    ap.add_argument("--hi", type=float, default=25.0, help="elevation (deg) above which it is all grey (default 25)")
    ap.add_argument("--grey", type=float, default=249.0, help="the grey level, 0..255 (default 249)")
    a = ap.parse_args()
    if not a.hi > a.lo:
        raise SystemExit("--hi must be above --lo")
    lo, hi, grey = a.lo, a.hi, a.grey
    F = load_faces(a.src)
    R = F.shape[1]
    up, updesc = scene_up(a.up, a.sparse)
    d = texel_dirs(R)
    el = np.degrees(np.arcsin(np.clip((d * up).sum(-1), -1, 1)))
    x = np.clip((el - lo) / (hi - lo), 0, 1)
    w = (x * x * (3 - 2 * x))[..., None]          # 0 = source, 1 = grey
    G = (1 - w) * F + w * (grey / 255.0)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "faces.npy"), G.astype(np.float32))
    for i, f in enumerate(FACES):
        cv2.imwrite(os.path.join(a.out, f"{f}.png"), (np.clip(G[i], 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))
    sa = solid_angle_weights(R)
    for e0, e1 in ((-90, lo), (lo, 0), (0, hi), (hi, 90)):
        m = (el >= e0) & (el < e1)
        if not m.any():
            continue
        c = (G[m] * sa[m][:, None]).sum(0) / sa[m].sum() * 255
        print(f"  {e0:+6.1f} .. {e1:+5.1f} deg: mean RGB ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}), "
              f"grey share {float((w[..., 0][m] * sa[m]).sum() / sa[m].sum()):.2f}")
    print(f"{a.out}: {a.src} below {lo:g} deg, {grey:g} grey above {hi:g} deg (smoothstep between); "
          f"up {np.round(up, 3).tolist()} = {updesc}")


if __name__ == "__main__":
    main()
