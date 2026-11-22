# FPV splat recipe: what this branch adds

This branch is **LichtFeld Studio 0.5.4.2** (upstream `fac5e49`) plus a set of training features built to turn
**raw FPV drone footage** into Gaussian splats for a **real-time game renderer**.

That footage is hard on a standard 3DGS trainer:

- an ultra-wide fisheye lens;
- a slow rolling shutter on a fast-moving camera;
- motion blur;
- auto exposure that drifts and gets confused by sun flare;
- a large open scene with sky.

The game also has a fixed frame-time budget, so the splats have to be cheap to draw as well as accurate.

**Every feature is opt-in.** With none of the flags below set, the trainer behaves as upstream. The flags work
headless (`--headless`), and most also work through a `--config` JSON.

## At a glance

| Area | Flags | What it fixes |
|---|---|---|
| [Fisheye lens](#native-fisheye-training) | `--native-fisheye`, `--lens-radial` | Trains on the raw fisheye frames, with no undistortion crop or resampling |
| [Camera poses](#pose-refinement) | `--pose-opt` | SfM pose error, refined per camera during training |
| [Rolling shutter](#rolling-shutter) | `--rolling-shutter`, `--rs-learn`, `--rs-time-map` | Bent, doubled geometry from a 30 ms readout on a fast camera |
| [Motion blur](#motion-blur) | `--motion-blur`, `--exposure-ms` | Blurred frames pulling splats fat |
| [Exposure and colour](#appearance-ppisp-locks-and-the-exposure-guard) | `--ppisp-lock-*`, `--ppisp-exposure-guard`, `--loss-ycc` | Per-frame appearance that drifts or soaks up real scene colour |
| [Sky](#learned-background-cubemap) | `--bg-mode cubemap` | Sky and far background filled with floaters |
| [Kernel shape](#kernel-shape) | `--kernel-shape` | Gaussians that can't make a hard edge (leaves, branches) |
| [Matching the game](#the-games-cut) | `--max-std-dev`, `--max-std-dev-distance` | Training draws a splat differently from the game |
| [Draw cost](#footprint-loss-draw-cost) | `--loss-footprint*` | Overdraw the game can't afford |
| [Splat shape](#geometry-regularisers) | `--thick-*`, `--aniso-*`, `--distortion-*`, `--opacity-binarize-*` | Needles, paper-thin discs, smeared depth, half-transparent fog |
| [Floaters](#near-camera-gradient-scaling-floaters) | `--grad-scale-near` | Splats grown right in front of the camera |
| [Densification](#densification-options) | `--growth-visit-*`, `--screen-share-axis` | Rarely seen areas losing out on growth; long thin splats being shrunk |
| [Extra supervision](#extra-supervision) | `--loss-weight-dir`, `--generated-*` | Per-pixel weighting; generated (non-photo) training views |
| [Rendering](#render-path-through-a-fisheye) | `--render-fisheye` | Rendering a camera path through a fisheye lens |
| [Build](#build-one-binary-for-several-gpus) | `LFS_CUDA_ARCHS` | One binary for several GPU generations |
| [Tools](#headless-tools) | `lfe-render`, `lfe-register`, `lfe-*` | Scoring through the trainer's own renderer; self-tests |

**Iteration numbers.** The default start iterations below are for a 30k-step run, and they scale with
`--steps-scaler`. A value you pass on the command line is used as an absolute iteration in this build. For example, the
recipe's `--distortion-start 37500` is 10 % of its 375k steps.

---

## Cameras and lens

### Native fisheye training

`--native-fisheye` trains `OPENCV_FISHEYE` cameras as they are: the standard rasterizer projects each splat
through the fisheye lens itself, and the images and masks stay in fisheye space. This replaces `--undistort`, which
crops and resamples the frames, and `--gut`.

- The projection, its Jacobian, culling and tile bounds all follow the lens.
- MRNF's blob seeding follows the lens too.
- `LFS_FISHEYE_MAX_DEG` (default 88°) is the largest off-axis angle at which a splat centre is still drawn.

### Radial lens table

`--lens-radial FILE` replaces the `OPENCV_FISHEYE` k1–k4 polynomial with a **tabulated radial profile θ → θd**. This
is for lenses that a four-term polynomial can't follow out to the edge of the frame.

- The table is used everywhere the lens is: the projection and its Jacobian, the rolling-shutter row solve, the
  culling bound and blob seeding.
- It applies only to the camera the table was made for. Any other fisheye camera in the dataset is an error.

**File format** (text; `#` starts a comment):

```
LENS_RADIAL_TABLE 1
camera W H fx fy cx cy k1 k2 k3 k4     # the camera the table belongs to (as in cameras.txt)
nodes N dtheta theta_end                # N nodes uniform in theta (radians), theta_end = (N-1) dtheta < 90 deg
valid THETA                             # the angle up to which the profile was measured
extrap A B                              # beyond theta_end: theta_d = A + B theta
centre C                                # optional
seg 0 u0 inv_h c0 c1 c2 c3 c4 c5        # N-1 segments, in order
seg 1 ...
```

- θd = θ·P(u), with u = θ².
- On each segment, P is a quintic Hermite polynomial in t = (u − u0)·inv_h, so the profile is C2.
- The loader checks three things: θd/θ ≈ 1 on the axis, θd increasing over the whole table, and B > 0.

### Pose refinement

`--pose-opt` refines every training camera's pose during training. After each render, each camera takes its own Adam
step on its rotation and on an offset of its camera centre. It works on the native-fisheye standard rasterizer, with
or without the rolling shutter, but not with `--gut`.

| Flag | Default | Meaning |
|---|---|---|
| `--pose-lr-deg` | 0.01 | Rotation step, in degrees per camera update |
| `--pose-lr-units` | 5e-4 | Centre step, in scene units per camera update |
| `--pose-opt-start` | 3000 | First iteration that moves the cameras |
| `--pose-opt-hold` | 15000 | Full learning rate until here, then a smooth halving |
| `--pose-opt-half-life` | 0 (auto) | Iterations per halving; auto = (last − hold) / 7, so the rate ends at 1/128 |
| `--pose-prior` | 0 | Pull back to the start pose (offset measured in pixels) |
| `--pose-smooth` | 0 | Tie each camera to its time neighbours (adjacent images at most 12 video frames apart) |

Outputs:

- `cameras_refined/`: COLMAP text, in the splat's frame;
- `pose_refinement.tsv`: the per-camera moves;
- `pose_snapshots/`, only when `LFS_POSE_SNAPSHOTS=1` is set.

---

## Rolling shutter

FPV cameras read the sensor out one row at a time over tens of milliseconds, and the drone moves and turns a lot in
that time. `--rolling-shutter FILE` draws **every splat through the pose at the moment its sensor row was read**, and
the gradients flow through that time as well.

**Motion file**: one line per image, in the image's mid-readout camera axes (`#` comments allowed):

```
name  wx wy wz  ax ay az  vx vy vz  Ax Ay Az
```

- `wx wy wz`: angular velocity (rad/s);
- `ax ay az`: angular acceleration (rad/s²);
- `vx vy vz`: linear velocity (scene units/s);
- `Ax Ay Az`: linear acceleration (scene units/s²).

An image without a line is trained with a global shutter. The motion usually comes from the camera path (for example,
differentiated SfM poses or a flight controller's gyro log).

| Flag | Default | Meaning |
|---|---|---|
| `--rs-readout-ms` | 10 | Readout time, top row to bottom row |
| `--rs-learn LIST` | none | Refine any of `omega,alpha,vel,acc,readout` during training |
| `--rs-lr` | 0.05 | Per-camera Adam step of the motion correction, in pixels at the readout ends |
| `--rs-readout-lr` | 0.002 | Adam step of the shared readout time (ms per iteration) |
| `--rs-prior` | 1e-4 | Pull each correction back to zero |
| `--rs-smooth` | 0 | Tie each camera's correction to its time neighbours' |
| `--rs-learn-start` | 3000 | First iteration of the RS refinement |
| `--rs-motion-start` | 0 | With both readout and motion learned: learn the readout first, then the motion from here (avoids the readout/speed trade-off) |

The learned motion and readout are written to `rs_motion_learned.txt`, in the same format with a `# readout_ms` line.

### Per-pixel readout time (`--rs-time-map`)

Some cameras remap the sensor image radially after capture, so the lines of equal readout time are curved rather than
straight rows. The map can be measured by filming a fast-blinking LED. `--rs-time-map FILE` (needs
`--rolling-shutter`) reads each splat at the time of its centre's pixel, instead of using the row rule. The background
cubemap's rays follow the same rule.

```
LFS_RS_TIME_MAP 1
size W H                         # the image the form is defined on; any camera with this aspect uses it, scaled
radial K A B C CX CY             # t = K y (1 + A r^2 + B r^4 + C r^6), in ms
                                 #   x = (u - W/2)/(W/2) - CX, y = (v - H/2)/(W/2) - CY, r^2 = x^2 + y^2
# or: grid NX NY, followed by NY rows of NX values (ms), bilinear between the nodes
readout_ms R                     # optional: the form's linear-equivalent readout (default: fitted)
```

The time is normalised so that `--rs-readout-ms` keeps its meaning:

- s = (t(u, v) − t(centre)) / R, and the splat is drawn at s × `--rs-readout-ms`;
- R is the least-squares slope of t against the row, times H;
- a linear form reproduces the row rule exactly.

## Motion blur

`--motion-blur` (with `--rolling-shutter`) models each frame's exposure blur on top of the row times:

- each splat's 2D covariance is widened along its exposure smear (the camera's rotation and travel at that splat);
- its opacity is scaled down so that its total coverage stays the same;
- with `--kernel-shape`, the smear is moment-matched to the splat's β.

The model learns sharp splats that blur to match the blurred photos, rather than learning the blur itself.

| Flag | Default | Meaning |
|---|---|---|
| `--exposure-ms` | 2 | Exposure time (1/500 s = 2 ms); 0 = no blur |
| `--motion-blur-learn-exposure` | off | Learn a per-camera exposure scale (start 1, clamped to [0.5, 2]); written to `motion_blur_exposure.txt` |
| `--motion-blur-exposure-lr` | 0.01 | Its Adam step |

---

## Appearance: PPISP locks and the exposure guard

Upstream's `--ppisp` learns a per-frame exposure and colour (white balance) plus a per-camera vignette and response
curve. On long video, the per-frame terms can absorb real scene colour, or get stuck.

| Flag | Meaning |
|---|---|
| `--ppisp-lock-frames` | Keep every frame's exposure and colour at its initial value (identity, or EXIF-seeded). Only the per-camera vignette and response curve train. |
| `--ppisp-lock-colour` | Keep every frame's colour (white point and primaries) at identity, i.e. the footage as shot; per-frame exposure still trains. **Used by the recipe.** |
| `--ppisp-exposure-guard EV` | After every PPISP step, a frame whose learned exposure is more than `EV` off the median of its ±6 temporal neighbours is projected back to that median, and its exposure momentum (Adam first moment) is cleared. Catches frames that run away, e.g. after sun flare. **The recipe uses 0.7.** |
| `--ppisp-reg-weight` | The PPISP regularisation weight (default 0.001; the recipe uses 1). |

The final PPISP state is dumped to `ppisp_final.txt`. `lfe-render --ppisp` reads it back.

### Y'CbCr loss

`--loss-ycc` computes the photometric loss in BT.709 full-range Y'CbCr instead of RGB:

- L1 + D-SSIM on luma, at full resolution;
- L1 on Cb and Cr after a 2×2 box downsample, because 4:2:0 video carries no finer colour.

The loss sees the image after the cubemap composite and PPISP. Masks keep only the 2×2 blocks that are fully kept.

| Flag | Default | Meaning |
|---|---|---|
| `--loss-ycc-chroma-weight` | 1 | Chroma L1 relative to luma L1 |
| `--loss-ycc-luma-scale` | 1 | Scales the luma L1 to match the RGB L1's magnitude |

---

## Learned background cubemap

`--bg-mode cubemap` learns a **sky cube** behind the splats. A pixel's colour is the splats plus (1 − α) times the
cube, looked up along the pixel's ray. That ray goes through the radial lens, the refined pose and the rolling shutter,
exactly as the splats are drawn. Without the cube, the sky gets filled with far floaters.

| Flag | Default | Meaning |
|---|---|---|
| `--bg-cubemap-res` | 256 | Texels per face edge (6 × res × res RGB) |
| `--bg-cubemap-lr` | 0.002 | Adam learning rate at iteration 0; 0 freezes the cube |
| `--bg-cubemap-lr-final` | 0.1 | The rate decays exponentially to lr × this factor |
| `--bg-cubemap-smooth` | 100 | Neighbouring-texel smoothness weight (keeps it sky-like and low-frequency) |
| `--bg-cubemap-init` | none | Start from a `faces.npy` (float32 `[6, R, R, 3]`, faces +X −X +Y −Y +Z −Z; resampled if R differs) |

The cube is exported to `bg_cubemap/faces.npy`. `LFS_CUBEMAP_DUMP=<iter>` also writes
`cubemap_dump/bg_<iter>.npy` at that iteration.

The recipe starts from a prepared horizon cube and freezes it (`--bg-cubemap-lr 0`), so the splats can't push detail
into the sky.

---

## Kernel shape

`--kernel-shape` gives every splat a **trainable kernel shape**, a generalized Gaussian:

- α = opacity · exp(−½ d^β), where d is the Mahalanobis distance in the 2D conic;
- β = 1 + 7·sigmoid(r − ln 6), which lies in [1, 8] and starts at 2 (the Gaussian);
- β > 2 gives a flat top and a sharp edge (leaves, branch edges); β < 2 gives a peak with long tails;
- the tile extent follows (2 ln(255·opacity))^(1/β).

It is exported as the PLY vertex property **`shape`** (β), and the PLY reader loads it back. A renderer that ignores
`shape` draws the splats as plain Gaussians.

It needs the standard rasterizer and MRNF. It can't be combined with `--enable-mip` or `--enable-sparsity`. It can be
combined with `--max-std-dev` only through `--max-std-dev-distance`.

| Flag | Default | Meaning |
|---|---|---|
| `--kernel-shape-lr` | 0.005 | Adam learning rate of the raw shape r; 0 freezes β at 2 |
| `--kernel-shape-start` | 0 | First iteration that learns the shape |

## The game's cut

A real-time renderer draws each splat as a quad cut at a few standard deviations. If the trainer draws more of the
tail than the game does, the model leans on tails that the game never shows.

- `--max-std-dev X` cuts every splat as the game does: a pixel takes part only within X standard deviations
  (Mahalanobis, in the 0.3-px-dilated 2D covariance) and above α 0.5/255. The cut applies in both the forward and the
  backward pass, and the tile radius shrinks to match. The target game draws at **√5 = 2.2360680**.
- `--max-std-dev-distance` (with `--kernel-shape`) makes the cut on distance, d ≤ X for any β, with bounds
  min(α contour, X). This matches a renderer that gives each splat a radius of min(contour, C). For a plain Gaussian
  it is identical to `--max-std-dev`.

## Footprint loss (draw cost)

`--loss-footprint λ` adds λ × **the mean number of splats the game draws per pixel** of the view. A splat's cost is
the area of its game quad: 4·min(512 px, R√λ1)·min(512 px, R√λ2) of the dilated 2D covariance, where:

- R = √5 for a Gaussian;
- R = min(√5, (2 ln(510·o))^(1/β)) for a shaped splat.

The gradient goes into the log-scales only; means, rotations and β get none. The loss trades a little image error for
a model the game can draw within its frame budget.

| Flag | Default | Meaning |
|---|---|---|
| `--loss-footprint-fraction F` | 0 | Instead of a fixed λ: set λ at the start so that the term starts at F × the image loss. **The recipe uses 0.025.** |
| `--loss-footprint-start` | 6000 | First iteration of the loss |
| `--loss-footprint-ramp` | 2000 | Linear ramp from 0 to λ |
| `--loss-footprint-opacity` | off | Also send the gradient into opacity, so that faint big splats fade out |

---

## Geometry regularisers

| Feature | Flags | Loss |
|---|---|---|
| **Thickness floor** | `--thick-max-ratio R --thick-weight λ [--thick-start] [--thick-longest]` | λ·mean max(0, ln(s_mid/s_min) − ln R). With `--thick-longest` it is measured against the longest axis, which bounds every splat's projected aspect at about R. Stops paper-thin discs that draw as hairlines and needles when seen edge-on. **The recipe uses R 10, λ 1, longest.** |
| **Anti-needle** (PhysGaussian) | `--aniso-max-ratio r --aniso-weight λ [--aniso-start]` | λ·mean max(0, s_max/s_mid − r). |
| **Depth distortion** (2DGS) | `--distortion-weight`, `--distortion-start`, `--distortion-near/far` (0.2/1000), `--distortion-depth distance\|z`, `--distortion-skip-plane-crossing k` (2), `--distortion-min-dist` | Pulls the splats seen through a pixel onto one depth. `distance` measures from the camera centre, which suits fisheye. Splats whose k-sigma extent crosses the camera plane, or that are nearer than min-dist, are left out per view, so that very close objects aren't widened into opaque walls. Standard rasterizer only. |
| **Opacity binarize** | `--opacity-binarize-weight`, `--opacity-binarize-start` (15000) | mean(o·(1 − o)): pushes every opacity toward 0 or 1. |
| MRNF decay rates (now on the command line) | `--opacity-decay` (0.004), `--scale-decay` (0.002) | Per-refine decay rates. |

## Near-camera gradient scaling (floaters)

`--grad-scale-near D0` applies *Floaters No More* (Philip & Deschaintre, 2023). Each training view's gradients on a
splat's own parameters are multiplied by min(1, (d/D0)²), where d is the splat's distance from that view's camera.
Splats right in front of one camera then can't grow fast enough to explain that one view.

## Densification options

| Flag | Meaning |
|---|---|
| `--growth-visit-normalize` | Rank MRNF growth candidates by their mean error signal per visit, instead of the window maximum, so that rarely seen splats compete on equal terms |
| `--growth-visit-stat mean\|weighted` | How to average the visits: `mean`, or `weighted` (Σs²/Σs, led by the closest views) |
| `--screen-share-axis max\|area` | Which radius the screen-share cap measures: the longest axis (upstream), or the area-equivalent radius of the two longest axes, so that long thin splats (straps, ropes, branches) aren't shrunk |

## Extra supervision

- **Per-pixel loss weights.** `--loss-weight-dir FOLDER` takes one `<image name>.png` per training image (16-bit PNG,
  weight = value / 10000; or float EXR/TIFF). The weights multiply the per-pixel L1 + SSIM and the densification error
  map. A missing map is weight 1. Use them to down-weight flare, propellers, or the drone's own frame.
- **Generated views.** Training images whose file name starts with `--generated-prefix` are treated as generated (for
  example, views made by an image model). Instead of L1 + SSIM, they get an alignment-tolerant loss: a low-pass L1
  plus a local-texture term.

  | Flag | Default | Meaning |
  |---|---|---|
  | `--generated-lowpass-sigma` | 2 px | Blur before the low-pass L1 |
  | `--generated-texture-radius` | 3 px | Box radius of the local luma standard deviation |
  | `--generated-lowpass-weight` | 1 | Weight of the low-pass L1 |
  | `--generated-texture-weight` | 1 | Weight of the texture term |
  | `--generated-pixel-weight` | 0 | Weight of a plain per-pixel L1 |
  | `--generated-loss-weight` | 1 | Scale of the whole generated-view loss |

## Render path through a fisheye

`--render-fisheye HFOV` (with `--render-camera-path`) renders the path through a centred equidistant fisheye with that
horizontal field of view, instead of the path's pinhole camera.

## Build: one binary for several GPUs

`-DLFS_CUDA_ARCHS="89;120"` builds one fat binary with real SASS for each listed SM (here an RTX 40-series and an
RTX 50-series card), instead of a single architecture. The entries must be integer SMs ≥ 70.

---

## Headless tools

These are subcommands of the main executable: `LichtFeld-Studio <tool> ...`.

### `lfe-render`: score through the trainer's own renderer

```
LichtFeld-Studio lfe-render --ply MODEL.ply [model options] (--serve | --views REQUESTS.jsonl | --colmap SPARSE_DIR)
```

It renders a trained model **exactly as training drew it**: the FastGS forward pass, the native fisheye and radial
lens, the rolling shutter (with the time map), the learned cubemap, PPISP and the game's cut. Scoring scripts then
don't need a third renderer of their own.

- **Model options:**
  - `--sh-degree D`;
  - `--lens-radial FILE`. An equidistant camera, such as a game lens, is drawn through the polynomial path with the
    table switched off for that view;
  - `--max-std-dev X`, `--max-std-dev-distance`;
  - `--bg R,G,B` in [0, 1], or `--bg-cubemap DIR|faces.npy`;
  - `--ppisp ppisp_final.txt`;
  - `--rs-time-map FILE`.
- **Modes:**
  - `--serve`: one JSON request per stdin line, each answered with `@@LFS ok <id> <ms>` or `@@LFS err <id> <why>`;
  - `--views FILE`: the same requests, read from a file;
  - `--colmap DIR`: one request per image of a COLMAP text model, with motion from `--motion`.
- **Request fields:**
  - `id`, `out`;
  - `camera {model, width, height, params}`, `camera_id`, `factor`;
  - `pose` (`qvec` + `tvec`, or `w2c`);
  - `motion[12]`, `readout_ms`, `rotation_only`;
  - `bg`, `cube`, `sh_degree`, `sphere [x, y, z, r]`;
  - `post` (`none | curve | curve_vig | frame`), `post_ev`, `frame`;
  - `save` (`image, post, alpha, depth`), `dtype` (`f32 | f16`).
- **Outputs**, as `<out>/<id>_<what>.npy`:
  - `image [H,W,3]` (before any post);
  - `post [H,W,3]`;
  - `alpha [H,W]`;
  - `depth [H,W]` (Σ w·d, not normalised by α).

The full reference is in the header of `src/app/lfe_render_tool.cpp`.

### `lfe-register`: register cameras against a frozen model

```
LichtFeld-Studio lfe-register DS MODEL.ply OUT [STEPS [POSE_LR]] [--max-std-dev X] [--sh-degree D] [--rs-time-map FILE]
```

Refines a dataset's camera poses against a fixed model (the splats never move), with an affine colour fit per camera.
Use it to score held-out views fairly, after their poses have settled.

- Writes `OUT/pose_deltas.npz`, `model/sparse/0`, `settle.tsv` and `percam.tsv`.
- Configured through `FC_*` / `POSE_*` environment variables, listed in the tool's header.

### Self-tests

Each feature has a developer check. Most are analytic-against-numeric gradient checks on small synthetic scenes, run
on the GPU; each tool's usage is in the header of its `src/app/lfe_*_tool.cpp`.

| Tool | Checks |
|---|---|
| `lfe-fisheye` | Native fisheye and radial lens |
| `lfe-rs` | Rolling shutter |
| `lfe-blur` | Motion blur |
| `lfe-pose` | Pose gradients |
| `lfe-shape` | Kernel shape |
| `lfe-footprint` | Footprint loss |
| `lfe-distortion` | Depth distortion |
| `lfe-thick` | Thickness floor |
| `lfe-aniso` | Anti-needle |
| `lfe-gradnear` | Near-camera gradient scaling |
| `lfe-lossweight` | Per-pixel loss weights |
| `lfe-genloss` | Generated-view loss |

---

## Environment variables

| Variable | Effect |
|---|---|
| `LFS_POSE_SNAPSHOTS=1` | With `--pose-opt`: write pose snapshots during training to `pose_snapshots/` |
| `LFS_CUBEMAP_DUMP=<iter>` | Write the background cube at that iteration to `cubemap_dump/bg_<iter>.npy` |
| `LFS_FISHEYE_MAX_DEG` | Largest off-axis angle at which a splat centre is drawn (default 88°) |
| `LFS_RASTER_STATS=1` | `lfe-fisheye`: print rasterizer statistics |
| `LFS_RS_ITERS` | Iterations of the rolling-shutter row solve (default 2) |
| `LFS_LENS_RADIAL`, `LFS_MAX_STD_DEV`, `LFS_MAX_STD_DEV_DISTANCE` | The same settings as the flags, for the `lfe-*` tools |
| `LFS_YCC_DUMP`, `LFS_PPISP_GUARD_INJECT` | Test hooks for the Y'CbCr loss and the exposure guard |

## Output files (in `-o`, besides the splat)

| File | Written by |
|---|---|
| `cameras_refined/` (COLMAP text), `pose_refinement.tsv` | `--pose-opt` |
| `rs_motion_learned.txt` | `--rs-learn` |
| `motion_blur_exposure.txt` | `--motion-blur-learn-exposure` |
| `ppisp_final.txt` | `--ppisp` (the final per-frame and per-camera state) |
| `bg_cubemap/faces.npy` | `--bg-mode cubemap` |
| PLY vertex property `shape` | `--kernel-shape` |

---

## The recipe

This is the full command that trains the current woodland FPV map: 4,367 raw 4096×3072 fisheye frames, 375k
iterations, a 16M splat cap, about 7.5 hours on an RTX 5090.

```bash
LFS_POSE_SNAPSHOTS=1 LichtFeld-Studio --headless -d <dataset> -o <out> \
  --strategy mrnf --mask-mode ignore --max-width 0 --export ply \
  --steps-scaler 12.5 --max-cap 16000000 \
  --config growth_exchange_config.json \
  --native-fisheye --lens-radial <lens_radial.txt> \
  --rolling-shutter <motion.txt> --rs-readout-ms 31 --rs-time-map <rs_time_map.txt> \
  --rs-learn omega,alpha,vel,acc --rs-lr 0.1 --rs-smooth 1e-3 \
  --motion-blur --exposure-ms 2.0 \
  --pose-opt \
  --no-exposure-correction --ppisp --ppisp-reg-weight 1 --ppisp-lock-colour --ppisp-exposure-guard 0.7 \
  --bg-color "(249,249,249)" --bg-mode cubemap --bg-cubemap-res 256 --bg-cubemap-init <horizon_faces.npy> --bg-cubemap-lr 0 \
  --kernel-shape --max-std-dev 2.2360680 --max-std-dev-distance \
  --loss-footprint-fraction 0.025 \
  --thick-max-ratio 10 --thick-weight 1 --thick-longest \
  --distortion-weight 20 --distortion-start 37500 --distortion-min-dist 1.5 \
  --distortion-depth distance --distortion-skip-plane-crossing 2
```

`growth_exchange_config.json`:

```json
{
  "strategy": "mrnf", "iterations": 30000, "sh_degree": 3,
  "means_lr": 2.17871e-05, "opacity_lr": 0.012, "scaling_lr": 0.00828016, "rotation_lr": 0.0015,
  "lambda_dssim": 0.22, "refine_every": 163, "stop_refine": 28500, "min_opacity": 0.00392157,
  "growth_exchange": true
}
```

**Not used by the recipe:**

- `--grad-scale-near` (floaters are culled by hand after training);
- `--aniso-*` (the thickness floor is what fixed the needles on this scene);
- `--loss-ycc`;
- `--growth-visit-*`;
- `--opacity-binarize-*`;
- `--loss-weight-dir` and `--generated-*`: sun flare is masked through the dataset's own masks (`--mask-mode ignore`).

These flags all work. They just weren't adopted for this scene.

**Inputs this repo doesn't make.** The lens table, the motion file, the time map and the horizon cube come from
external calibration scripts. The formats above are everything the trainer needs.

---

## Lineage

Each commit builds on the one before it. The tags in the commit messages, such as `LFQPLCYKMR`, name the trainer
build that was used for each round of experiments.

| Commit | Adds |
|---|---|
| `8facb31` LFE | Native fisheye path, screen-share axis, generated-view and loss-weight losses |
| `8fda226` LFR | Dynamic rolling shutter (`--rolling-shutter`) + `lfe-rs` |
| `1def90b` LFP | `--pose-opt` |
| `ee11399` LFR_mg | `LFS_CUDA_ARCHS` fat binary |
| `cdf2afe` LFQ | `--ppisp-lock-frames`, `--ppisp-lock-colour`, `ppisp_final.txt` |
| `cbb61fd` LFC, `025fd90` LFC2 | Learned background cubemap; `--bg-cubemap-init`, smoothness default 100 |
| `7637c3d` LFQP | Pose refinement + PPISP locks + fat binary in one build |
| `6c74b00` LFRL, `b0f6b1a` LFQPL | `--lens-radial` |
| `48d369a` LFQPLC | The cubemap's rays through the radial lens and the refined pose |
| `31db566` LFQPLCY | `--loss-ycc`, `--max-std-dev` |
| `92ab4ad` LFQPLCYR | `lfe-render`, `lfe-register` |
| `3e0126e` LFQPLCYK | `--kernel-shape` |
| `3b02dae` LFQPLCYM | `--motion-blur` |
| `f245630` LFQPLCYKM | Kernel shape + motion blur together (smear moment-matched to β) |
| `f6c1af4` LFQPLCYKD | `--max-std-dev-distance` |
| `9c23d50` LFQPLCYKMO, `33718ad` LFQPLCYKMOD | `--loss-footprint*` |
| `400dd3b` LFQPLCYKMGS | `--grad-scale-near` |
| `78e4eff` LFQPLCYKMAN | `--aniso-*` |
| `8026a04` LFQPLCYKMT | `--thick-*` |
| `4231262` LFQPLCYKMR | `--rs-time-map` |
| `6a1c6a5`, `c97b564` LFQPLCYKMREG | `--ppisp-exposure-guard` |
