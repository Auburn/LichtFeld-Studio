# FPV recipe inputs

Scripts that make the four input files of the FPV splat recipe (see [`FPV_SPLAT_RECIPE.md`](../../FPV_SPLAT_RECIPE.md)):

| Script | Makes | Trainer flag |
|---|---|---|
| [`rs_motion.py`](#rs_motionpy-rolling-shutter-motion) | the per-image rolling-shutter motion file | `--rolling-shutter` |
| [`build_rs_map.py`](#build_rs_mappy-and-fit_rs_radialpy-readout-time-map) + [`fit_rs_radial.py`](#build_rs_mappy-and-fit_rs_radialpy-readout-time-map) | the per-pixel readout-time map, from a video of a flickering LED | `--rs-time-map` |
| [`lens_radial.py`](#lens_radialpy-radial-lens-table) | the radial lens table | `--lens-radial` |
| [`cubemap_horizon.py`](#cubemap_horizonpy-horizon-cube) | the fixed horizon cube | `--bg-cubemap-init` |

The scripts are standalone: Python 3 with numpy, scipy and opencv-python (`pip install -r requirements.txt`), plus
`ffmpeg` and `ffprobe` on `PATH` for `build_rs_map.py`. None needs a GPU. Each script's full reference is in its
docstring (`python SCRIPT.py --help`).

All paths below are placeholders.

---

## `rs_motion.py`: rolling-shutter motion

Differentiates a camera path to get each training image's motion during its readout: angular velocity and
acceleration, linear velocity and acceleration, in the image's own camera axes.

```bash
python rs_motion.py <dataset>/sparse/0 <path_model> motion.txt --frames-csv <frames.csv>
python rs_motion.py <dataset>/sparse/0 <dataset>/sparse/0 motion.txt --fps 23.976
```

**Inputs**

- `SPARSE`: the training dataset's COLMAP text model (`images.txt`). Each image's pose is its mid-readout pose, which
  is how the trainer uses it.
- `PATH`: a COLMAP text model of the camera path, in the same world frame: a denser registration of the same video, a
  larger set of frames, or `SPARSE` itself.
- Frame times: `--fps`, or `--frames-csv` with the columns `frame` and `time_s` (fps = the last row's frame / its time).
  Frame numbers come from the image names (`--frame-regex`, default `frame_(\d+)`).

**Method.** For every image, a local polynomial in time (up to cubic) through the ±3 nearest other path frames within
0.4 s (`--half`, `--max-gap-s`), passing exactly through the image's own pose. Its first and second derivatives at the
image's time are the motion. An image with path frames on one side only gets a constant-velocity fit, and one with
none gets zero motion; the script counts both.

**Output** (text, `#` comments), one line per image of `SPARSE`:

```
name  wx wy wz  ax ay az  vx vy vz  Ax Ay Az
```

`w`: angular velocity (rad/s), `a`: angular acceleration (rad/s²), `v`: linear velocity (scene units/s), `A`: linear
acceleration (scene units/s²), all in the image's mid-readout camera axes.

**Trainer:** `--rolling-shutter motion.txt --rs-readout-ms <readout>`. Add `--rs-learn omega,alpha,vel,acc` to refine
the motion during training.

The recipe's file was made with `SPARSE` = the training images and `PATH` = the pose-refined cameras of every usable
frame of the flight (a superset of the training images), with the defaults (`--half 3 --max-gap-s 0.4`) and the video's
frame list.

---

## `build_rs_map.py` and `fit_rs_radial.py`: readout-time map

Some cameras remap the sensor image radially after capture, so the lines of equal readout time are curved rather than
straight rows. These two scripts measure the readout time of every pixel from a video of a flickering LED and fit the
trainer's radial form to it.

**Recording.** Mount the camera so it can't move and film a wall lit by an LED that flickers at a steady, known
frequency (1 kHz works well). Use the same video mode as the footage: resolution, frame rate, and any lens-correction or
stabilisation setting. Light the wall evenly, without clipping. One or two minutes of video is enough.

```bash
python build_rs_map.py <led_video.mp4> --out <map_dir> --led-khz 1.0 --step 13
python fit_rs_radial.py <map_dir>                     # writes <map_dir>/rs_time_radial.txt
```

**`build_rs_map.py`** decodes every `--step`-th frame and every I-frame through ffmpeg, binned 4×4. It then fits, per
bin, the LED's phase over the frames (temporal phase shifting, with the LED's frequency searched against the frame
clock and its harmonics fitted), unwraps the phase to a time, and fits smooth 2D polynomials whose degree is chosen by
cross-validation. Pick `--step` so the LED's phase spreads evenly over the used frames: at 23.976 fps and 1 kHz, any
step coprime with 24 does (the recipe's map used `--step 13`). `--led-khz` must be right to about 1 %, because the frame
clock fixes the frequency only modulo the frame rate.

Outputs in `--out`:

- `rs_time_map_raw.npz`: the per-bin measured times, their noise, SNR, the fit, and the image size (the input of
  `fit_rs_radial.py`);
- `rs_time_map_grid.txt`: the polynomial as a grid-form map (also usable with `--rs-time-map`);
- `rs_time_map_poly.json`: the polynomial's coefficients, the cross-validation table and summary numbers;
- check images (`rs_time_map_isochrones.png`, `_residual.png`, `_sigma.png`) and the log `rs_map_report.txt`.

**`fit_rs_radial.py`** fits t = t0 + k y (1 + a r² + b r⁴ + c r⁶) to the measured bins (x, y in half image widths
from the centre, minus a fitted centre offset), with the same noise weights, and writes the trainer's file.

**Output** (`rs_time_radial.txt`; times in ms):

```
LFS_RS_TIME_MAP 1
size W H
radial k a b c cx cy
```

with x = (u − W/2)/(W/2) − cx, y = (v − H/2)/(W/2) − cy, r² = x² + y², t = k y (1 + a r² + b r⁴ + c r⁶), in output
pixels with the pixel-corner convention (pixel (c, r) has its centre at (c + 0.5, r + 0.5)). The grid form is
`grid NX NY` followed by NY rows of NX values, nodes at u = i W/(NX − 1), v = j H/(NY − 1), and an optional
`readout_ms R`.

**Trainer:** `--rolling-shutter motion.txt --rs-time-map rs_time_radial.txt --rs-readout-ms <readout>`. The trainer
normalises the map by its own linear-equivalent readout, so `--rs-readout-ms` keeps its meaning. Any image with the
map's aspect ratio uses it, scaled. `fit_rs_radial.py --linear-test MS` also writes the plain row rule at MS ms in both
forms, to check that the time-map path reproduces the row rule.

---

## `lens_radial.py`: radial lens table

Builds a tabulated radial profile θ → θd that replaces the `OPENCV_FISHEYE` k1–k4 polynomial, for lenses a four-term
polynomial can't follow out to the edge of the frame. Each builder writes `lens_radial.txt` and `cameras.txt` (the
paired `OPENCV_FISHEYE` camera line) to `OUT_DIR`.

**From any calibration (`profile`).** Sample your calibration's θ → θd at many angles and write them as text or CSV:

```
# my lens, sampled every 0.05 deg
camera 4096 3072 1165.94 1164.42 2046.54 1528.97 0.3862 -0.0092 -0.0319 0.0096
theta_rad,theta_d_rad
0.000873,0.000867
0.001745,0.001734
...
```

```bash
python lens_radial.py profile <profile.csv> <out_dir> [--valid-deg 75.6] [--cameras <dataset>/sparse/0/cameras.txt]
```

- θ is the ray's angle from the optical axis; θd is the image radius over the focal length (u = fx θd x/ρ + cx).
- The `camera` line is `W H fx fy cx cy`, optionally followed by `k1 k2 k3 k4`. `--cameras` instead takes the first
  `OPENCV_FISHEYE` camera of a COLMAP `cameras.txt`, with its k's.
- With k's given, the table is paired with exactly those k's, so a dataset whose `cameras.txt` already holds them
  needs no change. Without them, the k's are fitted to the profile (least squares over the frame), and the dataset's
  camera must then be the one in `out_dir/cameras.txt`.
- The profile should be smooth (sampled from a calibration model, not raw measurements) and increasing, sampled from
  near the axis (every 0.05–0.25° works) to the edge of the data. The table runs from 0 to `--valid-deg` (default: the
  last sample) and continues linearly beyond.
- P = θd/θ is interpolated as a quintic spline in θ². Fed 0.05°-spaced samples of the recipe's own table, this
  rebuilds that table's θd to within 2·10⁻⁶ px (0.25° spacing: 3·10⁻⁴ px).

**From a fisheye + B-spline calibration (`rspl`).** The recipe's table came from a ChArUco calibration whose model is
the fisheye polynomial plus a 20-point uniform cubic B-spline correction in θ over [0, 1.45 rad]. The JSON format is in
the script's docstring.

```bash
python lens_radial.py rspl <results.json> <out_dir> --size 4096x3072 --valid-deg 75.6
```

Near the axis (`--centre-deg`, default 1°), the B-spline correction is replaced by an odd polynomial matched to it to
second order, so the profile starts smoothly at θ = 0.

**Checks.** `lens_radial.py poly <cameras.txt> <out_dir>` tabulates an `OPENCV_FISHEYE` camera's own polynomial (the
table reproduces it to about 10⁻¹² px). `lens_radial.py info <lens_radial.txt>` prints a table's summary, the
trainer's load checks and θd at a range of angles.

**Output** (`lens_radial.txt`; text, `#` comments):

```
LENS_RADIAL_TABLE 1
camera W H fx fy cx cy k1 k2 k3 k4     # the camera the table belongs to
nodes N dtheta theta_end                # N nodes uniform in theta (radians), theta_end = (N-1) dtheta < 90 deg
valid THETA                             # the angle up to which the profile is supported by data
extrap A B                              # beyond theta_end: theta_d = A + B theta
centre C                                # optional (rspl)
source ...                              # provenance, not read by the trainer
seg 0 u0 inv_h c0 c1 c2 c3 c4 c5        # N-1 segments: P = sum c_i t^i, t = (u - u0) inv_h, u = theta^2
...
node 0 theta P dP/du d2P/du2 theta_d    # N reference rows, not read by the trainer
...
```

θd = θ·P(θ²). Each segment is the quintic Hermite interpolant of P, dP/du and d²P/du² at its two nodes, so the
profile is C2.

**Trainer:** `--native-fisheye --lens-radial lens_radial.txt`. The table applies only to a camera whose k1–k4 match
the table's `camera` line (relative 10⁻⁶) and whose fx/W, fy/H, cx/W, cy/H match it; any other fisheye camera in the
dataset is an error. Images downscaled from the table's size are fine. Mask out pixels beyond `valid` in the dataset's
masks: the profile there is an extrapolation.

---

## `cubemap_horizon.py`: horizon cube

Makes a fixed sky for training: a background cube kept below the horizon and faded to a solid grey above it. The
recipe trains against it frozen (`--bg-cubemap-lr 0`), so the splats can't push detail into the sky.

```bash
python cubemap_horizon.py <run>/bg_cubemap <out_dir> --up up.json --sparse <dataset>/sparse/0
python cubemap_horizon.py <src>/faces.npy <out_dir> --up=0.022,-0.957,-0.289 --lo -25 --hi 25 --grey 249
```

**Inputs**

- `SRC`: the cube to keep below the horizon: a folder with `faces.npy` (or the six face PNGs), or a `faces.npy` file.
  The simplest source is the trainer's own export `<output>/bg_cubemap/faces.npy` from a run with `--bg-mode cubemap`.
  Any cube in the same layout works; the recipe's was painted from renders of an earlier model of the scene.
- `--up`: the scene's up direction in the dataset's world frame, as one of:
  - `x,y,z` (write `--up=x,y,z` when the first value is negative);
  - a JSON file with `"up": [x, y, z]`;
  - `cams`: the mean up (−y axis) of the cameras in `--sparse`.

  The cameras' mean up is only an estimate of the vertical. An FPV camera is usually tilted up on the drone, and the
  drone's average pitch and roll add to that, so the mean camera up can be several degrees off the true vertical
  (2.8° on the recipe's flight), which tilts the horizon. A measured vertical is better: from the flight
  controller's accelerometer log, or from vertical structures such as tree trunks. The recipe used the accelerometer's.
- `--sparse`: a COLMAP text model, needed for `--up cams`; otherwise only used to report the angle between `--up` and
  the cameras' mean up.
- `--lo`, `--hi`, `--grey`: the source is kept below elevation `lo` degrees, the cube is all `grey` (0–255) above
  `hi`, with a smoothstep between. The recipe uses −25, 25 and 249, the same grey as `--bg-color`.

**Output:** `out_dir/faces.npy`, float32 `[6, R, R, 3]`, RGB in 0–1, faces +X −X +Y −Y +Z −Z in the dataset's world
axes (the trainer's cube layout, documented in the script), plus the six faces as PNGs for viewing.

**Trainer:** `--bg-color "(249,249,249)" --bg-mode cubemap --bg-cubemap-res 256 --bg-cubemap-init faces.npy
--bg-cubemap-lr 0`. A cube of another resolution is resampled.
