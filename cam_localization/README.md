# cam_localization: camera-only localization on a race track

Constructor Labs challenge "Camera-based localization on a race track" (National AI Hackathon 2026).
At every camera set (7 cameras, 10 Hz) the localizer estimates the car pose `(x, y, yaw)` in the track frame from
the images, the camera calibration, the track borders and the car's motion state (speeds, yaw rate). It is causal:
the pose for set *t* uses only data up to *t*. GT is never read at inference.

## Quick start

```bash
./setup.sh                              # .venv with the pinned dependencies (Python 3.12, CUDA 12.8)
./download_model.sh                     # trained model -> model/ (~8 GB, checked with sha256)
./run.sh RUN.mcap out/RUN               # first pose unknown: found by global relocalization
./run.sh RUN.mcap out/RUN X Y YAW       # first pose of the run given
```

`out/RUN/poses.csv` is the submission file: `timestamp_ns,x,y,yaw` (one row per camera set, `timestamp_ns` = header
stamp of the newest image of the set, metres / radians, track frame). `frames.csv` has the measured time per set, `summary.json` the
runtime statistics, the hardware and the localizer statistics.

Inputs read from the mcap: `/constructor0/sensor/camera_*/compressed_image` and `camera_info`, `/tf_static`,
`/constructor0/track_bounds` (if missing, the bundled borders of the same circuit are used), and the car state from
`/constructor0/state_estimation` (only the fields whitelisted in `camloc/car_state.py`: body-frame velocity `vx_mps`,
`vy_mps` and yaw rate `wz_radps`; pose fields are never read) or, if that topic is missing, the same quantities from
`/constructor0/validation/twist` (body frame `constructor0/base_link`).

## Method

```
camera set (7 images) ──► SuperPoint (FP16, 1024 kp/camera) ──► guided matching to the keyframe map ──► robust pose
                                                                 (window around the predicted projection,      refinement,
car state (vx, vy, yaw rate) ──► EKF prediction ──► prior ───────►  keypoints + dense search, night OR day        all cameras
                                       ▲                             descriptor of each map point)                  │
                                       └──────────────────────────── EKF update ◄──────────────────────────────────┘
start / lost: DINOv2 retrieval + LightGlue + PnP-RANSAC over the whole map (global relocalization)
```

* **Map** (built from the night training run): keyframes along all laps with SuperPoint keypoints, descriptors and
  3D points (triangulated with the GT poses of the training run), DINOv2 global descriptors per keyframe.
* **Tracking**: the EKF (car state as motion input, speed scale and yaw-rate bias in the state) predicts the pose;
  map points of the 3 nearest keyframes are projected with it and matched inside a window (descriptor mutual best,
  plus a dense search in the SuperPoint descriptor map for the points left unmatched); the pose is refined on all
  cameras at once with robust least squares and fused into the EKF. A correction runs on every set.
* **Daylight**: the map is from a night session, the test may be by day. (1) Per-camera gamma maps the image
  brightness to the night median, saturated pixels (sun, glare) are masked, a failed correction is retried with
  relaxed thresholds, poses off the drivable area are rejected. (2) **Day descriptor layer**: every map point also
  has a day descriptor, computed at the same keyframe pixel on a day-styled copy of the training run (Qwen image
  editing: same geometry, other light). Matching uses `max(similarity to night, similarity to day)`, so night runs
  behave as before and day runs find ~3x more matches.
* **No first pose**: global relocalization on the first camera sets; the solutions are carried forward with the car
  state and voted on (median of the agreeing ones within 2 m); tracking starts once >= 3 (and at least half) agree.

Code (`camloc/`):

| module | content |
|---|---|
| `localizer.py` | **the localizer**: guided correction (exposure normalisation, glare mask, FP16 / compiled SuperPoint, keypoint + dense guided search with night and day descriptors, Gauss-Newton refinement, relaxed retry, drivable-area check), global relocalization when lost, start without a first pose |
| `localize.py` | keyframe map access; LightGlue + PnP-RANSAC + multi-camera refinement for the global relocalization |
| `track_filter.py` | EKF with car state (calibrated motion model); constant-velocity EKF for runs without car state |
| `global_desc.py`, `vislib.py` | DINOv2 global descriptor; SuperPoint + LightGlue of the global relocalization |
| `camera_model.py` | cameras: half-res intrinsics, extrinsics, body masks, projection |
| `bag.py`, `car_state.py` | mcap reading (calibration, camera sets, whitelisted car state, GT for validation only) |

## Results

Validation on the held-out laps K3, K6 of the training run (map without their keyframes: `configs/map_val_k3k6.json`,
first pose = GT at the lap start), RTX 5060 Ti 16 GB.

| run | pos RMSE [m] | lateral RMSE [m] | yaw RMSE [deg] | < 1 m | time / set |
|---|---|---|---|---|---|
| K3, night | 0.071 | 0.025 | 0.050 | 100 % | 67 ms |
| K6, night | 0.081 | 0.045 | 0.082 | 100 % | 66 ms |
| K3, day (Qwen re-lit copy) | 0.078 | 0.037 | 0.063 | 100 % | 65 ms |
| organisers' day lap (no GT) | – | – | – | 739/739 corrections, 0 failed | 59 ms |

Time per set = image decoding + `step()`; the camera period is 100 ms.
Without a first pose the localizer starts after 10 sets (1 s): at 42 starts spread over K3 / K6 the start pose was
within 0.20 m of GT every time (median 0.09 m), and the full-lap error was the same as with the GT first pose.
Caveat: the K3 day copy and the day layer come from the same image editing model (other laps), so that number is
optimistic; the organisers' real day lap has no GT.

Validation command (K3; K6: `--start-s 896.1 --max-s 111.6`):

```bash
.venv/bin/python tools/run.py --mcap TRAIN.mcap --out out/K3 --start-s 523.4 --max-s 110.6 \
    --map-config configs/map_val_k3k6.json --first-pose-from-gt
```

With GT in the bag, `run.py` writes `gt_window.csv` (format of the organisers' `tools/extract_gt.py`) and scores the
poses with their unchanged `tools/evaluate.py` (`../c0_2026-10-13_22-53-42_camloc/tools/`) into `evaluate.txt`.

Current model state: the map (`features/`) holds the keyframes of all laps and test runs use all of them
(`configs/map_full.json`, K3 / K6 included). The day layer (`model/day/`) is the validation build: keyframes of
K3 / K6 keep their night descriptor. For test runs rebuild it from the complete Qwen day copy of the training run
(command below, all laps) and keep the validation build separately (`--map-config configs/map_val_k3k6.json
--out model/day_val`, run with `CAMLOC_DAY_DIR=model/day_val`).

## Model (`model/`, ~8 GB, not in git)

| file | content | built by |
|---|---|---|
| `features/db_<cam>.npz` | keyframes: keypoints, SuperPoint descriptors, 3D points | map builder (not in this repository yet) |
| `features/global_<cam>.npy`, `set_stamps.npy` | DINOv2 global descriptors of the training-run frames | descriptor of `camloc/global_desc.py`, map builder |
| `kf_poses.npz`, `motion_calib.json` | keyframe poses; car state -> motion calibration | from GT of the training run |
| `masks/<cam>.png` | car-body masks | from the training images |
| `night_stats.json` | median grey of the night keyframes per camera | from the map keyframes |
| `day/day_<cam>.npy` | day descriptor layer | `tools/build_day_layer.py` |
| `track_bounds.npz` | track / pit-lane borders | `/constructor0/track_bounds` of the training run |

Rebuilding the day layer (e.g. from a new day-styled copy of the training run, all laps including K3 / K6):

```bash
.venv/bin/python tools/build_day_layer.py --mcap DAY_COPY.mcap          # -> model/day/
```

## Pretrained models and external data (declared)

* DINOv2 ViT-S/14 (Meta, `facebookresearch/dinov2` via torch.hub) - global descriptors.
* SuperPoint and LightGlue weights (`cvg/LightGlue`) - keypoints, descriptors, matching.
* Qwen image editing model - only offline, to re-light the training images night -> day for the day layer.
* No external datasets; the map is built from the provided training run.

## Configuration

| variable | default | meaning |
|---|---|---|
| `CAMLOC_MODEL` | `model/` | model directory |
| `LOC_MAP_CONFIG` | `configs/map_full.json` | map subset (`tools/run.py --map-config`) |
| `CAMLOC_DAY` | `1` | day descriptor layer on / off (0: night descriptors only, ablation) |
| `CAMLOC_DAY_DIR` | `model/day/` | day descriptor layer directory |
| `CAMLOC_COMPILE` | `1` | `torch.compile` of the SuperPoint backbone (0: eager, if compiling fails on a machine) |
| `CAMLOC_BOOT_*` | see `camloc/localizer.py` | start without a first pose |
| `TORCH_HOME` | `.torch/` | cache of the pretrained weights |
