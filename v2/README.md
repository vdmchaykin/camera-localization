# v2: correction from the filter prior

A separate localizer variant. Nothing outside `v2/` was changed: v1 (`sim/my_loc.py`, `localize.py`,
`track_filter.py`, the `features/` map) works as before. Rolling back means not using `v2/`.

## What changed relative to v1

| | v1 (`sim/my_loc.py`) | v2 (`v2/my_loc_v2.py`) | flag |
|---|---|---|---|
| keyframe selection | DINOv2 over the whole map | keyframe nearest to the filter's predicted pose (2 keyframes) | `V2_GUIDED` |
| matching | LightGlue, 7 pairs | map points projected with the prior, pairs searched only in a window around the projection (GPU) | `V2_GUIDED` |
| pose | PnP-RANSAC per camera + scipy | robust Gauss–Newton from the prior over all cameras | `V2_FAST_SOLVE` |
| SuperPoint | 7 calls, upscaled to 1024 px | one batch, native 753 px | `V2_BATCH`, `V2_SP_RESIZE` |
| motion model | `vy` as is | `0.79·vy + 0.48·wz` (lever arm), yaw-rate scale | `V2_CALIB` |
| scheduling | every 0.5 s | in corners (\|wz\| > 0.1) back to back, on straights every 0.3 s | `V2_CORNER` |
| fallback | — | if the filter is uncertain (σ > 1.5 m) or 3 failures in a row → v1 path (DINOv2 + LightGlue) | |

All flags are on by default; `V2_xxx=0` turns one off (for ablations).

## Files

- `my_loc_v2.py` — the localizer (inherits from `sim/my_loc.py`, which it does not modify).
- `build_model.py` — training artefacts in `model/`: keyframe poses (`kf_poses.npz`) and the motion model
  calibration (`motion_calib.json`, development laps K1, K2, K4, K5, K7 only).
- `profile_v2.py` — timing of the correction stages (lap K4).
- `run_dev.sh` — v1 vs v2 on development laps K1, K5.
- `runs/` — results (`summary.txt`, `evaluate.txt`, `pred.csv`, `frames.csv`).

## Running

```bash
source .venv/bin/activate
python3 v2/build_model.py          # once
python3 sim/simulate.py --gt gt.csv --mcap data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap --localizer v2/my_loc_v2.py:Localizer --speed 1 --start-s 523.4 --max-s 110.6
```

## Rules

Tuning is done only on K1, K2, K4, K5, K7. Held-out K3 and K6 are run once for the final check.
