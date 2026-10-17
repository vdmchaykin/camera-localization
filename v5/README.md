# v5: more keypoints, dense guided search, keyframes from different passes

v5 builds on v3 (`v5/my_loc_v5.py` inherits v3 → v2 → v1; nothing outside `v5/` was changed).

| change | why | flag (default) |
|---|---|---|
| SuperPoint keypoints per camera 512 → 1024 (guided path) | more candidates inside the prior windows; helps when detector repeatability drops (light, blur) | `V5_KP=1024` |
| dense guided search: each projected map point is searched in SuperPoint's dense descriptor map on a 1 px grid in its window (≤ 8 px), best position + parabola sub-pixel, accepted if similarity ≥ 0.75 and ≥ 0.03 above the rest of the window | does not depend on where the detector fires | `V5_DENSE=1` (0 off, 1 keypoints + dense for unmatched points, 2 dense only if keypoint matches < 150, 3 dense only) |
| keyframes: nearest first but one per pass (stamps > 20 s apart), 3 instead of 2 | map errors of one pass average out | `V5_KF=3` |

Running: `v5/run_dev.sh` (ablations on development laps K1, K5, sync, no drops; `CFGS="full kp"` for a subset).
Results: `v5/runs/`, summary in `v5/run_dev.log`. Held-out laps K3/K6 are not used here.
