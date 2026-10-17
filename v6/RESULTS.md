# v3 / v5 / v6: night and synthetic sun (development laps K1, K5)

All runs: `LOC_SYNC=1`, `sim/simulate.py --speed 0` (no drops), RTX 5060 Ti. Held-out K3/K6 were not run.
Sources: `v5/run_dev.sh` → `v5/runs/`, `v6/run_stress.sh` → `v6/runs/`. ms = decode + step per frame, minus the
stress perturbation time. inl = median guided inliers. fail = failed guided corrections. Global fallback: 0 in every run.

## v5 ablations (night)

| config | K1 pos | K5 pos | mean | ms |
|---|---|---|---|---|
| v3 | 0.108 | 0.103 | 0.106 | 37 |
| + 1024 keypoints | 0.108 | 0.100 | 0.104 | 40 |
| + keyframes from different passes (3) | 0.103 | 0.098 | 0.100 | 37 |
| + dense on unmatched points | 0.110 | 0.095 | 0.103 | 47 |
| dense only (no detector) | 0.109 | 0.102 | 0.106 | 51 |
| dense only when keypoint matches < 150, 1024 kp, 3 kf | 0.103 | 0.097 | 0.100 | 40 |
| **v5 full** (1024 kp, dense on unmatched, 3 kf) | 0.104 | 0.093 | **0.099** | 56–60 (p95 82) |

## Synthetic sun

`day_all` = day + shadow + glare, `hard` = day_all + relight (smooth random gain field). See `v6/sun_preview.jpg`.

| run | K1 pos | K5 pos | K5 lat | K5 max | inl (K5) | fail (K5) | ms |
|---|---|---|---|---|---|---|---|
| v3 night | 0.108 | 0.103 | 0.067 | 0.59 | 626 | 0 | 36 |
| v5 night | 0.104 | 0.093 | 0.056 | 0.49 | 1047 | 0 | 60 |
| v6 night | 0.104 | 0.091 | 0.054 | 0.45 | 980 | 0 | 61 |
| v3 day_all | 0.109 | 0.104 | 0.068 | 0.63 | 480 | 0 | 36 |
| v5 day_all | 0.105 | 0.091 | 0.052 | 0.45 | 783 | 0 | 55 |
| v6 day_all | 0.106 | **0.088** | **0.046** | **0.36** | 756 | 0 | 61 |
| v3 hard | 0.111 | 0.113 | 0.076 | 0.59 | 141 | 2 | 32 |
| v5 hard | 0.108 | 0.101 | 0.065 | 0.55 | 213 | 0 | 53 |
| v6 hard | 0.109 | 0.103 | 0.064 | 0.50 | 188 | 0 | 59 |

## Findings

- Keyframes from different passes are the cheapest gain (0.106 → 0.100 m, +0 ms). v5 full is best at night
  (0.099 m) at ~60 ms per frame, still real-time (p95 74–82 ms).
- Dense search alone, with no detector, matches v3's accuracy (0.106 m), so it can stand in if the detector stops
  firing on the same spots.
- The synthetic sun barely moves the accuracy. What it does move is the margin: under `hard` the guided inliers
  fall about 4× (v3 626 → 141), and v3 starts to fail corrections (2 on K5). v5/v6 keep ~50 % more inliers than v3
  and do not fail. The real night→day change can be harsher than this proxy, and the inlier margin is what runs
  out first.
- v6's safety nets (relaxed retry, drivable-area check, rate-limited global path) almost never fired (1 relaxed
  retry, 0 rejections, 0 global calls), because the synthetic sun did not make matching fail. They need
  failure-injection tests (blackout, wrong initial pose) to be verified. Exposure normalisation was active
  (gamma ≈ 2 by "day", ≈ 1 at night) and changed nothing at night.

## Final check on held-out laps K3, K6 (run once, `v6/run_val.sh` → `v6/runs_val/`)

No parameter was changed after these runs.

| version | K3 pos | K3 lat | K3 yaw° | K6 pos | K6 lat | K6 yaw° | K6 max | ms mean / p95 | inl |
|---|---|---|---|---|---|---|---|---|---|
| v3 | 0.077 | 0.025 | 0.052 | 0.084 | 0.039 | 0.075 | 0.30 | 37 / 42 | 690 / 657 |
| v5 | 0.072 | 0.025 | 0.050 | 0.082 | 0.045 | 0.082 | 0.26 | 57 / 72 | 1241 / 1188 |
| v6 | **0.071** | 0.025 | 0.050 | **0.081** | 0.045 | 0.082 | 0.26 | 62 / 77 | 1150 / 1102 |

Coverage 100 %, <1 m 100 %, no failed corrections, no global fallback, in all six runs. v5/v6 vs v3: position
−6 % on K3, −3 % on K6 (from the longitudinal error: 0.073 → 0.066–0.068), lateral the same on K3, +6 mm on K6,
yaw +0.007° on K6. Nearly twice the inliers, i.e. more margin for other lighting. Time is still below the 100 ms frame period.
