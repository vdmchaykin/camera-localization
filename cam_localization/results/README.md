# Validation results

All runs: RTX 5060 Ti 16 GB, every camera set processed (no drops), correction inside `step()` (`LOC_SYNC=1`).
K3 / K6 are the held-out laps of the training run (their keyframes excluded from the map, first pose = GT at the lap
start unless stated). Per run: `poses.csv` (submission format), `summary.json` (errors, timing, localizer statistics),
`evaluate.txt` (output of the organisers' `tools/evaluate.py`, where GT exists).

These runs were made during development with the same code under earlier names; the `localizer` field in
`summary.json` shows the old module path: `v14/my_loc_v14.py` = this repository, `v6/my_loc_v6.py` = without the
day layer (`CAMLOC_DAY=0`), `v12/my_loc_v12.py` = day layer built from a CycleGAN-Turbo day copy instead of Qwen.

| folder | data | pos RMSE [m] | lat RMSE [m] | yaw RMSE [deg] | p95 [m] | max [m] | inliers (median) | failed corrections |
|---|---|---|---|---|---|---|---|---|
| `night/K3_this` | night, K3 | 0.071 | 0.025 | 0.050 | 0.143 | 0.196 | 1147 | 0 |
| `night/K6_this` | night, K6 | 0.081 | 0.045 | 0.082 | 0.175 | 0.261 | 1101 | 0 |
| `night/K3_ablation_no_day_layer` | night, K3 | 0.071 | 0.025 | 0.050 | 0.143 | 0.195 | 1150 | 0 |
| `night/K6_ablation_no_day_layer` | night, K6 | 0.081 | 0.045 | 0.082 | 0.175 | 0.258 | 1102 | 0 |
| `night/K3_ablation_cyclegan_layer` | night, K3 | 0.072 | 0.025 | 0.050 | 0.146 | 0.202 | 1132 | 0 |
| `night/K6_ablation_cyclegan_layer` | night, K6 | 0.082 | 0.045 | 0.082 | 0.175 | 0.264 | 1085 | 0 |
| `qwen_day/K3_this` | Qwen day copy, K3 | **0.078** | **0.037** | **0.063** | **0.154** | **0.200** | **463** | **0** |
| `qwen_day/K3_ablation_no_day_layer` | Qwen day copy, K3 | 0.114 | 0.054 | 0.085 | 0.215 | 0.531 | 117 | 23 |
| `qwen_day/K3_ablation_cyclegan_layer` | Qwen day copy, K3 | 0.109 | 0.050 | 0.083 | 0.216 | 0.470 | 163 | 5 |

The Qwen day copy of K3 and the Qwen day layer come from the same image editing model (the layer from the other
laps), so the day numbers of `K3_this` are optimistic.

Organisers' day lap (`real_day_lap/`, 739 sets, up to 68 m/s, no GT; first pose found by global relocalization):

| folder | failed corrections | inliers (median) | relaxed retries | time / set (mean / p95) |
|---|---|---|---|---|
| `real_day_lap/this` | **0** | **207** | 1 | 59 / 70 ms |
| `real_day_lap/ablation_no_day_layer` | 17 | 124 | 127 | 57 / 70 ms |
| `real_day_lap/ablation_cyclegan_layer` | 2 | 144 | 70 | 60 / 74 ms |

The three trajectories agree to 3-4 cm (median) / 10-14 cm (p95).

Start without a first pose (`no_first_pose/`, CycleGAN-layer build; the start does not use the day layer):
full laps K3 / K6 0.072 / 0.081 m (with the GT first pose: 0.072 / 0.082); 42 starts spread over K3 / K6
(`boot_starts/`, `boot_analysis.txt`): all initialised after 10 sets, start pose error median 0.089 m, max 0.199 m,
no single-frame global relocalization more than 0.33 m off.
