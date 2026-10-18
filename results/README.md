# Validation results

Made with `tools/reproduce.sh` (this repository, unchanged), RTX 5060 Ti 16 GB, every camera set processed (no
drops). K3 / K6 runs use the validation model (K3 / K6 excluded from the map, the motion calibration and the day
layer) and, unless the folder says `no_first_pose`, the GT pose at the lap start. `qwen_day` = the Qwen re-lit day copy
of the training run (same stamps, same GT). Per run: `poses.csv` (submission format), `summary.json` (errors, timing,
hardware, localizer statistics), `evaluate.txt` (output of the organisers' `tools/evaluate.py`).

| folder | pos RMSE [m] | lat RMSE [m] | yaw RMSE [deg] | p95 [m] | max [m] | inliers (median) | failed corrections | time / set |
|---|---|---|---|---|---|---|---|---|
| `night/K3` | 0.071 | 0.025 | 0.050 | 0.143 | 0.197 | 1146 | 0 | 67 ms |
| `night/K6` | 0.081 | 0.044 | 0.082 | 0.176 | 0.256 | 1104 | 0 | 66 ms |
| `night/K3_no_first_pose` | 0.072 | 0.025 | 0.050 | 0.147 | 0.250 | 1153 | 0 | 68 ms |
| `night/K6_no_first_pose` | 0.080 | 0.044 | 0.082 | 0.172 | 0.256 | 1106 | 0 | 68 ms |
| `qwen_day/K3` | 0.076 | 0.034 | 0.059 | 0.151 | 0.211 | 632 | 0 | 65 ms |
| `qwen_day/K6` | 0.086 | 0.051 | 0.081 | 0.184 | 0.333 | 613 | 0 | 65 ms |
| `qwen_day/K3_no_first_pose` | 0.080 | 0.035 | 0.061 | 0.156 | 0.530 | 634 | 0 | 67 ms |
| `qwen_day/K6_no_first_pose` | 0.088 | 0.052 | 0.081 | 0.181 | 0.613 | 615 | 0 | 68 ms |
| `qwen_day/K3_no_day_layer` | 0.114 | 0.053 | 0.085 | 0.218 | 0.535 | 118 | 23 | 63 ms |
| `qwen_day/K6_no_day_layer` | 0.134 | 0.067 | 0.102 | 0.260 | 0.626 | 121 | 25 | 62 ms |

`no_day_layer` = the same model with the day layer switched off (`CAMLOC_DAY=0`). In the `no_first_pose` runs the
maximum error is in the first second, while the localizer is still finding itself. The day copy of K3 / K6 and the day
layer come from the same image editing model, so the day numbers are optimistic.

Organisers' day lap (`real_day_lap/`, 739 sets, up to 68 m/s, no GT, no first pose), submission model:

| folder | failed corrections | inliers (median) | relaxed retries | time / set (mean / p95) |
|---|---|---|---|---|
| `real_day_lap/this` | 0 | 260 | 0 | 69 / 82 ms |
| `real_day_lap/no_day_layer` | 21 | 127 | 117 | 63 / 75 ms |
