#!/bin/bash
# Final check on held-out laps K3, K6 (run ONCE, no tuning afterwards): v3 / v5 / v6, sync, no drops.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home" LOC_SYNC=1
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v6/runs_val
run() { python3 -W ignore sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $1 --speed 0 --start-s $3 --max-s $4 \
          --no-gui --tag $2 --out v6/runs_val/$2 2>&1 | grep -E "^pos RMSE|Traceback|Error" | sed "s/^/$2  /"; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for v in v3 v5 v6; do
  run $v/my_loc_$v.py:Localizer ${v}_K3 523.4 110.6
  run $v/my_loc_$v.py:Localizer ${v}_K6 896.1 111.6
done
echo ALL_DONE
