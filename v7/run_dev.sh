#!/bin/bash
# v7 (bright map points removed) on development laps K1, K5, sync, no drops. Compare with v6/runs/v6_night_K*.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home" LOC_SYNC=1
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v7/runs
run() { python3 -W ignore sim/simulate.py --gt gt.csv --mcap $MCAP --localizer v7/my_loc_v7.py:Localizer --speed 0 \
          --start-s $2 --max-s $3 --no-gui --tag $1 --out v7/runs/$1 2>&1 | grep -E "^pos RMSE|Traceback|Error" | sed "s/^/$1  /"; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for p in ${PEAKS:-200 250}; do
  V7_PEAK=$p run v7p${p}_K1 185.5 180.1
  V7_PEAK=$p run v7p${p}_K5 752.2 143.9
done
echo ALL_DONE
