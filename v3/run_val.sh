#!/bin/bash
# v3 final check on held-out laps K3, K6 (run once). Compare with v2/runs_cluster/v2_val_K3, v2_val_K6.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home"
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v3/runs
run() { python3 -W ignore sim/simulate.py --gt gt.csv --mcap $MCAP --localizer v3/my_loc_v3.py:Localizer --speed 1 --start-s $1 --max-s $2 --no-gui --tag $3 --out v3/runs/$3 2>&1 | grep -v -i "warning\|SMs"; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
run 523.4 110.6 v3_val_K3
run 896.1 111.6 v3_val_K6
echo ALL_DONE
