#!/bin/bash
# v3 on development laps K1, K5 (compare with v2/runs_cluster/v2_K1, v2_K5). Run from the project root.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home"
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v3/runs
run() { python3 -W ignore sim/simulate.py --gt gt.csv --mcap $MCAP --localizer v3/my_loc_v3.py:Localizer --speed 1 --start-s $1 --max-s $2 --no-gui --tag $3 --out v3/runs/$3 2>&1 | grep -v -i "warning\|SMs"; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
V3_PERIOD=0   run 185.5 180.1 v3_K1
V3_PERIOD=0   run 752.2 143.9 v3_K5
V3_PERIOD=0.3 run 185.5 180.1 v3p03_K1
V3_PERIOD=0.3 run 752.2 143.9 v3p03_K5
echo ALL_DONE
