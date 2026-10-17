#!/bin/bash
# v3 in synchronous mode (LOC_SYNC=1: the correction of frame t is computed inside step(t) and applied with no delay;
# same mode as Vlad's featmap.py validate without --async-fix). Dev laps K1, K5 first, then held-out K3, K6.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home" LOC_SYNC=1
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v3/runs
run() { python3 -W ignore sim/simulate.py --gt gt.csv --mcap $MCAP --localizer v3/my_loc_v3.py:Localizer --speed 1 --start-s $1 --max-s $2 --no-gui --tag $3 --out v3/runs/$3 2>&1 | grep -v -i "warning\|SMs"; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
run 185.5 180.1 v3sync_K1
run 752.2 143.9 v3sync_K5
run 523.4 110.6 v3sync_val_K3
run 896.1 111.6 v3sync_val_K6
echo ALL_DONE
