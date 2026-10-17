#!/bin/bash
# FINAL check of the frozen v2 config on held-out K3, K6 (run once; no tuning after this).
cd "$(dirname "$0")/.." && source .venv/bin/activate
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
run() { python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $1 --speed 1 --start-s $2 --max-s $3 --no-gui --tag $4 --out v2/runs/$4 2>&1 | grep -v Warning; }
run v2/my_loc_v2.py:Localizer 523.4 110.6 v2_val_K3
run v2/my_loc_v2.py:Localizer 896.1 111.6 v2_val_K6
echo ALL_DONE
