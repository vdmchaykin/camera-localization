#!/bin/bash
# v1 vs v2 on development laps K1, K5 (live, real time). Held-out K3/K6 are NOT run here.
cd "$(dirname "$0")/.." && source .venv/bin/activate
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
run() { python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $1 --speed 1 --start-s $2 --max-s $3 --no-gui --tag $4 --out v2/runs/$4 2>&1 | grep -v Warning; }
run sim/my_loc.py:Localizer   185.5 180.1 v1_K1
run v2/my_loc_v2.py:Localizer 185.5 180.1 v2_K1
run sim/my_loc.py:Localizer   752.2 143.9 v1_K5
run v2/my_loc_v2.py:Localizer 752.2 143.9 v2_K5
echo ALL_DONE
