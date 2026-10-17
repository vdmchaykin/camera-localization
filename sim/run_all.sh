#!/bin/bash
# All mode-B runs of the report, sequentially (one GPU, timings must not overlap).
cd "$(dirname "$0")/.." && source .venv/bin/activate
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
L=sim/my_loc.py:Localizer
python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $L --speed 1 --start-s 300 --max-s 120 --no-gui --tag B_state_300_120
python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $L --speed 1 --start-s 300 --max-s 120 --no-gui --tag B_nostate_300_120 --state-topic ''
python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $L --speed 1 --no-gui --tag B_state_full
python3 sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $L --speed 1 --no-gui --tag B_nostate_full --state-topic ''
