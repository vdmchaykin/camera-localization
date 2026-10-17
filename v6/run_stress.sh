#!/bin/bash
# Synthetic-sun stress matrix (v6/sun_stress.py) on development laps K1, K5, sync, no drops. Never K3/K6.
# Night baselines of v3 / v5 come from v5/run_dev.sh (v5/runs/v3_K*, v5full_K*).
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home" LOC_SYNC=1
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v6/runs
run() {  # localizer tag start len
  python3 -W ignore sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $1 --speed 0 --start-s $3 --max-s $4 \
    --no-gui --tag $2 --out v6/runs/$2 2>&1 | grep -E "^pos RMSE" | sed "s/^/$2  /"
}
laps() { run $1 $2_K1 185.5 180.1; run $1 $2_K5 752.2 143.9; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
laps v6/my_loc_v6.py:Localizer v6_night
for scen in ${SCENS:-day_all hard}; do
  case $scen in day_all) S=day,shadow,glare ;; hard) S=day,shadow,glare,relight ;; *) S=$scen ;; esac
  for loc in v3 v5 v6; do
    case $loc in v3) B=v3/my_loc_v3.py:Localizer ;; v5) B=v5/my_loc_v5.py:Localizer ;; v6) B=v6/my_loc_v6.py:Localizer ;; esac
    STRESS=$S STRESS_BASE=$B laps v6/sun_stress.py:Localizer ${loc}_$scen
  done
done
echo ALL_DONE
