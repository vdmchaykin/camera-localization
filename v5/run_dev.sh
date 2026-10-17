#!/bin/bash
# v5 ablations on development laps K1, K5 (sync, no drops). Never K3/K6 here. Run from anywhere.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export TORCH_HOME="$PWD/torch_home" LOC_SYNC=1
MCAP=data/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
mkdir -p v5/runs
run() {  # loc tag start len
  python3 -W ignore sim/simulate.py --gt gt.csv --mcap $MCAP --localizer $1 --speed 0 --start-s $3 --max-s $4 \
    --no-gui --tag $2 --out v5/runs/$2 2>&1 | grep -E "^pos RMSE|^step ms" | sed "s/^/$2  /"
}
both() {  # loc tag   (env set by caller)
  run $1 $2_K1 185.5 180.1
  run $1 $2_K5 752.2 143.9
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for cfg in ${CFGS:-v3 full kp kf dense dense_only dense_fb}; do
  case $cfg in
    v3)         both v3/my_loc_v3.py:Localizer v3 ;;
    full)       V5_KP=1024 V5_DENSE=1 V5_KF=3 both v5/my_loc_v5.py:Localizer v5full ;;
    kp)         V5_KP=1024 V5_DENSE=0 V5_KF=2 both v5/my_loc_v5.py:Localizer v5kp ;;
    kf)         V5_KP=512  V5_DENSE=0 V5_KF=3 both v5/my_loc_v5.py:Localizer v5kf ;;
    dense)      V5_KP=512  V5_DENSE=1 V5_KF=2 both v5/my_loc_v5.py:Localizer v5dense ;;
    dense_only) V5_KP=512  V5_DENSE=3 V5_KF=3 both v5/my_loc_v5.py:Localizer v5denseonly ;;
    dense_fb)   V5_KP=1024 V5_DENSE=2 V5_KF=3 both v5/my_loc_v5.py:Localizer v5densefb ;;
  esac
done
echo ALL_DONE
