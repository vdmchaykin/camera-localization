#!/bin/bash
# Rebuilds the submission model and the validation model from the data, then re-runs every validation in results/.
#   TRAIN=.../c0_2026-10-13_22-53-42_camloc_0.mcap DAY=.../qwen_full.mcap EVAL=.../tools/evaluate.py \
#   [DAYLAP=.../rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap] tools/reproduce.sh [OUT]
# Needs model/features and model/masks (the map, from download_model.sh). Writes model/, model_val/, OUT/ (out/val).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYBIN:-.venv/bin/python}"
R="${1:-out/val}"; mkdir -p "$R"
: "${TRAIN:?training mcap}" "${DAY:?day-styled copy of the training mcap}" "${EVAL:?organisers tools/evaluate.py}"
set -e
# 1. submission model: all laps
$PY -W ignore tools/build_model.py --mcap "$TRAIN"
$PY -W ignore tools/build_day_layer.py --mcap "$DAY"
# 2. validation model: K3 / K6 held out of the motion calibration and the day layer
rm -rf model_val && cp -al model model_val
$PY -W ignore tools/build_model.py --mcap "$TRAIN" --laps 1 2 4 5 7 --out model_val
$PY -W ignore tools/build_day_layer.py --mcap "$DAY" --map-config configs/map_val_k3k6.json --out model_val/day
set +e
# 3. validation on K3 / K6 (map without their keyframes)
val() { tag=$1; mcap=$2; s=$3; l=$4; shift 4
  CAMLOC_MODEL=model_val $PY -W ignore tools/run.py --mcap "$mcap" --out "$R/$tag" --start-s $s --max-s $l \
     --map-config configs/map_val_k3k6.json --evaluate "$EVAL" "$@" > "$R/$tag.log" 2>&1
  echo "$tag: $(grep 'vs GT' "$R/$tag.log")"; }
val night_K3 "$TRAIN" 523.4 110.6 --first-pose-from-gt
val night_K6 "$TRAIN" 896.1 111.6 --first-pose-from-gt
val night_K3_no_first_pose "$TRAIN" 523.4 110.6
val night_K6_no_first_pose "$TRAIN" 896.1 111.6
val qwen_K3 "$DAY" 523.4 110.6 --first-pose-from-gt
val qwen_K6 "$DAY" 896.1 111.6 --first-pose-from-gt
val qwen_K3_no_first_pose "$DAY" 523.4 110.6
val qwen_K6_no_first_pose "$DAY" 896.1 111.6
CAMLOC_DAY=0 val qwen_K3_no_day_layer "$DAY" 523.4 110.6 --first-pose-from-gt
CAMLOC_DAY=0 val qwen_K6_no_day_layer "$DAY" 896.1 111.6 --first-pose-from-gt
# 4. organisers' day lap (no GT, no first pose) with the submission model
if [ -n "${DAYLAP:-}" ]; then
  $PY -W ignore tools/run.py --mcap "$DAYLAP" --out "$R/real_day_lap" > "$R/real_day_lap.log" 2>&1
  CAMLOC_DAY=0 $PY -W ignore tools/run.py --mcap "$DAYLAP" --out "$R/real_day_lap_no_day_layer" > "$R/real_day_lap_no_day_layer.log" 2>&1
fi
echo "done: $R"
