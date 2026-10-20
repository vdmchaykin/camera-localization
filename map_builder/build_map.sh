#!/bin/bash
# Builds the whole model from the data: the night keyframe map, keyframe poses / motion calibration / night
# brightness / track borders, and the day descriptor layer.
#   map_builder/build_map.sh TRAIN.mcap DAY.mcap [DAY2.mcap ...]
#     TRAIN = c0_2026-10-13_22-53-42_camloc_0.mcap (night training run with GT /tf)
#     DAY   = the day-styled (Qwen re-lit) copy of TRAIN, see README.md
# Writes model/ (or CAMLOC_MODEL). Night-map work files (export/, frames/ ~4 GB, features/ ~4.4 GB, masks/) are
# written next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
PY="$(realpath "${PYBIN:-.venv/bin/python}")"
TRAIN="$(realpath "${1:?training mcap}")"; shift
[ $# -ge 1 ] || { echo "day-styled mcap(s) missing" >&2; exit 1; }
DAY=(); for d in "$@"; do DAY+=("$(realpath "$d")"); done
MODEL="${CAMLOC_MODEL:-$PWD/model}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch}"

# 1. night keyframe map (original standalone scripts, run inside map_builder/)
( cd "$HERE"
  $PY export_lite.py "$TRAIN" export         # calibration, GT /tf, image stamps            (~3 min, CPU)
  $PY extract_frames.py "$TRAIN" frames      # half-res frames of all 7 cameras             (~2 min, CPU)
  $PY make_masks.py                          # car-body masks                               (seconds)
  $PY global_desc.py                         # DINOv2 global descriptors                    (~25 min, GPU)
  $PY db_build.py                            # keyframes, SuperPoint, 3D points (GT poses)  (~2.5 h, GPU)
  $PY write_set_stamps.py )
mkdir -p "$MODEL/features" "$MODEL/masks"
cp "$HERE"/features/db_*.npz "$HERE"/features/global_*.npy "$HERE"/features/set_stamps.npy "$MODEL/features/"
cp "$HERE"/masks/{fl,fr,cl,cr,rl,rr,r}.png "$MODEL/masks/"
# 2. keyframe poses, motion calibration (complete laps 1-7), night brightness, track borders     (~30 s, CPU)
$PY -W ignore map_builder/build_model.py --mcap "$TRAIN"
# 3. day descriptor layer from the day-styled copy                         (~2 min per 8 GB of day bags, GPU)
$PY -W ignore map_builder/build_day_layer.py --mcap "${DAY[@]}"
echo "model -> $MODEL"
