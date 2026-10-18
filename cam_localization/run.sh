#!/bin/bash
# Single run script: raw run (mcap) -> submission CSV.
#   ./run.sh RUN.mcap OUT_DIR              first pose unknown: the localizer finds it (global relocalization)
#   ./run.sh RUN.mcap OUT_DIR X Y YAW      first pose of the run given (track frame, yaw in rad)
# Writes OUT_DIR/poses.csv (timestamp_ns,x,y,yaw), frames.csv (time per camera set), summary.json.
# Needs: ./setup.sh (environment) and ./download_model.sh (model/, ~8 GB) once.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ $# -eq 2 ] || [ $# -eq 5 ] || { sed -n 2,6p "$0"; exit 2; }
MCAP=$1; OUT=$2; shift 2
PYBIN="${PYBIN:-$HERE/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN=python3
export TORCH_HOME="${TORCH_HOME:-$HERE/.torch}"          # DINOv2 / SuperPoint / LightGlue weights (downloaded once)
[ -f "${CAMLOC_MODEL:-$HERE/model}/features/db_fl.npz" ] || "$HERE/download_model.sh"
FP=()
[ $# -eq 3 ] && FP=(--first-pose "$1" "$2" "$3")
"$PYBIN" -W ignore "$HERE/tools/run.py" --mcap "$MCAP" --out "$OUT" "${FP[@]}"
