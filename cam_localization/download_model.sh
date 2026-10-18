#!/bin/bash
# Downloads the trained model (~8 GB) into model/ (or $CAMLOC_MODEL): the archive is split into parts
# (model_manifest.txt: sha256 + file name), each part is checked, then the parts are unpacked.
#   ./download_model.sh                         from MODEL_BASE_URL below
#   MODEL_BASE_URL=file:///path/to/parts ./download_model.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_BASE_URL="${MODEL_BASE_URL:-https://github.com/vdmchaykin/camera-localization/releases/download/cam-localization-model-v1}"
DEST="${CAMLOC_MODEL:-$HERE/model}"
TMP="${TMPDIR:-/tmp}/camloc_model_parts"
mkdir -p "$TMP" "$DEST"
while read -r sum name; do
  [ -z "$name" ] && continue
  if [ -f "$TMP/$name" ] && echo "$sum  $TMP/$name" | sha256sum -c --quiet - 2>/dev/null; then
    echo "have $name"; continue
  fi
  echo "download $name"
  curl -fL --retry 3 -o "$TMP/$name" "$MODEL_BASE_URL/$name"
  echo "$sum  $TMP/$name" | sha256sum -c --quiet -
done < "$HERE/model_manifest.txt"
echo "unpack -> $DEST"
cat $(awk '{print "'"$TMP"'/"$2}' "$HERE/model_manifest.txt") | tar -x -C "$DEST" --strip-components=1
[ -f "$DEST/features/db_fl.npz" ] && echo "model ready: $DEST"
