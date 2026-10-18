#!/bin/bash
# Downloads the trained model (cam_localization_model.tar, ~8.6 GB), checks its sha256 (model_manifest.txt) and
# unpacks it into model/ (or $CAMLOC_MODEL).
#   ./download_model.sh                                    from MODEL_URL below (Google Drive)
#   MODEL_URL=https://... ./download_model.sh              another location (http(s) or file://)
#   MODEL_TAR=/path/cam_localization_model.tar ./download_model.sh     an archive already downloaded
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_URL="${MODEL_URL:-TODO_GOOGLE_DRIVE_LINK}"
DEST="${CAMLOC_MODEL:-$HERE/model}"
read -r SUM NAME < "$HERE/model_manifest.txt"
TAR="${MODEL_TAR:-${TMPDIR:-/tmp}/$NAME}"
if ! { [ -f "$TAR" ] && echo "$SUM  $TAR" | sha256sum -c --quiet - 2>/dev/null; }; then
  [ -n "${MODEL_TAR:-}" ] && { echo "$TAR: missing or wrong sha256"; exit 1; }
  case "$MODEL_URL" in
    TODO*) echo "no model link yet: set MODEL_URL or MODEL_TAR"; exit 1 ;;
    *drive.google.com*)                          # large Drive files need gdown (virus-scan confirmation page)
      PYBIN="${PYBIN:-$HERE/.venv/bin/python}"; [ -x "$PYBIN" ] || PYBIN=python3
      "$PYBIN" -m pip install -q gdown
      "$PYBIN" -m gdown --fuzzy "$MODEL_URL" -O "$TAR" ;;
    *) curl -fL --retry 3 -o "$TAR" "$MODEL_URL" ;;
  esac
  echo "$SUM  $TAR" | sha256sum -c -
fi
mkdir -p "$DEST"
echo "unpack -> $DEST"
tar -xf "$TAR" -C "$DEST" --strip-components=1
[ -f "$DEST/features/db_fl.npz" ] && echo "model ready: $DEST"
