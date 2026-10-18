#!/bin/bash
# Downloads the trained model (cam_localization_model.tar, ~8.6 GB), checks its sha256 (model_manifest.txt) and
# unpacks it into model/ (or $CAMLOC_MODEL).
#   ./download_model.sh                                    from MODEL_URL below (Google Drive)
#   MODEL_URL=https://... ./download_model.sh              another location (http(s) or file://)
#   MODEL_TAR=/path/cam_localization_model.tar ./download_model.sh     an archive already downloaded
set -euo pipefail
# needs: curl, tar, sha256sum (no Python)
HERE="$(cd "$(dirname "$0")" && pwd)"
# clean-v14 model, Drive folder https://drive.google.com/drive/folders/1MiI1nSciCO-bJB_dFHq23Rb4BqsDERa2
MODEL_URL="${MODEL_URL:-https://drive.google.com/file/d/1K050xbkpVElLbSnwAx3x5Msb2TL-sk6n/view}"
DEST="${CAMLOC_MODEL:-$HERE/model}"
read -r SUM NAME < "$HERE/model_manifest.txt"
TAR="${MODEL_TAR:-${TMPDIR:-/tmp}/$NAME}"
if ! { [ -f "$TAR" ] && echo "$SUM  $TAR" | sha256sum -c --quiet - >/dev/null 2>&1; }; then
  [ -n "${MODEL_TAR:-}" ] && { echo "$TAR: missing or wrong sha256"; exit 1; }
  rm -f "$TAR"                                   # a broken earlier download (e.g. a 2 KB HTML page)
  case "$MODEL_URL" in
    *drive.google.com*)          # Drive answers a large file with an HTML "virus scan" page: confirm it directly
      ID="$(sed -nE 's#.*(/d/|[?&]id=)([A-Za-z0-9_-]+).*#\2#p' <<< "$MODEL_URL")"
      curl -fL --retry 5 -o "$TAR" \
        "https://drive.usercontent.google.com/download?id=${ID}&export=download&confirm=t" ;;
    *) curl -fL --retry 3 -o "$TAR" "$MODEL_URL" ;;
  esac
  if [ "$(head -c 1 "$TAR")" = "<" ]; then                # an HTML page instead of the archive
    echo "download failed, the server sent: $(grep -o '<title>[^<]*' "$TAR" | head -1 | cut -c8-)"
    echo "Google Drive limits downloads of large shared files ('Quota exceeded', up to 24 h). Retry later, or get"
    echo "the archive another way and run: MODEL_TAR=/path/$NAME ./download_model.sh"
    exit 1
  fi
  echo "$SUM  $TAR" | sha256sum -c -
fi
mkdir -p "$DEST"
echo "unpack -> $DEST"
tar -xf "$TAR" -C "$DEST" --strip-components=1
[ -f "$DEST/features/db_fl.npz" ] && echo "model ready: $DEST"
