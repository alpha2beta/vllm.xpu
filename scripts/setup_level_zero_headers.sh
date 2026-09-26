#!/usr/bin/env bash
# Why this exists:
#   The Intel Triton XPU backend JIT-compiles its `spirv_utils` helper with icpx and
#   needs `<level_zero/ze_api.h>`, which ships in Arch's `level-zero-headers` package.
#   This host only has `level-zero-loader` (runtime), and installing the headers needs
#   an interactive sudo. So we fetch the package and extract it into ./tools/sysroot,
#   then expose it through CPATH (honoured by icpx/clang).
#
# The clean alternative, if you have sudo:   sudo pacman -S level-zero-headers
# and you can skip exporting CPATH entirely.
#
# Usage: scripts/setup_level_zero_headers.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tools/sysroot"
PKG="level-zero-headers"
STAMP="$DEST/.stamp"

if [ -f "$STAMP" ] && [ -f "$DEST/usr/include/level_zero/ze_api.h" ]; then
  echo "already present: $DEST/usr/include/level_zero/ze_api.h"
  exit 0
fi

URL="$(pacman -S --print-format '%l' "$PKG")" || {
  echo "could not resolve the package URL (is pacman available?)"; exit 1; }
echo "downloading $URL"

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fL --retry 3 -o "$TMP/pkg.tar.zst" "$URL"
tar --use-compress-program=unzstd -xf "$TMP/pkg.tar.zst" -C "$DEST"

test -f "$DEST/usr/include/level_zero/ze_api.h" || {
  echo "FAILED: ze_api.h not extracted"; exit 1; }
echo "$(date -Is) $URL" >"$STAMP"
echo "OK -> $DEST/usr/include/level_zero/ze_api.h"
echo "Add to your environment before running vLLM:"
echo "  export CPATH=$DEST/usr/include:\${CPATH:-}"
