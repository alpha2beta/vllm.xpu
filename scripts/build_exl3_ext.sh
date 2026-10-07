#!/usr/bin/env bash
# E2.1: reliable EXL3 ESIMD extension build for Arc 140V.
#
# Fixes vs upstream scripts/build_ext.sh at pinned SHA:
#   - honors ${PYTHON:-<workspace .venv python>} for Torch discovery
#   - derives -D_GLIBCXX_USE_CXX11_ABI from that exact Torch build
#   - no `grep ... || true` output masking; full log preserved, failures propagate
#   - builds to a fresh output path; only publishes to exl3xpu/_C.so after validation
#
# Usage (from workspace root):
#   PYTHON=.venv/bin/python EXL3_SRC=~/Projects/exl3xpu bash scripts/build_exl3_ext.sh [log_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
EXL3_SRC="${EXL3_SRC:-$HOME/Projects/exl3xpu}"
LOG_DIR="${1:-$ROOT/logs/exl3-build-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/build.log"
exec > >(tee "$LOG") 2>&1

echo "=== exl3xpu ESIMD build ==="
echo "date: $(date -u +%FT%TZ)"
echo "python: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])'))"
echo "exl3 src: $EXL3_SRC ($(git -C "$EXL3_SRC" rev-parse HEAD))"
echo "worktree patch:"
git -C "$EXL3_SRC" diff --stat

# oneAPI environment (setvars references unset vars; suspend nounset).
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force
set -u
echo "icpx: $(which icpx) ($(icpx --version 2>/dev/null | head -1))"

T="$("$PYTHON" -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
echo "torch dir: $T"
ABI="$("$PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
echo "CXX11 ABI: $ABI"
"$PYTHON" -c "import torch; print('torch', torch.__version__)"

# Baseline flags (see logs/<run>/baseline-env.sh): no ALL_CODEBOOKS needed for
# K=4/6 mul1; DNNL explicitly off (no oneDNN tree on this machine).
export EXL3_NO_DNNL=1
unset EXL3_FLAGS || true
echo "EXL3_FLAGS: ${EXL3_FLAGS-<unset>}  EXL3_NO_DNNL=$EXL3_NO_DNNL"

FRESH="$EXL3_SRC/exl3xpu/_C.so.fresh-$(date +%Y%m%d-%H%M%S)"
rm -f "$FRESH"
echo "fresh output: $FRESH"

# shellcheck disable=SC2086
icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared \
  -fsycl-device-code-split=per_kernel "-D_GLIBCXX_USE_CXX11_ABI=$ABI" \
  -I "$EXL3_SRC/csrc" -I"$T/include" -I"$T/include/torch/csrc/api/include" \
  -x c++ "$EXL3_SRC/csrc/exl3_ops.sycl" -x none -o "$FRESH" \
  -L"$T/lib" -Wl,-rpath,"$T/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
echo "icpx exit: $?"

ls -la "$FRESH"
sha256sum "$FRESH" | tee "$LOG_DIR/artifact.sha256"

# Validate: load the FRESH artifact in a pristine interpreter (no vLLM import).
EXL3_LIB="$FRESH" "$PYTHON" - <<'PY'
import os, torch
lib = os.environ["EXL3_LIB"]
torch.ops.load_library(lib)
ops = torch.ops.exl3xpu_C
print("ops loaded from:", lib)
print("exl3_supported(4,2):", ops.exl3_supported(4, 2))
print("exl3_supported(6,2):", ops.exl3_supported(6, 2))
assert ops.exl3_supported(4, 2) and ops.exl3_supported(6, 2)
PY
echo "fresh-artifact load validation: PASS"

cp "$FRESH" "$EXL3_SRC/exl3xpu/_C.so"
ls -la "$EXL3_SRC/exl3xpu/_C.so"
echo "published to exl3xpu/_C.so"
echo "BUILD OK"
