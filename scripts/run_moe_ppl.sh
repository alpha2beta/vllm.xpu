#!/usr/bin/env bash
# Runs WikiText-2 perplexity evaluation with oneAPI environment.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT_DIR}" || exit 1

if [ ! -f tools/sysroot/usr/include/level_zero/ze_api.h ]; then
  bash scripts/setup_level_zero_headers.sh || exit 1
fi
export CPATH="$PWD/tools/sysroot/usr/include${CPATH:+:$CPATH}"

set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u

export LD_LIBRARY_PATH="$PWD/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source .venv/bin/activate
unset PYTHONHOME

export EXL3_VLLM_PATCHES=0
export EXL3_SMALL_M_MAX=8
export EXL3_INT8_PREFILL=0
export EXL3_BACKEND=auto
export EXL3_RECON_SLICE_N=1024
export EXL3_NO_DNNL=1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Pre-flight reclaim
python scripts/reclaim_gpu_cache.py || true

mkdir -p logs/moe-p3
python scripts/eval_moe_ppl.py "$@" 2>&1 | tee logs/moe-p3/eval-ppl.log
