#!/usr/bin/env bash
# Phase 5 wrapper: run the offline load test under a hard timeout while
# sampling host memory/swap. Designed to survive the known
# `profile_run()` startup hang on Lunar Lake (see compatibility.md §3.5).
#
# Usage: scripts/run_phase5.sh <label> [extra smoke_offline.py args...]
# e.g.  scripts/run_phase5.sh smoke-01 --language-model-only --max-model-len 1024
set -uo pipefail

LABEL=${1:?usage: run_phase5.sh <label> [args...]}
shift
TIMEOUT_S=${TIMEOUT_S:-1800}

mkdir -p logs
LOG="logs/smoke-${LABEL}.log"
echo "logging to $LOG ; timeout ${TIMEOUT_S}s"

# environment: oneAPI + the experiment venv
cd "$(dirname "$0")/.." || exit 1

# Triton-XPU JIT needs <level_zero/ze_api.h> (Arch's `level-zero-headers`).
if [ ! -f tools/sysroot/usr/include/level_zero/ze_api.h ]; then
  bash scripts/setup_level_zero_headers.sh || exit 1
fi
export CPATH="$PWD/tools/sysroot/usr/include${CPATH:+:$CPATH}"
echo "CPATH=$CPATH"
# setvars.sh references unset variables (e.g. OCL_ICD_FILENAMES), which kills a
# `set -u` shell — suspend nounset while sourcing it.
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >logs/oneapi-setvars.log 2>&1
SETVARS_RC=$?
set -u
[ "$SETVARS_RC" -ne 0 ] && echo "warning: setvars.sh rc=$SETVARS_RC (see logs/oneapi-setvars.log)"
source .venv/bin/activate || { echo "failed to activate .venv"; exit 1; }
unset PYTHONHOME  # a stale PYTHONHOME breaks the venv interpreter
echo "python: $(command -v python) ($(python -V 2>&1))"
python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)" \
  || { echo "FATAL: torch/vllm not importable from the venv python"; exit 1; }

# Env: pin the SYCL/UR device. Without this the Level-Zero device, Mesa
# "rusticl" and Intel OpenCL NEO are all visible as GPUs, and the vLLM-XPU
# SYCL kernels fail with "Device does not support device USM allocations".
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"

# Pre-flight: the Xe driver caches freed device memory in GPUReclaim, which
# Level-Zero does not count as free -> vLLM's startup check fails after any
# previous big/crashed run. Drain it first (no-op when the pool is empty).
if [ "${WARMUP:-1}" = "1" ]; then
  echo "--- pre-flight: draining Xe GPUReclaim pool ---"
  python scripts/reclaim_gpu_cache.py || echo "warning: reclaim incomplete (see above)"
fi

# memory / swap sampling alongside the test
bash scripts/mem_monitor.sh "$LABEL" 2 >/dev/null 2>&1 &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null' EXIT

set -o pipefail
timeout --signal=TERM --kill-after=60 "$TIMEOUT_S" \
  python scripts/smoke_offline.py "$@" 2>&1 | tee "$LOG"
rc=$?

kill $MON_PID 2>/dev/null
trap - EXIT

echo "exit code: $rc"
if [ "$rc" -eq 124 ]; then
  echo "TIMEOUT after ${TIMEOUT_S}s — check $LOG tail and logs/mem-${LABEL}.csv for a startup hang"
fi
if [ "$rc" -eq 137 ]; then
  echo "KILLED (SIGKILL) — likely OOM-killed by the kernel; check dmesg / logs/mem-${LABEL}.csv"
fi

echo "--- memory sample (last 5 rows) ---"
tail -5 "logs/mem-${LABEL}.csv" 2>/dev/null
exit "$rc"
