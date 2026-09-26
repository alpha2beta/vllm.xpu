#!/usr/bin/env bash
# Phase 7: launch the OpenAI-compatible vLLM server for the MXFP4 Qwen3.6 model.
#
# Every flag below was confirmed present in `vllm serve --help` for the pinned
# build (captured in logs/vllm-serve-help.txt):
#   --model --served-model-name --host --port --max-model-len --max-num-seqs
#   --max-num-batched-tokens --gpu-memory-utilization --enforce-eager
#   --language-model-only --seed --kv-cache-dtype --tool-call-parser
# NOTE: `--swap-space` does NOT exist in this build (V1 engine) — vLLM will not
# page to host swap itself; host swap can only come from the OS.
#
# Usage:
#   scripts/launch_vllm.sh [label] [extra vllm serve args...]
# Env overrides:
#   MODEL, MAX_LEN, MAX_SEQS, BATCHED_TOKENS, GPU_UTIL, PORT, HOST, EAGER=1/0
set -uo pipefail

LABEL=${1:-server-01}
shift || true

MODEL=${MODEL:-models/Qwen3.6-35B-A3B-MXFP4}
SERVED=${SERVED:-Qwen3.6-35B-A3B-MXFP4}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-1}
BATCHED_TOKENS=${BATCHED_TOKENS:-4096}
# Measured ceiling: Level-Zero reports 22.03/28.58 GiB free at startup with a
# desktop session running, so util must be <= 0.77; text-only weights are 19.24 GiB.
GPU_UTIL=${GPU_UTIL:-0.74}
EAGER=${EAGER:-1}
LOG="logs/server-${LABEL}.log"

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

# Triton-XPU JIT needs <level_zero/ze_api.h> (Arch's `level-zero-headers`).
if [ ! -f tools/sysroot/usr/include/level_zero/ze_api.h ]; then
  bash scripts/setup_level_zero_headers.sh || exit 1
fi
export CPATH="$PWD/tools/sysroot/usr/include${CPATH:+:$CPATH}"

# setvars.sh references unset variables (e.g. OCL_ICD_FILENAMES), which kills a
# `set -u` shell — suspend nounset while sourcing it.
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >logs/oneapi-setvars.log 2>&1
set -u
source .venv/bin/activate || { echo "failed to activate .venv"; exit 1; }
unset PYTHONHOME

# Pin the SYCL/UR device — otherwise the vLLM-XPU SYCL kernels can land on the
# Mesa "rusticl" GPU and fail with "Device does not support device USM allocations".
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"

# Pre-flight: drain the Xe driver's GPUReclaim pool (see scripts/reclaim_gpu_cache.py);
# Level-Zero does not count it as free, which otherwise fails vLLM's startup check.
if [ "${WARMUP:-1}" = "1" ]; then
  echo "--- pre-flight: draining Xe GPUReclaim pool ---"
  python scripts/reclaim_gpu_cache.py || echo "warning: reclaim incomplete (see above)"
fi

EAGER_FLAG=(--enforce-eager)
[ "$EAGER" = "0" ] && EAGER_FLAG=()

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_UTIL"
  --language-model-only
  --seed 0
  "${EAGER_FLAG[@]}"
  "$@"
)

echo "vllm serve ${ARGS[*]}" | tee "$LOG"
echo "logging to $LOG" >&2
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >>"$LOG" 2>&1
