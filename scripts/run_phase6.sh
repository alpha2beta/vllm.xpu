#!/usr/bin/env bash
# Phase 6 wrapper: run the fixed prompt suite (5 prompts x 3 runs) offline,
# with the same environment fixes the load test needs.
#
# Usage: scripts/run_phase6.sh <label> [extra run_prompt_suite.py args...]
set -uo pipefail

LABEL=${1:?usage: run_phase6.sh <label> [args...]}
shift
TIMEOUT_S=${TIMEOUT_S:-3600}
KV_BYTES=${KV_BYTES:-1073741824}      # 1 GiB pinned KV cache (~5k tokens)
MAX_LEN=${MAX_LEN:-4096}

mkdir -p logs
LOG="logs/prompt-suite-${LABEL}.log"
echo "logging to $LOG ; timeout ${TIMEOUT_S}s ; KV_BYTES=$KV_BYTES MAX_LEN=$MAX_LEN"

cd "$(dirname "$0")/.." || exit 1

if [ ! -f tools/sysroot/usr/include/level_zero/ze_api.h ]; then
  bash scripts/setup_level_zero_headers.sh || exit 1
fi
export CPATH="$PWD/tools/sysroot/usr/include${CPATH:+:$CPATH}"

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"

set +u
source /opt/intel/oneapi/setvars.sh >logs/oneapi-setvars.log 2>&1
set -u
source .venv/bin/activate || { echo "failed to activate .venv"; exit 1; }
unset PYTHONHOME

if [ "${WARMUP:-1}" = "1" ]; then
  echo "--- pre-flight: draining Xe GPUReclaim pool ---"
  python scripts/reclaim_gpu_cache.py || echo "warning: reclaim incomplete (see above)"
fi

bash scripts/mem_monitor.sh "$LABEL" 2 >/dev/null 2>&1 &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null' EXIT

set -o pipefail
timeout --signal=TERM --kill-after=60 "$TIMEOUT_S" \
  python scripts/run_prompt_suite.py \
    --mode offline \
    --model-path models/Qwen3.6-35B-A3B-MXFP4 \
    --max-model-len "$MAX_LEN" \
    --max-num-batched-tokens "$MAX_LEN" \
    --kv-cache-memory-bytes "$KV_BYTES" \
    --out "logs/prompt-suite/run-${LABEL}.jsonl" \
    "$@" 2>&1 | tee "$LOG"
rc=$?

kill $MON_PID 2>/dev/null
trap - EXIT
echo "exit code: $rc"
echo "--- memory sample (last 5 rows) ---"
tail -5 "logs/mem-${LABEL}.csv" 2>/dev/null
exit "$rc"
