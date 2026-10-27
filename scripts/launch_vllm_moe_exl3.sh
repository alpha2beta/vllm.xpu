#!/usr/bin/env bash
# Dedicated launcher for Qwen3.5-35B-A3B EXL3 MoE on Arc 140V (P3-05).
# Loopback port 8002 reserved for MoE testing (Tiel uses 8000/8080, Qwen3.8 dense uses 8001).
#
# Usage:
#   scripts/launch_vllm_moe_exl3.sh [label] [extra vllm serve args...]
# Env overrides:
#   MODEL, SERVED, MAX_LEN, MAX_SEQS, BATCHED_TOKENS, GPU_UTIL, PORT, HOST,
#   EAGER=1/0, PINNED=1/0, WARMUP=1/0,
#   SPEC=mtp-k1|mtp-k2|ngram-k<N> (empty disables speculation; default empty)

set -uo pipefail

LABEL=${1:-moe-3.09bpw-2k}
shift || true

MODEL=${MODEL:-models/turboderp-Qwen3.5-35B-A3B-exl3-3.09bpw}
SERVED=${SERVED:-qwen3.5-35b-a3b-exl3}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8002}
MAX_LEN=${MAX_LEN:-4096}
MAX_SEQS=${MAX_SEQS:-1}
BATCHED_TOKENS=${BATCHED_TOKENS:-4096}
GPU_UTIL=${GPU_UTIL:-0.65}
KV_CACHE_BYTES=${KV_CACHE_BYTES:-536870912}
EAGER=${EAGER:-1}
PINNED=${PINNED:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}" || exit 1
mkdir -p logs/moe-p3
LOG="logs/moe-p3/server-${LABEL}.log"

# Level-Zero headers
if [ ! -f tools/sysroot/usr/include/level_zero/ze_api.h ]; then
  bash scripts/setup_level_zero_headers.sh || exit 1
fi
export CPATH="$PWD/tools/sysroot/usr/include${CPATH:+:$CPATH}"

# Intel oneAPI environment
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >logs/moe-p3/oneapi-setvars.log 2>&1
set -u
export LD_LIBRARY_PATH="$PWD/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source .venv/bin/activate || { echo "failed to activate .venv"; exit 1; }
unset PYTHONHOME

# EXL3 environment
export EXL3_VLLM_PATCHES=0
export EXL3_SMALL_M_MAX=8
export EXL3_INT8_PREFILL=0
export EXL3_BACKEND="${EXL3_BACKEND:-auto}"
export EXL3_RECON_SLICE_N=1024
export EXL3_NO_DNNL=1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
echo "EXL3 MoE env: PATCHES=$EXL3_VLLM_PATCHES SMALL_M_MAX=$EXL3_SMALL_M_MAX BACKEND=$EXL3_BACKEND RECON_SLICE_N=$EXL3_RECON_SLICE_N SPEC=${SPEC:-off}"

# Pre-flight: drain GPUReclaim
if [ "${WARMUP:-1}" = "1" ]; then
  echo "--- pre-flight: draining Xe GPUReclaim pool ---"
  python scripts/reclaim_gpu_cache.py || echo "warning: reclaim incomplete"
fi

EAGER_FLAG=(--enforce-eager)
[ "$EAGER" = "0" ] && EAGER_FLAG=()

# Speculative decoding (SPEC=mtp-k1/mtp-k2/ngram-k<N>; empty = off, the
# certified Gate 4 default). Note: ${SPEC-} (no colon) so an explicit empty
# SPEC= disables speculation and an unset SPEC stays off.
SPEC=${SPEC-}
SPEC_FLAG=()
case "$SPEC" in
  "") ;;
  mtp-k1) SPEC_FLAG=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}') ;;
  mtp-k2) SPEC_FLAG=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}') ;;
  ngram-k*)
    K="${SPEC#ngram-k}"
    SPEC_FLAG=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${K},\"prompt_lookup_max\":${K},\"prompt_lookup_min\":2}")
    ;;
  *) echo "unknown SPEC='$SPEC' (expected mtp-k1, mtp-k2, ngram-k<N>, or empty)" >&2; exit 2 ;;
esac

KV_BYTES_FLAG=()
if [ -n "${KV_CACHE_BYTES:-}" ]; then
  if [[ ! " $* " =~ " --kv-cache-memory-bytes " ]]; then
    KV_BYTES_FLAG=(--kv-cache-memory-bytes "$KV_CACHE_BYTES")
  fi
fi

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_UTIL"
  "${KV_BYTES_FLAG[@]}"
  --language-model-only
  --seed 0
  "${EAGER_FLAG[@]}"
  "${SPEC_FLAG[@]}"
  "$@"
)

echo "vllm serve ${ARGS[*]}" | tee "$LOG"
echo "logging to $LOG" >&2

if [ "$PINNED" = "1" ] && command -v taskset >/dev/null 2>&1; then
  if taskset -c 0-3 true 2>/dev/null; then
    echo "pinning to P-cores 0-3 (PINNED=1)" | tee -a "$LOG"
    exec taskset -c 0-3 python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >>"$LOG" 2>&1
  else
    echo "warning: P-core pinning unavailable, running unpinned" | tee -a "$LOG"
  fi
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >>"$LOG" 2>&1
