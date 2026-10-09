#!/usr/bin/env bash
# Launch the OpenAI-compatible vLLM server for Qwen3.8-27B EXL3 2.20bpw.
#
# Defaults (override via env): MTP k=1 speculation, fp8 KV cache, 128k context.
# Isolated from the production Tiel launcher (scripts/launch_vllm.sh):
#   - loopback port 8001 (EXL3 reserved; Tiel uses 8000)
#   - text-only, eager, conservative memory defaults
#   - EXL3 baseline env (see logs/<run>/baseline-env.sh): no vLLM string
#     patches, SMALL_M_MAX=8, no INT8 prefill, 1024-col recon slices, no DNNL
#
# Usage:
#   ./launch_vllm_qwen38_exl3.sh [label] [extra vllm serve args...]
# Env overrides:
#   MODEL, SERVED, MAX_LEN, MAX_SEQS, BATCHED_TOKENS, GPU_UTIL, PORT, HOST,
#   EAGER=1/0, KV_DTYPE (default fp8), EXL3_BACKEND (default auto),
#   SPEC=mtp-k1/mtp-k2/ngram-k<N> (default mtp-k1; empty disables speculation),
#   PINNED=1/0, WARMUP=1/0
set -uo pipefail

LABEL=${1:-exl3-2bpw-mtp-fp8-128k}
shift || true

MODEL=${MODEL:-models/turboderp-Qwen3.8-27B-exl3-2.20bpw}
SERVED=${SERVED:-qwen3.8-27b-2bpw}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
MAX_LEN=${MAX_LEN:-131072}
MAX_SEQS=${MAX_SEQS:-1}
BATCHED_TOKENS=${BATCHED_TOKENS:-1024}
GPU_UTIL=${GPU_UTIL:-0.65}
EAGER=${EAGER:-1}
KV_DTYPE=${KV_DTYPE:-fp8}
PINNED=${PINNED:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}" || exit 1
mkdir -p logs
LOG="logs/server-${LABEL}.log"

# Triton-XPU JIT needs <level_zero/ze_api.h> (same sysroot pattern as Tiel launcher).
if [ ! -f tools/sysroot/usr/include/level_zero/ze_api.h ]; then
  bash scripts/setup_level_zero_headers.sh || exit 1
fi
export CPATH="$PWD/tools/sysroot/usr/include${CPATH:+:$CPATH}"

set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >logs/oneapi-setvars-exl3.log 2>&1
set -u
# oneCCL 2022.1 (torch 2.14) dlopens its plugin (libccl.so.1) via the loader
# search path — prepend the venv lib dir holding the pip oneAPI runtime.
export LD_LIBRARY_PATH="$PWD/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source .venv/bin/activate || { echo "failed to activate .venv"; exit 1; }
unset PYTHONHOME

# --- EXL3 baseline environment (approved E1.4; do not change without a task) ---
export EXL3_VLLM_PATCHES=0
export EXL3_SMALL_M_MAX=8
export EXL3_INT8_PREFILL=0
export EXL3_BACKEND="${EXL3_BACKEND:-auto}"
export EXL3_RECON_SLICE_N=1024
export EXL3_NO_DNNL=1
# K=4/6 mul1 are default kernel instantiations: leave EXL3_FLAGS unset.
# Fused path stays off; no EXL3_DRAFT_VOCAB (text-only baseline, no MTP).
# --- end EXL3 baseline ---

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
echo "EXL3 env: PATCHES=$EXL3_VLLM_PATCHES SMALL_M_MAX=$EXL3_SMALL_M_MAX BACKEND=$EXL3_BACKEND RECON_SLICE_N=$EXL3_RECON_SLICE_N"

if [ "${WARMUP:-0}" = "1" ]; then
  echo "--- pre-flight: draining Xe GPUReclaim pool ---"
  python scripts/reclaim_gpu_cache.py || echo "warning: reclaim incomplete (see above)"
fi

EAGER_FLAG=(--enforce-eager)
[ "$EAGER" = "0" ] && EAGER_FLAG=()

# Speculative decoding (SPEC=mtp-k1 default; mtp-k2, ngram-k<N> e.g.
# ngram-k4; SPEC= for off).
# Note: ${SPEC-mtp-k1} (no colon) so an explicit empty SPEC= disables,
# while an unset SPEC defaults to mtp-k1.
SPEC=${SPEC-mtp-k1}
SPEC_FLAG=()
[ "$SPEC" = "mtp-k1" ] && SPEC_FLAG=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}')
[ "$SPEC" = "mtp-k2" ] && SPEC_FLAG=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
case "$SPEC" in
  ngram-k*)
    K="${SPEC#ngram-k}"
    SPEC_FLAG=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${K},\"prompt_lookup_max\":${K},\"prompt_lookup_min\":2}")
    ;;
esac

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_UTIL"
  --kv-cache-dtype "$KV_DTYPE"
  --language-model-only
  --seed 0
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  "${EAGER_FLAG[@]}"
  "${SPEC_FLAG[@]}"
  "$@"
)

echo "vllm serve ${ARGS[*]}" | tee "$LOG"
echo "logging to $LOG" >&2
if [ "$PINNED" = "1" ] && command -v taskset >/dev/null 2>&1; then
  if taskset -c 0-3 true 2>/dev/null; then
    echo "pinning to P-cores 0-3 (PINNED=1; skip with PINNED=0)" | tee -a "$LOG"
    exec taskset -c 0-3 python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >>"$LOG" 2>&1
  else
    echo "warning: P-core pinning unavailable in this cpuset ($(taskset -pc $$ 2>&1)); running unpinned (PINNED=0 to silence)" | tee -a "$LOG"
  fi
fi
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >>"$LOG" 2>&1
