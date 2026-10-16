#!/usr/bin/env bash
# Production launcher for Tiel-Coder-35B-A3B on Intel Arc 140V (Lunar Lake 258V).
#
# Config: Config G (Selected Production Configuration)
#   - Model: Tiel-Coder-35B-A3B MXFP4 (19.71 GiB resident weights)
#   - Speculative Decoding: MTP K=2 with in-place MXFP4 draft model (SPEC=mtp-k2)
#   - CPU Affinity: Lion Cove P-cores 0-3 (taskset -c 0-3)
#   - Memory: FP8 KV cache default (2.5 GiB budget fits ~180k+ tokens, >1.8x concurrency @ 96k)
#   - Context Length: 98304 (96k; configurable up to 131072 / 128k)
#   - Sampling: Calibrated anti-loop generation config (temp 0.6, rep_pen 1.08, pres_pen 0.2, top_p 0.95)
#   - Tool Calling: OpenAI auto tool choice with qwen3_xml parser
#
# Usage:
#   ./scripts/launch_vllm_tiel_coder.sh [label] [extra vllm serve args...]
#   or from project root:
#   ./launch_vllm_tiel_coder.sh [label] [extra vllm serve args...]
#
# Examples:
#   ./launch_vllm_tiel_coder.sh
#   ./launch_vllm_tiel_coder.sh prod-server --port 8080 --host 0.0.0.0
#   MAX_LEN=131072 ./launch_vllm_tiel_coder.sh 128k
#   KV_DTYPE=auto KV_BYTES=3000000000 ./launch_vllm_tiel_coder.sh bf16kv
set -euo pipefail

CALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${CALL_DIR}/launch_vllm.sh" ]; then
  ROOT_DIR="$(cd "${CALL_DIR}/.." && pwd)"
  LAUNCH_SCRIPT="${CALL_DIR}/launch_vllm.sh"
elif [ -f "${CALL_DIR}/scripts/launch_vllm.sh" ]; then
  ROOT_DIR="${CALL_DIR}"
  LAUNCH_SCRIPT="${CALL_DIR}/scripts/launch_vllm.sh"
else
  # Fallback to physical path resolution
  PHYS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd "${PHYS_DIR}/.." && pwd)"
  LAUNCH_SCRIPT="${PHYS_DIR}/launch_vllm.sh"
fi

LABEL="${1:-tiel-coder}"
shift || true

# Production defaults (Config G)
export MODEL="${MODEL:-models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4}"
export SERVED="${SERVED:-Tiel-Coder-35B-A3B}"
export MAX_LEN="${MAX_LEN:-98304}"
export MAX_SEQS="${MAX_SEQS:-1}"
export GPU_UTIL="${GPU_UTIL:-0.74}"
export SPEC="${SPEC:-mtp-k2}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"

# KV cache dtype and budget:
# With KV_DTYPE=fp8 (default), 2.5 GB yields ~180k+ tokens (concurrency >1.8x @ 96k, >1.38x @ 128k) while
# keeping total resident memory at ~22.0 GiB (safe 1.5+ GiB headroom on 24 GiB free).
# If KV_DTYPE=auto (BF16), 3.0 GB yields ~71k tokens.
KV_DTYPE="${KV_DTYPE:-fp8}"
if [ "${KV_DTYPE}" = "fp8" ]; then
  KV_BYTES="${KV_BYTES:-2500000000}"
else
  KV_BYTES="${KV_BYTES:-3000000000}"
fi

# Anti-loop / calibrated generation settings
# Server-wide sampling defaults to prevent repetitive reasoning loops:
# temperature=0.6, repetition_penalty=1.08, presence_penalty=0.2, top_p=0.95
OVERRIDE_GEN_CONFIG=${OVERRIDE_GEN_CONFIG:-'{"temperature":0.6,"repetition_penalty":1.08,"presence_penalty":0.2,"top_p":0.95}'}

GEN_CONFIG_FLAG=()
[ -n "${OVERRIDE_GEN_CONFIG:-}" ] && GEN_CONFIG_FLAG=(--override-generation-config "$OVERRIDE_GEN_CONFIG")

cd "${ROOT_DIR}" || exit 1

exec "${LAUNCH_SCRIPT}" "${LABEL}" \
  --kv-cache-memory-bytes "${KV_BYTES}" \
  --kv-cache-dtype "${KV_DTYPE}" \
  "${GEN_CONFIG_FLAG[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  "$@"
