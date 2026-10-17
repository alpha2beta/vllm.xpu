#!/usr/bin/env bash
# Production launcher for Tiel-Coder-35B-A3B W4A16-GPTQ on Intel Arc 140V (Lunar Lake 258V).
#
# Model: arcitech-psp/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP
#   - Format: compressed-tensors pack-quantized W4A16 (INT4 group 128) routed MoE experts,
#             BF16 attention, shared experts, and MTP draft head.
#   - Resident Weights: ~20.53 GiB (including draft model)
#   - Speculative Decoding: MTP K=2 default (SPEC=mtp-k2, supports SPEC=mtp-k3)
#   - CPU Affinity: Lion Cove P-cores 0-3 (taskset -c 0-3)
#   - Context Length: 131072 (128k; config ceiling is 262144)
#   - KV Cache: FP8 default (2.5 GB budget fits ~200k tokens, >1.5x concurrency @ 128k)
#   - Sampling: Calibrated anti-loop generation config (temp 0.6, rep_pen 1.08, pres_pen 0.2, top_p 0.95)
#   - Tool Calling: OpenAI auto tool choice with qwen3_xml parser
#
# Usage:
#   ./scripts/launch_vllm_tiel_gptq.sh [label] [extra vllm serve args...]
#   or from project root:
#   ./launch_vllm_tiel_gptq.sh [label] [extra vllm serve args...]
#
# Examples:
#   ./launch_vllm_tiel_gptq.sh
#   ./launch_vllm_tiel_gptq.sh prod-server --port 8000 --host 0.0.0.0
#   SPEC=mtp-k3 ./launch_vllm_tiel_gptq.sh k3-test
#   MAX_LEN=98304 ./launch_vllm_tiel_gptq.sh 96k
#   KV_DTYPE=auto KV_BYTES=3000000000 ./launch_vllm_tiel_gptq.sh bf16kv
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

LABEL="${1:-tiel-gptq}"
shift || true

# Production defaults
export MODEL="${MODEL:-models/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP}"
export SERVED="${SERVED:-Tiel-Coder-35B-A3B-W4A16-GPTQ}"
export MAX_LEN="${MAX_LEN:-131072}"
export MAX_SEQS="${MAX_SEQS:-1}"
export GPU_UTIL="${GPU_UTIL:-0.74}"
export SPEC="${SPEC:-mtp-k2}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# KV cache dtype and budget:
# With KV_DTYPE=fp8 (default), 2.5 GB yields ~200k tokens (concurrency >1.5x @ 128k) while
# keeping total resident memory at ~23.0 GiB (safe headroom on 28.58 GiB addressable, 24 GiB free).
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
