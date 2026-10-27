#!/usr/bin/env bash
# Production launcher for Sage-A1.5-EXL3-4.0bpw-mtp on Intel Arc 140V (Lunar Lake 258V).
#
# Config: Config G pattern for EXL3 MoE
#   - Model: Sage-A1.5-EXL3-4.0bpw-mtp (~17.55 GiB resident weights, incl. MTP draft)
#   - Speculative Decoding: MTP K=2 with in-place EXL3 draft model (SPEC=mtp-k2)
#   - CPU Affinity: Lion Cove P-cores 0-3 (taskset -c 0-3, launcher default)
#   - Context Length: 131072 (128k; max_position_embeddings ceiling is 262144)
#   - KV Cache: FP8 default (2.5 GB budget fits ~244k tokens, >1.8x concurrency @ 128k)
#   - Sampling: Calibrated anti-loop generation config (temp 0.6, rep_pen 1.08, pres_pen 0.2, top_p 0.95)
#   - Tool Calling: OpenAI auto tool choice with qwen3_xml parser
#   - Port: 8002 (dedicated EXL3 MoE port; configurable via PORT=...)
#
# Usage:
#   ./scripts/launch_vllm_sage_coder.sh [label] [extra vllm serve args...]
#   or from project root:
#   ./launch_vllm_sage_coder.sh [label] [extra vllm serve args...]
#
# Examples:
#   ./launch_vllm_sage_coder.sh
#   ./launch_vllm_sage_coder.sh prod-server --port 8002 --host 0.0.0.0
#   MAX_LEN=98304 ./launch_vllm_sage_coder.sh 96k
#   KV_DTYPE=auto ./launch_vllm_sage_coder.sh bf16kv
set -euo pipefail

CALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${CALL_DIR}/launch_vllm_moe_exl3.sh" ]; then
  ROOT_DIR="$(cd "${CALL_DIR}/.." && pwd)"
  LAUNCH_SCRIPT="${CALL_DIR}/launch_vllm_moe_exl3.sh"
elif [ -f "${CALL_DIR}/scripts/launch_vllm_moe_exl3.sh" ]; then
  ROOT_DIR="${CALL_DIR}"
  LAUNCH_SCRIPT="${CALL_DIR}/scripts/launch_vllm_moe_exl3.sh"
else
  # Fallback to physical path resolution
  PHYS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  ROOT_DIR="$(cd "${PHYS_DIR}/.." && pwd)"
  LAUNCH_SCRIPT="${PHYS_DIR}/launch_vllm_moe_exl3.sh"
fi

LABEL="${1:-sage-coder}"
shift || true

# Production defaults
export MODEL="${MODEL:-/home/yanchun/models/Sage-A1.5-EXL3-4.0bpw-mtp}"
export SERVED="${SERVED:-Sage-A1.5-Coder}"
export MAX_LEN="${MAX_LEN:-131072}"
export MAX_SEQS="${MAX_SEQS:-1}"
export BATCHED_TOKENS="${BATCHED_TOKENS:-4096}"
export GPU_UTIL="${GPU_UTIL:-0.74}"
export SPEC="${SPEC:-mtp-k2}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8002}"

# KV cache dtype and budget:
# Qwen3.5 has 10 full attention layers (full_attention_interval=4), 2 KV heads, head_dim=256.
# With KV_DTYPE=fp8 (default), 1 token = 10 layers * 2 heads * 256 dim * 2 (K+V) * 1 byte = 10,240 bytes (~10 KB).
# A 2.5 GB budget yields ~244k tokens (>1.8x concurrency @ 128k context) while
# keeping total resident memory at ~20.5-21.5 GiB (safe 2.5+ GiB headroom on 24 GiB free).
# If KV_DTYPE=auto (BF16), 3.0 GB yields ~146k tokens (>1.1x @ 128k).
KV_DTYPE="${KV_DTYPE:-fp8}"
if [ "${KV_DTYPE}" = "fp8" ]; then
  KV_BYTES="${KV_BYTES:-2500000000}"
else
  KV_BYTES="${KV_BYTES:-3000000000}"
fi
export KV_CACHE_BYTES="${KV_BYTES}"

# Anti-loop / calibrated generation settings
# Server-wide sampling defaults to prevent repetitive reasoning loops:
# temperature=0.6, repetition_penalty=1.08, presence_penalty=0.2, top_p=0.95
OVERRIDE_GEN_CONFIG=${OVERRIDE_GEN_CONFIG:-'{"temperature":0.6,"repetition_penalty":1.08,"presence_penalty":0.2,"top_p":0.95}'}

GEN_CONFIG_FLAG=()
[ -n "${OVERRIDE_GEN_CONFIG:-}" ] && GEN_CONFIG_FLAG=(--override-generation-config "$OVERRIDE_GEN_CONFIG")

cd "${ROOT_DIR}" || exit 1

exec "${LAUNCH_SCRIPT}" "${LABEL}" \
  --kv-cache-dtype "${KV_DTYPE}" \
  "${GEN_CONFIG_FLAG[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  "$@"
