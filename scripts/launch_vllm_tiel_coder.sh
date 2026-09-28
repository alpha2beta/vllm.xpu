#!/usr/bin/env bash
# Production launcher for Tiel-Coder-35B-A3B on Intel Arc 140V (Lunar Lake 258V).
#
# Config: Config G (Selected Production Configuration)
#   - Model: Tiel-Coder-35B-A3B MXFP4 (19.71 GiB resident weights)
#   - Speculative Decoding: MTP K=2 with in-place MXFP4 draft model (SPEC=mtp-k2)
#   - CPU Affinity: Lion Cove P-cores 0-3 (taskset -c 0-3)
#   - Memory: 0.5 GiB pinned KV cache (--kv-cache-memory-bytes 500000000), GPU_UTIL=0.68
#   - Context Length: 4096 (maxlen fits comfortably within memory envelope)
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
export GPU_UTIL="${GPU_UTIL:-0.68}"
export SPEC="${SPEC:-mtp-k2}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"

cd "${ROOT_DIR}" || exit 1

exec "${LAUNCH_SCRIPT}" "${LABEL}" \
  --kv-cache-memory-bytes 3000000000 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  "$@"
