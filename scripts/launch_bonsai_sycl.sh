#!/usr/bin/env bash
# Dedicated launcher for Ternary Bonsai models on Intel Arc using the optimized SYCL engine.
set -euo pipefail

MODEL="${MODEL:-/home/yanchun/llama.cpp/models/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
CONTEXT="${CONTEXT:-16384}"
REASONING="${REASONING:-medium}"
THREADS="${THREADS:-8}"
BIN="/home/yanchun/arc-b580/build-sycl/bin/llama-server"

set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u

export LD_LIBRARY_PATH="/home/yanchun/arc-b580/build-sycl/bin:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export ZES_ENABLE_SYSMAN=1

echo "Starting llama-server with SYCL engine..."
echo "Model: $MODEL"
echo "Port: $PORT | Context: $CONTEXT | Reasoning: $REASONING"

exec "$BIN" \
  --model "$MODEL" \
  --ctx-size "$CONTEXT" \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0 \
  --repeat-penalty 1.0 \
  --alias "ternary-bonsai-2-27b" \
  --host "$HOST" \
  --port "$PORT" \
  --device SYCL0 \
  --n-gpu-layers 999 \
  --no-host \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning-effort "$REASONING" \
  "$@"
