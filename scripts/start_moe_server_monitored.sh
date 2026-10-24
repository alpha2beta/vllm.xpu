#!/usr/bin/env bash
# Starts memory monitor and vLLM MoE server on port 8002 in background.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

mkdir -p logs/moe-p3

# Pre-flight reclaim
echo "Draining GPUReclaim pool..."
python scripts/reclaim_gpu_cache.py || true

# Start memory monitor
echo "Starting memory monitor (sampling every 1s)..."
nohup bash scripts/mem_monitor.sh moe-serving 1 >logs/moe-p3/mem_monitor.out 2>&1 &
MEM_PID=$!
disown "$MEM_PID" || true
echo "$MEM_PID" > logs/moe-p3/mem_monitor.pid

# Start server
echo "Starting vLLM MoE server on port 8002..."
nohup bash scripts/launch_vllm_moe_exl3.sh moe-serve-01 >logs/moe-p3/launcher.log 2>&1 &
SERVER_PID=$!
disown "$SERVER_PID" || true
echo "$SERVER_PID" > logs/moe-p3/server.pid

echo "Server started with PID $SERVER_PID, memory monitor PID $MEM_PID"
echo "Log file: logs/moe-p3/server-moe-serve-01.log"
