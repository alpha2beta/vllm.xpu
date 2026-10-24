#!/usr/bin/env bash
# Stops memory monitor and vLLM MoE server.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

if [ -f logs/moe-p3/server.pid ]; then
  SPID=$(cat logs/moe-p3/server.pid)
  echo "Killing server tree rooted at PID $SPID..."
  # Kill child processes as well (EngineCore etc.)
  pkill -TERM -P "$SPID" 2>/dev/null || true
  kill -TERM "$SPID" 2>/dev/null || true
  sleep 2
  pkill -9 -P "$SPID" 2>/dev/null || true
  kill -9 "$SPID" 2>/dev/null || true
  rm -f logs/moe-p3/server.pid
fi

# Clean up any lingering python api_server processes on port 8002
fuser -k 8002/tcp 2>/dev/null || true

if [ -f logs/moe-p3/mem_monitor.pid ]; then
  MPID=$(cat logs/moe-p3/mem_monitor.pid)
  echo "Stopping memory monitor (PID $MPID)..."
  pkill -TERM -P "$MPID" 2>/dev/null || true
  kill -TERM "$MPID" 2>/dev/null || true
  sleep 1
  kill -9 "$MPID" 2>/dev/null || true
  rm -f logs/moe-p3/mem_monitor.pid
fi

echo "Copying memory traces..."
if [ -f logs/mem-moe-serving.csv ]; then
  cp logs/mem-moe-serving.csv logs/moe-p3/mem-trace.csv
  cp logs/mem-moe-serving.csv logs/moe-p3/full-load-mem-trace.csv
  echo "Wrote logs/moe-p3/mem-trace.csv and logs/moe-p3/full-load-mem-trace.csv"
fi

echo "Draining GPU cache..."
python scripts/reclaim_gpu_cache.py || true
echo "Done."
