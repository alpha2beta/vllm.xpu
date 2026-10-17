#!/usr/bin/env bash
# serve_exl3.sh — alternative server: Qwen3.8-27B EXL3 4.00bpw via the
# 0xSero/exl3xpu vLLM plugin, adapted from the B70 recipe to the B60.
# A/B counterpart to vllm-xpu-b60 (GPTQ-Int4, vLLM 0.27.2).
#
# B70 recipe (models/qwen3.8-27b-exl3-4.00bpw/model.yaml): ctx 262144,
#   seqs 16, util 0.965, MTP k=3, fp8 KV, prefix caching on.
# B60 adaptations (--set, serve.py feature): ctx 65536, seqs 4, util 0.95.
#   262144 is impossible here (8.4 GiB KV + 14.91 weights = 23.3 GiB >
#   22.71 before any overhead; see REPRODUCIBILITY.md §6).
#
# Only ONE server runs at a time: GPTQ resident is ~18.1 GiB, EXL3 will be
# ~17-19 GiB; both do not fit in 22.71 GiB. This script refuses to start
# while vllm-xpu-b60 is running.
set -u
IMG='ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8'
MODEL_DIR='/mnt/usb/models/turboderp-Qwen3.8-27B-exl3-4.00bpw'
NAME='vllm-exl3-b60'
HOST_PORT='8081'
B60_RENDER_NODE='/dev/dri/renderD128'
RENDER_GID="$(stat -c '%g' "$B60_RENDER_NODE")"

if [ "${1:-}" = "--check" ]; then
  echo "=== $NAME ==="
  sg docker -c "docker ps --format '{{.Names}}  {{.Status}}  {{.Ports}}'" | grep -E "$NAME" || echo "$NAME not running"
  echo "--- API ---"
  curl -s -m 10 "http://localhost:$HOST_PORT/v1/models" | head -c 600; echo
  LAN_IP="$(hostname -I | awk '{print $1}')"
  echo "--- LAN: http://$LAN_IP:$HOST_PORT/v1  (model: qwen3.8-27b-exl3) ---"
  echo '--- opencode snippet (npm: @ai-sdk/openai-compatible):'
  echo "baseURL: 'http://$LAN_IP:$HOST_PORT/v1',  model: 'qwen3.8-27b-exl3'"
  exit 0
fi

if sg docker -c "docker ps --format '{{.Names}}'" | grep -qx 'vllm-xpu-b60'; then
  echo "REFUSING: vllm-xpu-b60 is RUNNING — both servers do not fit in 22.71 GiB."
  echo "Stop it first:  sg docker -c 'docker stop vllm-xpu-b60'"
  exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "MISSING: $MODEL_DIR (hf download still running? see /tmp/opencode/exl3/hf.log)"
  exit 1
fi

echo "B60_RENDER_NODE=$B60_RENDER_NODE  RENDER_GID=$RENDER_GID  ZE_AFFINITY_MASK=0"
sg docker -c "docker rm -f '$NAME'" >/dev/null 2>&1 || true

sg docker -c "docker run -d --name '$NAME' -p '$HOST_PORT:8081' \
  --device /dev/dri \
  --group-add '$RENDER_GID' \
  -v /dev/dri:/dev/dri:ro \
  -v '$MODEL_DIR:/models:ro' \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_LOG_STATS_INTERVAL=2 \
  '$IMG' \
  models/qwen3.8-27b-exl3-4.00bpw --gpu 0 --port 8081 --model-path /models \
  --set vllm.max_model_len=90000 \
  --set vllm.max_num_seqs=4 \
  --set vllm.gpu_memory_utilization=0.95 \
  -- --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"

echo "started $NAME; waiting for healthy (cold start, up to ~15 min)…"
for i in $(seq 1 90); do
  if curl -s -m 5 "http://localhost:$HOST_PORT/v1/models" | grep -q 'qwen3.8-27b-exl3'; then
    echo "HEALTHY after ~$((i*10)) s"
    exit 0
  fi
  sleep 10
done
echo "NOT healthy after 15 min — inspect: sg docker -c 'docker logs $NAME' | tail -30"
exit 1
