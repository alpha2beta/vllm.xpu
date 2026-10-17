#!/usr/bin/env bash
# serve_mxfp4.sh — coder390 MXFP4 server: Huihui Qwen3.8-27B Coder390
# EfficientThink (abliterated) via compressed-tensors mxfp4-pack-quantized,
# served by stock vLLM 0.30.0+xpu (CompressedTensorsW4A4Mxfp4 XPU kernel).
# Third A/B leg next to vllm-xpu-b60 (GPTQ-Int4) and vllm-exl3-b60 (EXL3).
#
# Grounding (model README under /mnt/stk/models/...):
#   496 packed modules (weight_packed + weight_scale, E2M1 group-32); bf16:
#   embed, lm_head, norms, GDN A_log/conv1d/dt_bias, 27-block vision tower,
#   and the full MTP draft block. Measured on Arc 140V with vLLM 0.30.0+xpu:
#     --max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.68
#     --language-model-only --enforce-eager
# B60 adaptations here: util 0.95, ctx 32768 (first boot), fp8 KV, seqs 1.
#   Weights are ~18.44 GiB, so at 0.95 x 22.71 only ~1-2 GiB is left for
#   KV+acts — read the engine's reported pool on first boot and adjust
#   ctx up/down exactly as in REPRODUCIBILITY.md §6 (EXL3: 65536->90000).
#   MTP is OFF on first boot (README's measured config); mtp.* ships bf16
#   so speculative decoding is the follow-up experiment, not day one.
#
# Only ONE server runs at a time (each resident is ~17-19 GiB of 22.71).
# This script refuses to start while vllm-xpu-b60 or vllm-exl3-b60 is up.
set -u
IMG='vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9'
MODEL_DIR='/mnt/stk/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4'
NAME='vllm-mxfp4-b60'
HOST_PORT='8002'
SERVED_NAME='coder390-mxfp4'
B60_RENDER_NODE='/dev/dri/renderD128'
RENDER_GID="$(stat -c '%g' "$B60_RENDER_NODE")"

if [ "${1:-}" = "--check" ]; then
  echo "=== $NAME ==="
  sg docker -c "docker ps --format '{{.Names}}  {{.Status}}  {{.Ports}}'" | grep -E "$NAME" || echo "$NAME not running"
  echo "--- API ---"
  curl -s -m 10 "http://localhost:$HOST_PORT/v1/models" | head -c 600; echo
  LAN_IP="$(hostname -I | awk '{print $1}')"
  echo "--- LAN: http://$LAN_IP:$HOST_PORT/v1  (model: $SERVED_NAME) ---"
  echo '--- opencode snippet (npm: @ai-sdk/openai-compatible):'
  echo "baseURL: 'http://$LAN_IP:$HOST_PORT/v1',  model: '$SERVED_NAME'"
  exit 0
fi

for other in vllm-xpu-b60 vllm-exl3-b60; do
  if sg docker -c "docker ps --format '{{.Names}}'" | grep -qx "$other"; then
    echo "REFUSING: $other is RUNNING — servers do not fit together in 22.71 GiB."
    echo "Stop it first:  sg docker -c 'docker stop $other'"
    exit 1
  fi
done
if [ ! -d "$MODEL_DIR" ]; then
  echo "MISSING: $MODEL_DIR (is /mnt/stk mounted?)"
  exit 1
fi

echo "B60_RENDER_NODE=$B60_RENDER_NODE  RENDER_GID=$RENDER_GID  ZE_AFFINITY_MASK=0"
sg docker -c "docker rm -f '$NAME'" >/dev/null 2>&1 || true

sg docker -c "docker run -d --name '$NAME' -p '$HOST_PORT:8000' \
  --device /dev/dri \
  --group-add '$RENDER_GID' \
  -v /dev/dri:/dev/dri:ro \
  -v '$MODEL_DIR:/model:ro' \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_LOG_STATS_INTERVAL=2 \
  --entrypoint bash '$IMG' -lc \
  'exec vllm serve /model --served-model-name $SERVED_NAME --port 8000 \
    --max-model-len 81920 --max-num-seqs 1 --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --max-num-batched-tokens 4096 \
    --language-model-only --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder'"

echo "started $NAME; waiting for healthy (cold start, up to ~15 min)…"
for i in $(seq 1 90); do
  if curl -s -m 5 "http://localhost:$HOST_PORT/v1/models" | grep -q "$SERVED_NAME"; then
    echo "HEALTHY after ~$((i*10)) s"
    echo "NEXT: read the engine pool from the log and tune ctx:"
    echo "  sg docker -c 'docker logs $NAME 2>&1' | grep -E 'KV cache size|Available KV|ValueError'"
    exit 0
  fi
  sleep 10
done
echo "NOT healthy after 15 min — inspect: sg docker -c 'docker logs $NAME' | tail -30"
exit 1
