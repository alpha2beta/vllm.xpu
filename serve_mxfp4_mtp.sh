#!/usr/bin/env bash
# serve_mxfp4_mtp.sh — experiment 1: coder390 MXFP4 + MTP speculative decoding.
#
# Identical to serve_mxfp4.sh in EVERY respect except one variable:
#   --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
# (same image, ctx 81920, util 0.95, fp8 KV, max-num-batched-tokens 4096,
#  --enforce-eager kept ON, same served name) so the A/B against
# serve_mxfp4.sh isolates MTP alone.
#
# Why this should load without the GPTQ stack's /patch_mtp.py: that patch exists
# because the auto_gptq checkpoint carries quantization_config but has
# dynamic=null, so qwen3_5_mtp.py's upstream mechanism (which nulls
# vllm_config.quant_config for the draft only when dynamic contains a
# "-:*mtp*" entry) never fires. This compressed-tensors checkpoint instead
# lists every mtp.* tensor in quantization_config.ignore, and all 15 are BF16
# on disk, so compressed_tensors.py's should_ignore_layer() builds the draft
# unquantized on its own. If the load fails on mtp.*, that assumption is wrong
# and the fix is to port patch_mtp_nightly.py to this image.
#
# Prediction before running: 7.962 x ~3.0 (mean acceptance) ≈ ~24 tok/s.
#
# First attempt used ctx 81920 (as serve_mxfp4.sh) and failed to INITIALIZE:
#   ValueError: To serve at least one request with the model's max seq len
#   (81920), (3.67 GiB KV cache is needed, but only 2.49 GiB is available).
#   Based on the available memory, the estimated maximum model length is 46592
# Not an MTP or quantization problem: the draft loaded fine (the log shows
# "Overriding draft model max model len from 262144 to 81920" and
# speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=4), and
# the compressed-tensors ignore list built it unquantized with no patch).
# It is the one-memory-budget effect (known issue 10): the draft's BF16 weights
# (~0.9 GiB) come out of the same pool, so leg A's 3.4 GiB of KV becomes 2.49 GiB
# while the requirement rises to 3.67 GiB. ctx is therefore reduced to 40960,
# comfortably inside the engine's own 46592 estimate so a ±8 % low-roll boot
# still fits. ctx is irrelevant to the p512/g128 decode rate being measured, so
# the A/B stays single-variable apart from this forced fit.
#
# Only ONE server runs at a time (each resident is ~17-19 GiB of 22.71).
set -u
IMG='vllm/vllm-openai-xpu@sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9'
MODEL_DIR='/mnt/stk/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4'
NAME='vllm-mxfp4-mtp-b60'
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
  exit 0
fi

for other in vllm-xpu-b60 vllm-exl3-b60 vllm-coder390-gptq vllm-mxfp4-b60; do
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
echo "MTP: method=mtp num_speculative_tokens=4  (all else identical to serve_mxfp4.sh)"
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
    --max-model-len 40960 --max-num-seqs 1 --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --max-num-batched-tokens 4096 \
    --language-model-only --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --speculative-config \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":4}\"'"

echo "started $NAME; waiting for healthy (cold start, up to ~15 min)…"
for i in $(seq 1 90); do
  if curl -s -m 5 "http://localhost:$HOST_PORT/v1/models" | grep -q "$SERVED_NAME"; then
    echo "HEALTHY after ~$((i*10)) s"
    echo "NEXT: confirm the draft resolved as BF16 (not quantized):"
    echo "  sg docker -c 'docker logs $NAME 2>&1' | grep -iE 'mtp|speculat|acceptance|draft' | head -20"
    exit 0
  fi
  # Fail fast instead of polling a corpse for 15 minutes: if the container is
  # gone, print the root cause now. The first (81920) attempt exited at 90 s and
  # this loop burned the remaining 13 minutes waiting on a dead container.
  if ! sg docker -c "docker ps --format '{{.Names}}'" 2>/dev/null | grep -qx "$NAME"; then
    echo "FAILED after ~$((i*10)) s: container $NAME is no longer running."
    echo "--- root cause ---"
    sg docker -c "docker logs $NAME 2>&1" | grep -E "ValueError|RuntimeError|Error:|error" | tail -5
    exit 1
  fi
  sleep 10
done
echo "NOT healthy after 15 min — inspect: sg docker -c 'docker logs $NAME' | tail -40"
exit 1
