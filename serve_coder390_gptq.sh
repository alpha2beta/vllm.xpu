#!/usr/bin/env bash
# serve_coder390_gptq.sh - serve the freshly quantized GPTQ-Int4 coder390 model
# (Huihui Qwen3.8-27B Coder390 EfficientThink abliterated) on the B60.
#
# Same launch recipe as phase5_launch.sh (the launch command of record):
#   * GPTQ dispatch, fp16, fp8 KV cache, --language-model-only
#   * MTP speculative decoding k=4 via the BF16 draft block
#     (the `dynamic` exclusion in the quantize config keeps mtp.* unquantized)
#   * the two engine patches the recorded launch applies
# Differences vs. the recorded launch: new model dir, host port 8003, served
# name `coder390-gptq`, and `--reasoning-parser qwen3`.
#
# The reasoning parser is not cosmetic. Without it the engine emits `</think>`
# as a token but the detokenizer *suppresses* it from `content`, so the whole
# reasoning trace arrives concatenated in front of the answer with nothing to
# split on — `reasoning` stays null too. Any client that wants the final answer
# alone (vLLM's own quality harnesses, agent loops) then scores the model on its
# private musings. `--reasoning-parser qwen3` matches `vllm/parser/qwen3.py`
# (THINK_START/THINK_END are exactly this model's tags, matched by token id) and
# is what serve_exl3.sh already does. See REPRODUCIBILITY.md.
#
# `--restart unless-stopped` is load-bearing, not boilerplate: vLLM's *first*
# engine init on a cold container deterministically fails with
#   ValueError: ... 1.24 GiB KV cache is needed, which is larger than the
#   available KV cache memory (0.18 GiB)
# because torch.compile's transient pool eats the KV budget; the retry succeeds
# against a warm compile cache (`REPRODUCIBILITY.md` known issue 1). Docker
# restarting the *same* container preserves that cache — so without this flag a
# cold start just exits (1) and never comes up. phase5_launch.sh and
# serve_8081.sh both carry it for the same reason.
#
# Usage:
#   ./serve_coder390_gptq.sh              # start (refuses if another server runs)
#   ./serve_coder390_gptq.sh --check      # no docker changes: state + LAN URL
#   ./serve_coder390_gptq.sh --dry-run    # print the docker command
set -u

IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL_DIR='/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-GPTQ-Int4-sym-G128-MTP-BF16'
COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
NAME='vllm-coder390-gptq'
HOST_PORT='8003'
SERVED_NAME='coder390-gptq'
B60_RENDER_NODE='/dev/dri/renderD128'
RENDER_GID="$(stat -c '%g' "$B60_RENDER_NODE")"

if [ "${1:-}" = "--check" ]; then
  echo "=== $NAME ==="
  sg docker -c "docker ps --format '{{.Names}}  {{.Status}}  {{.Ports}}'" | grep -F "$NAME" || echo "$NAME not running"
  echo "=== model ==="
  { [ -f "$MODEL_DIR/model.safetensors.index.json" ] && du -sh "$MODEL_DIR"; } || echo "NOT READY: $MODEL_DIR"
  python3 - "$MODEL_DIR" <<'PY' 2>/dev/null || echo "(index not readable yet)"
import json,sys
p=sys.argv[1]+"/model.safetensors.index.json"
wm=json.load(open(p))["weight_map"]
print(f"index tensors: {len(wm)}  mtp tensors: {sum(1 for k in wm if k.startswith('mtp.'))}")
print("qweight tensors:", sum(1 for k in wm if k.endswith('.qweight')))
PY
  echo "=== API ==="
  curl -s -m 8 "http://localhost:$HOST_PORT/v1/models" | head -c 400; echo
  LAN_IP="$(hostname -I | awk '{print $1}')"
  echo "--- LAN: http://$LAN_IP:$HOST_PORT/v1  (model: $SERVED_NAME) ---"
  echo "=== other servers ==="
  sg docker -c "docker ps --format '{{.Names}}  {{.Status}}'" | grep -E 'vllm-(xpu|exl3|mxfp4|coder390)' || echo "none"
  exit 0
fi

for other in vllm-xpu-b60 vllm-exl3-b60 vllm-mxfp4-b60; do
  if sg docker -c "docker ps --format '{{.Names}}'" | grep -qx "$other"; then
    echo "REFUSING: $other is RUNNING — one server at a time (22.71 GiB)."
    echo "Stop it first:  sg docker -c 'docker stop $other'"
    exit 1
  fi
done

if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "MISSING model: $MODEL_DIR (quantization still running? try ./quantize_coder390.sh --check)"
  exit 1
fi

CMD=(sg docker -c "docker run -d --name '$NAME' -p '$HOST_PORT:8000' \
  --restart unless-stopped \
  --device /dev/dri --group-add '$RENDER_GID' \
  -v /dev/dri:/dev/dri:ro \
  -v '$MODEL_DIR:/model:ro' \
  -v '$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro' \
  -v '$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro' \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VLLM_LOG_STATS_INTERVAL=2 \
  --entrypoint bash '$IMAGE' -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model \
     --quantization gptq --dtype float16 --max-model-len 12288 \
     --gpu-memory-utilization 0.93 --kv-cache-dtype fp8 --port 8000 \
     --max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-prefix-caching \
     --served-model-name $SERVED_NAME --language-model-only \
     --enable-auto-tool-choice --tool-call-parser qwen3_xml \
     --reasoning-parser qwen3 \
     --speculative-config \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":4}\"'")

echo "B60_RENDER_NODE=$B60_RENDER_NODE  RENDER_GID=$RENDER_GID  ZE_AFFINITY_MASK=0"
echo "model: $MODEL_DIR"
echo "host port: $HOST_PORT (container 8000), served name: $SERVED_NAME"

if [ "${1:-}" = "--dry-run" ]; then
  printf '%s\n' "${CMD[@]}"
  exit 0
fi

sg docker -c "docker rm -f '$NAME'" >/dev/null 2>&1 || true
"${CMD[@]}"

echo "started $NAME; waiting for healthy (cold start ~3-5 min, retries once)…"
for i in $(seq 1 90); do
  if curl -s -m 5 "http://localhost:$HOST_PORT/v1/models" | grep -q "$SERVED_NAME"; then
    echo "HEALTHY after ~$((i*10)) s"
    LAN_IP="$(hostname -I | awk '{print $1}')"
    echo "LAN: http://$LAN_IP:$HOST_PORT/v1  (model: $SERVED_NAME)"
    exit 0
  fi
  if ! sg docker -c "docker ps --format '{{.Names}}'" | grep -qx "$NAME"; then
    state="$(sg docker -c "docker inspect -f '{{.State.Status}}' '$NAME'" 2>/dev/null || echo missing)"
    case "$state" in
      restarting|running) : ;;  # mid first-init failure; retry is expected
      *)
        echo "container $NAME is '$state' (not restarting) — giving up."
        echo "inspect: sg docker -c 'docker logs $NAME' | tail -40"
        exit 1
        ;;
    esac
  fi
  sleep 10
done
echo "NOT healthy after 15 min — inspect: sg docker -c 'docker logs $NAME' | tail -40"
exit 1
