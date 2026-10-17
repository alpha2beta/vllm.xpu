#!/usr/bin/env bash
# quantize_coder390.sh - GPTQ INT4 (sym, g128) quantization of the Huihui
# Qwen3.8-27B Coder390 EfficientThink (abliterated) BF16 model, mirroring the
# recipe of Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16.
#
# Runs inside vllm-quantizer:gptqmodel7.6.0 (gptqmodel 7.6.0 on vLLM 0.30.0+xpu;
# torch 2.13.0+xpu, transformers 5.16.1) with the Arc Pro B60 as accelerator.
# The 52 GB BF16 source is streamed layer-by-layer (meta shell + LazyTurtle), so
# it is quantizable on this 45 GB-RAM host.
#
# Usage:
#   ./quantize_coder390.sh                 # run (detached-ish, foreground container)
#   ./quantize_coder390.sh --check         # no run: verify inputs/handles
#   ./quantize_coder390.sh --dry-run       # print the docker command only
#   ./quantize_coder390.sh --calib-samples 16 --calib-length 512   # quick trial
#
# NOTE: the B60 and host RAM are needed in full (52 GB streaming + 1.2 GB
# Hessians): this refuses to start while any vLLM server container is running.
#
# Only the B60 render/card nodes are exposed to the container (NOT all of
# /dev/dri). The first run exposed both GPUs, L0 then probed the B580 for P2P,
# the B580 GuC timed out and the B60 ccs engine was reset -> DEVICE_LOST mid-run.
# Isolation keeps L0/level-zero on the B60 only.
#
# Accelerator is overridable (defaults keep the B60 command of record):
#   QDEV_RENDER_NODE=/dev/dri/renderD129 QDEV_CARD=/dev/dri/card1 \
#   QDEV_ZE_MASK=1 QDEV_DEVICE=xpu:1 ./quantize_coder390.sh ...
set -u
IMG='vllm-quantizer:gptqmodel7.6.0'
SRC_HOST='/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-bf16'
OUT_HOST='/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-GPTQ-Int4-sym-G128-MTP-BF16'
SCRIPTS_HOST='/mnt/usb/vllm.xpu'
HFCACHE_HOST='/mnt/usb/models/.hf-cache'
B60_RENDER_NODE="${QDEV_RENDER_NODE:-/dev/dri/renderD128}"
CARD_NODE="${QDEV_CARD:-/dev/dri/card0}"
ZE_MASK="${QDEV_ZE_MASK:-0}"
TORCH_DEVICE="${QDEV_DEVICE:-xpu:0}"
# PYTORCH_ALLOC_CONF override. gptqmodel sets
#   expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7
# which grows VMM 'reserved' far past physical VRAM (40 GiB on a 24 GiB card)
# and fills the device. Suspected cause of the stochastic UR_RESULT_ERROR_DEVICE_LOST
# engine timeouts, so allow disabling expandable_segments:
#   QDEV_ALLOC_CONF='expandable_segments:False,max_split_size_mb:256' ./quantize_coder390.sh
ALLOC_CONF="${QDEV_ALLOC_CONF:-}"
ALLOC_ENV=""
[ -n "$ALLOC_CONF" ] && ALLOC_ENV="-e PYTORCH_ALLOC_CONF=$ALLOC_CONF"
RENDER_GID="$(stat -c '%g' "$B60_RENDER_NODE")"
NAME='vllm-quant-coder390'
DRY=0
CHECK=0
PASSTHRU=()
for a in "$@"; do
  case "$a" in
    --check)   CHECK=1 ;;
    --dry-run) DRY=1 ;;
    *)         PASSTHRU+=("$a") ;;
  esac
done

if [ "$CHECK" = 1 ]; then
    sg docker -c "docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -F '$IMG'" || echo "IMAGE MISSING"
    echo "=== source ==="
    du -sh "$SRC_HOST" 2>/dev/null || echo "SOURCE MISSING: $SRC_HOST"
    echo "=== target ==="
    { [ -e "$OUT_HOST" ] && du -sh "$OUT_HOST"; } || echo "not present (will be created): $OUT_HOST"
    echo "=== running servers ==="
    sg docker -c "docker ps --format '{{.Names}}  {{.Status}}'" | grep -E 'vllm-(xpu|exl3|mxfp4)' || echo "none"
    echo "=== existing quant container ==="
    sg docker -c "docker ps -a --format '{{.Names}}  {{.Status}}'" | grep -F "$NAME" || echo "none"
    echo "=== LAN/host ==="
    echo "RENDER_GID=$RENDER_GID  ZE_AFFINITY_MASK=$ZE_MASK (accel = $B60_RENDER_NODE, device=$TORCH_DEVICE)"
    exit 0
fi

for other in vllm-xpu-b60 vllm-exl3-b60 vllm-mxfp4-b60; do
  if sg docker -c "docker ps --format '{{.Names}}'" | grep -qx "$other"; then
    echo "REFUSING: $other is RUNNING — stop it first (GPU + RAM are needed in full):"
    echo "  sg docker -c 'docker stop $other'"
    exit 1
  fi
done

if [ ! -d "$SRC_HOST" ]; then
  echo "MISSING source: $SRC_HOST"; exit 1
fi
mkdir -p "$OUT_HOST" "$HFCACHE_HOST"

CMD=(sg docker -c "docker run --rm --name '$NAME' \
  --device '$B60_RENDER_NODE' --device '$CARD_NODE' --group-add '$RENDER_GID' \
  --shm-size=16g \
  -v '$SRC_HOST:/work/src:ro' \
  -v '$(dirname "$OUT_HOST"):/work/models' \
  -v '$SCRIPTS_HOST:/work/scripts:ro' \
  -v '$HFCACHE_HOST:/root/.cache/huggingface' \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_AFFINITY_MASK='$ZE_MASK' \
  $ALLOC_ENV \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 '$IMG' /work/scripts/quantize_coder390_gptq.py \
  --src /work/src --out /work/models/$(basename "$OUT_HOST") \
  --device '$TORCH_DEVICE' ${PASSTHRU[*]:-}")

echo "accel node=$B60_RENDER_NODE  card=$CARD_NODE  RENDER_GID=$RENDER_GID  ZE_AFFINITY_MASK=$ZE_MASK  device=$TORCH_DEVICE"
echo "src:  $SRC_HOST"
echo "out:  $OUT_HOST"

if [ "$DRY" = 1 ]; then
  printf '%s\n' "${CMD[@]}"
  exit 0
fi

sg docker -c "docker rm -f '$NAME'" >/dev/null 2>&1 || true
echo "--- starting quantization (streaming logs below) ---"
exec "${CMD[@]}"
