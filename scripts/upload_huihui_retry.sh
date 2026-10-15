#!/usr/bin/env bash
# Detached resilient uploader for the Huihui-Qwen3.8 MXFP4 checkpoint (USB disk).
cd /home/yanchun/Projects/vllm.xpu || exit 1
OUT="/run/media/yanchun/CCCOMA_X64FRE_EN-GB_DV9/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4"
for attempt in $(seq 1 10); do
  echo "=== attempt $attempt $(date -Is) ===" >> logs/upload-huihui-mxfp4.log
  hf upload "alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4" \
    "$OUT" \
    --exclude "quantize_state.json" >> logs/upload-huihui-mxfp4.log 2>&1
  code=$?
  echo "EXIT:$code $(date -Is)" >> logs/upload-huihui-mxfp4.log
  [ "$code" -eq 0 ] && break
  sleep 120
done
