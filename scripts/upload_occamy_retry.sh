#!/usr/bin/env bash
# Detached resilient uploader for the Occamy MXFP4 checkpoint.
cd /home/yanchun/Projects/vllm.xpu || exit 1
for attempt in $(seq 1 10); do
  echo "=== attempt $attempt $(date -Is) ===" >> logs/upload-occamy-mxfp4.log
  hf upload "alpha2beta/Qwen3.6-35B-A3B-occamy-1.0-MXFP4" \
    "models/Qwen3.6-35B-A3B-occamy-1.0-MXFP4" \
    --exclude "*.bak" --exclude "quantize_state.json" >> logs/upload-occamy-mxfp4.log 2>&1
  code=$?
  echo "EXIT:$code $(date -Is)" >> logs/upload-occamy-mxfp4.log
  [ "$code" -eq 0 ] && break
  sleep 120
done
