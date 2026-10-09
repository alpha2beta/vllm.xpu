#!/usr/bin/env bash
# Detached resilient uploader: retries `hf upload` until it exits 0.
# Safe: hf re-checks/skips completed files each attempt.
cd /home/yanchun/Projects/vllm.xpu || exit 1
for attempt in $(seq 1 10); do
  echo "=== attempt $attempt $(date -Is) ===" >> logs/upload-kat-mxfp4.log
  hf upload "alpha2beta/KAT-Coder-V2.5-Dev-MXFP4" \
    "models/KAT-Coder-V2.5-Dev-MXFP4" \
    --exclude "*.bak" --exclude "quantize_state.json" >> logs/upload-kat-mxfp4.log 2>&1
  code=$?
  echo "EXIT:$code $(date -Is)" >> logs/upload-kat-mxfp4.log
  [ "$code" -eq 0 ] && break
  sleep 120
done
