#!/usr/bin/env bash
# Detached resilient uploader for KAT-Coder-V2.5-Dev-W4A16-GPTQ checkpoint to Hugging Face.
# Safe: hf re-checks/skips completed files each attempt.
set -u

REPO_ID="alpha2beta/KAT-Coder-V2.5-Dev-W4A16-GPTQ"
LOCAL_DIR="models/KAT-Coder-V2.5-Dev-W4A16-GPTQ"
LOG_FILE="logs/upload-kat-gptq.log"

cd /home/yanchun/Projects/vllm.xpu || exit 1
mkdir -p logs

echo "=== Upload initiated for ${REPO_ID} at $(date -Is) ===" | tee -a "${LOG_FILE}"

for attempt in $(seq 1 15); do
  echo "=== attempt $attempt $(date -Is) ===" >> "${LOG_FILE}"
  hf upload "${REPO_ID}" \
    "${LOCAL_DIR}" \
    --exclude "*.bak" \
    --exclude "hidden-after.pt" \
    --exclude "quantize_state.json" >> "${LOG_FILE}" 2>&1
  code=$?
  echo "EXIT:$code $(date -Is)" >> "${LOG_FILE}"
  if [ "$code" -eq 0 ]; then
    echo "=== Upload completed successfully at $(date -Is) ===" | tee -a "${LOG_FILE}"
    break
  fi
  echo "Attempt $attempt failed with code $code. Retrying in 30s..." | tee -a "${LOG_FILE}"
  sleep 30
done
