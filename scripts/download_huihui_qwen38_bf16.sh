#!/usr/bin/env bash
# Download Huihui-Qwen3.8-27B-Coder390 BF16 source shards one at a time (resumable).
set -u
cd "$(dirname "$0")/.."
REPO="huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated"
STAGING="models/_staging_huihui_qwen38"
mkdir -p "$STAGING"
for f in model.safetensors.index.json \
         config.json generation_config.json chat_template.jinja \
         tokenizer.json tokenizer_config.json \
         preprocessor_config.json \
         video_preprocessor_config.json \
         model-00001-of-00002.safetensors \
         model-00002-of-00002.safetensors \
         mtp-bf16.safetensors; do
  if [ -f "$STAGING/$f" ]; then echo "SKIP (done): $f"; continue; fi
  echo "GET: $f"
  ./.venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$REPO', '$f', local_dir='$STAGING'))" || { echo "FAILED: $f"; exit 1; }
done
echo "ALL-DONE"
