#!/usr/bin/env bash
# Download Tiel BF16 source shards one at a time (resumable) into models/_staging_tiel/.
set -u
cd "$(dirname "$0")/.."
REPO="symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized"
for i in $(seq 1 17); do
  f=$(printf "model.safetensors-%05d-of-00017.safetensors" $i)
  if [ -f "models/_staging_tiel/$f" ]; then echo "SKIP (done): $f"; continue; fi
  echo "GET: $f"
  ./.venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$REPO', '$f', local_dir='models/_staging_tiel'))" || { echo "FAILED: $f"; exit 1; }
done
echo "ALL-DONE"
