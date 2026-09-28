#!/usr/bin/env bash
# Download KAT-Coder-V2.5-Dev BF16 source shards one at a time (resumable).
set -u
cd "$(dirname "$0")/.."
REPO="Kwaipilot/KAT-Coder-V2.5-Dev"
N=${N:-13}
for i in $(seq 0 $((N - 1))); do
  f=$(printf "model-%05d-of-%05d.safetensors" $i $N)
  if [ -f "models/_staging_kat/$f" ]; then echo "SKIP (done): $f"; continue; fi
  echo "GET: $f"
  ./.venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$REPO', '$f', local_dir='models/_staging_kat'))" || { echo "FAILED: $f"; exit 1; }
done
echo "ALL-DONE"
