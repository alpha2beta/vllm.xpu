#!/usr/bin/env bash
# Watch-quantize-delete loop for Tiel MXFP4 conversion.
# Waits for each source shard to land, quantizes it, deletes the source.
set -u
cd "$(dirname "$0")/.."
for i in $(seq 3 17); do
  f=$(printf "model.safetensors-%05d-of-00017.safetensors" $i)
  for w in $(seq 1 120); do
    # complete when download script moved past it (next GET appears) or file stable
    if [ -f "models/_staging_tiel/$f" ]; then
      # wait until the downloader starts the NEXT shard (current file complete)
      nxt=$(printf "model.safetensors-%05d-of-00017.safetensors" $((i+1)))
      if [ $i -eq 17 ] || grep -q "GET: $nxt" logs/download-tiel.log 2>/dev/null || grep -q "ALL-DONE" logs/download-tiel.log 2>/dev/null; then
        break
      fi
    fi
    sleep 60
  done
  if [ ! -f "models/_staging_tiel/$f" ]; then echo "GIVE-UP waiting for $f"; exit 1; fi
  echo "QUANT: $f"
  ./.venv/bin/python scripts/quantize_tiel_mxfp4.py --shards $i-$i || exit 1
  rm "models/_staging_tiel/$f" && echo "DELETED source $f"
done
echo "LOOP-DONE"
