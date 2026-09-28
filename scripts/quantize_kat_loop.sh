#!/usr/bin/env bash
# Watch-quantize-delete loop for KAT MXFP4 conversion.
# Quantizes each source shard soon after it lands, then deletes the source.
# Safe to re-run (resume via quantize_state.json + SKIP).
set -u
cd "$(dirname "$0")/.."
N=13
for i in $(seq 0 $((N - 1))); do
  f=$(printf "model-%05d-of-%05d.safetensors" $i $N)
  for w in $(seq 1 180); do
    if [ -f "models/_staging_kat/$f" ]; then
      nxt=$(printf "model-%05d-of-%05d.safetensors" $((i+1)) $N)
      if [ $i -eq $((N - 1)) ] || grep -q "GET: $nxt" logs/download-kat.log 2>/dev/null || grep -q "ALL-DONE" logs/download-kat.log 2>/dev/null; then
        break
      fi
    fi
    sleep 60
  done
  if [ ! -f "models/_staging_kat/$f" ]; then echo "GIVE-UP waiting for $f"; exit 1; fi
  echo "QUANT: $f"
  ./.venv/bin/python scripts/quantize_kat_mxfp4.py --shards $i-$i || exit 1
  rm "models/_staging_kat/$f" && echo "DELETED source $f"
done
echo "LOOP-DONE"
