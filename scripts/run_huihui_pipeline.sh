#!/usr/bin/env bash
# Wait for the huihui qwen3.8 source download, quantizing shard-by-shard
# on the fly: once model-00001-of-00002 lands, quantize it (--shards 1-1)
# while shard 2 + MTP keep downloading. Final pass (--shards 2-2 via the
# default 1-2 run, which SKIP-resumes) merges shard 2 + MTP + config.
# Detached-safe: logs everything; aborts loudly on unrecoverable states.
cd /home/yanchun/Projects/vllm.xpu || exit 1
LOG=logs/download-huihui-qwen38.log
QLOG=logs/quantize-huihui-qwen38.log
STAGING=models/_staging_huihui_qwen38
EARLY_MARKER=logs/.huihui-shard1-early
# shard 1 is 49,825,162,944 bytes; treat >49e9 as complete.
SHARD1_MIN=49000000000

echo "=== pipeline start $(date -Is) ===" >> "$QLOG"
for i in $(seq 1 720); do   # up to 6h waiting on download
  # on-the-fly: shard 1 complete but early quant never started -> start it
  if [ ! -f "$EARLY_MARKER" ]; then
    sz=$(stat -c %s "$STAGING/model-00001-of-00002.safetensors" 2>/dev/null || echo 0)
    if [ "$sz" -gt "$SHARD1_MIN" ]; then
      if ! pgrep -f "quantize_huihui_qwen38_mxfp4.py" >/dev/null; then
        echo "shard 1 landed (${sz}B), starting early --shards 1-1 $(date -Is)" >> "$QLOG"
        touch "$EARLY_MARKER"
        setsid nohup .venv/bin/python scripts/quantize_huihui_qwen38_mxfp4.py --shards 1-1 >> "$QLOG" 2>&1 < /dev/null &
      fi
    fi
  fi
  if grep -q "ALL-DONE" "$LOG" 2>/dev/null; then
    echo "download complete $(date -Is)" >> "$QLOG"
    break
  fi
  if grep -q "^FAILED:" "$LOG" 2>/dev/null; then
    echo "ABORT: download reported failure $(date -Is)" >> "$QLOG"
    exit 3
  fi
  if ! pgrep -f download_huihui_qwen38_bf16.sh >/dev/null; then
    # process gone without ALL-DONE/FAILED -> one relaunch (resumable), then give up
    if [ "$relaunch" = "" ]; then
      relaunch=1
      echo "download proc gone, relaunching (resumable) $(date -Is)" >> "$QLOG"
      setsid nohup bash scripts/download_huihui_qwen38_bf16.sh >> "$LOG" 2>&1 < /dev/null &
      sleep 30
      continue
    fi
    echo "ABORT: download dead twice $(date -Is)" >> "$QLOG"
    exit 4
  fi
  sleep 30
done
grep -q "ALL-DONE" "$LOG" || { echo "ABORT: wait window expired $(date -Is)" >> "$QLOG"; exit 5; }

# never race the early job on shared OUTDIR/state: wait for it first.
while pgrep -f "quantize_huihui_qwen38_mxfp4.py" >/dev/null; do
  echo "waiting on early shard-1 job $(date -Is)" >> "$QLOG"
  sleep 60
done

# sanity: all three payload files present at expected sizes
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors mtp-bf16.safetensors; do
  sz=$(stat -c %s "$STAGING/$f" 2>/dev/null || echo 0)
  [ "$sz" -gt 100000000 ] || { echo "ABORT: missing/undersized $f ($sz bytes)" >> "$QLOG"; exit 6; }
done

echo "=== quantizer final pass $(date -Is) ===" >> "$QLOG"
.venv/bin/python scripts/quantize_huihui_qwen38_mxfp4.py >> "$QLOG" 2>&1
code=$?
echo "EXIT:$code $(date -Is)" >> "$QLOG"
exit "$code"
