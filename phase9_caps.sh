#!/bin/bash
# Phase 9 - power-cap sweep: benchmark + power/thermal sampling at 180 W and 150 W,
# then restore the original cap. Run with ONE sudo invocation:
#
#     sudo bash /mnt/usb/vllm.xpu/phase9_caps.sh
#
# The original cap is restored on exit (EXIT/INT/TERM trap), so a Ctrl-C is safe.
# Needs root only for the power1_cap writes (mode 664 root:root).
set -u

HW=/sys/class/drm/card0/device/hwmon/hwmon8
DIR="$(cd "$(dirname "$0")" && pwd)"
RUNAS="${SUDO_USER:-yanchun}"

[ -w "$HW/power1_cap" ] || { echo "ERROR: $HW/power1_cap not writable - run with sudo"; exit 1; }
ORIG="$(cat "$HW/power1_cap")"
restore() { echo "$ORIG" > "$HW/power1_cap"; echo "[done] cap restored -> $(( $(cat "$HW/power1_cap") / 1000000 )) W"; }
trap restore EXIT INT TERM

echo "original cap: $((ORIG / 1000000)) W"
echo "runner user : $RUNAS"
curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null \
  && echo "server      : healthy" \
  || { echo "ERROR: vLLM not healthy at :8000"; exit 1; }
echo

for w in 180 150; do
  echo "$((w * 1000000))" > "$HW/power1_cap"
  readback="$(cat "$HW/power1_cap")"
  echo "=========================================================="
  echo " target ${w} W   readback $((readback / 1000000)) W"
  [ "$readback" = "$((w * 1000000))" ] || echo " !! readback differs from target - driver may have clamped"
  echo "=========================================================="
  # power1_cap_interval is 15 s: let the limit actually take effect before measuring
  sleep 16

  runuser -u "$RUNAS" -- python3 "$DIR/phase9_sampler.py" --hwmon "$HW" \
      --out "$DIR/phase9_${w}w.csv" --duration 45 &
  SAMPLER=$!
  sleep 2
  runuser -u "$RUNAS" -- python3 "$DIR/bench_p512_g128.py" \
      --base-url http://127.0.0.1:8000 --model qwen38 \
      --out "$DIR/bench_${w}w_therm.json"
  wait $SAMPLER
  echo
done

echo "=== all cap points ==="
python3 - "$DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
order = {"bench_power_default.json": 220, "bench_220w_a.json": 220, "bench_220w_b.json": 220,
         "bench_180w.json": 180, "bench_180w_therm.json": 180,
         "bench_150w.json": 150, "bench_150w_therm.json": 150}
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    b = os.path.basename(f)
    if b not in order:
        continue
    j = json.load(open(f))
    if not isinstance(j, dict) or "decode_tok_s" not in j:
        continue
    rows.append((order[b], b, j["decode_tok_s"]["median"], j["decode_tok_s"]["mean"]))
rows.sort()
print(f"{'cap':>5} {'file':26} {'median':>9} {'mean':>9}")
for cap, b, m, mn in rows:
    print(f"{cap:>4}W {b:26} {m:9.3f} {mn:9.3f}")
PY
