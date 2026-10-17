#!/bin/bash
# Phase 9 box 4 - "final" recorded baseline at the CHOSEN cap (180 W).
# One sudo invocation; original cap restored on any exit.
#
#     sudo bash /mnt/usb/vllm.xpu/phase9_final.sh
set -u
HW=/sys/class/drm/card0/device/hwmon/hwmon8
DIR="$(cd "$(dirname "$0")" && pwd)"
RUNAS="${SUDO_USER:-yanchun}"
CHOSEN=180

[ -w "$HW/power1_cap" ] || { echo "ERROR: run with sudo"; exit 1; }
ORIG="$(cat "$HW/power1_cap")"
restore(){ echo "$ORIG" > "$HW/power1_cap"; echo "[done] cap restored -> $(( $(cat "$HW/power1_cap") / 1000000 )) W"; }
trap restore EXIT INT TERM

curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null || { echo "ERROR: server unhealthy"; exit 1; }

echo "$((CHOSEN * 1000000))" > "$HW/power1_cap"
echo "cap -> $(cat "$HW/power1_cap" | awk '{print $1/1000000}') W (target ${CHOSEN} W)"
echo "settling 16s (power1_cap_interval = 15s) ..."
sleep 16

runuser -u "$RUNAS" -- python3 "$DIR/phase9_sampler.py" --hwmon "$HW" \
    --out "$DIR/phase9_final_${CHOSEN}w.csv" --duration 45 &
SP=$!
sleep 2
runuser -u "$RUNAS" -- python3 "$DIR/bench_p512_g128.py" \
    --base-url http://127.0.0.1:8000 --model qwen38 \
    --out "$DIR/bench_final_${CHOSEN}w.json"
wait $SP
