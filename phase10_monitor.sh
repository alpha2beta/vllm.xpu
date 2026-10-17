#!/bin/bash
# Phase 10 monitoring: append one row of B60 telemetry to a CSV.
#
# Runs fully unprivileged -- the xe hwmon directory is world-readable and vLLM's
# /metrics endpoint is on localhost -- so the systemd timer needs no sudo.
#
#   phase10_monitor.sh                 # one sample, append, exit
#   phase10_monitor.sh --interval 30   # loop, sampling every 30 s
#   MON_OUT=/path.csv phase10_monitor.sh
#
# Throttle state: xe exposes no throttle attribute, so it is derived as
# at_cap_pct = 100 * card_W / cap_W (sustained ~100% == power-limited).
set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${MON_OUT:-$DIR/b60_monitor.csv}"
STATE="$DIR/.b60_monitor.state"
PCI="${B60_PCI:-0000:03:00.0}"
URL="${METRICS_URL:-http://127.0.0.1:8000/metrics}"
INTERVAL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --once)     INTERVAL=0; shift ;;
    -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HDR="ts_iso,card_W,pkg_W,temp_pkg_C,temp_vram_C,temp_mctrl_C,temp_pcie_C,fan_rpm,cap_W,at_cap_pct,cache_pct,running,waiting,vllm_ok"

# hwmon from the B60's PCI address; falls back to card0's hwmon dir.
find_hwmon() {
  local h
  for h in /sys/bus/pci/devices/$PCI/hwmon/hwmon*; do
    [ -e "$h/energy1_input" ] && { echo "$h"; return 0; }
  done
  for h in /sys/class/drm/card0/device/hwmon/hwmon*; do
    [ -e "$h/energy1_input" ] && { echo "$h"; return 0; }
  done
  return 1
}

rd() { if [ -r "$1" ]; then cat "$1" 2>/dev/null; fi; }

# mC -> "59.0" ("" if the sensor is absent)
deg() { local v; v=$(rd "$1"); [ -n "$v" ] && awk -v v="$v" 'BEGIN{printf "%.1f", v/1000}'; }

sample_once() {
  local h now e1 e2 cap card pkg hw
  h=$(find_hwmon) || { echo "monitor: no xe hwmon found" >&2; return 1; }
  now=$(date +%s)
  e1=$(rd "$h/energy1_input"); e2=$(rd "$h/energy2_input")
  cap=$(rd "$h/power1_cap")

  # watts from the energy counter delta since the previous sample
  card=""; pkg=""
  if [ -r "$STATE" ]; then
    local p t0 x1 x2
    read -r p t0 x1 x2 < "$STATE" 2>/dev/null || true
    if [ -n "${t0:-}" ] && [ -n "${x1:-}" ] && [ -n "${x2:-}" ] && [ "$now" -gt "$t0" ] && [ "$e1" -ge "$x1" ] && [ "$e2" -ge "$x2" ] 2>/dev/null; then
      card=$(awk -v d="$((e1 - x1))" -v dt="$((now - t0))" 'BEGIN{if(dt>0) printf "%.1f", d/1e6/dt}')
      pkg=$(awk  -v d="$((e2 - x2))" -v dt="$((now - t0))" 'BEGIN{if(dt>0) printf "%.1f", d/1e6/dt}')
    fi
  fi
  printf '%s %s %s %s\n' "$h" "$now" "$e1" "$e2" > "$STATE"

  local fan t2 t3 t4 t5
  fan=$(rd "$h/fan1_input")
  t2=$(deg "$h/temp2_input")   # pkg
  t3=$(deg "$h/temp3_input")   # vram
  t4=$(deg "$h/temp4_input")   # mctrl
  t5=$(deg "$h/temp5_input")   # pcie

  local capw="" atp=""
  if [ -n "$cap" ]; then
    capw=$(awk -v v="$cap" 'BEGIN{printf "%d", v/1e6}')
    [ -n "$card" ] && [ "$card" != "0.0" ] && atp=$(awk -v c="$card" -v m="$capw" 'BEGIN{printf "%.0f", 100*c/m}')
  fi

  # vLLM: /metrics doubles as the health probe
  local m health=0 cache="" run="" wait=""
  m=$(curl -fsS --max-time 3 "$URL" 2>/dev/null) && [ -n "$m" ] && health=1
  if [ "$health" = 1 ]; then
    cache=$(printf '%s\n' "$m" | awk '/^vllm:kv_cache_usage_perc/{print $2; exit}')
    run=$(printf '%s\n'   "$m" | awk '/^vllm:num_requests_running/{print $2; exit}')
    wait=$(printf '%s\n'  "$m" | awk '/^vllm:num_requests_waiting/{print $2; exit}')
    cache=$(printf '%s' "$cache" | awk '{printf "%.3f",$1}')
    run=$(printf '%s' "$run"   | awk '{printf "%.0f",$1}')
    wait=$(printf '%s' "$wait"  | awk '{printf "%.0f",$1}')
  fi

  [ -f "$OUT" ] || printf '%s\n' "$HDR" > "$OUT"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${card:-}" "${pkg:-}" "${t2:-}" "${t3:-}" \
    "${t4:-}" "${t5:-}" "${fan:-}" "${capw:-}" "${atp:-}" "${cache:-}" \
    "${run:-}" "${wait:-}" "$health" >> "$OUT"

  # keep the file bounded (~1.3 MB/month at 1/min)
  if [ "$(stat -c %s "$OUT" 2>/dev/null || echo 0)" -gt 20971520 ]; then
    mv -f "$OUT" "$OUT.1"; printf '%s\n' "$HDR" > "$OUT"
  fi
}

if [ "$INTERVAL" -gt 0 ]; then
  echo "sampling every ${INTERVAL}s -> $OUT  (Ctrl-C to stop)"
  trap 'echo; echo "stopped"; exit 0' INT TERM
  while :; do sample_once || true; sleep "$INTERVAL"; done
else
  sample_once
fi
