#!/usr/bin/env bash
# Samples host memory / swap while a workload runs, so that "swap activity"
# (tasks.md Phase 8) is measurable evidence rather than a guess.
#
# Usage: scripts/mem_monitor.sh <label> [interval_s]   (Ctrl-C to stop)
# Writes logs/mem-<label>.csv with columns:
#   epoch,mem_total_kb,mem_available_kb,swap_total_kb,swap_used_kb,si,so,vmalloc_used_kb
set -uo pipefail

LABEL=${1:-run}
INTERVAL=${2:-2}
OUT="logs/mem-${LABEL}.csv"
mkdir -p logs

echo "epoch,mem_total_kb,mem_available_kb,swap_total_kb,swap_used_kb,si,so" >"$OUT"
echo "sampling every ${INTERVAL}s -> $OUT (Ctrl-C to stop)"

prev_si=0
prev_so=0
while true; do
  read -r mt ma st su <<<"$(awk '/^MemTotal:|^MemAvailable:|^SwapTotal:|^SwapUsed:/ {printf "%s ", $2}' /proc/meminfo 2>/dev/null)"
  # /proc/meminfo has no SwapUsed; derive it.
  st=$(awk '/^SwapTotal:/{print $2}' /proc/meminfo)
  su=$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print t-f}' /proc/meminfo)
  mt=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
  ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  # cumulative pswpin/pswpout (pages) from /proc/vmstat
  si=$(awk '/^pswpin /{print $2}' /proc/vmstat)
  so=$(awk '/^pswpout /{print $2}' /proc/vmstat)
  echo "$(date +%s),$mt,$ma,$st,$su,$si,$so" >>"$OUT"
  sleep "$INTERVAL"
done
