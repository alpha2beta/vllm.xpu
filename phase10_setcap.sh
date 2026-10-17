#!/bin/bash
# Re-apply the Arc Pro B60 power cap. power1_cap is volatile sysfs state and does NOT
# survive a reboot, so this exists to make the Phase-9-measured 180 W configuration
# persistent. Needs root.
#
#   sudo phase10_setcap.sh 180
#
# Resolves the hwmon directory from the B60's PCI address rather than a hardcoded
# hwmonN, because hwmon numbering is not stable across boots.
#
# POLLING: at boot the xe driver registers hwmon *after* systemd has already
# started this unit. Observed 2026-09-27: unit ran 10:55:33, hwmon8 appeared at
# 10:55:34.328 -- 1.3 s too late -- so the cap silently stayed at the 220 W
# firmware default for the whole session. Poll instead of failing on the first
# miss, and fail loudly if it never appears.
set -eu
WATTS="${1:?usage: phase10_setcap.sh <watts>}"
PCI="${B60_PCI:-0000:03:00.0}"

f=""
waited=0
for _ in $(seq 1 90); do
  for h in /sys/bus/pci/devices/$PCI/hwmon/hwmon*; do
    [ -e "$h/power1_cap" ] && { f="$h/power1_cap"; break 2; }
  done
  [ "$waited" -eq 0 ] && { echo "waiting for the B60 hwmon to appear (normal at boot)..."; waited=1; }
  sleep 1
done
if [ -z "$f" ]; then
  echo "no hwmon with power1_cap under /sys/bus/pci/devices/$PCI (waited 90s)" >&2
  echo "pass the B60's PCI address in B60_PCI if the slot changed" >&2
  exit 1
fi

want=$((WATTS * 1000000))
echo "$want" > "$f"
got=$(cat "$f")
echo "$f -> $((got / 1000000)) W (target ${WATTS} W)"
[ "$got" = "$want" ] || { echo "WARNING: driver clamped the request" >&2; exit 1; }
