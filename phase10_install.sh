#!/bin/bash
# Phase 10 - installs BOTH remaining operator items in one go:
#   * vllm-b60-power-cap.service : re-applies the chosen 180 W cap (volatile sysfs state)
#   * vllm-b60-monitor.{service,timer} : 1-minute telemetry CSV
#
#     sudo bash /mnt/usb/vllm.xpu/phase10_install.sh
#
# Uses `install -m 644` rather than `cp`: /mnt/usb is exFAT, which stores no unix
# permissions and hands these files a 755 mode -- systemd warns if a unit is executable.
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run with sudo" >&2; exit 1; }

echo "installing units to /etc/systemd/system/"
install -m 644 "$DIR"/vllm-b60-power-cap.service \
               "$DIR"/vllm-b60-monitor.service \
               "$DIR"/vllm-b60-monitor.timer /etc/systemd/system/

systemctl daemon-reload

echo
echo "--- applying the 180 W cap now and at every boot ---"
systemctl enable --now vllm-b60-power-cap.service
echo "power1_cap now: $(( $(cat /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap) / 1000000 )) W"

echo
echo "--- arming the telemetry sampler (every 1 min) ---"
systemctl enable --now vllm-b60-monitor.timer

echo
echo "--- verification ---"
systemctl is-enabled vllm-b60-power-cap.service
systemctl is-enabled vllm-b60-monitor.timer
systemctl list-timers vllm-b60-monitor.timer --no-pager | head -4

echo
echo "--- waiting for the first sample ---"
sleep 5
if [ -f "$DIR/b60_monitor.csv" ]; then
  echo "b60_monitor.csv:"
  tail -2 "$DIR/b60_monitor.csv"
else
  echo "no sample yet - check: journalctl -u vllm-b60-monitor.service -n 20" >&2
fi

echo
echo "done. logs: journalctl -u vllm-b60-power-cap -u vllm-b60-monitor -n 20"
