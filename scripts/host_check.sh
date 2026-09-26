#!/usr/bin/env bash
# Repeatable host / GPU environment capture for the vLLM XPU experiment.
# Usage: scripts/host_check.sh [output-file]   (default: logs/host-check.txt)
set -uo pipefail

OUT=${1:-logs/host-check.txt}
mkdir -p "$(dirname "$OUT")"

section() { printf '\n=== %s (%s) ===\n' "$1" "$(date -Is)"; }

{
  section "host basics"
  uname -a
  cat /etc/os-release
  echo
  lscpu
  echo
  free -h
  echo
  lsblk
  echo
  df -h / /home 2>/dev/null

  section "GPU PCI + device nodes"
  lspci -nn | grep -Ei 'vga|display|3d'
  ls -l /dev/dri
  id

  section "kernel driver / messages"
  for m in xe i915; do
    if [ -d "/sys/module/$m" ]; then
      echo "module $m loaded, version: $(cat "/sys/module/$m/version" 2>/dev/null || echo n/a)"
    fi
  done
  dmesg 2>/dev/null | grep -Ei '\bxe\b|i915|drm:' | tail -20 || echo "(dmesg not readable without privileges)"

  section "oneAPI / runtime packages (pacman)"
  pacman -Q 2>/dev/null | grep -Ei 'level-zero|intel-|ocl-icd|oneapi' || true

  section "GPU discovery: sycl-ls"
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
  sycl-ls 2>&1 | sed 's/^/  /'

  section "GPU discovery: clinfo"
  if command -v clinfo >/dev/null; then
    clinfo 2>&1 | grep -Ei 'platform name|device name|Device Version|Max compute units|Global memory size'
  else
    echo "clinfo not installed"
  fi

  section "GPU monitoring tool"
  if command -v intel_gpu_top >/dev/null; then
    echo "intel_gpu_top available: $(command -v intel_gpu_top)"
    timeout 10 intel_gpu_top -J -s 1 2>&1 | head -20
  else
    echo "intel_gpu_top NOT installed (pacman package: intel-gpu-tools)"
  fi

  section "memory / swap snapshot"
  free -h
  swapon --show 2>/dev/null || cat /proc/swaps
  vmstat 1 3 2>/dev/null || true
} >>"$OUT" 2>&1

echo "wrote $OUT"
