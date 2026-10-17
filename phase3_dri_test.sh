#!/bin/bash
# Phase 3, last box: "Can the container enumerate /dev/dri devices?"
# Go/no-go for Phase 5 (vLLM launch).
# Note: --device /dev/kfd from tasks.md is omitted - this Intel host has no
# /dev/kfd node (that is AMD/ROCm), and passing a missing node aborts the run.
# docker.io ships no udev rules, so devices are passed explicitly with --device
# (never --device-cgroup-rule), per cookbook 6b.
set -u

REF="vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"

echo "=== image under test ==="
echo "$REF"
echo
echo "=== /dev/dri INSIDE container ==="

docker run --rm --device /dev/dri --entrypoint bash "$REF" -c '
    echo "--- /dev/dri listing ---"
    ls -l /dev/dri
    echo
    echo "--- render node RW test ---"
    for n in /dev/dri/renderD128 /dev/dri/renderD129; do
      if [ -e "$n" ]; then
        if [ -r "$n" ] && [ -w "$n" ]; then
          echo "  $n: RW OK (mode $(stat -c %a "$n"))"
        else
          echo "  $n: present but NOT RW (mode $(stat -c %a "$n"))"
        fi
      else
        echo "  $n: ABSENT"
      fi
    done
    echo
    echo "--- card nodes ---"
    for n in /dev/dri/card0 /dev/dri/card1; do
      if [ -e "$n" ]; then echo "  $n: present (mode $(stat -c %a "$n"))"; else echo "  $n: ABSENT"; fi
    done
    echo
    if [ -e /dev/kfd ]; then echo "/dev/kfd: present"; else echo "/dev/kfd: ABSENT"; fi
    echo
    echo "--- Level Zero loader in image? ---"
    ldconfig -p 2>/dev/null | grep -E "ze_loader|ze_intel_gpu" || echo "  no ldconfig match"
    echo
    echo "--- kernel driver packages in image? ---"
    (dpkg -l 2>/dev/null | grep -E "intel-level-zero|intel-opencl|level-zero" | awk "{print \$1, \$2, \$3}") || true
    echo
    echo "--- intel_gpu_top available? ---"
    command -v intel_gpu_top || echo "  (not installed in image)"
'
rc=$?
echo
echo "container exit=$rc"
exit $rc
