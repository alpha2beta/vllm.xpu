#!/usr/bin/env python3
"""Probe whether the Xe driver's GPUReclaim pool is actually reclaimable.

After a crashed vLLM run, /proc/meminfo showed GPUReclaim ~12.8 GiB while
Level-Zero reported only 11.01/28.58 GiB free, which blocks startup (vLLM
requires free >= gpu_memory_utilization * total). This allocates XPU memory in
steps and reports free memory after each step, then after freeing.

    source .venv/bin/activate
    python scripts/probe_xpu_reclaim.py
"""
import sys

import torch

GB = 1024**3


def free_total() -> tuple[int, int]:
    info = torch.xpu.mem_get_info()
    return int(info[0]), int(info[1])


def gpu_meminfo() -> dict:
    out = {}
    for line in open("/proc/meminfo"):
        if line.startswith(("GPUActive", "GPUReclaim")):
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


def main() -> int:
    if not torch.xpu.is_available():
        print("no xpu"); return 1
    f, t = free_total()
    gm = gpu_meminfo()
    print(f"start: xpu free {f/GB:.2f} / {t/GB:.2f} GiB ; "
          f"GPUActive {gm.get('GPUActive',0)/GB:.2f} GiB "
          f"GPUReclaim {gm.get('GPUReclaim',0)/GB:.2f} GiB")

    held = []
    allocated = 0
    step = 1 * GB
    target = int(6 * GB)
    while allocated < target:
        try:
            held.append(torch.empty(step, dtype=torch.uint8, device="xpu"))
            torch.xpu.synchronize()
            allocated += step
        except Exception as e:
            print(f"allocation failed at {allocated/GB:.1f} GiB: {type(e).__name__}: {e}")
            break
        if len(held) % 2 == 0:
            f, _ = free_total()
            gm = gpu_meminfo()
            print(f"  allocated {allocated/GB:5.1f} GiB -> xpu free {f/GB:5.2f} GiB, "
                  f"GPUReclaim {gm.get('GPUReclaim',0)/GB:5.2f} GiB")

    print(f"peak allocated {allocated/GB:.2f} GiB")
    del held
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    f2, _ = free_total()
    gm = gpu_meminfo()
    print(f"after free: xpu free {f2/GB:.2f} GiB ; GPUReclaim {gm.get('GPUReclaim',0)/GB:.2f} GiB")
    print("RECLAIMABLE" if f2 > f else "NOT reclaimed (free did not recover)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
