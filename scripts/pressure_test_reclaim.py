#!/usr/bin/env python3
"""Does system memory pressure make the Xe driver give GPUReclaim back?

Background: after a crashed vLLM run, /proc/meminfo held GPUReclaim ~12.8 GiB
while Level-Zero reported only ~11 GiB free, so vLLM's startup check
(`free >= gpu_memory_utilization * total`) failed even though the memory is
demonstrably allocatable (scripts/probe_xpu_reclaim.py consumed it).

This allocates + touches anonymous host memory in 1 GiB steps until MemFree is
close to the low watermark, recording MemFree/GPUReclaim/MemAvailable at each
step, then frees everything and records the end state.

    source .venv/bin/activate
    python scripts/pressure_test_reclaim.py [min_free_mib]
"""
from __future__ import annotations

import gc
import sys
import time

STOP_FREE_MIB = int(sys.argv[1]) if len(sys.argv) > 1 else 1536
CHUNK = 1 << 30  # 1 GiB


def mem() -> dict:
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0])
    return d


def report(tag: str, d: dict) -> None:
    print(
        f"{tag:22s} MemFree {d['MemFree']/1048576:6.2f}  "
        f"MemAvailable {d['MemAvailable']/1048576:6.2f}  "
        f"GPUActive {d.get('GPUActive',0)/1048576:6.2f}  "
        f"GPUReclaim {d.get('GPUReclaim',0)/1048576:6.2f}  (GiB)",
        flush=True,
    )


def main() -> int:
    report("start", mem())
    held: list[bytearray] = []
    allocated = 0
    while True:
        d = mem()
        if d["MemFree"] < STOP_FREE_MIB * 1024:
            print(f"stopping: MemFree {d['MemFree']/1048576:.2f} GiB < "
                  f"{STOP_FREE_MIB} MiB threshold")
            break
        if allocated >= 14 * CHUNK:
            print("stopping: 14 GiB cap reached")
            break
        b = bytearray(CHUNK)
        # touch one byte per page so every page is actually allocated
        b[::4096] = b"\x01" * (CHUNK // 4096)
        held.append(b)
        allocated += CHUNK
        report(f"held {allocated//CHUNK} GiB", mem())
        time.sleep(0.3)

    peak = len(held)
    print(f"peak held {peak} GiB; holding 3 s to let kswapd/shrinkers run")
    time.sleep(3)
    report("under pressure", mem())

    del held
    gc.collect()
    time.sleep(2)
    report("after free", mem())
    time.sleep(5)
    report("after free +5s", mem())
    return 0


if __name__ == "__main__":
    sys.exit(main())
