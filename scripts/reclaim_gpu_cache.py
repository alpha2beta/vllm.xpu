#!/usr/bin/env python3
"""Reclaim GPU memory cached by the Xe driver before starting vLLM.

Why this exists
---------------
`/proc/meminfo` exposes `GPUActive` / `GPUReclaim` (xe driver). After a big GPU
run — or a crashed one — the driver keeps the freed device memory in
`GPUReclaim` instead of returning it to the OS. Level-Zero's "free memory"
query does *not* count that pool, so vLLM's startup check

    free_memory >= total_memory * gpu_memory_utilization

fails (e.g. "Free memory ... 11.01/28.58 GiB ... less than desired ...
0.74, 21.15 GiB") even though the memory is perfectly allocatable.

Applying host memory pressure makes the kernel's shrinkers drain `GPUReclaim`
back to the OS. Measured on this host (scripts/pressure_test_reclaim.py):

    held 0 GiB : MemFree 12.01  MemAvailable 15.52  GPUReclaim 12.38 (GiB)
    held 10GiB : MemFree 18.18  MemAvailable 18.17  GPUReclaim  0.00 (GiB)
    after free : MemFree 26.90  MemAvailable 27.06  GPUReclaim  0.00 (GiB)

Usage:
    source .venv/bin/activate
    python scripts/reclaim_gpu_cache.py [--max-reclaim-mib 256] [--max-hold-gib 16]

Exit code 0 when GPUReclaim is at/below the target (or already was).
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

CHUNK = 1 << 30


def mem() -> dict[str, int]:
    d: dict[str, int] = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0])
    return d


def gib(kib: int) -> float:
    return kib / 1048576


def report(tag: str, d: dict[str, int]) -> None:
    print(
        f"{tag:24s} MemFree {gib(d['MemFree']):6.2f}  "
        f"MemAvailable {gib(d['MemAvailable']):6.2f}  "
        f"GPUActive {gib(d.get('GPUActive', 0)):5.2f}  "
        f"GPUReclaim {gib(d.get('GPUReclaim', 0)):5.2f} (GiB)",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-reclaim-mib", type=int, default=256,
                    help="stop once GPUReclaim is at or below this")
    ap.add_argument("--max-hold-gib", type=int, default=16,
                    help="safety cap on host memory we hold")
    ap.add_argument("--min-available-mib", type=int, default=6144,
                    help="safety floor on MemAvailable (NOT MemFree: page cache "
                         "makes MemFree look tiny even when the system is idle)")
    ap.add_argument("--min-free-mib", type=int, default=256,
                    help="hard floor on MemFree to avoid the OOM killer")
    ap.add_argument("--settle-s", type=float, default=3.0)
    args = ap.parse_args()

    before = mem()
    report("start", before)
    target = args.max_reclaim_mib * 1024
    if before.get("GPUReclaim", 0) <= target:
        print(f"nothing to do: GPUReclaim already <= {args.max_reclaim_mib} MiB")
        return 0

    held: list[bytearray] = []
    allocated = 0
    while True:
        d = mem()
        if d.get("GPUReclaim", 0) <= target:
            print(f"target reached: GPUReclaim {gib(d['GPUReclaim']):.2f} GiB")
            break
        if d["MemAvailable"] < args.min_available_mib * 1024:
            print(f"stop: MemAvailable {gib(d['MemAvailable']):.2f} GiB hit the safety floor")
            break
        if d["MemFree"] < args.min_free_mib * 1024:
            print(f"stop: MemFree {gib(d['MemFree']):.2f} GiB hit the hard floor")
            break
        if allocated >= args.max_hold_gib * CHUNK:
            print(f"stop: reached the {args.max_hold_gib} GiB hold cap")
            break
        b = bytearray(CHUNK)
        b[::4096] = b"\x01" * (CHUNK // 4096)  # touch every page
        held.append(b)
        allocated += CHUNK
        report(f"holding {allocated // CHUNK} GiB", mem())
        time.sleep(0.2)

    time.sleep(args.settle_s)
    report("under pressure", mem())

    del held
    gc.collect()
    time.sleep(2)
    after = mem()
    report("after release", after)

    ok = after.get("GPUReclaim", 0) <= max(target, before.get("GPUReclaim", 0))
    print("RESULT:", "RECLAIMED" if after.get("GPUReclaim", 0) < before.get("GPUReclaim", 0)
          else "NO CHANGE")
    print(f"MemAvailable {gib(before['MemAvailable']):.2f} -> "
          f"{gib(after['MemAvailable']):.2f} GiB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
