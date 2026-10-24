#!/usr/bin/env python3
"""Empirical platform roofline for the EXL3 MoE bring-up (P0-12).

Three measurements on the Arc 140V:

1. **Memory bandwidth** vs working-set size. ``read`` is a reduction
   (``tensor.sum()``) and ``copy`` is a full read+write; the smallest sizes sit
   in the 8 MiB L2, the largest are DRAM-resident. Peak copy bandwidth is
   reported as ``2 * bytes / time`` (read + write).

2. **Trellis decode ALU throughput**: ``exl3_reconstruct`` (ESIMD) and the
   Triton ``reconstruct`` for the (K, codebook) pairs the checkpoint actually
   uses, reported in GoW/s (giga original-weights per second) so it can be
   compared with the expert GEMM demand.

3. **EXL3 linear achieved bandwidth** for the routed-expert shapes
   (2048x512 gate/up, 512x2048 down) at M=1, which is decode-bound: the number
   to compare against the ~136 GB/s DRAM spec and the 60 tok/s target.

Writes ``logs/moe-preflight/roofline-140v.{json,txt}``.

Usage:
  .venv/bin/python bench/roofline.py
  .venv/bin/python bench/roofline.py --sizes 1 8 64 1024 --reps 50
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "logs" / "moe-preflight"

# LPDDR5X-8533, 64-bit bus: 8533 MT/s * 8 B = 68.3 GB/s per direction; the
# platform number widely quoted for this part is 136 GB/s (shared read+write).
SPEC_BANDWIDTH_GBPS = 136.0
SPEC_FLOPS_FP16_TFLOPS = 3.9  # 64 EU * 8 lanes? see docs/compatibility_manifest.md


def timed(fn, reps: int, warmup: int = 3) -> float:
    """Median wall time in seconds over ``reps`` runs, XPU-synchronised."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return statistics.median(samples)


def bandwidth_sweep(device: torch.device, sizes_mib: list[int], reps: int) -> list[dict]:
    results = []
    for mib in sizes_mib:
        n = mib * 1024 * 1024 // 2  # fp16 elements
        x = torch.randn(n, dtype=torch.float16, device=device)
        y = torch.empty_like(x)
        total = x.numel() * x.element_size()

        read_s = timed(lambda: torch.sum(x), reps)
        copy_s = timed(lambda: y.copy_(x), reps)
        results.append(
            {
                "working_set_mib": mib,
                "bytes": total,
                "read_gbps": total / read_s / 1e9,
                "copy_gbps": 2 * total / copy_s / 1e3 / 1e6,
                "read_seconds": read_s,
                "copy_seconds": copy_s,
            }
        )
        print(
            f"  {mib:>5} MiB  read {total / read_s / 1e9:7.2f} GB/s   "
            f"copy {2 * total / copy_s / 1e9:7.2f} GB/s",
            flush=True,
        )
        del x, y
        torch.xpu.empty_cache()
    return results


def decode_throughput(device: torch.device, reps: int) -> list[dict]:
    from exl3xpu import ops, triton_kernels as tk

    esimd = ops._get_esimd()
    cases = [
        # (label, k, n, K, cb) -- shapes and bitrates from the 3.09bpw checkpoint
        ("routed gate/up 2048x512 K3/mcg", 2048, 512, 3, 1),
        ("routed down 512x2048 K3/mcg", 512, 2048, 3, 1),
        ("shared expert 2048x512 K4/mcg", 2048, 512, 4, 1),
        ("shared expert 2048x512 K4/mul1", 2048, 512, 4, 2),
    ]
    results = []
    for label, k, n, K, cb in cases:
        tk_tiles, tn_tiles = k // 16, n // 16
        trellis = torch.randint(
            -(2**15), 2**15, (tk_tiles, tn_tiles, 16 * K), dtype=torch.int16, device=device
        )
        out = torch.empty((k, n), dtype=torch.float16, device=device)
        weights = k * n
        trellis_bytes = trellis.numel() * 2

        row = {
            "case": label,
            "k": k,
            "n": n,
            "K": K,
            "codebook": {0: "3inst", 1: "mcg", 2: "mul1"}[cb],
            "weights": weights,
            "trellis_bytes": trellis_bytes,
            "esimd_supported": bool(esimd and esimd.exl3_supported(K, cb)),
        }

        # Triton reconstruct (always available)
        s = timed(lambda: tk.reconstruct(trellis, K, cb, out=out), reps)
        row["triton_seconds"] = s
        row["triton_gow_per_s"] = weights / s / 1e9
        row["triton_gbps"] = trellis_bytes / s / 1e9

        # ESIMD reconstruct (only for supported K/cb)
        if row["esimd_supported"] and hasattr(esimd, "exl3_reconstruct"):
            s2 = timed(lambda: esimd.exl3_reconstruct(trellis, out, 0, K, cb), reps)
            row["esimd_seconds"] = s2
            row["esimd_gow_per_s"] = weights / s2 / 1e9
            row["esimd_gbps"] = trellis_bytes / s2 / 1e9
        results.append(row)
        esimd_txt = (
            f"{row['esimd_gow_per_s']:7.1f} GoW/s" if "esimd_gow_per_s" in row else "unsupported  "
        )
        print(
            f"  {label:<34} triton {row['triton_gow_per_s']:7.1f} GoW/s   esimd {esimd_txt}",
            flush=True,
        )
        del trellis, out
        torch.xpu.empty_cache()
    return results


def linear_bandwidth(device: torch.device, reps: int) -> list[dict]:
    """Achieved bandwidth of the full EXL3 linear at M=1 (decode-bound)."""
    from exl3xpu import ops

    esimd = ops._get_esimd()
    results = []
    for label, k, n, K, cb in [
        ("routed gate/up 2048x512 K3/mcg", 2048, 512, 3, 1),
        ("routed down 512x2048 K3/mcg", 512, 2048, 3, 1),
    ]:
        trellis = torch.randint(
            -(2**15), 2**15, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=device
        )
        suh = torch.randn(1, k, dtype=torch.float16, device=device) * 0.0025
        svh = torch.randn(n, dtype=torch.float16, device=device) * 0.02
        shard = torch.zeros(n // 128, dtype=torch.int32, device=device)
        x = torch.randn(1, k, dtype=torch.float16, device=device) * 0.05
        weights = k * n
        trellis_bytes = trellis.numel() * 2

        s = timed(
            lambda: ops.exl3_linear(x, trellis, suh, svh, shard, [0, n], K, cb), reps
        )
        row = {
            "case": label,
            "k": k,
            "n": n,
            "K": K,
            "codebook": {0: "3inst", 1: "mcg", 2: "mul1"}[cb],
            "esimd_path": bool(esimd and esimd.exl3_supported(K, cb)),
            "seconds_per_gemv": s,
            "effective_gbps": trellis_bytes / s / 1e9,
            "weights_per_second": weights / s,
        }
        results.append(row)
        print(
            f"  {label:<34} {s * 1e6:8.1f} us  {row['effective_gbps']:6.2f} GB/s  "
            f"({row['esimd_path'] and 'esimd' or 'triton'})",
            flush=True,
        )
        del trellis, suh, svh, shard, x
        torch.xpu.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sizes", type=int, nargs="*", default=[1, 4, 8, 16, 64, 256, 1024])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--device", default="xpu:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    props = torch.xpu.get_device_properties(0)
    record = {
        "device": props.name,
        "platform": props.platform_name,
        "driver_version": props.driver_version,
        "torch": torch.__version__,
        "kernel": platform.release(),
        "l2_bytes": props.last_level_cache_size,
        "eu_count": props.max_compute_units,
        "spec": {
            "bandwidth_gbps": SPEC_BANDWIDTH_GBPS,
            "fp16_tflops": SPEC_FLOPS_FP16_TFLOPS,
        },
        "notes": [
            "copy_gbps counts read+write (2 * bytes / time).",
            "GoW/s = 10^9 original (post-dequant) weights per second.",
            "ESIMD is the production fast path; Triton is the K-generic "
            "reference/fallback. Both are measured here so the gap is visible.",
            "K=3/mcg decode requires the (K=3, cb=1) ESIMD instantiation; on a "
            "build without it the linear rows report esimd_path=false.",
        ],
    }

    print("1) memory bandwidth sweep")
    record["memory"] = bandwidth_sweep(device, args.sizes, args.reps)
    print("\n2) trellis decode throughput")
    record["decode"] = decode_throughput(device, args.reps)
    print("\n3) EXL3 linear achieved bandwidth (M=1)")
    record["linear"] = linear_bandwidth(device, args.reps)

    peak_copy = max(r["copy_gbps"] for r in record["memory"])
    peak_read = max(r["read_gbps"] for r in record["memory"])
    record["achieved"] = {
        "peak_read_gbps": peak_read,
        "peak_copy_gbps": peak_copy,
        "peak_copy_pct_of_spec": 100.0 * peak_copy / SPEC_BANDWIDTH_GBPS,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "roofline-140v.json").write_text(json.dumps(record, indent=2) + "\n")

    lines = [
        "# Arc 140V empirical roofline (P0-12)",
        f"device: {record['device']} ({record['platform']}, driver {record['driver_version']})",
        f"torch: {record['torch']}  kernel: {record['kernel']}",
        f"L2: {record['l2_bytes'] / 2**20:.0f} MiB   EU: {record['eu_count']}",
        f"spec bandwidth: {SPEC_BANDWIDTH_GBPS:.0f} GB/s",
        "",
        "## Memory bandwidth vs working set",
        "| MiB | read GB/s | copy GB/s (r+w) |",
        "| --- | --- | --- |",
    ]
    for row in record["memory"]:
        lines.append(
            f"| {row['working_set_mib']} | {row['read_gbps']:.2f} | {row['copy_gbps']:.2f} |"
        )
    lines += [
        "",
        f"peak read {peak_read:.2f} GB/s, peak copy {peak_copy:.2f} GB/s "
        f"({100.0 * peak_copy / SPEC_BANDWIDTH_GBPS:.1f}% of the {SPEC_BANDWIDTH_GBPS:.0f} GB/s spec)",
        "",
        "## Trellis decode throughput",
        "| case | K | codebook | triton GoW/s | esimd GoW/s |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in record["decode"]:
        esimd = (
            f"{row['esimd_gow_per_s']:.1f}" if "esimd_gow_per_s" in row else "unsupported"
        )
        lines.append(
            f"| {row['case']} | {row['K']} | {row['codebook']} | "
            f"{row['triton_gow_per_s']:.1f} | {esimd} |"
        )
    lines += [
        "",
        "## EXL3 linear at M=1 (routed-expert shapes)",
        "| case | us/gemv | effective GB/s | path |",
        "| --- | --- | --- | --- |",
    ]
    for row in record["linear"]:
        lines.append(
            f"| {row['case']} | {row['seconds_per_gemv'] * 1e6:.1f} | "
            f"{row['effective_gbps']:.2f} | {'esimd' if row['esimd_path'] else 'triton'} |"
        )
    lines += ["", "## Notes", *[f"- {n}" for n in record["notes"]]]
    (args.out_dir / "roofline-140v.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out_dir / 'roofline-140v.txt'}")
    print(f"wrote {args.out_dir / 'roofline-140v.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
