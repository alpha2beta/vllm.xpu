"""Benchmark Layer-0 Prefill Latency: Grouped vs Eager across T in [16, 64, 128, 256]."""

from __future__ import annotations

import json
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from _moe_harness import HIDDEN, Harness

def benchmark_prefill():
    print("Initializing Harness for Layer 0...")
    h = Harness()
    h.method._build_exec_cache(h.layer)
    dev = torch.device("xpu")
    dtype = torch.bfloat16

    results = {}
    tokens_list = [16, 64, 128, 256]
    iters = 15
    warmup = 5

    print(f"\n{'T':>4} | {'Active Exp':>10} | {'<=8 Exp':>7} | {'>8 Exp':>7} | {'Eager (ms)':>11} | {'Grouped (ms)':>12} | {'Speedup':>8} | {'Max Diff':>10}")
    print("-" * 85)

    for T in tokens_list:
        torch.manual_seed(42 + T)
        x = torch.randn(T, HIDDEN, dtype=dtype, device=dev)
        logits = torch.randn(T, 256, device=dev)
        weights, ids = torch.topk(logits.softmax(dim=-1), 8, dim=-1)

        # Count active experts and distribution
        flat_ids = ids.reshape(-1)
        unique_experts, counts = torch.unique(flat_ids, return_counts=True)
        active_u = unique_experts.shape[0]
        small_u = (counts <= 8).sum().item()
        large_u = (counts > 8).sum().item()

        # Warmup and verify parity
        h.method._grouped_gemm_supported = False
        out_eager = h.method.apply(h.layer, x, weights, ids, None, None)
        h.method._grouped_gemm_supported = True
        out_grouped = h.method.apply(h.layer, x, weights, ids, None, None)

        diff = (out_grouped.float() - out_eager.float()).abs().max().item()

        # Measure eager latency
        h.method._grouped_gemm_supported = False
        for _ in range(warmup):
            h.method.apply(h.layer, x, weights, ids, None, None)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            h.method.apply(h.layer, x, weights, ids, None, None)
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        ms_eager = (t1 - t0) * 1000.0 / iters

        # Measure grouped latency
        h.method._grouped_gemm_supported = True
        for _ in range(warmup):
            h.method.apply(h.layer, x, weights, ids, None, None)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            h.method.apply(h.layer, x, weights, ids, None, None)
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        ms_grouped = (t1 - t0) * 1000.0 / iters

        speedup = ms_eager / ms_grouped if ms_grouped > 0 else 0.0

        print(f"{T:4d} | {active_u:10d} | {small_u:7d} | {large_u:7d} | {ms_eager:11.2f} | {ms_grouped:12.2f} | {speedup:7.2f}x | {diff:10.2e}")

        results[f"T_{T}"] = {
            "tokens": T,
            "active_experts": active_u,
            "small_experts_le8": small_u,
            "large_experts_gt8": large_u,
            "ms_eager": ms_eager,
            "ms_grouped": ms_grouped,
            "speedup": speedup,
            "max_abs_diff": diff,
            "projected_40L_eager_ms": ms_eager * 40,
            "projected_40L_grouped_ms": ms_grouped * 40,
        }

    out_file = Path("logs/moe-p4/prefill-speedup.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    log_file = Path("logs/moe-p4/prefill-speedup.log")
    with open(log_file, "w") as f:
        f.write("=== P4-03 Prefill Speedup Benchmark (Layer 0) ===\n")
        f.write(f"Device: {torch.xpu.get_device_name(0)}\n")
        f.write(f"{'T':>4} | {'Active Exp':>10} | {'<=8 Exp':>7} | {'>8 Exp':>7} | {'Eager (ms)':>11} | {'Grouped (ms)':>12} | {'Speedup':>8} | {'Max Diff':>10}\n")
        f.write("-" * 85 + "\n")
        for T in tokens_list:
            r = results[f"T_{T}"]
            f.write(f"{T:4d} | {r['active_experts']:10d} | {r['small_experts_le8']:7d} | {r['large_experts_gt8']:7d} | {r['ms_eager']:11.2f} | {r['ms_grouped']:12.2f} | {r['speedup']:7.2f}x | {r['max_abs_diff']:10.2e}\n")
        f.write("-" * 85 + "\n")
        for T in tokens_list:
            r = results[f"T_{T}"]
            f.write(f"T={T:3d}: 40-layer prefill projected eager={r['projected_40L_eager_ms']:.1f} ms, grouped={r['projected_40L_grouped_ms']:.1f} ms (saving {r['projected_40L_eager_ms'] - r['projected_40L_grouped_ms']:.1f} ms)\n")

    print(f"\nResults written to {out_file} and {log_file}")

if __name__ == "__main__":
    benchmark_prefill()
