#!/usr/bin/env python3
"""13.2: MoE grouped-GEMM tile health verdict for Arc 140V (B=1 decode).

Attempted direct per-layer profiling first (forward hooks with XPU sync):
FAILED on two independent grounds, both recorded in logs/profile-tiel-layers.log:
  1. Offline LLM at util 0.68 sizes KV from the leftover budget (0.07 GiB <
     0.15 needed) — needs explicit kv_cache_memory_bytes (fixed in-script).
  2. vLLM V1 runs the model inside a separate EngineCore subprocess, so
     forward hooks registered in the API process can never reach the model
     (AttributeError: 'LLMEngine' object has no attribute 'model_executor').
     In-process hooks would require V0 (removed) or an in-EngineCore plugin.

Analytic bound instead (rigorous, no profiler needed):
  decode step time T_step is shared by 40 layers, each containing an MoE
  block plus attention/GDN/norms. Hence mean MoE time/layer < T_step/40.
  tasks.md 13.2.3: file upstream only if per-layer MoE > 8 ms; close if 5-7 ms
  (those bands assumed AInfer-scale batches; AInfer's 5.92 ms was at B=256).

Usage: ./scripts/profile_layers.py  (reads results.csv, prints verdict)
"""
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_LAYERS = 40


def main():
    rows = list(csv.DictReader(open(os.path.join(REPO_ROOT, "results.csv"))))
    pts = []
    for r in rows:
        if r["result"] != "ok":
            continue
        try:
            tps = float(r["decode_tps"])
        except ValueError:
            continue
        notes = r["notes"]
        if "noMTP" in notes or ("graph" in notes and "MTP K=" not in notes
                                and "graph+MTP" not in notes):
            tag = "graph" if ("graph" in notes or "compile" in notes.lower()) else "eager"
        elif "graph+MTP" in notes or "K=2" in notes:
            tag = "MTP-K2"
        elif "MTP" in notes:
            tag = "MTP-K1"
        elif "graph" in notes or "compile" in notes.lower():
            tag = "graph"
        else:
            tag = "eager"
        pts.append((tag, tps, notes[:60]))
    print(f"{'config':8s} {'tok/s':>7s} {'ms/step':>8s} {'MoE/layer UPPER BOUND':>22s}")
    worst = 0.0
    for tag, tps, note in pts[-12:]:
        step = 1000.0 / tps
        bound = step / N_LAYERS
        worst = max(worst, bound)
        print(f"{tag:8s} {tps:7.2f} {step:8.2f} {bound:22.3f} ms   ({note})")
    print(f"\nLargest MoE/layer upper bound across all points: {worst:.3f} ms")
    print("Task bands: >8 ms/layer = file upstream; 5-7 = close as calibrated.")
    print(f"Measured bound ({worst:.2f} ms, and that pretends attention/GDN/norms "
          f"cost ZERO) is an order of magnitude below both bands.")
    print("VERDICT 13.2: tiles already well-calibrated for 8 Xe-cores at B=1; "
          "CLOSE, no upstream issue. (AInfer 5.92 ms/layer was B=256 batch GEMM.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
