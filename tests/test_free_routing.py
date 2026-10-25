#!/usr/bin/env python3
"""P2-05b: free-routing differential test.

Runs the dynamic router (top-8 + softmax renormalization) on seeded random
inputs for several batch sizes and compares the eager ``apply`` output against a
CPU fp32 reference built from the same checkpoint experts.

    .venv/bin/python tests/test_free_routing.py

Evidence: ``logs/moe-p2/free-routing.{log,json}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _moe_harness import Harness, metrics, T2_MAX_ABS, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "fixtures" / "reference_moe_traces.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    h = Harness(args.model_dir, args.layer)
    records = []
    failures = []

    # The oracle trace input must select exactly the recorded top-8.
    if TRACES.is_file():
        traces = np.load(TRACES)
        x = torch.from_numpy(traces["inputs/x"].copy()).to(torch.float16).to("xpu")
        _, ids, w = h.router(x)
        ref_ids = torch.from_numpy(traces["router/topk_ids"].copy()).long()
        agree = (torch.sort(ids.cpu(), dim=-1).values
                 == torch.sort(ref_ids, dim=-1).values).all().item()
        got = h.routed(x, ids, w).cpu().float()
        want = h.free_routing_reference(x.cpu(), ids.cpu(), w.cpu())
        st = metrics(got, want)
        rec = {"case": "oracle_input_top8", "top8_agreement": float(agree), **st,
               "pass": bool(agree) and st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0}
        records.append(rec)
        if not rec["pass"]:
            failures.append(str(rec))

    for tokens, seed in ((1, 201), (4, 202), (16, 203)):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(tokens, 2048, generator=g) * 0.05).half().to("xpu")
        _, ids, w = h.router(x)
        got = h.routed(x, ids, w).cpu().float()
        want = h.free_routing_reference(x.cpu(), ids.cpu(), w.cpu())
        st = metrics(got, want)
        rec = {
            "case": "free_routing",
            "tokens": tokens,
            "unique_experts": int(torch.unique(ids).numel()),
            "assignments": int(ids.numel()),
            **st,
            "pass": st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0,
        }
        records.append(rec)
        if not rec["pass"]:
            failures.append(str(rec))

    summary = {
        "task": "P2-05b",
        "pass": not failures,
        "device": torch.xpu.get_device_properties(0).name,
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "records": records,
        "failures": failures,
    }
    out = ROOT / "logs/moe-p2/free-routing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    (out.with_suffix(".log")).write_text(
        "\n".join([f"P2-05b free routing T2={T2_MAX_ABS}"]
                  + [json.dumps(r, sort_keys=True) for r in records]
                  + [f"pass={summary['pass']}"]) + "\n"
    )
    for r in records:
        print("OK  " if r["pass"] else "FAIL", json.dumps(r, sort_keys=True))
    print(f"pass={summary['pass']}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
