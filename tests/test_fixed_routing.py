#!/usr/bin/env python3
"""P2-05a: fixed-routing differential test.

Forces every token onto the same fixed expert subset {0..7} with fixed weights,
isolating expert execution from router selection, and compares the eager
``apply`` output against a CPU fp32 reference.

    .venv/bin/python tests/test_fixed_routing.py

Evidence: ``logs/moe-p2/fixed-routing.{log,json}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _moe_harness import Harness, metrics, T2_MAX_ABS, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[1]
EXPERTS = (0, 1, 2, 3, 4, 5, 6, 7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    h = Harness(args.model_dir, args.layer)
    records = []
    failures = []

    # Uniform fixed weights over {0..7}.
    for tokens, seed in ((8, 101), (1, 102)):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(tokens, 2048, generator=g) * 0.05).half().to("xpu")
        ids = torch.tensor([list(EXPERTS)], dtype=torch.int32, device="xpu").repeat(tokens, 1)
        w = torch.full((tokens, len(EXPERTS)), 1.0 / len(EXPERTS), device="xpu")
        got = h.routed(x, ids, w).cpu().float()
        want = torch.stack([
            sum(h.expert_mlp_reference(e, x[t].cpu(), 1.0 / len(EXPERTS)) for e in EXPERTS)
            for t in range(tokens)
        ])
        st = metrics(got, want)
        rec = {"case": "fixed_uniform", "tokens": tokens, "experts": list(EXPERTS), **st,
               "pass": st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0}
        records.append(rec)
        if not rec["pass"]:
            failures.append(str(rec))

    # Non-uniform fixed weights (still fixed routing).
    g = torch.Generator().manual_seed(103)
    x = (torch.randn(4, 2048, generator=g) * 0.05).half().to("xpu")
    ids = torch.tensor([list(EXPERTS)], dtype=torch.int32, device="xpu").repeat(4, 1)
    w = torch.softmax(torch.randn(4, len(EXPERTS), generator=torch.Generator().manual_seed(104)), -1).to("xpu")
    got = h.routed(x, ids, w).cpu().float()
    want = torch.stack([
        sum(h.expert_mlp_reference(e, x[t].cpu(), float(w[t, j]))
            for j, e in enumerate(EXPERTS))
        for t in range(4)
    ])
    st = metrics(got, want)
    rec = {"case": "fixed_nonuniform", "tokens": 4, "experts": list(EXPERTS), **st,
           "pass": st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0}
    records.append(rec)
    if not rec["pass"]:
        failures.append(str(rec))

    summary = {
        "task": "P2-05a",
        "pass": not failures,
        "device": torch.xpu.get_device_properties(0).name,
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "records": records,
        "failures": failures,
    }
    out = ROOT / "logs/moe-p2/fixed-routing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    (out.with_suffix(".log")).write_text(
        "\n".join([f"P2-05a fixed routing T2={T2_MAX_ABS}"]
                  + [json.dumps(r, sort_keys=True) for r in records]
                  + [f"pass={summary['pass']}"]) + "\n"
    )
    for r in records:
        print("OK  " if r["pass"] else "FAIL", json.dumps(r, sort_keys=True))
    print(f"pass={summary['pass']}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
