#!/usr/bin/env python3
"""P2-08: intermediate-stage validation probe.

Enables ``Exl3MoEMethod.enable_debug_capture`` and verifies each stage of the
MoE block against the CPU reference / P0-08 oracle:

1. router logits and normalized top-8 weights;
2. per-expert pre-activation ``gate`` and ``up``;
3. activated hidden state ``SiLU(gate) * up``;
4. down-projection output;
5. summed routed tensor and the shared-expert contribution.

    .venv/bin/python tests/test_intermediate_probe.py

Evidence: ``logs/moe-p2/intermediate-stage-validation.json``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from _moe_harness import Harness, metrics, T2_MAX_ABS, DEFAULT_MODEL
from exl3xpu import ref

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "fixtures" / "reference_moe_traces.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p2/intermediate-stage-validation.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    traces = np.load(TRACES)
    h = Harness(args.model_dir, args.layer)
    x_h = torch.from_numpy(traces["inputs/x"].copy()).to(torch.float16).to("xpu")
    x_ref = x_h.cpu()

    stages: dict = {}
    failures: list[str] = []

    # 1. Router logits + normalized weights.
    logits, ids, w = h.router(x_h)
    ref_logits = torch.from_numpy(traces["router/logits"].copy()).float()
    ref_w = torch.from_numpy(traces["router/topk_weights"].copy()).float()
    stages["router"] = {
        "logits": {**metrics(logits.cpu(), ref_logits)},
        "weights": {**metrics(w.cpu(), ref_w)},
        "top8_agreement": float(
            (torch.sort(ids.cpu(), dim=-1).values
             == torch.sort(torch.from_numpy(traces["router/topk_ids"].copy()).long(),
                           dim=-1).values).all().item()
        ),
    }

    # Run the block with capture on.
    h.method.enable_debug_capture()
    try:
        routed = h.routed(x_h, ids, w)
    finally:
        cap = h.method.disable_debug_capture()
    assert cap is not None, "capture hook returned nothing"

    # 2-4. Per-expert stages.
    per_expert = {}
    for e, got in sorted(cap["experts"].items()):
        token_idx, _ = (ids == e).nonzero(as_tuple=True)
        rows = x_ref.index_select(0, token_idx.cpu())
        v = h.method.expert_views(h.layer, e)
        gate_ref = ref.linear_forward(
            rows, v["gate_proj"]["trellis"].cpu(), v["gate_proj"]["suh"].cpu(),
            v["gate_proj"]["svh"].cpu(), v["gate_proj"]["K"], v["gate_proj"]["cb"],
        )
        up_ref = ref.linear_forward(
            rows, v["up_proj"]["trellis"].cpu(), v["up_proj"]["suh"].cpu(),
            v["up_proj"]["svh"].cpu(), v["up_proj"]["K"], v["up_proj"]["cb"],
        )
        act_ref = torch.nn.functional.silu(gate_ref) * up_ref
        down_ref = ref.linear_forward(
            act_ref, v["down_proj"]["trellis"].cpu(), v["down_proj"]["suh"].cpu(),
            v["down_proj"]["svh"].cpu(), v["down_proj"]["K"], v["down_proj"]["cb"],
        )
        per_expert[str(e)] = {
            "rows": int(token_idx.numel()),
            "gate": metrics(got["gate"].cpu(), gate_ref),
            "up": metrics(got["up"].cpu(), up_ref),
            "act": metrics(got["act"].cpu(), act_ref),
            "down": metrics(got["down"].cpu(), down_ref),
        }
    stages["per_expert"] = per_expert

    # 5. Summed routed + shared.
    routed_ref = h.free_routing_reference(x_ref, ids.cpu(), w.cpu())
    stages["routed_sum"] = metrics(cap["routed_fp32"].cpu(), routed_ref)
    shared_ref = torch.from_numpy(traces["shared/y"].copy()).float()
    stages["shared"] = metrics(h.shared(x_h).cpu(), shared_ref)

    # Gate on the worst per-stage error (except router weights, tiny by nature).
    worst = 0.0
    for stage in ("router", "per_expert", "routed_sum", "shared"):
        if stage == "per_expert":
            for rec in stages[stage].values():
                for k in ("gate", "up", "act", "down"):
                    worst = max(worst, rec[k]["max_abs"])
        elif stage == "router":
            worst = max(worst, stages[stage]["logits"]["max_abs"],
                        stages[stage]["weights"]["max_abs"])
        else:
            worst = max(worst, stages[stage]["max_abs"])
    if stages["router"]["top8_agreement"] != 1.0:
        failures.append("router top-8 disagreement")

    summary = {
        "task": "P2-08",
        "pass": not failures and worst <= T2_MAX_ABS,
        "device": torch.xpu.get_device_properties(0).name,
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "worst_max_abs": worst,
        "experts_captured": len(per_expert),
        "failures": failures,
        "stages": stages,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    (out.with_suffix(".log")).write_text(
        "\n".join([
            f"P2-08 intermediate stages: experts={len(per_expert)} worst={worst:.3e} T2={T2_MAX_ABS}",
            "  router: " + json.dumps(stages["router"], sort_keys=True),
            "  routed_sum: " + json.dumps(stages["routed_sum"], sort_keys=True),
            "  shared: " + json.dumps(stages["shared"], sort_keys=True),
            f"pass={summary['pass']}",
        ]) + "\n"
    )
    print(f"experts captured: {len(per_expert)}, worst max_abs={worst:.3e}")
    print(f"pass={summary['pass']} failures={failures}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
