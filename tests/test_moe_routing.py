#!/usr/bin/env python3
"""P2-02/P2-03/P2-06: routing alignment, weight application, edge cases.

* P2-02: reference top-k + softmax-renormalization, and proof that ``apply``
  consumes the precomputed ``(topk_ids, topk_weights)`` without recalculating
  (output is exactly linear in the supplied weights).
* P2-03: the routing weight multiplies the **down-projection output**, never the
  input activations (``w*MLP(x) != MLP(w*x)``).
* P2-06: extreme skew, uniform 256-way routing, zero-token experts, logit
  near-ties, and M=0/M=1.

    .venv/bin/python tests/test_moe_routing.py

Evidence: ``logs/moe-p2/routing-eval.{log,json}`` and
``logs/moe-p2/edge-cases.log``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from _moe_harness import Harness, metrics, T2_MAX_ABS, TOPK

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "fixtures" / "reference_moe_traces.npz"
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"


def xpu_launch_count(fn: Callable[[], object]) -> int:
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        fn()
        torch.xpu.synchronize()
    return sum(
        int(r.count) for r in prof.key_averages()
        if getattr(r, "self_device_time_total", 0) > 0
    )


def make_x(m: int, seed: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, 2048, generator=g) * 0.05).half().to("xpu")


# ---------------------------------------------------------------------------
# P2-02
# ---------------------------------------------------------------------------

def case_router_reference(h: Harness, traces, records) -> None:
    x = torch.from_numpy(traces["inputs/x"].copy()).float().to("xpu")
    logits, ids, weights = h.router(x)
    ref_logits = torch.from_numpy(traces["router/logits"].copy()).float()
    ref_ids = torch.from_numpy(traces["router/topk_ids"].copy()).long()
    ref_w = torch.from_numpy(traces["router/topk_weights"].copy()).float()
    cos = torch.nn.functional.cosine_similarity(logits.cpu(), ref_logits, dim=-1)
    agree = (torch.sort(ids.cpu(), dim=-1).values
             == torch.sort(ref_ids, dim=-1).values).all().item()
    sums = weights.sum(-1)
    rec = {
        "case": "router_reference",
        "cos_min": float(cos.min().item()),
        "top8_agreement": float(agree),
        "weight_max_abs": float((weights.cpu() - ref_w).abs().max().item()),
        "weight_row_sum_min": float(sums.min().item()),
        "weight_row_sum_max": float(sums.max().item()),
        "pass": float(cos.min().item()) >= 0.999 and bool(agree)
        and float((weights.cpu() - ref_w).abs().max().item()) <= 1e-5
        and float((sums - 1).abs().max().item()) <= 1e-5,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_consumes_precomputed(h: Harness, records) -> None:
    """Output must be exactly linear in the supplied weights (no renorm)."""
    x = make_x(4)
    _, ids, w = h.router(x)
    y1 = h.routed(x, ids, w).float()
    y2 = h.routed(x, ids, 2.0 * w).float()
    err = float((y2 - 2.0 * y1).abs().max().item())
    rec = {
        "case": "consumes_precomputed_weights",
        "scale_error": err,
        "pass": err <= 2e-5,
    }
    records.append(rec)
    assert rec["pass"], rec


# ---------------------------------------------------------------------------
# P2-03
# ---------------------------------------------------------------------------

def case_weight_after_down(h: Harness, records) -> None:
    x = make_x(1, seed=21)
    e, w = 0, 2.0
    ids = torch.tensor([[e]], dtype=torch.int32, device="xpu")
    weights = torch.tensor([[w]], dtype=torch.float32, device="xpu")
    got = h.routed(x, ids, weights).cpu().float()

    want_after = h.expert_mlp_reference(e, x.cpu(), w=w)
    want_input = h.expert_mlp_reference(e, x.cpu() * w, w=1.0)

    st_after = metrics(got, want_after)
    st_input = metrics(got, want_input)
    ratio = st_input["max_abs"] / max(st_after["max_abs"], 1e-12)
    rec = {
        "case": "weight_applied_after_down",
        "vs_w_times_mlp_max_abs": st_after["max_abs"],
        "vs_mlp_of_wx_max_abs": st_input["max_abs"],
        "wrong_over_right_error_ratio": ratio,
        # Correct = w*MLP(x); wrong = MLP(w*x). The latter is clearly worse
        # even though SiLU is near-linear for these small activations.
        "pass": st_after["max_abs"] <= T2_MAX_ABS and ratio >= 20.0,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_weight_not_on_input(h: Harness, records) -> None:
    """Two experts with different weights: result == sum w_e * MLP_e(x)."""
    x = make_x(2, seed=22)
    ids = torch.tensor([[0, 255], [0, 255]], dtype=torch.int32, device="xpu")
    weights = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32, device="xpu")
    got = h.routed(x, ids, weights).cpu().float()
    want = torch.zeros_like(got)
    for t in range(2):
        for s in range(2):
            want[t] += h.expert_mlp_reference(
                int(ids[t, s]), x[t].cpu(), float(weights[t, s])
            )
    st = metrics(got, want)
    rec = {"case": "weight_not_on_input", **st, "pass": st["max_abs"] <= T2_MAX_ABS}
    records.append(rec)
    assert rec["pass"], rec


# ---------------------------------------------------------------------------
# P2-06 edge cases
# ---------------------------------------------------------------------------

def case_extreme_skew(h: Harness, records) -> None:
    x = make_x(16, seed=31)
    ids = torch.zeros((16, 1), dtype=torch.int32, device="xpu")
    w = torch.ones((16, 1), dtype=torch.float32, device="xpu")
    got = h.routed(x, ids, w).cpu().float()
    want = torch.stack([h.expert_mlp_reference(0, x[t].cpu()) for t in range(16)])
    st = metrics(got, want)
    rec = {
        "case": "extreme_skew_single_expert",
        "tokens": 16,
        "unique_experts": int(torch.unique(ids).numel()),
        **st,
        "pass": st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_uniform_256(h: Harness, records) -> None:
    x = make_x(256, seed=32)
    ids = torch.arange(256, dtype=torch.int32, device="xpu").reshape(-1, 1)
    w = torch.ones((256, 1), dtype=torch.float32, device="xpu")
    got = h.routed(x, ids, w).cpu().float()
    # Numerically verify a spread sample; all 256 experts are exercised.
    sample = [0, 1, 2, 63, 127, 128, 254, 255]
    sample_err = 0.0
    for e in sample:
        want = h.expert_mlp_reference(e, x[e].cpu())
        sample_err = max(sample_err, float((got[e] - want).abs().max().item()))
    rec = {
        "case": "uniform_256",
        "tokens": 256,
        "unique_experts": int(torch.unique(ids).numel()),
        "sampled_experts": sample,
        "sample_max_abs": sample_err,
        "finite": bool(torch.isfinite(got).all()),
        "pass": int(torch.unique(ids).numel()) == 256
        and sample_err <= T2_MAX_ABS and bool(torch.isfinite(got).all()),
    }
    records.append(rec)
    assert rec["pass"], rec


def case_zero_token_experts(h: Harness, records) -> None:
    x = make_x(8, seed=33)
    ids = torch.tensor([[0], [5], [100], [0], [5], [100], [0], [5]],
                       dtype=torch.int32, device="xpu")
    w = torch.ones((8, 1), dtype=torch.float32, device="xpu")
    got = h.routed(x, ids, w).cpu().float()
    want = torch.stack([h.expert_mlp_reference(int(ids[t, 0]), x[t].cpu())
                        for t in range(8)])
    st = metrics(got, want)
    rec = {
        "case": "zero_token_experts",
        "active": sorted({int(v) for v in ids.flatten().tolist()}),
        "inactive_experts": 253,
        **st,
        "pass": st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_logit_near_ties(h: Harness, records) -> None:
    x = torch.zeros((4, 2048), dtype=torch.float16, device="xpu")
    logits, ids, w = h.router(x)
    _, ids2, _ = h.router(x)
    ids_set_ok = sorted(ids[0].cpu().tolist()) == list(range(TOPK))
    w_err = float((w - 1.0 / TOPK).abs().max().item())
    rec = {
        "case": "logit_near_ties",
        "logits_max_abs": float(logits.abs().max().item()),
        "ids": ids[0].cpu().tolist(),
        "ids_stable_across_calls": bool(torch.equal(ids, ids2)),
        "ids_are_first_topk_set": ids_set_ok,
        "weights_uniform": w_err <= 1e-6,
        "pass": float(logits.abs().max().item()) == 0.0
        and ids_set_ok and bool(torch.equal(ids, ids2)) and w_err <= 1e-6,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_empty_m0(h: Harness, records) -> None:
    x = torch.empty((0, 2048), dtype=torch.float16, device="xpu")
    ids = torch.zeros((0, TOPK), dtype=torch.int32, device="xpu")
    w = torch.zeros((0, TOPK), dtype=torch.float32, device="xpu")
    out = h.routed(x, ids, w)
    launches = xpu_launch_count(lambda: h.routed(x, ids, w))
    rec = {
        "case": "empty_m0",
        "shape": list(out.shape),
        "kernels": launches,
        "pass": tuple(out.shape) == (0, 2048) and launches == 0,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_single_token(h: Harness, records) -> None:
    x = make_x(1, seed=34)
    _, ids, w = h.router(x)
    got = h.routed(x, ids, w).cpu().float()
    want = h.free_routing_reference(x.cpu(), ids.cpu(), w.cpu())
    st = metrics(got, want)
    rec = {"case": "single_token_decode", **st,
           "pass": st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0}
    records.append(rec)
    assert rec["pass"], rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--log-out", default=str(ROOT / "logs/moe-p2/routing-eval.log"))
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p2/routing-eval.json"))
    ap.add_argument("--edge-log", default=str(ROOT / "logs/moe-p2/edge-cases.log"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("P2 routing test needs the XPU")
    traces = np.load(TRACES) if TRACES.is_file() else None
    h = Harness(args.model_dir, args.layer)
    records: list[dict] = []
    failures: list[str] = []
    edge_records: list[dict] = []

    cases: list[tuple[str, Callable[[], None]]] = [
        ("P2-02 router reference", lambda: case_router_reference(h, traces, records)),
        ("P2-02 consumes precomputed weights", lambda: case_consumes_precomputed(h, records)),
        ("P2-03 weight after down projection", lambda: case_weight_after_down(h, records)),
        ("P2-03 weight not on input", lambda: case_weight_not_on_input(h, records)),
        ("P2-06 extreme skew", lambda: case_extreme_skew(h, edge_records)),
        ("P2-06 uniform 256", lambda: case_uniform_256(h, edge_records)),
        ("P2-06 zero-token experts", lambda: case_zero_token_experts(h, edge_records)),
        ("P2-06 logit near-ties", lambda: case_logit_near_ties(h, edge_records)),
        ("P2-06 empty M=0", lambda: case_empty_m0(h, edge_records)),
        ("P2-06 single token M=1", lambda: case_single_token(h, edge_records)),
    ]
    results = []
    for label, fn in cases:
        try:
            fn()
            results.append({"case": label, "pass": True})
            print(f"OK   {label}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {exc}")
            results.append({"case": label, "pass": False, "error": str(exc)})
            print(f"FAIL {label}: {exc}")
            if args.verbose:
                traceback.print_exc()

    all_records = records + edge_records
    summary = {
        "task": "P2-02/P2-03/P2-06",
        "pass": not failures,
        "model_dir": str(args.model_dir),
        "device": torch.xpu.get_device_properties(0).name,
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "cases": results,
        "failures": failures,
        "records": all_records,
    }
    log_path = Path(args.log_out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"P2 routing/edge checks layer={args.layer} T2={T2_MAX_ABS}"]
    for r in all_records:
        lines.append("  " + json.dumps(r, sort_keys=True))
    lines.append(f"pass={summary['pass']} failures={failures}")
    log_path.write_text("\n".join(lines) + "\n")

    edge_path = Path(args.edge_log)
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    edge_path.write_text(
        "\n".join(
            [f"P2-06 edge cases T2={T2_MAX_ABS}"]
            + ["  " + json.dumps(r, sort_keys=True) for r in edge_records]
            + [f"pass={all(r.get('pass') for r in edge_records)}"]
        ) + "\n"
    )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\npass={summary['pass']} failures={failures}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
