#!/usr/bin/env python3
"""P2-01/P2-04/P2-07: eager EXL3 MoE block.

Builds a fully-loaded layer-0 routed-expert layer from the real 3.09bpw
checkpoint, then exercises ``Exl3MoEMethod.apply`` (eager per-expert dispatch):

* restricted-routing block vs the P0-08 oracle ``block/y`` (experts {0,255});
* shared-expert isolation and routed+shared accumulation;
* router logits / top-8 / renormalization vs ``router/*``;
* free top-8 routing vs a CPU reference built from the same expert weights;
* token accounting: every one of the B*k assignments contributes.

    .venv/bin/python tests/test_moe_block.py

Evidence: ``logs/moe-p2/moe-block-eval.{log,json}``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _moe_harness import Harness, metrics, T2_MAX_ABS

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "fixtures" / "reference_moe_traces.npz"
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"


def case_router(h: Harness, traces, records) -> None:
    x = torch.from_numpy(traces["inputs/x"].copy()).float().to("xpu")
    logits, ids, weights = h.router(x)
    ref_logits = torch.from_numpy(traces["router/logits"].copy()).float()
    ref_ids = torch.from_numpy(traces["router/topk_ids"].copy()).long()
    ref_w = torch.from_numpy(traces["router/topk_weights"].copy()).float()
    cos = F.cosine_similarity(logits.cpu(), ref_logits, dim=-1)
    agree = (torch.sort(ids.cpu(), dim=-1).values
             == torch.sort(ref_ids, dim=-1).values).all().item()
    w_err = float((weights.cpu() - ref_w).abs().max().item())
    rec = {
        "case": "router_alignment",
        "cos_min": float(cos.min().item()),
        "top8_agreement": float(agree),
        "weight_max_abs": w_err,
        "pass": float(cos.min().item()) >= 0.999 and bool(agree) and w_err <= 1e-5,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_restricted_block(h: Harness, traces, records) -> None:
    x = torch.from_numpy(traces["inputs/x"].copy()).to(torch.float16).to("xpu")
    ids, w = h.restricted_routing(x)
    routed = h.routed(x, ids, w)
    block = torch.from_numpy(traces["block/y"].copy()).float()
    shared = torch.from_numpy(traces["shared/y"].copy()).float()
    st = metrics(routed.cpu(), block - shared)
    rec = {"case": "restricted_block_routed", **st, "pass": st["max_abs"] <= T2_MAX_ABS}
    records.append(rec)
    assert rec["pass"], rec


def case_full_block(h: Harness, traces, records) -> None:
    """Routed (restricted) + shared, with shared computed by the block itself."""
    x = torch.from_numpy(traces["inputs/x"].copy()).to(torch.float16).to("xpu")
    ids, w = h.restricted_routing(x)
    got = h.routed(x, ids, w).float() + h.shared(x)
    want = torch.from_numpy(traces["block/y"].copy()).float()
    st = metrics(got.cpu(), want)
    rec = {"case": "full_block_routed_plus_shared", **st, "pass": st["max_abs"] <= T2_MAX_ABS}
    records.append(rec)
    assert rec["pass"], rec


def case_free_routing(h: Harness, traces, records) -> None:
    x = torch.from_numpy(traces["inputs/x"].copy()).to(torch.float16).to("xpu")
    _, ids, w = h.router(x)
    got = h.routed(x, ids, w).cpu()
    want = h.free_routing_reference(x.cpu().to(torch.float16), ids.cpu(), w.cpu())
    st = metrics(got, want)
    rec = {
        "case": "free_routing_vs_reference",
        "unique_experts": int(torch.unique(ids).numel()),
        "assignments": int(ids.numel()),
        **st,
        "pass": st["max_abs"] <= T2_MAX_ABS and st["nan_inf"] == 0,
    }
    records.append(rec)
    assert rec["pass"], rec


class _SharedMustNotRun:
    """Sentinel: the runner owns the shared expert, so ``apply`` must not call it."""

    def __call__(self, *args, **kwargs):
        raise AssertionError("apply() must not execute the runner-owned shared expert")

    def forward(self, *args, **kwargs):
        raise AssertionError("apply() must not execute the runner-owned shared expert")


def case_shared_ownership(h: Harness, traces, records) -> None:
    """P2-04: shared expert is computed once, by MoERunner, not by ``apply``."""
    x = torch.from_numpy(traces["inputs/x"].copy()).to(torch.float16).to("xpu")
    ids, w = h.restricted_routing(x)
    plain = h.routed(x, ids, w)
    with_sentinel = h.method.apply(
        layer=h.layer,
        x=x,
        topk_weights=w,
        topk_ids=ids,
        shared_experts=_SharedMustNotRun(),
        shared_experts_input=x,
    )
    same = torch.equal(plain, with_sentinel)
    rec = {
        "case": "shared_ownership",
        "apply_ignores_shared_argument": same,
        "pass": same,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_token_accounting(h: Harness, traces, records) -> None:
    """Every (token, slot) assignment must contribute (P2-07)."""
    x = torch.from_numpy(traces["inputs/x"].copy()).to(torch.float16).to("xpu")
    _, ids, w = h.router(x)
    full = h.routed(x, ids, w).cpu().float()
    zero = h.routed(x, ids, torch.zeros_like(w)).cpu().float()
    per_slot = []
    for slot in range(ids.shape[1]):
        w_ablate = w.clone()
        w_ablate[:, slot] = 0.0
        per_slot.append(
            float((full - h.routed(x, ids, w_ablate).cpu().float()).abs().max().item())
        )
    rec = {
        "case": "token_accounting",
        "assignments": int(ids.numel()),
        "tokens": int(ids.shape[0]),
        "topk": int(ids.shape[1]),
        "zero_weights_max_abs": float(zero.abs().max().item()),
        "per_slot_contribution_max_abs": per_slot,
        "all_slots_contribute": all(v > 0 for v in per_slot),
        "pass": float(zero.abs().max().item()) == 0.0 and all(v > 0 for v in per_slot),
    }
    records.append(rec)
    assert rec["pass"], rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--log-out", default=str(ROOT / "logs/moe-p2/moe-block-eval.log"))
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p2/moe-block-eval.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("P2 block test needs the XPU")
    if not TRACES.is_file():
        raise SystemExit(f"missing oracle {TRACES}")

    traces = np.load(TRACES)
    h = Harness(args.model_dir, args.layer)
    records: list[dict] = []
    failures: list[str] = []

    cases: list[tuple[str, Callable[[], None]]] = [
        ("router alignment", lambda: case_router(h, traces, records)),
        ("restricted-routing block vs oracle", lambda: case_restricted_block(h, traces, records)),
        ("routed + shared block vs oracle", lambda: case_full_block(h, traces, records)),
        ("free top-8 routing vs reference", lambda: case_free_routing(h, traces, records)),
        ("shared ownership (no double compute)", lambda: case_shared_ownership(h, traces, records)),
        ("token accounting (all 8 slots)", lambda: case_token_accounting(h, traces, records)),
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

    summary = {
        "task": "P2-01/P2-04/P2-07",
        "pass": not failures,
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "device": torch.xpu.get_device_properties(0).name,
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "cases": results,
        "failures": failures,
        "records": records,
    }
    log_path = Path(args.log_out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"P2 MoE block layer={args.layer} T2={T2_MAX_ABS}"]
    for r in records:
        lines.append("  " + json.dumps(r, sort_keys=True))
    lines.append(f"pass={summary['pass']} failures={failures}")
    log_path.write_text("\n".join(lines) + "\n")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\npass={summary['pass']} failures={failures}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
