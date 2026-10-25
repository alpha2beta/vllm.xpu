#!/usr/bin/env python3
"""P1-07: single-expert MLP forward pass on XPU.

Runs the full routed-expert MLP for one expert through the production fused op
(``exl3xpu.ops.exl3_linear``):

    x -> gate = linear(x)          # FWHT(x*su) -> GEMV -> FWHT(.)*sv
       -> up   = linear(x)
       -> h    = SiLU(gate) * up   # activation strictly post-Hadamard
       -> y    = linear(h)         # down projection, same transforms

and compares it with the CPU dequantized reference MLP built from
``exl3xpu.ref.weight_orig`` (fp32), gating on the P0-09 tolerances:

* T-2  ``max_abs <= 0.08`` for M in {1,2,4,8};
* T-5  repeat run is bit-identical;
* T-6  zero NaN/Inf.

It also replays the P0-08 oracle trace inputs for experts 0/255 and checks the
end-to-end MLP output against ``expert/{id}/down_proj/y``.

    .venv/bin/python tests/test_expert_mlp.py

Evidence: ``logs/moe-p1/single-expert-eval.log`` (+ ``--json-out``).
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from exl3xpu import ops
from exl3xpu import ref

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures.safetensors"
TRACES = ROOT / "fixtures" / "reference_moe_traces.npz"
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
T2_MAX_ABS = 0.08
M_VALUES = (1, 2, 4, 8)


class ShardSource:
    def __init__(self, model_dir: Path, layer: int):
        self.model_dir = model_dir
        self.layer = layer
        self.prefix = f"model.language_model.layers.{layer}.mlp.experts"
        self.weight_map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._handles: dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.model_dir / shard), framework="pt")
        return self._handles[shard]

    def has(self, expert: int) -> bool:
        key = f"{self.prefix}.{expert}.gate_proj.trellis"
        shard = self.weight_map.get(key)
        return shard is not None and (self.model_dir / shard).is_file()

    def get(self, expert: int, proj: str, suffix: str) -> torch.Tensor:
        key = f"{self.prefix}.{expert}.{proj}.{suffix}"
        return self._handle(self.weight_map[key]).get_tensor(key)

    def codebook(self, expert: int, proj: str) -> int:
        for suffix, cb in (("mcg", ref.CB_MCG), ("mul1", ref.CB_MUL1)):
            if f"{self.prefix}.{expert}.{proj}.{suffix}" in self.weight_map:
                return cb
        return ref.CB_3INST


class FixtureSource:
    def __init__(self, layer: int):
        self.layer = layer
        self.prefix = f"model.language_model.layers.{layer}.mlp.experts"
        self._f = safe_open(str(FIXTURES), framework="pt")
        self._keys = set(self._f.keys())

    def has(self, expert: int) -> bool:
        return f"{self.prefix}.{expert}.gate_proj.trellis" in self._keys

    def get(self, expert: int, proj: str, suffix: str) -> torch.Tensor:
        return self._f.get_tensor(f"{self.prefix}.{expert}.{proj}.{suffix}")

    def codebook(self, expert: int, proj: str) -> int:
        if f"{self.prefix}.{expert}.{proj}.mcg" in self._keys:
            return ref.CB_MCG
        if f"{self.prefix}.{expert}.{proj}.mul1" in self._keys:
            return ref.CB_MUL1
        return ref.CB_3INST


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_modules(source, expert: int) -> dict[str, dict]:
    out = {}
    for proj in PROJECTIONS:
        trellis = source.get(expert, proj, "trellis")
        suh = source.get(expert, proj, "suh")
        svh = source.get(expert, proj, "svh")
        out[proj] = {
            "trellis": trellis,
            "suh": suh,
            "svh": svh,
            "K": trellis.shape[-1] // 16,
            "cb": source.codebook(expert, proj),
            "k": trellis.shape[0] * 16,
            "n": trellis.shape[1] * 16,
        }
    return out


def reference_mlp(x: torch.Tensor, modules: dict[str, dict]) -> tuple[torch.Tensor, dict]:
    """Literal fp32 dequantized MLP in the original basis (CPU)."""
    w = {
        proj: ref.weight_orig(
            m["trellis"].cpu(), m["suh"].cpu(), m["svh"].cpu(), m["K"], m["cb"]
        )
        for proj, m in modules.items()
    }
    g = x.float() @ w["gate_proj"]
    u = x.float() @ w["up_proj"]
    h = F.silu(g) * u
    y = h @ w["down_proj"]
    return y, {"gate": g, "up": u, "act": h}


def kernel_mlp(x: torch.Tensor, modules: dict[str, dict]) -> tuple[torch.Tensor, dict]:
    """Production fused path (all transforms inside ``ops.exl3_linear``)."""
    def lin(t: torch.Tensor, m: dict) -> torch.Tensor:
        n = m["n"]
        nb = torch.zeros(n // 128, dtype=torch.int32, device=t.device)
        return ops.exl3_linear(
            t, m["trellis"], m["suh"], m["svh"], nb, [0, n], m["K"], m["cb"]
        )

    g = lin(x, modules["gate_proj"])
    u = lin(x, modules["up_proj"])
    h = F.silu(g) * u
    y = lin(h, modules["down_proj"])
    return y, {"gate": g, "up": u, "act": h}


def metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    a = got.float()
    b = want.float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-3)
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99).item())
        if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().item()) if diff.numel() else 0.0,
        "nan_inf": int(torch.isnan(a).sum().item() + torch.isinf(a).sum().item()),
    }


def make_x(m: int, k: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed + m)
    x = (torch.randn((m, k), generator=g, dtype=torch.float32) * 0.05).to(dtype)
    return x


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def case_mlp_vs_dequantized(source, expert: int, dtype_name: str,
                            records: list[dict]) -> None:
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    modules = load_modules(source, expert)
    device = "xpu"
    mods = {
        proj: {
            **m,
            "trellis": m["trellis"].to(device),
            "suh": m["suh"].to(device),
            "svh": m["svh"].to(device),
        }
        for proj, m in modules.items()
    }
    k = modules["gate_proj"]["k"]
    for m_tokens in M_VALUES:
        x = make_x(m_tokens, k, dtype, seed=1234).to(device)
        y_k, parts_k = kernel_mlp(x, mods)
        y_k2, _ = kernel_mlp(x, mods)
        y_ref, parts_ref = reference_mlp(x.cpu(), modules)

        stat = metrics(y_k.cpu(), y_ref)
        stat_gate = metrics(parts_k["gate"].cpu(), parts_ref["gate"])
        stat_up = metrics(parts_k["up"].cpu(), parts_ref["up"])
        stat_down = metrics(y_k.cpu(), y_ref)
        det = bool(torch.equal(y_k, y_k2))
        ok = (
            stat["max_abs"] <= T2_MAX_ABS
            and stat["nan_inf"] == 0
            and det
            and all(s["nan_inf"] == 0 for s in (stat_gate, stat_up, stat_down))
        )
        records.append({
            "case": "mlp_vs_dequantized",
            "expert": expert,
            "dtype": dtype_name,
            "M": m_tokens,
            "K": modules["gate_proj"]["K"],
            "cb": modules["gate_proj"]["cb"],
            "down_max_abs": stat["max_abs"],
            "down_p99_abs": stat["p99_abs"],
            "down_max_rel": stat["max_rel"],
            "gate_max_abs": stat_gate["max_abs"],
            "up_max_abs": stat_up["max_abs"],
            "deterministic": det,
            "nan_inf": stat["nan_inf"],
            "T2_pass": stat["max_abs"] <= T2_MAX_ABS,
            "pass": ok,
        })
        assert ok, (expert, dtype_name, m_tokens, stat, det)


def case_trace_mlp(source, expert: int, records: list[dict]) -> None:
    """Replay the P0-08 trace input through the real MLP for one expert."""
    if not TRACES.is_file():
        records.append({"case": "trace_mlp", "expert": expert, "status": "missing"})
        return
    data = np.load(TRACES)
    key = f"expert/{expert}/down_proj/y"
    if key not in data or not source.has(expert):
        records.append({"case": "trace_mlp", "expert": expert, "status": "missing"})
        return
    modules = load_modules(source, expert)
    mods = {
        proj: {
            **m,
            "trellis": m["trellis"].to("xpu"),
            "suh": m["suh"].to("xpu"),
            "svh": m["svh"].to("xpu"),
        }
        for proj, m in modules.items()
    }
    x = torch.from_numpy(data["inputs/x"].copy()).to(torch.float16).to("xpu")
    y_k, _ = kernel_mlp(x, mods)
    want = torch.from_numpy(data[key].copy()).float()
    stat = metrics(y_k.cpu(), want)
    ok = stat["max_abs"] <= T2_MAX_ABS and stat["nan_inf"] == 0
    records.append({
        "case": "trace_mlp",
        "expert": expert,
        "M": int(x.shape[0]),
        "max_abs": stat["max_abs"],
        "p99_abs": stat["p99_abs"],
        "max_rel": stat["max_rel"],
        "nan_inf": stat["nan_inf"],
        "T2_pass": stat["max_abs"] <= T2_MAX_ABS,
        "pass": ok,
    })
    assert ok, (expert, stat)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def make_source(model_dir: Path, layer: int):
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        src = ShardSource(model_dir, layer)
        if any(src.has(e) for e in (0, 255)):
            return src, "checkpoint"
    if layer == 0:
        src = FixtureSource(layer)
        if any(src.has(e) for e in (0, 255)):
            return src, "fixtures"
    raise SystemExit(f"no EXL3 expert tensors available ({model_dir}, layer {layer})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--dtypes", nargs="*", default=["float16", "bfloat16"])
    ap.add_argument("--log-out", default=str(ROOT / "logs/moe-p1/single-expert-eval.log"))
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p1/single-expert-eval.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("P1-07 needs the XPU; none available")
    assert bool(ops._get_esimd().exl3_supported(3, 1)), "no (K=3, cb=1) kernel"

    source, source_kind = make_source(Path(args.model_dir), args.layer)
    records: list[dict] = []
    failures: list[str] = []

    cases: list[tuple[str, Callable[[], None]]] = []
    for dtype_name in args.dtypes:
        cases.append((
            f"MLP vs dequantized ({dtype_name}, expert {args.expert})",
            lambda d=dtype_name: case_mlp_vs_dequantized(source, args.expert, d, records),
        ))
    for expert in (0, 255):
        if source.has(expert):
            cases.append((
                f"P0-08 trace MLP (expert {expert})",
                lambda e=expert: case_trace_mlp(source, e, records),
            ))

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

    passes = [r for r in records if r.get("pass") is not None]
    summary = {
        "task": "P1-07",
        "pass": not failures and all(r["pass"] for r in passes),
        "source": source_kind,
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "device": torch.xpu.get_device_properties(0).name,
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "cases": results,
        "failures": failures,
        "records": records,
        "worst_max_abs": max((r.get("down_max_abs", r.get("max_abs", 0.0))
                              for r in records), default=0.0),
    }

    log_path = Path(args.log_out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"P1-07 single-expert MLP  source={source_kind} layer={args.layer}",
        f"T2_max_abs={T2_MAX_ABS} worst={summary['worst_max_abs']:.6g}",
    ]
    for r in records:
        lines.append("  " + json.dumps(r, sort_keys=True))
    lines.append(f"pass={summary['pass']} failures={failures}")
    log_path.write_text("\n".join(lines) + "\n")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nworst max_abs={summary['worst_max_abs']:.6g} (T-2 <= {T2_MAX_ABS}); "
          f"wrote {log_path}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
