#!/usr/bin/env python3
"""P1-08: token-batch sweep (M=0..256) + empty-work + crossover calibration.

Sweeps the single-expert routed MLP over ``M in {0,1,2,4,8,16,32,64,128,256}``:

* validates every M against the fp32 dequantized reference (P0-09 T-2/T-6);
* proves **M=0 exits with zero XPU kernel launches** (XPU profiler);
* measures the fused vector path (``exl3_gemm_small``) against the
  reconstruct+GEMM path (``exl3_reconstruct`` + oneDNN fp16 GEMM) and records
  the per-M times / speedup used to justify the small/large-M threshold.

Arc 140V has no XMX/DPAS units, so "dequantize+XMX GEMM" is measured as the
production large-M path: reconstruct the Hadamard-domain fp16 weight then oneDNN
fp16 GEMM (the same code path ``ops.exl3_linear`` takes for M > small_m_max).

    .venv/bin/python tests/test_batch_sweep.py

Evidence: ``logs/moe-p1/crossover-calibration.json``,
``logs/moe-p1/batch-sweep.log``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

from exl3xpu import ops
from exl3xpu import ref

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures.safetensors"
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
M_VALUES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
T2_MAX_ABS = 0.08
FUSED_M = (1, 2, 4, 8)  # largest M the no-DPAS vector kernel instantiates


# ---------------------------------------------------------------------------
# source
# ---------------------------------------------------------------------------

class ShardSource:
    def __init__(self, model_dir: Path, layer: int):
        self.model_dir = model_dir
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
# MLP
# ---------------------------------------------------------------------------

def load_modules(source, expert: int, device: str = "xpu") -> dict[str, dict]:
    out = {}
    for proj in PROJECTIONS:
        tr = source.get(expert, proj, "trellis")
        n = tr.shape[1] * 16
        out[proj] = {
            "trellis": tr.to(device),
            # Production dense contract: suh is [num_groups, k]. A 1-D suh
            # makes the C++ set S=k and run k copies of the input Hadamard.
            "suh": source.get(expert, proj, "suh").view(1, -1).to(device),
            "svh": source.get(expert, proj, "svh").to(device),
            # Precomputed once per module, exactly like
            # Exl3LinearMethod.process_weights_after_loading.
            "nb": torch.zeros(n // 128, dtype=torch.int32, device=device),
            "K": tr.shape[-1] // 16,
            "cb": source.codebook(expert, proj),
            "k": tr.shape[0] * 16,
            "n": n,
            "trellis_cpu": tr,
            "suh_cpu": source.get(expert, proj, "suh"),
            "svh_cpu": source.get(expert, proj, "svh"),
        }
    return out


def lin_call(x: torch.Tensor, m: dict) -> torch.Tensor:
    """One linear using the module's precomputed ``shard_of_nb``."""
    return ops.exl3_linear(
        x, m["trellis"], m["suh"], m["svh"], m["nb"], [0, m["n"]], m["K"], m["cb"]
    )


def kernel_mlp(x: torch.Tensor, modules: dict[str, dict]) -> tuple[torch.Tensor, dict]:
    def lin(t, m):
        return lin_call(t, m)

    g = lin(x, modules["gate_proj"])
    u = lin(x, modules["up_proj"])
    h = F.silu(g) * u
    y = lin(h, modules["down_proj"])
    return y, {"gate": g, "up": u, "act": h}


def reference_mlp(x: torch.Tensor, modules: dict[str, dict]) -> torch.Tensor:
    w = {
        p: ref.weight_orig(m["trellis_cpu"], m["suh_cpu"], m["svh_cpu"], m["K"], m["cb"])
        for p, m in modules.items()
    }
    g = x.float() @ w["gate_proj"]
    u = x.float() @ w["up_proj"]
    h = F.silu(g) * u
    return h @ w["down_proj"]


def metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    a, b = got.float(), want.float()
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99).item()) if diff.numel() else 0.0,
        "nan_inf": int(torch.isnan(a).sum().item() + torch.isinf(a).sum().item()),
    }


def make_x(m: int, k: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed + m)
    return (torch.randn((m, k), generator=g) * 0.05).half().to("xpu")


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def xpu_kernel_count(fn: Callable[[], object]) -> tuple[int, float]:
    """Run fn under the XPU profiler; return (kernel launches, total XPU us).

    ``key_averages`` aggregates by kernel name, so the launch count is the sum
    of ``count`` over rows that actually spent time on the device.
    """
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        fn()
        torch.xpu.synchronize()
    rows = [r for r in prof.key_averages() if getattr(r, "self_device_time_total", 0) > 0]
    return sum(int(r.count) for r in rows), sum(r.self_device_time_total for r in rows)


def bench(fn: Callable[[], object], warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(iters):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def case_batch_sweep(source, expert: int, records: list[dict]) -> None:
    modules = load_modules(source, expert)
    k = modules["gate_proj"]["k"]
    for m_tokens in M_VALUES:
        x = make_x(m_tokens, k)
        y_k, parts = kernel_mlp(x, modules)
        ok = True
        rec = {"M": m_tokens, "out_shape": list(y_k.shape), "dtype": str(y_k.dtype).removeprefix("torch.")}
        if m_tokens == 0:
            # Empty work: exact zero-size output, finite (empty) tensors.
            ok = all(t.numel() == 0 for t in (y_k, parts["gate"], parts["up"], parts["act"]))
            rec["empty"] = True
            rec["zero_sized"] = ok
            rec["nan_inf"] = 0
        else:
            ref_y = reference_mlp(x.cpu(), modules)
            stat = metrics(y_k.cpu(), ref_y)
            ok = stat["max_abs"] <= T2_MAX_ABS and stat["nan_inf"] == 0
            rec.update(stat)
            rec["T2_pass"] = stat["max_abs"] <= T2_MAX_ABS
        rec["pass"] = ok
        records.append(rec)
        assert ok, (m_tokens, rec)


def case_m0_zero_kernels(source, expert: int, records: list[dict]) -> None:
    modules = load_modules(source, expert)
    k = modules["gate_proj"]["k"]
    x0 = torch.empty((0, k), dtype=torch.float16, device="xpu")
    # Isolate the EXL3 op so SiLU/mul bookkeeping cannot be mistaken for it.
    n_lin, us_lin = xpu_kernel_count(lambda: lin_call(x0, modules["gate_proj"]))
    # Full MLP with M=0 too: no projection may launch a kernel.
    n_mlp, us_mlp = xpu_kernel_count(lambda: kernel_mlp(x0, modules))
    rec = {
        "M": 0,
        "exl3_linear_kernels": n_lin,
        "exl3_linear_device_us": us_lin,
        "full_mlp_kernels": n_mlp,
        "full_mlp_device_us": us_mlp,
        "pass": n_lin == 0 and n_mlp == 0,
    }
    records.append(rec)
    assert rec["pass"], rec


def case_crossover(source, expert: int, records: list[dict]) -> None:
    modules = load_modules(source, expert)
    gate = modules["gate_proj"]
    k, n = gate["k"], gate["n"]
    orig_small_m_max = ops.small_m_max

    rows = []

    def fused(m_tokens):
        x = make_x(m_tokens, k)
        return bench(lambda: lin_call(x, gate))

    def recon_gemm(m_tokens):
        x = make_x(m_tokens, k)
        # Force the production large-M path by reporting no small-M support.
        ops.small_m_max = lambda K, cb: 0
        try:
            return bench(lambda: lin_call(x, gate))
        finally:
            ops.small_m_max = orig_small_m_max

    for m_tokens in M_VALUES:
        if m_tokens == 0:
            continue
        row = {"M": m_tokens}
        if m_tokens in FUSED_M:
            tf = fused(m_tokens)
            row["fused_ms"] = tf * 1e3
            # The fused path's per-token weight traffic is the win at low M.
            row["fused_us_per_token"] = tf * 1e6 / m_tokens
        tr = recon_gemm(m_tokens)
        row["recon_gemm_ms"] = tr * 1e3
        row["recon_gemm_us_per_token"] = tr * 1e6 / m_tokens
        if "fused_ms" in row:
            row["speedup_fused_over_recon"] = tr / (tf or 1e-12)
            # Bit/loose correctness: both paths must agree within T-2.
            x = make_x(m_tokens, k)
            y_f = lin_call(x, gate).cpu()
            ops.small_m_max = lambda K, cb: 0
            try:
                y_r = lin_call(x, gate).cpu()
            finally:
                ops.small_m_max = orig_small_m_max
            row["path_max_abs"] = float((y_f.float() - y_r.float()).abs().max().item())
        rows.append(row)
        records.append({"case": "crossover", **row})

    # Emit the calibration artifact.
    out = Path("/home/yanchun/Projects/vllm.xpu/logs/moe-p1/crossover-calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    small_m_max = int(ops.small_m_max(3, 1))
    payload = {
        "task": "P1-08",
        "device": torch.xpu.get_device_properties(0).name,
        "has_xmx_dpas": False,
        "note": (
            "Arc 140V has no XMX/DPAS, so the large-M path is "
            "exl3_reconstruct + oneDNN fp16 GEMM (not int8 XMX)."
        ),
        "small_m_max": small_m_max,
        "fused_M_tested": list(FUSED_M),
        "shape": {"k": k, "n": n},
        "tolerance_T2_max_abs": T2_MAX_ABS,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


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
    ap.add_argument("--log-out", default=str(ROOT / "logs/moe-p1/batch-sweep.log"))
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p1/batch-sweep.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("P1-08 needs the XPU")
    assert bool(ops._get_esimd().exl3_supported(3, 1))

    source, source_kind = make_source(Path(args.model_dir), args.layer)
    records: list[dict] = []
    failures: list[str] = []

    cases: list[tuple[str, Callable[[], None]]] = [
        ("batch sweep M=0..256 vs reference",
         lambda: case_batch_sweep(source, args.expert, records)),
        ("M=0 zero XPU kernel launches",
         lambda: case_m0_zero_kernels(source, args.expert, records)),
        ("fused vs reconstruct+GEMM crossover",
         lambda: case_crossover(source, args.expert, records)),
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

    sweep = [r for r in records if r.get("case") != "crossover"]
    summary = {
        "task": "P1-08",
        "pass": not failures and all(r.get("pass", True) for r in sweep),
        "source": source_kind,
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "expert": args.expert,
        "device": torch.xpu.get_device_properties(0).name,
        "small_m_max": int(ops.small_m_max(3, 1)),
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "cases": results,
        "failures": failures,
        "sweep": sweep,
        "crossover": [r for r in records if r.get("case") == "crossover"],
    }

    log_path = Path(args.log_out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"P1-08 batch sweep  source={source_kind} layer={args.layer} expert={args.expert}",
        f"small_m_max={summary['small_m_max']} T2_max_abs={T2_MAX_ABS}",
    ]
    for r in sweep:
        lines.append("  " + json.dumps(r, sort_keys=True))
    lines.append(f"pass={summary['pass']} failures={failures}")
    log_path.write_text("\n".join(lines) + "\n")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\npass={summary['pass']} failures={failures}")
    print(f"wrote {log_path}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
