#!/usr/bin/env python3
"""P1-09: reentrancy and reconstruction-workspace scratch audit.

Audits the large-M reconstruction path at ``RECON_SLICE_N=1024`` and proves:

1. **Small-M never materializes FP16 weights.** For M<=8 the XPU profiler shows
   only ``HadInKernel`` + ``GemvKernel`` + ``HadOutKernel``; there is no
   ``ReconstructKernel`` and no GEMM. Peak device memory is the bounded
   split-K partial buffer, not a weight copy.
2. **Large-M workspace is bounded, not persistent.** With
   ``RECON_SLICE_N=1024`` the reconstruct slice is ``k*1024`` fp16 (<= 4 MiB at
   k=2048); repeated calls do not grow the peak, and the slice is freed after
   the call.
3. **Reentrant across in-flight streams.** The old per-device static scratch
   was replaced by a per-call stream-ordered allocation
   (``at::empty({k, n1-n0})``), so concurrent calls on two XPU streams return
   the same bits as serial execution.

    .venv/bin/python tests/test_reentrancy.py

Evidence: ``logs/moe-p1/reentrancy-audit.log``,
``logs/moe-p1/reentrancy-memory.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile

from exl3xpu import ops

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures.safetensors"
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"
RECON_SLICE_N = 1024
SMALL_M = 8
LARGE_M = 64
PEAK_SLACK_MIB = 1.0  # run-to-run allocator jitter


class ShardSource:
    def __init__(self, model_dir: Path, layer: int):
        self.model_dir = model_dir
        self.prefix = f"model.language_model.layers.{layer}.mlp.experts"
        self.weight_map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._handles: dict[str, object] = {}

    def has(self, expert: int) -> bool:
        key = f"{self.prefix}.{expert}.gate_proj.trellis"
        shard = self.weight_map.get(key)
        return shard is not None and (self.model_dir / shard).is_file()

    def get(self, expert: int, proj: str, suffix: str) -> torch.Tensor:
        key = f"{self.prefix}.{expert}.{proj}.{suffix}"
        if key not in self.weight_map:
            raise KeyError(key)
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.model_dir / shard), framework="pt")
        return self._handles[shard].get_tensor(key)

    def codebook(self, expert: int, proj: str) -> int:
        for suffix, cb in (("mcg", 1), ("mul1", 2)):
            if f"{self.prefix}.{expert}.{proj}.{suffix}" in self.weight_map:
                return cb
        return 0


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
        for suffix, cb in (("mcg", 1), ("mul1", 2)):
            if f"{self.prefix}.{expert}.{proj}.{suffix}" in self._keys:
                return cb
        return 0


def load_module(source, expert: int, proj: str) -> dict:
    tr = source.get(expert, proj, "trellis")
    n = tr.shape[1] * 16
    return {
        "trellis": tr.to("xpu"),
        "suh": source.get(expert, proj, "suh").view(1, -1).to("xpu"),
        "svh": source.get(expert, proj, "svh").to("xpu"),
        "nb": torch.zeros(n // 128, dtype=torch.int32, device="xpu"),
        "k": tr.shape[0] * 16,
        "n": n,
        "K": tr.shape[-1] // 16,
        "cb": source.codebook(expert, proj),
    }


def lin(x: torch.Tensor, m: dict) -> torch.Tensor:
    return ops.exl3_linear(
        x, m["trellis"], m["suh"], m["svh"], m["nb"], [0, m["n"]], m["K"], m["cb"]
    )


def kernel_rows(fn: Callable[[], object]) -> list[dict]:
    fn()
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        fn()
        torch.xpu.synchronize()
    return [
        {"name": r.key, "count": int(r.count), "device_us": float(r.self_device_time_total)}
        for r in prof.key_averages()
        if getattr(r, "self_device_time_total", 0) > 0
    ]


def peak_delta_mib(fn: Callable[[], object], iters: int = 20) -> dict[str, float]:
    fn()
    torch.xpu.synchronize()
    base = torch.xpu.memory_allocated()
    torch.xpu.reset_peak_memory_stats()
    torch.xpu.synchronize()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    peak = torch.xpu.max_memory_allocated()
    after = torch.xpu.memory_allocated()
    return {
        "baseline_mib": base / 2**20,
        "peak_mib": peak / 2**20,
        "delta_mib": (peak - base) / 2**20,
        "after_mib": after / 2**20,
        "persistent_mib": (after - base) / 2**20,
    }


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def case_small_m_no_weight_materialization(mods, records, mem_rows) -> None:
    assert ops.RECON_SLICE_N == RECON_SLICE_N
    for proj in ("gate_proj", "down_proj"):
        m = mods[proj]
        x = (torch.randn(SMALL_M, m["k"]) * 0.05).half().to("xpu")
        rows = kernel_rows(lambda: lin(x, m))
        names = [r["name"] for r in rows]
        has_recon = any("Reconstruct" in n for n in names)
        has_gemm = any("gemm" in n.lower() for n in names)
        has_gemv = any("Gemv" in n for n in names)
        mem = peak_delta_mib(lambda: lin(x, m), iters=40)
        rec = {
            "case": "small_m_no_weight_materialization",
            "proj": proj,
            "M": SMALL_M,
            "kernels": [{"name": r["name"].split("(")[0], "count": r["count"]} for r in rows],
            "reconstruct_kernel": has_recon,
            "gemm_kernel": has_gemm,
            "gemv_kernel": has_gemv,
            **mem,
            "pass": (not has_recon) and (not has_gemm) and has_gemv and mem["persistent_mib"] <= 0.5,
        }
        records.append(rec)
        mem_rows.append({**rec, "path": "fused_vector"})
        assert rec["pass"], rec


def case_large_m_workspace_bounded(mods, records, mem_rows) -> None:
    for proj in ("gate_proj", "down_proj"):
        m = mods[proj]
        x = (torch.randn(LARGE_M, m["k"]) * 0.05).half().to("xpu")
        rows = kernel_rows(lambda: lin(x, m))
        recon = [r for r in rows if "Reconstruct" in r["name"]]
        expected_slices = (m["n"] + RECON_SLICE_N - 1) // RECON_SLICE_N
        recon_count = sum(r["count"] for r in recon)
        mem1 = peak_delta_mib(lambda: lin(x, m), iters=1)
        mem40 = peak_delta_mib(lambda: lin(x, m), iters=40)
        # Slice bound: k*RECON_SLICE_N*2 bytes, plus xh/y; slack for overlapping
        # slices and allocator padding.
        slice_mib = m["k"] * min(RECON_SLICE_N, m["n"]) * 2 / 2**20
        bound = slice_mib + (m["k"] * LARGE_M * 2 + LARGE_M * m["n"] * 2) / 2**20 + PEAK_SLACK_MIB
        no_growth = mem40["delta_mib"] <= mem1["delta_mib"] + PEAK_SLACK_MIB
        rec = {
            "case": "large_m_workspace_bounded",
            "proj": proj,
            "M": LARGE_M,
            "recon_slice_n": RECON_SLICE_N,
            "reconstruct_launches": recon_count,
            "expected_slices": expected_slices,
            "slice_mib": slice_mib,
            "bound_mib": bound,
            "peak_1_call_delta_mib": mem1["delta_mib"],
            "peak_40_calls_delta_mib": mem40["delta_mib"],
            "persistent_mib": mem40["persistent_mib"],
            "pass": (
                len(recon) > 0
                and recon_count == expected_slices
                and mem40["delta_mib"] <= bound
                and no_growth
                and mem40["persistent_mib"] <= 0.5
            ),
        }
        records.append(rec)
        mem_rows.append({**rec, "path": "reconstruct_gemm"})
        assert rec["pass"], rec


def case_concurrent_streams_reentrant(mods, records) -> None:
    """Two XPU streams must match serial execution (no shared scratch race)."""
    gate = mods["gate_proj"]
    g = torch.Generator().manual_seed(7)
    x1 = (torch.randn(LARGE_M, gate["k"], generator=g) * 0.05).half().to("xpu")
    x2 = (torch.randn(LARGE_M, gate["k"], generator=g) * 0.05).half().to("xpu")
    xs = (torch.randn(SMALL_M, gate["k"], generator=g) * 0.05).half().to("xpu")

    y1 = lin(x1, gate)
    y2 = lin(x2, gate)
    ys = lin(xs, gate)
    torch.xpu.synchronize()

    s1, s2 = torch.xpu.Stream(), torch.xpu.Stream()
    mismatches = 0
    trials = 100
    for _ in range(trials):
        with torch.xpu.stream(s1):
            a = lin(x1, gate)
        with torch.xpu.stream(s2):
            b = lin(x2, gate)
        with torch.xpu.stream(s1):
            c = lin(xs, gate)
        torch.xpu.synchronize()
        if not (torch.equal(a, y1) and torch.equal(b, y2) and torch.equal(c, ys)):
            mismatches += 1
    rec = {
        "case": "concurrent_streams_reentrant",
        "trials": trials,
        "streams": 2,
        "mismatches": mismatches,
        "pass": mismatches == 0,
    }
    records.append(rec)
    assert rec["pass"], rec


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def make_source(model_dir: Path, layer: int):
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        src = ShardSource(model_dir, layer)
        if src.has(0):
            return src, "checkpoint"
    if layer == 0:
        src = FixtureSource(layer)
        if src.has(0):
            return src, "fixtures"
    raise SystemExit(f"no EXL3 expert tensors available ({model_dir}, layer {layer})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--log-out", default=str(ROOT / "logs/moe-p1/reentrancy-audit.log"))
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p1/reentrancy-audit.json"))
    ap.add_argument("--csv-out", default=str(ROOT / "logs/moe-p1/reentrancy-memory.csv"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("P1-09 needs the XPU")
    assert bool(ops._get_esimd().exl3_supported(3, 1))
    ops.RECON_SLICE_N = RECON_SLICE_N

    source, source_kind = make_source(Path(args.model_dir), args.layer)
    mods = {p: load_module(source, args.expert, p) for p in ("gate_proj", "down_proj")}
    records: list[dict] = []
    mem_rows: list[dict] = []
    failures: list[str] = []

    cases: list[tuple[str, Callable[[], None]]] = [
        ("small-M materializes no FP16 weights",
         lambda: case_small_m_no_weight_materialization(mods, records, mem_rows)),
        ("large-M workspace bounded at slice=1024",
         lambda: case_large_m_workspace_bounded(mods, records, mem_rows)),
        ("concurrent streams bit-identical to serial",
         lambda: case_concurrent_streams_reentrant(mods, records)),
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
        "task": "P1-09",
        "pass": not failures,
        "source": source_kind,
        "model_dir": str(args.model_dir),
        "device": torch.xpu.get_device_properties(0).name,
        "recon_slice_n": RECON_SLICE_N,
        "workspace": "per-call stream-ordered at::empty (no static scratch)",
        "cases": results,
        "failures": failures,
        "records": records,
    }

    log_path = Path(args.log_out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"P1-09 reentrancy audit  source={source_kind} RECON_SLICE_N={RECON_SLICE_N}",
        "workspace: per-call stream-ordered at::empty (static per-device scratch removed)",
    ]
    for r in records:
        lines.append("  " + json.dumps(r, sort_keys=True))
    lines.append(f"pass={summary['pass']} failures={failures}")
    log_path.write_text("\n".join(lines) + "\n")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["case", "path", "proj", "M", "baseline_mib", "peak_mib", "delta_mib",
                  "after_mib", "persistent_mib", "slice_mib", "bound_mib", "pass"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in mem_rows:
                w.writerow(row)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\npass={summary['pass']} failures={failures}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
