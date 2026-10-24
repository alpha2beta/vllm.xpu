#!/usr/bin/env python3
"""P1-06: exact weight reconstruction for representative experts.

Reconstructs full Hadamard-domain weights for experts ``{0, 1, 127, 254, 255}``
across ``gate``/``up``/``down`` with the **C++ dequantizer**
(``exl3xpu_C::exl3_reconstruct``) and compares them **bit-for-bit** against the
deterministic CPU oracle (:func:`exl3xpu.ref.reconstruct_inner`, the same
bit-exact dequantizer that backs the P0-08 traces).

Runs as a plain script (this venv has no pytest):

    .venv/bin/python tests/test_expert_reconstruction.py \
        --model-dir /home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw

Sources (auto-detected):
  * real checkpoint shards via ``model.safetensors.index.json`` when
    ``--model-dir`` exists (all five experts);
  * otherwise the layer-0 fixtures ``fixtures/exl3-moe/moe-fixtures.safetensors``
    (experts 0 and 255 only), so the harness is usable pre-download.

Covered contract:
  * C++ ``exl3_reconstruct`` == CPU oracle, zero differing fp16 elements, for
    both ``w13`` (k=2048, n=512) and ``w2`` (k=512, n=2048) tile geometry;
  * the ``n0`` column-offset path reconstructs the requested slice exactly;
  * ``suh``/``svh`` lengths equal ``k``/``n`` and the checkpoint codebook agrees
    with the reconstructed ``cb``;
  * for experts covered by the P0-08 traces (0, 255), the C++-reconstructed
    weights reproduce the recorded CPU-oracle linear outputs.

Evidence: ``logs/moe-p1/reconstruction-audit.log`` (+ ``--json-out``).
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from exl3xpu import ops
from exl3xpu import ref

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures.safetensors"
TRACES = ROOT / "fixtures" / "reference_moe_traces.npz"
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_EXPERTS = (0, 1, 127, 254, 255)


# ---------------------------------------------------------------------------
# tensor sources
# ---------------------------------------------------------------------------

class ShardSource:
    """Reads EXL3 tensors straight from the real checkpoint shards."""

    def __init__(self, model_dir: Path, layer: int):
        self.model_dir = model_dir
        self.layer = layer
        self.prefix = f"model.language_model.layers.{layer}.mlp.experts"
        index = model_dir / "model.safetensors.index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"]
        self._handles: dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                str(self.model_dir / shard), framework="pt"
            )
        return self._handles[shard]

    def has(self, expert: int, proj: str) -> bool:
        key = f"{self.prefix}.{expert}.{proj}.trellis"
        shard = self.weight_map.get(key)
        return shard is not None and (self.model_dir / shard).is_file()

    def get(self, expert: int, proj: str, suffix: str) -> torch.Tensor:
        key = f"{self.prefix}.{expert}.{proj}.{suffix}"
        return self._handle(self.weight_map[key]).get_tensor(key)

    def codebook(self, expert: int, proj: str) -> int:
        if self.has(expert, proj):
            if f"{self.prefix}.{expert}.{proj}.mcg" in self.weight_map:
                return ref.CB_MCG
            if f"{self.prefix}.{expert}.{proj}.mul1" in self.weight_map:
                return ref.CB_MUL1
        return ref.CB_3INST


class FixtureSource:
    """Fallback: layer-0 fixtures (experts 0/255 only)."""

    def __init__(self, layer: int):
        self.layer = layer
        self.prefix = f"model.language_model.layers.{layer}.mlp.experts"
        self._f = safe_open(str(FIXTURES), framework="pt")
        self._keys = set(self._f.keys())

    def has(self, expert: int, proj: str) -> bool:
        return f"{self.prefix}.{expert}.{proj}.trellis" in self._keys

    def get(self, expert: int, proj: str, suffix: str) -> torch.Tensor:
        return self._f.get_tensor(f"{self.prefix}.{expert}.{proj}.{suffix}")

    def codebook(self, expert: int, proj: str) -> int:
        if f"{self.prefix}.{expert}.{proj}.mcg" in self._keys:
            return ref.CB_MCG
        if f"{self.prefix}.{expert}.{proj}.mul1" in self._keys:
            return ref.CB_MUL1
        return ref.CB_3INST


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def cpp_reconstruct(trellis: torch.Tensor, K: int, cb: int, n0: int, n_slice: int) -> torch.Tensor:
    """C++ dequantizer for columns ``[n0, n0 + n_slice)`` of the inner weight."""
    k = trellis.shape[0] * 16
    w = torch.empty((k, n_slice), dtype=torch.float16, device=trellis.device)
    ops._get_esimd().exl3_reconstruct(trellis.contiguous(), w, n0, K, cb)
    return w


def case_reconstruct_bit_exact(source, experts, records: list[dict]) -> None:
    """C++ reconstruct == CPU oracle, full width, for every expert/projection."""
    for expert in experts:
        for proj in PROJECTIONS:
            if not source.has(expert, proj):
                records.append({"expert": expert, "proj": proj, "status": "missing"})
                continue
            trellis = source.get(expert, proj, "trellis")
            suh = source.get(expert, proj, "suh")
            svh = source.get(expert, proj, "svh")
            k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
            K = trellis.shape[-1] // 16
            cb = source.codebook(expert, proj)

            assert trellis.dtype == torch.int16, (expert, proj, trellis.dtype)
            assert suh.numel() == k, (expert, proj, "suh", suh.numel(), k)
            assert svh.numel() == n, (expert, proj, "svh", svh.numel(), n)
            assert K == 3, (expert, proj, "unexpected K", K)
            assert cb == ref.CB_MCG, (expert, proj, "unexpected codebook", cb)

            oracle = ref.reconstruct_inner(trellis.cpu(), K, cb).to(torch.float16)
            got = cpp_reconstruct(trellis.to("xpu"), K, cb, 0, n).cpu()

            assert got.shape == oracle.shape == (k, n), (got.shape, oracle.shape)
            n_diff = int((got != oracle).sum().item())
            max_abs = float((got.float() - oracle.float()).abs().max().item())
            exact = bool(torch.equal(got, oracle))
            records.append({
                "expert": expert, "proj": proj, "status": "ok" if exact else "FAIL",
                "k": k, "n": n, "K": K, "cb": cb,
                "elements": got.numel(), "differing": n_diff, "max_abs": max_abs,
                "bit_exact": exact,
            })
            assert exact, f"expert {expert} {proj}: {n_diff} fp16 elements differ"


def case_reconstruct_sliced(source, experts, records: list[dict]) -> None:
    """The ``n0`` column-offset path reconstructs the requested slice exactly."""
    # (expert, proj) is chosen so both w13 (n=512) and w2 (n=2048) are covered.
    checks = [(experts[0], "gate_proj"), (experts[-1], "down_proj")]
    for expert, proj in checks:
        if not source.has(expert, proj):
            continue
        trellis = source.get(expert, proj, "trellis")
        n = trellis.shape[1] * 16
        K = trellis.shape[-1] // 16
        cb = source.codebook(expert, proj)
        oracle = ref.reconstruct_inner(trellis.cpu(), K, cb).to(torch.float16)
        n_slice = 256 if n <= 512 else 512
        for n0 in (128, max(0, n - n_slice)):
            if n0 % 128 or n0 + n_slice > n:
                continue
            got = cpp_reconstruct(trellis.to("xpu"), K, cb, n0, n_slice).cpu()
            ref_slice = oracle[:, n0:n0 + n_slice]
            exact = bool(torch.equal(got, ref_slice))
            records.append({
                "expert": expert, "proj": proj, "status": "ok" if exact else "FAIL",
                "slice": [n0, n0 + n_slice],
                "differing": int((got != ref_slice).sum().item()),
                "bit_exact": exact,
            })
            assert exact, f"expert {expert} {proj} slice [{n0}:{n0 + n_slice}] differs"


def case_trace_crosscheck(source, records: list[dict]) -> None:
    """C++-reconstructed weights reproduce the P0-08 CPU-oracle linear outputs."""
    if not TRACES.is_file():
        records.append({"status": "missing", "note": "no reference_moe_traces.npz"})
        return
    data = np.load(TRACES)
    if not all(f"expert/{e}/gate_proj/y" in data for e in (0, 255)):
        records.append({"status": "missing", "note": "traces lack experts 0/255"})
        return
    x = torch.from_numpy(data["inputs/x"].copy()).float()
    for expert in (0, 255):
        if not source.has(expert, "gate_proj"):
            continue
        for proj in PROJECTIONS:
            trellis = source.get(expert, proj, "trellis")
            suh = source.get(expert, proj, "suh").float()
            svh = source.get(expert, proj, "svh").float()
            K = trellis.shape[-1] // 16
            cb = source.codebook(expert, proj)
            w = cpp_reconstruct(trellis.to("xpu"), K, cb, 0, trellis.shape[1] * 16).cpu()
            xin = x if proj != "down_proj" else torch.from_numpy(
                data[f"expert/{expert}/act"].copy()).float()
            xh = ref.had_rows(xin * suh)
            y = ref.had_rows(xh @ w.float()) * svh
            ref_y = torch.from_numpy(data[f"expert/{expert}/{proj}/y"].copy()).float()
            max_abs = float((y - ref_y).abs().max().item())
            # The reconstructed weights are bit-exact; this last leg is a CPU
            # fp32 matmul against the stored trace, so BLAS reassociation can
            # move the result by a few ULP. Require a tight bound, not 0.
            ok = max_abs <= 1e-5
            records.append({
                "expert": expert, "proj": proj, "status": "ok" if ok else "FAIL",
                "trace_max_abs": max_abs,
            })
            assert ok, f"expert {expert} {proj}: trace cross-check max abs {max_abs}"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def make_source(model_dir: Path, layer: int):
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        src = ShardSource(model_dir, layer)
        present = [e for e in DEFAULT_EXPERTS if src.has(e, "gate_proj")]
        if present:
            return src, "checkpoint"
    if layer == 0:
        src = FixtureSource(layer)
        if any(src.has(e, "gate_proj") for e in DEFAULT_EXPERTS):
            return src, "fixtures"
    raise SystemExit(
        f"no EXL3 expert tensors available (model-dir={model_dir}, layer={layer}); "
        "download the checkpoint shards or use --layer 0 fixtures"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, nargs="*", default=list(DEFAULT_EXPERTS))
    ap.add_argument("--log-out", default=str(ROOT / "logs/moe-p1/reconstruction-audit.log"))
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p1/reconstruction-audit.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("P1-06 needs the XPU (C++ dequantizer); none available")
    esimd = ops._get_esimd()
    assert bool(esimd.exl3_supported(3, 1)), "C++ dequantizer lacks (K=3, cb=1)"

    source, source_kind = make_source(Path(args.model_dir), args.layer)
    experts = [e for e in args.experts if source.has(e, "gate_proj")]
    records: list[dict] = []
    failures: list[str] = []

    cases: list[tuple[str, Callable[[], None]]] = [
        ("reconstruct bit-exact (C++ vs oracle)",
         lambda: case_reconstruct_bit_exact(source, experts, records)),
        ("reconstruct n0-slice path",
         lambda: case_reconstruct_sliced(source, experts, records)),
        ("P0-08 trace cross-check",
         lambda: case_trace_crosscheck(source, records)),
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

    exact = [r for r in records if r.get("bit_exact") is True]
    summary = {
        "task": "P1-06",
        "pass": not failures,
        "source": source_kind,
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "experts": experts,
        "device": torch.xpu.get_device_properties(0).name,
        "cases": results,
        "failures": failures,
        "reconstructions": records,
        "bit_exact_count": len(exact),
    }

    log_path = Path(args.log_out)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"P1-06 reconstruction audit  source={source_kind} layer={args.layer}",
        f"experts={experts}",
        f"reconstructions={len(records)} bit_exact={len(exact)}",
    ]
    for r in records:
        lines.append("  " + json.dumps(r, sort_keys=True))
    lines.append(f"pass={summary['pass']} failures={failures}")
    log_path.write_text("\n".join(lines) + "\n")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{len(records)} reconstructions, {len(exact)} bit-exact; "
          f"wrote {log_path}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
