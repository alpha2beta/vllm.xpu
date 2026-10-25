#!/usr/bin/env python3
"""Deterministic CPU EXL3 MoE oracle (P0-08).

Builds reference traces from *real* 3.09bpw checkpoint tensors (see
``scripts/fetch_exl3_fixtures.py``) using the bit-exact dequantizer in
``exl3xpu.ref``. Every trace is fp32 on CPU, seeded, and checksummed, so kernel
and layer tests can compare against it without a CUDA node.

Traces produced (``fixtures/reference_moe_traces.npz``):

  inputs/x                        [4, 2048] fp16 deterministic activations
  expert/{id}/{proj}/y            routed-expert linear outputs, fp32
  shared/{proj}/y                 shared-expert linear outputs, fp32
  router/logits                   [4, 256] fp32 router logits
  router/topk_ids, router/topk_weights  [4, 8] softmax-renormalised top-8
  block/y                         [4, 2048] MoE block output with *synthetic*
                                  restricted routing over experts {0, 255}
                                  (all other expert ids treated as absent), so
                                  a block-level test can run before the full
                                  12 GB of expert weights is on the XPU.

The activated-expert math follows vLLM's ``Qwen3NextSparseMoeBlock``:
``y = sum_i w_i * down(silu(gate(x)) * up(x)) + sigmoid(sg . x) * shared(x)``
with the routed weights renormalised over the top-8.

Usage:
  .venv/bin/python tests/ref_moe_oracle.py
  .venv/bin/python tests/ref_moe_oracle.py --verify   # re-run and diff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures.safetensors"
MANIFEST = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures-manifest.json"
OUT = ROOT / "fixtures" / "reference_moe_traces.npz"
LAYER = 0
EXPERTS = (0, 255)
BATCH = 4


def load_module(tensors: dict, base: str) -> dict:
    trellis = torch.from_numpy(tensors[f"{base}.trellis"].copy())
    K = trellis.shape[-1] // 16
    cb = 1 if f"{base}.mcg" in tensors else (2 if f"{base}.mul1" in tensors else 0)
    return {
        "trellis": trellis,
        "suh": torch.from_numpy(tensors[f"{base}.suh"].copy()),
        "svh": torch.from_numpy(tensors[f"{base}.svh"].copy()),
        "K": K,
        "cb": cb,
    }


def forward(module: dict, x: torch.Tensor) -> torch.Tensor:
    from exl3xpu import ref

    return ref.linear_forward(
        x, module["trellis"], module["suh"], module["svh"], module["K"], module["cb"]
    )


def build(cache: Path, fixtures: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    tensors = load_file(str(fixtures))
    prefix = f"model.language_model.layers.{LAYER}.mlp"

    rng = np.random.default_rng(20261023)
    x = (rng.standard_normal((BATCH, 2048)) * 0.05).astype(np.float16)
    xf = torch.from_numpy(x.copy()).float()

    traces: dict[str, np.ndarray] = {"inputs/x": x}

    # --- single-expert linear traces ---------------------------------------
    for expert in EXPERTS:
        modules = {
            proj: load_module(tensors, f"{prefix}.experts.{expert}.{proj}")
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        gate = forward(modules["gate_proj"], xf)
        up = forward(modules["up_proj"], xf)
        act = torch.nn.functional.silu(gate) * up
        traces[f"expert/{expert}/gate_proj/y"] = gate.numpy()
        traces[f"expert/{expert}/up_proj/y"] = up.numpy()
        traces[f"expert/{expert}/down_proj/y"] = forward(
            modules["down_proj"], act
        ).numpy()
        traces[f"expert/{expert}/act"] = act.numpy()
        for proj, module in modules.items():
            traces[f"meta/expert/{expert}/{proj}"] = np.array(
                [
                    module["K"],
                    module["cb"],
                    module["trellis"].shape[0] * 16,
                    module["trellis"].shape[1] * 16,
                ],
                dtype=np.int32,
            )

    # --- shared expert ------------------------------------------------------
    shared = {
        proj: load_module(tensors, f"{prefix}.shared_expert.{proj}")
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    shared_gate_h = forward(shared["gate_proj"], xf)
    shared_up_h = forward(shared["up_proj"], xf)
    traces["shared/gate_proj/y"] = shared_gate_h.numpy()
    traces["shared/up_proj/y"] = shared_up_h.numpy()

    shared_gate = torch.from_numpy(
        tensors[f"{prefix}.shared_expert_gate.weight"].copy()
    ).float()
    activated = torch.nn.functional.silu(shared_gate_h) * shared_up_h
    traces["shared/act"] = activated.numpy()
    shared_h = forward(shared["down_proj"], activated)
    traces["shared/down_proj/y"] = shared_h.numpy()
    shared_scalar = torch.sigmoid(shared_gate @ xf.T).T  # [B, 1]
    traces["shared/y"] = (shared_h * shared_scalar).numpy()
    traces["shared/gate_scalar"] = shared_scalar.numpy()

    # --- router -------------------------------------------------------------
    router = torch.from_numpy(tensors[f"{prefix}.gate.weight"].copy()).float()
    logits = xf @ router.T  # [B, 256], fp32
    traces["router/logits"] = logits.numpy()
    top_w, top_i = torch.topk(logits, 8, dim=-1)
    top_w = torch.softmax(top_w.float(), dim=-1)
    traces["router/topk_ids"] = top_i.numpy().astype(np.int32)
    traces["router/topk_weights"] = top_w.numpy()

    # --- block output with routing restricted to the fetched experts --------
    # Synthetic fixture: only ids 0 and 255 exist locally, so take the best 8
    # of {0, 255} (alternating by layer-position parity would be arbitrary);
    # instead give both experts the top-2 router weight share and blend.
    available = torch.tensor(EXPERTS)
    sub_logits = logits[:, available]  # [B, 2]
    sub_w = torch.softmax(sub_logits, dim=-1)
    block = torch.zeros_like(xf)
    for j, expert in enumerate(EXPERTS):
        g = torch.from_numpy(traces[f"expert/{expert}/gate_proj/y"])
        u = torch.from_numpy(traces[f"expert/{expert}/up_proj/y"])
        d = forward(load_module(tensors, f"{prefix}.experts.{expert}.down_proj"),
                    torch.nn.functional.silu(g) * u)
        block += sub_w[:, j : j + 1] * d
    block += torch.from_numpy(traces["shared/y"])
    traces["block/y"] = block.numpy()
    traces["block/routing_weights"] = sub_w.numpy()

    # --- checksum manifest --------------------------------------------------
    checks = {}
    for key, value in traces.items():
        checks[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    (cache.parent / "reference_moe_traces.checksums.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    np.savez_compressed(cache, **traces)
    return traces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if not args.fixtures.exists():
        raise SystemExit(
            f"missing fixtures: {args.fixtures}\n"
            "run scripts/fetch_exl3_fixtures.py first"
        )

    traces = build(args.out, args.fixtures)
    total = sum(v.nbytes for v in traces.values())
    print(f"wrote {args.out} ({len(traces)} arrays, {total / 2**20:.1f} MiB uncompressed)")

    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        print(
            f"fixtures from {manifest['repo']}@{manifest['revision']} "
            f"commit {manifest.get('commit')}"
        )

    if args.verify:
        again = build(args.out.with_suffix(".verify.npz"), args.fixtures)
        bad = []
        for key, value in traces.items():
            if not np.array_equal(value, again[key]):
                bad.append(key)
        print(f"verify: {len(traces) - len(bad)}/{len(traces)} arrays identical")
        if bad:
            print("differing arrays:", ", ".join(bad[:10]))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
