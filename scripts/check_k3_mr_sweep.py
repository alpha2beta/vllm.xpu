#!/usr/bin/env python3
"""P0-06 follow-up: validate the (K=3, cb=1) ESIMD instantiation at every M
bucket used by the production path.

`scripts/check_exl3_backends.py` exercises M in {1,2,4,8} only. The plan's
routed-expert tensors are also sent through the *same* `exl3_gemm_small` entry
point for `SMALL_M_MAX >> M > 8` (per-MR vector kernel selection) and through
`exl3_reconstruct` + a dense GEMM above that. This script closes the middle
band, where the new `dispatch_mr<3,1>` cases are actually selected:

    M <= 8    -> dispatch_mr<3,1> with MR = 1/2/4/8
    9..128    -> dispatch_mr<3,1> with the next MR bucket (up to SMALL_M_MAX)
    > 128     -> exl3_reconstruct<3,1> + oneDNN/ATen GEMM

For each M the XPU result is compared against the CPU fp32 reference
(`exl3xpu.ref.linear_forward`) using the T-2 tolerance from
`docs/tolerance_spec.md` (max abs <= 0.08, cosine >= 0.999).

Usage:
  .venv/bin/python scripts/check_k3_mr_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_exl3_backends import FIXTURES, collect_modules, load_module  # noqa: E402

OUT_JSON = ROOT / "logs" / "moe-preflight" / "k3-mr-sweep.json"
OUT_LOG = ROOT / "logs" / "moe-preflight" / "k3-mr-sweep.log"

M_LIST = [1, 2, 3, 4, 8, 9, 16, 17, 24, 32, 33, 40, 48, 64, 96, 128]
TOL_ABS = 0.08
TOL_COS = 0.999


def main() -> int:
    from safetensors.numpy import load_file

    from exl3xpu import ops, ref

    tensors = load_file(str(FIXTURES))
    # One representative of each routed-expert direction.
    wanted = [
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.0.mlp.experts.0.down_proj",
    ]
    modules = [b for b in collect_modules(list(tensors)) if b in wanted]
    if len(modules) != len(wanted):
        print(f"fixtures missing: found {modules}", file=sys.stderr)
        return 2

    device = torch.device(os.environ.get("EXL3_DEVICE", "xpu:0"))
    esimd = ops._get_esimd()
    print(f"device: {torch.xpu.get_device_properties(0).name}")
    print(f"SMALL_M_MAX={ops.SMALL_M_MAX}  small-M limit K=3/mcg="
          f"{ops.small_m_max(3, 1)}  esimd K=3/cb=1 supported="
          f"{bool(esimd and esimd.exl3_supported(3, 1))}")

    reports, failures = [], []
    for base in modules:
        m = load_module(tensors, base)
        K, cb = m["K"], m["cb"]
        k, n = m["trellis"].shape[0] * 16, m["trellis"].shape[1] * 16
        short = base.replace("model.language_model.", "")
        trellis = m["trellis"].to(device)
        suh = m["suh"].to(device)
        svh = m["svh"].to(device)
        shard = torch.zeros(n // 128, dtype=torch.int32, device=device)
        w_ref = ref.reconstruct_inner(m["trellis"], K, cb)  # CPU, bit-exact oracle

        for M in M_LIST:
            torch.manual_seed(2000 + M)
            x = (torch.randn(M, k, dtype=torch.float32) * 0.05)
            xh = ref.had_rows(x * m["suh"].float())
            y_ref = ref.had_rows(xh @ w_ref.float()) * m["svh"].float()
            y = ops.exl3_linear(x.to(torch.float16).to(device), trellis, suh, svh,
                                shard, [0, n], K, cb).float().cpu()
            d = (y - y_ref).abs()
            max_abs = float(d.max())
            denom = float(torch.linalg.vector_norm(y_ref) * torch.linalg.vector_norm(y))
            cos = float((y * y_ref).sum() / denom) if denom else 1.0
            path = ("esimd" if bool(esimd and esimd.exl3_supported(K, cb))
                    and M <= ops.small_m_max(K, cb) else "recon+gemm")
            ok = max_abs <= TOL_ABS and cos >= TOL_COS
            reports.append({"module": short, "M": M, "K": K, "codebook": "mcg",
                            "path": path, "max_abs": max_abs, "cosine": cos, "ok": ok})
            if not ok:
                failures.append(f"{short} M={M}: max_abs={max_abs:.4g} cos={cos:.6f}")
            print(f"  {short:<42} M={M:>3} {path:<10} max_abs={max_abs:.2e} cos={cos:.6f} "
                  f"{'OK' if ok else 'FAIL'}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": torch.xpu.get_device_properties(0).name,
        "small_m_max": ops.SMALL_M_MAX,
        "small_m_max_k3_cb1": ops.small_m_max(3, 1),
        "tolerance": {"max_abs": TOL_ABS, "cosine": TOL_COS},
        "cases": reports,
        "failures": failures,
        "pass": not failures,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_LOG.write_text(
        f"K=3/mcg M-bucket sweep\nresults: {len(reports) - len(failures)}/{len(reports)} pass\n"
        + ("\n".join(failures) if failures else "no failures")
        + f"\njson: {OUT_JSON}\n"
    )
    print(f"\n{len(reports) - len(failures)}/{len(reports)} pass -> {OUT_JSON}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
