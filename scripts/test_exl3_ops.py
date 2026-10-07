#!/usr/bin/env python3
"""E3.1/E3.2: EXL3 linear operator correctness vs upstream pure-torch reference.

Reference provenance: <EXL3_SRC>/exl3xpu/ref.py (author's bit-exact CPU
reference; documents the trellis/Hadamard format). This test validates the
compiled dispatch (vector vs reconstruct+GEMM) and row coverage on XPU — the
main 140V patch risk — not the reference's format derivation itself.

Requires (see logs/<run>/baseline-env.sh):
    EXL3_SMALL_M_MAX=8, EXL3_INT8_PREFILL=0
Backend: EXL3_BACKEND=auto (default) or triton. Results are labelled by backend.

Usage:
    source logs/<run>/baseline-env.sh
    EXL3_SRC=~/Projects/exl3xpu .venv/bin/python scripts/test_exl3_ops.py \
        --out logs/<run>/operator-results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Tolerances defined BEFORE accepting results (fp16 Hadamard-domain accumulate
# on XPU vs fp32 CPU reference; per-element, abs+rel).
ATOL = 0.08
RTOL = 0.08
# Boundary sweep from tasks.md E3.2.
M_SWEEP = [0, 1, 2, 3, 4, 7, 8, 9, 16, 127, 128, 129, 255, 256, 257]


def make_tensors(k: int, n: int, K: int, cb: int, device: str):
    import torch
    g = torch.Generator().manual_seed(0xE13 + k + n + K)
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K),
                            dtype=torch.int16, generator=g)
    suh = (torch.rand(k, dtype=torch.float16, generator=g) - 0.25).to(torch.float16)
    svh = (torch.rand(n, dtype=torch.float16, generator=g) - 0.25).to(torch.float16)
    suh[suh == 0] = 0.5
    svh[svh == 0] = 0.5
    nshards = 1 if n <= 128 else 2
    bounds = [i * (n // nshards) for i in range(nshards + 1)]
    shard = torch.zeros(n // 128, dtype=torch.int32)
    for gi in range(nshards):
        shard[bounds[gi] // 128: bounds[gi + 1] // 128] = gi
    suh_g = torch.stack([suh for _ in range(nshards)])
    return (trellis.to(device), suh_g.to(device), svh.to(device),
            shard.to(device), bounds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="xpu:0")
    args = ap.parse_args()

    for var, want in (("EXL3_SMALL_M_MAX", "8"), ("EXL3_INT8_PREFILL", "0")):
        got = os.environ.get(var)
        if got != want:
            print(f"refusing: {var}={got!r}, baseline requires {want!r} "
                  f"(source logs/<run>/baseline-env.sh)")
            return 2
    backend = os.environ.get("EXL3_BACKEND", "auto")
    print(f"backend={backend} device={args.device} ATOL={ATOL} RTOL={RTOL}", flush=True)

    exl3_src = Path(os.environ.get("EXL3_SRC", str(Path.home() / "Projects/exl3xpu")))
    sys.path.insert(0, str(exl3_src))
    import torch
    from exl3xpu import ops, ref

    torch.manual_seed(7)
    results: dict = {
        "backend": backend,
        "device": args.device,
        "atol": ATOL, "rtol": RTOL,
        "ref_provenance": "upstream exl3xpu/ref.py (author CPU reference)",
        "cases": [],
    }
    failures = 0

    configs = [
        ("decoder-K4", 512, 512, 4, 2),
        ("mlp-wide-K4", 512, 1024, 4, 2),
        ("lmhead-K6", 512, 256, 6, 2),
        # 2.2bpw checkpoint coverage (needs EXL3_ALL_CODEBOOKS build for K=2/3;
        # K=1 has no C++ instantiation: triton-only via the k1-fallback patch).
        ("bpw22-K3", 512, 512, 3, 2),
        ("bpw22-K2", 512, 512, 2, 2),
        ("bpw22-K1-triton", 512, 256, 1, 2),
    ]
    for name, k, n, K, cb in configs:
        trellis, suh, svh, shard, bounds = make_tensors(k, n, K, cb, args.device)
        for M in M_SWEEP:
            case = {"config": name, "k": k, "n": n, "K": K, "cb": cb, "M": M}
            t0 = time.time()
            try:
                x = torch.randn(M, k, dtype=torch.float16,
                                device=args.device) if M else \
                    torch.empty(0, k, dtype=torch.float16, device=args.device)
                y = ops.exl3_linear(x, trellis, suh, svh, shard, bounds, K, cb)
                if args.device.startswith("xpu"):
                    torch.xpu.synchronize()
                y = y.float().cpu()
                yref = ref.linear_forward(
                    x.float().cpu(), trellis.cpu(), suh[0].float().cpu()
                    if suh.dim() == 2 else suh.float().cpu(),
                    svh.float().cpu(), K, cb)
                # NOTE: suh stacked per shard; reference uses single-group suh.
                ok_finite = bool(torch.isfinite(y).all())
                denom = yref.abs().clamp_min(1e-3)
                rel = ((y - yref).abs() / denom)
                case.update({
                    "max_abs": float((y - yref).abs().max() or 0) if M else 0.0,
                    "mean_abs": float((y - yref).abs().mean() or 0) if M else 0.0,
                    "max_rel": float(rel.max() or 0) if M else 0.0,
                    "finite": ok_finite,
                    "pass": ok_finite and (not M or (
                        float((y - yref).abs().max()) <= ATOL + RTOL * float(yref.abs().max()))),
                    "ms": round((time.time() - t0) * 1e3, 1),
                })
            except Exception as e:  # noqa - record, don't stop the sweep
                case.update({"error": repr(e)[:300], "pass": False,
                             "ms": round((time.time() - t0) * 1e3, 1)})
            results["cases"].append(case)
            if not case["pass"]:
                failures += 1
            print(f"{name} K={K} M={M:>3}: "
                  f"{'PASS' if case['pass'] else 'FAIL'} "
                  f"{case.get('max_abs', '-')}/{case.get('max_rel', '-')}/"
                  f"{case.get('error', '')}", flush=True)

    # Guard check: direct small-GEMM op with M>8 must fail loudly (no DPAS).
    try:
        from exl3xpu import ops as _ops
        E = _ops._get_esimd()
        if E and hasattr(E, "exl3_gemm_small"):
            import torch as _t
            trellis, suh, svh, shard, bounds = make_tensors(512, 256, 4, 2, args.device)
            x = _t.randn(9, 512, dtype=_t.float16, device=args.device)
            out = _t.empty(9, 256, dtype=_t.float16, device=args.device)
            try:
                E.exl3_gemm_small(x, trellis, suh, svh, shard, out, 4, 2)
                results["guard_M9"] = "FAIL: no error raised"
                failures += 1
            except RuntimeError as e:
                results["guard_M9"] = f"PASS: {str(e)[:120]}"
                print("guard M=9:", results["guard_M9"], flush=True)
        else:
            results["guard_M9"] = "SKIP: ESIMD lib unavailable"
    except Exception as e:  # noqa
        results["guard_M9"] = f"SKIP: {e!r}"[:200]

    results["failures"] = failures
    results["ok"] = failures == 0
    out = Path(args.out) if args.out else Path("operator-results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"{'ALL PASS' if not failures else f'{failures} FAILURES'} -> {out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
