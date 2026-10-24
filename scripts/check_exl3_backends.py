#!/usr/bin/env python3
"""EXL3 backend parity check on real checkpoint fixtures (P0-04 / P0-06).

For every fetched MoE/attention module this compares, on the real 3.09bpw /
EXL3-0.0.22 payload:

  1. Trellis reconstruction in the Hadamard domain:
       exl3xpu.ref.reconstruct_inner   (CPU, bit-exact oracle)
       exl3xpu.triton_kernels.reconstruct (XPU Triton)
     -> expected bit-exact equality for the identical codebook/LUT.

  2. End-to-end linear at M in {1, 2, 4, 8}:
       exl3xpu.ops.exl3_linear         (XPU: ESIMD when supported, else Triton)
       exl3xpu.ref.linear_forward      (CPU fp32 reference)
     -> reports max abs / relative error against the P0-09 tolerances.

  3. The ESIMD (Level Zero / DPAS) support matrix for the (K, codebook) pairs
     present in this checkpoint, which decides whether the fast path is usable.

Writes ``logs/moe-preflight/exl3-backend-parity.{log,json}``.

Usage:
  .venv/bin/python scripts/check_exl3_backends.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures.safetensors"
DEFAULT_LOG_DIR = ROOT / "logs" / "moe-preflight"
CB_NAMES = {0: "3inst", 1: "mcg", 2: "mul1"}
MODULE_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\."
    r"(?:experts\.\d+|shared_expert)\.(gate_proj|up_proj|down_proj)$"
)


def collect_modules(keys: list[str]) -> list[str]:
    found = set()
    for key in keys:
        base, _, leaf = key.rpartition(".")
        if leaf in ("trellis", "suh", "svh", "mcg", "mul1"):
            if MODULE_RE.match(base):
                found.add(base)
    return sorted(found)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--recon-tolerance",
        type=int,
        default=0,
        help="allowed differing fp16 elements between CPU and Triton reconstruction",
    )
    args = parser.parse_args()

    from safetensors.numpy import load_file

    from exl3xpu import ops, ref, triton_kernels as tk

    tensors = load_file(str(args.fixtures))
    modules = collect_modules(list(tensors))
    device = torch.device(args.device)
    esimd = ops._get_esimd()

    print(f"fixtures: {args.fixtures}")
    print(f"modules:  {len(modules)}")
    print(f"device:   {torch.xpu.get_device_properties(0).name}")
    print()

    # --- ESIMD support matrix for every (K, codebook) in the checkpoint -------
    pairs = sorted({(load_module(tensors, b)["K"], load_module(tensors, b)["cb"]) for b in modules})
    matrix = [
        {
            "K": K,
            "codebook": CB_NAMES[cb],
            "cb": cb,
            "esimd_supported": bool(esimd and esimd.exl3_supported(K, cb)),
            "modules": sum(
                1
                for b in modules
                if (m := load_module(tensors, b))["K"] == K and m["cb"] == cb
            ),
        }
        for K, cb in pairs
    ]
    print("ESIMD support matrix (as built):")
    for row in matrix:
        flag = "yes" if row["esimd_supported"] else "NO  "
        print(f"  K={row['K']} cb={row['codebook']:<6} supported={flag} modules={row['modules']}")
    print()

    torch.manual_seed(args.seed)
    reports: list[dict] = []
    failures: list[str] = []

    for base in modules:
        module = load_module(tensors, base)
        K, cb = module["K"], module["cb"]
        k, n = module["trellis"].shape[0] * 16, module["trellis"].shape[1] * 16
        short = base.replace("model.language_model.", "")

        # 1. reconstruction parity (CPU oracle vs XPU Triton)
        w_ref = ref.reconstruct_inner(module["trellis"], K, cb).float()
        w_tri = tk.reconstruct(module["trellis"].to(device), K, cb).float().cpu()
        diff = (w_ref - w_tri).abs()
        n_diff = int((diff > 0).sum())
        recon_ok = n_diff <= args.recon_tolerance
        if not recon_ok:
            failures.append(f"{short}: reconstruction differs in {n_diff} elements")

        # 2. end-to-end linear parity
        linear = {"module": short, "K": K, "codebook": CB_NAMES[cb], "k": k, "n": n,
                  "esimd_path": bool(esimd and esimd.exl3_supported(K, cb)),
                  "recon_max_abs_diff": float(diff.max()),
                  "recon_differing_elements": n_diff, "batch": {}}
        g4 = module["trellis"].to(device)
        suh = module["suh"].to(device).unsqueeze(0)
        svh = module["svh"].to(device)
        shard = torch.zeros(n // 128, dtype=torch.int32, device=device)
        for M in args.batch_sizes:
            x = (torch.randn(M, k, dtype=torch.float32) * 0.05).to(torch.float16)
            y_ref = ref.linear_forward(x, module["trellis"], module["suh"], module["svh"], K, cb)
            y_xpu = ops.exl3_linear(
                x.to(device), g4, suh, svh, shard, [0, n], K, cb
            ).float().cpu()
            err = (y_ref - y_xpu).abs()
            scale = y_ref.abs().mean().clamp_min(1e-6)
            max_abs = float(err.max())
            # P0-09: absolute error <= 0.08, relative to output magnitude.
            ok = max_abs <= 0.08
            if not ok:
                failures.append(f"{short} M={M}: max abs err {max_abs:.4f} > 0.08")
            linear["batch"][str(M)] = {
                "max_abs_err": max_abs,
                "mean_abs_err": float(err.mean()),
                "rel_err_vs_mean_out": float(err.mean() / scale),
                "within_tolerance": ok,
            }
        reports.append(linear)
        worst = max(linear["batch"].values(), key=lambda v: v["max_abs_err"])
        print(
            f"  {short:<58} K={K} {CB_NAMES[cb]:<5} recon_diff={n_diff:<6} "
            f"worst_max_abs={worst['max_abs_err']:.5f} "
            f"{'OK' if recon_ok and all(b['within_tolerance'] for b in linear['batch'].values()) else 'FAIL'}"
        )

    # --- report ---------------------------------------------------------------
    args.log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fixtures": str(args.fixtures),
        "fixture_manifest": str(args.fixtures.with_name("moe-fixtures-manifest.json")),
        "device": torch.xpu.get_device_properties(0).name,
        "torch": torch.__version__,
        # Records how the backend was selected so a forced-ESIMD run is
        # self-describing: EXL3_BACKEND=esimd makes a library-load failure raise
        # instead of silently degrading to Triton (ops.py:45).
        "requested_backend": os.environ.get("EXL3_BACKEND", "auto"),
        "small_m_max": ops.SMALL_M_MAX,
        "esimd_library_loaded": bool(esimd),
        "esimd_support_matrix": matrix,
        "modules": reports,
        "failures": failures,
        "pass": not failures,
    }
    (args.log_dir / "exl3-backend-parity.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# EXL3 backend parity on Qwen3.5-35B-A3B 3.09bpw fixtures (P0-04 / P0-06)",
        f"fixtures: {args.fixtures}",
        f"device: {summary['device']}",
        f"torch: {summary['torch']}",
        f"modules checked: {len(reports)}",
        f"result: {'PASS' if summary['pass'] else 'FAIL'}",
        "",
        "## ESIMD support matrix (as-built _C.so)",
        f"{'K':>2}  {'codebook':<8} {'esimd':<6} modules",
    ]
    for row in matrix:
        lines.append(
            f"{row['K']:>2}  {row['codebook']:<8} "
            f"{'yes' if row['esimd_supported'] else 'NO':<6} {row['modules']}"
        )
    lines += ["", "## Per-module results", 
              "module,K,codebook,esimd_path,recon_diff,worst_max_abs_err,M"]
    for rep in reports:
        for M, stats in rep["batch"].items():
            lines.append(
                f"{rep['module']},{rep['K']},{rep['codebook']},"
                f"{'esimd' if rep['esimd_path'] else 'triton'},"
                f"{rep['recon_differing_elements']},{stats['max_abs_err']:.6f},{M}"
            )
    if failures:
        lines += ["", "## Failures", *[f"- {f}" for f in failures]]
    (args.log_dir / "exl3-backend-parity.log").write_text("\n".join(lines) + "\n")
    print()
    print(f"wrote {args.log_dir / 'exl3-backend-parity.log'}")
    print(f"wrote {args.log_dir / 'exl3-backend-parity.json'}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
