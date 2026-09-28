#!/usr/bin/env python3
"""Quantize Kwaipilot/KAT-Coder-V2.5-Dev (BF16, per-expert-split MoE) to the exact
MXFP4 safetensors schema of pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4 so the
vLLM XPU path (CompressedTensorsW4A4Mxfp4 + XPUExpertsMxFp4) loads it.

Schema (same as reference):
  - routed experts: per-expert, per-proj
      .../experts.E.{gate_proj,up_proj,down_proj}.weight_packed  U8 [out, in/2]
      .../experts.E.{gate_proj,up_proj,down_proj}.weight_scale   U8 [out, in/32]
    (KAT source experts are ALREADY per-expert 2D BF16 — quantize directly,
    no unfuse needed, unlike Tiel's fused layout)
  - linear_attn (30 layers): in_proj_qkv / in_proj_z / out_proj / in_proj_a /
    in_proj_b  ->  <name>.weight_packed + <name>.weight_scale (U8, shape-agnostic)
  - everything else (self_attn, norms, shared experts, embed, lm_head):
    copied unchanged (source is text-only: no vision, no MTP block)
  - config.json = KAT config + reference quantization_config
    with a KAT-adapted ignore list (every BF16 module)

Quantizer: torch.ops.vllm.xpu_mxfp4_quantize (the inference kernel package's
own quantizer — format match by construction). E2M1 group-32 symmetric,
E8M0 uint8 scales, low-nibble-even packing.

Usage:
  ./scripts/quantize_kat_mxfp4.py [--shards A-B]   (default 0-12, KAT numbering)
  Output: models/KAT-Coder-V2.5-Dev-MXFP4/
  Resume: completed source shards are recorded in
  models/KAT-Coder-V2.5-Dev-MXFP4/quantize_state.json
"""

import argparse
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(REPO_ROOT, "models", "_staging_kat")
OUTDIR = os.path.join(
    REPO_ROOT, "models", "KAT-Coder-V2.5-Dev-MXFP4")
SRC_REPO = "Kwaipilot/KAT-Coder-V2.5-Dev"
N_SHARDS = 13
SHARD_TARGET_BYTES = 1 << 30  # ~1 GiB output shards like the reference

LINEAR_ATTN_QUANT = ("in_proj_qkv", "in_proj_z", "out_proj",
                     "in_proj_a", "in_proj_b")

EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")


def quantize_mxfp4(w2d_xpu):
    """[rows, K] BF16 XPU tensor -> (packed U8 [rows, K/2], scale U8 [rows, K/32])."""
    assert w2d_xpu.dim() == 2 and w2d_xpu.size(1) % 32 == 0
    q, s = torch.ops.vllm.xpu_mxfp4_quantize(w2d_xpu)
    return q.view(torch.uint8).to("cpu"), s.view(torch.uint8).to("cpu")


def shard_name(i):
    return f"model-{i:05d}-of-{N_SHARDS:05d}.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=f"0-{N_SHARDS - 1}")
    ap.add_argument("--validate-every", type=int, default=64,
                    help="dequant-validate 1 in N expert-projs")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.shards.split("-"))

    import vllm_xpu_kernels  # noqa: F401 (registers torch.ops._xpu_C.*)
    import vllm._xpu_ops  # noqa: F401 (registers torch.ops.vllm.*)
    from vllm_xpu_kernels.moe_utils import dequant_mxfp4
    assert torch.xpu.is_available(), "XPU required for xpu_mxfp4_quantize"

    # source index: local copy if present, else fetch from HF once
    src_index_path = os.path.join(STAGING, "kat_source_index.json")
    if not os.path.exists(src_index_path):
        from huggingface_hub import hf_hub_download
        os.makedirs(STAGING, exist_ok=True)
        print("fetching source index ...", flush=True)
        p = hf_hub_download(SRC_REPO, "model.safetensors.index.json",
                            local_dir=STAGING)
        os.rename(p, src_index_path)
    idx = json.load(open(src_index_path))["weight_map"]
    by_shard = {}
    for k, f in idx.items():
        by_shard.setdefault(f, []).append(k)
    os.makedirs(OUTDIR, exist_ok=True)

    state_path = os.path.join(OUTDIR, "quantize_state.json")
    state = {"done_shards": [], "out_shard_no": 1, "weight_map": {},
             "total_out_bytes": 0}
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"resume: {len(state['done_shards'])} source shards done, "
              f"next output shard #{state['out_shard_no']}")

    pending, pending_bytes = [], 0
    n_q, n_val, max_rel, sum_rel = 0, 0, 0.0, 0.0

    def save_state():
        with open(state_path, "w") as f:
            json.dump(state, f)

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        n = state["out_shard_no"]
        fn = f"model-{n:05d}-of-XXXXX.safetensors"
        tensors = {k: v.contiguous() for k, v in pending}
        save_file(tensors, os.path.join(OUTDIR, fn))
        for k in tensors:
            state["weight_map"][k] = fn
        state["total_out_bytes"] += pending_bytes
        state["out_shard_no"] += 1
        print(f"  wrote {fn}: {len(tensors)} tensors, "
              f"{pending_bytes / 2**30:.2f} GiB", flush=True)
        pending, pending_bytes = [], 0
        save_state()

    def emit(name, t):
        nonlocal pending_bytes
        pending.append((name, t))
        pending_bytes += t.nelement() * t.element_size()
        if pending_bytes >= SHARD_TARGET_BYTES:
            flush()

    def validate(src2d_cpu, q_cpu, s_cpu, tag):
        """Dequantize packed pair on XPU, compare vs BF16 source."""
        nonlocal n_val, max_rel, sum_rel
        n_val += 1
        with torch.no_grad():
            qr = q_cpu.to("xpu").view(torch.float4_e2m1fn_x2)
            sr = s_cpu.to("xpu").view(torch.float8_e8m0fnu)
            dq = dequant_mxfp4(qr, sr).float().cpu()
        ref = src2d_cpu.float()
        rel = (dq - ref).abs().mean().item() / ref.abs().mean().item()
        max_rel = max(max_rel, rel)
        sum_rel += rel
        if n_val <= 3 or n_val % 200 == 0:
            print(f"  [val] {tag}: mean-rel-err={rel:.5f}", flush=True)

    def is_expert_proj(name):
        # ...mlp.experts.{E}.{gate,up,down}_proj.weight (already per-expert 2D)
        m = re.search(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$",
                      name)
        return m.group(1) if m else None

    t0 = time.time()
    for si in range(lo, hi + 1):
        src = shard_name(si)
        if src in state["done_shards"]:
            print(f"[{si}/{N_SHARDS - 1}] {src}: SKIP (done)")
            continue
        path = os.path.join(STAGING, src)
        if not os.path.exists(path):
            print(f"MISSING source shard (download first): {src}")
            sys.exit(2)
        names = by_shard[src]
        print(f"[{si}/{N_SHARDS - 1}] {src}: {len(names)} tensors", flush=True)
        with safe_open(path, framework="pt") as h:
            for name in names:
                t = h.get_slice(name)[:].contiguous()
                is_text = ".language_model.layers." in name
                proj = is_expert_proj(name) if is_text else None
                if proj is not None:
                    assert t.dim() == 2 and t.size(1) % 32 == 0, (
                        name, tuple(t.shape))
                    w = t.to("xpu", dtype=torch.bfloat16)
                    q, s = quantize_mxfp4(w)
                    del w
                    torch.xpu.synchronize()
                    n_q += 1
                    if n_q % args.validate_every == 0:
                        validate(t, q, s, name)
                    stem = name[: -len(".weight")]
                    emit(stem + ".weight_packed", q)
                    emit(stem + ".weight_scale", s)
                    del q, s
                elif is_text and ".linear_attn." in name and any(
                        name.endswith(f".{p}.weight")
                        for p in LINEAR_ATTN_QUANT):
                    assert t.dim() == 2 and t.size(1) % 32 == 0, (
                        name, tuple(t.shape))
                    w = t.to("xpu", dtype=torch.bfloat16)
                    q, s = quantize_mxfp4(w)
                    del w
                    torch.xpu.synchronize()
                    n_q += 1
                    if n_q % args.validate_every == 0:
                        validate(t, q, s, name)
                    stem = name[: -len(".weight")]
                    emit(stem + ".weight_packed", q)
                    emit(stem + ".weight_scale", s)
                    del q, s
                else:
                    emit(name, t)
        state["done_shards"].append(src)
        save_state()
        print(f"  shard done ({time.time() - t0:.0f}s elapsed)", flush=True)
    flush()

    # finalize index with real shard count. Normalize EVERY output shard name
    # (incremental runs stamp provisional -of-NNNNN counts).
    n_shards = state["out_shard_no"] - 1
    final_map = {}
    for k, fn in state["weight_map"].items():
        m = re.fullmatch(r"model-(\d+)-of-(?:XXXXX|\d+)\.safetensors", fn)
        final_map[k] = (f"model-{int(m.group(1)):05d}-of-{n_shards:05d}"
                        f".safetensors")
    for fn in os.listdir(OUTDIR):
        m = re.fullmatch(r"model-(\d+)-of-(?:XXXXX|\d+)\.safetensors", fn)
        if m:
            os.rename(os.path.join(OUTDIR, fn),
                      os.path.join(OUTDIR, f"model-{int(m.group(1)):05d}-"
                                           f"of-{n_shards:05d}.safetensors"))
    with open(os.path.join(OUTDIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": state["total_out_bytes"]},
                   "weight_map": final_map}, f)
    print(f"DONE: {len(final_map)} tensors, {n_shards} shards, "
          f"{state['total_out_bytes'] / 2**30:.2f} GiB in "
          f"{time.time() - t0:.0f}s")
    print(f"quantized projs: {n_q}, validated: {n_val}, "
          f"max mean-rel-err={max_rel:.5f}, mean={sum_rel / max(1, n_val):.5f}")


if __name__ == "__main__":
    main()
