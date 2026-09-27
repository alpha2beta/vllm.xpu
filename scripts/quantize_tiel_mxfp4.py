#!/usr/bin/env python3
"""Quantize Tiel-Coder-35B-A3B (BF16, fused/stacked experts) to the exact
MXFP4 safetensors schema of pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4 so the
vLLM XPU path (CompressedTensorsW4A4Mxfp4 + XPUExpertsMxFp4) loads it.

Schema (verified against the reference checkpoint):
  - routed experts: per-expert, per-proj
      .../experts.E.{gate_proj,up_proj,down_proj}.weight_packed  U8 [out, in/2]
      .../experts.E.{gate_proj,up_proj,down_proj}.weight_scale   U8 [out, in/32]
    (gate/up split from Tiel's fused experts.gate_up_proj [E,1024,2048])
  - linear_attn (30 layers): in_proj_qkv / in_proj_z / out_proj / in_proj_a /
    in_proj_b  ->  <name>.weight_packed + <name>.weight_scale (U8)
  - everything else (self_attn, norms, gate, shared experts, embed, lm_head,
    visual, MTP incl. fused MTP experts): copied unchanged
  - config.json = Tiel text/vision config + reference quantization_config
    with a Tiel-adapted ignore list (every BF16 module)

Quantizer: torch.ops.vllm.xpu_mxfp4_quantize (the inference kernel package's
own quantizer — format match by construction). E2M1 group-32 symmetric,
E8M0 uint8 scales, low-nibble-even packing. Stacked expert tensors are
reshaped to 2D and quantized in ONE call per layer-proj (row-major reshape
preserves rows exactly), then split per expert.

Usage:
  ./scripts/quantize_tiel_mxfp4.py [--shards A-B]   (default 1-17)
  Output: models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4/
  Resume: completed source shards are recorded in
  models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4/quantize_state.json
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
STAGING = os.path.join(REPO_ROOT, "models", "_staging_tiel")
OUTDIR = os.path.join(
    REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4")
SRC_INDEX = ("/home/yanchun/Projects/AInfer/models/"
             "Tiel-Coder-35B-A3B-Genesis-Hermes/model.safetensors.index.json")
N_EXPERTS = 256
SHARD_TARGET_BYTES = 1 << 30  # ~1 GiB output shards like the reference

LINEAR_ATTN_QUANT = ("in_proj_qkv", "in_proj_z", "out_proj",
                     "in_proj_a", "in_proj_b")

EXPECTED_LIN = {
    "in_proj_qkv": (8192, 2048),
    "in_proj_z": (4096, 2048),
    # NOTE: Tiel out_proj is [2048, 4096] (DeltaNet concat), unlike the
    # Qwen3.6 reference ([2048, 2048]). Same arch otherwise.
    "out_proj": (2048, 4096),
    "in_proj_a": (32, 2048),
    "in_proj_b": (32, 2048),
}


def quantize_mxfp4(w2d_xpu):
    """[rows, K] BF16 XPU tensor -> (packed U8 [rows, K/2], scale U8 [rows, K/32])."""
    assert w2d_xpu.dim() == 2 and w2d_xpu.size(1) % 32 == 0
    q, s = torch.ops.vllm.xpu_mxfp4_quantize(w2d_xpu)
    return q.view(torch.uint8).to("cpu"), s.view(torch.uint8).to("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1-17")
    ap.add_argument("--validate-every", type=int, default=64,
                    help="dequant-validate 1 in N expert-projs")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.shards.split("-"))

    import vllm_xpu_kernels  # noqa: F401 (registers torch.ops._xpu_C.*)
    import vllm._xpu_ops  # noqa: F401 (registers torch.ops.vllm.*)
    from vllm_xpu_kernels.moe_utils import dequant_mxfp4
    assert torch.xpu.is_available(), "XPU required for xpu_mxfp4_quantize"

    idx = json.load(open(SRC_INDEX))["weight_map"]
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
            # dequant_mxfp4 dispatches on dtype: U8 inputs are read as raw
            # floats, so view back to the MX dtypes first (same as the vLLM
            # loader does after reading U8 from safetensors).
            qr = q_cpu.to("xpu").view(torch.float4_e2m1fn_x2)
            sr = s_cpu.to("xpu").view(torch.float8_e8m0fnu)
            dq = dequant_mxfp4(qr, sr).float().cpu()
        ref = src2d_cpu.float()
        rel = (dq - ref).abs().mean().item() / ref.abs().mean().item()
        max_rel = max(max_rel, rel)
        sum_rel += rel
        if n_val <= 3 or n_val % 200 == 0:
            print(f"  [val] {tag}: mean-rel-err={rel:.5f}", flush=True)

    t0 = time.time()
    for si in range(lo, hi + 1):
        src = f"model.safetensors-{si:05d}-of-00017.safetensors"
        if src in state["done_shards"]:
            print(f"[{si}/17] {src}: SKIP (done)")
            continue
        path = os.path.join(STAGING, src)
        if not os.path.exists(path):
            print(f"MISSING source shard (download first): {src}")
            sys.exit(2)
        names = by_shard[src]
        print(f"[{si}/17] {src}: {len(names)} tensors", flush=True)
        with safe_open(path, framework="pt") as h:
            for name in names:
                t = h.get_slice(name)[:].contiguous()
                # NOTE: only the text tower is quantized. MTP fused experts
                # (mtp.layers.*.mlp.experts.*) stay BF16 like the reference.
                is_text = ".language_model.layers." in name
                if is_text and name.endswith("mlp.experts.down_proj"):
                    assert tuple(t.shape) == (N_EXPERTS, 2048, 512), (name, tuple(t.shape))
                    # output base must match the reference layout exactly:
                    #   ...mlp.experts.{e}.down_proj.weight_packed
                    base = name[: -len(".experts.down_proj")] + ".experts"
                    w = t.view(N_EXPERTS * 2048, 512).to(
                        "xpu", dtype=torch.bfloat16)
                    q, s = quantize_mxfp4(w)
                    del w
                    torch.xpu.synchronize()
                    for e in range(N_EXPERTS):
                        sl = slice(e * 2048, (e + 1) * 2048)
                        emit(f"{base}.{e}.down_proj.weight_packed",
                             q[sl].clone())
                        emit(f"{base}.{e}.down_proj.weight_scale",
                             s[sl].clone())
                        n_q += 1
                        if n_q % args.validate_every == 0:
                            validate(t[e].view(2048, 512), q[sl], s[sl],
                                     f"{name} expert {e} down")
                    del q, s
                elif is_text and name.endswith("mlp.experts.gate_up_proj"):
                    assert tuple(t.shape) == (N_EXPERTS, 1024, 2048), (name, tuple(t.shape))
                    base = name[: -len(".experts.gate_up_proj")] + ".experts"
                    w = t.view(N_EXPERTS * 1024, 2048).to(
                        "xpu", dtype=torch.bfloat16)
                    q, s = quantize_mxfp4(w)
                    del w
                    torch.xpu.synchronize()
                    for e in range(N_EXPERTS):
                        base_e = e * 1024
                        for proj, r0 in (("gate_proj", base_e),
                                         ("up_proj", base_e + 512)):
                            sl = slice(r0, r0 + 512)
                            emit(f"{base}.{e}.{proj}.weight_packed",
                                 q[sl].clone())
                            emit(f"{base}.{e}.{proj}.weight_scale",
                                 s[sl].clone())
                            n_q += 1
                            if n_q % args.validate_every == 0:
                                validate(t[e][r0 - base_e:r0 - base_e + 512],
                                         q[sl], s[sl],
                                         f"{name} expert {e} {proj}")
                    del q, s
                elif is_text and ".linear_attn." in name and any(
                        name.endswith(f".{p}.weight")
                        for p in LINEAR_ATTN_QUANT):
                    proj = name.rsplit(".", 2)[-2]
                    assert tuple(t.shape) == EXPECTED_LIN[proj], (name, tuple(t.shape))
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
                # note: emitted tensors are referenced by `pending`; the
                # source `t` is freed here (`safe_open` closes at block end)
        state["done_shards"].append(src)
        save_state()
        print(f"  shard done ({time.time() - t0:.0f}s elapsed)", flush=True)
    flush()

    # finalize index with real shard count. Normalize EVERY output shard name
    # (earlier incremental runs already stamped provisional -of-NNNNN counts).
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
