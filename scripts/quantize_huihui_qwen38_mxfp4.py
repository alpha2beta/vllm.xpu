#!/usr/bin/env python3
"""Quantize Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated (BF16, DENSE
qwen3_5 arch with hybrid linear/full attention) to the exact MXFP4
safetensors schema of the proven-dense reference
Solstice-AI/Qwen3.8-27B-TURBO-Fable-...-MXFP4 (verified loadable on this box,
2.88 tok/s dense wall) so the vLLM XPU path (CompressedTensorsW4A4Mxfp4) loads it.

Schema (verified against the dense reference index):
  Quantized -> <name>.weight_packed U8 [out, in/2] + <name>.weight_scale U8 [out, in/32]:
    - mlp.{gate,up,down}_proj           (64 layers, dense MLP)
    - self_attn.{q,k,v,o}_proj          (16 full-attention layers)
    - linear_attn.{in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b}  (48 layers)
  Everything else BF16: embed, lm_head, all norms (incl. q_norm/k_norm,
  linear_attn.norm), A_log/conv1d/dt_bias, visual tower, MTP block.
  config.json = source config + reference quantization_config with an
  exhaustive ignore list (every BF16 module, Occamy-style).

MTP: source ships mtp-bf16.safetensors separately (NOT in the main index);
its tensors are merged into the output index at root `mtp.*` BF16 (reference
layout) — text_config.mtp_num_hidden_layers is already 1 in the source config.

Quantizer: torch.ops.vllm.xpu_mxfp4_quantize (inference kernel's own quantizer
— format match by construction). E2M1 group-32 symmetric, E8M0 uint8 scales.

Usage:
  ./scripts/quantize_huihui_qwen38_mxfp4.py [--shards A-B]   (default 1-2)
  Output: /run/media/yanchun/CCCOMA_X64FRE_EN-GB_DV9/models/
          Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4/
  Resume: OUTDIR/quantize_state.json
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
STAGING = os.path.join(REPO_ROOT, "models", "_staging_huihui_qwen38")
OUTDIR = ("/run/media/yanchun/CCCOMA_X64FRE_EN-GB_DV9/models/"
          "Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4")
SRC_INDEX = os.path.join(STAGING, "model.safetensors.index.json")
SRC_CONFIG = os.path.join(STAGING, "config.json")
REF_CONFIG = os.path.join(REPO_ROOT, "scripts", "ref",
                         "solstice-qwen38-dense-mxfp4-config.json")
MTP_FILE = os.path.join(STAGING, "mtp-bf16.safetensors")
N_SRC_SHARDS = 2
SHARD_TARGET_BYTES = 1 << 30  # ~1 GiB output shards like the reference

# Text-tower Linear modules that get MXFP4-packed (dense arch, no experts).
QUANT_PROJS = (
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
    "linear_attn.out_proj", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
)


def is_quant_target(name):
    """True for text-tower Linear weights we pack (not visual, not mtp)."""
    if not name.startswith("model.language_model.layers."):
        return False
    if not name.endswith(".weight"):
        return False
    return any(name[: -len(".weight")].endswith(p) for p in QUANT_PROJS)


def module_of(tensor_name):
    """Tensor name -> module name for the ignore list (strip leaf attr)."""
    for suf in (".weight", ".bias"):
        if tensor_name.endswith(suf):
            return tensor_name[: -len(suf)]
    return tensor_name


def quantize_mxfp4(w2d_xpu):
    """[rows, K] BF16 XPU tensor -> (packed U8 [rows, K/2], scale U8 [rows, K/32])."""
    assert w2d_xpu.dim() == 2 and w2d_xpu.size(1) % 32 == 0, \
        (tuple(w2d_xpu.shape),)
    q, s = torch.ops.vllm.xpu_mxfp4_quantize(w2d_xpu)
    return q.view(torch.uint8).to("cpu"), s.view(torch.uint8).to("cpu")


def build_config(src, quantized_modules):
    """Source config + reference quantization_config, exhaustive ignore list."""
    ref = json.load(open(REF_CONFIG))
    out = dict(src)
    qc = json.loads(json.dumps(ref["quantization_config"]))

    # Ignore every module in the checkpoint that is NOT a quantized Linear:
    # collect all module names from source tensor names (+ MTP file when present;
    # early partial runs rewrite config.json again on the final pass).
    modules = set()
    wm = json.load(open(SRC_INDEX))["weight_map"]
    for t in list(wm.keys()):
        modules.add(module_of(t))
    if os.path.exists(MTP_FILE):
        with safe_open(MTP_FILE, framework="pt") as h:
            for t in h.keys():
                modules.add(module_of(t))
    ignore = sorted(m for m in modules if m not in quantized_modules)
    qc["ignore"] = ignore
    out["quantization_config"] = qc
    return out, ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1-2")
    ap.add_argument("--validate-every", type=int, default=64,
                    help="dequant-validate 1 in N packed projs")
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
    state = {"done_shards": [], "done_mtp": False, "out_shard_no": 1,
             "weight_map": {}, "total_out_bytes": 0,
             "quantized_modules": [], "n_q": 0}
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"resume: {len(state['done_shards'])} source shards done, "
              f"mtp={'done' if state['done_mtp'] else 'pending'}, "
              f"next output shard #{state['out_shard_no']}")

    quantized_modules = set(state["quantized_modules"])
    pending, pending_bytes = [], 0
    n_q, n_val, max_rel, sum_rel = state.get("n_q", 0), 0, 0.0, 0.0

    def save_state():
        state["quantized_modules"] = sorted(quantized_modules)
        state["n_q"] = n_q
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

    t0 = time.time()
    for si in range(lo, hi + 1):
        src = f"model-{si:05d}-of-{N_SRC_SHARDS:05d}.safetensors"
        if src in state["done_shards"]:
            print(f"[{si}/{N_SRC_SHARDS}] {src}: SKIP (done)")
            continue
        path = os.path.join(STAGING, src)
        if not os.path.exists(path):
            print(f"MISSING source shard (download first): {src}")
            sys.exit(2)
        names = by_shard[src]
        print(f"[{si}/{N_SRC_SHARDS}] {src}: {len(names)} tensors", flush=True)
        with safe_open(path, framework="pt") as h:
            for name in names:
                if is_quant_target(name):
                    t = h.get_slice(name)[:].contiguous()
                    assert t.dim() == 2, (name, tuple(t.shape))
                    w = t.to("xpu", dtype=torch.bfloat16)
                    q, s = quantize_mxfp4(w)
                    del w
                    torch.xpu.synchronize()
                    n_q += 1
                    quantized_modules.add(module_of(name))
                    if n_q % args.validate_every == 0:
                        validate(t, q, s, name)
                    stem = name[: -len(".weight")]
                    emit(stem + ".weight_packed", q)
                    emit(stem + ".weight_scale", s)
                    del q, s
                else:
                    t = h.get_slice(name)[:].contiguous()
                    emit(name, t)
        state["done_shards"].append(src)
        save_state()
        print(f"  shard done ({time.time() - t0:.0f}s elapsed)", flush=True)

    # MTP block: separate BF16 file, merged as root `mtp.*` (reference layout).
    # May be absent on early partial runs (shard-1-while-downloading) — the
    # final pass merges it; the pipeline pre-flights its presence.
    if not state["done_mtp"]:
        if not os.path.exists(MTP_FILE):
            print("[mtp] SKIP (file not present yet — partial run)")
        else:
            with safe_open(MTP_FILE, framework="pt") as h:
                keys = list(h.keys())
                print(f"[mtp] {len(keys)} tensors: {keys[:4]} ...", flush=True)
                for name in keys:
                    emit(name, h.get_slice(name)[:].contiguous())
            state["done_mtp"] = True
            save_state()
            print(f"  mtp merged ({time.time() - t0:.0f}s elapsed)", flush=True)

    flush()

    # config.json (+ exhaustive ignore list) and ancillary files.
    src_cfg = json.load(open(SRC_CONFIG))
    cfg, ignore = build_config(src_cfg, quantized_modules)
    with open(os.path.join(OUTDIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"config written: {len(ignore)} ignored modules, "
          f"{len(quantized_modules)} packed modules")
    for fn in ("chat_template.jinja", "generation_config.json",
               "tokenizer.json", "tokenizer_config.json",
               "preprocessor_config.json", "processor_config.json",
               "video_preprocessor_config.json"):
        src = os.path.join(STAGING, fn)
        if os.path.exists(src):
            with open(src, "rb") as a, open(os.path.join(OUTDIR, fn), "wb") as b:
                b.write(a.read())

    # finalize index with real shard count (normalize provisional names).
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
