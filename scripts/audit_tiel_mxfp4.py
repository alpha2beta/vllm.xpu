#!/usr/bin/env python3
"""Independent post-hoc audit of the shipped Tiel MXFP4 checkpoint.

Re-downloads ONE BF16 source shard, CPU-dequantizes EVERY shipped packed
tensor derived from it with an independent MXFP4 decoder (E2M1 table +
E8M0 scales + low-nibble-even packing, same spec as
scripts/mxfp4_kernel_check.py::_dequant_mx), and reports rel-err vs the
fresh BF16 source. Catches save/load corruption and any in-quant bug the
1-in-64 conversion spot-checks missed. No XPU memory needed.

Usage: ./scripts/audit_tiel_mxfp4.py --shard 9
"""
import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(REPO_ROOT, "models", "_staging_tiel")
OUTDIR = os.path.join(
    REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4")
SRC_REPO = "symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized"

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_cpu(packed_u8, scale_u8, group=32):
    """packed [..., K/2] U8, scale [..., K/32] U8 -> float32 [..., K]."""
    lo = (packed_u8 & 0x0F).to(torch.int64)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
    vals = torch.empty((*packed_u8.shape[:-1], packed_u8.shape[-1] * 2),
                       dtype=torch.float32)
    for nib, sl in ((lo, slice(0, None, 2)), (hi, slice(1, None, 2))):
        sign = torch.where((nib & 0x08) != 0, -1.0, 1.0)
        vals[..., sl] = sign * _E2M1[(nib & 0x07)]
    scales = torch.pow(2.0, scale_u8.float() - 127.0)
    return vals * scales.repeat_interleave(group, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    args = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import load_file

    src = f"model.safetensors-{args.shard:05d}-of-00017.safetensors"
    spath = os.path.join(STAGING, src)
    if not os.path.exists(spath):
        from huggingface_hub import hf_hub_download
        print(f"downloading {src} ...", flush=True)
        hf_hub_download(SRC_REPO, src, local_dir=STAGING)
    out_index = json.load(
        open(os.path.join(OUTDIR, "model.safetensors.index.json"))
    )["weight_map"]

    # map: source tensor -> list of (packed, scale, src_slice_fn, shape)
    jobs = []
    with safe_open(spath, framework="pt") as h:
        names = list(h.keys())
    for name in names:
        if ".language_model.layers." not in name:
            continue
        if name.endswith("mlp.experts.down_proj"):
            base = name[: -len(".experts.down_proj")] + ".experts"
            for e in range(256):
                jobs.append((f"{base}.{e}.down_proj", (e, slice(None)),
                             "down"))
        elif name.endswith("mlp.experts.gate_up_proj"):
            base = name[: -len(".experts.gate_up_proj")] + ".experts"
            for e in range(256):
                for proj, r in (("gate_proj", slice(0, 512)),
                                ("up_proj", slice(512, 1024))):
                    jobs.append((f"{base}.{e}.{proj}", (e, r),
                                 os.path.basename(name)))
        elif ".linear_attn." in name and any(
                name.endswith(f".{p}.weight") for p in
                ("in_proj_qkv", "in_proj_z", "out_proj",
                 "in_proj_a", "in_proj_b")):
            stem = name[: -len(".weight")]
            jobs.append((stem, None, os.path.basename(name)))
    print(f"{len(jobs)} packed pairs to audit from {src}", flush=True)

    # group jobs by output shard file to minimize opens (packed and scale
    # of one pair may straddle a shard boundary, so resolve each blob's
    # file independently)
    by_file = {}
    for stem, ref, tag in jobs:
        for suf in (".weight_packed", ".weight_scale"):
            fn = out_index[stem + suf]
            by_file.setdefault(fn, []).append(stem + suf)
    # invert: stem -> {packed: tensor, scale: tensor} loaded lazily per file
    worst, worst_name, tot, s = 0.0, "", 0, 0.0
    cache = {}
    stacked = {}
    with safe_open(spath, framework="pt") as h:
        for fn in sorted(by_file):
            blobs = load_file(os.path.join(OUTDIR, fn))
            for blob_name in by_file[fn]:
                cache[blob_name] = blobs[blob_name]
            del blobs
        # preload each distinct stacked source tensor once
        need_stacked = set()
        need_lin = set()
        for st, ref, _ in jobs:
            if ref is None:
                need_lin.add(st + ".weight")
            else:
                layer_base = st.split(".mlp.experts.")[0]
                need_stacked.add(layer_base + ".mlp.experts." + (
                    "down_proj" if st.endswith(".down_proj")
                    else "gate_up_proj"))
        for sname in sorted(need_stacked):
            stacked[sname] = h.get_slice(sname)[:].float()
            print(f"  preloaded source {sname[-50:]} "
                  f"{tuple(stacked[sname].shape)}", flush=True)
        for sname in sorted(need_lin):
            stacked[sname] = h.get_slice(sname)[:].float()
        for stem, ref, tag in jobs:
            q = cache.pop(stem + ".weight_packed")
            sc = cache.pop(stem + ".weight_scale")
            dq = dequant_cpu(q, sc)
            del q, sc
            if ref is None:
                src_t = stacked[stem + ".weight"]
            else:
                e, r = ref
                layer_base = stem.split(".mlp.experts.")[0]
                if stem.endswith(".down_proj"):
                    src_t = stacked[
                        layer_base + ".mlp.experts.down_proj"][e].reshape(
                            *dq.shape)
                else:
                    src_t = stacked[
                        layer_base + ".mlp.experts.gate_up_proj"][e][r]
            rel = ((dq - src_t).abs().mean().item()
                   / src_t.abs().mean().item())
            tot += 1
            s += rel
            if rel > worst:
                worst, worst_name = rel, stem
            if tot % 500 == 0:
                print(f"  ... {tot}/{len(jobs)} pairs, worst so far "
                      f"{worst:.5f}", flush=True)
    print(f"AUDIT-OK: {tot} pairs, mean rel-err={s / tot:.5f}, "
          f"max={worst:.5f} ({worst_name})")


if __name__ == "__main__":
    main()
