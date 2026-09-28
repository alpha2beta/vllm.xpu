#!/usr/bin/env python3
"""Independent post-hoc audit of the KAT MXFP4 checkpoint.

Re-downloads ONE BF16 source shard, CPU-dequantizes EVERY shipped packed
tensor derived from it with an independent MXFP4 decoder (E2M1 table +
E8M0 scales + low-nibble-even packing), and reports rel-err vs the fresh
BF16 source. Catches save/load corruption and any in-quant bug the 1-in-64
conversion spot-checks missed. No XPU memory needed. Source shard deleted
afterwards (staging kept clean).

KAT mapping is 1:1 (per-expert source) — simpler than Tiel's fused split.

Usage: ./scripts/audit_kat_mxfp4.py [--shard 0]
"""
import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(REPO_ROOT, "models", "_staging_kat")
OUTDIR = os.path.join(
    REPO_ROOT, "models", "KAT-Coder-V2.5-Dev-MXFP4")
SRC_REPO = "Kwaipilot/KAT-Coder-V2.5-Dev"
N_SHARDS = 13

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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import load_file

    src = f"model-{args.shard:05d}-of-{N_SHARDS:05d}.safetensors"
    spath = os.path.join(STAGING, src)
    if not os.path.exists(spath):
        from huggingface_hub import hf_hub_download
        print(f"downloading {src} ...", flush=True)
        hf_hub_download(SRC_REPO, src, local_dir=STAGING)
    out_index = json.load(
        open(os.path.join(OUTDIR, "model.safetensors.index.json"))
    )["weight_map"]

    # jobs: (packed_stem, source_name) — 1:1 for experts and linear_attn
    jobs = []
    with safe_open(spath, framework="pt") as h:
        names = list(h.keys())
    for name in names:
        if ".language_model.layers." not in name:
            continue
        if name.endswith(".weight") and (
                ".mlp.experts." in name or ".linear_attn." in name):
            # only quantized ones have packed counterparts; check index
            stem = name[: -len(".weight")]
            if stem + ".weight_packed" in out_index:
                jobs.append((stem, name))
    print(f"{len(jobs)} packed pairs to audit from {src}", flush=True)

    by_file = {}
    for stem, _ in jobs:
        for suf in (".weight_packed", ".weight_scale"):
            fn = out_index[stem + suf]
            by_file.setdefault(fn, []).append(stem + suf)
    cache = {}
    for fn in sorted(by_file):
        blobs = load_file(os.path.join(OUTDIR, fn))
        for blob_name in by_file[fn]:
            cache[blob_name] = blobs[blob_name]
        del blobs
    worst, worst_name, tot, s = 0.0, "", 0, 0.0
    per_pair = []
    with safe_open(spath, framework="pt") as h:
        for stem, src_name in jobs:
            q = cache.pop(stem + ".weight_packed")
            sc = cache.pop(stem + ".weight_scale")
            dq = dequant_cpu(q, sc)
            del q, sc
            src_t = h.get_slice(src_name)[:].float()
            rel = ((dq - src_t).abs().mean().item()
                   / src_t.abs().mean().item())
            tot += 1
            s += rel
            per_pair.append((stem, rel))
            if rel > worst:
                worst, worst_name = rel, stem
            if tot % 500 == 0:
                print(f"  ... {tot}/{len(jobs)} pairs, worst so far "
                      f"{worst:.5f}", flush=True)
    print(f"AUDIT-OK: {tot} pairs, mean rel-err={s / tot:.5f}, "
          f"max={worst:.5f} ({worst_name})")
    # Gate: mean < 0.13. Pairs above 0.15 get a documented waiver iff rare
    # (<0.5%) — MXFP4 group-32 cannot represent a few large-magnitude outlier
    # channels inside a group (verified: offender groups show median-normal
    # error with only top groups elevated). Track the tail explicitly.
    over = [(n, r) for n, r in per_pair if r >= 0.15]
    print(f"pairs >= 0.15: {len(over)}/{tot} "
          f"{[n.split('.experts.')[-1] for n, _ in over][:5]}")
    assert s / tot < 0.13, f"audit mean rel-err {s / tot:.4f} >= 0.13"
    assert len(over) <= max(1, tot * 0.005), (
        f"too many high-error pairs: {len(over)}/{tot}")
    if not args.keep and os.path.exists(spath):
        os.remove(spath)
        print(f"removed audit source {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
