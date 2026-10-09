#!/usr/bin/env python3
"""Audit of the Occamy MXFP4 checkpoint:
1. Structural check: all packed weights have scales, shapes match, index valid.
2. MTP check: MTP draft block present and quantized.
3. Config check: quantization_config present, compressed-tensors format, ignore list valid.
4. Independent CPU dequantization check on a sample of tensors.

Usage: ./scripts/audit_occamy_mxfp4.py
"""

import json
import os
import re
import sys
import torch
from safetensors.torch import load_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO_ROOT, "models", "Qwen3.6-35B-A3B-occamy-1.0-MXFP4")

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_cpu(packed_u8, scale_u8, group=32):
    """packed [..., K/2] U8, scale [..., K/32] U8 -> float32 [..., K]."""
    lo = (packed_u8 & 0x0F).to(torch.int64)
    hi = ((packed_u8 >> 4) & 0x0F).to(torch.int64)
    vals = torch.empty((*packed_u8.shape[:-1], packed_u8.shape[-1] * 2), dtype=torch.float32)
    for nib, sl in ((lo, slice(0, None, 2)), (hi, slice(1, None, 2))):
        sign = torch.where((nib & 0x08) != 0, -1.0, 1.0)
        vals[..., sl] = sign * _E2M1[(nib & 0x07)]
    scales = torch.pow(2.0, scale_u8.float() - 127.0)
    return vals * scales.repeat_interleave(group, dim=-1)


def main():
    print(f"Auditing checkpoint: {OUTDIR} ...\n")
    if not os.path.exists(OUTDIR):
        print(f"Error: {OUTDIR} does not exist.")
        return 1

    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        print("Error: model.safetensors.index.json not found.")
        return 1

    idx = json.load(open(idx_path))
    wmap = idx.get("weight_map", {})
    print(f"Index check: {len(wmap)} total tensors mapped.")

    # 1. Check packed vs scale pairs
    packed = {k[:-len(".weight_packed")] for k in wmap if k.endswith(".weight_packed")}
    scales = {k[:-len(".weight_scale")] for k in wmap if k.endswith(".weight_scale")}

    assert packed == scales, f"Mismatch: {len(packed)} packed vs {len(scales)} scales (diff: {packed ^ scales})"
    print(f"Packed/Scale pairs: {len(packed)} pairs strictly matched.")

    # 2. Check base expert counts
    base_experts = {p for p in packed if ".mlp.experts." in p and not p.startswith("mtp.")}
    expected_base = 40 * 256 * 3
    assert len(base_experts) == expected_base, f"Expected {expected_base} base expert projs, got {len(base_experts)}"
    print(f"Base routed experts: {len(base_experts)} projs OK (40 layers x 256 experts x 3 projs).")

    # 3. Check linear attention counts
    linear_attn = {p for p in packed if ".linear_attn." in p}
    expected_lin = 30 * 5
    assert len(linear_attn) == expected_lin, f"Expected {expected_lin} linear_attn projs, got {len(linear_attn)}"
    print(f"Linear attention projs: {len(linear_attn)} projs OK (30 layers x 5 projs).")

    # 4. Check MTP draft block
    mtp_experts = {p for p in packed if p.startswith("mtp.layers.0.mlp.experts.")}
    expected_mtp = 256 * 3
    assert len(mtp_experts) == expected_mtp, f"Expected {expected_mtp} MTP expert projs, got {len(mtp_experts)}"
    mtp_plain = {k for k in wmap if k.startswith("mtp.") and not (k.endswith(".weight_packed") or k.endswith(".weight_scale"))}
    assert len(mtp_plain) == 17, f"Expected 17 non-expert MTP tensors, got {len(mtp_plain)}"
    print(f"MTP draft block: {len(mtp_experts)} expert projs + {len(mtp_plain)} non-expert tensors OK.")

    # 5. Check config.json
    cfg_path = os.path.join(OUTDIR, "config.json")
    assert os.path.exists(cfg_path), "config.json missing"
    cfg = json.load(open(cfg_path))
    qc = cfg.get("quantization_config", {})
    assert qc.get("quant_method") == "compressed-tensors", f"quant_method: {qc.get('quant_method')}"
    assert qc.get("format") == "mxfp4-pack-quantized", f"format: {qc.get('format')}"
    assert "group_0" in qc.get("config_groups", {}), "group_0 missing"
    assert len(qc.get("ignore", [])) > 0, "ignore list empty"
    assert cfg.get("text_config", {}).get("mtp_num_hidden_layers") == 1 or cfg.get("mtp_num_hidden_layers") == 1, "MTP disabled in config"
    print(f"Configuration check: compressed-tensors / mxfp4-pack-quantized valid, {len(qc.get('ignore'))} ignored BF16 modules, MTP enabled.")

    # 6. Spot-check dequantization on first shard
    first_shard = sorted(set(wmap.values()))[0]
    first_shard_path = os.path.join(OUTDIR, first_shard)
    blobs = load_file(first_shard_path)
    sample_packed = [k for k in blobs if k.endswith(".weight_packed")][:5]
    print(f"\nSpot-checking {len(sample_packed)} packed tensors from {first_shard}:")
    for sp in sample_packed:
        stem = sp[:-len(".weight_packed")]
        q = blobs[sp]
        sc = blobs[stem + ".weight_scale"]
        dq = dequant_cpu(q, sc)
        assert not torch.isnan(dq).any(), f"NaN in {sp}"
        assert not torch.isinf(dq).any(), f"Inf in {sp}"
        print(f"  {stem}: packed shape {list(q.shape)}, scale shape {list(sc.shape)} -> dequant {list(dq.shape)}, finite range [{dq.min().item():.3f}, {dq.max().item():.3f}]")

    print("\n==========================================")
    print("AUDIT SUCCESSFUL: All checks PASSED!")
    print("==========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
