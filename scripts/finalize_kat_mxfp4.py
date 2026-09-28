#!/usr/bin/env python3
"""Finalize models/KAT-Coder-V2.5-Dev-MXFP4/ after quantization:
config.json (KAT config + MXFP4 quantization_config), tokenizer sidecars,
and schema verification against the reference layout.

Usage: ./scripts/finalize_kat_mxfp4.py
"""
import json
import os
import re
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(
    REPO_ROOT, "models", "KAT-Coder-V2.5-Dev-MXFP4")
SRC_REPO = "Kwaipilot/KAT-Coder-V2.5-Dev"
REF_CFG = os.path.join(
    REPO_ROOT, "models", "Qwen3.6-35B-A3B-MXFP4", "config.json")

SIDECARS = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "chat_template.jinja", "generation_config.json",
            "configuration.json"]


def main():
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    assert os.path.exists(idx_path), "run quantize_kat_mxfp4.py first"
    weight_map = json.load(open(idx_path))["weight_map"]

    # --- schema verification: every packed tensor has its scale, shapes sane
    files = {}
    packed, scales, plain = set(), set(), {}
    for name, fn in weight_map.items():
        if name.endswith(".weight_packed"):
            packed.add(name[: -len(".weight_packed")])
        elif name.endswith(".weight_scale"):
            scales.add(name[: -len(".weight_scale")])
        else:
            plain[name] = fn
    assert packed == scales, (
        "packed/scale mismatch: "
        f"{sorted(packed ^ scales)[:5]}")
    print(f"packed pairs: {len(packed)}, plain BF16 tensors: {len(plain)}")

    n_exp = sum(1 for p in packed if ".mlp.experts." in p)
    assert n_exp == 40 * 256 * 3, f"expert projs: {n_exp}"
    n_lin = sum(1 for p in packed if ".linear_attn." in p)
    assert n_lin == 30 * 5, f"linear_attn projs: {n_lin}"
    print(f"expert projs OK ({n_exp}), linear_attn projs OK ({n_lin})")

    # no leftover fused/stacked expert tensors
    leftovers = [n for n in list(packed) + list(plain)
                 if ".language_model.layers." in n and
                 ("gate_up_proj" in n
                  or re.search(r"experts\.down_proj$", n))]
    assert not leftovers, f"unconverted fused tensors: {leftovers[:5]}"
    print("no fused leftovers")

    # --- ignore list: every BF16 module (reference convention: full tensor
    # name minus trailing .weight)
    ignore = set()
    for n in plain:
        ignore.add(n[: -len(".weight")] if n.endswith(".weight") else n)
    ignore = sorted(ignore)
    print(f"ignore entries: {len(ignore)}")

    # --- config.json: KAT config + reference quantization_config
    kat_cfg_path = os.path.join(OUTDIR, "config.json")
    if not os.path.exists(kat_cfg_path):
        print("downloading KAT config.json ...")
        hf_hub_download(SRC_REPO, "config.json", local_dir=OUTDIR)
    kat_cfg = json.load(open(kat_cfg_path))
    ref_q = json.load(open(REF_CFG))["quantization_config"]
    assert ref_q["quant_method"] == "compressed-tensors"
    assert ref_q["format"] == "mxfp4-pack-quantized"
    kat_cfg["quantization_config"] = {
        "quant_method": ref_q["quant_method"],
        "format": ref_q["format"],
        "config_groups": ref_q["config_groups"],
        "ignore": ignore,
    }
    with open(os.path.join(OUTDIR, "config.json"), "w") as f:
        json.dump(kat_cfg, f, indent=2)
    print("wrote config.json with quantization_config "
          f"(ignore={len(ignore)})")

    # --- sidecars from the source repo
    for fn in SIDECARS:
        dest = os.path.join(OUTDIR, fn)
        if not os.path.exists(dest):
            try:
                hf_hub_download(SRC_REPO, fn, local_dir=OUTDIR)
                print(f"downloaded {fn}")
            except Exception as e:
                print(f"skip {fn}: {e}")
    # README provenance
    with open(os.path.join(OUTDIR, "README.md"), "w") as f:
        f.write(
            "# KAT-Coder-V2.5-Dev-MXFP4\n\n"
            "MXFP4 (compressed-tensors `mxfp4-pack-quantized`, E2M1 group-32 "
            "symmetric, E8M0 scales) quantization of the text tower's routed "
            "MoE experts (already per-expert in source) + linear-attention "
            "projections, in the exact safetensors schema of "
            "`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` for the vLLM XPU "
            "backend (`CompressedTensorsW4A4Mxfp4` + `XPUExpertsMxFp4`).\n\n"
            "Source: `Kwaipilot/KAT-Coder-V2.5-Dev` "
            "(BF16 safetensors, text-only, no MTP block).\n"
            "Quantizer: `torch.ops.vllm.xpu_mxfp4_quantize` "
            "(see `scripts/quantize_kat_mxfp4.py`).\n"
            "Everything else (self_attn, norms, shared experts, embed, "
            "lm_head) is unchanged BF16.\n")
    print("FINALIZE-OK")


if __name__ == "__main__":
    sys.exit(main())
