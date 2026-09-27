#!/usr/bin/env python3
"""Finalize models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4/ after quantization:
config.json (Tiel config + MXFP4 quantization_config), tokenizer sidecars,
and schema verification against the reference layout.

Usage: ./scripts/finalize_tiel_mxfp4.py
"""
import json
import os
import re
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(
    REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4")
SRC_META = ("/home/yanchun/Projects/AInfer/models/"
            "Tiel-Coder-35B-A3B-Genesis-Hermes")
REF_CFG = os.path.join(
    REPO_ROOT, "models", "Qwen3.6-35B-A3B-MXFP4", "config.json")


def main():
    from safetensors import safe_open
    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    assert os.path.exists(idx_path), "run quantize_tiel_mxfp4.py first"
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

    # shape spot-check on a few shards (first file with experts + linear attn)
    checked = 0
    for fn in sorted(set(weight_map.values())):
        path = os.path.join(OUTDIR, fn)
        with safe_open(path, framework="pt") as h:
            for k in h.keys():
                if k.endswith(".weight_packed") and (
                        ".experts.0.gate_proj" in k or "in_proj_qkv" in k):
                    sl = h.get_slice(k)
                    print(" ", sl.get_shape(), k)
                    checked += 1
                    if checked >= 4:
                        break
        if checked >= 4:
            break

    # no leftover fused TEXT-TOWER tensors (MTP fused experts intentionally
    # stay BF16, mirroring the reference)
    leftovers = [n for n in list(packed) + list(plain)
                 if ".language_model.layers." in n and
                 ("gate_up_proj" in n
                  or re.search(r"experts\.down_proj$", n))]
    assert not leftovers, f"unconverted fused tensors: {leftovers[:5]}"
    print("no fused leftovers (mtp fused experts kept as BF16 by design)")

    # --- ignore list: every BF16 module (reference convention: full tensor
    # name minus trailing .weight)
    ignore = set()
    for n in plain:
        ignore.add(n[: -len(".weight")] if n.endswith(".weight") else n)
    ignore = sorted(ignore)
    print(f"ignore entries: {len(ignore)}")

    # --- config.json
    tiel_cfg = json.load(open(os.path.join(SRC_META, "config.json")))
    ref_q = json.load(open(REF_CFG))["quantization_config"]
    assert ref_q["quant_method"] == "compressed-tensors"
    assert ref_q["format"] == "mxfp4-pack-quantized"
    tiel_cfg["quantization_config"] = {
        "quant_method": ref_q["quant_method"],
        "format": ref_q["format"],
        "config_groups": ref_q["config_groups"],
        "ignore": ignore,
    }
    with open(os.path.join(OUTDIR, "config.json"), "w") as f:
        json.dump(tiel_cfg, f, indent=2)
    print("wrote config.json with quantization_config "
          f"(ignore={len(ignore)})")

    # --- sidecars (same file set as the reference + Tiel chat template)
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "chat_template.jinja", "generation_config.json",
               "preprocessor_config.json", "video_preprocessor_config.json",
               "configuration.json"]:
        src = os.path.join(SRC_META, fn)
        if fn == "configuration.json":
            src = os.path.join(REPO_ROOT, "models", "Qwen3.6-35B-A3B-MXFP4",
                               "configuration.json")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUTDIR, fn))
            print(f"copied {fn}")
    # README provenance
    with open(os.path.join(OUTDIR, "README.md"), "w") as f:
        f.write(
            "# Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4\n\n"
            "MXFP4 (compressed-tensors `mxfp4-pack-quantized`, E2M1 group-32 "
            "symmetric, E8M0 scales) quantization of the text tower's routed "
            "MoE experts + linear-attention projections, in the exact "
            "safetensors schema of "
            "`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` for the vLLM XPU "
            "backend (`CompressedTensorsW4A4Mxfp4` + `XPUExpertsMxFp4`).\n\n"
            "Source: `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-GGUF-dequantized` "
            "(BF16 safetensors, fused `experts.gate_up_proj`/`down_proj` "
            "unfused per expert here).\n"
            "Quantizer: `torch.ops.vllm.xpu_mxfp4_quantize` "
            "(see `scripts/quantize_tiel_mxfp4.py`).\n"
            "Everything else (self_attn, norms, shared experts, embed, "
            "lm_head, vision, MTP) is unchanged BF16.\n")
    print("FINALIZE-OK")


if __name__ == "__main__":
    sys.exit(main())
