#!/usr/bin/env python3
"""E0.3: read-only audit of a local EXL3 checkpoint (headers only, no full weight load).

Usage:
    .venv/bin/python scripts/audit_exl3_checkpoint.py \
        --model models/turboderp-Qwen3.8-27B-exl3-4.00bpw \
        --out logs/<run>/checkpoint-audit.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

DTYPE_BYTES = {
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "I64": 8, "U64": 8, "F64": 8,
    "U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}


def group_of(name: str) -> str:
    if ".visual." in name or ".vision" in name or name.startswith("visual"):
        return "vision"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if "embed_tokens" in name:
        return "embeddings"
    if name.startswith("lm_head"):
        return "lm_head"
    if ".language_model.layers." in name or name.startswith("model.language_model."):
        return "decoder_layers"
    if name in ("model.norm.weight",):
        return "other"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors import safe_open

    model = Path(args.model)
    out_path = Path(args.out) if args.out else model / "checkpoint-audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"model": str(model), "checks": {}, "warnings": []}
    cfg = json.loads((model / "config.json").read_text())
    quant = json.loads((model / "quantization_config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    report["architectures"] = cfg.get("architectures")
    report["model_type"] = cfg.get("model_type")
    report["quant_top"] = {k: quant.get(k) for k in
                           ("quant_method", "version", "bits", "head_bits",
                            "codebook", "mtp_bits", "out_scales", "calibration")}
    storage = quant.get("tensor_storage", {})
    report["tensor_storage_modules"] = len(storage)
    bpw = collections.Counter(
        v.get("bits_per_weight") for v in storage.values() if "bits_per_weight" in v)
    report["tensor_storage_bpw"] = dict(sorted(bpw.items(), key=lambda kv: str(kv[0])))

    wm: dict = index["weight_map"]
    report["indexed_tensors"] = len(wm)
    report["index_total_size"] = index.get("metadata", {}).get("total_size")

    # Shard presence + header scan (no tensor data loaded).
    totals = collections.Counter()
    counts = collections.Counter()
    suffix = collections.Counter()
    trellis_K = collections.Counter()
    trellis_shapes: dict = {}
    scale_dtypes = collections.Counter()
    errors: list[str] = []
    shard_files = sorted(set(wm.values()))
    report["shard_files"] = shard_files
    seen: set[str] = set()
    for shard in shard_files:
        fp = model / shard
        if not fp.is_file():
            errors.append(f"missing shard file: {shard}")
            continue
        with safe_open(fp, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            seen.update(keys)
            for key in keys:
                s = f.get_slice(key)
                shape = list(s.get_shape())
                dtype = s.get_dtype()
                suffix[key.rsplit(".", 1)[-1]] += 1
                nbytes = DTYPE_BYTES.get(dtype)
                if nbytes is None:
                    errors.append(f"unknown dtype {dtype} for {key}")
                    continue
                size = nbytes
                for d in shape:
                    size *= d
                g = group_of(key)
                totals[g] += size
                counts[g] += 1
                if key.endswith((".suh", ".svh")):
                    scale_dtypes[dtype] += 1
                if key.endswith(".trellis"):
                    if len(shape) == 3 and shape[2] % 16 == 0:
                        trellis_K[shape[2] // 16] += 1
                        if len(trellis_shapes) < 8:
                            trellis_shapes[key] = shape
                    else:
                        errors.append(f"unexpected trellis shape {key}={shape}")

    indexed = set(wm)
    report["tensors_missing_from_shards"] = sorted(indexed - seen)[:10]
    report["tensors_missing_count"] = len(indexed - seen)
    report["tensors_extra_in_shards"] = sorted(seen - indexed)[:10]
    report["tensors_extra_count"] = len(seen - indexed)
    report["group_bytes"] = dict(totals)
    report["group_counts"] = dict(counts)
    report["group_gib"] = {k: round(v / 2**30, 4) for k, v in totals.items()}
    report["total_payload_bytes"] = sum(totals.values())
    report["suffix_counts"] = dict(suffix)
    report["trellis_K_counts"] = dict(trellis_K)
    report["trellis_shape_samples"] = trellis_shapes
    report["scale_dtypes"] = dict(scale_dtypes)

    # Structural expectations for this checkpoint.
    report["checks"]["lm_head_quantized"] = any(
        k.startswith("lm_head.") and k.endswith(".trellis") for k in seen)
    report["checks"]["in_proj_ba_unquantized"] = all(
        ("in_proj_a.weight" in k or "in_proj_b.weight" in k) for k in seen
        if "in_proj_a" in k or "in_proj_b" in k) if any(
        "in_proj_a" in k or "in_proj_b" in k for k in seen) else False
    report["checks"]["fused_qkv_present"] = any(
        k.endswith("in_proj_qkv.trellis") for k in seen)
    report["checks"]["qkvz_split_present"] = any(
        k.endswith("in_proj_z.trellis") for k in seen)

    # Stop tokens / chat template (report only; server behavior verified later).
    tcfg = json.loads((model / "tokenizer_config.json").read_text())
    gen = json.loads((model / "generation_config.json").read_text())
    report["tokenizer"] = {
        "bos_token_id_cfg": cfg.get("text_config", {}).get("bos_token_id"),
        "eos_token_id_cfg": cfg.get("text_config", {}).get("eos_token_id"),
        "eos_gen": gen.get("eos_token_id"),
        "chat_template_file": (model / "chat_template.jinja").is_file(),
        "tokenizer_eos_id": tcfg.get("eos_token"),
    }
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model), trust_remote_code=False)
        msgs = [{"role": "user", "content": "Hello"}]
        report["tokenizer"]["apply_chat_template_ok"] = bool(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        report["tokenizer"]["decoded_eos"] = [
            tok.decode([i]) if isinstance(i, int) else None
            for i in ([gen["eos_token_id"]] if isinstance(gen.get("eos_token_id"), int)
                      else (gen.get("eos_token_id") or []))]
    except Exception as e:  # noqa - audit must not fail on optional dep
        report["tokenizer"]["apply_chat_template_ok"] = False
        report["tokenizer"]["error"] = repr(e)[:300]

    # Supplied checksums: record, do not claim validation (format not specified).
    crc = model / "crc32.txt"
    if crc.is_file():
        report["crc32_txt"] = crc.read_text().splitlines()[:10]
        report["warnings"].append("crc32.txt recorded only; checksum format not validated")
    else:
        report["crc32_txt"] = None

    report["errors"] = errors
    report["ok"] = not errors and report["tensors_missing_count"] == 0
    out_path.write_text(json.dumps(report, indent=2))
    print(f"payload bytes: {report['total_payload_bytes']} "
          f"({report['total_payload_bytes']/2**30:.3f} GiB)")
    print(f"groups GiB: {report['group_gib']}")
    print(f"trellis K counts: {dict(trellis_K)}")
    print(f"errors: {errors[:5] if errors else 'none'}; "
          f"missing={report['tensors_missing_count']} extra={report['tensors_extra_count']}")
    print(f"wrote {out_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
