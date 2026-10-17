#!/usr/bin/env python3
"""quant_merge_mtp.py -- re-inject the MTP draft block into a GPTQ output dir.

Why this exists
---------------
transformers' `Qwen3_5ForConditionalGeneration` declares
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, i.e. the single MTP draft
layer is *not* a parameter of the HF model class. The source checkpoint ships it
as a standalone `mtp-bf16.safetensors` (15 tensors, BF16). A plain GPTQModel run
therefore loads language+vision, quantizes the language linears, and **saves
without mtp**.

The reference artifact (`Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`) keeps the
15 `mtp.*` tensors in BF16 (see its README: "the single MTP draft layer is
preserved unquantized in BF16"). This script restores that parity: it copies
the source `mtp-bf16.safetensors` next to the quantized shards and adds its keys
to `model.safetensors.index.json`, so index-based loaders (vLLM, transformers)
pick the block up again.

Idempotent: re-running replaces the copied file and re-merges the index keys.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path


def header_keys(path: Path) -> list[str]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return [k for k in hdr if k != "__metadata__"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-mtp", required=True,
                    help="source mtp-bf16.safetensors (BF16 draft block)")
    ap.add_argument("--out", required=True,
                    help="quantized output dir (holds model.safetensors.index.json)")
    ap.add_argument("--name", default="mtp-bf16.safetensors",
                    help="filename to copy the block to inside --out")
    args = ap.parse_args()

    src = Path(args.src_mtp)
    out = Path(args.out)
    if not src.is_file():
        print(f"ERROR: missing {src}", file=sys.stderr)
        return 2

    idx_path = out / "model.safetensors.index.json"
    if not idx_path.is_file():
        print(f"ERROR: missing {idx_path} (is --out a saved model dir?)", file=sys.stderr)
        return 2

    keys = header_keys(src)
    mtp_keys = [k for k in keys if k.startswith("mtp.")]
    if len(mtp_keys) != len(keys):
        print(f"WARNING: {len(keys) - len(mtp_keys)} non-mtp keys in {src.name}", file=sys.stderr)

    # 1) copy the block file
    dst = out / args.name
    if dst.resolve() != src.resolve():
        shutil.copy2(src, dst)
    print(f"copied {len(keys)} tensors -> {dst}")

    # 2) merge into the index
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]
    added = 0
    for k in keys:
        if wm.get(k) == args.name:
            continue
        if k in wm:
            print(f"  note: {k} was mapped to {wm[k]}, remapping to {args.name}")
        wm[k] = args.name
        added += 1
    idx.setdefault("metadata", {})
    total = idx["metadata"].get("total_size")
    if isinstance(total, int):
        # true byte size of the block, from its own header
        with src.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        block_bytes = sum(
            v["data_offsets"][1] - v["data_offsets"][0]
            for k, v in hdr.items() if k != "__metadata__"
        )
        idx["metadata"]["total_size"] = total + block_bytes
    idx_path.write_text(json.dumps(idx, indent=2))
    print(f"merged {added} new mtp keys into {idx_path} (map now {len(wm)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
