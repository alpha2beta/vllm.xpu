#!/usr/bin/env python3
"""Fetch small EXL3 MoE tensor fixtures from a checkpoint (P0-03 / P0-04 / P0-08).

Downloads only the named tensors via safetensors byte ranges, so a 15 GB
checkpoint becomes a ~2 MB fixture set. Useful for CPU oracles, Triton/ESIMD
reconstruction tests and later kernel unit tests.

Outputs (default ``fixtures/exl3-moe/``):
  moe-fixtures.safetensors   selected tensors, checkpoint shapes preserved
  moe-fixtures-manifest.json provenance: repo/revision/commit, per-tensor shard
                             and byte range, sha256 of the raw payload

Example:
  .venv/bin/python scripts/fetch_exl3_fixtures.py \
      --repo turboderp/Qwen3.5-35B-A3B-exl3 --revision 3.09bpw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "fixtures" / "exl3-moe"

DTYPES = {
    "F16": np.float16,
    "BF16": None,  # handled via uint16 view
    "F32": np.float32,
    "I16": np.int16,
    "I32": np.int32,
    "U8": np.uint8,
    "I8": np.int8,
    "I64": np.int64,
}


def default_selection() -> list[str]:
    """Two layers (GDN layer 0, full-attention layer 3), two experts, all proj."""
    names: list[str] = []
    for layer in (0, 3):
        for expert in (0, 255):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for leaf in ("suh", "svh", "mcg", "trellis"):
                    names.append(
                        f"model.language_model.layers.{layer}.mlp.experts."
                        f"{expert}.{proj}.{leaf}"
                    )
        # Router, shared-expert gate and the shared expert itself.
        names.append(f"model.language_model.layers.{layer}.mlp.gate.weight")
        names.append(f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for leaf in ("suh", "svh", "mcg", "trellis"):
                names.append(
                    f"model.language_model.layers.{layer}.mlp.shared_expert."
                    f"{proj}.{leaf}"
                )
    names.append("model.language_model.embed_tokens.weight")
    names.append("model.language_model.layers.0.input_layernorm.weight")
    return names


def to_numpy(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        array = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    else:
        array = np.frombuffer(raw, dtype=DTYPES[dtype]).reshape(shape)
    return array


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--tensor",
        action="append",
        default=None,
        help="explicit tensor name (repeatable); defaults to the standard MoE set",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="refuse any single tensor larger than this",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from audit_moe_checkpoint import HeaderSource  # local helper, same range logic

    source = HeaderSource(args.repo, args.revision, None)
    index = json.loads(source.read_text("model.safetensors.index.json"))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    headers = {shard: source.read_header(shard) for shard in shards}
    tensors = {name: (meta, shard) for shard, h in headers.items() for name, meta in h.items()}

    wanted = args.tensor or default_selection()
    payload: dict[str, np.ndarray] = {}
    manifest: list[dict] = []
    for name in wanted:
        if name not in tensors:
            print(f"skip (absent): {name}", file=sys.stderr)
            continue
        meta, shard = tensors[name]
        begin, end = meta["data_offsets"]
        size = end - begin
        if size > args.max_bytes:
            print(f"skip (too large, {size} B): {name}", file=sys.stderr)
            continue
        raw = source.read_tensor_bytes(shard, begin, size)
        if len(raw) != size:
            raise SystemExit(f"{name}: short read {len(raw)} != {size}")
        payload[name] = to_numpy(raw, meta["dtype"], meta["shape"])
        manifest.append(
            {
                "name": name,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "shard": shard,
                "data_offsets": [begin, end],
                "n_bytes": size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        print(f"got {name} {meta['dtype']} {meta['shape']} {size} B", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file

    out_path = args.out_dir / "moe-fixtures.safetensors"
    save_file(payload, str(out_path))

    commit = None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(args.repo, revision=args.revision)
        commit = info.sha
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not resolve commit: {exc}", file=sys.stderr)

    manifest_path = args.out_dir / "moe-fixtures-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repo": args.repo,
                "revision": args.revision,
                "commit": commit,
                "source": "HF safetensors byte-range reads (bodies only, no full download)",
                "fixture_file": out_path.name,
                "fixture_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                "tensor_count": len(manifest),
                "tensors": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out_path} ({len(manifest)} tensors)")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
