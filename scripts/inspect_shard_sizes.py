#!/usr/bin/env python3
"""Fetch safetensors headers over HTTP range requests and report tensor sizes
by logical group, without downloading the weight data.

Usage:
    python scripts/inspect_shard_sizes.py <repo_id> [revision]

Writes logs/model/<repo>-size-breakdown.json and prints a summary table.
"""
import collections
import json
import struct
import sys
import urllib.request
from pathlib import Path

DTYPE_BYTES = {
    "F4_E2M1": 0.5,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "F32": 4,
    "I32": 4,
    "I64": 8,
}


def group_of(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision_tower (bf16)"
    if name.startswith("mtp."):
        return "mtp_draft (bf16)"
    if ".mlp.experts." in name:
        return "moe_routed_experts (mx fp4)"
    if name.startswith("lm_head"):
        return "lm_head"
    if "embed_tokens" in name:
        return "embed_tokens"
    if "self_attn" in name:
        return "self_attn (bf16)"
    if "shared_expert" in name:
        return "shared_expert (bf16)"
    if "linear_attn" in name:
        return "linear_attn/gdn (bf16)"
    if "layernorm" in name or name.endswith(".norm") or ".norm." in name:
        return "norms"
    if "mlp.gate" in name or "mlp.up" in name or "mlp.down" in name:
        return "dense_mlp (mx fp4?)"
    return "other"


def fetch(url: str, offset: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def read_header(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"}), timeout=60) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    raw = fetch(url, 8, n)
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    return hdr


def main() -> int:
    repo = sys.argv[1]
    revision = sys.argv[2] if len(sys.argv) > 2 else "main"
    index_url = f"https://huggingface.co/{repo}/raw/{revision}/model.safetensors.index.json"
    with urllib.request.urlopen(index_url, timeout=60) as r:
        index = json.loads(r.read())
    weight_map: dict = index["weight_map"]

    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    groups = collections.Counter()
    group_tensors = collections.Counter()
    total = 0
    shard_totals = {}
    for shard in sorted(set(weight_map.values())):
        hdr = read_header(f"{base}/{shard}")
        shard_bytes = 0
        for name, meta in hdr.items():
            shape = meta["shape"]
            dtype = meta["dtype"]
            n = 1
            for s in shape:
                n *= s
            mult = DTYPE_BYTES.get(dtype)
            if mult is None:
                raise SystemExit(f"unknown dtype {dtype} for {name}")
            nbytes = int(n * mult)
            g = group_of(name)
            groups[g] += nbytes
            group_tensors[g] += 1
            shard_bytes += nbytes
        shard_totals[shard] = shard_bytes
        total += shard_bytes
        print(f"{shard}: {shard_bytes / 1e9:.3f} GB")

    print(f"\nTOTAL weights: {total / 1e9:.3f} GB ({total / (1 << 30):.2f} GiB)\n")
    print(f"{'group':32s} {'tensors':>8s} {'GB':>10s} {'GiB':>8s} {'share':>7s}")
    for g, b in groups.most_common():
        print(f"{g:32s} {group_tensors[g]:8d} {b / 1e9:10.3f} {b / (1 << 30):8.3f} {100 * b / total:6.1f}%")

    out_dir = Path(__file__).resolve().parent.parent / "logs" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = repo.replace("/", "-")
    payload = {
        "repo": repo,
        "revision": revision,
        "total_bytes": total,
        "groups_bytes": dict(groups),
        "groups_tensors": dict(group_tensors),
        "shard_bytes": shard_totals,
    }
    (out_dir / f"{safe}-size-breakdown.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir}/{safe}-size-breakdown.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
