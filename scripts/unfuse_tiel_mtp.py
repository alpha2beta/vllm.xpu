#!/usr/bin/env python3
"""Unfuse Tiel's MTP routed experts for the vLLM qwen3_5_mtp draft loader.

Tiel stores MTP experts fused (like its trunk):
  mtp.layers.0.mlp.experts.gate_up_proj  [256,1024,2048] BF16
  mtp.layers.0.mlp.experts.down_proj     [256,2048,512]  BF16
The vLLM MTP draft model builds standard unfused w13/w2 expert params, so it
needs per-expert split tensors (same convention as the Qwen3.6 reference):
  mtp.layers.0.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight  BF16

CPU-only reshape pass (no XPU needed). Rewrites output shard 30 (drops the 2
fused tensors), appends split experts as shards 32+ (1 GiB each), updates the
index + config.json ignore list + SHA256SUMS. Idempotent: skips if the split
tensors already exist in the index.

Usage: ./scripts/unfuse_tiel_mtp.py
"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(
    REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4")
N_EXPERTS = 256
SHARD_TARGET_BYTES = 1 << 30

FUSED_GU = "mtp.layers.0.mlp.experts.gate_up_proj"
FUSED_DOWN = "mtp.layers.0.mlp.experts.down_proj"
BASE = "mtp.layers.0.mlp.experts"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rebuild_tail_by_position(weight_map):
    """Repair path: files on disk are already consistently numbered
    (model-00001..N-of-N) but the index still references stale names.
    Rebuild the index by POSITION: the i-th indexed file (sorted) maps to
    the i-th file on disk (sorted). Safe here because the content set is
    complete (all 63,321 tensors resolve to existing files)."""
    import re
    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    files = sorted(f for f in os.listdir(OUTDIR)
                   if re.fullmatch(r"model-\d+-of-\d+\.safetensors", f))
    old_names = sorted({fn for fn in weight_map.values()})
    assert len(files) == len(old_names), (len(files), len(old_names))
    remap = dict(zip(old_names, files))
    final_map = {k: remap[fn] for k, fn in weight_map.items()}
    # every indexed file must exist and every tensor must be openable
    from safetensors import safe_open
    seen = set()
    for fn in sorted(set(final_map.values())):
        with safe_open(os.path.join(OUTDIR, fn), framework="pt") as h:
            seen.update(h.keys())
    assert set(final_map) == seen, (
        f"tensor/file mismatch: index={len(final_map)} on-disk={len(seen)}")
    total = sum(os.path.getsize(os.path.join(OUTDIR, f)) for f in files)
    with open(idx_path, "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": final_map}, f)
    return final_map, len(files)


def rebuild_tail(weight_map):
    """Global consistent renumbering + index refresh (full-pass path)."""
    import re
    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    files = sorted(f for f in os.listdir(OUTDIR)
                   if re.fullmatch(r"model-\d+-of-(?:XXXXX|\d+)\.safetensors", f))
    n_shards = len(files)
    tmp_map = {}
    for i, fn in enumerate(files, 1):
        tmp_map[fn] = f"model-{i:05d}-of-{n_shards:05d}.safetensors.tmp"
    for old, tmp in tmp_map.items():
        os.rename(os.path.join(OUTDIR, old), os.path.join(OUTDIR, tmp))
    new_names = {}
    for old, tmp in tmp_map.items():
        final = tmp[: -len(".tmp")]
        os.rename(os.path.join(OUTDIR, tmp), os.path.join(OUTDIR, final))
        new_names[old] = final
    final_map = {k: new_names[fn] for k, fn in weight_map.items()}
    # sanity: every indexed file exists
    missing = sorted({fn for fn in final_map.values()}
                     - set(os.listdir(OUTDIR)))
    assert not missing, f"indexed files missing on disk: {missing[:3]}"
    total = sum(os.path.getsize(os.path.join(OUTDIR, f))
                for f in new_names.values())
    with open(idx_path, "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": final_map}, f)
    return final_map, n_shards


def main():
    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]
    # Rebuild the tail (rename pass) whenever shard file names disagree
    # with the index (e.g. a previous pass appended -of-XXXXX shards but
    # exited before renaming). Full re-unfuse only if split tensors are
    # absent; otherwise jump straight to the rename tail.
    import re as _re

    def shard_files():
        return sorted(f for f in os.listdir(OUTDIR)
                      if _re.fullmatch(r"model-\d+-of-(?:XXXXX|\d+)"
                                       r"\.safetensors", f))

    def consistent():
        files = shard_files()
        n = len(files)
        want = {f"model-{i:05d}-of-{n:05d}.safetensors"
                for i in range(1, n + 1)}
        if set(files) != want:
            return False
        return all(_re.fullmatch(r"model-\d+-of-%05d\.safetensors" % n, fn)
                   for fn in weight_map.values())

    split_done = f"{BASE}.0.gate_proj.weight" in weight_map
    if split_done and consistent():
        print("already unfused — nothing to do")
        return 0
    if split_done and not consistent():
        # tail-only repair: fused tensors already converted and old shard
        # rewritten; just rebuild a consistent numbering + index. Map by
        # POSITION (sorted order), not by old name, since a prior pass may
        # have renamed files without updating the index.
        print("split tensors present but shard set inconsistent — "
              "repairing numbering", flush=True)
        rebuild_tail_by_position(weight_map)
        return 0

    gu_file = weight_map[FUSED_GU]
    dn_file = weight_map[FUSED_DOWN]
    assert gu_file == dn_file, (gu_file, dn_file)
    print(f"reading fused tensors from {gu_file} ...", flush=True)
    blobs = load_file(os.path.join(OUTDIR, gu_file))
    gu = blobs[FUSED_GU]
    dn = blobs[FUSED_DOWN]
    assert tuple(gu.shape) == (N_EXPERTS, 1024, 2048), tuple(gu.shape)
    assert tuple(dn.shape) == (N_EXPERTS, 2048, 512), tuple(dn.shape)
    rest = {k: v for k, v in blobs.items()
            if k not in (FUSED_GU, FUSED_DOWN)}
    del blobs

    # rewrite shard 30 without the fused tensors
    keep = {k: v.contiguous() for k, v in rest.items()}
    save_file(keep, os.path.join(OUTDIR, gu_file))
    for k in (FUSED_GU, FUSED_DOWN):
        del weight_map[k]
    print(f"rewrote {gu_file}: dropped 2 fused, kept {len(keep)}",
          flush=True)

    # append split experts as new shards (drop -of- total; fixed at the end)
    existing = sorted(f for f in os.listdir(OUTDIR)
                      if f.startswith("model-") and f.endswith(".safetensors"))
    nxt = max(int(f.split("-")[1]) for f in existing) + 1
    pending, pending_bytes = {}, 0

    def flush():
        nonlocal pending, pending_bytes, nxt
        if not pending:
            return
        fn = f"model-{nxt:05d}-of-XXXXX.safetensors"
        save_file({k: v.contiguous() for k, v in pending.items()},
                  os.path.join(OUTDIR, fn))
        for k in pending:
            weight_map[k] = fn
        print(f"  wrote {fn}: {len(pending)} tensors, "
              f"{pending_bytes / 2**30:.2f} GiB", flush=True)
        pending, pending_bytes = {}, 0
        nxt += 1

    for e in range(N_EXPERTS):
        for proj, src in (("gate_proj", gu[e][:512]),
                          ("up_proj", gu[e][512:]),
                          ("down_proj", dn[e])):
            name = f"{BASE}.{e}.{proj}.weight"
            t = src.clone()
            pending[name] = t
            pending_bytes += t.nelement() * t.element_size()
        if pending_bytes >= SHARD_TARGET_BYTES:
            flush()
        if (e + 1) % 64 == 0:
            print(f"  ... {e + 1}/{N_EXPERTS} experts", flush=True)
    del gu, dn
    flush()

    # finalize via the shared tail (consistent numbering + index refresh)
    final_map, n_shards = rebuild_tail(weight_map)
    # sanity: no fused MTP experts left, all split present
    assert not any("gate_up_proj" in k or
                   (k.startswith("mtp.") and k.endswith("experts.down_proj"))
                   for k in final_map)
    n_split = sum(1 for k in final_map if k.startswith(BASE + "."))
    assert n_split == N_EXPERTS * 3, n_split
    total = sum(os.path.getsize(os.path.join(OUTDIR, f))
                for f in os.listdir(OUTDIR) if f.endswith(".safetensors"))
    with open(idx_path, "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": final_map}, f)

    n_split = sum(1 for k in final_map if k.startswith(BASE + "."))
    assert n_split == N_EXPERTS * 3, n_split

    # config.json ignore: drop fused entries, add split ones (minus .weight)
    cfg_path = os.path.join(OUTDIR, "config.json")
    cfg = json.load(open(cfg_path))
    ign = [i for i in cfg["quantization_config"]["ignore"]
           if i not in (FUSED_GU, FUSED_DOWN)]
    for e in range(N_EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ign.append(f"{BASE}.{e}.{proj}")
    cfg["quantization_config"]["ignore"] = sorted(set(ign))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"ignore list: {len(set(ign))} entries")

    # SHA256SUMS: rebuild from current shard files (old names may be stale)
    sums_path = os.path.join(OUTDIR, "SHA256SUMS")
    with open(sums_path, "w") as f:
        for fn in sorted(os.listdir(OUTDIR)):
            if fn.endswith(".safetensors"):
                h = sha256(os.path.join(OUTDIR, fn))
                f.write(f"{h}  {fn}\n")
    print("  SHA256SUMS rebuilt")
    total = sum(os.path.getsize(os.path.join(OUTDIR, f))
                for f in os.listdir(OUTDIR) if f.endswith(".safetensors"))
    print(f"UNFUSE-OK: {len(final_map)} tensors, {n_shards} shards, "
          f"{total / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
