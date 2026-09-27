#!/usr/bin/env python3
"""Quantize Tiel's unfused MTP draft experts (BF16) to MXFP4 in-place.

Task 11.1.2: the 768 MTP expert tensors
  mtp.layers.0.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight  BF16
  gate/up [512,2048], down [2048,512]
become
  ...{proj}.weight_packed  U8 ([512,1024] / [2048,256])
  ...{proj}.weight_scale   U8 E8M0 ([512,64] / [2048,16])
in the exact schema CompressedTensorsW4A4Mxfp4.create_weights() expects
(same as the trunk: low-nibble-even packing, group-32 symmetric E2M1).

Quantizer: torch.ops.vllm.xpu_mxfp4_quantize (inference kernel's own op).
Source tensors are read from the LOCAL checkpoint (staging was deleted);
no network, no XPU bulk memory (one expert-proj at a time).

Reversible: backs up config.json + model.safetensors.index.json to
*.pre-mtp-mxfp4.bak on first run; idempotent (skips if packed tensors exist).
Does NOT delete the BF16 shards until the new shards verify — old MTP shards
are rewritten without the 768 BF16 tensors only after all packed shards flush.

Updates (task 11.1.3/11.1.4 scope, done here atomically):
  - index: -768 BF16, +1536 packed/scale; consistent -of-N renumbering
  - config.json ignore: drop the 768 MTP expert entries, keep 17 non-expert
  - SHA256SUMS rebuilt

Usage:
  ./scripts/quantize_tiel_mtp_mxfp4.py [--dry-run] [--validate-every 64]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(
    REPO_ROOT, "models", "Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4")
BASE = "mtp.layers.0.mlp.experts"
N_EXPERTS = 256
PROJS = ("gate_proj", "up_proj", "down_proj")
EXPECTED = {"gate_proj": (512, 2048), "up_proj": (512, 2048),
            "down_proj": (2048, 512)}
SHARD_TARGET_BYTES = 1 << 30


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def quantize_mxfp4(w2d_xpu):
    assert w2d_xpu.dim() == 2 and w2d_xpu.size(1) % 32 == 0
    q, s = torch.ops.vllm.xpu_mxfp4_quantize(w2d_xpu)
    return q.view(torch.uint8).to("cpu"), s.view(torch.uint8).to("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-every", type=int, default=64)
    args = ap.parse_args()

    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    cfg_path = os.path.join(OUTDIR, "config.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    packed_done = sum(1 for k in weight_map
                      if k.startswith(BASE + ".") and k.endswith(".weight_packed"))
    bf16_left = [k for k in weight_map
                 if k.startswith(BASE + ".") and k.endswith(".weight")]
    print(f"MTP packed pairs present: {packed_done // 2}, BF16 left: {len(bf16_left)}")
    if packed_done == 768 * 2 and not bf16_left:
        print("already quantized — nothing to do")
        return 0
    assert len(bf16_left) == 768, f"expected 768 BF16 MTP experts, got {len(bf16_left)}"

    if args.dry_run:
        print("dry-run: would quantize 768 BF16 -> 1536 packed/scale, "
              "update index + ignore + SHA256SUMS")
        return 0

    import vllm_xpu_kernels  # noqa: F401 (registers torch.ops._xpu_C.*)
    import vllm._xpu_ops  # noqa: F401 (registers torch.ops.vllm.*)
    from vllm_xpu_kernels.moe_utils import dequant_mxfp4
    assert torch.xpu.is_available(), "XPU required for xpu_mxfp4_quantize"

    # backups (first run only)
    for p in (idx_path, cfg_path):
        bak = p + ".pre-mtp-mxfp4.bak"
        if not os.path.exists(bak):
            import shutil
            shutil.copy(p, bak)
            print(f"backup: {bak}")

    # group BF16 source tensors by file to minimize opens
    by_file = {}
    for k in bf16_left:
        by_file.setdefault(weight_map[k], []).append(k)

    # load all 768 BF16 (1.6 GiB, fits in host RAM; moved to XPU one at a time)
    src = {}
    for fn in sorted(by_file):
        blobs = load_file(os.path.join(OUTDIR, fn))
        for k in by_file[fn]:
            t = blobs[k]
            assert tuple(t.shape) == EXPECTED[k.rsplit(".", 2)[-2]], (k, tuple(t.shape))
            src[k] = t
        del blobs
    print(f"loaded {len(src)} BF16 source tensors", flush=True)

    # quantize -> new shards
    existing = sorted(f for f in os.listdir(OUTDIR)
                      if re.fullmatch(r"model-\d+-of-(?:XXXXX|\d+)\.safetensors", f))
    nxt = max(int(f.split("-")[1]) for f in existing) + 1
    pending, pending_bytes = {}, 0
    n_val, max_rel, sum_rel = 0, 0.0, 0.0
    t0 = time.time()

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
        for proj in PROJS:
            name = f"{BASE}.{e}.{proj}.weight"
            cpu_ref = src.pop(name)
            t = cpu_ref.to("xpu", dtype=torch.bfloat16)
            q, s = quantize_mxfp4(t)
            torch.xpu.synchronize()
            if (e * 3 + PROJS.index(proj) + 1) % args.validate_every == 0:
                n_val += 1
                with torch.no_grad():
                    qr = q.to("xpu").view(torch.float4_e2m1fn_x2)
                    sr = s.to("xpu").view(torch.float8_e8m0fnu)
                    dq = dequant_mxfp4(qr, sr).float().cpu()
                ref = cpu_ref.float()
                rel = ((dq - ref).abs().mean().item()
                       / ref.abs().mean().item())
                max_rel = max(max_rel, rel)
                sum_rel += rel
                if n_val <= 3 or n_val % 4 == 0:
                    print(f"  [val] {name}: mean-rel-err={rel:.5f}", flush=True)
                assert rel < 0.15, f"{name} rel-err {rel:.4f} >= 0.15"
                del dq, ref
            stem = name[: -len(".weight")]
            pending[stem + ".weight_packed"] = q
            pending_bytes += q.nelement()
            pending[stem + ".weight_scale"] = s
            pending_bytes += s.nelement()
            del q, s, t, cpu_ref
        if pending_bytes >= SHARD_TARGET_BYTES:
            flush()
        if (e + 1) % 64 == 0:
            print(f"  ... {e + 1}/{N_EXPERTS} experts "
                  f"({time.time() - t0:.0f}s)", flush=True)
    flush()
    assert not src, f"unconsumed sources: {len(src)}"
    print(f"in-conversion spot checks: {n_val}, max rel-err={max_rel:.5f}, "
          f"mean={sum_rel / max(1, n_val):.5f}", flush=True)

    # Drop the 768 BF16 entries, rewrite their old shards without them.
    old_files = sorted({fn for k, fn in list(weight_map.items())
                        if k.startswith(BASE + ".") and k.endswith(".weight")})
    # weight_map still points BF16 names at old files; collect then delete keys
    for k in bf16_left:
        del weight_map[k]
    for fn in old_files:
        path = os.path.join(OUTDIR, fn)
        blobs = load_file(path)
        keep = {k: v for k, v in blobs.items() if not k.startswith(BASE + ".")}
        del blobs
        if len(keep) == 0:
            os.remove(path)
            print(f"  removed empty {fn}")
        else:
            save_file({k: v.contiguous() for k, v in keep.items()}, path)
            print(f"  rewrote {fn}: kept {len(keep)}")

    # consistent renumbering + index refresh
    files = sorted(f for f in os.listdir(OUTDIR)
                   if re.fullmatch(r"model-\d+-of-(?:XXXXX|\d+)\.safetensors", f))
    n_shards = len(files)
    tmp_map = {fn: f"model-{i:05d}-of-{n_shards:05d}.safetensors.tmp"
               for i, fn in enumerate(files, 1)}
    for old, tmp in tmp_map.items():
        os.rename(os.path.join(OUTDIR, old), os.path.join(OUTDIR, tmp))
    new_names = {}
    for old, tmp in tmp_map.items():
        final = tmp[: -len(".tmp")]
        os.rename(os.path.join(OUTDIR, tmp), os.path.join(OUTDIR, final))
        new_names[old] = final
    # weight_map values may reference removed files (BF16 gone) or XXXXX names
    final_map = {}
    for k, fn in weight_map.items():
        assert fn in new_names, f"stale ref {k} -> {fn}"
        final_map[k] = new_names[fn]
    # sanity
    assert not any(k.startswith(BASE + ".") and k.endswith(".weight")
                   for k in final_map)
    assert sum(1 for k in final_map if k.startswith(BASE + ".")
               ) == 768 * 2, "MTP packed count"
    # every indexed tensor opens
    from safetensors import safe_open
    seen = set()
    for fn in sorted(set(final_map.values())):
        with safe_open(os.path.join(OUTDIR, fn), framework="pt") as h:
            seen.update(h.keys())
    assert set(final_map) == seen, "tensor/file mismatch"
    total = sum(os.path.getsize(os.path.join(OUTDIR, f))
                for f in new_names.values())
    with open(idx_path, "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": final_map}, f)
    print(f"index: {len(final_map)} tensors, {n_shards} shards, "
          f"{total / 2**30:.2f} GiB")

    # config.json ignore: drop 768 MTP expert entries, keep 17 non-expert
    cfg = json.load(open(cfg_path))
    ign = cfg["quantization_config"]["ignore"]
    before = len(ign)
    drop = {f"{BASE}.{e}.{p}" for e in range(N_EXPERTS) for p in PROJS}
    ign = sorted(set(ign) - drop)
    assert not any(i in drop for i in ign)
    assert sum(1 for i in ign if i.startswith("mtp")) == 17, \
        [i for i in ign if i.startswith("mtp")]
    cfg["quantization_config"]["ignore"] = ign
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"ignore: {before} -> {len(ign)} (dropped 768 MTP experts)")

    # SHA256SUMS rebuild
    with open(os.path.join(OUTDIR, "SHA256SUMS"), "w") as f:
        for fn in sorted(os.listdir(OUTDIR)):
            if fn.endswith(".safetensors"):
                f.write(f"{sha256(os.path.join(OUTDIR, fn))}  {fn}\n")
    print("SHA256SUMS rebuilt")
    print(f"QUANT-MTP-OK in {time.time() - t0:.0f}s "
          f"(post-hoc audit: scripts/audit_tiel_mxfp4.py --mtp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
