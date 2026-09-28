#!/usr/bin/env python3
"""Add an MXFP4 MTP draft block to models/KAT-Coder-V2.5-Dev-MXFP4/.

Source: SpectreSystems/KAT-Coder-V2.5-Dev-MTP model-00014-of-mtp.safetensors
(19 BF16 tensors, fused experts like Tiel's MTP block was).

One pass, no intermediate BF16 in the checkpoint:
  - 17 non-expert MTP tensors -> appended BF16, added to ignore list
  - fused gate_up [256,1024,2048] / down [256,2048,512] -> split per expert,
    MXFP4-quantized via torch.ops.vllm.xpu_mxfp4_quantize ->
    mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight_packed/.weight_scale
    (1536 tensors, same schema CompressedTensorsW4A4Mxfp4 expects)
  - config.json: text_config.mtp_num_hidden_layers 0 -> 1 (KAT base has no
    MTP; the draft builder reads this — 0 would build zero layers), ignore
    list gains the 17 non-expert entries (experts stay quantized)
  - index renumbered consistently, SHA256SUMS rebuilt

Reversible: backs up index + config.json to *.pre-kat-mtp.bak on first run;
idempotent (skips if MTP packed tensors already present).

Usage: ./scripts/quantize_kat_mtp_mxfp4.py [--validate-every 64]
  Log: logs/quantize-kat-mtp-mxfp4.log
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
STAGING = os.path.join(REPO_ROOT, "models", "_staging_kat_mtp")
MTP_SHARD = "model-00014-of-mtp.safetensors"
OUTDIR = os.path.join(
    REPO_ROOT, "models", "KAT-Coder-V2.5-Dev-MXFP4")
BASE = "mtp.layers.0.mlp.experts"
N_EXPERTS = 256
PROJS = ("gate_proj", "up_proj", "down_proj")
FUSED_GU = f"{BASE}.gate_up_proj"
FUSED_DOWN = f"{BASE}.down_proj"
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
    ap.add_argument("--validate-every", type=int, default=64)
    args = ap.parse_args()

    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    cfg_path = os.path.join(OUTDIR, "config.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    packed_done = sum(1 for k in weight_map
                      if k.startswith(BASE + ".") and k.endswith(".weight_packed"))
    if packed_done == N_EXPERTS * 3 * 2:
        print("MTP draft already quantized — nothing to do")
        return 0
    assert packed_done == 0, f"partial MTP state: {packed_done} packed"

    import vllm_xpu_kernels  # noqa: F401
    import vllm._xpu_ops  # noqa: F401
    from vllm_xpu_kernels.moe_utils import dequant_mxfp4
    assert torch.xpu.is_available(), "XPU required for xpu_mxfp4_quantize"

    for p in (idx_path, cfg_path):
        bak = p + ".pre-kat-mtp.bak"
        if not os.path.exists(bak):
            import shutil
            shutil.copy(p, bak)
            print(f"backup: {bak}")

    src_path = os.path.join(STAGING, MTP_SHARD)
    if not os.path.exists(src_path):
        print(f"MISSING MTP source shard: {src}")
        sys.exit(2)
    blobs = load_file(src_path)
    gu, dn = blobs.pop(FUSED_GU), blobs.pop(FUSED_DOWN)
    assert tuple(gu.shape) == (N_EXPERTS, 1024, 2048), tuple(gu.shape)
    assert tuple(dn.shape) == (N_EXPERTS, 2048, 512), tuple(dn.shape)
    rest = dict(blobs)
    del blobs
    print(f"fused experts: {tuple(gu.shape)}, {tuple(dn.shape)}; "
          f"non-expert MTP tensors: {len(rest)}", flush=True)
    nonexpert_names = sorted(rest)

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

    # non-expert BF16 first
    for k, v in sorted(rest.items()):
        pending[k] = v.contiguous()
        pending_bytes += v.nelement() * v.element_size()

    # experts: split + quantize one proj at a time (flat host RAM)
    cpu_gu, cpu_dn = gu, dn
    del gu, dn
    for e in range(N_EXPERTS):
        for proj, src2d in (("gate_proj", cpu_gu[e][:512]),
                            ("up_proj", cpu_gu[e][512:]),
                            ("down_proj", cpu_dn[e])):
            name = f"{BASE}.{e}.{proj}.weight"
            t = src2d.clone().to("xpu", dtype=torch.bfloat16)
            q, s = quantize_mxfp4(t)
            torch.xpu.synchronize()
            if (e * 3 + PROJS.index(proj) + 1) % args.validate_every == 0:
                n_val += 1
                with torch.no_grad():
                    qr = q.to("xpu").view(torch.float4_e2m1fn_x2)
                    sr = s.to("xpu").view(torch.float8_e8m0fnu)
                    dq = dequant_mxfp4(qr, sr).float().cpu()
                ref = src2d.float()
                rel = ((dq - ref).abs().mean().item()
                       / ref.abs().mean().item())
                max_rel = max(max_rel, rel)
                sum_rel += rel
                if n_val <= 3:
                    print(f"  [val] {name}: mean-rel-err={rel:.5f}", flush=True)
                assert rel < 0.15, f"{name} rel-err {rel:.4f} >= 0.15"
                del dq, ref
            stem = name[: -len(".weight")]
            pending[stem + ".weight_packed"] = q
            pending_bytes += q.nelement()
            pending[stem + ".weight_scale"] = s
            pending_bytes += s.nelement()
            del q, s, t, src2d
        if pending_bytes >= SHARD_TARGET_BYTES:
            flush()
        if (e + 1) % 64 == 0:
            print(f"  ... {e + 1}/{N_EXPERTS} experts "
                  f"({time.time() - t0:.0f}s)", flush=True)
    del cpu_gu, cpu_dn
    flush()
    print(f"spot checks: {n_val}, max rel-err={max_rel:.5f}, "
          f"mean={sum_rel / max(1, n_val):.5f}", flush=True)

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
    final_map = {}
    for k, fn in weight_map.items():
        assert fn in new_names, f"stale ref {k} -> {fn}"
        final_map[k] = new_names[fn]
    assert sum(1 for k in final_map if k.startswith(BASE + ".")) == N_EXPERTS * 3 * 2
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

    # config: mtp_num_hidden_layers 0 -> 1 (else draft builds zero layers),
    # ignore gains the 17 non-expert MTP entries (experts stay quantized)
    cfg = json.load(open(cfg_path))
    tc = cfg.get("text_config", cfg)
    assert tc.get("mtp_num_hidden_layers", 0) == 0, tc.get("mtp_num_hidden_layers")
    tc["mtp_num_hidden_layers"] = 1
    if cfg is not tc and "mtp_num_hidden_layers" in cfg:
        cfg["mtp_num_hidden_layers"] = 1
    ign = set(cfg["quantization_config"]["ignore"])
    before = len(ign)
    for k in nonexpert_names:
        ign.add(k[: -len(".weight")] if k.endswith(".weight") else k)
    n_mtp = sum(1 for i in ign if i.startswith("mtp"))
    assert n_mtp == len(nonexpert_names), (n_mtp, len(nonexpert_names))
    cfg["quantization_config"]["ignore"] = sorted(ign)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"config: mtp_num_hidden_layers=1, ignore {before} -> {len(ign)} "
          f"(+{len(nonexpert_names)} MTP non-expert)")

    with open(os.path.join(OUTDIR, "SHA256SUMS"), "w") as f:
        for fn in sorted(os.listdir(OUTDIR)):
            if fn.endswith(".safetensors"):
                f.write(f"{sha256(os.path.join(OUTDIR, fn))}  {fn}\n")
    print("SHA256SUMS rebuilt")
    print(f"QUANT-KAT-MTP-OK in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
