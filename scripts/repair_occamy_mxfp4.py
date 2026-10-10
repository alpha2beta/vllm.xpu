#!/usr/bin/env python3
"""Repair Occamy MXFP4 checkpoint by recovering missing tensors from shards 0 and 4:
1. Identifies missing/incomplete tensors from model.safetensors.index.json.
2. Downloads source shards 0 and 4 with retry and deletes each immediately after use.
3. Quantizes missing expert and linear attention projections to MXFP4.
4. Flushes the recovered tensors into model-00021-of-00021.safetensors.
5. Renames existing shards from of-00020 to of-00021.
6. Updates model.safetensors.index.json and config.json.
7. Executes audit to verify completeness.

Usage: ./.venv/bin/python scripts/repair_occamy_mxfp4.py
"""

import json
import os
import re
import shutil
import sys
import time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(REPO_ROOT, "models", "_staging_occamy")
OUTDIR = os.path.join(REPO_ROOT, "models", "Qwen3.6-35B-A3B-occamy-1.0-MXFP4")
SRC_REPO = "AMAImedia/Qwen3.6-35B-A3B-occamy-1.0-BF16-GGUF-MTP"

LINEAR_ATTN_QUANT = ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b")


def is_expert_proj(name):
    m = re.search(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", name)
    return m.group(1) if m else None


def download_with_retry(repo_id: str, filename: str, local_dir: str, max_retries: int = 5) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            return hf_hub_download(repo_id, filename, local_dir=local_dir)
        except Exception as e:
            if "404" in str(e) or "Entry Not Found" in str(e) or attempt == max_retries:
                raise
            wait = 15 * attempt
            print(f"  Transient download error ({e}); retrying in {wait}s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(wait)


def main():
    import vllm_xpu_kernels  # noqa: F401
    import vllm._xpu_ops  # noqa: F401

    idx_path = os.path.join(OUTDIR, "model.safetensors.index.json")
    assert os.path.exists(idx_path), f"{idx_path} missing"
    idx = json.load(open(idx_path))
    wmap = idx["weight_map"]

    src_index_path = os.path.join(STAGING, "occamy_source_index.json")
    assert os.path.exists(src_index_path), f"{src_index_path} missing"
    src_idx = json.load(open(src_index_path))["weight_map"]

    missing_by_shard = {}
    for k, src in src_idx.items():
        is_text = ".language_model.layers." in k
        proj = is_expert_proj(k) if is_text else None
        is_lin = is_text and any(f".{p}.weight" in k for p in LINEAR_ATTN_QUANT)
        if proj or is_lin:
            stem = k[:-len(".weight")] if k.endswith(".weight") else k
            p = stem + ".weight_packed"
            s = stem + ".weight_scale"
            if p not in wmap or s not in wmap:
                missing_by_shard.setdefault(src, []).append((k, "quant"))
        else:
            if k not in wmap:
                missing_by_shard.setdefault(src, []).append((k, "plain"))

    total_missing = sum(len(v) for v in missing_by_shard.values())
    print(f"Found {total_missing} missing tensors across {len(missing_by_shard)} shards:")
    for src, items in sorted(missing_by_shard.items()):
        print(f"  {src}: {len(items)} tensors", flush=True)

    if total_missing == 0:
        print("Nothing to repair!")
        return 0

    recovered_tensors = {}
    recovered_bytes = 0

    for src, items in sorted(missing_by_shard.items()):
        shard_path = os.path.join(STAGING, src)
        if not os.path.exists(shard_path):
            print(f"\nDownloading {src} from {SRC_REPO}...", flush=True)
            dl_path = download_with_retry(SRC_REPO, src, local_dir=STAGING)
            if dl_path != shard_path:
                shutil.copy(dl_path, shard_path)

        print(f"Processing {len(items)} missing tensors from {src}...", flush=True)
        with safe_open(shard_path, framework="pt") as h:
            for name, kind in items:
                t = h.get_slice(name)[:].contiguous()
                if kind == "quant":
                    stem = name[:-len(".weight")] if name.endswith(".weight") else name
                    assert t.dim() == 2 and t.size(1) % 32 == 0, f"{name}: bad shape {t.shape}"
                    w = t.to("xpu", dtype=torch.bfloat16)
                    q, s = torch.ops.vllm.xpu_mxfp4_quantize(w)
                    del w
                    torch.xpu.synchronize()
                    qp = q.view(torch.uint8).cpu().contiguous()
                    sp = s.view(torch.uint8).cpu().contiguous()
                    del q, s
                    recovered_tensors[stem + ".weight_packed"] = qp
                    recovered_tensors[stem + ".weight_scale"] = sp
                    recovered_bytes += qp.nelement() + sp.nelement()
                else:
                    recovered_tensors[name] = t
                    recovered_bytes += t.nelement() * t.element_size()

        # Clean up source shard immediately to maintain flat disk
        if os.path.exists(shard_path):
            os.remove(shard_path)
            print(f"Cleaned up {src}", flush=True)

    print(f"\nTotal recovered tensors: {len(recovered_tensors)} ({recovered_bytes / (1024**3):.2f} GiB)")

    # Rename existing 20 shards from of-00020 to of-00021
    print("Renaming existing shards of-00020 -> of-00021...", flush=True)
    new_wmap = {}
    for k, fn in wmap.items():
        new_fn = fn.replace("-of-00020.safetensors", "-of-00021.safetensors")
        new_wmap[k] = new_fn

    for i in range(1, 21):
        old_fn = os.path.join(OUTDIR, f"model-{i:05d}-of-00020.safetensors")
        new_fn = os.path.join(OUTDIR, f"model-{i:05d}-of-00021.safetensors")
        if os.path.exists(old_fn):
            os.rename(old_fn, new_fn)

    # Save recovered tensors to shard 21
    shard_21_name = "model-00021-of-00021.safetensors"
    shard_21_path = os.path.join(OUTDIR, shard_21_name)
    print(f"Writing {shard_21_name} ({len(recovered_tensors)} tensors)...", flush=True)
    save_file(recovered_tensors, shard_21_path)

    # Update weight_map with recovered tensors
    for k in recovered_tensors:
        new_wmap[k] = shard_21_name

    new_total_size = idx["metadata"].get("total_size", 0) + recovered_bytes
    new_index = {
        "metadata": {"total_size": new_total_size},
        "weight_map": new_wmap
    }
    with open(idx_path, "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"Updated {idx_path} with {len(new_wmap)} total mapped tensors ({new_total_size / (1024**3):.2f} GiB)", flush=True)

    # Update config.json ignore list
    cfg_path = os.path.join(OUTDIR, "config.json")
    cfg = json.load(open(cfg_path))
    plain_modules = set()
    for name in new_wmap:
        if not (name.endswith(".weight_packed") or name.endswith(".weight_scale")):
            mod = name[:-len(".weight")] if name.endswith(".weight") else name
            plain_modules.add(mod)
    cfg["quantization_config"]["ignore"] = sorted(plain_modules)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Updated config.json ignore list ({len(plain_modules)} modules)", flush=True)

    print("\nRepair completed! Running verification audit...")
    audit_script = os.path.join(REPO_ROOT, "scripts", "audit_occamy_mxfp4.py")
    ret = os.system(f"{sys.executable} {audit_script}")
    return ret


if __name__ == "__main__":
    sys.exit(main())
