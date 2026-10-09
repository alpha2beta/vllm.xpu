#!/usr/bin/env python3
"""Quantize AMAImedia/Qwen3.6-35B-A3B-occamy-1.0-BF16-GGUF-MTP to MXFP4
with MTP draft block, targeting vLLM on Intel Arc 140V (Lunar Lake Xe2-LPG).

Pipeline:
  1. Base text & vision shards (0-13) streamed one at a time:
     - Routed experts (40L x 256E x 3 projs) -> MXFP4 (.weight_packed, .weight_scale)
     - Linear attention (30L x 5 projs) -> MXFP4 (.weight_packed, .weight_scale)
     - Vision, self-attention, norms, shared experts, embed, lm_head -> BF16
     - Delete each source shard immediately after quantization (keeps disk flat)
  2. MTP draft block (MTP/mtp-trained.safetensors):
     - Fused experts (gate_up [256, 1024, 2048] / down [256, 2048, 512]) -> split per expert & quantized to MXFP4
     - 17 non-expert MTP tensors -> BF16
  3. Finalization:
     - Sidecars downloaded from HF repo
     - config.json updated with compressed-tensors mxfp4-pack-quantized + ignore list + text_config.mtp_num_hidden_layers=1
     - model.safetensors.index.json generated
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from huggingface_hub import hf_hub_download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join(REPO_ROOT, "models", "_staging_occamy")
OUTDIR = os.path.join(REPO_ROOT, "models", "Qwen3.6-35B-A3B-occamy-1.0-MXFP4")
SRC_REPO = "AMAImedia/Qwen3.6-35B-A3B-occamy-1.0-BF16-GGUF-MTP"
REF_CFG = os.path.join(REPO_ROOT, "models", "Qwen3.6-35B-A3B-MXFP4", "config.json")

N_BASE_SHARDS = 13  # model-00000-of-00013 .. model-00012-of-00013
SHARD_TARGET_BYTES = 1 << 30  # ~1 GiB output shards

LINEAR_ATTN_QUANT = ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b")
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")

MTP_BASE = "mtp.layers.0.mlp.experts"
N_EXPERTS = 256
PROJS = ("gate_proj", "up_proj", "down_proj")
FUSED_GU = f"{MTP_BASE}.gate_up_proj"
FUSED_DOWN = f"{MTP_BASE}.down_proj"

SIDECARS = [
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "preprocessor_config.json", "processor_config.json", "generation_config.json"
]


def quantize_mxfp4(w2d_xpu):
    """[rows, K] BF16 XPU tensor -> (packed U8 [rows, K/2], scale U8 [rows, K/32])."""
    assert w2d_xpu.dim() == 2 and w2d_xpu.size(1) % 32 == 0, f"bad shape: {w2d_xpu.shape}"
    q, s = torch.ops.vllm.xpu_mxfp4_quantize(w2d_xpu)
    return q.view(torch.uint8).to("cpu"), s.view(torch.uint8).to("cpu")


def get_source_shards():
    """List of all source shard names in order."""
    shards = [f"model-{i:05d}-of-{N_BASE_SHARDS:05d}.safetensors" for i in range(N_BASE_SHARDS)]
    shards.append("model-visual.safetensors")
    return shards


def is_expert_proj(name):
    m = re.search(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", name)
    return m.group(1) if m else None


def download_with_retry(repo_id: str, filename: str, local_dir: str, max_retries: int = 5) -> str:
    """Download a file from Hugging Face with exponential backoff on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return hf_hub_download(repo_id, filename, local_dir=local_dir)
        except Exception as e:
            if attempt == max_retries:
                print(f"  Download failed after {max_retries} attempts: {e}", flush=True)
                raise
            wait = 15 * attempt
            print(f"  Transient download error ({e}); retrying in {wait}s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Quantize Occamy-1.0 to MXFP4 with MTP")
    parser.add_argument("--validate-every", type=int, default=64, help="Validate dequant error every N tensors")
    parser.add_argument("--skip-mtp", action="store_true", help="Skip MTP draft block quantization")
    args = parser.parse_args()

    import vllm_xpu_kernels  # noqa: F401
    import vllm._xpu_ops  # noqa: F401
    from vllm_xpu_kernels.moe_utils import dequant_mxfp4
    assert torch.xpu.is_available(), "XPU required for xpu_mxfp4_quantize"

    os.makedirs(STAGING, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)

    # 1. Fetch source index
    src_index_path = os.path.join(STAGING, "occamy_source_index.json")
    if not os.path.exists(src_index_path):
        print(f"Fetching source index from {SRC_REPO}...", flush=True)
        p = hf_hub_download(SRC_REPO, "model.safetensors.index.json", local_dir=STAGING)
        if p != src_index_path:
            shutil.copy(p, src_index_path)
    src_idx = json.load(open(src_index_path))["weight_map"]
    by_shard = {}
    for k, f in src_idx.items():
        by_shard.setdefault(f, []).append(k)

    # 2. State management
    state_path = os.path.join(OUTDIR, "quantize_state.json")
    state = {
        "done_shards": [],
        "out_shard_no": 1,
        "weight_map": {},
        "total_out_bytes": 0,
        "mtp_done": False,
    }
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"Resuming: {len(state['done_shards'])} source shards done, out_shard_no #{state['out_shard_no']}, mtp_done={state.get('mtp_done')}", flush=True)

    pending, pending_bytes = [], 0
    n_q, n_val, max_rel, sum_rel = 0, 0, 0.0, 0.0

    def save_state():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        n = state["out_shard_no"]
        fn = f"model-{n:05d}-of-XXXXX.safetensors"
        tensors = {k: v.contiguous() for k, v in pending}
        save_file(tensors, os.path.join(OUTDIR, fn))
        for k in tensors:
            state["weight_map"][k] = fn
        state["total_out_bytes"] += pending_bytes
        state["out_shard_no"] += 1
        print(f"  [flush] wrote {fn}: {len(tensors)} tensors, {pending_bytes / (1024**3):.2f} GiB (total: {state['total_out_bytes'] / (1024**3):.2f} GiB)", flush=True)
        pending, pending_bytes = [], 0
        save_state()

    def emit(name, t):
        nonlocal pending_bytes
        pending.append((name, t))
        pending_bytes += t.nelement() * t.element_size()
        if pending_bytes >= SHARD_TARGET_BYTES:
            flush()

    def validate(src2d_cpu, q_cpu, s_cpu, tag):
        nonlocal n_val, max_rel, sum_rel
        n_val += 1
        with torch.no_grad():
            qr = q_cpu.to("xpu").view(torch.float4_e2m1fn_x2)
            sr = s_cpu.to("xpu").view(torch.float8_e8m0fnu)
            dq = dequant_mxfp4(qr, sr).float().cpu()
        ref = src2d_cpu.float()
        rel = (dq - ref).abs().mean().item() / ref.abs().mean().item()
        max_rel = max(max_rel, rel)
        sum_rel += rel
        if n_val <= 3 or n_val % 200 == 0:
            print(f"  [val] {tag}: mean-rel-err={rel:.5f} (max={max_rel:.5f})", flush=True)

    t0 = time.time()
    source_shards = get_source_shards()

    # 3. Process base shards
    for si, src in enumerate(source_shards):
        if src in state["done_shards"]:
            print(f"[{si+1}/{len(source_shards)}] {src}: SKIP (already done)", flush=True)
            continue

        shard_path = os.path.join(STAGING, src)
        if not os.path.exists(shard_path):
            print(f"[{si+1}/{len(source_shards)}] Downloading {src} from {SRC_REPO}...", flush=True)
            dl_path = download_with_retry(SRC_REPO, src, local_dir=STAGING)
            if dl_path != shard_path:
                shutil.copy(dl_path, shard_path)

        names = by_shard.get(src, [])
        print(f"[{si+1}/{len(source_shards)}] Quantizing {src} ({len(names)} tensors)...", flush=True)
        with safe_open(shard_path, framework="pt") as h:
            for name in names:
                t = h.get_slice(name)[:].contiguous()
                is_text = ".language_model.layers." in name
                proj = is_expert_proj(name) if is_text else None

                if proj is not None:
                    # Routed expert projection
                    assert t.dim() == 2 and t.size(1) % 32 == 0, f"{name}: {t.shape}"
                    w = t.to("xpu", dtype=torch.bfloat16)
                    q, s = quantize_mxfp4(w)
                    del w
                    torch.xpu.synchronize()
                    n_q += 1
                    if n_q % args.validate_every == 0:
                        validate(t, q, s, name)
                    stem = name[: -len(".weight")]
                    emit(stem + ".weight_packed", q)
                    emit(stem + ".weight_scale", s)
                    del q, s
                elif is_text and ".linear_attn." in name and any(name.endswith(f".{p}.weight") for p in LINEAR_ATTN_QUANT):
                    # Linear attention projection
                    assert t.dim() == 2 and t.size(1) % 32 == 0, f"{name}: {t.shape}"
                    w = t.to("xpu", dtype=torch.bfloat16)
                    q, s = quantize_mxfp4(w)
                    del w
                    torch.xpu.synchronize()
                    n_q += 1
                    if n_q % args.validate_every == 0:
                        validate(t, q, s, name)
                    stem = name[: -len(".weight")]
                    emit(stem + ".weight_packed", q)
                    emit(stem + ".weight_scale", s)
                    del q, s
                else:
                    # Plain BF16 tensor (embed, norms, shared experts, self_attn, vision, etc.)
                    emit(name, t)

        state["done_shards"].append(src)
        save_state()
        print(f"  Completed {src} ({time.time() - t0:.0f}s elapsed)", flush=True)

        # Flat disk discipline: remove downloaded source shard immediately
        if os.path.exists(shard_path):
            os.remove(shard_path)
            print(f"  Cleaned up source shard {src}", flush=True)

    # 4. Process MTP draft block
    if not args.skip_mtp and not state.get("mtp_done"):
        print("\n--- Processing MTP Draft Block ---", flush=True)
        mtp_file = "MTP/mtp-trained.safetensors"
        mtp_local = os.path.join(STAGING, "mtp-trained.safetensors")
        if not os.path.exists(mtp_local):
            print(f"Downloading {mtp_file} from {SRC_REPO}...", flush=True)
            p = download_with_retry(SRC_REPO, mtp_file, local_dir=STAGING)
            if p != mtp_local:
                shutil.copy(p, mtp_local)

        mtp_blobs = load_file(mtp_local)
        gu = mtp_blobs.pop(FUSED_GU)
        dn = mtp_blobs.pop(FUSED_DOWN)
        assert tuple(gu.shape) == (N_EXPERTS, 1024, 2048), f"bad MTP gate_up: {gu.shape}"
        assert tuple(dn.shape) == (N_EXPERTS, 2048, 512), f"bad MTP down: {dn.shape}"
        rest = dict(mtp_blobs)
        del mtp_blobs
        print(f"MTP fused experts: {gu.shape}, {dn.shape}; non-expert tensors: {len(rest)}", flush=True)

        # Emit non-expert MTP tensors as BF16
        for k, v in sorted(rest.items()):
            emit(k, v.contiguous())

        # Split and quantize 256 MTP routed experts
        cpu_gu, cpu_dn = gu, dn
        del gu, dn
        for e in range(N_EXPERTS):
            for proj, src2d in (("gate_proj", cpu_gu[e][:512]),
                                ("up_proj", cpu_gu[e][512:]),
                                ("down_proj", cpu_dn[e])):
                name = f"{MTP_BASE}.{e}.{proj}.weight"
                t = src2d.clone().to("xpu", dtype=torch.bfloat16)
                q, s = quantize_mxfp4(t)
                torch.xpu.synchronize()
                if (e * 3 + PROJS.index(proj) + 1) % args.validate_every == 0:
                    validate(src2d, q, s, name)
                stem = name[: -len(".weight")]
                emit(stem + ".weight_packed", q)
                emit(stem + ".weight_scale", s)
                del q, s, t, src2d

            if (e + 1) % 64 == 0:
                print(f"  ... {e + 1}/{N_EXPERTS} MTP experts quantized", flush=True)

        del cpu_gu, cpu_dn
        state["mtp_done"] = True
        save_state()
        if os.path.exists(mtp_local):
            os.remove(mtp_local)
            print(f"  Cleaned up MTP source shard {mtp_local}", flush=True)

    # 5. Flush all remaining tensors
    flush()

    # 6. Finalize output shards: normalize naming to model-NNNNN-of-TOTAL.safetensors
    n_out_shards = state["out_shard_no"] - 1
    print(f"\nFinalizing {n_out_shards} output shards...", flush=True)
    final_map = {}
    for k, fn in state["weight_map"].items():
        m = re.fullmatch(r"model-(\d+)-of-(?:XXXXX|\d+)\.safetensors", fn)
        final_map[k] = f"model-{int(m.group(1)):05d}-of-{n_out_shards:05d}.safetensors"

    for fn in os.listdir(OUTDIR):
        m = re.fullmatch(r"model-(\d+)-of-(?:XXXXX|\d+)\.safetensors", fn)
        if m:
            target_name = f"model-{int(m.group(1)):05d}-of-{n_out_shards:05d}.safetensors"
            if fn != target_name:
                os.rename(os.path.join(OUTDIR, fn), os.path.join(OUTDIR, target_name))

    # Write model.safetensors.index.json
    index_out = {
        "metadata": {"total_size": state["total_out_bytes"]},
        "weight_map": final_map
    }
    with open(os.path.join(OUTDIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)
    print(f"Wrote model.safetensors.index.json with {len(final_map)} tensors ({state['total_out_bytes'] / (1024**3):.2f} GiB)", flush=True)

    # 7. Finalize config.json with quantization_config & MTP settings
    print("Generating config.json and downloading sidecars...", flush=True)
    cfg_path = os.path.join(OUTDIR, "config.json")
    hf_hub_download(SRC_REPO, "config.json", local_dir=OUTDIR)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Build ignore list: all plain BF16 modules (full name minus trailing .weight)
    plain_modules = set()
    for name in final_map:
        if not (name.endswith(".weight_packed") or name.endswith(".weight_scale")):
            mod = name[: -len(".weight")] if name.endswith(".weight") else name
            plain_modules.add(mod)

    # Reference quantization_config format
    ref_q = json.load(open(REF_CFG))["quantization_config"]
    cfg["quantization_config"] = {
        "quant_method": ref_q["quant_method"],
        "format": ref_q["format"],
        "config_groups": ref_q["config_groups"],
        "ignore": sorted(plain_modules)
    }

    # Enable MTP hidden layer in text_config
    if "text_config" in cfg:
        cfg["text_config"]["mtp_num_hidden_layers"] = 1
    cfg["mtp_num_hidden_layers"] = 1

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"config.json updated: {len(plain_modules)} ignored BF16 modules, MTP enabled", flush=True)

    # Download sidecars
    for sidecar in SIDECARS:
        target = os.path.join(OUTDIR, sidecar)
        if not os.path.exists(target):
            try:
                hf_hub_download(SRC_REPO, sidecar, local_dir=OUTDIR)
                print(f"  Downloaded {sidecar}", flush=True)
            except Exception as e:
                print(f"  Skip optional sidecar {sidecar}: {e}", flush=True)

    # Write README provenance
    with open(os.path.join(OUTDIR, "README.md"), "w") as f:
        f.write(
            f"# Qwen3.6-35B-A3B-occamy-1.0-MXFP4\n\n"
            f"MXFP4 (`compressed-tensors` `mxfp4-pack-quantized`, E2M1 group-32 symmetric, E8M0 scales) "
            f"quantization of routed MoE experts and linear-attention projections for Intel Arc Xe2 (Lunar Lake).\n\n"
            f"- **Base Model:** `{SRC_REPO}`\n"
            f"- **Format:** CompressedTensorsW4A4Mxfp4 + XPUExpertsMxFp4\n"
            f"- **Speculative Decoding:** Integrated MXFP4 MTP draft block (MTP $K=2$ supported)\n"
            f"- **Quantizer:** `torch.ops.vllm.xpu_mxfp4_quantize`\n"
            f"- **Total Tensors:** {len(final_map)}\n"
            f"- **Total Payload:** {state['total_out_bytes'] / (1024**3):.2f} GiB across {n_out_shards} shards\n"
        )

    print(f"\n==========================================")
    print(f"ALL DONE in {time.time() - t0:.1f}s!")
    print(f"Total tensors: {len(final_map)}, Shards: {n_out_shards}")
    print(f"Quantized projs: {n_q}, Spot-checks: {n_val}, Max rel-err: {max_rel:.5f}")
    print(f"Output directory: {OUTDIR}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
