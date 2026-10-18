#!/usr/bin/env python3
"""quantize_kat_gptq.py -- Production W4A16 GPTQ Quantizer for Kwaipilot/KAT-Coder-V2.5-Dev
on Intel Arc 140V (Lunar Lake Xe2-LPG).

Exports compressed-tensors packed int4 safetensors directly compatible with
vLLM XPU's CompressedTensorsWNA16MoEMethod (XPU WNA16 MoE and Linear kernels).

Recipe:
  - 256 routed experts per layer (40 layers): symmetric INT4, group size 128,
    column nibble-packed int32, bfloat16 scales, int64 shapes.
  - Linear attention (GDN), self-attention, and shared experts can be quantized to W4A16
    or kept in BF16; default is reference parity with Tiel-Coder-35B-A3B-W4A16-GPTQ.
  - Embeddings, norms, router gate, conv1d, output head: remain BF16.
  - MTP draft block (19 tensors): preserved in BF16 for speculative decoding (k=2).
  - Calibration: 64 samples x 1024 tokens = 65,536 tokens from pile-val (or custom).
  - Resumable: saves hidden-after.pt checkpoint state layer-by-layer.
  - Disk-constrained: fetches BF16 shards on demand and purges completed shards if requested.

Usage:
  ./.venv/bin/python scripts/quantize_kat_gptq.py \
      --out models/KAT-Coder-V2.5-Dev-W4A16-GPTQ \
      --calib-samples 64 --calib-length 1024
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Deterministic descending sort for XPU top-k routing selection
_orig_topk = torch.topk


def _safe_topk(x, k, dim=-1, largest=True, sorted=True, **kw):
    if x.device.type != 'xpu':
        return _orig_topk(x, k, dim=dim, largest=largest, sorted=sorted, **kw)
    v, i = torch.sort(x, dim=dim, descending=largest)
    return torch.return_types.topk((v.narrow(dim, 0, k), i.narrow(dim, 0, k)))


torch.topk = _safe_topk

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_REPO = "Kwaipilot/KAT-Coder-V2.5-Dev"
MTP_STAGING = REPO_ROOT / "models" / "_staging_kat_mtp" / "model-00014-of-mtp.safetensors"
KAT_MXFP4_DIR = REPO_ROOT / "models" / "KAT-Coder-V2.5-Dev-MXFP4"
STAGING_DIR = REPO_ROOT / "models" / "_staging_kat"

PREFIX = "model.language_model."
LAYER_PREFIX = PREFIX + "layers.{}."
GROUP = 128
BLOCK = 128
DAMP = 0.01
CLIP_FACTORS = (1.0, 0.96, 0.92, 0.88, 0.84)
DEV = "xpu"


def log(*args) -> None:
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + " ".join(str(x) for x in args)
    print(line, flush=True)


def get_mem_stats() -> str:
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
        host_avail = meminfo.get("MemAvailable", 0) / 1024
        swap_free = meminfo.get("SwapFree", 0) / 1024
        xpu_alloc = torch.xpu.memory_allocated() / (1024**2) if torch.xpu.is_available() else 0
        xpu_res = torch.xpu.memory_reserved() / (1024**2) if torch.xpu.is_available() else 0
        return (f"hostAvail={host_avail:.0f}MiB swapFree={swap_free:.0f}MiB "
                f"xpuAlloc={xpu_alloc:.0f}MiB xpuRes={xpu_res:.0f}MiB")
    except Exception:
        return "memStat=?"


# -----------------------------------------------------------------------------
# GPTQ Algorithm (Batched or Single Matrix)
# -----------------------------------------------------------------------------
def gptq_quantize(W: torch.Tensor, Hs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4 GPTQ with group size 128.

    Args:
        W: [B, rows, cols] fp32 weight matrices (B can be number of experts or 1)
        Hs: [B, cols, cols] fp32 Hessian matrices (covariance of inputs)

    Returns:
        Q: [B, rows, cols] uint8 quantized weights (0..15, zero-point 8)
        S: [B, rows, cols // GROUP] bfloat16 scales
    """
    W = W.clone()
    Hs = Hs.clone()
    B, rows, cols = W.shape
    ar = torch.arange(cols, device=W.device)

    # Enforce exact symmetry across all blocks
    Hs = 0.5 * (Hs + Hs.transpose(1, 2))

    # Regularize zero or near-zero diagonals
    diag = Hs[:, ar, ar]
    dead = diag <= 1e-8
    Hs[:, ar, ar] = torch.where(dead, torch.ones_like(diag), diag)

    # Any block with dead diagonal elements must fallback to RTN using true weights W
    dead_blocks = set((dead.any(dim=-1)).nonzero().flatten().tolist())

    # Proportional diagonal damping
    mean_diag = Hs[:, ar, ar].mean(dim=-1, keepdim=True).clamp(min=1e-5)
    Hs[:, ar, ar] += DAMP * mean_diag

    # Cholesky decomposition with progressive damping boost
    L, info = torch.linalg.cholesky_ex(Hs)
    for boost in (0.1, 1.0, 10.0, 100.0, 1000.0):
        bad = (info != 0).nonzero().flatten().tolist()
        if not bad:
            break
        for b in bad:
            Hs[b, ar, ar] += boost * mean_diag[b] + 0.1
        L, info = torch.linalg.cholesky_ex(Hs)

    bad_blocks = dead_blocks.union((info != 0).nonzero().flatten().tolist())
    good_blocks = [i for i in range(B) if i not in bad_blocks]

    Q = torch.empty(B, rows, cols, dtype=torch.uint8, device=W.device)
    S = torch.empty(B, rows, cols // GROUP, dtype=torch.bfloat16, device=W.device)

    # Fallback to optimal RTN with clipping factor for any residual singular/unobserved blocks
    for b in bad_blocks:
        for i1 in range(0, cols, GROUP):
            i2 = min(i1 + GROUP, cols)
            w_grp = W[b, :, i1:i2]
            amax = w_grp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
            best_s, best_err = None, None
            for c in CLIP_FACTORS:
                s_cand = (amax * c * 2.0 / 15.0).to(torch.bfloat16).float()
                q_cand = torch.clamp(torch.round(w_grp / s_cand) + 8.0, 0.0, 15.0)
                err_cand = ((q_cand - 8.0) * s_cand - w_grp).pow(2).sum(dim=-1)
                if best_s is None:
                    best_s, best_err = s_cand, err_cand
                else:
                    better = err_cand < best_err
                    best_s = torch.where(better.unsqueeze(-1), s_cand, best_s)
                    best_err = torch.where(better, err_cand, best_err)
            S[b, :, i1 // GROUP] = best_s.squeeze(-1).to(torch.bfloat16)
            Q[b, :, i1:i2] = torch.clamp(torch.round(w_grp / best_s) + 8.0, 0.0, 15.0).to(torch.uint8)

    # Standard GPTQ for positive definite blocks
    if good_blocks:
        W_g = W[good_blocks]
        L_g = L[good_blocks]
        try:
            Hinv_mat = torch.cholesky_inverse(L_g)
            Hinv_mat = 0.5 * (Hinv_mat + Hinv_mat.transpose(1, 2))
            Hinv_g = torch.linalg.cholesky(Hinv_mat, upper=True)
        except Exception:
            for b in good_blocks:
                for i1 in range(0, cols, GROUP):
                    i2 = min(i1 + GROUP, cols)
                    w_grp = W[b, :, i1:i2]
                    amax = w_grp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
                    best_s, best_err = None, None
                    for c in CLIP_FACTORS:
                        s_cand = (amax * c * 2.0 / 15.0).to(torch.bfloat16).float()
                        q_cand = torch.clamp(torch.round(w_grp / s_cand) + 8.0, 0.0, 15.0)
                        err_cand = ((q_cand - 8.0) * s_cand - w_grp).pow(2).sum(dim=-1)
                        if best_s is None:
                            best_s, best_err = s_cand, err_cand
                        else:
                            better = err_cand < best_err
                            best_s = torch.where(better.unsqueeze(-1), s_cand, best_s)
                            best_err = torch.where(better, err_cand, best_err)
                    S[b, :, i1 // GROUP] = best_s.squeeze(-1).to(torch.bfloat16)
                    Q[b, :, i1:i2] = torch.clamp(torch.round(w_grp / best_s) + 8.0, 0.0, 15.0).to(torch.uint8)
            good_blocks = []

        for i1 in range(0, cols, BLOCK):
            i2 = min(i1 + BLOCK, cols)
            W1 = W_g[:, :, i1:i2].clone()
            Err = torch.zeros_like(W1)
            Hinv1 = Hinv_g[:, i1:i2, i1:i2]

            amax = W1.abs().amax(dim=-1).clamp(min=1e-10)
            best_scale, best_err = None, None
            for c in CLIP_FACTORS:
                s_cand = (amax * c * 2.0 / 15.0).to(torch.bfloat16).float()
                q_cand = torch.clamp(torch.round(W1 / s_cand.unsqueeze(-1)) + 8.0, 0.0, 15.0)
                err_cand = ((q_cand - 8.0) * s_cand.unsqueeze(-1) - W1).pow(2).sum(dim=-1)
                if best_scale is None:
                    best_scale, best_err = s_cand, err_cand
                else:
                    better = err_cand < best_err
                    best_scale = torch.where(better, s_cand, best_scale)
                    best_err = torch.where(better, err_cand, best_err)

            s = best_scale
            S[good_blocks, :, i1 // GROUP] = s.to(torch.bfloat16)

            for col_idx in range(i2 - i1):
                w_col = W1[:, :, col_idx]
                d = Hinv1[:, col_idx, col_idx].unsqueeze(-1)
                q_col = torch.clamp(torch.round(w_col / s) + 8.0, 0.0, 15.0)
                Q[good_blocks, :, i1 + col_idx] = q_col.to(torch.uint8)

                err_col = (w_col - (q_col - 8.0) * s) / d
                W1[:, :, col_idx:] -= err_col.unsqueeze(-1) * Hinv1[:, col_idx, col_idx:].unsqueeze(1)
                Err[:, :, col_idx] = err_col

            if i2 < cols:
                W_g[:, :, i2:] -= torch.bmm(Err, Hinv_g[:, i1:i2, i2:])

    return Q, S


def dequantize(Q: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """[B, rows, cols] uint8 + [B, rows, cols//G] scale -> [B, rows, cols] bfloat16."""
    return ((Q.float() - 8.0) * S.float().repeat_interleave(GROUP, dim=-1)).to(torch.bfloat16)


def pack_ct(q: torch.Tensor) -> torch.Tensor:
    """[rows, cols] uint8 0..15 -> [rows, cols//8] int32 packed nibbles."""
    assert q.shape[1] % 8 == 0, f"Columns {q.shape[1]} must be multiple of 8 for int32 packing"
    nib = (q.to(torch.int32) & 0xF).reshape(q.shape[0], -1, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=q.device).view(1, 1, 8)
    return (nib << shifts).sum(dim=2, dtype=torch.int32)


# -----------------------------------------------------------------------------
# Calibration Data Loader
# -----------------------------------------------------------------------------
def build_calibration_dataset(dataset_name: str, split: str, samples: int, seq_len: int,
                             tokenizer, seed: int = 42) -> torch.Tensor:
    """Fetch text from HF dataset, tokenize, and chunk into [samples, seq_len] tensor."""
    log(f"Building calibration stream from {dataset_name} ({split}) ...")
    from datasets import load_dataset

    name, _, cfg = dataset_name.partition(":")
    load_kwargs = dict(split=split, streaming=True)
    if cfg:
        ds = load_dataset(name, cfg, **load_kwargs)
    else:
        ds = load_dataset(name, **load_kwargs)

    try:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    except Exception:
        pass

    stream: List[int] = []
    scanned = 0
    eos_id = tokenizer.eos_token_id or 248046

    for row in ds:
        scanned += 1
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        stream.extend(ids + [eos_id])
        if len(stream) >= samples * seq_len * 2:
            break
        if scanned > 100_000:
            break

    windows = [stream[i:i + seq_len] for i in range(0, len(stream) - seq_len + 1, seq_len)]
    random.seed(seed)
    random.shuffle(windows)

    if len(windows) < samples:
        log(f"WARNING: Only {len(windows)} full windows available; using {len(windows)}.")
        samples = len(windows)

    calib_tensor = torch.tensor(windows[:samples], dtype=torch.long)
    log(f"Calibration built: {calib_tensor.shape[0]} windows x {calib_tensor.shape[1]} tokens ({scanned} rows scanned)")
    return calib_tensor


# -----------------------------------------------------------------------------
# Model Weight Shard Manager (Disk-Constrained Staging)
# -----------------------------------------------------------------------------
class WeightShardManager:
    def __init__(self, src_repo: str, staging_dir: Path, index_map: Dict[str, str],
                 clean_shards: bool = False):
        self.src_repo = src_repo
        self.staging_dir = staging_dir
        self.index_map = index_map
        self.clean_shards = clean_shards
        self.open_shards: Dict[str, any] = {}
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def ensure_shard(self, shard_file: str) -> Path:
        local_path = self.staging_dir / shard_file
        if local_path.is_file():
            return local_path

        # Check if file exists in MXFP4 dir (e.g. for shard 12 non-expert tensors)
        alt_path = KAT_MXFP4_DIR / shard_file
        if alt_path.is_file():
            return alt_path

        log(f"Downloading required shard {shard_file} from {self.src_repo} ...")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(self.src_repo, shard_file, local_dir=str(self.staging_dir))
        return Path(downloaded)

    def get_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        shard_file = self.index_map.get(name)
        if not shard_file:
            raise KeyError(f"Tensor {name} not found in model index.")

        shard_path = self.ensure_shard(shard_file)
        if shard_file not in self.open_shards:
            self.open_shards[shard_file] = safe_open(str(shard_path), framework="pt")

        f = self.open_shards[shard_file]
        return f.get_tensor(name).to(device)

    def close_all(self) -> None:
        self.open_shards.clear()

    def purge_shard(self, shard_file: str) -> None:
        if not self.clean_shards:
            return
        if shard_file in self.open_shards:
            del self.open_shards[shard_file]
        local_path = self.staging_dir / shard_file
        if local_path.is_file():
            local_path.unlink()
            log(f"Purged staging shard {shard_file} to free disk space.")


# -----------------------------------------------------------------------------
# Main Quantization Pipeline
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="W4A16 GPTQ Quantization for KAT-Coder-V2.5-Dev on Arc 140V")
    ap.add_argument("--out", required=True, help="Output directory for quantized model")
    ap.add_argument("--device", default="xpu", help="Accelerator device (xpu or cpu)")
    ap.add_argument("--calib-dataset", default="mit-han-lab/pile-val-backup", help="HF calibration dataset")
    ap.add_argument("--calib-split", default="validation")
    ap.add_argument("--calib-samples", type=int, default=64, help="Calibration sample chunks (default: 64)")
    ap.add_argument("--calib-length", type=int, default=1024, help="Tokens per calibration chunk (default: 1024)")
    ap.add_argument("--batch-size", type=int, default=4, help="Forward batch size (default: 4)")
    ap.add_argument("--expert-chunk", type=int, default=32, help="Expert GPTQ batch size (default: 32)")
    ap.add_argument("--layers", default=None, help="Layers to process: 'start-end' or 'N' (default: all 40)")
    ap.add_argument("--clean-shards", action="store_true", help="Purge source shards after use to conserve disk")
    ap.add_argument("--quantize-attn", action="store_true", default=True,
                    help="Quantize linear attention and self attention to W4A16 (default: True)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Target Output Directory: {out_dir}")
    log(f"Device: {args.device} | Memory Status: {get_mem_stats()}")

    # 1. Load Tokenizer & AutoConfig
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

    tok_src = str(KAT_MXFP4_DIR) if (KAT_MXFP4_DIR / "tokenizer.json").is_file() else SRC_REPO
    log(f"Loading tokenizer from {tok_src} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=False)

    cfg_src = str(KAT_MXFP4_DIR / "config.json") if (KAT_MXFP4_DIR / "config.json").is_file() else SRC_REPO
    log(f"Loading AutoConfig from {cfg_src} ...")
    cfg = AutoConfig.from_pretrained(cfg_src, trust_remote_code=False)
    tc = getattr(cfg, "text_config", cfg)
    tc._attn_implementation = "sdpa"

    n_layers = tc.num_hidden_layers
    H = tc.hidden_size
    E = tc.num_experts
    I = tc.moe_intermediate_size
    top_k = tc.num_experts_per_tok
    log(f"Architecture: layers={n_layers}, hidden={H}, experts={E}, top_k={top_k}, moe_inter={I}")

    # 2. Source Shard Index
    index_path = STAGING_DIR / "kat_source_index.json"
    if not index_path.is_file():
        from huggingface_hub import hf_hub_download
        log("Fetching source model.safetensors.index.json ...")
        p = hf_hub_download(SRC_REPO, "model.safetensors.index.json", local_dir=str(STAGING_DIR))
        shutil.copy2(p, str(index_path))

    index_map = json.load(open(index_path))["weight_map"]
    shard_mgr = WeightShardManager(SRC_REPO, STAGING_DIR, index_map, clean_shards=args.clean_shards)

    # 3. Build Calibration Tensor
    calib_ids = build_calibration_dataset(
        args.calib_dataset, args.calib_split, args.calib_samples, args.calib_length, tokenizer
    )
    N_SAMPLES, SEQ = calib_ids.shape
    BATCH = args.batch_size
    N_SAMPLES -= (N_SAMPLES % BATCH)
    calib_ids = calib_ids[:N_SAMPLES]

    # Compute initial hidden states from embed_tokens
    log("Computing initial hidden states from embed_tokens ...")
    embed_weight = shard_mgr.get_tensor(PREFIX + "embed_tokens.weight", device=DEV)
    hs = torch.empty((N_SAMPLES, SEQ, H), dtype=torch.bfloat16, device=DEV)
    for b in range(0, N_SAMPLES, 16):
        b_end = min(b + 16, N_SAMPLES)
        hs[b:b_end] = F.embedding(calib_ids[b:b_end].to(DEV), embed_weight)
    del embed_weight
    torch.xpu.empty_cache()

    nxt_hs = torch.empty_like(hs)

    # Rotary embeddings
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(config=tc, device=DEV)
    pos = torch.arange(SEQ, device=DEV).view(1, 1, -1).expand(4, BATCH, -1)
    text_pos = pos[0]
    with torch.no_grad():
        pos_emb = rotary(hs[:BATCH], pos[1:])

    # Base tensors (embed_tokens, lm_head, norm) - save early so shard 0 can be safely purged
    base_shard_path = out_dir / "model-base.safetensors"
    if not base_shard_path.is_file():
        log("Assembling early base model tensors (embed_tokens, lm_head, norms) ...")
        base_tensors: Dict[str, torch.Tensor] = {}
        for k in index_map:
            if not k.startswith(PREFIX + "layers.") and not k.startswith("mtp."):
                base_tensors[k] = shard_mgr.get_tensor(k, device="cpu")
        save_file(base_tensors, str(base_shard_path), metadata={"format": "pt"})
        log(f"Saved {len(base_tensors)} base tensors to {base_shard_path.name}")
        del base_tensors

    # Map each shard to the last layer that accesses it
    shard_last_layer: Dict[str, int] = {}
    for k, f in index_map.items():
        if "layers." in k:
            l_num = int(k.split("layers.")[1].split(".")[0])
            shard_last_layer[f] = max(shard_last_layer.get(f, -1), l_num)

    # 4. Resume check
    state_file = out_dir / "hidden-after.pt"
    start_layer = 0
    if state_file.is_file():
        saved_state = torch.load(str(state_file), map_location="cpu")
        start_layer = saved_state.get("next_layer", 0)
        hs.copy_(saved_state["hs"].to(DEV))
        del saved_state
        log(f"Resuming quantization at layer {start_layer} from checkpoint.")

    end_layer = n_layers
    if args.layers:
        if "-" in args.layers:
            s_str, e_str = args.layers.split("-")
            start_layer, end_layer = int(s_str), int(e_str)
        else:
            end_layer = int(args.layers)

    log(f"Processing decoder layers: [{start_layer} -> {end_layer})")

    # 5. Layer-by-Layer Loop
    for li in range(start_layer, end_layer):
        t0 = time.time()
        pfx = LAYER_PREFIX.format(li)
        layer_keys = [k for k in index_map if k.startswith(pfx)]

        log(f"--- Layer {li:02d}/{n_layers} [{get_mem_stats()}] ---")

        # Instantiate layer structure on meta device
        with torch.device("meta"):
            layer = M.Qwen3_5MoeDecoderLayer(tc, li)

        # In KAT, experts are stored as 256 individual 2D weights:
        #   experts.{e}.gate_proj.weight [512, 2048]
        #   experts.{e}.up_proj.weight   [512, 2048]
        #   experts.{e}.down_proj.weight [2048, 512]
        # Assemble them into 3D tensors: gu [E, 2*I, H] and dn [E, H, I]
        gu = torch.empty((E, 2 * I, H), dtype=torch.bfloat16, device=DEV)
        dn = torch.empty((E, H, I), dtype=torch.bfloat16, device=DEV)

        log(f"Loading expert weights for layer {li} ...")
        for e in range(E):
            g_k = f"{pfx}mlp.experts.{e}.gate_proj.weight"
            u_k = f"{pfx}mlp.experts.{e}.up_proj.weight"
            d_k = f"{pfx}mlp.experts.{e}.down_proj.weight"
            g_w = shard_mgr.get_tensor(g_k, device=DEV)
            u_w = shard_mgr.get_tensor(u_k, device=DEV)
            d_w = shard_mgr.get_tensor(d_k, device=DEV)
            gu[e, :I, :] = g_w
            gu[e, I:, :] = u_w
            dn[e, :, :] = d_w
            del g_w, u_w, d_w

        # Load non-expert weights into layer state dict
        non_expert_keys = [k for k in layer_keys if ".mlp.experts." not in k]
        non_expert_sd = {}
        for k in non_expert_keys:
            sub_k = k[len(pfx):]
            non_expert_sd[sub_k] = shard_mgr.get_tensor(k, device=DEV)

        # Assign non-expert weights to layer
        layer.load_state_dict(non_expert_sd, strict=False, assign=True)
        # Materialize real expert weights into layer.mlp.experts on DEV
        layer.mlp.experts.gate_up_proj = torch.nn.Parameter(gu, requires_grad=False)
        layer.mlp.experts.down_proj = torch.nn.Parameter(dn, requires_grad=False)
        layer.eval()

        # Hessian capture setup
        Hgu = torch.zeros((E, H, H), dtype=torch.float32, device=DEV)
        Hdn = torch.zeros((E, I, I), dtype=torch.float32, device=DEV)
        counts = torch.zeros(E, dtype=torch.long, device=DEV)

        def moe_capture_hook(module, args_tuple):
            x = args_tuple[0].reshape(-1, H)
            _, _, routing_idx = module.gate(x)
            for expert_id in torch.unique(routing_idx).tolist():
                rows = (routing_idx == expert_id).any(dim=-1).nonzero().squeeze(1)
                xe = x[rows]
                counts[expert_id] += rows.numel()
                xf = xe.float()
                Hgu[expert_id].addmm_(xf.T, xf)
                g, u = F.linear(xe, gu[expert_id]).chunk(2, dim=-1)
                a = (F.silu(g) * u).float()
                Hdn[expert_id].addmm_(a.T, a)

        hook_handle = layer.mlp.register_forward_pre_hook(moe_capture_hook)

        # Run calibration forward to capture activations
        with torch.no_grad():
            for b in range(0, N_SAMPLES, BATCH):
                layer(hs[b:b + BATCH], position_embeddings=pos_emb, attention_mask=None, position_ids=text_pos)
        hook_handle.remove()
        t1 = time.time()

        # Run GPTQ on routed experts (in chunks)
        out_layer_tensors: Dict[str, torch.Tensor] = {}
        qerr = []

        chunk_size = args.expert_chunk
        for c0 in range(0, E, chunk_size):
            c1 = min(c0 + chunk_size, E)
            Qg, Sg = gptq_quantize(gu[c0:c1].float(), Hgu[c0:c1])
            Qd, Sd = gptq_quantize(dn[c0:c1].float(), Hdn[c0:c1])

            dq_g = dequantize(Qg, Sg)
            dq_d = dequantize(Qd, Sd)
            rel_err = ((dq_g.float() - gu[c0:c1].float()).norm() / gu[c0:c1].float().norm()).item()
            qerr.append(rel_err)

            for j in range(Qg.shape[0]):
                e = c0 + j
                base_e = f"{pfx}mlp.experts.{e}."
                for proj_name, q, s in [
                    ("gate_proj", Qg[j, :I], Sg[j, :I]),
                    ("up_proj", Qg[j, I:], Sg[j, I:]),
                    ("down_proj", Qd[j], Sd[j]),
                ]:
                    out_layer_tensors[base_e + proj_name + ".weight_packed"] = pack_ct(q).cpu()
                    out_layer_tensors[base_e + proj_name + ".weight_scale"] = s.contiguous().cpu()
                    out_layer_tensors[base_e + proj_name + ".weight_shape"] = torch.tensor(list(q.shape), dtype=torch.int64)

            gu[c0:c1] = dq_g
            dn[c0:c1] = dq_d
            del Qg, Sg, Qd, Sd, dq_g, dq_d

        del Hgu, Hdn
        torch.xpu.empty_cache()
        t2 = time.time()

        # Quantize non-expert linear projections (Linear Attention or Self Attention)
        if args.quantize_attn:
            attn_proj_names = []
            if hasattr(layer, "linear_attn"):
                attn_proj_names.extend([
                    "linear_attn.in_proj_a", "linear_attn.in_proj_b",
                    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                    "linear_attn.out_proj"
                ])
            elif hasattr(layer, "self_attn"):
                attn_proj_names.extend([
                    "self_attn.q_proj", "self_attn.k_proj",
                    "self_attn.v_proj", "self_attn.o_proj"
                ])
            # Shared experts
            attn_proj_names.extend([
                "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj"
            ])

            for ap_name in attn_proj_names:
                full_k = f"{pfx}{ap_name}.weight"
                if full_k in non_expert_keys:
                    w = non_expert_sd[ap_name + ".weight"].float()
                    # Use identity-scaled diagonal Hessian for attention / shared experts
                    h_dummy = torch.eye(w.shape[1], dtype=torch.float32, device=DEV).unsqueeze(0)
                    Q_a, S_a = gptq_quantize(w.unsqueeze(0), h_dummy)
                    q_2d = Q_a.squeeze(0)
                    s_2d = S_a.squeeze(0)

                    out_layer_tensors[f"{pfx}{ap_name}.weight_packed"] = pack_ct(q_2d).cpu()
                    out_layer_tensors[f"{pfx}{ap_name}.weight_scale"] = s_2d.contiguous().cpu()
                    out_layer_tensors[f"{pfx}{ap_name}.weight_shape"] = torch.tensor(list(q_2d.shape), dtype=torch.int64)

                    # Replace layer weight with dequantized value for propagation
                    non_expert_sd[ap_name + ".weight"] = dequantize(Q_a, S_a).squeeze(0)
                    del Q_a, S_a, h_dummy

        # Non-quantized tensors remain BF16
        for k in non_expert_keys:
            base_k = k[: -len(".weight")] if k.endswith(".weight") else k
            if f"{base_k}.weight_packed" not in out_layer_tensors:
                out_layer_tensors[k] = shard_mgr.get_tensor(k, device="cpu")

        # Save layer safetensors
        layer_shard_path = out_dir / f"model-layer-{li:02d}.safetensors"
        save_file(out_layer_tensors, str(layer_shard_path), metadata={"format": "pt"})

        # Update layer.mlp.experts with dequantized weights so hidden state propagation
        # passes through the actual quantized MoE block
        layer.mlp.experts.gate_up_proj = torch.nn.Parameter(gu, requires_grad=False)
        layer.mlp.experts.down_proj = torch.nn.Parameter(dn, requires_grad=False)

        # Propagate hidden states through quantized layer to next layer
        with torch.no_grad():
            for b in range(0, N_SAMPLES, BATCH):
                nxt_hs[b:b + BATCH] = layer(
                    hs[b:b + BATCH], position_embeddings=pos_emb, attention_mask=None, position_ids=text_pos
                )
        hs, nxt_hs = nxt_hs, hs

        # Save resumable state
        torch.save({"next_layer": li + 1, "hs": hs.cpu()}, str(state_file))

        avg_err = sum(qerr) / len(qerr) if qerr else 0.0
        cold_experts = int((counts == 0).sum())
        t_tot = time.time() - t0
        log(f"Layer {li:02d} done in {t_tot:.1f}s (hess: {t1-t0:.1f}s, gptq: {t2-t1:.1f}s, fwd: {time.time()-t2:.1f}s) | "
            f"RelErr: {avg_err:.4f} | Cold Experts: {cold_experts}/{E}")

        del layer, non_expert_sd, out_layer_tensors, gu, dn
        gc.collect()
        torch.xpu.empty_cache()

        # Evict source shards that are no longer referenced by any remaining layer
        for s_file, last_l in shard_last_layer.items():
            if last_l == li and s_file != "model-00012-of-00013.safetensors":
                shard_mgr.purge_shard(s_file)

    if end_layer < n_layers:
        log(f"Stopping run at layer {end_layer} as requested. Partial state saved.")
        return 0

    # 6. Base Tensors (ensure model-base.safetensors is saved)
    base_shard_path = out_dir / "model-base.safetensors"
    if not base_shard_path.is_file():
        log("Assembling base model tensors (embed_tokens, lm_head, norms) ...")
        base_tensors: Dict[str, torch.Tensor] = {}
        for k in index_map:
            if not k.startswith(PREFIX + "layers.") and not k.startswith("mtp."):
                base_tensors[k] = shard_mgr.get_tensor(k, device="cpu")
        save_file(base_tensors, str(base_shard_path), metadata={"format": "pt"})
        log(f"Saved {len(base_tensors)} base tensors to {base_shard_path.name}")

    # 7. MTP Speculative Decoding Draft Model Block
    log("Grafting MTP draft block (19 tensors in BF16) ...")
    mtp_shard_path = out_dir / "model-mtp.safetensors"
    if MTP_STAGING.is_file():
        mtp_tensors = load_file(str(MTP_STAGING))
        save_file(mtp_tensors, str(mtp_shard_path), metadata={"format": "pt"})
        log(f"Saved {len(mtp_tensors)} MTP draft tensors to {mtp_shard_path.name}")
    else:
        log(f"WARNING: MTP staging shard {MTP_STAGING} not found; skipping MTP draft block.")

    # 8. Build model.safetensors.index.json
    log("Building model.safetensors.index.json ...")
    weight_map = {}
    total_bytes = 0
    for p in sorted(out_dir.glob("model-*.safetensors")):
        fn = p.name
        with safe_open(str(p), framework="pt") as f:
            for k in f.keys():
                weight_map[k] = fn
        total_bytes += p.stat().st_size

    index_json = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map
    }
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_json, f, indent=2)
    log(f"Index built: {len(weight_map)} tensors across {len(list(out_dir.glob('model-*.safetensors')))} shards.")

    # 9. Build config.json with quantization_config & MTP enabled
    log("Building config.json with W4A16 compressed-tensors configuration ...")
    kat_cfg = json.load(open(cfg_src))
    kat_cfg["text_config"]["mtp_num_hidden_layers"] = 1  # Enable MTP draft head
    kat_cfg["text_config"]["_attn_implementation"] = "sdpa"

    # Assemble ignore list (every tensor that remains BF16)
    ignore_list = [
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*norm.*",
        "re:.*visual.*",
        "re:.*[.]mtp[.].*",
        "re:^mtp[.].*",
        "re:.*mlp[.]gate$",
        "re:.*shared_expert_gate.*",
        "re:.*conv1d.*"
    ]
    if not args.quantize_attn:
        ignore_list.extend([
            "re:.*self_attn.*",
            "re:.*linear_attn.*",
            "re:.*shared_expert[.].*"
        ])

    kat_cfg["quantization_config"] = {
        "config_groups": {
            "config_group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": GROUP,
                    "num_bits": 4,
                    "observer": "gptq",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None
                }
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": sorted(set(ignore_list)),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "version": "0.17.1"
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(kat_cfg, f, indent=2)

    # 10. Copy Sidecar Files (tokenizers, jinja templates, etc.)
    sidecars = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "chat_template.jinja", "generation_config.json",
        "preprocessor_config.json", "configuration.json"
    ]
    for sc in sidecars:
        src_sc = KAT_MXFP4_DIR / sc
        dst_sc = out_dir / sc
        if src_sc.is_file() and not dst_sc.is_file():
            shutil.copy2(str(src_sc), str(dst_sc))

    log(f"=== QUANTIZATION COMPLETE: {out_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
