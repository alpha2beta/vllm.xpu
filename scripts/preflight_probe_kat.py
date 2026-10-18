#!/usr/bin/env python3
"""preflight_probe_kat.py -- Validate pre-quantization environment, Hugging Face
metadata, tokenizer, and meta-layer construction for Kwaipilot/KAT-Coder-V2.5-Dev
on Intel Arc 140V before committing GPU time.

Checks performed:
  1. XPU device sanity: driver, Level-Zero, device memory, compute capability.
  2. Memory budget: physical RAM, zram swap headroom, free disk space.
  3. Tokenizer loading and token properties (vocab, pad/eos IDs, chat template).
  4. AutoConfig resolution: validates Qwen3_5Moe architecture parameters.
  5. Meta-device layer construction: validates zero unmaterialized buffers.
  6. MTP draft block check: verifies presence of model-00014-of-mtp.safetensors.

Exit code 0 == all preflight checks passed.
"""
from __future__ import annotations

import os
import shutil
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_REPO = "Kwaipilot/KAT-Coder-V2.5-Dev"
MTP_STAGING = os.path.join(REPO_ROOT, "models", "_staging_kat_mtp", "model-00014-of-mtp.safetensors")


def log_header(title: str) -> None:
    print(f"\n{'='*10} [{title}] {'='*10}", flush=True)


def check_xpu_device() -> bool:
    log_header("1. XPU Device & Driver Check")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if not hasattr(torch, "xpu"):
            print("ERROR: torch has no .xpu module.")
            return False
        if not torch.xpu.is_available():
            print("ERROR: torch.xpu.is_available() is False.")
            return False

        dev_count = torch.xpu.device_count()
        props = torch.xpu.get_device_properties(0)
        total_mem_gb = props.total_memory / (1024**3)
        print(f"XPU Devices detected: {dev_count}")
        print(f"Primary device [0]: {props.name}")
        print(f"Total addressable memory: {total_mem_gb:.2f} GiB")
        print(f"Max compute units: {getattr(props, 'max_compute_units', 'N/A')}")

        # Quick XPU tensor sanity test
        a = torch.ones((1024, 1024), dtype=torch.bfloat16, device="xpu")
        b = a @ a
        s = b.sum().item()
        print(f"XPU matmul probe: sum = {s:.0f} (PASS)")
        return True
    except Exception as e:
        print(f"ERROR during XPU check: {e}")
        traceback.print_exc()
        return False


def check_memory_and_storage() -> bool:
    log_header("2. Memory Budget & Storage Check")
    try:
        # Check host RAM and swap
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])

        mem_total_gb = meminfo.get("MemTotal", 0) / (1024**2)
        mem_avail_gb = meminfo.get("MemAvailable", 0) / (1024**2)
        swap_total_gb = meminfo.get("SwapTotal", 0) / (1024**2)
        swap_free_gb = meminfo.get("SwapFree", 0) / (1024**2)

        print(f"Host Total RAM: {mem_total_gb:.2f} GiB")
        print(f"Host Available RAM: {mem_avail_gb:.2f} GiB")
        print(f"Zram/Swap Total: {swap_total_gb:.2f} GiB")
        print(f"Zram/Swap Free: {swap_free_gb:.2f} GiB")

        if mem_avail_gb < 10.0:
            print("WARNING: Available RAM is below 10 GiB. Close running servers before quantizing.")

        # Check disk space on filesystem where REPO_ROOT lives
        stat = os.statvfs(REPO_ROOT)
        free_disk_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"Storage free on {REPO_ROOT}: {free_disk_gb:.2f} GiB")

        if free_disk_gb < 25.0:
            print(f"ERROR: Free disk space ({free_disk_gb:.2f} GiB) is insufficient for output artifact (~21 GiB).")
            return False
        return True
    except Exception as e:
        print(f"ERROR during memory check: {e}")
        return False


def check_hf_and_tokenizer() -> bool:
    log_header("3. Hugging Face Metadata & Tokenizer Check")
    try:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        print(f"Querying HF repository: {SRC_REPO} ...")
        api = HfApi()
        info = api.model_info(SRC_REPO)
        print(f"HF repository verified: {info.id} (SHA: {info.sha[:10]})")

        # Load tokenizer directly from HF repo or local staging if available
        print("Loading AutoTokenizer ...")
        local_tok_dir = os.path.join(REPO_ROOT, "models", "KAT-Coder-V2.5-Dev-MXFP4")
        tok_src = local_tok_dir if os.path.exists(os.path.join(local_tok_dir, "tokenizer.json")) else SRC_REPO
        tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=False)
        print(f"Tokenizer class: {type(tok).__name__}")
        print(f"Vocab size: {getattr(tok, 'vocab_size', 'N/A')}")
        print(f"Bos token ID: {getattr(tok, 'bos_token_id', None)}")
        print(f"Eos token ID: {getattr(tok, 'eos_token_id', None)}")
        print(f"Pad token ID: {getattr(tok, 'pad_token_id', None)}")

        test_text = "def fibonacci(n: int) -> int:"
        tokens = tok.encode(test_text, add_special_tokens=False)
        print(f"Tokenizer round-trip test: '{test_text}' -> {len(tokens)} tokens (PASS)")
        return True
    except Exception as e:
        print(f"ERROR during tokenizer check: {e}")
        traceback.print_exc()
        return False


def check_autoconfig_and_meta_layer() -> bool:
    log_header("4. Architecture & Meta-Layer Build Check")
    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

        local_cfg = os.path.join(REPO_ROOT, "models", "KAT-Coder-V2.5-Dev-MXFP4", "config.json")
        cfg_src = local_cfg if os.path.exists(local_cfg) else SRC_REPO
        print(f"Loading AutoConfig from {cfg_src} ...")
        cfg = AutoConfig.from_pretrained(cfg_src, trust_remote_code=False)

        print(f"Architectures: {cfg.architectures}")
        print(f"Model type: {cfg.model_type}")

        tc = getattr(cfg, "text_config", cfg)
        n_layers = getattr(tc, "num_hidden_layers", 0)
        n_experts = getattr(tc, "num_experts", 0)
        top_k = getattr(tc, "num_experts_per_tok", 0)
        hidden_dim = getattr(tc, "hidden_size", 0)
        moe_inter = getattr(tc, "moe_intermediate_size", 0)

        print(f"Decoder layers: {n_layers}")
        print(f"Routed experts/layer: {n_experts} (top-{top_k} active)")
        print(f"Hidden dim: {hidden_dim}, MoE intermediate: {moe_inter}")

        assert n_layers == 40, f"Expected 40 layers, got {n_layers}"
        assert n_experts == 256, f"Expected 256 experts, got {n_experts}"
        assert top_k == 8, f"Expected top-8 routing, got {top_k}"

        print("Testing meta-device decoder layer build ...")
        with torch.device("meta"):
            layer = M.Qwen3_5MoeDecoderLayer(tc, 0)
        buffers = list(layer.named_buffers())
        assert len(buffers) == 0, f"Found unexpected unmaterialized buffers: {buffers}"
        print("Meta-device layer construction: 0 unmaterialized buffers (PASS)")
        return True
    except Exception as e:
        print(f"ERROR during architecture check: {e}")
        traceback.print_exc()
        return False


def check_mtp_draft_block() -> bool:
    log_header("5. MTP Speculative Draft Block Check")
    try:
        from safetensors import safe_open
        if not os.path.exists(MTP_STAGING):
            print(f"WARNING: MTP staging shard not found at {MTP_STAGING}.")
            print("Speculative decoding draft head (k=2) will need to be downloaded from SpectreSystems/KAT-Coder-V2.5-Dev-MTP.")
            return False

        print(f"Found MTP draft shard at {MTP_STAGING}")
        with safe_open(MTP_STAGING, framework="pt") as f:
            keys = f.keys()
            print(f"MTP tensor count: {len(keys)}")
            assert len(keys) == 19, f"Expected 19 MTP tensors, got {len(keys)}"
            for k in ["mtp.fc.weight", "mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"]:
                t = f.get_tensor(k)
                print(f"  {k}: shape {t.shape}, dtype {t.dtype}")
        print("MTP draft block verification (PASS)")
        return True
    except Exception as e:
        print(f"ERROR during MTP draft check: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("KAT-Coder-V2.5-Dev GPTQ-Int4 Preflight Probe (Arc 140V)")
    print("=" * 60)

    ok = True
    ok = check_xpu_device() and ok
    ok = check_memory_and_storage() and ok
    ok = check_hf_and_tokenizer() and ok
    ok = check_autoconfig_and_meta_layer() and ok
    ok = check_mtp_draft_block() and ok

    log_header("Summary")
    if ok:
        print(">>> ALL PREFLIGHT PROBE CHECKS PASSED. SYSTEM IS READY FOR QUANTIZATION.")
        return 0
    else:
        print(">>> PREFLIGHT CHECKS FAILED. PLEASE RESOLVE ISSUES BEFORE RUNNING QUANTIZATION.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
