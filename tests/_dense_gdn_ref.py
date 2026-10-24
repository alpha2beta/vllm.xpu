#!/usr/bin/env python3
"""P2-09 helper: dense (unquantized) GDN attention reference on XPU.

Runs in a clean process (the mixed-config/global-state collision between the
EXL3 decoder layer and a second plain vLLM config makes an in-process second
layer unreliable). Builds ``QwenGatedDeltaNetAttention`` from the probe config
(quantization stripped), loads the layer-0 GDN weights with the EXL3
projections dequantized through ``ref.weight_orig``, and writes the attention
output for a given input.

    python tests/_dense_gdn_ref.py --input norm1.npz --out attn_ref.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The parent test process holds its own torch.distributed rendezvous; use a
# distinct port so this reference process does not collide with it.
import os
os.environ["MASTER_PORT"] = "29631"

import exl3xpu.vllm_plugin  # noqa: F401,E402
from exl3xpu import ref  # noqa: E402
from safetensors import safe_open  # noqa: E402

import gdn_fixture_probe as gdn_probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--fp16-weights", action="store_true",
                    help="round dequantized projections to fp16 (sensitivity probe)")
    ap.add_argument("--proj", default=None,
                    help="npz with EXL3 qkvz/ba projections; bypass the linears")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    data = np.load(args.input)
    x = torch.from_numpy(data["x"].copy()).to(torch.bfloat16).to("xpu")

    probe_dir = gdn_probe.prepare_probe_config(model_dir)
    vc = gdn_probe.make_vllm_config(probe_dir)
    with gdn_probe.standalone_vllm(probe_dir):
        cfg = vc.model_config.hf_text_config
        torch.set_default_dtype(torch.bfloat16)
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        ckpt_prefix = f"model.language_model.layers.{args.layer}.linear_attn"
        prefix = f"model.language_model.layers.{args.layer}.linear_attn_ref"
        layer = QwenGatedDeltaNetAttention(
            config=cfg, vllm_config=vc, prefix=prefix, gqa_interleaved_layout=False
        )
        wm = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        handles: dict[str, object] = {}

        def get(key: str) -> torch.Tensor:
            shard = wm[key]
            if shard not in handles:
                handles[shard] = safe_open(str(model_dir / shard), framework="pt")
            return handles[shard].get_tensor(key)

        def dequant(base: str) -> torch.Tensor:
            tr = get(f"{base}.trellis")
            suh = get(f"{base}.suh")
            svh = get(f"{base}.svh")
            cb = 1 if f"{base}.mcg" in wm else (2 if f"{base}.mul1" in wm else 0)
            w = ref.weight_orig(tr, suh, svh, tr.shape[-1] // 16, cb).t().contiguous()
            if args.fp16_weights:
                w = w.half().float()
            return w

        with torch.no_grad():
            layer.in_proj_qkvz.weight.data.copy_(
                torch.cat([dequant(f"{ckpt_prefix}.in_proj_qkv"),
                           dequant(f"{ckpt_prefix}.in_proj_z")], 0)
                .to(layer.in_proj_qkvz.weight.dtype)
            )
            layer.in_proj_ba.weight.data.copy_(
                torch.cat([get(f"{ckpt_prefix}.in_proj_b.weight"),
                           get(f"{ckpt_prefix}.in_proj_a.weight")], 0)
                .to(layer.in_proj_ba.weight.dtype)
            )
            layer.out_proj.weight.data.copy_(
                dequant(f"{ckpt_prefix}.out_proj").to(layer.out_proj.weight.dtype)
            )
            layer.conv1d.weight.data.copy_(get(f"{ckpt_prefix}.conv1d.weight"))
            layer.A_log.data.copy_(get(f"{ckpt_prefix}.A_log"))
            layer.dt_bias.data.copy_(get(f"{ckpt_prefix}.dt_bias"))
            layer.norm.weight.data.copy_(get(f"{ckpt_prefix}.norm.weight"))

        layer = layer.to("xpu")
        if args.proj:
            p = np.load(args.proj)
            qkvz = torch.from_numpy(p["qkvz"]).to(torch.bfloat16).to("xpu")
            ba = torch.from_numpy(p["ba"]).to(torch.bfloat16).to("xpu")
            # Same input as the parent run, so the saved projections are valid.
            object.__setattr__(layer, "in_proj_qkvz", lambda x: (qkvz, None))
            object.__setattr__(layer, "in_proj_ba", lambda x: (ba, None))
        conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_world_size=1, num_k_heads=cfg.linear_num_key_heads,
            num_v_heads=cfg.linear_num_value_heads, head_k_dim=cfg.linear_key_head_dim,
            head_v_dim=cfg.linear_value_head_dim, conv_kernel_size=cfg.linear_conv_kernel_dim,
        )
        op_in: dict[str, torch.Tensor] = {}
        layer.out_proj.register_forward_hook(
            lambda m, i, o: op_in.__setitem__("x", i[0].detach())
        )
        conv = torch.zeros((1, *conv_shape), dtype=torch.bfloat16, device="xpu")
        ssm = torch.zeros((1, *ssm_shape), dtype=torch.float32, device="xpu")
        layer.kv_cache = (conv, ssm)
        vc.compilation_config.static_forward_context[prefix] = layer

        n = x.shape[0]
        meta = GDNAttentionMetadata(
            num_prefills=1, num_prefill_tokens=n, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=n,
            has_initial_state=torch.tensor([False], dtype=torch.bool, device="xpu"),
            non_spec_query_start_loc=torch.tensor([0, n], dtype=torch.int32, device="xpu"),
            non_spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32, device="xpu"),
        )
        with set_forward_context(attn_metadata={prefix: meta}, vllm_config=vc, num_tokens=n):
            out = layer.forward_xpu(x)
        torch.xpu.synchronize()

    np.savez(
        args.out,
        attn=out.float().cpu().numpy(),
        out_proj_in=op_in["x"].float().cpu().numpy(),
    )
    print(f"dense GDN ref: {tuple(out.shape)} finite={bool(torch.isfinite(out.float()).all())} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
