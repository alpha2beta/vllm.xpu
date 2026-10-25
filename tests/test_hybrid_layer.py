#!/usr/bin/env python3
"""P2-09: complete hybrid decoder layer (GDN + EXL3 MoE), end-to-end.

Assembles the real layer-0 ``Qwen3_5DecoderLayer`` (input GemmaRMSNorm → GDN
linear attention → post-attention GemmaRMSNorm → EXL3 MoE → residual) with the
pinned checkpoint weights, and validates it stage by stage against independent
references:

* **attention** — a second, *unquantized* GDN layer on XPU loaded with the EXL3
  projections dequantized through ``ref.weight_orig`` (same GDN core kernel,
  plain projections);
* **norms** — a CPU ``GemmaRMSNorm`` reference (``x*(1+w)``);
* **MoE** — the CPU fp32 dequantized routed+shared reference from
  ``tests/_moe_harness.py``;
* **full layer** — ``residual + MoE(norm2(x + attn(x)))``.

The P0-08 CUDA trace does not exist (documented deviation), so the CPU/dense
references above are the authority.

    .venv/bin/python tests/test_hybrid_layer.py

Evidence: ``logs/moe-p2/full-layer-eval.{log,json}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import exl3xpu.vllm_plugin  # noqa: F401,E402  registers "exl3"
from _moe_harness import Harness, metrics, T2_MAX_ABS, DEFAULT_MODEL  # noqa: E402
from safetensors import safe_open  # noqa: E402


def make_vc(model_dir: Path):
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

    mc = ModelConfig(
        model=str(model_dir), tokenizer=str(model_dir), dtype="bfloat16",
        max_model_len=2048, trust_remote_code=True, enforce_eager=True,
    )
    return VllmConfig(
        model_config=mc,
        cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.5),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
    )


def standalone(vc):
    import os
    from contextlib import contextmanager
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29621")

    @contextmanager
    def _ctx():
        with set_current_vllm_config(vc):
            init_distributed_environment(world_size=1, rank=0, local_rank=0)
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
            yield

    return _ctx()


class Ckpt:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.wm = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._h: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self.wm[key]
        if shard not in self._h:
            self._h[shard] = safe_open(str(self.dir / shard), framework="pt")
        return self._h[shard].get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.wm

    def batch(self, prefix: str):
        for key in self.wm:
            if key.startswith(prefix):
                yield key, self.get(key)


def gdn_state(cfg, device):
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_world_size=1, num_k_heads=cfg.linear_num_key_heads,
        num_v_heads=cfg.linear_num_value_heads, head_k_dim=cfg.linear_key_head_dim,
        head_v_dim=cfg.linear_value_head_dim, conv_kernel_size=cfg.linear_conv_kernel_dim,
    )
    conv = torch.zeros((1, *conv_shape), dtype=torch.bfloat16, device=device)
    ssm = torch.zeros((1, *ssm_shape), dtype=torch.float32, device=device)
    return conv, ssm


def gdn_meta(prefix, num_tokens, device):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    return GDNAttentionMetadata(
        num_prefills=1, num_prefill_tokens=num_tokens, num_decodes=0,
        num_decode_tokens=0, num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=num_tokens,
        has_initial_state=torch.tensor([False], dtype=torch.bool, device=device),
        non_spec_query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        non_spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32, device=device),
    )


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(x.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--log-out", default=str(ROOT / "logs/moe-p2/full-layer-eval.log"))
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p2/full-layer-eval.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    ckpt = Ckpt(model_dir)
    records: list[dict] = []
    failures: list[dict] = []
    stages: dict = {}

    exl3_vc = make_vc(model_dir)
    x = (torch.randn(4, 2048, generator=torch.Generator().manual_seed(1234)) * 0.05)
    x = x.to(torch.bfloat16).to("xpu")

    # Reference MoE/shared/router (independent of the vLLM config context).
    h = Harness(model_dir, args.layer)

    try:
        with standalone(exl3_vc):
            cfg = exl3_vc.model_config.hf_text_config
            torch.set_default_dtype(torch.bfloat16)

            from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5Model
            from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            from vllm.forward_context import set_forward_context

            with torch.device("xpu"):
                layer = Qwen3_5DecoderLayer(
                    exl3_vc, "linear_attention", prefix=f"model.layers.{args.layer}"
                )
            mapper = (
                WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
                | Qwen3_5Model.hf_to_vllm_mapper
            )
            import torch.nn as nn

            class Wrapper(nn.Module):
                def __init__(self, layer):
                    super().__init__()
                    self.model = nn.Module()
                    self.model.layers = nn.ModuleList([layer])

            wrapper = Wrapper(layer)
            prefix = f"model.language_model.layers.{args.layer}."
            AutoWeightsLoader(wrapper).load_weights(ckpt.batch(prefix), mapper=mapper)
            process_weights_after_loading(wrapper, exl3_vc.model_config, torch.device("xpu"))

            attn = layer.linear_attn
            conv, ssm = gdn_state(cfg, "xpu")
            attn.kv_cache = (conv, ssm)
            exl3_vc.compilation_config.static_forward_context[attn.prefix] = attn

            caps: dict[str, torch.Tensor] = {}

            def norm_hook(name):
                # GemmaRMSNorm returns a bare tensor when residual is None and
                # ``(normed, residual)`` otherwise. Clone: the storage of these
                # intermediates is recycled by later allocations.
                def hook(m, i, o):
                    caps[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()
                return hook

            layer.input_layernorm.register_forward_hook(norm_hook("norm1"))
            layer.linear_attn.register_forward_hook(
                lambda m, i, o: caps.__setitem__("attn", o.detach().clone()))
            layer.post_attention_layernorm.register_forward_hook(norm_hook("norm2"))
            layer.mlp.register_forward_hook(
                lambda m, i, o: caps.__setitem__("moe", o.detach().clone()))

            # ``post_attention_layernorm`` uses fused_add_rms_norm, which
            # mutates the residual (the input tensor) in place — keep a copy.
            x_in = x.clone()
            pos = torch.arange(x.shape[0], device="xpu")
            meta = gdn_meta(attn.prefix, x.shape[0], "xpu")
            with set_forward_context(attn_metadata={attn.prefix: meta},
                                     vllm_config=exl3_vc, num_tokens=x.shape[0]):
                moe_delta, residual = layer(x, None, pos)
            torch.xpu.synchronize()
            final = (residual + moe_delta).detach()

            # -- dense attention reference (clean subprocess) --------------
            import subprocess
            import tempfile

            work = Path(tempfile.mkdtemp(prefix="exl3-hybrid-", dir="/tmp/opencode"))
            np.savez(work / "norm1.npz", x=caps["norm1"].float().cpu().numpy())
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tests/_dense_gdn_ref.py"),
                 "--model-dir", str(model_dir), "--layer", str(args.layer),
                 "--input", str(work / "norm1.npz"),
                 "--out", str(work / "attn_ref.npz")],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "dense GDN reference failed:\n" + proc.stderr[-3000:]
                )
            attn_ref_f = (
                torch.from_numpy(np.load(work / "attn_ref.npz")["attn"])
                .float()
                .to("xpu")
            )
            stages["attention_vs_dense"] = metrics(caps["attn"].cpu(), attn_ref_f.cpu())

            # -- norms + MoE reference -------------------------------------
            eps = cfg.rms_norm_eps
            p_w = ckpt.get(f"model.language_model.layers.{args.layer}.post_attention_layernorm.weight")
            # Compose the reference from the layer's actual attention output:
            # the attention sublayer is validated independently above; this
            # chain verifies the norms, residual wiring, MoE and final add.
            residual_ref = x_in + caps["attn"]
            norm2_ref = gemma_rmsnorm(residual_ref, p_w.to("xpu"), eps)
            stages["norm2"] = metrics(caps["norm2"].cpu(), norm2_ref.cpu())

            _, ids, w = h.router(norm2_ref)
            moe_ref = (h.free_routing_reference(norm2_ref.cpu(), ids.cpu(), w.cpu())
                       + h.shared(norm2_ref).cpu())
            stages["moe"] = metrics(caps["moe"].cpu(), moe_ref)
            out_ref = residual_ref.cpu().float() + moe_ref
            stages["full_layer"] = metrics(final.cpu(), out_ref)

            worst = max(v["max_abs"] for v in stages.values())
            ok = worst <= T2_MAX_ABS and all(
                v["nan_inf"] == 0 for v in stages.values()
            )
            stages["worst_max_abs"] = worst
            stages["pass"] = ok
            if not ok:
                failures.append({"stage": "full_layer", "worst": worst})
            print("stage max_abs:", {
                k: (round(v["max_abs"], 8) if isinstance(v, dict) else v)
                for k, v in stages.items()
            })
    except Exception as exc:  # noqa: BLE001
        failures.append({"error": repr(exc)[:500]})
        if args.verbose:
            traceback.print_exc()
        print(f"FAIL: {exc}")

    summary = {
        "task": "P2-09",
        "pass": not failures and stages.get("pass", False),
        "device": torch.xpu.get_device_properties(0).name,
        "tolerance": {"T2_max_abs": T2_MAX_ABS},
        "layer": args.layer,
        "note": (
            "P0-08 has no CUDA trace (documented deviation); references are a "
            "dequantized dense GDN layer on XPU, a CPU GemmaRMSNorm, and the "
            "CPU fp32 dequantized MoE."
        ),
        "stages": stages,
        "failures": failures,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    (out.with_suffix(".log")).write_text(
        "\n".join([f"P2-09 hybrid layer {args.layer} T2={T2_MAX_ABS}"]
                  + [f"  {k}: {json.dumps(v, sort_keys=True)}"
                     for k, v in stages.items() if isinstance(v, dict)]
                  + [f"pass={summary['pass']} failures={failures}"]) + "\n"
    )
    print(f"pass={summary['pass']} failures={failures}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
