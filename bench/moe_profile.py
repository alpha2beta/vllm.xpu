#!/usr/bin/env python3
"""P4-01: MoE / decoder-layer decode profiler and host-sync auditor.

Profiles a single-token (M=1) decode step through the real layer-0
``Qwen3_5DecoderLayer`` on XPU and reports:

* **per-stage device time** (XPU profiler): norms, GDN attention, routed MoE,
  shared expert;
* **host-side synchronization audit**: counts calls to ``torch.Tensor.item``,
  ``.tolist()``, ``__int__``, ``__float__`` made inside the decode loop — each
  of these forces a device→host sync and serialises the pipeline. The Gate 4
  target for single-sequence decode is **zero**.

    .venv/bin/python bench/moe_profile.py [--decode-steps 50] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import exl3xpu.vllm_plugin  # noqa: F401,E402  registers "exl3"

MODEL_DIR = Path("/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw")
LAYER_ID = 0


class SyncCounter:
    """Counts device->host sync primitives touched during a scope."""

    def __init__(self):
        self.calls = {"item": 0, "tolist": 0, "__int__": 0, "__float__": 0,
                      "__index__": 0}
        self._saved = {}

    def __enter__(self):
        self._saved = {
            "item": torch.Tensor.item,
            "tolist": torch.Tensor.tolist,
            "__int__": torch.Tensor.__int__,
            "__float__": torch.Tensor.__float__,
            "__index__": torch.Tensor.__index__,
        }
        calls = self.calls

        def _wrap(orig, name):
            def f(self, *a, **k):
                calls[name] += 1
                return orig(self, *a, **k)
            return f

        torch.Tensor.item = _wrap(self._saved["item"], "item")
        torch.Tensor.tolist = _wrap(self._saved["tolist"], "tolist")
        torch.Tensor.__int__ = _wrap(self._saved["__int__"], "__int__")
        torch.Tensor.__float__ = _wrap(self._saved["__float__"], "__float__")
        torch.Tensor.__index__ = _wrap(self._saved["__index__"], "__index__")
        return self

    def __exit__(self, *exc):
        torch.Tensor.item = self._saved["item"]
        torch.Tensor.tolist = self._saved["tolist"]
        torch.Tensor.__int__ = self._saved["__int__"]
        torch.Tensor.__float__ = self._saved["__float__"]
        torch.Tensor.__index__ = self._saved["__index__"]
        return False

    def total(self) -> int:
        return sum(self.calls.values())


def build_layer(args):
    """Real layer-0 decoder layer with checkpoint weights on XPU."""
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig, VllmConfig,
                             set_current_vllm_config)
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)

    mc = ModelConfig(model=str(MODEL_DIR), tokenizer=str(MODEL_DIR),
                     dtype="bfloat16", max_model_len=2048,
                     trust_remote_code=True, enforce_eager=True)
    vc = VllmConfig(
        model_config=mc,
        cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.5),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
    )
    import os
    os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")

    with set_current_vllm_config(vc):
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1,
                                  pipeline_model_parallel_size=1)

        # Arc 140V has no XMX: M>8 must take reconstruct+GEMM, not DPAS.
        from exl3xpu import ops as _ops
        _ops.SMALL_M_MAX = 8
        if args.target_threads:
            esimd = _ops._get_esimd()
            esimd.exl3_set_target_threads(args.target_threads)

        torch.set_default_dtype(torch.bfloat16)
        from vllm.model_executor.models.qwen3_5 import (Qwen3_5DecoderLayer,
                                                        Qwen3_5Model)
        from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                                      WeightsMapper)
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        import torch.nn as nn

        with torch.device("xpu"):
            layer = Qwen3_5DecoderLayer(
                vc, "linear_attention", prefix=f"model.layers.{LAYER_ID}")

        class Wrapper(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([layer])

        wrapper = Wrapper(layer)
        mapper = (
            WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
            | Qwen3_5Model.hf_to_vllm_mapper
        )
        wm = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"model.language_model.layers.{LAYER_ID}."
        from safetensors import safe_open
        handles = {}

        def gen():
            for key, shard in wm.items():
                if key.startswith(prefix):
                    if shard not in handles:
                        handles[shard] = safe_open(str(MODEL_DIR / shard), framework="pt")
                    yield key, handles[shard].get_tensor(key)

        AutoWeightsLoader(wrapper).load_weights(gen(), mapper=mapper)
        process_weights_after_loading(wrapper, mc, torch.device("xpu"))

        # GDN recurrent state + forward context for a 1-token decode step.
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        cfg = mc.hf_text_config
        conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_world_size=1, num_k_heads=cfg.linear_num_key_heads,
            num_v_heads=cfg.linear_num_value_heads, head_k_dim=cfg.linear_key_head_dim,
            head_v_dim=cfg.linear_value_head_dim,
            conv_kernel_size=cfg.linear_conv_kernel_dim)
        attn = layer.linear_attn
        conv = torch.zeros((1, *conv_shape), dtype=torch.bfloat16, device="xpu")
        ssm = torch.zeros((1, *ssm_shape), dtype=torch.float32, device="xpu")
        attn.kv_cache = (conv, ssm)
        vc.compilation_config.static_forward_context[attn.prefix] = attn

        meta = GDNAttentionMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=1,
            num_decode_tokens=1, num_spec_decodes=0, num_spec_decode_tokens=0,
            num_actual_tokens=1,
            has_initial_state=torch.tensor([True], dtype=torch.bool, device="xpu"),
            non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="xpu"),
            non_spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32, device="xpu"),
        )

        def decode_step(x):
            with set_forward_context(attn_metadata={attn.prefix: meta},
                                     vllm_config=vc, num_tokens=1):
                moe_delta, residual = layer(x, None, torch.zeros(1, dtype=torch.int32, device="xpu"))
            return residual + moe_delta

        return decode_step, layer, vc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json-out", default=str(ROOT / "logs/moe-p4/moe-profile.json"))
    ap.add_argument("--label", default="")
    ap.add_argument("--target-threads", type=int, default=0,
                    help="override exl3_set_target_threads (0 = driver default)")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        raise SystemExit("P4-01 profiler needs the XPU")

    from vllm.config import set_current_vllm_config

    decode_step, layer, vc = build_layer(args)
    # Re-enter the config context for the forward pass (custom ops read it).
    with set_current_vllm_config(vc):
        return run_benchmark(decode_step, layer, args)


def run_benchmark(decode_step, layer, args) -> int:
    x = (torch.randn(1, 2048) * 0.05).to(torch.bfloat16).to("xpu")

    # The layer's fused_add_rms_norm mutates its residual/input in place, so
    # feed a fresh copy each step (mirrors a real decode, where the input is
    # the previous step's hidden state).
    def step():
        return decode_step(x.clone())

    # --- warmup (JIT / autotune) -----------------------------------------
    for _ in range(args.warmup):
        step()
    torch.xpu.synchronize()

    # --- timed decode loop -------------------------------------------------
    ev_start = torch.xpu.Event(enable_timing=True)
    ev_end = torch.xpu.Event(enable_timing=True)
    ev_start.record()
    for _ in range(args.decode_steps):
        step()
    ev_end.record()
    torch.xpu.synchronize()
    gpu_ms = ev_start.elapsed_time(ev_end)

    # --- host-sync audit ----------------------------------------------------
    with SyncCounter() as counter:
        for _ in range(args.decode_steps):
            step()
    torch.xpu.synchronize()
    syncs_per_token = counter.total() / max(args.decode_steps, 1)

    # --- per-stage device time ----------------------------------------------
    with profile(activities=[ProfilerActivity.XPU]) as prof:
        for _ in range(10):
            step()
        torch.xpu.synchronize()
    stages = {}
    for row in prof.key_averages():
        if getattr(row, "self_device_time_total", 0) <= 0:
            continue
        name = row.key.split("(")[0]
        stages[name] = {"calls": int(row.count),
                        "device_us_total": round(row.self_device_time_total, 1),
                        "device_us_per_token": round(row.self_device_time_total / 10, 2)}

    tok_per_s = args.decode_steps / (gpu_ms / 1e3)
    per_token_ms = gpu_ms / args.decode_steps
    summary = {
        "task": "P4-01",
        "label": args.label,
        "device": torch.xpu.get_device_properties(0).name,
        "decode_steps": args.decode_steps,
        "gpu_ms_total": round(gpu_ms, 3),
        "per_token_ms": round(per_token_ms, 4),
        "tok_per_s": round(tok_per_s, 3),
        "host_syncs_per_token": round(syncs_per_token, 3),
        "host_syncs": counter.calls,
        "kernel_launches_per_token": round(
            sum(v["calls"] for v in stages.values()) / 10, 2),
        "stages": stages,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"decode per-token {per_token_ms:.3f} ms ({tok_per_s:.2f} tok/s) "
          f"| host syncs/token {syncs_per_token:.1f} | "
          f"kernel launches/token {summary['kernel_launches_per_token']}")
    print(f"  syncs: {counter.calls}")
    print(f"  stages (device us/token):")
    for name, v in sorted(stages.items(), key=lambda kv: -kv[1]["device_us_total"]):
        print(f"    {v['device_us_per_token']:>9.2f}  x{v['calls']:>3}  {name}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
