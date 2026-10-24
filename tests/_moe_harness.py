"""Shared harness for Phase 2 MoE tests (P2-01..P2-09).

Loads a fully-populated layer-0 routed-expert layer from the pinned 3.09bpw
checkpoint, plus the router and shared expert, and exposes:

* ``Harness.router`` / ``restricted_routing`` — reference routing;
* ``Harness.routed`` — ``Exl3MoEMethod.apply``;
* ``Harness.shared`` — the EXL3 shared expert with its learned gate;
* ``Harness.free_routing_reference`` — CPU fp32 reference for selected experts.

It is imported by ``tests/test_moe_block.py`` and ``tests/test_moe_routing.py``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from exl3xpu import ops
from exl3xpu import ref
from exl3xpu.moe import Exl3MoEMethod, stream_expert_shards
from exl3xpu.vllm_plugin import Exl3Config

NUM_EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 512
TOPK = 8
T2_MAX_ABS = 0.08
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"


class Checkpoint:
    """Key -> tensor access through the shard index."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.weight_map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._handles: dict[str, object] = {}

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.model_dir / shard), framework="pt")
        return self._handles[shard].get_tensor(key)


def make_layer(layer_id: int) -> RoutedExperts:
    """Minimal but faithful ``RoutedExperts`` (the real ctor needs an engine)."""
    layer = object.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
    layer.layer_name = prefix
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.lora_base_layer_prefix = ""
    layer.is_fused_checkpoint_transposed = False
    layer.moe_config = types.SimpleNamespace(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_dim=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        moe_parallel_config=types.SimpleNamespace(
            tp_size=1, ep_size=1, use_ep=False, enable_eplb=False
        ),
    )
    layer.expert_map_manager = types.SimpleNamespace(num_fused_shared_experts=0)
    return layer


class Harness:
    """Layer-0 routed experts + shared expert + router, from the checkpoint."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL, layer_id: int = 0):
        model_dir = Path(model_dir)
        self.ckpt = Checkpoint(model_dir)
        self.layer_id = layer_id
        self.mlp = f"model.language_model.layers.{layer_id}.mlp"
        # Platform baseline: M>8 never takes the DPAS path on Arc 140V.
        ops.SMALL_M_MAX = 8

        qc = json.loads((model_dir / "quantization_config.json").read_text())
        self.config = Exl3Config(
            bits=float(qc["bits"]),
            head_bits=int(qc["head_bits"]),
            codebook=qc["codebook"],
        )
        self.config.maybe_update_config(str(model_dir))

        self.layer = make_layer(layer_id)
        self.method = Exl3MoEMethod(self.config, self.layer, f"{self.mlp}.experts")
        with torch.device("xpu"):
            self.method.create_weights(
                self.layer, NUM_EXPERTS, HIDDEN, INTERMEDIATE, torch.bfloat16
            )
        self.layer.quant_method = self.method
        key = f"layers.{layer_id}.mlp.experts"
        shards = sorted(set(self.ckpt.weight_map.values()))
        stream_expert_shards([str(model_dir / s) for s in shards], {key: self.layer})
        self.method.process_weights_after_loading(self.layer)

        self.router_w = self.ckpt.get(f"{self.mlp}.gate.weight").float().to("xpu")
        self.shared_gate_w = (
            self.ckpt.get(f"{self.mlp}.shared_expert_gate.weight").float().to("xpu")
        )
        self._wcache: dict[int, dict[str, torch.Tensor]] = {}

    # -- routing -----------------------------------------------------------
    def router(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = x.float() @ self.router_w.T
        top_w, top_i = torch.topk(logits, TOPK, dim=-1)
        top_w = torch.softmax(top_w.float(), dim=-1)
        return logits, top_i.to(torch.int32), top_w

    def restricted_routing(self, x: torch.Tensor, experts=(0, 255)):
        """Oracle block routing: softmax over two available experts."""
        logits = x.float() @ self.router_w.T
        w = torch.softmax(logits[:, list(experts)].float(), dim=-1)
        ids = torch.tensor([list(experts)], dtype=torch.int32, device=x.device).repeat(
            x.shape[0], 1
        )
        return ids, w

    # -- execution ---------------------------------------------------------
    def routed(self, x, ids, weights):
        return self.method.apply(
            layer=self.layer,
            x=x,
            topk_weights=weights,
            topk_ids=ids,
            shared_experts=None,
            shared_experts_input=None,
        )

    def dense_linear(self, x, base: str):
        tr = self.ckpt.get(f"{base}.trellis")
        suh = self.ckpt.get(f"{base}.suh")
        svh = self.ckpt.get(f"{base}.svh")
        cb = 1 if self.ckpt.has(f"{base}.mcg") else (
            2 if self.ckpt.has(f"{base}.mul1") else 0
        )
        K = tr.shape[-1] // 16
        n = tr.shape[1] * 16
        nb = torch.zeros(n // 128, dtype=torch.int32, device=x.device)
        return ops.exl3_linear(
            x, tr.to("xpu"), suh.view(1, -1).to("xpu"), svh.to("xpu"),
            nb, [0, n], K, cb,
        )

    def shared(self, x: torch.Tensor) -> torch.Tensor:
        base = f"{self.mlp}.shared_expert"
        g = self.dense_linear(x, f"{base}.gate_proj")
        u = self.dense_linear(x, f"{base}.up_proj")
        h = self.dense_linear(F.silu(g) * u, f"{base}.down_proj")
        scalar = torch.sigmoid(x.float() @ self.shared_gate_w.T)
        return h.float() * scalar

    # -- reference ---------------------------------------------------------
    def _expert_weights_cpu(self, e: int) -> dict[str, torch.Tensor]:
        if e not in self._wcache:
            views = self.method.expert_views(self.layer, e)
            self._wcache[e] = {
                p: ref.weight_orig(
                    views[p]["trellis"].cpu(),
                    views[p]["suh"].cpu(),
                    views[p]["svh"].cpu(),
                    views[p]["K"],
                    views[p]["cb"],
                )
                for p in PROJECTIONS
            }
        return self._wcache[e]

    def expert_mlp_reference(self, e: int, x: torch.Tensor, w: float = 1.0) -> torch.Tensor:
        m = self._expert_weights_cpu(e)
        xi = x.float()
        h = F.silu(xi @ m["gate_proj"]) * (xi @ m["up_proj"])
        return w * (h @ m["down_proj"])

    def free_routing_reference(
        self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros(x.shape, dtype=torch.float32)
        for t in range(x.shape[0]):
            for slot in range(ids.shape[1]):
                out[t] += self.expert_mlp_reference(
                    int(ids[t, slot]), x[t], float(weights[t, slot])
                )
        return out


def metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    a, b = got.float(), want.float()
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99).item())
        if diff.numel() else 0.0,
        "nan_inf": int(torch.isnan(a).sum().item() + torch.isinf(a).sum().item()),
    }
