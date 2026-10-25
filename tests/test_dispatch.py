#!/usr/bin/env python3
"""P1-01 dispatch tests: dense -> Exl3LinearMethod, experts -> Exl3MoEMethod.

Runs as a plain script (this venv has no pytest):

    .venv/bin/python tests/test_dispatch.py [--verbose]

Every case exercises ``Exl3Config.get_quant_method`` through vLLM's real
``resolve_quant_method`` helper, so the tests cover the exact seam that used to
produce ``UnquantizedFusedMoEMethod`` (``routed_experts.py:219-220``).
"""
from __future__ import annotations

import argparse
import sys
import traceback
import types
from collections.abc import Callable

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import resolve_quant_method

from exl3xpu.moe import Exl3MoEMethod, Exl3MoEUnsupported
from exl3xpu.vllm_plugin import Exl3Config, Exl3LinearMethod

DENSE_PREFIX = "layers.0.mlp.gate_proj"
UNQUANTIZED_DENSE_PREFIX = "layers.0.mlp.shared_expert_gate"
EXPERTS_PREFIX = "layers.0.mlp.experts"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def expert_storage(
    num_experts: int = 256,
    K: int = 3,
    skip: tuple[str, ...] = (),
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    """Checkpoint storage map for one routed-expert layer."""
    st: dict[str, int] = {}
    for e in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            st[f"{EXPERTS_PREFIX}.{e}.{proj}"] = K
    for key in skip:
        st.pop(key, None)
    st.update(overrides or {})
    return st


def dense_storage() -> dict[str, int]:
    # K=4 is what the real checkpoint uses for attention/shared-expert linears.
    return {
        DENSE_PREFIX: 4,
        "layers.0.mlp.up_proj": 4,
        "layers.0.mlp.down_proj": 4,
        "lm_head": 6,
    }


def make_config(storage: dict[str, int], codebook: str = "mcg") -> Exl3Config:
    cfg = Exl3Config(bits=3, head_bits=6, codebook=codebook, storage=dict(storage))
    # One real-shaped entry exercises the member_trellis (n, K) path.
    if f"{EXPERTS_PREFIX}.0.gate_proj" in cfg.storage:
        cfg.member_trellis = {
            f"{EXPERTS_PREFIX}.0.gate_proj": (512, cfg.storage[f"{EXPERTS_PREFIX}.0.gate_proj"]),
        }
    return cfg


def dense_layer() -> LinearBase:
    return object.__new__(LinearBase)


def experts_layer(
    num_experts: int = 256, tp_size: int = 1, ep_size: int = 1
) -> RoutedExperts:
    layer = object.__new__(RoutedExperts)
    layer.layer_name = EXPERTS_PREFIX
    layer.moe_config = types.SimpleNamespace(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_dim=2048,
        intermediate_size_per_partition=512,
        activation="silu",
        moe_parallel_config=types.SimpleNamespace(
            tp_size=tp_size, ep_size=ep_size, use_ep=ep_size > 1
        ),
    )
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    return layer


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def case_dense_resolves_to_exl3() -> None:
    cfg = make_config(dense_storage())
    method = resolve_quant_method(cfg, dense_layer(), DENSE_PREFIX)
    assert isinstance(method, Exl3LinearMethod), type(method)
    assert method.K == 4 and method.cb == 1, (method.K, method.cb)


def case_unquantized_dense_still_unquantized() -> None:
    cfg = make_config(dense_storage())
    method = resolve_quant_method(cfg, dense_layer(), UNQUANTIZED_DENSE_PREFIX)
    assert isinstance(method, UnquantizedLinearMethod), type(method)


def case_experts_resolve_to_exl3_moe() -> None:
    cfg = make_config(expert_storage())
    layer = experts_layer()
    method = resolve_quant_method(cfg, layer, EXPERTS_PREFIX)
    assert isinstance(method, Exl3MoEMethod), type(method)
    assert not isinstance(method, UnquantizedFusedMoEMethod)
    assert method.K == 3 and method.cb == 1, (method.K, method.cb)
    assert method.num_local_experts == 256
    # Modular-kernel contract: no vLLM fused-MoE kernel, runner computes top-k
    # and calls apply().
    assert method.moe_kernel is None
    assert method.supports_internal_mk is False
    assert method.is_monolithic is False
    assert method.get_fused_moe_quant_config(layer) is None


def case_experts_direct_get_quant_method() -> None:
    cfg = make_config(expert_storage())
    method = cfg.get_quant_method(experts_layer(), EXPERTS_PREFIX)
    assert isinstance(method, Exl3MoEMethod), type(method)


def case_model_qualified_prefix_normalizes() -> None:
    # vLLM passes the full model prefix; checkpoint keys are normalized.
    cfg = make_config(expert_storage())
    qualified = "model.language_model.layers.0.mlp.experts"
    method = cfg.get_quant_method(experts_layer(), qualified)
    assert isinstance(method, Exl3MoEMethod), type(method)
    assert method.key_prefix == EXPERTS_PREFIX, method.key_prefix


def case_apply_implemented() -> None:
    """P2-01: the eager routed-expert path is implemented (no stub/fallback)."""
    cfg = make_config(expert_storage())
    layer = experts_layer()
    method = cfg.get_quant_method(layer, EXPERTS_PREFIX)
    assert method.is_monolithic is False
    assert callable(method.apply)
    # Empty work returns an empty result before touching any expert weight.
    out = method.apply(
        layer, torch.zeros(0, 2048), torch.zeros(0, 8),
        torch.zeros(0, 8, dtype=torch.int32), None, None,
    )
    assert tuple(out.shape) == (0, 2048), tuple(out.shape)


def expect_unsupported(
    fn: Callable[[], object], contains: str, label: str
) -> None:
    try:
        fn()
    except Exl3MoEUnsupported as exc:
        assert contains in str(exc), f"{label}: expected {contains!r} in {exc!s}"
    else:
        raise AssertionError(f"{label}: expected Exl3MoEUnsupported, got a method")


def case_no_expert_tensors_raise() -> None:
    cfg = make_config(dense_storage())
    layer = experts_layer()
    expect_unsupported(
        lambda: cfg.get_quant_method(layer, EXPERTS_PREFIX),
        "no EXL3 expert tensors",
        "unquantized MoE",
    )


def case_partial_expert_raises() -> None:
    st = expert_storage(skip=(f"{EXPERTS_PREFIX}.7.down_proj",))
    cfg = make_config(st)
    expect_unsupported(
        lambda: cfg.get_quant_method(experts_layer(), EXPERTS_PREFIX),
        "expert 7 is missing quantized projections ['down_proj']",
        "partial expert",
    )


def case_expert_count_mismatch_raises() -> None:
    cfg = make_config(expert_storage(num_experts=255))
    expect_unsupported(
        lambda: cfg.get_quant_method(experts_layer(), EXPERTS_PREFIX),
        "checkpoint has 255 experts",
        "expert count",
    )


def case_mixed_K_raises() -> None:
    st = expert_storage(overrides={f"{EXPERTS_PREFIX}.0.gate_proj": 4})
    cfg = make_config(st)
    expect_unsupported(
        lambda: cfg.get_quant_method(experts_layer(), EXPERTS_PREFIX),
        "mixed trellis K",
        "mixed K",
    )


def case_unsupported_K_raises() -> None:
    cfg = make_config(expert_storage(K=5))
    expect_unsupported(
        lambda: cfg.get_quant_method(experts_layer(), EXPERTS_PREFIX),
        "no ESIMD kernel for K=5",
        "unsupported K",
    )


def case_empty_storage_raises() -> None:
    cfg = make_config({})
    expect_unsupported(
        lambda: cfg.get_quant_method(experts_layer(), EXPERTS_PREFIX),
        "storage map is empty",
        "empty storage",
    )


def case_tp_ep_raise() -> None:
    cfg = make_config(expert_storage())
    for kwargs, label in (
        ({"tp_size": 2}, "tp>1"),
        ({"ep_size": 2}, "ep>1"),
    ):
        expect_unsupported(
            lambda kw=kwargs: cfg.get_quant_method(experts_layer(**kw), EXPERTS_PREFIX),
            "parallelism is not supported",
            label,
        )


def case_unknown_codebook_raises() -> None:
    cfg = make_config(expert_storage(), codebook="mul2")
    expect_unsupported(
        lambda: cfg.get_quant_method(experts_layer(), EXPERTS_PREFIX),
        "unknown codebook",
        "unknown codebook",
    )


def case_no_unquantized_symbol_in_moe_module() -> None:
    import exl3xpu.moe as moe_mod

    src = open(moe_mod.__file__).read()
    # The MoE adapter must not import or construct the forbidden fallback. The
    # class name may appear in prose (docstrings explain why), so key off code
    # constructs only.
    assert "import UnquantizedFusedMoEMethod" not in src
    assert "UnquantizedFusedMoEMethod(" not in src


CASES: list[tuple[str, Callable[[], None]]] = [
    ("dense linear -> Exl3LinearMethod", case_dense_resolves_to_exl3),
    ("unquantized dense stays unquantized", case_unquantized_dense_still_unquantized),
    ("routed experts -> Exl3MoEMethod", case_experts_resolve_to_exl3_moe),
    ("direct get_quant_method -> Exl3MoEMethod", case_experts_direct_get_quant_method),
    ("model-qualified prefix normalizes", case_model_qualified_prefix_normalizes),
    ("apply implemented (empty work)", case_apply_implemented),
    ("unquantized MoE raises", case_no_expert_tensors_raise),
    ("partially-quantized expert raises", case_partial_expert_raises),
    ("expert-count mismatch raises", case_expert_count_mismatch_raises),
    ("mixed-K experts raise", case_mixed_K_raises),
    ("unsupported K raises", case_unsupported_K_raises),
    ("empty storage raises", case_empty_storage_raises),
    ("TP/EP raise", case_tp_ep_raise),
    ("unknown codebook raises", case_unknown_codebook_raises),
    ("moe.py has no unquantized fallback", case_no_unquantized_symbol_in_moe_module),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", type=str, default=None,
                    help="write a machine-readable summary (evidence artifact)")
    args = ap.parse_args()

    failures: list[str] = []
    results: list[dict] = []
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:
            failures.append(name)
            results.append({"case": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}")
            if args.verbose:
                traceback.print_exc()
        else:
            results.append({"case": name, "ok": True})
            print(f"OK   {name}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} dispatch cases pass")

    if args.json_out:
        import hashlib
        import json
        from pathlib import Path

        import exl3xpu.moe as moe_mod
        from exl3xpu import ops

        def sha(path: str) -> str:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        summary = {
            "task": "P1-01",
            "pass": not failures,
            "cases": results,
            "failures": failures,
            "device": torch.xpu.get_device_properties(0).name,
            "torch": torch.__version__,
            "requested_backend": __import__("os").environ.get("EXL3_BACKEND", "auto"),
            "small_m_max_k3_cb1": ops.small_m_max(3, 1),
            "moe_method_module": moe_mod.__file__,
            "moe_method_sha256": sha(moe_mod.__file__),
            "test_sha256": sha(__file__),
        }
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.json_out}")

    if failures:
        print("failures:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
