#!/usr/bin/env python3
"""P1-02 tests: compressed expert buffer registration and weight loading.

Runs as a plain script (this venv has no pytest):

    .venv/bin/python tests/test_expert_loader.py [--verbose] [--json-out PATH]

The loader tests use *real* 3.09bpw checkpoint tensors from
``fixtures/exl3-moe/moe-fixtures.safetensors`` (experts 0 and 255 of layer 0,
fetched by ``scripts/fetch_exl3_fixtures.py``) and load them through the same
paths vLLM uses: the per-layer expert loader and ``AutoWeightsLoader``.

Covered contract (``docs/moe_api_audit.md`` §3.2):

* one allocation per tensor kind per layer, all below 4 GiB;
* no fp16/BF16 copy of any expert matrix;
* bit-exact per-expert copies into ``[expert, slot]``;
* fail-closed on incomplete loads, wrong shape/dtype/shard/name, and a
  codebook id that disagrees with the config;
* the stock 3-D "fused tensor" heuristic is not used.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
import types
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.models.utils import AutoWeightsLoader

from exl3xpu.moe import (
    CODEBOOK_IDS,
    CODEBOOK_MULTIPLIERS,
    DESCRIPTOR_FIELDS,
    MAX_ALLOCATION_BYTES,
    Exl3ExpertStore,
    Exl3MoEMethod,
    Exl3MoEUnsupported,
)
from exl3xpu.vllm_plugin import Exl3Config

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "exl3-moe" / "moe-fixtures.safetensors"
CKPT_PREFIX = "model.language_model."
LAYER_PREFIX = "layers.0.mlp.experts"
LAYER_NAME = "model.language_model.layers.0.mlp.experts"
NUM_EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 512
K = 3
CODEBOOK = "mcg"

EXPECTED_PARAMS = {
    "w13_trellis": ((NUM_EXPERTS, 2, 128, 32, 48), torch.int16),
    "w13_suh": ((NUM_EXPERTS, 2, HIDDEN), torch.float16),
    "w13_svh": ((NUM_EXPERTS, 2, INTERMEDIATE), torch.float16),
    "w2_trellis": ((NUM_EXPERTS, 1, 32, 128, 48), torch.int16),
    "w2_suh": ((NUM_EXPERTS, 1, INTERMEDIATE), torch.float16),
    "w2_svh": ((NUM_EXPERTS, 1, HIDDEN), torch.float16),
    "w13_mcg": ((NUM_EXPERTS, 2), torch.int32),
    "w2_mcg": ((NUM_EXPERTS, 1), torch.int32),
}

PROJECTIONS = (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2"))


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

_FIXTURES: dict[str, torch.Tensor] | None = None


def fixtures() -> dict[str, torch.Tensor]:
    global _FIXTURES
    if _FIXTURES is None:
        with safe_open(str(FIXTURES), framework="pt") as f:
            _FIXTURES = {k: f.get_tensor(k) for k in f.keys()}
    return _FIXTURES


def expert_slices(expert_id: int) -> dict[str, torch.Tensor]:
    """Layer-relative EXL3 tensors for one expert: ``<proj>.<suffix>``."""
    out = {}
    prefix = f"{CKPT_PREFIX}{LAYER_PREFIX}.{expert_id}."
    for name, tensor in fixtures().items():
        if name.startswith(prefix):
            out[name[len(prefix):]] = tensor
    assert len(out) == 12, (expert_id, sorted(out))
    return out


def expert_storage(num_experts: int = NUM_EXPERTS) -> dict[str, int]:
    return {
        f"{LAYER_PREFIX}.{e}.{proj}": K
        for e in range(num_experts)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }


def make_config(num_experts: int = NUM_EXPERTS) -> Exl3Config:
    cfg = Exl3Config(
        bits=3, head_bits=6, codebook=CODEBOOK, storage=expert_storage(num_experts)
    )
    cfg.member_trellis = {f"{LAYER_PREFIX}.0.gate_proj": (INTERMEDIATE, K)}
    return cfg


def make_layer() -> RoutedExperts:
    """Minimal but faithful ``RoutedExperts`` (the real ctor needs an engine)."""
    layer = object.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.layer_name = LAYER_NAME
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


def make_layer_with_weights() -> tuple[Exl3MoEMethod, RoutedExperts]:
    layer = make_layer()
    method = Exl3MoEMethod(make_config(), layer, LAYER_NAME)
    with torch.device("cpu"):
        method.create_weights(layer, NUM_EXPERTS, HIDDEN, INTERMEDIATE, torch.bfloat16)
    return method, layer


def synthetic_tensor(spec) -> torch.Tensor:
    tensor = torch.zeros(spec.shape[2:], dtype=spec.dtype)
    if spec.tensor == CODEBOOK:
        tensor.fill_(CODEBOOK_MULTIPLIERS[CODEBOOK])
    return tensor


def all_checkpoint_tensors(method: Exl3MoEMethod) -> list[tuple[str, torch.Tensor]]:
    """Synthetic full-layer stream (3072 tensors) for completeness tests."""
    by_name = {spec.name: synthetic_tensor(spec) for spec in method.store.specs}
    out: list[tuple[str, torch.Tensor]] = []
    for expert_id in range(NUM_EXPERTS):
        for proj, shard in PROJECTIONS:
            family = "w13" if shard in ("w1", "w3") else "w2"
            for spec in method.store.specs:
                if spec.name.startswith(family + "_"):
                    out.append(
                        (f"{expert_id}.{proj}.{spec.tensor}", by_name[spec.name])
                    )
    return out


def expect_unsupported(fn: Callable[[], object], contains: str, label: str) -> None:
    try:
        fn()
    except Exl3MoEUnsupported as exc:
        assert contains in str(exc), f"{label}: expected {contains!r} in {exc!s}"
    else:
        raise AssertionError(f"{label}: expected Exl3MoEUnsupported")


# ---------------------------------------------------------------------------
# shared state
# ---------------------------------------------------------------------------

_BASE: tuple[Exl3MoEMethod, RoutedExperts] | None = None


def base() -> tuple[Exl3MoEMethod, RoutedExperts]:
    global _BASE
    if _BASE is None:
        _BASE = make_layer_with_weights()
    return _BASE


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def case_plan_matches_checkpoint_schema() -> None:
    store = Exl3ExpertStore(NUM_EXPERTS, HIDDEN, INTERMEDIATE, K, CODEBOOK)
    assert {s.name for s in store.specs} == set(EXPECTED_PARAMS)
    fixture = expert_slices(0)
    for proj, shard in PROJECTIONS:
        family = "w13" if shard in ("w1", "w3") else "w2"
        for suffix in ("trellis", "suh", "svh", CODEBOOK):
            name = f"{family}_{suffix}"
            spec = store.get(name)
            expected = tuple(fixture[f"{proj}.{suffix}"].shape)
            assert spec.shape[2:] == expected, (name, spec.shape[2:], expected)
            assert spec.dtype == fixture[f"{proj}.{suffix}"].dtype, (name, spec.dtype)
            assert spec.slots == (2 if family == "w13" else 1), (name, spec.slots)


def case_allocations_within_budget() -> None:
    store = Exl3ExpertStore(NUM_EXPERTS, HIDDEN, INTERMEDIATE, K, CODEBOOK)
    assert len(store.specs) == 8
    for spec in store.specs:
        assert spec.nbytes < MAX_ALLOCATION_BYTES, (spec.name, spec.nbytes)
    assert store.total_bytes < MAX_ALLOCATION_BYTES, store.total_bytes
    # 3.09 bpw reality check: ~292 MiB per routed layer.
    assert 280 * 2**20 < store.total_bytes < 310 * 2**20, store.total_bytes


def case_registration_has_no_fp16_experts() -> None:
    method, layer = base()
    params = dict(layer.named_parameters())
    assert set(params) == set(EXPECTED_PARAMS), sorted(params)
    for name, (shape, dtype) in EXPECTED_PARAMS.items():
        p = params[name]
        assert tuple(p.shape) == shape, (name, tuple(p.shape), shape)
        assert p.dtype == dtype, (name, p.dtype, dtype)
        assert not p.requires_grad
    # No fp16 matrix of any expert projection exists (those are the ~64 GiB
    # unquantized layout this method exists to avoid).
    forbidden = {
        (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        (NUM_EXPERTS, HIDDEN, INTERMEDIATE),
    }
    for name, p in params.items():
        assert tuple(p.shape) not in forbidden, name
        if p.is_floating_point():
            assert p.numel() < NUM_EXPERTS * HIDDEN * INTERMEDIATE, name
    assert layer.exl3_moe_meta["prefix"] == LAYER_PREFIX
    assert layer.exl3_moe_meta["K"] == K
    assert layer.exl3_moe_meta["codebook"] == CODEBOOK
    assert layer.exl3_moe_meta["allocations"] == 8


def case_loader_override_bound() -> None:
    _, layer = base()
    assert isinstance(layer.load_weights, types.MethodType)
    assert layer.load_weights.__func__ is not RoutedExperts.load_weights
    assert layer.load_weights.__func__.__name__ == "_layer_load_weights"
    # Why the override is required: the stock loader keys the "fused tensor"
    # path off `loaded_weight.dim() == 3`, and a per-expert trellis is 3-D.
    assert expert_slices(0)["gate_proj.trellis"].dim() == 3


def case_real_experts_bit_exact() -> None:
    _, layer = base()
    stream = []
    for expert_id in (0, 255):
        slices = expert_slices(expert_id)
        stream.extend(
            (f"{expert_id}.{rel}", tensor) for rel, tensor in sorted(slices.items())
        )
    yielded = list(layer.load_weights(iter(stream)))
    assert len(yielded) == 24, len(yielded)  # 12 tensors x 2 experts
    for expert_id in (0, 255):
        slices = expert_slices(expert_id)
        for rel, tensor in slices.items():
            proj, suffix = rel.split(".")
            shard = dict(PROJECTIONS)[proj]
            family = "w13" if shard in ("w1", "w3") else "w2"
            slot = 0 if shard in ("w1", "w2") else 1
            got = getattr(layer, f"{family}_{suffix}").data[expert_id, slot]
            assert got.dtype == tensor.dtype, (rel, got.dtype, tensor.dtype)
            assert torch.equal(got, tensor), f"expert {expert_id} {rel} not bit-exact"


def case_autoweightsloader_end_to_end() -> None:
    _, layer = base()

    class Parent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = layer

    parent = Parent()
    slices = expert_slices(255)
    weights = [(f"experts.255.{rel}", t) for rel, t in sorted(slices.items())]
    loaded = AutoWeightsLoader(parent).load_weights(iter(weights))
    assert {"experts.w13_trellis", "experts.w2_mcg"} <= loaded, sorted(loaded)
    trellis = slices["gate_proj.trellis"]
    assert torch.equal(layer.w13_trellis.data[255, 0], trellis)
    assert torch.equal(layer.w2_svh.data[255, 0], slices["down_proj.svh"])


def case_descriptors_packed() -> None:
    method, layer = base()
    d = layer.exl3_moe_descriptors
    assert d.shape == (NUM_EXPERTS, len(PROJECTIONS), len(DESCRIPTOR_FIELDS)), d.shape
    assert d.dtype == torch.int64
    names = layer.exl3_moe_tensor_names
    assert names == [s.name for s in method.store.specs], names
    assert layer.exl3_moe_meta["descriptor_fields"] == list(DESCRIPTOR_FIELDS)
    for expert_id in (0, 255):
        for proj_index, (proj, shard) in enumerate(PROJECTIONS):
            row = method.descriptor(layer, expert_id, proj)
            family = "w13" if proj_index < 2 else "w2"
            slot = 0 if shard == "w1" else 1 if shard == "w3" else 0
            for kind in ("trellis", "suh", "svh", CODEBOOK):
                spec = method.store.get(f"{family}_{kind}")
                name = spec.name
                field = "codebook" if kind == CODEBOOK else kind
                per = math.prod(spec.shape[2:])
                assert row[f"{field}_tensor"] == names.index(name), (kind, row)
                assert row[f"{field}_offset"] == (expert_id * spec.slots + slot) * per, (
                    kind,
                    row,
                )
                param = getattr(layer, name)
                assert 0 <= row[f"{field}_offset"] <= param.numel()
                assert row[f"{field}_offset"] + per <= param.numel(), (kind, row)
            assert row["k"] == (HIDDEN if proj_index < 2 else INTERMEDIATE), row
            assert row["n"] == (INTERMEDIATE if proj_index < 2 else HIDDEN), row
            assert row["K"] == K and row["cb"] == CODEBOOK_IDS[CODEBOOK], row
            assert row["family"] == (0 if proj_index < 2 else 1), row
            assert row["slot"] == slot, row
    # gate and up share the w13 allocation but not the slot.
    gate = method.descriptor(layer, 0, "gate_proj")
    up = method.descriptor(layer, 0, "up_proj")
    assert gate["trellis_tensor"] == up["trellis_tensor"]
    assert (gate["slot"], up["slot"]) == (0, 1), (gate, up)


def case_descriptor_views_bit_exact() -> None:
    method, layer = base()
    slices = expert_slices(255)
    list(layer.load_weights(iter([(f"255.{rel}", t) for rel, t in sorted(slices.items())])))
    views = method.expert_views(layer, 255)
    for proj, shard in PROJECTIONS:
        got = views[proj]
        assert got["k"] == (HIDDEN if shard in ("w1", "w3") else INTERMEDIATE)
        assert got["n"] == (INTERMEDIATE if shard in ("w1", "w3") else HIDDEN)
        assert got["K"] == K and got["cb"] == CODEBOOK_IDS[CODEBOOK]
        for kind in ("trellis", "suh", "svh"):
            assert torch.equal(got[kind], slices[f"{proj}.{kind}"]), (proj, kind)
            assert got[kind].dtype == slices[f"{proj}.{kind}"].dtype
        assert got["codebook"].item() == CODEBOOK_MULTIPLIERS[CODEBOOK]


def loaded_layer() -> tuple[Exl3MoEMethod, RoutedExperts]:
    """A fresh layer with a complete synthetic load already settled."""
    method, layer = make_layer_with_weights()
    list(layer.load_weights(iter(all_checkpoint_tensors(method))))
    method.process_weights_after_loading(layer)
    return method, layer


def _descriptor_offsets(
    method: Exl3MoEMethod, layer: RoutedExperts, spec
) -> set[int]:
    """Descriptor offsets for one allocation, across all experts/projections."""
    field = "codebook" if spec.tensor == CODEBOOK else spec.tensor
    index = layer.exl3_moe_tensor_names.index(spec.name)
    out: set[int] = set()
    for expert_id in range(NUM_EXPERTS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            row = method.descriptor(layer, expert_id, projection)
            if row[f"{field}_tensor"] == index:
                out.add(row[f"{field}_offset"])
    return out


def case_integrity_arena_tiling() -> None:
    method, layer = loaded_layer()
    summary = method.verify_expert_integrity(layer)
    assert summary["experts"] == NUM_EXPERTS
    assert summary["projections"] == 3
    assert summary["allocations"] == 8
    assert summary["tiled_slices"] == 3072
    assert summary["trellis_tile_w13"] == [HIDDEN // 16, INTERMEDIATE // 16, 16 * K]
    assert summary["trellis_tile_w2"] == [INTERMEDIATE // 16, HIDDEN // 16, 16 * K]

    # Trellis is tile-major [k/16, n/16, 16K], never a dense [k, n] row store.
    w13 = tuple(layer.w13_trellis.shape[2:])
    assert w13 == (128, 32, 48), w13
    assert (w13[0] * 16, w13[1] * 16) == (HIDDEN, INTERMEDIATE)
    assert w13 != (HIDDEN, INTERMEDIATE)
    w2 = tuple(layer.w2_trellis.shape[2:])
    assert w2 == (32, 128, 48), w2
    assert (w2[0] * 16, w2[1] * 16) == (INTERMEDIATE, HIDDEN)

    # Every (expert, slot) slice owns distinct storage, so per-expert scales /
    # Hadamard vectors / codebooks are never aliased.
    for name in (
        "w13_trellis", "w13_suh", "w13_svh", "w13_mcg",
        "w2_trellis", "w2_suh", "w2_svh", "w2_mcg",
    ):
        param = getattr(layer, name)
        ptrs = {
            param[e, s].data_ptr()
            for e in range(NUM_EXPERTS)
            for s in range(param.shape[1])
        }
        assert len(ptrs) == NUM_EXPERTS * param.shape[1], (name, len(ptrs))
    # Gate and up keep independent input transforms (distinct suh rows).
    assert layer.w13_suh[0, 0].data_ptr() != layer.w13_suh[0, 1].data_ptr()

    # Descriptor offsets tile each arena exactly: a permutation of
    # {0, per, 2*per, ...} ending at the allocation size.
    for spec in method.store.specs:
        per = math.prod(spec.shape[2:])
        param = getattr(layer, spec.name)
        assert param.numel() == NUM_EXPERTS * spec.slots * per
        expected = {i * per for i in range(NUM_EXPERTS * spec.slots)}
        assert _descriptor_offsets(method, layer, spec) == expected, spec.name


def case_integrity_detects_corruption() -> None:
    method, layer = loaded_layer()
    d = layer.exl3_moe_descriptors

    def restore(patch: tuple[int, int, str, int]) -> None:
        e, p, field, value = patch
        d[e, p, DESCRIPTOR_FIELDS.index(field)] = value

    # A wrong stride/offset must be rejected (codes are never read as raw
    # dense rows).
    f = DESCRIPTOR_FIELDS.index("suh_offset")
    saved = int(d[7, 1, f].item())
    d[7, 1, f] = saved + 1
    expect_unsupported(
        lambda: method.verify_expert_integrity(layer),
        "strided offset",
        "offset corruption",
    )
    restore((7, 1, "suh_offset", saved))

    # A wrong per-projection k must be rejected.
    f = DESCRIPTOR_FIELDS.index("k")
    saved = int(d[3, 0, f].item())
    d[3, 0, f] = saved + 16
    expect_unsupported(
        lambda: method.verify_expert_integrity(layer), "k/n", "dim corruption"
    )
    restore((3, 0, "k", saved))

    # A tensor-index that points at the wrong allocation must be rejected.
    f = DESCRIPTOR_FIELDS.index("trellis_tensor")
    saved = int(d[5, 2, f].item())
    d[5, 2, f] = 1 if saved != 1 else 2
    expect_unsupported(
        lambda: method.verify_expert_integrity(layer),
        "trellis_tensor",
        "allocation-index corruption",
    )
    restore((5, 2, "trellis_tensor", saved))

    # process_weights_after_loading must run the check too.
    f = DESCRIPTOR_FIELDS.index("svh_offset")
    saved = int(d[9, 0, f].item())
    d[9, 0, f] = saved + 1
    expect_unsupported(
        lambda: method.process_weights_after_loading(layer),
        "strided offset",
        "integrity check in settle path",
    )
    restore((9, 0, "svh_offset", saved))
    method.verify_expert_integrity(layer)


def case_integrity_rejects_dense_layout() -> None:
    method, layer = make_layer_with_weights()
    try:
        # A dense [E, 2, k, n] row store is exactly the packing P1-04 forbids.
        dense = nn.Parameter(
            torch.empty(NUM_EXPERTS, 2, HIDDEN, INTERMEDIATE, dtype=torch.int16),
            requires_grad=False,
        )
        layer.register_parameter("w13_trellis", dense)
        expect_unsupported(
            lambda: method.verify_expert_integrity(layer),
            "expected tiled",
            "dense-row trellis",
        )
    finally:
        del method, layer


def case_incomplete_load_fails() -> None:
    method, layer = make_layer_with_weights()
    try:
        slices = expert_slices(0)
        list(
            layer.load_weights(
                iter([(f"0.{rel}", t) for rel, t in sorted(slices.items())])
            )
        )
        total_slices = sum(
            s.slots * NUM_EXPERTS for s in method.store.specs
        )
        expect_unsupported(
            lambda: method.process_weights_after_loading(layer),
            f"{total_slices - 12} expert tensor slice(s) were not loaded",
            "incomplete load",
        )
    finally:
        del method, layer


def case_malformed_tensors_fail() -> None:
    method, layer = make_layer_with_weights()
    try:
        good = expert_slices(0)
        trellis = good["gate_proj.trellis"]

        expect_unsupported(
            lambda: list(layer.load_weights(iter([("0.gate_proj.trellis", trellis[1:])]))),
            "has shape",
            "bad shape",
        )
        expect_unsupported(
            lambda: list(
                layer.load_weights(iter([("0.gate_proj.trellis", trellis.int())]))
            ),
            "has dtype",
            "bad dtype",
        )
        expect_unsupported(
            lambda: list(layer.load_weights(iter([("0.gate_proj.scale", trellis)]))),
            "unexpected tensor suffix",
            "unknown suffix",
        )
        expect_unsupported(
            lambda: list(layer.load_weights(iter([("0.gate_proj2.trellis", trellis)]))),
            "unexpected projection",
            "unknown projection",
        )
        expect_unsupported(
            lambda: list(layer.load_weights(iter([("256.gate_proj.trellis", trellis)]))),
            "out of range",
            "expert out of range",
        )
        expect_unsupported(
            lambda: list(layer.load_weights(iter([("gate_proj.trellis", trellis)]))),
            "unexpected routed-expert tensor",
            "unqualified name",
        )
        # Direct loader misuse: w2 param fed a w1 shard.
        expect_unsupported(
            lambda: layer.w2_suh.weight_loader(
                param=layer.w2_suh,
                loaded_weight=good["up_proj.suh"],
                weight_name="0.up_proj.suh",
                shard_id="w1",
                expert_id=0,
                return_success=True,
            ),
            "does not match parameter",
            "shard mismatch",
        )
    finally:
        del method, layer


def case_complete_load_passes_and_codebook_checked() -> None:
    method, layer = base()
    list(layer.load_weights(iter(all_checkpoint_tensors(method))))
    method.process_weights_after_loading(layer)

    # A codebook multiplier that disagrees with the config must abort start-up.
    try:
        layer.w13_mcg.data.fill_(CODEBOOK_MULTIPLIERS[CODEBOOK] + 1)
        expect_unsupported(
            lambda: method.process_weights_after_loading(layer),
            "checkpoint and config disagree",
            "codebook mismatch",
        )
    finally:
        layer.w13_mcg.data.fill_(CODEBOOK_MULTIPLIERS[CODEBOOK])


def case_dummy_load_path_skips_completeness() -> None:
    method, layer = make_layer_with_weights()
    try:
        assert method._load_attempted is False
        # --load-format dummy never calls load_weights; skipping is explicit,
        # not an accidental "empty set is complete".
        method.process_weights_after_loading(layer)
    finally:
        del method, layer


CASES: list[tuple[str, Callable[[], None]]] = [
    ("plan shapes/dtypes match real checkpoint", case_plan_matches_checkpoint_schema),
    ("allocations below 4 GiB / layer", case_allocations_within_budget),
    ("registration has no fp16 expert matrices", case_registration_has_no_fp16_experts),
    ("custom expert loader is bound", case_loader_override_bound),
    ("real experts 0/255 load bit-exact", case_real_experts_bit_exact),
    ("AutoWeightsLoader end-to-end", case_autoweightsloader_end_to_end),
    ("packed expert descriptors", case_descriptors_packed),
    ("descriptor views bit-exact", case_descriptor_views_bit_exact),
    ("arena tiling + metadata integrity", case_integrity_arena_tiling),
    ("integrity detects corruption", case_integrity_detects_corruption),
    ("integrity rejects dense-row layout", case_integrity_rejects_dense_layout),
    ("incomplete load fails closed", case_incomplete_load_fails),
    ("malformed tensors fail closed", case_malformed_tensors_fail),
    ("complete load passes; codebook checked", case_complete_load_passes_and_codebook_checked),
    ("dummy-load path skips completeness", case_dummy_load_path_skips_completeness),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    failures: list[str] = []
    results: list[dict] = []
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:
            failures.append(name)
            results.append(
                {"case": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"FAIL {name}")
            if args.verbose:
                traceback.print_exc()
        else:
            results.append({"case": name, "ok": True})
            print(f"OK   {name}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} expert-loader cases pass")

    if args.json_out:
        import hashlib

        def sha(path: str | Path) -> str:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        import exl3xpu.moe as moe_mod

        method, layer = base()
        summary = {
            "task": "P1-02..P1-04",
            "pass": not failures,
            "cases": results,
            "failures": failures,
            "store": layer.exl3_moe_meta,
            "descriptors": {
                "shape": list(layer.exl3_moe_descriptors.shape),
                "dtype": str(layer.exl3_moe_descriptors.dtype).removeprefix("torch."),
                "fields": list(DESCRIPTOR_FIELDS),
                "tensor_order": layer.exl3_moe_tensor_names,
            },
            "integrity": method.verify_expert_integrity(layer),
            "per_layer_total_mib": layer.exl3_moe_meta["total_bytes"] / 2**20,
            "max_allocation_bytes": MAX_ALLOCATION_BYTES,
            "fixtures": {
                "file": str(FIXTURES.relative_to(ROOT)),
                "sha256": sha(FIXTURES),
                "experts_checked": [0, 255],
            },
            "device": torch.xpu.get_device_properties(0).name,
            "torch": torch.__version__,
            "moe_module": moe_mod.__file__,
            "moe_module_sha256": sha(moe_mod.__file__),
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
