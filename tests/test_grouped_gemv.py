"""P4-02 parity: grouped M=1 decode GEMV vs the eager per-expert path.

Gate 4 / P4-02. ``exl3_moe_grouped_gemv`` (csrc/exl3_moe_gemv.sycl) folds every
active decode expert into ONE launch per projection:

* gate+up : nslots=2, x_unit_stride=0  (all units share the token row)
* down    : nslots=1, x_unit_stride=Kdim (each unit consumes its own SiLU act)

The kernel replicates the eager micro-sequence (input Hadamard -> split-K GEMV
-> output Hadamard + SVH) with a row-granular part layout and the exact
HadOutKernel reduction chain, so the contract is BIT-FOR-BIT equality with the
P4-01 ``_expert_mlp_cached`` outputs — asserted for both fp16 (harness) and
bf16 (the real model's activation dtype) at the kernel and ``apply()`` level.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from exl3xpu import ops

from _moe_harness import HIDDEN, INTERMEDIATE, Harness

K = 3        # routed-expert trellis K (routed tensors are codebook 1)
CB = 1
IDS = [3, 7, 42, 100, 255, 0, 17, 200]   # deterministic top-8 stand-ins
NSLOTS_13 = 2
NSLOTS_2 = 1


def _shard(method, device, n: int) -> torch.Tensor:
    return method._shard_of_nb(device, n)


def _parity_for_dtype(dtype: torch.dtype) -> None:
    """Full P4-02 parity battery at one activation dtype (fp16 harness / bf16 model)."""
    h = Harness()
    h.method._build_exec_cache(h.layer)
    dev = torch.device("xpu")
    layer = h.layer
    shard13 = h.method._shard_of_nb(dev, INTERMEDIATE)
    shard2 = h.method._shard_of_nb(dev, HIDDEN)

    # gate+up: shared token row, nslots=2, x_unit_stride=0
    torch.manual_seed(1234)
    x = torch.randn(1, HIDDEN, dtype=dtype, device=dev)
    ids13 = torch.tensor([e for e in IDS for _ in range(2)], dtype=torch.int32, device=dev)
    gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
        x, layer.w13_trellis, layer.w13_suh, layer.w13_svh, ids13, shard13,
        2, INTERMEDIATE, K, CB, 0,
    )
    assert list(gu.shape) == [len(IDS) * 2, 1, INTERMEDIATE], list(gu.shape)
    worst = 0.0
    for u, e in enumerate(IDS):
        for slot, proj in ((0, "gate"), (1, "up")):
            ref = h.method._cached_linear(proj, e, x)
            got = gu[u * NSLOTS_13 + slot]
            assert got.dtype == dtype, f"{got.dtype} != {dtype}"
            assert torch.equal(got, ref), (
                f"{proj} expert {e} ({dtype}): bit mismatch "
                f"max_abs={(got.float() - ref.float()).abs().max().item():.3e}"
            )
            worst = max(worst, (got.float() - ref.float()).abs().max().item())
    print(f"grouped gate+up [{dtype}]: {len(IDS)*2} units bit-exact "
          f"(worst max_abs={worst:.3e})")

    # down: per-expert activations, nslots=1, x_unit_stride=Kdim
    acts = torch.cat(
        [
            F.silu(h.method._cached_linear("gate", e, x))
            * h.method._cached_linear("up", e, x)
            for e in IDS
        ],
        dim=0,
    ).contiguous()
    assert acts.dtype == dtype
    dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
        acts, layer.w2_trellis, layer.w2_suh, layer.w2_svh,
        torch.tensor(IDS, dtype=torch.int32, device=dev), shard2,
        NSLOTS_2, HIDDEN, K, CB, INTERMEDIATE,
    )
    assert list(dn.shape) == [len(IDS), 1, HIDDEN], list(dn.shape)
    worst = 0.0
    for u, e in enumerate(IDS):
        ref = h.method._cached_linear("down", e, acts[u : u + 1])
        got = dn[u]
        assert got.dtype == dtype
        assert torch.equal(got, ref), (
            f"down expert {e} ({dtype}): bit mismatch "
            f"max_abs={(got.float() - ref.float()).abs().max().item():.3e}"
        )
        worst = max(worst, (got.float() - ref.float()).abs().max().item())
    print(f"grouped down [{dtype}]: {len(IDS)} units bit-exact "
          f"(worst max_abs={worst:.3e})")

    # end-to-end grouped expert MLP vs the P4-01 cached eager MLP
    ids13 = torch.tensor([e for e in IDS for _ in range(2)], dtype=torch.int32, device=dev)
    gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
        x, layer.w13_trellis, layer.w13_suh, layer.w13_svh, ids13, shard13,
        2, INTERMEDIATE, K, CB, 0,
    )
    acts2 = torch.cat(
        [F.silu(gu[2 * u]) * gu[2 * u + 1] for u in range(len(IDS))],
        dim=0,
    ).contiguous()
    dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
        acts2, layer.w2_trellis, layer.w2_suh, layer.w2_svh,
        torch.tensor(IDS, dtype=torch.int32, device=dev), shard2,
        NSLOTS_2, HIDDEN, K, CB, INTERMEDIATE,
    )
    worst = 0.0
    for u, e in enumerate(IDS):
        ref = h.method._expert_mlp_cached(e, x)
        got = dn[u]
        assert torch.equal(got, ref), (
            f"mlp expert {e} ({dtype}): bit mismatch "
            f"max_abs={(got.float() - ref.float()).abs().max().item():.3e}"
        )
        worst = max(worst, (got.float() - ref.float()).abs().max().item())
    print(f"grouped mlp (gate+up+down) [{dtype}]: {len(IDS)} experts bit-exact "
          f"(worst max_abs={worst:.3e})")

    # apply() end-to-end parity: grouped path vs the eager per-expert loop
    _, ids, wts = h.router(x.unsqueeze(0))
    ids = ids.view(1, -1)
    wts = wts.view(1, -1)
    X = x.view(1, 1, HIDDEN)
    h.method._grouped_supported = False   # force eager reference
    ref = h.method.apply(layer=layer, x=X, topk_weights=wts, topk_ids=ids,
                         shared_experts=None, shared_experts_input=None)
    del h.method._grouped_supported        # re-probe the grouped path
    got = h.method.apply(layer=layer, x=X, topk_weights=wts, topk_ids=ids,
                         shared_experts=None, shared_experts_input=None)
    assert h.method._grouped_supported, "grouped path not engaged"
    assert got.dtype == ref.dtype == dtype
    assert torch.equal(got, ref), (
        f"apply() ({dtype}): bit mismatch "
        f"max_abs={(got.float() - ref.float()).abs().max().item():.3e}"
    )
    print(f"apply() grouped vs eager [{dtype}]: bit-exact, dtype={dtype}")


def test_grouped_fp16_parity():
    """Harness-dtype (fp16) battery: gate+up / down / mlp / apply()."""
    _parity_for_dtype(torch.float16)


def test_grouped_bf16_parity():
    """Model-dtype (bf16) battery — the real serving activation type."""
    _parity_for_dtype(torch.bfloat16)


def test_grouped_rejects_bad_shapes():
    h = Harness()
    dev = torch.device("xpu")
    x = torch.randn(3, HIDDEN, dtype=torch.float16, device=dev)   # rows != 1
    layer = h.layer
    ids = torch.tensor([0, 0], dtype=torch.int32, device=dev)
    try:
        torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
            x, layer.w13_trellis, layer.w13_suh, layer.w13_svh, ids,
            _shard(h.method, dev, INTERMEDIATE), NSLOTS_13,
            INTERMEDIATE, K, CB, 0,
        )
    except RuntimeError as exc:
        assert "shared-x mode needs x rows==1" in str(exc), exc
        print(f"grouped shape validation OK: {str(exc).splitlines()[0][:80]}")
    else:
        raise AssertionError("expected RuntimeError for shared-x with rows!=1")


if __name__ == "__main__":
    test_grouped_fp16_parity()
    test_grouped_bf16_parity()
    test_grouped_rejects_bad_shapes()
    print("P4-02 grouped GEMV parity: PASS (3/3)")
