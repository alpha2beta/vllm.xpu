"""P4-03 parity: grouped small-M prefill GEMM vs eager per-expert path.

Tests exl3_moe_grouped_gemm across variable token counts M_u in [1, 8]
for all active experts in a single kernel launch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from exl3xpu import ops
from _moe_harness import HIDDEN, INTERMEDIATE, Harness

K = 3
CB = 1
EXPERTS = [3, 7, 42, 100, 255, 0, 17, 200]
# Variable token counts M_u across the 8 experts: 1, 2, 3, 4, 5, 6, 7, 8
COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]


def test_grouped_gemm_parity(dtype: torch.dtype = torch.bfloat16):
    print(f"\n--- Testing Grouped Prefill GEMM Parity [{dtype}] ---")
    h = Harness()
    h.method._build_exec_cache(h.layer)
    dev = torch.device("xpu")
    layer = h.layer
    shard13 = h.method._shard_of_nb(dev, INTERMEDIATE)
    shard2 = h.method._shard_of_nb(dev, HIDDEN)

    total_tokens = sum(COUNTS)
    U = len(EXPERTS)

    # Prepare device arrays
    expert_ids = torch.tensor(EXPERTS, dtype=torch.int32, device=dev)
    counts = torch.tensor(COUNTS, dtype=torch.int32, device=dev)
    offsets = torch.zeros(U, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(counts[:-1], dim=0)

    # Random input tokens
    torch.manual_seed(42)
    x = torch.randn(total_tokens, HIDDEN, dtype=dtype, device=dev)

    # 1. Gate + Up projection
    # nslots = 2, units = 2 * U
    ids13 = expert_ids.repeat_interleave(2)
    m13 = counts.repeat_interleave(2)
    x_off13 = offsets.repeat_interleave(2)
    out_off13 = offsets.repeat_interleave(2)

    gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemm(
        x,
        layer.w13_trellis,
        layer.w13_suh,
        layer.w13_svh,
        ids13,
        shard13,
        m13,
        x_off13,
        out_off13,
        total_tokens,
        2,
        INTERMEDIATE,
        K,
        CB,
    )
    assert gu.shape == (2, total_tokens, INTERMEDIATE), f"Shape mismatch: {gu.shape}"

    # Verify gate and up against reference eager per-expert linear
    worst_diff = 0.0
    for i, exp in enumerate(EXPERTS):
        m = COUNTS[i]
        off = offsets[i].item()
        x_slice = x[off : off + m]

        ref_gate = h.method._cached_linear("gate", exp, x_slice)
        ref_up = h.method._cached_linear("up", exp, x_slice)

        got_gate = gu[0, off : off + m]
        got_up = gu[1, off : off + m]

        diff_gate = (got_gate.float() - ref_gate.float()).abs().max().item()
        diff_up = (got_up.float() - ref_up.float()).abs().max().item()

        assert torch.equal(got_gate, ref_gate), f"Gate mismatch for expert {exp}: max_abs={diff_gate:.3e}"
        assert torch.equal(got_up, ref_up), f"Up mismatch for expert {exp}: max_abs={diff_up:.3e}"
        worst_diff = max(worst_diff, diff_gate, diff_up)

    print(f"Grouped gate+up [{dtype}]: {U*2} units bit-exact across variable M in [1, 8] (worst_abs={worst_diff:.3e})")

    # 2. Gated activation
    act = (F.silu(gu[0]) * gu[1]).contiguous()

    # 3. Down projection
    dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemm(
        act,
        layer.w2_trellis,
        layer.w2_suh,
        layer.w2_svh,
        expert_ids,
        shard2,
        counts,
        offsets,
        offsets,
        total_tokens,
        1,
        HIDDEN,
        K,
        CB,
    )
    assert dn.shape == (total_tokens, HIDDEN), f"Shape mismatch: {dn.shape}"

    worst_diff_down = 0.0
    for i, exp in enumerate(EXPERTS):
        m = COUNTS[i]
        off = offsets[i].item()
        act_slice = act[off : off + m]

        ref_down = h.method._cached_linear("down", exp, act_slice)
        got_down = dn[off : off + m]

        diff_down = (got_down.float() - ref_down.float()).abs().max().item()
        assert torch.equal(got_down, ref_down), f"Down mismatch for expert {exp}: max_abs={diff_down:.3e}"
        worst_diff_down = max(worst_diff_down, diff_down)

    print(f"Grouped down [{dtype}]: {U} units bit-exact across variable M in [1, 8] (worst_abs={worst_diff_down:.3e})")
    print(f"P4-03 grouped prefill GEMM parity [{dtype}]: PASS")


def test_apply_prefill_parity(dtype: torch.dtype = torch.bfloat16):
    print(f"\n--- Testing Exl3MoEMethod.apply Prefill Parity [{dtype}] ---")
    h = Harness()
    h.method._build_exec_cache(h.layer)
    dev = torch.device("xpu")

    for T in [2, 4, 16, 64, 128]:
        torch.manual_seed(100 + T)
        x = torch.randn(T, HIDDEN, dtype=dtype, device=dev)
        logits = torch.randn(T, 256, device=dev)
        weights, ids = torch.topk(logits.softmax(dim=-1), 8, dim=-1)

        # 1. Grouped prefill path
        h.method._grouped_gemm_supported = True
        out_grouped = h.method.apply(h.layer, x, weights, ids, None, None)

        # 2. Eager fallback path
        h.method._grouped_gemm_supported = False
        out_eager = h.method.apply(h.layer, x, weights, ids, None, None)

        # Re-enable
        h.method._grouped_gemm_supported = True

        diff = (out_grouped.float() - out_eager.float()).abs().max().item()
        print(f"T={T:3d} tokens: diff={diff:.3e} (grouped vs eager)")
        assert diff <= 1e-4, f"T={T} prefill diverged: max_abs={diff:.3e}"

    print(f"Exl3MoEMethod.apply prefill parity [{dtype}]: PASS")


if __name__ == "__main__":
    test_grouped_gemm_parity(torch.float16)
    test_grouped_gemm_parity(torch.bfloat16)
    test_apply_prefill_parity(torch.float16)
    test_apply_prefill_parity(torch.bfloat16)
    print("\nALL PREFILL GROUPED GEMM PARITY TESTS PASS!")
