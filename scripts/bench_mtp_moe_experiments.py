#!/usr/bin/env python3
"""Benchmark MTP MoE: §11.4 sync experiment and §11.5 prototype on Layer 0."""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
from _moe_harness import Harness, HIDDEN

def main():
    print("=" * 78)
    print("MTP MoE Benchmarking: §11.4 Sync Fix vs §11.5 Token-Parallel Prototype")
    print("Model: Sage-A1.5-EXL3-4.0bpw-mtp (Layer 0)")
    print("=" * 78)

    h = Harness("/home/yanchun/models/Sage-A1.5-EXL3-4.0bpw-mtp", layer_id=0)
    h.method._build_exec_cache(h.layer)
    dev = torch.device("xpu")
    dtype = torch.bfloat16

    T, k = 2, 8
    torch.manual_seed(42)
    x2 = torch.randn(T, HIDDEN, dtype=dtype, device=dev)
    logits = torch.randn(T, 256, device=dev)
    weights, ids = torch.topk(logits.softmax(dim=-1), k, dim=-1)

    # Distinct expert count analysis
    flat_ids = ids.reshape(-1)
    unique_exp = torch.unique(flat_ids)
    num_unique = unique_exp.numel()
    print(f"\n[Routing Analysis] Top-8 across T=2 tokens:")
    print(f"Total token-expert units : {T * k}")
    print(f"Unique active experts    : {num_unique} / 16 (Collision rate: {(16 - num_unique)/16*100:.1f}%)")
    print(f"Token 0 experts          : {ids[0].tolist()}")
    print(f"Token 1 experts          : {ids[1].tolist()}")
    print(f"Overlapping experts      : {set(ids[0].tolist()) & set(ids[1].tolist())}")

    warmup = 15
    iters = 50

    # ---------------------------------------------------------
    # 1. Baseline: T=1 decode path (_grouped_mlp_decode)
    # ---------------------------------------------------------
    x1 = x2[0:1]
    ids1 = ids[0:1]
    w1 = weights[0:1]
    for _ in range(warmup):
        h.method._grouped_mlp_decode(h.layer, x1, ids1, w1)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        h.method._grouped_mlp_decode(h.layer, x1, ids1, w1)
    torch.xpu.synchronize()
    t_t1 = (time.perf_counter() - t0) * 1000.0 / iters

    # ---------------------------------------------------------
    # 2. Baseline: T=2 using existing _grouped_mlp_prefill (before §11.4)
    # ---------------------------------------------------------
    # Ensure standard _grouped_mlp_prefill is called
    for _ in range(warmup):
        h.method._grouped_mlp_prefill(h.layer, x2, ids, weights)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        h.method._grouped_mlp_prefill(h.layer, x2, ids, weights)
    torch.xpu.synchronize()
    t_prefill_orig = (time.perf_counter() - t0) * 1000.0 / iters

    # ---------------------------------------------------------
    # 3. §11.4: T=2 using _grouped_mlp_prefill with sync fix
    # (hardcoded max_m_target = 8 for T <= 8, bypassing .item() / .any())
    # ---------------------------------------------------------
    def prefill_sync_fixed(method, layer, x2, topk_ids, weights):
        T, k = topk_ids.shape
        dev = x2.device
        E = method.num_local_experts
        flat_ids = topk_ids.reshape(-1)
        if flat_ids.dtype != torch.int32:
            flat_ids = flat_ids.to(torch.int32)
        flat_weights = weights.reshape(-1)
        flat_token_ids = (
            torch.arange(T, device=dev, dtype=torch.int32)
            .unsqueeze(1)
            .expand(T, k)
            .reshape(-1)
        )
        sort_order = torch.argsort(flat_ids)
        sorted_expert_ids = flat_ids[sort_order]
        sorted_token_ids = flat_token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]
        x_sorted = x2.index_select(0, sorted_token_ids)
        active_experts, counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True
        )
        active_experts = active_experts.to(torch.int32)
        counts = counts.to(torch.int32)
        U = active_experts.shape[0]
        offsets = torch.zeros(U, dtype=torch.int32, device=dev)
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        total_tokens = x_sorted.shape[0]

        inter = method.intermediate_size_per_partition
        shard13 = method._shard_of_nb(dev, inter)
        shard2 = method._shard_of_nb(dev, method.hidden_size)

        # §11.4 sync fix: when T <= 8, counts <= 8 always, max_m_target = 8 without host sync
        max_m_target = 8

        ids13 = active_experts.repeat_interleave(2)
        m13 = counts.repeat_interleave(2)
        off13 = offsets.repeat_interleave(2)

        gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemm(
            x_sorted,
            layer.w13_trellis,
            layer.w13_suh,
            layer.w13_svh,
            ids13,
            shard13,
            m13,
            off13,
            off13,
            total_tokens,
            2,
            inter,
            method.K,
            method.cb,
            max_m_target,
        )
        act = (F.silu(gu[0]) * gu[1]).contiguous()
        dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemm(
            act,
            layer.w2_trellis,
            layer.w2_suh,
            layer.w2_svh,
            active_experts,
            shard2,
            counts,
            offsets,
            offsets,
            total_tokens,
            1,
            method.hidden_size,
            method.K,
            method.cb,
            max_m_target,
        )
        out = torch.zeros((T, method.hidden_size), dtype=torch.float32, device=dev)
        out.index_add_(
            0,
            sorted_token_ids,
            dn.to(torch.float32) * sorted_weights.unsqueeze(-1),
        )
        return out.to(x2.dtype)

    for _ in range(warmup):
        prefill_sync_fixed(h.method, h.layer, x2, ids, weights)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        prefill_sync_fixed(h.method, h.layer, x2, ids, weights)
    torch.xpu.synchronize()
    t_prefill_sync_fixed = (time.perf_counter() - t0) * 1000.0 / iters

    # ---------------------------------------------------------
    # 4. §11.5: T=2 using token-parallel prototype
    # ---------------------------------------------------------
    def prototype_verify(method, layer, x2, topk_ids, weights):
        T, k = topk_ids.shape
        dev = x2.device
        inter = method.intermediate_size_per_partition
        ids = topk_ids.view(-1).to(torch.int32)
        x_rep = x2.repeat_interleave(2 * k, dim=0)
        gu = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
            x_rep,
            layer.w13_trellis,
            layer.w13_suh,
            layer.w13_svh,
            ids.repeat_interleave(2),
            method._shard_of_nb(dev, inter),
            2,
            inter,
            method.K,
            method.cb,
            method.hidden_size,
        )
        acts = (F.silu(gu[0::2]) * gu[1::2]).reshape(-1, inter).contiguous()
        dn = torch.ops.exl3xpu_C.exl3_moe_grouped_gemv(
            acts,
            layer.w2_trellis,
            layer.w2_suh,
            layer.w2_svh,
            ids,
            method._shard_of_nb(dev, method.hidden_size),
            1,
            method.hidden_size,
            method.K,
            method.cb,
            inter,
        )
        w = weights.view(-1).to(torch.float32).unsqueeze(1)
        y = (dn.squeeze(1).to(torch.float32) * w).view(T, k, -1).sum(dim=1)
        return y.to(x2.dtype)

    for _ in range(warmup):
        prototype_verify(h.method, h.layer, x2, ids, weights)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        prototype_verify(h.method, h.layer, x2, ids, weights)
    torch.xpu.synchronize()
    t_proto = (time.perf_counter() - t0) * 1000.0 / iters

    # ---------------------------------------------------------
    # Results Presentation
    # ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS (Layer 0 forward pass latency across 50 iterations)")
    print("=" * 78)
    print(f"{'Configuration':<45} | {'Latency (ms)':>12} | {'Scaling s (vs T=1)':>18}")
    print("-" * 78)
    print(f"{'T=1 Decode Baseline (_grouped_mlp_decode)':<45} | {t_t1:12.3f} | {'1.00x':>18}")
    print(f"{'T=2 Prefill Orig (with host syncs)':<45} | {t_prefill_orig:12.3f} | {t_prefill_orig / t_t1:17.2f}x")
    print(f"{'T=2 Prefill §11.4 (syncs removed)':<45} | {t_prefill_sync_fixed:12.3f} | {t_prefill_sync_fixed / t_t1:17.2f}x")
    print(f"{'T=2 §11.5 Token-Parallel Prototype':<45} | {t_proto:12.3f} | {t_proto / t_t1:17.2f}x")
    print("-" * 78)

    print("\n[Analysis & Diagnosis]")
    delta_sync = t_prefill_orig - t_prefill_sync_fixed
    speedup_proto = t_prefill_orig / t_proto
    print(f"* Host sync penalty in prefill path   : {delta_sync:.3f} ms / layer ({delta_sync * 40:.1f} ms / 40 layers)")
    print(f"* §11.5 Prototype vs Orig Prefill     : {speedup_proto:.2f}x faster ({t_proto:.3f} ms vs {t_prefill_orig:.3f} ms)")
    print(f"* Empirical scaling factor s (G1 gate): s = {t_proto / t_t1:.2f}")

    if (t_proto / t_t1) <= 1.5:
        print(f"--> VERDICT: G1 GATE PASSED! s = {t_proto / t_t1:.2f} <= 1.5.")
        print(f"    Estimated 40-layer MoE time: {t_proto * 40:.1f} ms (vs {t_t1 * 40:.1f} ms for T=1).")
    elif (t_proto / t_t1) <= 1.6:
        print(f"--> VERDICT: G1 GATE MARGINAL: s = {t_proto / t_t1:.2f} <= 1.6.")
    else:
        print(f"--> VERDICT: G1 GATE FAILED: s = {t_proto / t_t1:.2f} > 1.6.")

if __name__ == "__main__":
    main()
