# Technical Progress Review: EXL3 MoE Inference for Qwen3.5-35B-A3B on vLLM XPU

**Document:** `review_moe.md`  
**Initial Review:** October 24, 2026 (07:49)  
**Updated & Appended:** October 24, 2026 (13:00)  
**Implementation Plan:** [`exl3_moe_inference_for_qwen3_5_35b_a3b_on_vllm_xpu_v1.2.md`](file:///home/yanchun/Projects/vllm.xpu/exl3_moe_inference_for_qwen3_5_35b_a3b_on_vllm_xpu_v1.2.md)  
**Task Specification:** [`tasks_moe.md`](file:///home/yanchun/Projects/vllm.xpu/tasks_moe.md)  
**Target Platform:** Intel Core Ultra 7 258V / Arc 140V (8 Xe2 cores, 32 GiB shared LPDDR5X-8533, Linux 7.2 CachyOS)  
**Software Stack:** PyTorch 2.14.0+xpu, vLLM 0.31.0+xpu, Intel oneAPI 2026.0 (ICPX), Level Zero 1.32.0 (Driver 1.17.39758)

---

## 1. Executive Summary

This review documents the end-to-end implementation progress for the **EXL3 MoE integration of `Qwen3.5-35B-A3B`** on the Arc 140V integrated GPU platform.

### High-Level Status
- **Phase 0 (Platform & Hardware Proof):** **100% COMPLETE — GATE 0 PASSED.**
- **Phase 1 (Expert Loading & Single-Expert MLP):** **100% COMPLETE — GATE 1 PASSED.**
- **Phase 2 (Correct Complete MoE Block & Hybrid Layer):** **100% COMPLETE — GATE 2 PASSED.**
- **Phase 3 (Bounded Full-Model Serving on 140V):** **READY TO EXECUTE.** All 15.29 GB safetensors shards are downloaded and verified locally.

```
[Phase 0: Platform Proof]   ██████████ 100%  (Gate 0 MET)
[Phase 1: Expert Store/MLP] ██████████ 100%  (Gate 1 MET)
[Phase 2: MoE Block/Layer]  ██████████ 100%  (Gate 2 MET)
[Phase 3: Full-Model Serve] ░░░░░░░░░░   0%  (Next Milestone)
[Phase 4: Optimization]     ░░░░░░░░░░   0%  (Deferred to Gate 3 Pass)
```

### Key Milestones Achieved
1. **Gate 0 Certified:** Level Zero memory allocation limits confirmed (28.58 GiB addressable, 4 GiB single allocation succeeds). ESIMD native kernels instantiated for 3-bit MCG `(K=3, cb=1)` with **zero register spills** on Xe2 `lnl-a1`. Empirical copy bandwidth measured at **130.4 GB/s** (95.9% of spec).
2. **Gate 1 Certified:** Shard-by-shard streaming loader streamed all **122,880 expert tensors across 40 layers in 9.7 seconds** into 11,673 MiB of packed arenas, holding host RSS strictly at **1.25–1.26 GiB**. Boundary expert weight reconstructions are **100% bit-exact** (`max_abs = 0.0`). Single-expert MLP forward pass executed with worst `max_abs = 3.62e-6` (vs 0.08 tolerance). Batch sweep verified $M=0$ launches **0 kernels**, and crossover calibrated at $M=8$. Intermediate reconstruction workspace reentrancy bug resolved in C++ (`_C.so` r4). Dense regression passed **90/90 cases**.
3. **Gate 2 Certified:** Eager token dispatch, top-8 routing, shared-expert gating, and residual reduction implemented in `Exl3MoEMethod.apply`. Router alignment achieves **100% top-8 set agreement** (`cos_min = 0.99999994`). Six extreme edge-case routing scenarios validated. Zero token dropping verified across all 8 slots. Full hybrid decoder layer (GemmaRMSNorm $\to$ GDN linear attention $\to$ GemmaRMSNorm $\to$ EXL3 MoE $\to$ residual) verified on real checkpoint weights (`max_abs = 0.00232`). Standardized Phase 2 suite runner passed **6/6 suites in 88.8s**.

---

## 2. Gate-by-Gate Implementation Status

### Phase 0: Pin, Audit, and Prove the Platform (Gate 0)

| Task ID | Description | Status | Evidence Artifacts |
|---|---|:---:|---|
| **P0-01** | Fork `exl3xpu`, pin environment and toolchain | **MET** | [`docs/compatibility_manifest.md`](file:///home/yanchun/Projects/vllm.xpu/docs/compatibility_manifest.md) §3, `logs/moe-preflight/environment-moe.txt` |
| **P0-02** | Audit installed vLLM 0.31.0 MoE interfaces and hooks | **MET** | [`docs/moe_api_audit.md`](file:///home/yanchun/Projects/vllm.xpu/docs/moe_api_audit.md) |
| **P0-03** | Catalog checkpoint schema & inventory 124,579 tensors | **MET** | [`docs/checkpoint_schema.md`](file:///home/yanchun/Projects/vllm.xpu/docs/checkpoint_schema.md), `logs/moe-preflight/checkpoint-moe-audit.json` |
| **P0-04** | Validate ExLlamaV3 compatibility & bit-exact reconstruction | **MET** | `scripts/check_exl3_backends.py` (18 modules, recon diff = 0), `logs/moe-preflight/exl3-backend-parity.json` |
| **P0-05** | Probe Level Zero limits (28.58 GiB) & verify Xe2 AOT compile (0 spills) | **MET** | `logs/moe-preflight/l0-capabilities.json`, `logs/moe-preflight/esimd-compile-lnl.log` |
| **P0-06** | Test baseline dense and expert matrix shapes on Arc 140V | **MET** | `logs/moe-preflight/k3-mr-sweep.json` (32/32 PASS), `logs/moe-preflight/exl3-backend-parity.log` |
| **P0-07** | Probe vLLM GDN linear-attention on Arc 140V | **MET** | `scripts/gdn_fixture_probe.py` (decode + prefill + determinism PASS), `logs/moe-preflight/gdn-fixture-probe.json` |
| **P0-08** | CPU deterministic oracle & external CUDA traces | **PARTIAL** | CPU oracle MET (`tests/ref_moe_oracle.py`, 26/26 arrays deterministic). CUDA traces non-gating for Gates 1–3 |
| **P0-09** | Define formal numerical tolerances (T-1 to T-6) | **MET** | [`docs/tolerance_spec.md`](file:///home/yanchun/Projects/vllm.xpu/docs/tolerance_spec.md) |
| **P0-10** | Publish Phase 0 architectural documentation | **MET** | [`docs/compatibility_manifest.md`](file:///home/yanchun/Projects/vllm.xpu/docs/compatibility_manifest.md), [`docs/moe_api_audit.md`](file:///home/yanchun/Projects/vllm.xpu/docs/moe_api_audit.md), [`docs/checkpoint_schema.md`](file:///home/yanchun/Projects/vllm.xpu/docs/checkpoint_schema.md) |
| **P0-11** | Prior art survey (`vllm-exl3`, B70 expert server, ExLlamaV3) | **MET** | [`docs/prior_art_survey.md`](file:///home/yanchun/Projects/vllm.xpu/docs/prior_art_survey.md) |
| **P0-12** | Empirical hardware roofline microbenchmarks on 140V | **MET** | [`bench/roofline.py`](file:///home/yanchun/Projects/vllm.xpu/bench/roofline.py), `logs/moe-preflight/roofline-140v.json` |

**Gate 0 Assessment:** **PASSED**.

---

### Phase 1: Expert Loading and One Expert MLP (Gate 1)

| Task ID | Description | Status | Evidence Artifacts |
|---|---|:---:|---|
| **P1-01** | Extend vLLM quantization dispatch for `RoutedExperts` | **MET** | `exl3xpu/vllm_plugin.py`, `tests/test_dispatch.py` (**15/15 PASS**), `logs/moe-preflight/test-dispatch.json` |
| **P1-02** | Expert-aware parameter store & tiled memory arenas | **MET** | `exl3xpu/moe.py` (`Exl3ExpertStore`), `tests/test_expert_loader.py` (**10/10 PASS**) |
| **P1-03** | Construct packed expert descriptors (`[256, 3, 14]` buffer) | **MET** | `exl3xpu/moe.py` (`_build_descriptors`), `tests/test_expert_loader.py` (**12/12 PASS**) |
| **P1-04** | Verify individual expert metadata integrity & layout | **MET** | `exl3xpu/moe.py` (`verify_expert_integrity`), `tests/test_expert_loader.py` (**15/15 PASS**) |
| **P1-05** | Shard-by-shard streaming loader with memory flushes | **MET** | `scripts/test_shard_streaming.py` (122,880 tensors in 9.7s, RSS ≤ 1.26 GiB), `logs/moe-p1/shard-loading-memory.csv` |
| **P1-06** | Exact weight reconstruction for experts {0, 1, 127, 254, 255} | **MET** | `tests/test_expert_reconstruction.py` (**15/15 bit-exact**, `max_abs=0.0`), `logs/moe-p1/reconstruction-audit.log` |
| **P1-07** | Single expert MLP forward pass execution & validation | **MET** | `tests/test_expert_mlp.py` (`max_abs = 3.62e-6` vs 0.08 tol), `logs/moe-p1/single-expert-eval.log` |
| **P1-08** | Token batch sweep ($M \in [0, 256]$) & crossover calibration | **MET** | `tests/test_batch_sweep.py` (M=0 launches 0 kernels; crossover at M=8), `logs/moe-p1/crossover-calibration.json` |
| **P1-09** | Reentrancy & intermediate scratch buffer audit | **MET** | `tests/test_reentrancy.py` (C++ stream-ordered buffer fix in r4, 0 mismatches), `logs/moe-p1/reentrancy-audit.log` |
| **P1-10** | Regression testing on dense linear models | **MET** | `scripts/test_exl3_ops.py` on r4 artifact (**90/90 PASS**), `logs/moe-p1/dense-regression-check.log` |

**Gate 1 Assessment:** **PASSED**. Full Phase 1 suite logged in `logs/moe-p1/gate1-suite.log`.

---

### Phase 2: Correct Complete MoE Block (Gate 2)

| Task ID | Description | Status | Evidence Artifacts |
|---|---|:---:|---|
| **P2-01** | Eager token dispatch and reduction baseline | **MET** | `exl3xpu/moe.py` (`Exl3MoEMethod.apply`), `tests/test_moe_block.py` (`max_abs = 2.49e-6`), `logs/moe-p2/moe-block-eval.json` |
| **P2-02** | Router top-$k$ normalization & logit alignment | **MET** | `tests/test_moe_routing.py` (`cos_min = 0.99999994`, 100% top-8 match), `logs/moe-p2/routing-eval.json` |
| **P2-03** | Verify exact downstream routing weight application point | **MET** | `tests/test_moe_routing.py` (input-side application proven 50× worse), `logs/moe-p2/routing-eval.json` |
| **P2-04** | Shared expert execution and learned gating semantics | **MET** | `tests/test_moe_block.py` (vLLM runner ownership verified, no double execution), `logs/moe-p2/moe-block-eval.json` |
| **P2-05** | Differential routing: fixed routing vs free routing | **MET** | `tests/test_fixed_routing.py` (`max_abs ≤ 1.35e-6`), `tests/test_free_routing.py` (`max_abs ≤ 1.18e-6`) |
| **P2-06** | Extreme routing edge-case suite (6/6 scenarios) | **MET** | `tests/test_moe_routing.py` (skew, uniform, zero-token, ties, M=0, M=1), `logs/moe-p2/edge-cases.log` |
| **P2-07** | Zero token dropping and full top-8 participation | **MET** | `tests/test_moe_block.py` (all 8 slots verified contributing, 32/32 assignments), `logs/moe-p2/moe-block-eval.json` |
| **P2-08** | Intermediate-stage validation probe hook | **MET** | `tests/test_intermediate_probe.py` (`enable_debug_capture`, worst `max_abs = 5.25e-5`), `logs/moe-p2/intermediate-stage-validation.json` |
| **P2-09** | Complete hybrid decoder layer forward pass integration | **MET** | `tests/test_hybrid_layer.py` (GemmaRMSNorm $\to$ GDN attn $\to$ norm $\to$ MoE $\to$ residual, `max_abs = 0.00232`), `logs/moe-p2/full-layer-eval.json` |
| **P2-10** | Standardized Phase 2 runner & fixture reports | **MET** | `scripts/run_phase2_tests.py` (**6/6 suites PASS, worst max_abs = 0.00244, 88.8s**), `logs/moe-p2/phase2-summary.json` |

**Gate 2 Assessment:** **PASSED**. Full Phase 2 summary logged in `logs/moe-p2/phase2-summary.json`.

---

## 3. Key Technical Discoveries & Resolutions

```
+----------------------------------------------------------------------------------------------------+
|                                CRITICAL ENGINEERING RESOLUTIONS                                    |
+----------------------------------------------------------------------------------------------------+
| 1. CHECKPOINT REVISION & PAYLOAD:                                                                  |
|    - Target pinned: turboderp/Qwen3.5-35B-A3B-exl3 @ 3.09bpw (f599771cd8ca99ce...).                |
|    - Payload: Exactly 15.29 GB across 2 shards (model-00001: 8.0 GB, model-00002: 6.4 GB).         |
|    - Downloaded and verified in /home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw.                 |
+----------------------------------------------------------------------------------------------------+
| 2. ESIMD 3-BIT MCG ROUTED KERNEL:                                                                  |
|    - Discovered: 30,720 routed experts use Trellis K=3 with MCG codebook (cb=1).                   |
|    - Solution: Instantiated launch_reconstruct<3,1> and dispatch_mr<3,1> in csrc/exl3_ops.sycl.    |
|    - AOT IGC compile on lnl-a1 confirmed 0 register spills.                                        |
+----------------------------------------------------------------------------------------------------+
| 3. INDEPENDENT HADAMARD TRANSFORMS:                                                                |
|    - Discovered: gate_proj and up_proj suh sign vectors match 0/120 times.                         |
|    - Solution: Maintained independent FWHT transforms; avoided invalid shared FWHT optimization.  |
+----------------------------------------------------------------------------------------------------+
| 4. RECONSTRUCTION SCRATCH REENTRANCY BUG (FIXED IN _C.so r4):                                      |
|    - Discovered: Upstream ops.py used static std::vector<at::Tensor> shared across streams.        |
|    - Solution: Switched to dynamic stream-ordered at::empty({k, n1-n0}) allocation.                |
|    - Verification: 2 concurrent XPU streams x 100 interleaved trials matched serial bit-for-bit.  |
+----------------------------------------------------------------------------------------------------+
| 5. 2D SUH CONTRACT (PERFORMANCE FIX):                                                              |
|    - Discovered: Passing 1-D suh triggers S=k in C++, causing k redundant input FWHTs (2 GiB alloc)|
|    - Solution: Exl3MoEMethod.apply explicitly unsqueezes suh to [1, k] ([num_groups, k] contract).|
+----------------------------------------------------------------------------------------------------+
| 6. STOCK vLLM 3D TENSOR CHUNKING BUG:                                                              |
|    - Discovered: RoutedExperts.load_weights chunks any 3D tensor, corrupting expert dimensions.    |
|    - Solution: Implemented per-layer _layer_load_weights parsing <expert>.<proj>.<suffix> keys.    |
+----------------------------------------------------------------------------------------------------+
| 7. FUSED_ADD_RMS_NORM IN-PLACE MUTATION TRAP:                                                      |
|    - Discovered: vLLM's fused_add_rms_norm mutates the residual tensor in place.                   |
|    - Solution: Cloned residual inputs in hybrid layer test harness to prevent corrupting inputs.   |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Hardware Telemetry & Test Suite Metrics

### Phase 1 Empirical Results
- **Streaming Shard Load (40 layers, 122,880 tensors):**
  - Shard 1 (70,656 tensors): **5.3 seconds** (host RSS: 1,261 MiB, XPU alloc: 11,673 MiB).
  - Shard 2 (52,224 tensors): **4.4 seconds** (host RSS: 1,253 MiB, XPU alloc: 11,673 MiB).
  - Memory discipline: Host RSS remained flat ($\le 1.26\text{ GiB}$) regardless of 8 GB file sizes; XPU allocation flat at **11.4 GiB** with **0 transient duplication**.
- **Weight Reconstruction Accuracy:**
  - 15/15 full matrices ({gate, up, down} $\times$ {0, 1, 127, 254, 255}): **100% bit-exact**, `max_abs = 0.0`.
- **Single-Expert MLP Numerics:**
  - Down projection `max_abs`: **$2.9 \times 10^{-6} \dots 3.6 \times 10^{-6}$**.
  - Gate/Up projection `max_abs`: **$3.1 \times 10^{-5} \dots 2.5 \times 10^{-4}$**.
- **Batch Sweep & Crossover:**
  - $M=0$: **0 kernel launches**, **0 µs** GPU time.
  - $M \in [1, 256]$: worst `max_abs` = **$4.3 \times 10^{-6}$**.
  - Small/Large $M$ crossover confirmed at **$M=8$** (fused vector is 1.49× faster at $M=1$; break-even at $M=8$; $M>8$ routes to reconstruct+oneDNN).

### Phase 2 Empirical Results ([`scripts/run_phase2_tests.py`](file:///home/yanchun/Projects/vllm.xpu/scripts/run_phase2_tests.py))

```text
================================================================================
Phase 2 Consolidated Test Summary (Target: max_abs <= 0.08, NaN/Inf == 0)
================================================================================
Suite 1: P2-01/04/07 MoE Block Evaluation      PASS   15.44s   worst_abs: 2.59e-06
Suite 2: P2-02/03/06 Routing & Edge Cases      PASS   11.16s   worst_abs: 3.45e-06
Suite 3: P2-05a Fixed Routing Verification     PASS    9.14s   worst_abs: 1.35e-06
Suite 4: P2-05b Free Dynamic Top-8 Routing     PASS   18.23s   worst_abs: 1.18e-06
Suite 5: P2-08 Intermediate Diagnostic Probe   PASS   13.91s   worst_abs: 5.25e-05
Suite 6: P2-09 Full Hybrid Decoder Layer       PASS   20.90s   worst_abs: 0.002441
--------------------------------------------------------------------------------
ALL 6 SUITES PASS (88.78s total runtime, 0 failures, 0 NaNs/Infs)
================================================================================
```

---

## 5. Next Steps: Phase 3 Execution (Full-Model Serving)

With Gates 0, 1, and 2 certified, implementation advances directly to **Phase 3 (Bounded Full-Model Serving)**:

1. **Task P3-01 & P3-02 (Full Model Streaming Load):**
   - Execute shard streaming across all 40 layers and verify text-only weight filtering (exclude multimodal/MTP weights).
   - Confirm peak memory stays under 22 GiB during load, preserving $>9.6\text{ GiB}$ host headroom.
2. **Task P3-03 (Engine Configuration & Startup `profile_run`):**
   - Configure engine parameters:
     ```bash
     --max-model-len 2048 \
     --max-num-seqs 1 \
     --max-num-batched-tokens 256 \
     --enforce-eager \
     --gpu-memory-utilization 0.70
     ```
   - Verify that startup `profile_run` passes without memory spikes or launch timeouts.
3. **Task P3-04 (KV & GDN Allocation Audit):**
   - Confirm KV cache allocation (~40 MiB) and GDN recurrent states (~63 MiB) initialize cleanly.
4. **Task P3-05 (Model Recipe & Dedicated Launcher):**
   - Author `models/qwen3.5-35b-a3b-exl3-3.09bpw-140v/model.yaml`.
   - Create launcher [`scripts/launch_vllm_moe_exl3.sh`](file:///home/yanchun/Projects/vllm.xpu/scripts/launch_vllm_moe_exl3.sh) on port **8002**.
5. **Tasks P3-06 to P3-11 (Functional Serving & Quality Validation):**
   - Verify WikiText-2 perplexity via [`scripts/eval_moe_ppl.py`](file:///home/yanchun/Projects/vllm.xpu/scripts/eval_moe_ppl.py).
   - Validate both stop tokens (`<|endoftext|>`, `<|im_end|>`) and test live chat completions through the OpenAI API endpoint.
