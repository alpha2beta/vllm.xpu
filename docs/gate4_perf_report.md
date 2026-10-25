# Gate 4 Performance Report — Grouped M=1 Decode Fast Path (EXL3 MoE Qwen3.5-35B-A3B)

**Date:** 2026-10-24/25
**Hardware:** Intel Arc 140V (8 Xe2 cores, 32 GB LPDDR5X), driver xe + Level-Zero 1.17.39758, oneAPI 2026.1.0
**Software:** vLLM 0.31.0+xpu, torch 2.14.0+xpu, exl3xpu `306192e`, vllm.xpu `fae3379`
**Artifact:** `_C.so` sha256 `b627e3a4…` (r7, built with `-DEXL3_ALL_CODEBOOKS`), checkpoint `turboderp/Qwen3.5-35B-A3B-exl3` @ 3.09 bpw
**Config:** loopback port 8002, TP=1, `--max-model-len 2048`, `--max-num-seqs 1`, `--enforce-eager`, `gpu-memory-utilization 0.62`

## Executive summary

| Gate 4 criterion | Result | Evidence |
|---|---|---|
| 1. Substantial decode speedup over eager baseline | **PASS — 4.9–5.4× at the server (9.8× at the layer profiler)** | `results.csv` r7 rows, `logs/moe-p4/moe-profile-r7-p402.json` |
| 2. Zero host-device syncs during single-sequence decode | **PASS — 0.0 syncs/token** | `logs/moe-p4/moe-profile-r7-p402.json` |
| 3. Zero DRAM writeout of intermediate FP16 weights | **PASS — 0 `_reconstruct_kernel`, 0 D2H/H2D** | `logs/moe-p4/moe-profile-r7-p402.json`, `logs/moe-p4/bandwidth-efficiency.log` |
| 4. Numerical outputs match Gate 2/3 baselines | **PASS — bit-identical WikiText-2 PPL, Phase 2 6/6, dense 91/91** | `logs/moe-p4/gate4-c4-ppl-parity.log`, `logs/moe-p2/phase2-summary.json`, `logs/moe-p4/dense-regression-r7.log` |
| 5. Multi-hour stress stability | **PASS — 397/397 requests over 3 h, 0 errors, −3.0 % decode drift, GPU flat** | `logs/moe-p4/soak-r7.jsonl`, `logs/moe-p4/soak-r7.summary.json` |

## 1. What P4-02 changed

P4-01 (zero-copy exec cache + B=1 fast path) left **145 kernel launches and 2 host syncs per decode token**, of which 72 launches were the per-expert MoE chain (8 experts × 3 projections × [HadIn, GEMV, HadOut]) and both syncs were `.tolist()` reads of `topk_ids`/`topk_weights`.

P4-02 (`csrc/exl3_moe_gemv.sycl`, exl3xpu `306192e`) collapses this to **2 grouped launches and 0 syncs**:

* **unit = (expert, slot)** — for gate+up, `nslots=2` makes the existing `[E, 2, …]` arena a flat `[units, …]` view; the active experts are read from a device-side `expert_ids` array built from `topk_ids` with `repeat_interleave` (no host read).
* **gate+up**: one launch with `x_unit_stride=0` (all 16 units share the token row); **down**: one launch with `x_unit_stride=Kdim` (each unit consumes its own `SiLU(gate)×up` row, computed by one device expression between the two launches).
* dtype-templated `T ∈ {fp16, bf16}` — the real model runs bf16 activations (the first profiler attempt silently fell back to the eager loop because the op required fp16; that fallback path is what the r7 parity test now locks down).

### Layer-0 decode profile lineage (same harness, `bench/moe_profile.py`, 50 steps)

| build | ms/token | host syncs/token | launches/token |
|---|---|---|---|
| r5 eager (Gate 3 artifact) | 8.186 | 27.0 | 253 |
| r6 eager (`-DEXL3_ALL_CODEBOOKS`) | 6.137 | 27.0 | 245 |
| r6 + P4-01 (exec cache, B=1 fast path) | 3.064 | 2.0 | 145 |
| **r7 + P4-02 (grouped kernel)** | **0.835** | **0.0** | **41** |

**9.8× vs the r5 eager baseline, 3.67× over P4-01.** The remaining 41 launches/token are dominated by the shared expert and GDN attention: the profile's largest remaining stage is the dense K=4 path (`GemvKernel<4,1,1,8,false>` 299 µs/token + its `HadOutKernel` 127 µs/token + `HadInKernel` 25 µs/token) — the shared expert still runs the 3-launch eager chain, as does GDN — while the grouped routed kernels together cost 332 µs/token for all 8 experts × 3 projections. Prefill-side and shared-expert grouping is P4-03 territory.

## 2. Criterion 1 — server-level decode speedup

Same matrix as the Gate 3 baseline (`scripts/bench_envelope.py`, 32 output tokens, concurrency 1, port 8002):

| context | Gate 3 eager | P4-01 (r6) | **P4-02 (r7)** | vs Gate 3 | vs P4-01 |
|---|---|---|---|---|---|
| 128 | 4.69 tok/s | 16.07 | **25.27** | **5.39×** | 1.57× |
| 512 | 5.66 tok/s | 15.98 | **27.90** | **4.93×** | 1.75× |
| 1024 | 4.83 tok/s | 15.89 | **26.19** | **5.42×** | 1.65× |

TTFT: 1.75 / 4.97 / 9.08 s (Gate 3: 2.95 / 6.71 / 11.62 s). End-to-end: 3.02 / 6.11 / 10.30 s (Gate 3: 9.77 / 12.37 / 18.24 s).

Rows: `results.csv` (git `fae3379`, notes `Phase 4 grouped r7+P4-02 …`).

## 3. Criterion 2 — zero host-device syncs

`logs/moe-p4/moe-profile-r7-p402.json`: `host_syncs_per_token = 0.0`, all five audited primitives (`item`, `tolist`, `__int__`, `__float__`, `__index__`) at zero across 50 decode steps.

The two P4-01 syncs (routing decision read-back) are gone because `expert_ids` stays on the device: `topk_ids → repeat_interleave → torch.ops` crosses no host boundary, and the weighted fp32 reduction is a device `sum`.

## 4. Criterion 3 — zero DRAM writeout of intermediate FP16 weights

The r7 profile stage list contains **0 `_reconstruct_kernel` launches and 0 D2H/H2D copies**. The grouped kernel unpacks trellis + codebook + scales + Hadamard in registers inside `GroupedFusedKernel`, exactly as the single-expert fused path did (P4-05 analysis in `logs/moe-p4/bandwidth-efficiency.log`: 26.50 MiB/token/layer read volume, workspace bounded by `RECON_SLICE_N`, persistent workspace 0 MiB).

Effective bandwidth over the P4-07 sweep: 19.1–29.2 GB/s = **14.7–22.4 % of the 130.4 GB/s P0-12 copy roofline** — still latency/occupancy bound, not bandwidth bound, so the launch reduction converts directly into throughput.

## 5. Criterion 4 — numerical parity with Gate 2/3

Four independent layers of evidence:

1. **WikiText-2 PPL, bit-identical** (`logs/moe-p4/gate4-c4-ppl-parity.log`): every one of the 5 Gate-3 chunks matches to `0.000e+00` absolute; geometric mean 7.439027596 on both r7-grouped and Gate-3-eager (the residual 2.7e-15 is only the final `exp(mean(log))` rounding). Extended 10-chunk run: PPL 6.3971 over 20,470 tokens.
2. **Kernel/apply parity** (`tests/test_grouped_gemv.py`, 3/3): grouped vs eager is **bit-for-bit** at fp16 *and* bf16 — gate+up (16 units), down (8 units), the full grouped MLP, and `apply()` end-to-end, plus shape validation. A separate sweep ran **1920/1920** bit-exact comparisons (5 seeds × all 256 experts × 3 projections).
3. **Phase 2 suite** (`logs/moe-p2/phase2-summary.json`): 6/6 PASS, worst `max_abs = 0.00244140625` — unchanged from Gate 2/3 (the P4-02 path does not move the numerics needle at all).
4. **Dense regression** (`logs/moe-p4/dense-regression-r7.log`): 91/91 PASS, case set byte-identical to the r6 baseline run — the dense launchers are untouched (P4-02 is a new file + one `#include` + one op registration).

Dispatch/loader/reconstruction/reentrancy/batch-sweep all green on r7 (15/15, 15/15, 25 reconstructions, pass, pass).

### Why bit-exactness holds

For K=3 the eager path takes the **non-fused** branch of `exl3_gemm_small`: `HadIn → GEMV(rps=1) → HadOut`, where each split owns exactly one tile-row and `HadOutKernel` sums rows sequentially with `v = v * kRsqrt128 * svh`. The grouped kernel therefore writes **row-granular** `part[r]` (each row's fp32 lane-reduction is byte-identical to eager's `rps=1` split) and reduces it with the same sequential chain in `grouped_had_out`. The result is independent of the grouped kernel's own split-K geometry, which is what let the first implementation pass fp16 tests while still failing bf16-in-anger — the dtype template plus this layout is what the parity test now pins.

## 6. Criterion 5 — multi-hour stress stability

**PASS.** 3.00-hour soak on 2026-10-25 00:01–03:00 CEST against the live r7
server (`bench/soak_moe.py`, rotating real/uniform/skewed × 128/512/max-context
cells, ~7 min/request cycle including 5 s inter-request sleep):

| metric | result | bar |
|---|---|---|
| requests | **397 total, 397 ok, 0 errors** (all three routing patterns ≈133 each) | ≥100, 0 errors |
| duration | 3.00 h continuous | multi-hour |
| decode drift (2nd half vs 1st half median) | **−3.0 %** (17.70 → 17.17 tok/s) | ±10 % |
| TTFT drift | 4.96 → 4.94 s median (max 12.05 s at max-context cell) | no degradation |
| GPUActive | **18.36 GiB flat** (start = min = max; leak bar was < +0.5 GiB) | stable |
| temperature | 44 → max 52 °C, no throttling | no throttle |
| swap | 8.68 → 9.72 GiB (+1.02 GiB) while MemAvailable *rose* 2.16 → 3.02 GiB — host page-cache churn, not a serving-process leak (GPUActive and per-request latency flat throughout) | observed, benign |

Per-pattern decode during soak (median): real 17.47, uniform 17.30,
skewed 17.94 tok/s — no pattern-specific degradation; the soak's mix
includes max-context cells (in≈1912), so these medians sit below the
short-context P4-07 numbers by construction.

Evidence: `logs/moe-p4/soak-r7.jsonl` (397 JSONL records, one per request),
`logs/moe-p4/soak-r7.summary.json`.

## 7. P4-07 benchmark matrix (`results_moe.csv`)

18/18 cells OK on artifact `b627e3a4`, harness `bench/bench_moe_envelope.py`:

* inputs 128 / 512 / 2048 × outputs 128 / 512 × routing patterns **real / uniform / skewed**
* (nominal-2048 input cells are context-capped to `2048 − output − 8` margin, noted per row — the server hard-rejects prompt+output > 2048)

| metric | min | median | max |
|---|---|---|---|
| decode tok/s | 17.22 | 22.02 | 26.30 |
| TTFT s | 1.49 | 4.53 | 15.24 |
| prefill tok/s | 56.9 | 93.8 | 430.5 |
| bandwidth GB/s (% roofline) | 19.1 (14.7 %) | 23.9 (18.8 %) | 29.2 (22.4 %) |
| peak GPU GiB | — | 18.36 | — |
| thermal °C (start→end) | 44 | — | 68 (no throttle observed) |

By routing pattern (decode median): real 22.37, uniform 22.23, skewed 21.16 tok/s — routing skew costs ≈5 % at decode, i.e. the grouped kernel is robust to expert concentration.

Ranking check: the fastest cells are max-context (in≈1912): KV-heavy decode still reaches 25–26 tok/s, confirming the win is launch-bound removal, not short-context luck.

## 8. Attribution

| item | value |
|---|---|
| kernel | `csrc/exl3_moe_gemv.sycl` (new), exl3xpu `306192e` |
| wiring | `exl3xpu/moe.py::_grouped_mlp_decode` + `apply()` M=1 branch; `exl3_ops.sycl` gains 1 include + 1 op registration |
| untouched | `exl3_esimd.h`, all dense launchers (proved by 91/91 dense regression, identical case set) |
| build | `EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS scripts/build_exl3_ext.sh`, log `logs/moe-preflight/esimd-build-moegrouped-r7/` |
| artifact | `_C.so` `b627e3a4` |
| vllm.xpu | `54fedb2` (P4-02), `c2f27f9` (P4-07 harness + C4 evidence), `fae3379` (C1 rows) |
| fallback | any capability/shape failure latches `_grouped_supported=False` and the P4-01 eager loop serves the request — correctness never depends on the fast path |

## 9. Remaining work

* **P4-03**: grouped prefill (`exl3_moe_grouped_gemm.sycl`) — the profile shows the shared-expert/dense K=4 chain (299 + 127 µs/token) is now the largest decode term, and prefill TTFT is the next lever (4.97–9.08 s at 512/1024).
* **P4-04**: blocked FWHT sharing between gate/up (verify `s_u,gate == s_u,up` from P0-03 first).
* **P4-06**: TorchScript/Inductor custom-op schemas + fake-tensor tests.
