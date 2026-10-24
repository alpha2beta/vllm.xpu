# Compatibility Manifest — Qwen3.5-35B-A3B EXL3 on Arc 140V (P0-01 / P0-05 / P0-10)

Frozen baseline for the EXL3 MoE bring-up. Everything in this table was read
from the live environment by `scripts/moe_preflight.py`
(`logs/moe-preflight/environment-moe.json`) or measured by the probes cited
below. Numbers in this document are the contract for Phases 1–4.

## 1. Host and OS

| Item | Value |
| --- | --- |
| Host | `cachyos-x8664` |
| OS | CachyOS (Arch-based), rolling |
| Kernel | `7.2.11-1-cachyos-deckify` |
| CPU | Intel Core Ultra 7 258V (8 logical CPUs) |
| RAM | 30.88 GiB total, 23.18 GiB `MemAvailable` at probe time |
| Swap | 30.88 GiB total, 6.49 GiB in use at probe time |
| `ttm.pages_limit` | 4 047 107 pages (≈ 15.4 GiB of GEM/TTM pages) |

`ttm.pages_limit` is the *shared-memory window* for the i915/xe driver, not a
hard cap on GPU-visible memory. `torch.xpu.mem_get_info()` reports the kernel's
free-page-watermark rather than `MemAvailable` (integrated GPU shares system
RAM); the probe therefore records both. Plan on `MemAvailable`, not
`mem_get_info`.

## 2. GPU / Level Zero

Source: `logs/moe-preflight/l0-capabilities{,-relaxed}.json`,
`scripts/l0_caps.cpp`.

| Item | Value |
| --- | --- |
| Device | Intel(R) Arc(TM) Graphics, `device_id 0x64a0` (Xe2 / Lunar Lake) |
| Runtime | oneAPI Unified Runtime over Level-Zero V2 |
| Driver | `1.17.39758` (`driver_version_raw 17013582`) |
| Level Zero loader | `/usr/lib/libze_loader.so.1.32.0` |
| `globalMemSize` | 30 692 810 752 B = **28.58 GiB** |
| `maxMemAllocSize` | 30 692 810 752 B = **28.58 GiB** (no smaller single-allocation cap) |
| EU / slices / subslices | 64 EU, 2 slices, 4 subslices, 8 threads/EU, SIMD 16 |
| L2 cache | 8 MiB (8 388 608 B) |
| SLM | 128 KiB per work-group |
| Sub-group sizes | 16, 32 |
| Max work-group | 1024 |
| SPIR-V | 1.5 |

### Allocation ladder

| Request | Result |
| --- | --- |
| 0.25 GiB | `ZE_RESULT_SUCCESS` |
| 0.50 GiB | `ZE_RESULT_SUCCESS` |
| 1.00 GiB | `ZE_RESULT_SUCCESS` |
| 2.00 GiB | `ZE_RESULT_SUCCESS` |
| 4.00 GiB | `ZE_RESULT_SUCCESS` |

`ZE_experimental_relaxed_allocation_limits` is advertised by the driver, but the
4 GiB allocation already succeeds **without** setting
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS`. Gate 0 requirement “single-allocation
cap handled, ≥ 24 GiB addressable” is satisfied: the cap equals total memory
and there is no 4 GiB per-allocation cliff on this stack. The P1-02 budget
(≤ 4 GiB per expert slab) remains a deliberate safety margin, not a driver
requirement.

### ESIMD compile diagnostics (P0-05 item 5 — resolved)

Source: `scripts/esimd_aot_diag.sh` → `logs/moe-preflight/esimd-compile-lnl.{log,json}`.

The production build uses `-fsycl-targets=spir64`, which emits SPIR-V and
defers register allocation to IGC at JIT time, so the host build log carries no
register/spill information. The diagnostic therefore compiles the **real**
translation unit ahead of time for the 140V device:

```
icpx -fsycl -fsycl-targets=spir64_gen \
     -Xsycl-target-backend "-device lnl-a1" \
     -O3 -ffast-math -std=c++20 -c -fsycl-device-code-split=per_kernel \
     -DEXL3_ALL_CODEBOOKS -DEXL3_NO_DNNL=1 ...
```

**Detector.** IGC reports per-kernel spills itself on an AOT device build:

```
Spill memory used = 31296 bytes for kernel _ZTSZ11spill_heavy...
Compiling kernel with spill code may degrade performance. ...
```

The script first builds a synthetic negative control (300 live
`simd<float,32>` = 300 GRF, beyond the 256-GRF maximum) and aborts unless that
report appears, so a clean run cannot be a false negative.

| Approach | Verdict |
| --- | --- |
| `-Rpass-analysis=kernel-resource-usage` | **absent** from ICPX 2026.0; emits no remarks |
| `-cl-fail-on-spill` (via `-internal_options`) | accepted but **inert**: the 31 KB-spilling control still exits 0 |
| IGC `Spill memory used` report on `lnl-a1` AOT build | **works**, used by the script |

**Result: PASS.** All instantiated kernels compile for `lnl-a1`, with **0**
spill reports, and the routed-expert format is present in the device image:

| Instantiation | Rows served |
| --- | --- |
| `GemvKernel<3,1,MR=1,NT=8>` | M = 1 |
| `GemvKernel<3,1,MR=2,NT=4>` | M = 2 |
| `GemvKernel<3,1,MR=4,NT=4>` | M = 3–4 |
| `GemvKernel<3,1,MR=8,NT=4>` | M = 5–8 |
| `ReconstructKernel<3,1,NT=8,half>` | M > 8 (reconstruct+GEMM) |

Two consequences are recorded as deviations rather than failures:

1. The AOT image is **diagnostic only**; the shipped `_C.so` still targets
   `spir64` and relies on IGC JIT at first use (the AOT device build is not
   linked into the extension).
2. `(K=3, cb=1)` is **vector-only** on this part (no XMX, so no DPAS
   instantiation), and the vector kernel is limited to **M ≤ 8**
   (`exl3_gemm_small: M=9 exceeds no-DPAS vector limit 8`). `ops.small_m_max()`
   now queries the new `exl3_small_m_max(K, cb)` binding and routes M > 8 to
   reconstruct+GEMM instead of tripping that check.

## 3. Software pins

| Component | Version | Note |
| --- | --- | --- |
| Python | 3.12.14 | uv-managed CPython |
| vLLM | **0.31.0+xpu** | not the 0.30.0 assumed by the plan |
| PyTorch | **2.14.0+xpu** | plan assumed 2.13.0+xpu |
| Triton | 3.8.0+xpu | K-generic EXL3 fallback path |
| Transformers | 5.17.0 | affects `Qwen3_5MoeTextConfig` naming |
| Tokenizers | 0.23.2 | |
| safetensors | 0.8.0 | header-only audit path |
| huggingface-hub | 1.33.0 | |
| NumPy | 2.3.5 | fixtures / oracle |
| exl3xpu | 0.1.0 | local editable fork |
| ICPX | Intel oneAPI DPC++/C++ 2026.0.0 (2026.0.0.20260331) | SYCL/ESIMD builds |

### Repository pins

| Repo | Path | Commit |
| --- | --- | --- |
| vLLM XPU fork | `/home/yanchun/Projects/vllm.xpu` | `c93c5f317057fc16c3f3c7df512805f9218ee423` (`Qwen35B_MoE`, dirty) |
| exl3xpu (owned fork) | `/home/yanchun/Projects/exl3xpu` | `15ded2f3add148c4db3c900cba7de878238f53bc` (`main`, dirty) |
| Checkpoint | `turboderp/Qwen3.5-35B-A3B-exl3` | revision `3.09bpw` = `f599771cd8ca99ce9bb62d58a855ecfbac60e3c2` |

The upstream `sybil-solutions/exl3xpu` repository was archived 2026-10-12; the
local fork is now the source of truth for kernels and the vLLM plugin.

### Deviations from the written plan

1. **PyTorch 2.14.0+xpu**, not 2.13.0+xpu. The XPU wheel for 2.13 is no longer
   in the pinned index; 2.14 passes the full parity suite and is what the
   installed vLLM wheel links against.
2. **vLLM 0.31.0+xpu**, not 0.30.0. 0.31.0 is the first release that registers
   `vllm::gdn_attention_core_xpu` and the `Qwen3_5Moe` hybrid model.
3. **Bring-up checkpoint is `3.09bpw`**, not `4.00bpw`. The `4.00bpw` and
   `3.00bpw` revisions named in `tasks_moe.md` do not exist in the repo; the
   published 3.09 bpw revision is the closest documented lower-bitrate build and
   is the only one with a complete 40-layer expert set (see
   `docs/checkpoint_schema.md`).
4. **No MTP tensors** ship in the checkpoint despite `mtp_num_hidden_layers: 1`
   in `config.json`. Speculative decoding is out of scope for Phase 0–3.
5. **The extension must be built as C++20.** torch 2.14's headers contain
   `#error C++20 or later compatible compiler is required`; the upstream
   `exl3xpu/scripts/build_ext.sh` (`-std=c++17`) fails against this wheel.
   `scripts/build_exl3_ext.sh` uses `-std=c++20`. The canonical build command is
   `EXL3_FLAGS=-DEXL3_ALL_CODEBOOKS bash scripts/build_exl3_ext.sh <log_dir>`.
6. **The spill diagnostic is AOT-only.** The shipped `_C.so` targets `spir64`
   and is JIT-compiled by IGC, so register pressure can only be observed by
   compiling the real translation unit for `lnl-a1` (§2). `-cl-fail-on-spill`
   is inert on this stack; the diagnostic asserts on IGC's own
   `Spill memory used` report instead, with a validated negative control.
7. **`(K=3, cb=1)` is vector-only and capped at M = 8.** There is no XMX/DPAS
   instantiation for the routed format on the 140V, so `exl3_small_m_max(3,1)
   == 8`; the Python layer routes M > 8 to reconstruct+GEMM. The original plan
   assumed one `SMALL_M_MAX = 128` for all formats.

## 4. Empirical roofline

Source: `bench/roofline.py`,
`logs/moe-preflight/roofline-140v.{json,txt}`. Spec bandwidth for the 140V is
136 GB/s (LPDDR5X-8533, 64-bit).

### Memory bandwidth vs working set

| Working set | Read GB/s | Copy GB/s (read+write) |
| --- | --- | --- |
| 1 MiB | 5.45 | 24.36 |
| 4 MiB | 29.88 | 65.70 |
| 8 MiB | 53.37 | **130.39** |
| 16 MiB | 80.98 | 117.04 |
| 64 MiB | 98.04 | 113.17 |
| 256 MiB | 104.39 | 110.72 |
| 1024 MiB | **105.48** | 107.07 |

* Peak read: **105.48 GB/s** (77.6 % of spec), stable across runs.
* Peak copy: **130.39 GB/s** (95.9 % of spec) at 8 MiB, i.e. L2-resident.
* DRAM-resident copy settles at ~107–113 GB/s, which is the number to use for
  expert-weight streaming in the roofline model.
* Small working sets are noisy run-to-run (1 MiB read ranged 5.4–6.5 GB/s
  across three runs); the ≥ 64 MiB rows are reproducible within ~2 %.

### Trellis decode and GEMV throughput

Latest run (`logs/moe-preflight/roofline-140v.txt`); the pre-K=3 build is
preserved as `roofline-140v-before-k3cb1.{json,txt}`.

| Case | K | Codebook | Triton GoW/s | ESIMD GoW/s |
| --- | --- | --- | --- | --- |
| routed gate/up 2048×512 | 3 | mcg | 7.1 | **12.3** (was unsupported) |
| routed down 512×2048 | 3 | mcg | 7.5 | **10.7** (was unsupported) |
| shared expert 2048×512 | 4 | mcg | 8.7 | 12.1 |
| shared expert 2048×512 | 4 | mul1 | 8.2 | 11.3 |

`GoW/s` = 10⁹ reconstructed weights per second. The ESIMD/Triton gap is
1.4–1.7× on the routed shapes; run-to-run spread on these micro-benchmarks is
±10–20 %, so treat the ratio, not the absolute figure, as the signal.

### Existing EXL3 linear at M=1 (routed shapes)

| Case | µs / GEMV | Effective GB/s | Path |
| --- | --- | --- | --- |
| gate/up 2048×512, K=3/mcg | **130.3** | 3.02 | ESIMD (was 311.0 / 1.26 Triton) |
| down 512×2048, K=3/mcg | **135.7** | 2.90 | ESIMD (was 277.0 / 1.42 Triton) |

Two readings matter for Phase 1/4:

* The new `(K=3, cb=1)` instantiation removes the 2.1–2.4× Triton penalty on
  the routed-expert shape.
* Even so, a single M=1 expert GEMV reaches only ~3 GB/s of trellis streaming,
  i.e. it is **latency/occupancy bound**, not bandwidth bound. A decode step
  issues 40 layers × top-k experts × 3 matrices, so the production path must
  keep many expert GEMVs concurrently resident; a per-expert sequential loop
  would be ~30× slower than the 105 GB/s roofline predicts. This is why P1-03
  (expert grouping) is a correctness-adjacent requirement, not an optimisation.


## 5. Routed-expert fast path status

**Resolved in `_C.so` sha256 `4003cc2b…` (r3, K=3 instantiation), carried into
`7642ee7f…` (r4, per-stream workspace) and the current `d8c1e7ac…`
(r6, `-DEXL3_ALL_CODEBOOKS`): `exl3_supported(K=3, cb=1) == true`.**

> **Artifact lineage / P4 finding.** A Phase 3 rebuild (`0380e452…`, r5) was
> produced with `EXL3_FLAGS` **unset**, which silently dropped
> `exl3_supported(4,1)` and `exl3_supported(6,1)` — the shared expert and all
> three GDN projections — forcing them onto the Triton `reconstruct+GEMM`
> fallback (measured: +2000 µs/token of `_reconstruct_kernel`/`gemm_kernel`,
> −33 % decode throughput). Phase 4 rebuilt as **r6 with
> `-DEXL3_ALL_CODEBOOKS`**, restoring ESIMD for `(4,1)`/`(6,1)` and reclaiming
> that loss. Gate 3 serving numbers were produced on r5; Gate 4 targets use r6+.
>
> **r7 (Gate 4, current): `b627e3a4…`, `-DEXL3_ALL_CODEBOOKS` + the P4-02
> grouped decode op** (`csrc/exl3_moe_gemv.sycl`, op
> `exl3_moe_grouped_gemv(Tensor, …, int x_unit_stride) -> Tensor`). All r6
> instantiations are carried forward; the new op is additionally templated on
> fp16/bf16 activations. Build log:
> `logs/moe-preflight/esimd-build-moegrouped-r7/`.
The build in use at the start of Gate 0 had ESIMD instantiations for
`(K=4, mcg)` and the `mul1` variants only, so all 30 720 routed expert modules
fell back to the K-generic Triton path at ≈ 1.3 GB/s effective.

`launch_reconstruct<3,1>` and `dispatch_mr<3,1>` are in
`csrc/exl3_ops.sycl`; `exl3_supported(3,1)` returns `true`; and
`exl3_small_m_max(3,1)` returns **8** because the routed format is
vector-only on this no-XMX part.

Acceptance evidence:

| Item | Artifact | Result |
| --- | --- | --- |
| Build | `logs/moe-preflight/esimd-build-k3cb1-r3/` | BUILD OK, load-validated |
| Build (per-stream workspace) | `logs/moe-preflight/esimd-build-reentrancy-r4/` | BUILD OK, `exl3_supported(3,1)=true`, `small_m_max(3,1)=8`; sha256 `7642ee7f…` |
| Build (ALL_CODEBOOKS, r6) | `logs/moe-preflight/esimd-build-allcb-r6/` | BUILD OK, restores `(4,1)`/`(6,1)` for shared + GDN linears; sha256 `d8c1e7ac…` |
| Build (P4-02 grouped op, **r7 current**) | `logs/moe-preflight/esimd-build-moegrouped-r7/` | BUILD OK, load-validated; sha256 `b627e3a4…` |
| Bit-exact reconstruction, T-1 | `parity-esimd-forced/exl3-backend-parity.{log,json}` | 18/18 modules `recon_diff=0` |
| Linear accuracy, T-2 (M ≤ 8) | same | worst max abs 1.2e-4 ≤ 0.08, cosine 1.000000 |
| M-bucket routing (9…128) | `logs/moe-preflight/k3-mr-sweep.{log,json}` | 32/32 pass; M > 8 → reconstruct+GEMM |
| Register pressure | `logs/moe-preflight/esimd-compile-lnl.{log,json}` | 0 spills, K=3 kernels present |

`requested_backend: esimd` in the parity JSON records that the run forced the
ESIMD dispatch (`EXL3_BACKEND=esimd` makes a missing library a hard error
instead of a silent Triton fallback).

**Remaining performance gap (not a correctness blocker).** The routed path is
correct for every M, but only ~3 GB/s of trellis streaming per M=1 GEMV
(§4), so Phase 1 must group expert calls rather than loop over experts
sequentially.

## 6. Roofline model used for Phase 4 targets

Per decoded token:

| Stream | Bytes | Basis |
| --- | --- | --- |
| Routed experts (8 of 256 × 40 layers) | ≈ 389 MB | 8 × 3 × 2048·512 × 0.386 B × 40 |
| Attention / GDN (dense) | ≈ 0.61 GiB | 30 GDN + 10 full-attn layers |
| LM head | ≈ 0.36 GiB | 248 320 × 2048 × 6 bit |
| Shared expert + router | ≈ 0.10 GiB | 40 layers |
| Embedding | row lookup only | not streamed in full |

Routed-expert traffic alone is ≈ 389 MB/token ⇒ **3.7 ms/token at 105 GB/s**.
Adding the dense streams gives ≈ 14 ms/token ⇒ a hard ceiling of **~70 tok/s**
for single-sequence decode with all weights DRAM-resident, before any kernel
inefficiency. This is the number Phase 4 is measured against; it also shows why
the K=3/MCG ESIMD instantiation is the single highest-value kernel task.

## 7. Runtime environment variables

At probe time none of `EXL3_VLLM_PATCHES`, `EXL3_BACKEND`, `EXL3_SMALL_M_MAX`,
`EXL3_INT8_PREFILL`, `EXL3_RECON_SLICE_N`,
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS`, `ZE_AFFINITY_MASK`,
`VLLM_LOGGING_LEVEL`, `OMP_NUM_THREADS`, or `HF_HOME` were set. Any later
experiment must record its overrides alongside its logs.

## 8. Evidence index

| Claim | Artifact |
| --- | --- |
| Host / software pins | `logs/moe-preflight/environment-moe.{txt,json}` |
| L0 limits and allocation ladder | `logs/moe-preflight/l0-capabilities{,-relaxed}.json`, `scripts/l0_caps.cpp` |
| Bandwidth / decode / GEMV roofline | `logs/moe-preflight/roofline-140v.{json,txt}`, `bench/roofline.py` (pre-K=3 baseline: `roofline-140v-before-k3cb1.*`) |
| ESIMD support matrix + parity | `logs/moe-preflight/parity-esimd-forced/exl3-backend-parity.{log,json}` (forced `EXL3_BACKEND=esimd`) |
| `(K=3, cb=1)` build | `logs/moe-preflight/esimd-build-k3cb1-r3/` (BUILD OK, sha256 in `artifact.sha256`) |
| Current build (per-stream workspace) | `logs/moe-preflight/esimd-build-reentrancy-r4/` (BUILD OK, sha256 `7642ee7f…`) |
| M-bucket dispatch 1…128 | `logs/moe-preflight/k3-mr-sweep.{log,json}`, `scripts/check_k3_mr_sweep.py` |
| Register/spill diagnostic | `logs/moe-preflight/esimd-compile-lnl.{log,json}`, `scripts/esimd_aot_diag.sh` |
| Checkpoint schema | `docs/checkpoint_schema.md`, `logs/moe-preflight/checkpoint-moe-audit.json` |
| GDN state path | `logs/moe-preflight/gdn-fixture-probe.{log,json}` |
| MoE dispatch seam (P1-01) | `logs/moe-preflight/test-dispatch.{log,json}`, `logs/moe-preflight/p1-01-dispatch.patch`, `tests/test_dispatch.py` |
| Expert registration + loader (P1-02) | `logs/moe-preflight/test-expert-loader.{log,json}`, `logs/moe-preflight/p1-02-expert-loader.patch`, `tests/test_expert_loader.py` |
| Packed expert descriptors (P1-03) | `logs/moe-preflight/test-expert-loader.{log,json}`, `logs/moe-preflight/p1-03-descriptors.patch`, `tests/test_expert_loader.py` |
| Packed-arena integrity (P1-04) | `logs/moe-preflight/test-expert-loader.{log,json}`, `logs/moe-preflight/p1-04-integrity.patch`, `tests/test_expert_loader.py` |
| Shard streaming + memory flush (P1-05) | `logs/moe-p1/shard-loading-memory.csv`, `logs/moe-p1/shard-streaming-audit.json`, `logs/moe-preflight/p1-05-streaming.patch`, `scripts/test_shard_streaming.py` |
| Bit-exact expert reconstruction (P1-06) | `logs/moe-p1/reconstruction-audit.{log,json}`, `logs/moe-preflight/p1-06-reconstruction.patch`, `tests/test_expert_reconstruction.py` |
| Single-expert MLP forward (P1-07) | `logs/moe-p1/single-expert-eval.{log,json}`, `logs/moe-preflight/p1-07-mlp.patch`, `tests/test_expert_mlp.py` |
| Batch sweep + M=0 + crossover (P1-08) | `logs/moe-p1/batch-sweep.{log,json}`, `logs/moe-p1/crossover-calibration.json`, `logs/moe-preflight/p1-08-batch-sweep.patch`, `tests/test_batch_sweep.py` |
| Reentrancy + workspace scratch (P1-09) | `logs/moe-p1/reentrancy-audit.{log,json}`, `logs/moe-p1/reentrancy-memory.csv`, `logs/moe-preflight/esimd-build-reentrancy-r4/`, `logs/moe-preflight/p1-09-reentrancy.patch`, `tests/test_reentrancy.py` |
| Dense regression (P1-10) | `logs/moe-p1/dense-regression-check.{log,json}`, `logs/moe-preflight/p1-10-dense-regression.patch`, `scripts/test_exl3_ops.py` (baseline: `logs/exl3-2bpw-20261007-124800/operator-results-allcb.json`) |
| Gate 1 full suite | `logs/moe-p1/gate1-suite.log`, `tasks_moe.md` §Gate 1 Status |
| Eager MoE dispatch + block (P2-01/04/07) | `logs/moe-p2/moe-block-eval.{log,json}`, `logs/moe-preflight/p2-phase2.patch`, `tests/test_moe_block.py` |
| Router alignment + weight point + edge cases (P2-02/03/06) | `logs/moe-p2/routing-eval.{log,json}`, `logs/moe-p2/edge-cases.log`, `tests/test_moe_routing.py` |
| Fixed / free routing (P2-05) | `logs/moe-p2/fixed-routing.{log,json}`, `logs/moe-p2/free-routing.{log,json}`, `tests/test_fixed_routing.py`, `tests/test_free_routing.py` |
| Intermediate-stage probe (P2-08) | `logs/moe-p2/intermediate-stage-validation.{json,log}`, `tests/test_intermediate_probe.py` |
| Full hybrid decoder layer (P2-09) | `logs/moe-p2/full-layer-eval.{log,json}`, `tests/test_hybrid_layer.py`, `tests/_dense_gdn_ref.py` |
| Phase 2 consolidated suite (P2-10) | `logs/moe-p2/phase2-summary.json`, `scripts/run_phase2_tests.py` |
| Dispatch profiler + host-sync audit (P4-01) | `bench/moe_profile.py`, `logs/moe-p4/moe-profile-{r5-baseline,r6-allcb,r6-p401}.json`, `logs/moe-p4/phase4-dispatch-summary.json` |
| Bandwidth / weight-writeout discipline (P4-05) | `logs/moe-p4/bandwidth-efficiency.log` |
| Dense regression on r6 | `logs/moe-p4/dense-regression-r6.{log,json}` (90/90 ALL PASS) |
| Grouped M=1 decode kernel (P4-02, **r7 current**) | `csrc/exl3_moe_gemv.sycl` (exl3xpu `306192e`), `tests/test_grouped_gemv.py` (3/3: fp16+bf16 bit-exact parity), build `logs/moe-preflight/esimd-build-moegrouped-r7/` (sha256 `b627e3a4…`) |
| Gate 4 profile: 0 syncs, 41 launches, 0.835 ms/token | `logs/moe-p4/moe-profile-r7-p402.json` |
| Gate 4 C1 server sweep (4.9–5.4× vs Gate 3) | `results.csv` rows at git `fae3379` (notes `Phase 4 grouped r7+P4-02 …`) |
| Gate 4 C4 PPL parity (bit-identical to Gate 3) | `logs/moe-p4/gate4-c4-ppl-parity.log`, `logs/moe-p3/ppl-results.json` |
| Dense regression on r7 (grouped op present) | `logs/moe-p4/dense-regression-r7.{log,json}` (91/91, identical case set to r6) |
| P4-07 matrix (18 cells, 18 OK) | `bench/bench_moe_envelope.py`, `results_moe.csv` |
| Gate 4 perf report | `docs/gate4_perf_report.md` |
| C5 multi-hour soak (in progress) | `bench/soak_moe.py`, `logs/moe-p4/soak-r7.jsonl` |
