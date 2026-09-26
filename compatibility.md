# Compatibility Investigation: Qwen3.6-35B-A3B (MXFP4) via vLLM XPU on Intel Arc 140V

Status legend: **verified** = evidence captured in this repo; **static** = read from
pinned upstream source/docs; **pending** = must be demonstrated by a running test.

---

## 1. Model artifact (Phase 1.1)

### 1.1.1 Candidate checkpoints surveyed

No *official* MXFP4 release exists for this model. Qwen publishes only
`Qwen/Qwen3.6-35B-A3B` (BF16, 26 shards, ~64.6 GB) and `Qwen/Qwen3.6-35B-A3B-FP8`.
All MXFP4 variants are third-party. Survey results (HF API, 2026-09-25):

| Repository | Format | Loadable by vLLM XPU path? | Size |
|---|---|---|---|
| `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` | safetensors, compressed-tensors `mxfp4-pack-quantized` | **yes** (see §2) | 23.11 GB |
| `prithivMLmods/Qwen3.6-27B-MXFP4` | compressed-tensors, but 27B **dense** model | yes, wrong model | 27.14 GB |
| `mlx-community/*-mxfp4`, `majentik/*`, `OsaurusAI/*` | MLX tensor format | no (MLX-only layouts) | ~19 GB |
| `tcclaviger/Qwen3.6-35B-A3B-MXFP416-MTP` | `quant_method: mxfp4_16`, IQ codebook | no (not MXFP4) | — |
| `LeaderboardModel1/*-AutoRound-MXFP4` | auto-round `quant_method` | no (auto-round path) | 22.28 GB |
| `FreedomAISVR/*`, `noctrex/*` (GGUF) | GGUF `MXFP4_MOE` | not via the MXFP4 XPU path (llama.cpp fallback) | ~20 GB |

### 1.1.2 Selected artifact — **verified**

- Repository: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`
- Pinned revision: `7eceff3a9f7e6f916c824d197266d86676bce695` (main, last modified 2026-04-17)
- Base model: `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Quantizer (per model card): `olka/qstream`, MXFP4 group size 32, symmetric, `scale_dtype: uint8` (E8M0)
- Saved locally: `logs/model/pahajokiconsulting-Qwen3.6-35B-A3B-MXFP4-{config,README}.json/md`,
  `logs/model/index.json`, `logs/model/*-size-breakdown.json`
- Custom model code: **none required** — no `auto_map`, no `custom_code` tag; standard
  transformers `Qwen3_5MoeForConditionalGeneration`. `--trust-remote-code` not needed.
- Download size: **23.11 GB (21.52 GiB)** across 26 safetensors shards. Disk free before
  download: 131 GB → sufficient (budget was 25 GB+).

### 1.1.3 Architecture identifier — **verified**

```
architectures : ["Qwen3_5MoeForConditionalGeneration"]
model_type     : "qwen3_5_moe"
quantization_config:
  quant_method : "compressed-tensors"
  format       : "mxfp4-pack-quantized"
  config_groups.group_0: targets=[Linear], weights={num_bits:4, type:float,
                                   strategy:group, group_size:32, symmetric:true}
  ignore: [lm_head, embed_tokens, all self_attn.*, all linear_attn/gdn.*,
           all shared_expert.*, *.mlp.gate, layernorms, entire model.visual.*,
           entire mtp.*]
```

So the checkpoint is **native MXFP4 for routed MoE experts only**; attention, GDN
(linear_attn), shared expert, vision tower and the MTP draft block stay BF16.
This is genuinely MXFP4 (E2M1 weights + E8M0 per-32 group scales), not a
differently-named INT4 format — confirmed by `format: mxfp4-pack-quantized`
matching `CompressionFormat.mxfp4_pack_quantized` in compressed-tensors 0.17.0
(`src/compressed_tensors/config/base.py:26`).

### 1.1.4 Exact on-disk / in-memory composition — **verified**

From safetensors headers (script: `scripts/inspect_shard_sizes.py`):

| Group | Tensors | GB | Share |
|---|---|---|---|
| routed MoE experts (MXFP4) | 61440 | 17.113 | 74.1% |
| MTP draft block (BF16) | 785 | 1.689 | 7.3% |
| embed_tokens (BF16) | 1 | 1.017 | 4.4% |
| lm_head (BF16) | 1 | 1.017 | 4.4% |
| vision tower (BF16) | 333 | 0.893 | 3.9% |
| self_attn (BF16) | 60 | 0.545 | 2.4% |
| linear_attn/GDN (BF16) | 420 | 0.539 | 2.3% |
| shared expert (BF16) | 160 | 0.252 | 1.1% |
| dense mlp/gates (MXFP4-ish) | 40 | 0.042 | 0.2% |
| norms | 81 | ~0 | 0% |
| **Total** | **63321** | **23.108** | 100% |

**Memory implication:** if the vision tower (0.89 GB) and the MTP draft block
(1.69 GB) are not instantiated, resident weights ≈ **20.5 GB**, leaving ≈ 8 GB of
the 32 GB package for OS + Python runtime + KV cache. If both are instantiated,
resident weights ≈ 23.1 GB and the KV budget drops to ≈ 5 GB.

---

## 2. vLLM loader and quantization path (Phase 1.2) — **static, pinned to v0.30.0**

Pinned stack target: **vLLM 0.30.0 `+xpu` wheel** from `https://wheels.vllm.ai/0.30.0/xpu`
(commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, released 2026-09-22).

| Question | Finding | Evidence |
|---|---|---|
| Accepted MXFP4 identifier | The checkpoint's `quant_method: compressed-tensors` + weight args (group, float, 4 bits, group_size 32, symmetric) are matched by `CompressedTensorsConfig._is_mxfp4()` → `CompressedTensorsW4A4Mxfp4()` | `vllm/.../compressed_tensors/compressed_tensors.py` `_is_mxfp4`, `_get_scheme_from_parts` |
| (Alternative) `--quantization mxfp4` | exists as a canonical name (`Mxfp4Config.get_name() == "mxfp4"`) but its base `get_quant_method` **falls back to `UnquantizedLinearMethod`** for plain linear layers — not what we want for this checkpoint | `vllm/.../layers/quantization/mxfp4.py` |
| Format string accepted? | `mxfp4-pack-quantized` is a registered `CompressionFormat` and a registered compressor in the exact version vLLM 0.30.0 pins (`compressed-tensors==0.17.0`) | `compressed_tensors/config/base.py:26`, `compressors/mxfp4/base.py:25` |
| Architecture registered? | `Qwen3_5MoeForConditionalGeneration` **and** text-only `Qwen3_5MoeForCausalLM` are both in the model registry | `vllm/.../models/registry.py:596-599, 204` |
| Architecture validated on XPU? | `Qwen/Qwen3.5-35B-A3B` (`Qwen3_5MoeForConditionalGeneration`) is listed as a **validated XPU model**, BF16/Online-FP8 | docs `models/hardware_supported_models/xpu.md` (page dated 2026-08-25) |
| Conversion required? | No — checkpoint schema (packed `weight_packed` uint8 + `weight_scale` uint8 E8M0, group 32) is exactly what `CompressedTensorsW4A4Mxfp4.create_weights()` allocates | `compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py` |
| Does the XPU platform allow this quantization? | **Yes** — `XPUPlatform.supported_quantization` explicitly lists both `"mxfp4"` and `"compressed-tensors"` | `vllm/platforms/xpu.py:113-130` |
| Capability gate (`get_min_capability()==80`)? | **No blocker** — `XPUPlatform.get_device_capability()` deliberately returns `None` ("capacity format differs from cuda's"), so `_check_scheme_supported()` short-circuits to `True` | `vllm/platforms/xpu.py:250-256`, `compressed_tensors.py:386-413` |
| Hybrid KV cache (required by this hybrid-attention model)? | `XPUPlatform.support_hybrid_kv_cache() == True` | `vllm/platforms/xpu.py:439-441` |

---

## 3. XPU kernel path (Phase 1.3) — **static analysis + runtime verification (§3.6)**

### 3.1 Pinned stack (per `vllm-xpu-kernels` README + vLLM XPU install docs)

| Component | Pinned version | Host state |
|---|---|---|
| Python | 3.12 (docs: "MUST" for the prebuilt wheel; wheel is now abi3 `cp38`, `requires-python >=3.9,<3.15`) | 3.12.14 installed |
| PyTorch | `torch==2.14.0+xpu` (per `vllm-xpu-kernels` build requirement) / vLLM 0.30.0 XPU wheel metadata | pending install |
| oneAPI | 2026.0 (kernels) — `pip intel-*-rt==2026.1.1` recommended for torch 2.14 XPU graphs | system oneAPI 2026.0.0 installed (pacman) |
| vLLM | `0.30.0+xpu` from `wheels.vllm.ai/0.30.0/xpu` | pending install |
| `vllm-xpu-kernels` | `0.1.15.4` (2026-09-22), abi3 wheel on PyPI | pending install |
| Driver stack | compute-runtime ≥ 26.18 recommended | `intel-compute-runtime 26.35.39758.10` ✅ |
| Triton | `triton==3.7.2+xpu` shim from `wheels.vllm.ai/xpu` | resolved by installer |

### 3.2 Does the kernel package advertise MXFP4?

**Yes — verified** from `vllm-xpu-kernels` README "Supported Kernels":
`Quantization | FP8, MxFP4 quantization and GEMM`, plus repo files
`csrc/quantization/fp4/mxfp4_quant.{cpp,h}`, `csrc/xpu/onednn/fp4_gemm_w4a4.h`
(oneDNN-backed FP4 GEMM), `tests/test_mxfp4_quant.py`, `tests/test_fp4_gemm_onednn.py`.

### 3.3 Is the MXFP4 path model-specific or general?

**General.** vLLM exposes MXFP4 on XPU at two layers, both platform-selected rather
than model-selected:

1. **Linear layers** — `_POSSIBLE_MXFP4_KERNELS[PlatformEnum.XPU] = [XPUMxFp4LinearKernel]`
   in `vllm/model_executor/kernels/linear/__init__.py`; the kernel calls
   `torch.ops._xpu_C.fp4_gemm(...)` after `xpu_mxfp4_quantize` on activations (W4A4,
   dynamic MXFP4 activations). `is_supported()` only requires `current_platform.is_xpu()`.
2. **MoE experts** — `CompressedTensorsW4A4Mxfp4MoEMethod.__init__()` has an explicit
   `elif current_platform.is_xpu(): mxfp4_backend = Mxfp4MoeBackend.XPU;
   experts_cls = XPUExpertsMxFp4` branch, logging
   `"Using XPUExpertsMxFp4 for MXFP4 MoE on XPU platform"`.

Both exist at tag **v0.30.0** (HTTP 200 on all three raw files), so the pinned wheel
contains them.

### 3.4 Xe2 / Arc 140V support

- vLLM's documented *validated* XPU hardware is **Intel Arc Pro B-Series only**
  (docs `models/hardware_supported_models/xpu.md`, 2026-08-25). The Arc 140V / Xe2
  iGPU is **not** in the validated list → this remains a project risk (unvalidated device),
  but it is *not* an architectural blocker:
- **`XPUExperts` explicitly gates on Xe2/Xe3** — `xpu_moe.py:68` runs
  `torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()` and raises
  `NotImplementedError("XPUExperts is only supported on Intel Xe2/Xe3 GPUs")` otherwise.
  The Arc 140V **is** Xe2 (`xe` driver bound to PCI `8086:64A0`, Lunar Lake Xe2-LPG),
  so the MoE path is designed for this architecture class. `is_xe2_arch()` output will be
  recorded by `scripts/mxfp4_kernel_check.py`.
- `XPUMxFp4LinearKernel.is_supported()` performs **no** device/arch check beyond
  `is_xpu()`; whether oneDNN FP4 GEMM actually runs on Xe2 is a runtime question —
  answered by `scripts/mxfp4_kernel_check.py` (registration **and** numeric agreement
  with a dequantized float32 reference).
- Host driver stack (compute-runtime 26.35, level-zero 1.32, xe driver, `/dev/dri/renderD128`
  world-readable) is newer than the documented minimum (26.18).

### 3.5 Prior art and known issues on this exact hardware — **verified (external reports)**

Searches found reports on the **same machine** (MSI Claw 8 AI+, Core Ultra 7 258V, Arc 140V,
32 GB LPDDR5x), all from `intel/llm-scaler`:

| Report | Date | Content | Relevance to us |
|---|---|---|---|
| `intel/llm-scaler#335` (closed) "Add Lunar Lake (32GB) support: Xe2 compatibility fixes and benchmark results" | 2026-03-27 | Full Lunar Lake support for "vLLM SYCL" on Nobara 43. Benchmarks Qwen3-8B INT4: 17.6 tok/s single-request, 90 tok/s batched. **Key findings: "Qwen3 (standard attention) works on Xe2; Qwen3.5 (fla/linear attention) does not (Triton XPU broken)"; "AWQ/GPTQ compressed-tensors fail due to CUDA-only Marlin kernels"** | Our model is Qwen3.6 with **hybrid GDN/linear attention** → this is the single biggest open risk. Mitigating evidence: `vllm-xpu-kernels` now ships native SYCL **"GDN attention, XE2 attention variants"**, and `triton-xpu` ships with the pinned torch 2.14 XPU stack. Must be verified in Phases 5–6. |
| `intel/llm-scaler#340` (open) "VLLM_SKIP_PROFILE_RUN patch for Lunar Lake iGPU profile_run() hang" | 2026-04-01 | **`profile_run()` hangs indefinitely at startup for MoE models on this iGPU**, blocking server start (models verified to hang: gpt-oss-20b MXFP4, GLM-4.7-flash). Fix = local patch adding `VLLM_SKIP_PROFILE_RUN=1` (skips the dummy forward pass, estimates peak = `memory_allocated() × 1.2`) | Directly applicable: our model is MoE. **The env var does NOT exist in vLLM 0.30.0** (grep of `vllm/envs.py` and `vllm/v1/worker/*` in the installed wheel returns nothing), and `determine_available_memory()` in `vllm/v1/worker/gpu_worker.py:528-578` still calls `model_runner.profile_run()` even when `kv_cache_memory_bytes` is set. Fallback plan: hard-timeout the first load, then apply an equivalent local patch (saved under `scripts/patches/`) if the hang reproduces. |
| `intel/llm-scaler#342` (open) "Lunar Lake Xe2 iGPU compatibility report and benchmarks" | 2026-04-01 | Benchmarks **gpt-oss-20b (MXFP4)**, Qwen3.5-4B INT4, Qwen3-8B INT4, Qwen3.5-9B BF16/FP8/INT4 on this device with vLLM 0.14.1.dev0 XPU | **MXFP4 on Arc 140V has been demonstrated to work** (gpt-oss-20b is a native `quant_method: mxfp4` model). Positive evidence for Gate 4/5. |
| `vllm-project/vllm-xpu-kernels#92` | — | "only panther lake and lunar lake runs on Xe2" (maintainer discussion re: architecture support) | Lunar Lake Xe2 is a target the kernels explicitly consider. |

Caveat: those reports used the **Intel SYCL fork / vLLM 0.14.x (March–April 2026)**, whereas we
pin **upstream vLLM 0.30.0 + `vllm-xpu-kernels` 0.1.15.4 (September 2026)** — five months of
kernel work later, with native SYCL GDN attention and a shipped `triton-xpu`. The reports are
treated as *risk indicators to test*, not as current facts.

### 3.6 Runtime verification on the Arc 140V — **verified (2026-09-25, Gate 4)**

`logs/mxfp4-kernel-check.txt` (13/13 checks PASS, script `scripts/mxfp4_kernel_check.py`):

| Check | Result |
|---|---|
| `torch.ops.vllm.xpu_mxfp4_quantize` registered | PASS (needs `import vllm._xpu_ops`; registered by `xpu_ops.register_ops_once()` at `vllm/_xpu_ops.py`) |
| `torch.ops._xpu_C.fp4_gemm` registered | PASS (`vllm-xpu-kernels 0.1.14.1`) |
| `torch.ops._xpu_C.is_xe2_arch()` | PASS → **`True`** on Arc 140V |
| MXFP4 quantize output layout | `float4_e2m1fn_x2 [M, K/2]` + `float8_e8m0fnu [M, K/32]` |
| `fp4_gemm` executes on Xe2 | PASS → `[64, 384]` bf16 |
| `fp4_gemm` vs. independently dequantized f32 reference | **rel. err 1.17e-3** (bf16 output rounding) |
| `XPUExpertsMxFp4._supports_current_device()` | PASS |
| `XPUExpertsMxFp4` accepts `(kMxfp4Static, kMxfp4Dynamic)` / `(kMxfp4Static, None)` | PASS / PASS |
| bf16 matmul, finite | PASS |

**Finding worth recording:** the FP4 GEMM only produces correct results when the weight operand is
the **non-contiguous `.t()` view** produced by `XPUMxFp4LinearKernel.process_weights_after_loading()`.
Materialising the transpose with `.contiguous()` changes the result (rel. err 1.46). Isolated in
`scripts/debug_fp4_gemm.py` (candidate A = vLLM's layout → 1.2e-3; candidate B = `.t().contiguous()`
→ 1.46; candidate E → `RuntimeError('B_scale must be contiguous for fp4 matmul')`). Our decoder was
validated against `vllm_xpu_kernels.moe_utils.dequant_mxfp4` (max diff 0.0, `scripts/debug_fp4_decode.py`),
so the reference is trustworthy.

**Still unverified at this stage:** the MoE GEMM (`XpuFusedMoe` / `XPUExpertsMxFp4.apply`) can only be
exercised end-to-end with real expert weights → Phase 5/6. And the hybrid GDN/linear attention path
(`csrc/xpu/sycl/deepseek_*`, GDN attention) is untested until the model loads.

### 3.7 Blockers found while loading the model (Phase 5) and their fixes

| # | Symptom | Root cause | Fix (all encoded in the wrappers) |
|---|---|---|---|
| 1 | `ValueError: Free memory on device xpu:0 (11.01/28.58 GiB) … less than desired GPU memory utilization` | The `xe` driver parks freed device memory in `GPUReclaim` (12.8 GiB after a crashed run) and Level-Zero does not count it as free | `scripts/reclaim_gpu_cache.py` pre-flight drains the pool (host pressure → kernel shrinkers); **must run before every attempt** because each crash re-parks ~12 GiB |
| 2 | `icpx … fatal error: 'level_zero/ze_api.h' file not found` during the Triton-XPU JIT | Host had `level-zero-loader` only; `level-zero-headers` not installed (and installing needs sudo) | `scripts/setup_level_zero_headers.sh` extracts the Arch package into `tools/sysroot`; `CPATH` exported |
| 3 | `ValueError: Device does not support device USM allocations` from `_xpu_C.gdn_attention` and `_xpu_C.cutlass_grouped_gemm_interface` | **Device selection**: Mesa `rusticl` and Intel NEO both register as `opencl:gpu`; the failing kernels are SYCL and can land on rusticl, which lacks device USM. (`sycl-ls` itself lists `usm_device_allocations` for the Level-Zero Arc device, so capability was never the issue.) | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` — after this the same code paths run (verified: attempt 9 passed, attempt 8 with otherwise identical flags failed) |
| 4 | `source setvars.sh` silently killed the wrapper | `set -u` + oneAPI's `env/vars.sh` referencing `OCL_ICD_FILENAMES` | `set +u` around the source, stderr captured to `logs/oneapi-setvars.log` |
| 5 | Reference-only workaround for the Xe2 grouped GEMM | `vllm_xpu_kernels` ships `VLLM_XPU_FUSED_MOE_USE_REF=1` (slow Python reference MoE) | Used to isolate blocker 3 from the MoE kernel itself |

**Also disproven:** the reported Lunar-Lake `profile_run()` hang (`intel/llm-scaler#340`) did **not**
reproduce on vLLM 0.30.0 — profiling completed in seconds in every attempt that got that far.

**Also confirmed working:** the hybrid GDN/linear-attention path executes on Xe2 (the March-2026
"Qwen3.5 fla broken" report does not apply to this stack), Flash Attention v2 selects, and
`--language-model-only` drops the vision tower (`All limits … set to 0, running in text-only mode`).

---

## 4. Known gaps / risks

1. **Not a validated device.** Arc Pro B-Series is the only validated XPU hardware;
   Arc 140V is an efficiency-class Xe2 iGPU — **verified via `clinfo`: 64 compute units
   (8 Xe cores) @ 1950 MHz** — sharing 32 GB LPDDR5x with the CPU.
2. **Unified memory budget.** 20.5–23.1 GB of weights on a 32 GB package. Any OOM will
   appear as *host swap*, not GPU OOM — must be monitored explicitly (Phase 8).
   **Measured:** `total = 28.58 GiB`; text-only resident weights = **19.24 GiB**
   (`Model loading took 19.24 GiB memory`); the model has 10 full-attention layers
   (`full_attention_interval: 4`) × 2 KV heads × 256 head dim ≈ **20 KB/token**, so KV is
   cheap. The binding constraint is vLLM's startup check `free ≥ util × total`:
   with a desktop session running, Level-Zero reported 22.03 GiB free → `util ≤ 0.77`.
2b. **`GPUReclaim` hides memory from Level-Zero (new, measured).** `/proc/meminfo` on the
   `xe` driver exposes `GPUActive`/`GPUReclaim`; after any large GPU run the driver parks the
   freed memory in `GPUReclaim` and Level-Zero's free query excludes it. A crashed run left
   12.8 GiB parked → next start saw only 11.01 GiB free → startup rejected despite the memory
   being allocatable. Host memory pressure drains the pool (and it stays drained):
   `MemAvailable 15.52 → 27.06 GiB`, `GPUReclaim 12.38 → 0.00 GiB`.
   Mitigation: `scripts/reclaim_gpu_cache.py` runs as a pre-flight in both wrappers.
   **This is the single most important environment gotcha found so far** — without it every
   run after the first one fails before loading.
3. ~~**`get_min_capability() == 80`** declared by `Mxfp4Config`/`CompressedTensorsW4A4Mxfp4`.~~
   **Resolved (static):** `XPUPlatform.get_device_capability()` returns `None` by design, so
   every `_check_scheme_supported(80)` short-circuits to `True`. No CUDA-capability gate applies
   on XPU.
4. **Attention/GDN layers stay BF16** — the XPU attention kernel set is compiled from
   presets (`chunk_prefill`/`paged_decode`); Qwen is in the *default* preset, so no
   rebuild should be needed. Confirm no "kernel tuple not compiled" error in Phase 5.
5. **Vision tower + MTP block** add 2.58 GB. Prefer text-only operation for the first load
   test: `--language-model-only` marks tower components as skippable when all
   `--limit-mm-per-prompt` counts are zero (`vllm/config/multimodal.py:91,483`,
   `models/interfaces.py:345-346`), and MTP weights are unused without
   `--num-speculative-tokens`. Fallback: `--hf-overrides` to `Qwen3_5MoeForCausalLM`, which
   maps `model.language_model.` → `model.` and drops `mtp.`/`visual.` (`qwen3_5.py:321`).
6. Third-party quant provenance: produced with `olka/qstream`, validated by its author
   on **ROCm/RDNA4**, not on Intel. Numerical equivalence to BF16 must be judged in Phase 6.
7. **Hybrid GDN/linear attention on Xe2 is unproven** — a March-2026 report on this exact
   device found Qwen3.5 (fla/linear attention) non-functional due to "Triton XPU broken"
   (§3.5). This is now the highest-probability failure mode; if it reproduces, the model
   cannot run as-is on this stack and Gate 5/6 fail with a *missing operator* classification.
8. **`profile_run()` startup hang on this iGPU for MoE models** (§3.5, `intel/llm-scaler#340`).
   No upstream env-var escape hatch exists in 0.30.0 → plan a hard timeout plus an equivalent
   local patch if it reproduces.

---

## 5. Gate 1 decision

| Gate-1 requirement | State |
|---|---|
| Exact checkpoint + revision identified | ✅ verified (§1.1.2) |
| Accepted quantization identifier identified | ✅ static (§2) — `compressed-tensors` / `mxfp4-pack-quantized` |
| Architecture loader identified & registered | ✅ static (§2) — registry.py + XPU validated-model list |
| Required kernel path identified | ✅ static (§3.2–3.3) + **runtime-verified (§3.6)** — `fp4_gemm` correct on Xe2, `XPUExpertsMxFp4` supported |

**Gate 1: GO (static)** — every item is identified and traceable to pinned upstream
source. The remaining unknowns (does `fp4_gemm` actually run on Xe2? does the
checkpoint schema load without mismatch? does 20.5 GB fit without swap?) are
isolated to minimal tests in Phases 3–5, which is exactly the "HOLD → minimal
loader/import test" condition upgraded to a runnable experiment.
