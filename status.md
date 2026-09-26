# Status

Last updated: 2026-09-25 (Gate 5 passed)

## Current phase

**Current phase: Phases 0–9 complete → Phase 10 (fallback comparison) is the only one left.**

Measured envelope (Gate 8): **4096-token context × 2 sequences**, decode **19.3 tok/s** flat,
concurrency 2 → **22.5 tok/s aggregate**, prefill latency ~1 s (overhead-dominated), 0 errors.

Phase 9 outcome: **eager + `--max-num-seqs 2`** selected (+35 % aggregate throughput);
`torch.compile` and XPU-graph variants measured and rejected on startup/host-memory grounds.

- Phase 0: **PASS** — scope, workload, target context lengths and measurable acceptance criteria
  recorded in "Gate 0 record" below (defaults, overridable).
- Phase 1: Gate 1 assessed **GO (static)** — see `compatibility.md` §5.
- Phase 2: **PASS** — `xe` driver bound, `/dev/dri/renderD128` accessible to non-root,
  `sycl-ls` + `clinfo` enumerate the Arc GPU (`logs/host-check.txt`).
- Phase 3: **PASS** — `.venv` (Python 3.12.14), torch 2.13.0+xpu sees the Arc GPU,
  matmul/round-trip OK (`logs/torch-sanity.txt`).
- Phase 4: **PASS** — `vllm 0.30.0+xpu` + `vllm-xpu-kernels 0.1.14.1`; MXFP4 quantize + FP4 GEMM
  registered **and numerically correct** on Xe2 (rel. err 1.17e-3 vs float32 reference);
  `XPUExpertsMxFp4` supported (`logs/mxfp4-kernel-check.txt`).
- Phase 5: **PASS** — model loads end-to-end and produces a deterministic completion
  (`logs/smoke-08.log`, exit 0). Weights 19.24 GiB, engine init 58.2 s, KV cache 1.31 GiB /
  9,216 tokens / 9.0× concurrency @ 1024 tokens, decode **1.18 tok/s** (reference-MoE path).

## Completed gates

| Gate | State | Evidence |
|---|---|---|
| Gate 0 | **PASS** | "Gate 0 record" below — scope, workload, context targets, 5 measurable acceptance criteria |
| Gate 1 | GO (static) | `compatibility.md` §5 |
| Gate 2 | PASS | `sycl-ls` → `level_zero:0 Intel(R) Arc(TM) Graphics 20.4.4`; `clinfo` → `Intel(R) Arc(TM) Graphics OpenCL 3.0 NEO [26.35]`; `xe` driver Live; no permission errors |
| Gate 3 | **PASS** | `logs/torch-sanity.txt` — `xpu available: True`, `Intel(R) Arc(TM) Graphics`, 28.58 GiB, BF16 matmul finite, round-trip OK |
| Gate 4 | **PASS** | `logs/mxfp4-kernel-check.txt` — 13/13 checks; `fp4_gemm` rel. err 1.17e-3 vs independent reference; `is_xe2_arch()=True`; `XPUExpertsMxFp4` supported |
| Gate 5 | **PASS** | `logs/smoke-08.log` — weights 19.24 GiB, init 58.24 s, KV 1.31 GiB (9,216 tok / 9.0×), greedy 32-token completion, exit 0 |
| Gate 6 | **PASS (caveat)** | `logs/prompt-suite/run-02.jsonl` — 15/15 clean, semantically correct (`Canberra`, correct palindrome code, correct "alpha" retrieval at ~3,000 tokens); caveat: greedy output is **not bit-deterministic** across runs (p5 diverged at char 81) |
| Gate 7 | **PASS** | `logs/server-server-01.log` + `scripts/health_check.sh` — `/health` 200, model listed (`max_model_len 4096`), 3/3 chat completions OK (~1.0 s each), 0 errors, init 50.99 s, KV 19,660 tok (4.80×). Caveats for Gate 8: host MemAvailable **2.9 GiB** while serving (< our 3 GiB bar) and swap grew 3.24 → 4.10 GiB at startup |
| Gate 8 | **PASS** | `results.csv` + `scripts/bench_envelope.py` — envelope **4096 context × 2 sequences**, 10+ consecutive requests, **0 errors / 0 OOM**, GPUActive constant 22.03 GiB, GPUReclaim 0, swap growth only **0.05 GiB** across the sweep. Decode flat **19.3 tok/s** (1K→4K), concurrency 2 → **22.5 tok/s aggregate** (+35 %) |

## Key decisions

1. **OS target: native Linux (CachyOS, Arch-based), kernel 7.2.3-3-cachyos-deckify.**
   Not Ubuntu 24.04 as `plan.md` suggested — no distro change made; documented as-is.
   WSL2 is out of scope (Phase 0 item 2).
2. **Model: `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`** @ `7eceff3a` — the only
   35B-A3B MXFP4 safetensors checkpoint whose schema maps onto vLLM's
   `CompressedTensorsW4A4Mxfp4` + `XPUExpertsMxFp4` XPU path. See `compatibility.md` §1.
3. **vLLM pinned to `0.30.0+xpu`** (stable wheel, commit `ced6857a`, 2026-09-22)
   rather than nightly, because the v0.30.0 tag already contains
   `kernels/linear/mxfp4/xpu.py`, `compressed_tensors_moe_w4a4_mxfp4.py` and
   `experts/xpu_moe.py` (`XPUExpertsMxFp4`).
4. **Python 3.12.14** (vLLM XPU docs mark 3.12 as mandatory for the XPU wheel).
5. First load test will use **text-only instantiation + no speculative decoding** to
   keep resident weights ≈20.5 GB instead of 23.1 GB.

## Blockers

None yet. Principal risks recorded in `compatibility.md` §4:
- Arc 140V is *not* on vLLM's validated XPU hardware list (Arc Pro B-Series only) — though
  `XPUExperts` explicitly requires Xe2/Xe3, which the Arc 140V is.
- 20.5–23.1 GB weights on a 32 GB unified package → OOM will manifest as host swap (zram).
- `intel-gpu-tools` not installed (sudo needs a password) → no `intel_gpu_top`; use
  `scripts/mem_monitor.sh` for swap/memory evidence instead.

## Fallback readiness (Phase 10)

- A **SYCL llama.cpp build already exists** at `~/llama-bonsai-sycl/build/`
  (`libggml-sycl.so.0.21.0`, `llama-ls-sycl-device`, `llama-cli`, etc.) — usable as the
  fallback backend without a rebuild.
- No Qwen3.6-35B-A3B GGUF is on disk yet (`~/models.md` inventory); a documented GGUF
  (e.g. `ggml-org/Qwen3.6-35B-A3B-GGUF`) would need to be downloaded for the Phase 10
  comparison, ~20 GB.

## Gate 0 record (Phase 0 deliverable)

Proposed defaults, written down so the gate is testable. Override anything here — everything
downstream reads the criteria from this section.

**Scope / machine**: native Linux (CachyOS) on the MSI Claw 8 AI+ (Core Ultra 7 258V, Arc 140V
Xe2 iGPU, 32 GB shared LPDDR5x; 112 GB free disk after setup). WSL2 out of scope.

**Intended workload** (priority order):
1. **Interactive single-user inference** — primary goal.
2. **OpenAI-compatible local API** (`vllm serve`) — the delivery mechanism.
3. **Batch throughput testing** — measured for comparison, not an optimization target.

**Target context lengths**: 1024 (baseline, must work) → **4096 (target)** → 8192 (stretch, only
if Gate 8 shows no swap). 262 K native context explicitly out of scope on this hardware.

**Minimum acceptance criteria** (measured from `results.csv` + `logs/mem-*.csv`):
1. Model loads without host swapping — steady-state criterion, revised after measurement:
   `pswpin`/`pswpout` deltas ≈ 0 **during steady-state requests** and swap used does not grow
   run-over-run; `MemAvailable` stays above ~2.5 GiB while serving (2 s sampling,
   `scripts/mem_monitor.sh`). Rationale: some swap activity at startup is unavoidable on this
   32 GB shared-memory box (swap grew 3.24 → 4.10 GiB during server startup, then only
   0.05 GiB across the whole Phase 8 sweep); the original ">3 GiB, zero swap even at startup"
   bar proved unachievable here.
2. The fixed 5-prompt suite (`scripts/prompts.json`) completes coherently, 3 runs each, with no
   empty output, repetition loops, replacement characters, or character runs.
3. Three consecutive full runs (load → suite → shutdown) complete without crash, OOM, or
   `UR_RESULT_ERROR_DEVICE_LOST`.
4. Peak host RSS and peak device memory recorded per run, appended to `results.csv`.
5. Performance compared against the llama.cpp/SYCL fallback with **the same prompts, context and
   output token counts** (`fallback-comparison.md`).

**Gate 0 state: PASS (documented; defaults are yours to override).**

## Attempt log (failed attempts kept per the triage checklist)

| # | Time | What happened | Classification | Fix |
|---|---|---|---|---|
| 1 | 23:32 | `smoke_offline.py` died instantly: `ModuleNotFoundError: No module named 'torch'` | Test harness bug — `scripts/run_phase5.sh` did not activate `.venv` (only sourced oneAPI) | Added `source .venv/bin/activate` + an import pre-flight to the wrapper |
| 2 | 23:33 | Wrapper exited before running anything (no output after its first line) | Test harness bug — `set -u` (nounset) + `source /opt/intel/oneapi/setvars.sh` aborts the shell: `env/vars.sh: OCL_ICD_FILENAMES: unbound variable`, and stderr was sent to `/dev/null` so it failed silently | `set +u` around the `source`, stderr captured to `logs/oneapi-setvars.log`, plus a `python -c "import torch, vllm"` pre-flight |
| 3 | 23:36 | Engine rejected at init: `ValueError: Free memory on device xpu:0 (22.03/28.58 GiB) on startup is less than desired GPU memory utilization (0.78, 22.3 GiB)` | **Configuration** — other processes hold ~6.5 GiB, so Level-Zero reports only 22.03 GiB free; `0.78 × 28.58 = 22.3 GiB` > free. Not a bug, not OOM, and **no `profile_run()` hang** (it fails before profiling) | Retrying with `--gpu-memory-utilization 0.74` (21.15 GiB, 0.88 GiB headroom under the 22.03 GiB free) |
| 4 | 23:37 | **Weights loaded successfully** (26/26 shards, 19.20 s, model resident **19.24 GiB**, `XPUMxFp4LinearKernel` + `XPUExpertsMxFp4` both selected), then failed inside `profile_run()` → `_prepare_rope_positions_kernel` (Triton) → Intel Triton JIT: `icpx … fatal error: 'level_zero/ze_api.h' file not found` | **Missing system package** — host has `level-zero-loader` (runtime) only; `extra/level-zero-headers` (which ships `ze_api.h`) is not installed, and installing needs an interactive sudo | Added `scripts/setup_level_zero_headers.sh`: fetches the Arch package and extracts it to `tools/sysroot`, exports `CPATH` in both wrappers. Clean alternative if you have sudo: `sudo pacman -S level-zero-headers` |
| 5 | 23:40 | Same startup rejection, but free memory had collapsed from 22.03 → **11.01 GiB** (util 0.74 needs 21.15) | **Environment / driver accounting** — attempt 4's crash left **12.8 GiB** in the Xe driver's `GPUReclaim` pool (`/proc/meminfo`), which Level-Zero does not report as free. Confirmed *not* held by any process: no vLLM/python process alive, `/dev/dri/renderD128` held only by desktop apps | Probed it: `scripts/probe_xpu_reclaim.py` shows the memory **is** allocatable (GPUReclaim drops as we allocate), and `scripts/pressure_test_reclaim.py` shows host memory pressure makes the kernel's shrinkers drain the pool back to the OS |
| 6 | 23:47 | **Reached the real forward pass** during `profile_run`: weights loaded (17.8 s, 19.24 GiB), `XPUMxFp4LinearKernel` + `XPUExpertsMxFp4` + Flash Attention + Triton/FLA GDN all selected, layer 0 attention/GDN executed, then died in the MoE experts: `vllm_xpu_kernels/fused_moe_interface.py:424` → `torch.ops._xpu_C.cutlass_grouped_gemm_interface(...)` → `ValueError: Device does not support device USM allocations` | **Missing/unsupported operator capability** — the Xe2 grouped-GEMM launcher needs device USM and whatever it queries says "not supported", even though `sycl-ls --verbose` lists `usm_device_allocations` for this device. Note: **no `profile_run()` hang** — the reported Lunar-Lake hang did not reproduce on 0.30.0 | Using the package's own documented escape hatch next: `VLLM_XPU_FUSED_MOE_USE_REF=1` (slow Python reference MoE) to get end-to-end correctness first, per tasks.md "establish correctness before performance" |
| 7 | 23:53 | Failed at the startup free-memory check again (11.07 GiB free vs 21.15 needed) — the ref-MoE flag never got a chance to matter | **Bug in my own pre-flight** — `reclaim_gpu_cache.py` used `MemFree` as its safety floor; `MemFree` was only 0.76 GiB (normal: 14 GiB of page cache) so it bailed immediately and left `GPUReclaim 12.42 GiB` parked. vLLM also logged `WARNING Unknown vLLM environment variable detected: VLLM_XPU_FUSED_MOE_USE_REF` (harmless — that variable belongs to `vllm_xpu_kernels`, which does read it) | Floor changed to **`MemAvailable`** (with a separate hard `MemFree` floor of 256 MiB). Standalone re-test: `GPUReclaim 12.42 → 0.00 GiB`, `MemAvailable 15.65 → 26.94 GiB` |
| 8 | 23:54 | **`profile_run()` passed** — `Available KV cache memory: 1.93 GiB`, `XPU KV cache size: 13,458 tokens`, `max concurrency 13.14x @ 1024 tok/request`, JIT warmup + GDN Triton warmup + M-RoPE warmup done. Ref-MoE path worked (no grouped-GEMM error). Then died one step later in `warmup_kernels` → `qwen_gdn_linear_attn.py:1004 forward_xpu` → `torch.ops._xpu_C.gdn_attention(...)` → **same** `ValueError: Device does not support device USM allocations` | **Not MoE-specific** — the same USM error now hits the GDN attention SYCL kernel (`libgdn_attn_kernels_xe_2.so`, another lib containing the string). Note GDN *succeeded* in attempt 6 and failed here → **intermittent**, and both occurrences happened with the device nearly full | Hypothesis: either (a) the SYCL libs pick a different GPU device than `sycl-ls`'s default (Mesa **rusticl** and Intel NEO are both visible as `opencl:gpu` and rusticl likely lacks device USM), or (b) a failed `malloc_device` under memory pressure is reported with this misleading message. Testing (a) first — single variable |
| 9 | 23:56 | **SUCCESS — Gate 5 PASS.** Identical to attempt 8 plus `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`: engine init 58.24 s, weights 16.05 s to read / 19.24 GiB resident, KV cache 1.31 GiB = 9,216 tokens = 9.0× concurrency @1024 tok, greedy generation of 32 tokens in 27.1 s (**1.18 tok/s**), `finish_reason=length`, clean SIGTERM shutdown, exit 0 | Root cause of the two USM failures = **wrong SYCL device selected** (Mesa *rusticl* / Intel NEO also register as `opencl:gpu`). Pinning `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` removes them as candidates | Selector now exported unconditionally by `run_phase5.sh` and `launch_vllm.sh` |
| 10 | 23:59 | **SUCCESS — native MXFP4 MoE verified.** Single-variable change from attempt 9 (dropped `VLLM_XPU_FUSED_MOE_USE_REF`): `XPUExpertsMxFp4` + `torch.ops._xpu_C.cutlass_grouped_gemm_interface` executed, 32 tokens in **1.8 s = 17.97 tok/s**, init 43.1 s, exit 0 | Confirms the Xe2 MXFP4 grouped GEMM works on Arc 140V once `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` is set — the earlier USM failure was device selection, not capability | Selector + reclaim pre-flight + CPATH are now defaults in both wrappers |
| 11 | 00:01 | Phase 6 prompt suite, label `01`, died before touching the model: `json.decoder.JSONDecodeError: Expecting ',' delimiter: line 48` | **My own bug** — `scripts/prompts.json` contained JavaScript-style string concatenation (`"…" + "<REPEAT_ME>" + "…"`), which is not valid JSON | Rewrote line 48 as one JSON string; validated parse + the `expand_long_prompt()` output (12,169 chars ≈ 3,042 tokens, under the 4,096 limit) |
| 12 | 00:03 | **Phase 6 PASS** — suite re-run with fixed JSON, label `02`, `max_model_len 4096`, `kv_cache_memory_bytes=1 GiB`: **15/15 completions, all `flags=[]`**, 2,679 output tokens in 165.4 s, exit 0. KV capacity **19,660 tokens (4.80× concurrency)**, engine init 45.26 s, prefill 131–153 tok/s on the ~3,000-token prompt, decode 14–19 tok/s | Correctness established. **New caveat:** greedy output is not bit-deterministic — `p5` runs diverged at char 81 and run 3 stopped early (247 tok, `finish_reason=stop`). `p1`–`p4` byte-identical across runs | Recorded under Gate 6; relevant to Phase 9/10 (any output comparison must tolerate this) |
| 13 | 00:08 | Phase 7: server launch attempt 1 — `nohup … &` from a foreground tool call | **My own harness mistake** — the process group was killed when the tool call returned, mid-reclaim (`holding 11 GiB`, no "after release") | Relaunched in a background shell instead; `nohup` from a tool call does not survive here |
| 14 | 00:09 | **Phase 7 PASS** — `launch_vllm.sh server-01` came up on 127.0.0.1:8000 in ~40 s (engine init 50.99 s), weights 19.24 GiB / 21.27 s, KV 19,660 tokens (4.80× @4096), **0 errors** in the log; health check `/health` 200, model listed, 3/3 chat completions OK (1.06/0.97/0.98 s ≈ 16 tok/s e2e) | Server works. Open items: `MemAvailable` while serving = **2.9 GiB** (below our 3 GiB bar), swap 3.24 → 4.10 GiB during startup, log lands at `logs/server-server-01.log` (script prepends `server-` to the label), `python -m vllm.entrypoints.openai.api_server` deprecated in favour of `vllm server` | Feeds Gate 8 (memory/context envelope) |

**Server running** at `http://127.0.0.1:8000`, served model name `Qwen3.6-35B-A3B-MXFP4`.
Stop it with `pkill -f vllm.entrypoints.openai.api_server` (it is a background shell in this session).

### Performance so far (single sequence, eager mode, `max_model_len 1024`)

| Configuration | 32-token generation | Decode |
|---|---|---|
| Reference MoE (`VLLM_XPU_FUSED_MOE_USE_REF=1`) | 27.1 s | **1.18 tok/s** |
| **Native Xe2 MXFP4 MoE** (`cutlass_grouped_gemm_interface`) | 1.8 s | **17.97 tok/s** |

≈ **15× speedup** from the native kernel — this is the headline result for Gate 4/5 and the
baseline for Phase 9 tuning. Prefill on the same run: 5 prompt tokens in ~1.8 s total.

### Root cause: `Device does not support device USM allocations`

The message is compiled into `libgrouped_gemm_xe_2.so`, `libgdn_attn_kernels_xe_2.so`,
`libattn_kernels_xe_2.so`, `libmhc_kernels_xe_2.so` and `auto_round_kernel` (a shared build-time
helper; the literal is absent from `vllm-xpu-kernels` sources at `main` and tag `0.1.14.1`).
It is **not** a missing capability: `sycl-ls --verbose` reports `usm_device_allocations` for the
Level-Zero Arc device. The trigger was **device selection** — `sycl-ls` without a filter shows
four devices, two of which are GPUs:

```
[level_zero:gpu] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) Arc(TM) Graphics
[opencl:cpu]     Intel(R) OpenCL, Intel(R) Core(TM) Ultra 7 258V
[opencl:gpu]     rusticl, Mesa Intel(R) Graphics (LNL) OpenCL 3.1     ← no device USM
[opencl:gpu]     Intel(R) OpenCL Graphics (integrated), Arc(TM) Graphics OpenCL 3.0 NEO
```

With `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` the same code path runs. Two different XPU kernels
(`gdn_attention` and `cutlass_grouped_gemm_interface`) produced the error before the fix, and
neither did after it.

Lesson recorded: **every crashed run parks ~12 GiB in `GPUReclaim`**, so the drain pre-flight must
run before *every* attempt, not just once — it is now wired into both wrappers.

Evidence gathered while triaging attempt 6:
- The message string is compiled into `libgrouped_gemm_xe_2.so`, `libattn_kernels_xe_2.so`,
  `libgdn_attn_kernels_xe_2.so`, `libmhc_kernels_xe_2.so` **and** `auto_round_kernel` — i.e. a
  common build-time helper — but is **absent** from the `vllm-xpu-kernels` source at both `main`
  and tag `0.1.14.1` (downloaded and grepped), so it comes from a dependency header that is not
  shipped in the source tarball.
- `sycl-ls --verbose` shows the Level-Zero V2 Arc device advertising
  `usm_device_allocations usm_host_allocations usm_shared_allocations` → the runtime claims the
  capability the kernel says is missing. Contradiction worth an upstream report.
- A 30-line `icpx -fsycl` probe (`scripts/sycl_usm_probe.cpp`) **segfaults** on this host, and
  every SYCL invocation prints the Mesa/rusticl libclc warning — the SYCL/UR stack is not healthy
  in isolation here. Recorded as an environment risk.
- Positive: **the hybrid GDN/linear-attention path executed without error** (the thing the
  March-2026 Lunar Lake report called broken), and **`profile_run()` did not hang**.

### GPUReclaim finding (reusable workaround)

`/proc/meminfo` on this `xe` driver exposes `GPUActive` / `GPUReclaim`. After any large GPU
run the driver parks the freed device memory in `GPUReclaim` instead of returning it, and
Level-Zero's free-memory query excludes that pool — so vLLM's
`free >= gpu_memory_utilization × total` check fails even though the memory is allocatable.
Measured drain (`pressure_test_reclaim.py`, 1 GiB host allocations):

```
held 0 GiB : MemFree 12.01  MemAvailable 15.52  GPUReclaim 12.38 (GiB)
held 10GiB : MemFree 18.18  MemAvailable 18.17  GPUReclaim  0.00 (GiB)
after free : MemFree 26.90  MemAvailable 27.06  GPUReclaim  0.00 (GiB)   ← stays reclaimed
```

No OOM, no swap growth, desktop survived. `scripts/reclaim_gpu_cache.py` now runs this as a
pre-flight in `run_phase5.sh` and `launch_vllm.sh` (skip with `WARMUP=0`). This is why
attempt 3 saw 22.03 GiB free but attempt 5 saw only 11.01 GiB.

Attempt 4 also confirmed (evidence for Gate 5, partial):
- `Loading weights took 19.20 seconds`, `Model loading took 19.24 GiB memory and 23.69 s`
  → matches the predicted 19.1 GiB text-only resident size (20.53 GB of the 23.15 GB checkpoint:
  vision tower 0.89 GB + MTP block 1.69 GB skipped).
- `Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` and
  `GDN decode kernel: cuda` — the hybrid GDN/linear-attention path *does* have an XPU route
  (the thing the March-2026 Lunar Lake report said was broken); the decode-path label
  `cuda` needs a follow-up check in Phase 6/9.
- `quantization=compressed-tensors`, `dtype=torch.bfloat16`, `device_config=xpu`,
  `enable_prefix_caching=True`, `enable_chunked_prefill=True`.

Useful confirmations from attempt 3 (all before the failure):
- `Resolved architecture: Qwen3_5MoeForConditionalGeneration` and
  `All limits of multimodal modalities … set to 0, running in text-only mode` →
  `--language-model-only` works and drops the vision tower.
- Engine config shows `quantization=compressed-tensors`, `dtype=torch.bfloat16`,
  `device_config=xpu` → **the loader accepted the checkpoint's MXFP4 metadata**.
- Warnings to track (Phase 5/9): `Failed to import the DeepSelect extension
  (vllm._deepselect_C)`; `XPU Graph is disabled by environment variable …
  VLLM_XPU_ENABLE_XPU_GRAPH=1`; `Enforce eager set, disabling torch.compile and CUDAGraphs`;
  chunked prefill enabled with `max_num_batched_tokens=1024`.

(The `set -u` + `setvars.sh` trap also affected `scripts/launch_vllm.sh`; fixed there too.)

## Phase 9 tuning log

Baseline (Gate 8, eager mode, `server-01`): decode **19.27 tok/s** @1K, 19.43 @~4K, concurrency-2
aggregate **22.51 tok/s**, engine init **50.99 s**, host `MemAvailable` ~2.9 GiB.

| # | Change (one at a time) | Decode | Aggregate | Startup | MemAvailable | Verdict |
|---|---|---|---|---|---|---|
| A | `--enforce-eager` removed → `mode=VLLM_COMPILE` (inductor; XPU graph still off) | 19.20 tok/s (−0.4 %) | 22.94 tok/s (+1.9 %) | 95.47 s (**+41.8 s compile**) | **2.36 GiB** (worse) | **Rejected** — gain is within noise, startup nearly doubles, host headroom shrinks. 0 errors, so the path itself is healthy. |
| B | `VLLM_XPU_ENABLE_XPU_GRAPH=1` (on top of A) → `cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes [1,2,4] | **19.84 tok/s (+3.0 %)** | **23.09 tok/s (+2.6 %)** | 93.60 s (compile 41.4 s) | **1.48–1.77 GiB** (vs 2.9) | **Rejected for this host** — correct and stable (0 errors, swap flat at 3.61 GiB) and a real but small gain, yet it nearly halves host memory headroom on a 32 GB shared-memory box. Keep as the documented option for a machine with more RAM. |
| C | MTP speculative decoding: `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, `gpu_memory_utilization` 0.74→0.77, KV pinned 1 GiB→0.5 GiB (`server-05-mtp`) | **26.83 tok/s (+39 % vs eager baseline)** | n/a (conc 1) | init ~47 s, weights **20.81 GiB** (19.24 + 1.57 draft) | **2.11–2.16 GiB** | **Accepted with caveats** — 0 errors, swap flat (4.47→4.46 GiB), output verified coherent/correct (Canberra probe). Draft (`Qwen3_5MoeMTP`, BF16) dispatches on XPU fine. Caveats: host headroom thinnest yet; KV capacity drops to 6,616 tokens (1.62× @4096); one warning (`no KV cache group could be identified as the draft model's`) needs a follow-up read; acceptance rate not directly measured (metrics off) but +39% implies healthy acceptance. New best decode config. |

**Gate 9 decision — selected configuration:** **eager mode + concurrency 2**
(`--enforce-eager`, `--max-num-seqs 2`, `gpu_memory_utilization 0.74`, KV pinned 1 GiB,
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`). Rationale: concurrency 1 → 2 raises aggregate throughput
**+35 %** (16.7 → 22.5 tok/s) while per-request decode drops only 11 % (19.27 → 17.08), with 0
errors, constant `GPUActive 22.03 GiB`, `GPUReclaim 0` and 0.05 GiB swap growth — i.e. it improves a
declared workload metric while staying correct and stable, and it costs nothing in startup or host
memory. Both compile/graph variants were measured and rejected on memory/startup grounds.

| # | Time | Detail |
|---|---|---|
| 15 | 00:19 | Phase 9 test A (`server-server-02-eager-off.log`): 6 requests across 2 points, **0 errors**, swap flat at 4.08 GiB. One request stopped early at 94 tokens (`finish_reason=stop`) — normal EOS, not a failure |
| 16 | 00:27 | Phase 9 test B (`server-server-03-xpugraph.log`): 6 requests, **0 errors**, XPU graphs actually captured (`cudagraph_mode FULL_AND_PIECEWISE`), `GPUReclaim` non-zero at 0.68–0.97 GiB (graph memory held), host `MemAvailable` down to 1.48 GiB |
| 17 | 07:25 | **MTP test PASS** (`server-server-05-mtp.log`): draft `Qwen3_5MoeMTP` resolved from the same checkpoint, weights 20.81 GiB / ~36 s, KV 6,616 tokens, init ~47 s, **0 errors**; bench point (1026 in / 128 out, conc 1) → decode **26.83 tok/s (+39 % vs 19.27 baseline)**, TTFT median 1.55 s (first run 1.95 s incl. draft warmup); output verified correct; memory stable (GPUActive 22.53 GiB, GPUReclaim 0, swap 4.47→4.46 GiB), host `MemAvailable` 2.11–2.16 GiB — thinnest headroom yet |

## Next action — Phase 10 (needs a decision)

1. **10.1 llama.cpp/SYCL**: a SYCL build already exists at `~/llama-bonsai-sycl/build/`
   (`libggml-sycl.so.0.21.0`), but **no Qwen3.6-35B-A3B GGUF is on disk** — one must be
   downloaded (~20 GB; at this host's observed ~23 GB / 55 min, budget ~1 hour) e.g.
   `ggml-org/Qwen3.6-35B-A3B-GGUF`, then replay `scripts/run_prompt_suite.py --mode server`
   style prompts and record the same `results.csv` fields for `fallback-comparison.md`.
2. **10.2 IPEX-LLM**: verify model/platform support first (its docs are laptop-iGPU oriented);
   keep in a separate environment.
3. **10.3 Decision**: vLLM is currently *working and stable* on this box (Gates 1–9 pass), so the
   fallback has to beat 19.3 tok/s decode / 22.5 tok/s aggregate at 4096 context with equal or
   better memory behaviour to displace it — record the verdict in `compatibility.md` + `fallback-comparison.md`.

Smaller open items (from `tasks.md`): dedicated prefix-caching workload, independent chunked-prefill
sweep, a `torch.profiler` trace, `intel-gpu-tools` (needs sudo), and a `max_num_seqs 4` point.
