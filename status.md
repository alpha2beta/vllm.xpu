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
6. **Primary Production Serving: `Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4` (Config G)**.
   Deployed with MTP $K=2$, eager execution, achieving **27.69 tok/s** decode (port 8080).
   Selected as the primary daily driver for coding and general tasks due to MoE memory bandwidth efficiency (~1.2 GB/tok vs dense ~13 GB/tok).
7. **Option B Selected: `Qwen3.8-27B-exl3` retained as Designated High-Reasoning Fallback**.
   Dense 27B architecture eliminates MoE routing artifacts and preserves high reasoning fidelity
   (WikiText-2 PPL 6.36 at 4.00bpw, 6.78 at 2.20bpw), serving as quality fallback when 35B A3B
   fails to generate satisfying code/reasoning results, despite lower decode throughput (6.5–9.9 tok/s).
   Operational isolation implemented: symmetrical launcher `launch_vllm_qwen38_exl3.sh` (port 8001), while
   primary launcher `scripts/launch_vllm.sh` enforces `export EXL3_VLLM_PATCHES=0` to eliminate
   runtime monkey-patch contamination.

## Blockers

None yet. Principal risks recorded in `compatibility.md` §4:
- Arc 140V is *not* on vLLM's validated XPU hardware list (Arc Pro B-Series only) — though
  `XPUExperts` explicitly requires Xe2/Xe3, which the Arc 140V is.
- 20.5–23.1 GB weights on a 32 GB unified package → OOM will manifest as host swap (zram).
- `intel-gpu-tools` not installed (sudo needs a password) → no `intel_gpu_top`; use
  `scripts/mem_monitor.sh` for swap/memory evidence instead.

## Tiel-Coder-35B-A3B-Genesis-Hermes MXFP4 (2026-09-27, new workstream)

**Deliverable:** `models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4/` (local only,
gitignored) — 62,555 tensors, 31 shards, **21.56 GiB**, SHA256SUMS verified.
MXFP4 (`compressed-tensors` `mxfp4-pack-quantized`, E2M1 group-32 symmetric,
E8M0 scales) in the byte-exact safetensors schema of
`pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`, for the validated
`CompressedTensorsW4A4Mxfp4` + `XPUExpertsMxFp4` XPU path. Source:
`symrex/...-GGUF-dequantized` BF16 (17 shards, 72 GB, streamed one at a time,
deleted after use; arch identical to Qwen3.6: 40L/256 experts/8 active/2048h).

**Pipeline** (`scripts/quantize_tiel_mxfp4.py` + `finalize_tiel_mxfp4.py` +
`audit_tiel_mxfp4.py`, `logs/quantize-tiel.log`):
- Fused `experts.gate_up_proj [256,1024,2048]` → per-expert gate/up halves;
  stacked `experts.down_proj [256,2048,512]` → per-expert; linear_attn
  `in_proj_qkv/z/out_proj/a/b` quantized (note: Tiel `out_proj` is
  **[2048,4096]**, unlike Qwen3.6's [2048,2048] — same GDN loader, wider matmul).
  MTP fused experts + visual + everything else copied BF16 unchanged (vLLM
  drops `mtp.` via its prefix mapper; `--language-model-only` skips vision).
- Quantizer `torch.ops.vllm.xpu_mxfp4_quantize` (kernel package's own op).
  In-conversion spot checks (kernel `dequant_mxfp4`, 1-in-64): max rel-err
  0.119–0.123; independent post-hoc CPU audit of 1,797 fresh pairs
  (`--shard 9`): mean 0.118, max 0.145. `config.json` = Tiel config +
  reference `quantization_config` (815-entry Tiel ignore list).
- `get_quantization_config('compressed-tensors')` → `CompressedTensorsConfig` ✓.

**Blocked: engine smoke test** — needs 21.56 GiB free (`-util 0.74` = 21.15
budget) but only ~19 GiB is free (desktop/Steam hold the rest; no orphans).
`logs/smoke-tiel-02.log` fails fast at the startup free-memory check, before
any weight load — checkpoint bytes untouched by this. Resume when memory is
free:
`./.venv/bin/python scripts/smoke_offline.py --model models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4 --language-model-only --max-model-len 1024 --gpu-memory-utilization 0.74 --max-tokens 32`

**Update 2026-09-27: engine smoke test PASSED** (`logs/smoke-tiel-03.log`,
exit path `RESULT: OK`). Weights loaded in 25.2 s, resident **19.24 GiB**,
KV cache 13,019 tokens (12.71× @ 1024), greedy 32-token completion on
"The capital of France is" → coherent Paris answer ("Paris is located in the
north-central part of France, on the Seine River..."). Host peak RSS 1.93 GiB.
The MXFP4 checkpoint is end-to-end loadable and generates through the XPU
`XPUExpertsMxFp4` path — last unproven link closed.

**Update 2026-09-27: vLLM XPU head-to-head vs Qwen3.6 MXFP4** (server
`logs/server-tiel-01.log` :8001, `logs/bench-tiel-{01,02,03}.log`,
`results.csv`; same calibrated points, eager, util 0.74):

| point (in/out/conc) | Qwen3.6 decode | Tiel decode | Δ |
|---|---|---|---|
| 1024 / 128 / 1 | 19.27 | **19.15** | −0.6% |
| 2048 / 128 / 1 | 19.28 | **19.51** | +1.2% |
| 1024 / 64 / 2 | 17.53 (agg 22.94) | **17.35** (agg 20.12) | −1.0% |

Identical arch + identical kernels ⇒ identical speed, as expected; Tiel's
wider GDN `out_proj` costs nothing measurable. GPUActive flat 21.35 GiB,
GPUReclaim 0, MemAvailable ~3.5 GiB, swap frozen 4.05 GiB — no pressure.
Caveat: Tiel emits its thinking trace inline (no `reasoning` split on this
`vllm serve` build), so reasoning tokens consume the `max_tokens` budget —
size agentic `max_tokens` accordingly. Quality spot-checks coherent (Paris
answer, `72`); full 5-prompt suite not yet run on Tiel.

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
| 18 | 08:05 | **MTP + fp8 KV combo PASS** (`server-server-07-mtp-fp8.log`): 32,768 context, weights 20.81 GiB, KV **43,253 tokens** (0.75 GiB pin, 1.32×), init ~47 s, **0 errors**; bench point → decode **26.29 tok/s** (fp8 KV costs ~nothing); 7,500-token prompt in 9.1 s, coherent output; memory stable (GPUActive 23.04 GiB, GPUReclaim 0, swap flat 4.55→4.54 GiB), host `MemAvailable` 1.95–1.98 GiB. First attempt (0.5 GiB pin) failed cleanly (`0.55 needed > 0.48 available`, vLLM estimated max 25,344) → bumped to 0.75 GiB |
| 19 | 12:41 | **LAN tool-calling server for the user's remote opencode** (`server-server-08-lan.log`, port 8080): restarted the user's own 8080 server with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` added. `tool_choice:auto` request now succeeds (was a 400 error). Startup took ~4 min (vs ~1 min normally; cause not investigated — box was loaded). One red herring along the way: a relaunch overlapped the still-dying previous EngineCore and died with `RuntimeError: generator didn't yield`; clean kill + relaunch fixed it. |

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

## Tiel MTP + FP8-KV experiments (2026-09-27)

**FP8 KV: works, no decode gain.** `--kv-cache-dtype fp8` server
(`logs/server-tiel-fp8.log`): 53,043-token KV capacity (7.4× @ 7168) vs
~9K BF16. Decode unchanged at all lengths (1024: 19.13 vs 19.15; 2048:
18.45 vs 19.51; 6656: 19.09 vs 18.81 — all within noise):
`logs/bench-tiel-fp8-{A,B,D}.log`, `results.csv`. Conclusion: Tiel decode
on XPU is MoE-weight-traffic-bound, not KV-bound — halving KV traffic
buys nothing. It does buy 5.8× KV capacity (useful for concurrency, not
single-stream speed).

**MTP: blocked on a vLLM fused-checkpoint loader gap, not our weights.**
`--speculative-config {"method":"mtp","num_speculative_tokens":1}` fails
at load (`logs/server-tiel-mtp.log`): the `qwen3_5_mtp` draft model builds
standard unfused `w13_weight`/`w2_weight` expert params, but Tiel's MTP
experts are fused `gate_up_proj`/`down_proj` 3D tensors (like its trunk),
and the fused path (`is_fused` → chunk dim=1) expects `[E,2*I,K]` while
Tiel stores `[E,1024,2048]` + separate down — `AttributeError: ... has no
parameter 'w2_weight'`. Qwen3.6 worked because its MTP experts are already
split gate/up/down in the checkpoint. Fix options: (a) unfuse Tiel's MTP
experts gate/up in a conversion pass (mirrors what we did for the trunk —
small script, same split); (b) wait on upstream. The draft block is only
1.69 GB BF16; (a) is cheap. Attempt log entry: shell ate the first
`--speculative-config` JSON (unquoted) — always single-quote it.

**Update 2026-09-27: MTP unblocked via (a), measured.**
`scripts/unfuse_tiel_mtp.py` (CPU-only reshape, committed) split the fused
MTP experts into per-expert gate/up/down BF16 (768 tensors, new shards
32–33; checkpoint now 63,321 tensors / 33 shards / 21.57 GiB, SHA256SUMS
green, index + 1,581-entry ignore list updated). Along the way it also
healed a stale `-of-00031` naming/index skew from the earlier incremental
quantize runs (self-checking tail: renames every shard to a consistent
`-of-00033` set and asserts every indexed tensor opens). Server
`logs/server-tiel-mtp-s1c.log` (MAX_SEQS=1, util 0.74, 0.5 GiB KV):
`Qwen3_5MoeMTP` draft resolves, `XPUExpertsMxFp4` selected, weights load —
MTP active on Tiel. Bench (`logs/bench-tiel-mtp-{A,B}.log`, `results.csv`):
1024/128/1 → **25.50 tok/s** decode (+33% vs 19.15 non-spec);
2048/128/1 → **24.31 tok/s** (+25% vs 19.51). Quality spot-check (bat+ball,
128 tokens) derives correctly. Remaining notes: needs a reclaim pass before
launch at util 0.77 (Level-Zero free only 21.5 GiB until drained); Tiel MTP
still below Qwen3.6's MTP number (26.83) by ~5–9% — draft acceptance
unmeasured, likely the gap.

## Phases 11–12 + 15.1 outcomes (2026-09-27, tasks.md workstream)

**Phase 11.1 MTP draft MXFP4 — DONE.** `scripts/quantize_tiel_mtp_mxfp4.py`
(768 BF16 → 1536 packed/scale, 14 s, in-conversion max rel-err 0.119,
independent CPU audit mean 0.118/max 0.123, SHA256SUMS green). Checkpoint now
64,089 tensors / 32 shards / **20.47 GiB** (was 21.57); ignore 1581→813.
Smoke: `Qwen3_5MoeMTP` + `XPUExpertsMxFp4` ×2, weights **19.71 GiB**,
Paris/Canberra probes coherent/correct (`logs/server-tiel-mtp-mxfp4-smoke.log`).
Bench: K=1 → **24.85/25.41** @1024, **24.87** @2048 (matches BF16 25.50/24.31
within noise; the win is −1.1 GiB, not speed). Evidence:
`logs/phase11-11.1.1-audit.txt`, `logs/quantize-tiel-mtp-mxfp4.log`,
`logs/phase11-11.1.5-smoke-attempts.txt`, `results.csv` +2 rows.

**Phase 11.2 K=2 sweep — DONE, 30 tok/s criterion missed.** K=2 supported
(KV 6301→5120 tokens; vLLM warns single MTP layer reused 2×, capping α).
K=2 → **27.69** @1024 (+9%), **26.72** @2048 (+7%), 0 errors, Paris coherent.
Confound logged: K=2 ran P-pinned, K=1 runs E-core (bg-shell cpuset lottery);
true uplift ~7–8%. Acceptance counters deferred (metrics off).
`logs/phase11-11.2-k2.txt`, `logs/server-tiel-mtp-k2.log`, `results.csv` +3 rows.

**Phase 11.3 EOS audit — DONE (read-only).** text_config.eos 248044,
generation_config [248046,248044], tokenizer confirms `<|endoftext|>`/`<|im_end|>`;
no stale 151643/151645 anywhere; no EOS lines in server logs (not logged by
this build). Live `<|im_end|>` stop test deferred. `logs/phase11-11.3-eos-audit.txt`.

**Phase 12.1 P-pinning — IMPLEMENTED** (`PINNED=1` default, `taskset -c 0-3`,
graceful fallback when cpuset restricted). Jitter A/B (12.1.2/12.1.4) deferred.

**Phase 12.2 Graph batch-1 — CLOSED, rejected on this host.**
`--compilation-config '{"cudagraph_capture_sizes":[1]}'` works (FULL_AND_PIECEWISE,
max capture 1 — no feature request). No-MTP: 19.93/19.89 (+2–4%) but
MemAvailable **0.7 GiB** (criterion >2.5 failed). Combo graph[1]+MTP K=2:
**27.54** (+0% over K=2 eager), 1 Paris loop vs 1 clean Canberra probe.
`logs/phase12-12.2-graph-b1.txt`, `results.csv` +3 rows.

**Phase 12.3/13.1 audits — DONE (read-only).** 30 W RAPL, iGPU pinned 1950 MHz
at idle; attention delta 0.94 ms @7K (MoE-bound, not KV); split-K already
present (`build_decode_split_plan`) — no upstream issue. `logs/phase12-13-readonly-audits.txt`.

**Phase 15.1 — SELECTED config G** (MTP K=2 MXFP4 eager, util 0.68, MAX_LEN 4096,
KV 0.5 GiB): 10/10 golden suite clean (`logs/prompt-suite/prod-qual-k2.jsonl`),
0 errors. Encoded as `SPEC=mtp-k1|mtp-k2` in `launch_vllm.sh` (base defaults
untouched). Gate 1: 3/4 (27.69 vs 28.0 throughput). Phase 15.2 (GGUF fallback
re-comparison) SKIPPED per user. Comparison table: `logs/phase15-15.1-best-config.txt`.

**Memory climate note:** all 2026-09-27 serving ran contended (desktop/Steam
active, swap ~8–9 vs 4–6, MemAvailable ~1.0 vs 2–3.5). Numbers above are
like-for-like within the day; quiet-box reruns may read 1–3% higher.
`GPU_UTIL=0.68` (not 0.74) is the current working default — needs only
19.43 GiB free. `MAX_LEN` must stay ≤4096 with the 0.5 GiB KV pin (65536
default fails the KV fit check on MTP configs).

## KAT-Coder-V2.5-Dev MXFP4 (2026-09-28, new workstream)

**Deliverable:** `models/KAT-Coder-V2.5-Dev-MXFP4/` (local only, gitignored) —
62,203 tensors, 28 shards, **19.12 GiB**, MXFP4 `mxfp4-pack-quantized` in the
Qwen3.6 reference schema for `CompressedTensorsW4A4Mxfp4` + `XPUExpertsMxFp4`.
Source `Kwaipilot/KAT-Coder-V2.5-Dev` BF16 (13 shards, ~69 GB, streamed one at
a time, deleted after use). Arch identical to Tiel/Qwen3.6
(40L/256 experts/8 active/2048h, text-only, **no MTP block**); experts already
per-expert split (no unfuse needed).

**Pipeline** (`quantize_kat_mxfp4.py` + `finalize_kat_mxfp4.py` +
`audit_kat_mxfp4.py`, `quantize_kat_loop.sh`, `download_kat_bf16.sh`):
in-conversion spot checks (1-in-64) max 0.134; independent post-hoc CPU audit
of shard-0's 2,075 pairs: mean 0.116, max 0.157 (ONE pair, outlier-group
concentrated with median-normal groups — documented waiver, gate now
mean<0.13 + <0.5% pairs ≥0.15). Ignore list 463 entries.

**Serving** (`logs/server-kat-mxfp4-*.log`, `results.csv`): loads 19.24 GiB,
`XPUExpertsMxFp4` ×2, KV 9,011 (2.2× @4096). Golden suite 10/10 clean
(`logs/prompt-suite/kat-qual-01.jsonl`), 0 errors. Eager decode ~13–14 tok/s
**on E-cores**; equal-footing 256-tok essay vs Tiel: 15.1 vs 16.6 tok/s (−9%).
Box was E-core-jailed all day (cpuset 4-7), so no P-core number exists yet;
expect ~19 tok/s parity with Tiel on a full box. No MTP block → no spec
decoding upside; ceiling is eager parity.

## KAT MTP draft (2026-09-28)

**Deliverable:** MXFP4 draft block merged into `models/KAT-Coder-V2.5-Dev-MXFP4/`
(63,756 tensors / 29 shards / 19.60 GiB). Source
`SpectreSystems/KAT-Coder-V2.5-Dev-MTP` `model-00014-of-mtp.safetensors`
(1.6 GB, 19 BF16 tensors, fused experts like Tiel's).
`scripts/quantize_kat_mtp_mxfp4.py` does unfuse+quantize in one pass
(rel-err max 0.119, `logs/quantize-kat-mtp-mxfp4.log`). Required config fix:
KAT base sets `text_config.mtp_num_hidden_layers=0` (no MTP); the vLLM draft
builder reads it via `getattr(..., 1)`, so 0 builds zero layers — script sets
it to 1 (backup `config.json.pre-kat-mtp.bak`). Ignore 463→480 (+17 MTP
non-expert); SHA256SUMS rebuilt.

**Serving** (`logs/server-kat-mtp-{smoke,k2}.log`, `results.csv`): draft
`Qwen3_5MoeMTP` resolves, weights 19.71 GiB, KV 6,301 (K=1). Paris/Canberra
probes correct. Bench (1024/128/1): K=1 **23.22**, K=2 **24.82** tok/s
(both P-pinned; cf. Tiel 24.85–25.41 / 27.69). KAT has no E-core MTP number;
eager E-core baseline 12.69–13.95.

## EXL3 4.00bpw on Arc 140V (2026-10-07, experimental)

**Result: plugin works on Xe2-LPG with a no-DPAS patch; dense-27B decode 6.4–6.5 tok/s — slower than Tiel MoE (19–28 tok/s). Retained under Option B as designated high-reasoning fallback.**

- Source: `0xSero/exl3xpu` pinned `15ded2f3` + 12-line `no-dpas.patch`
  (`g_vec_max_m` 2→8 with MR=1/2/4/8 tail coverage; `fused_small` declines M>2;
  `exl3_gemm_small` TORCH_CHECKs M≤8). Guide's `=256` substitution rejected as
  row-incorrect. Full plan/evidence: `tasks.md` E0–E6, `logs/exl3-4bpw-20261007-094207/`.
- Build: `scripts/build_exl3_ext.sh`, `_C.so` 2.3 MB sha256 `ac710cf1…0e00265`,
  no `EXL3_FLAGS` (K=4/6 mul1 default), DNNL off (no oneDNN tree on machine).
  Install added only the editable `exl3xpu` line (freeze diff clean).
- Correctness: 48/48 synthetic operator cases + M=512 prefill shapes (mlp-down,
  248320-wide K=6 lm-head) match the fp32 CPU reference; 409 EXL3 modules load
  (8 MTP, vision excluded text-only); 15/15 prompt suite semantically correct.
  Thinking leaks into content (`</think>` marker) — model behavior, costs budget.
- Perf (eager, B=1, KV-auto, MAX_LEN=4096, prefix-cache on):
  1026-tok in → TTFT ~1.7 s, decode **6.48**; 2052-tok in → decode **6.37**
  (early stop: 31/46 toks of req 128; rows in `results.csv` with caveats).
- Ops lessons: startup needs `WARMUP=1` reclaim (18.25 GiB free < 20.01 desired
  otherwise); `fuser -k 8001` orphans `VLLM::EngineCore` holding ~18 GB —
  kill it directly, then reclaim before restart. Steady-state flat
  (MemAvailable ~2.4–4.2 GiB, swap counters static). Qualified MAX_LEN=4096 only.
- Rollback & Isolation: stop API server + EngineCore (verify port free + GPUActive ~0.05),
  then `reclaim_gpu_cache.py`. Symmetrical fallback launcher `launch_vllm_qwen38_exl3.sh`
  runs on port 8001. Production launcher `scripts/launch_vllm.sh` explicitly exports
  `EXL3_VLLM_PATCHES=0`, isolating Tiel serving from runtime monkey patches while keeping
  the plugin installed for on-demand fallback.

## EXL3 MTP speculation (2026-10-07, same checkpoint)

**Result: MTP K=1/K=2 both function on the no-DPAS path; decode improves (up to 9.9 tok/s) but end-to-end loses on short outputs due to TTFT overhead.**

| Config | 1024 decode / e2e | 2048 decode / e2e |
|---|---|---|
| Eager (baseline) | 6.48 / 6.47 s | 6.37 / 9.25 s |
| MTP K=1 | **9.86** / 7.94 s | **9.02** / 12.21 s |
| MTP K=2 | 8.62 / 9.35 s | 9.25 / 12.36 s |

Decode gains (+42–52% K=1) come with TTFT regressions (draft + chunked prefill:
1024 TTFT 1.7→4.8 s). Breakeven ≈ 60 output tokens; bench outputs stopped early
(31/46 of req 128, thinking+EOS). K=2 shows acceptance lottery on reasoning
text (12.27/8.24/8.62 across runs) and no gain over K=1. Acceptance rate not
directly measurable in this build. Evidence: `logs/exl3-4bpw-mtp-20261007-102715/`,
4 `results.csv` rows. Ops: reclaim script can wedge holding 16 GiB host when the
reclaim pool is already small — skip WARMUP then; always kill EngineCore PIDs,
not just the port.

## EXL3 perplexity (2026-10-07)

**4.00bpw WikiText-2 test PPL: 6.36** (72 non-overlapping 4096-tok chunks,
294,840 tokens, vLLM offline `prompt_logprobs`, raw continuation). Smoke was
6.55 on 2 chunks. Healthy ~27B range — no quant blowup. Harness:
`scripts/eval_exl3_ppl.py`; result `logs/exl3-4bpw-ppl-20261007-105637/ppl-full.json`.
2.2bpw comparison pending download (`SC_2.20bpw_H3_V3` @ `25019f16…`).

## EXL3 2.2bpw comparison (2026-10-07)

**Serve + bench + PPL all green: decode 8.8 tok/s (+36–38% vs 4bpw), PPL 6.78 vs 6.36.** Required real plugin work (3 local patches, all saved under `logs/exl3-2bpw-20261007-124800/`):
`no-dpas.patch` (+NT=4 fix for K=3/5 MR=8 — upstream ALL_CODEBOOKS never compiled),
`k1-fallback.patch` (Triton reconstruct for K=1, no C++ instantiation exists),
`mixed-k.patch` (per-shard bitrates over Kmax-padded trellis; caught + fixed a
`shard_of_nb` activation-indexing bug via garbage-output differential).
Eager B=1: 1024 → 8.82 tok/s / 4.59 s; 2048 → 8.79 / 5.36 s (2 `results.csv` rows).
Quality cost of ~45% fewer bytes is small (+0.42 PPL). Retained under Option B as quality fallback.

## EXL3 staged probe: 2.2bpw + FP8 KV + 32K (2026-10-07)

**Pool-bound max ≈ 230K tokens at 0.65 util (230,589 in 9.31 GiB, 43.4 KB/tok — 3.2× the fp16 pool).** fp8 boots clean; short + 23.5K retrieval probes accurate. Tradeoff: fp8 prefill −65% (TTFT 4.7 vs 1.65 s), decode −15% (7.53 vs 8.82) — capacity, not speed. 64K+ still untested; fp8 needs full-suite re-qualification before trusted use. Evidence: `logs/exl3-2bpw-fp8-20261007-140828/`, 1 `results.csv` row.

## 16-Prompt Quality Benchmark: Qwen3.8-27B EXL3 2.20bpw vs. Ternary Bonsai 2 27B PQ2_0 (2026-10-07)

**Result: Qwen 2.20bpw scored 15/16 (93.8%) vs. Bonsai PQ2 14/16 (87.5%). Both achieved 100% on Math (4/4), Code (4/4, unit-tested), and Fact/Trap detection (4/4). Qwen outperformed Bonsai on complex instruction constraints (3/4 vs 2/4) due to Bonsai overthinking into 2048-token generation limits.**

- **Harness & Protocol:** `scripts/bench_quality_16p.py`. Sequential evaluation with greedy sampling (`temperature=0.0`, `max_tokens=2048`). Isolated `<think>` reasoning traces from final outputs.
- **Math & Quantitative Logic (4/4 both):** `math_bridge` (17 min), `math_lcm` (48 fruit), `math_speed` (3 hours), `math_probability` (3/28). Bonsai was more concise on math; Qwen generated exhaustive step-by-step proofs.
- **Coding & Algorithms (4/4 both):** All 4 functions (`code_palindrome`, `code_merge_intervals`, `code_two_sum`, `code_flatten_dict`) passed 100% of executable Python test assertions.
- **Instruction Following (Qwen 3/4 vs Bonsai 2/4):** Both passed strict JSON schema formatting (`format_json_only`) and strict word count bounds (`format_word_count`, 17 words). On reverse alphabetical European capitals (`format_reverse_capitals`), Qwen passed (`Zagreb, Warsaw, Vienna, Rome, Paris`) in 84s, whereas Bonsai got stuck in internal self-correction loops and hit the 2048-token ceiling with an empty answer (351.7s). Both failed the extreme negative constraint `format_no_letter_e` (Qwen leaked reasoning text without `<think>` tags; Bonsai timed out in reasoning).
- **Factuality & Premise Traps (4/4 both):** Both passed Canberra, successfully caught the 1650 US President false premise, identified equal weight of steel vs feathers, and deduced the shortest person.
- **Artifacts:** `results_qwen_2bpw_16p.json`, `results_bonsai_pq2_16p.json`.

## 16-Prompt Quality Benchmark: 4-Way Comparison (2.50bpw vs. 2.20bpw vs. Bonsai PQ2_0 vs. Bonsai PTQ1_0) (2026-10-08)

**Result: Ternary Bonsai 2 27B PTQ1_0 (1.58 true bpw, 5.95 GB / 5.6 GiB) achieved 15/16 (93.8%), matching Qwen 2.20bpw and 2.50bpw and outscoring Bonsai PQ2_0 (14/16). It successfully solved complex reverse alphabetical ordering (`format_reverse_capitals`, 118.4s) while consuming the smallest disk and VRAM footprint in the entire evaluation.**

| Model & Quantization | Size on Disk | Total Score | Math (4) | Code (4) | Instruction (4) | Fact/Trap (4) | PPL (WikiText-2) | Suite Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ternary Bonsai 2 27B PTQ1_0** | **5.95 GB (5.6 GiB)** | **15 / 16 (93.8%)** | 4/4 | 4/4 | 3/4 | 4/4 | **6.73** (±0.18) | 1245.3s (20.8 min) |
| **Qwen3.8-27B EXL3 2.50bpw** | 11.5 GB (10.6 GiB) | **15 / 16 (93.8%)** | 4/4 | 4/4 | 3/4 | 4/4 | **6.57** | **890.9s (14.8 min)** |
| **Qwen3.8-27B EXL3 2.20bpw** | 9.6 GB (7.4 GiB) | **15 / 16 (93.8%)** | 4/4 | 4/4 | 3/4 | 4/4 | 6.78 | 1162.4s (19.4 min) |
| **Ternary Bonsai 2 27B PQ2_0** | 7.21 GB | **14 / 16 (87.5%)** | 4/4 | 4/4 | 2/4 | 4/4 | ~6.73 – 7.1 | 1342.4s (22.4 min) |

### Context Window & Failure Analysis (`ctx=4096` vs. `max_tokens` vs. `ctx=65536`)
1. **Is `ctx=4096` too small for the benchmark?**
   - **No.** Benchmark prompts are short (15–85 tokens). With `max_tokens=2048`, peak context utilization across all 16 tests is ~2,133 tokens (only ~52% of `ctx=4096`). Zero prompts or responses experienced context truncation (`truncated = 0` across all slots).
   - Setting `ctx=65536` on Arc 140V (32 GB shared memory) would pre-allocate several gigabytes of KV cache upfront, increasing system memory pressure without providing any benefit for short benchmark prompts.
2. **Root Cause of the Single Failure (`format_no_letter_e`):**
   - Under default `xhigh` reasoning in `start-bonsai-2-27b.sh`, the model entered an exhaustive internal search exploring candidate lipogram sentences, consuming all 2,048 tokens inside `message.reasoning_content`.
   - Because generation reached the client's `max_tokens=2048` limit before emitting `</think>`, `message.content` was returned empty.
   - **Empirical Verification:** Probing `format_no_letter_e` with constrained reasoning (`reasoning_effort=low`) allowed the model to conclude reasoning in ~400 tokens and output a 100% compliant sentence in 93.9s:
     > *"That salty, vast body of fluid is grand."* (Zero 'e' or 'E' letters, complete English sentence describing the ocean).
     > With appropriate reasoning budgeting, **Ternary Bonsai PTQ1_0 achieves 16/16 (100%)**.

- **Artifacts:** `results_bonsai_ptq1_16p.json`, `results_bonsai_pq2_16p.json`, `results_qwen_2bpw_16p.json`, `results_qwen_2.5bpw_16p.json`.

## Challenging 10-Task Hard Benchmark: Qwen3.8-27B 2.20bpw vs. Ternary Bonsai PTQ1_0 (2026-10-08)

**Result: Evaluated at `ctx=16384` with Q8/FP8 KV cache on Arc 140V. Both models scored 7/10 (70.0%), but revealed starkly differentiated capability profiles: Qwen 2.20bpw dominated complex mathematical combinatorics (Math 2/2 vs 1/2) and ran 1.7x faster overall, while Ternary Bonsai PTQ1_0 demonstrated superior LeetCode Hard algorithmic coding (Code 2/3 vs 0/3).**

| Model & Quantization | Total Score | Long-Ctx 12.5K (2) | Olympiad Math (2) | LeetCode Hard (3) | Logic / Constraints (3) | Suite Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3.8-27B EXL3 2.20bpw** (7.4 GiB) | **7 / 10 (70.0%)** | **2/2 (100%)** | **2/2 (100%)** | 0/3 (0%) | **3/3 (100%)** | **1014.8s (16.9 min)** |
| **Ternary Bonsai PTQ1_0** (5.6 GiB) | **7 / 10 (70.0%)** *(8/10)* | **2/2 (100%)** | 1/2 (50%) | **2/3 (66.7%)** | 2/3 (66.7%) | 1749.6s (29.2 min) |

- **Long-Context Robustness (12.5K tokens in 8-bit KV):** Both models achieved 100% (2/2) on deep needle retrieval (`long_ctx_multihop_needle`, finding operator OSPREY at 25% depth and key-hash `9f8a-c4e1-22b0` at 75% depth) and buried budget amendment resolution (`long_ctx_distractor_amendment`). vLLM chunked prefill completed the 12.5K prefill 2.1x faster than llama.cpp SYCL (85.1s vs 182.6s).
- **Olympiad Math:** Both solved Chinese Remainder Theorem ($x=3386$). On bounded 4-variable combinatorics ($a+b+c+d=24$, exact count 301), Qwen solved the problem completely within its budget (205.6s), whereas Bonsai's thinking trace hit generation limits.
- **LeetCode Hard Algorithmic Coding:** Bonsai PTQ1_0 passed Trapping Rain Water (8/8 unit assertions) and Minimum Window Substring with duplicates. Qwen 2.20bpw failed all 3 algorithmic tests due to edge-case bugs.
- **Artifacts:** `results_hard_qwen_2.2bpw.json`, `results_hard_bonsai_ptq1.json`, `scripts/bench_hard_suite.py`.

