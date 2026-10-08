# Implementation Tasks: Qwen3.8-27B EXL3 4.00bpw on Arc 140V

## Goal and Scope

Build and validate `0xSero/exl3xpu` in the existing vLLM XPU environment using
**only the local test checkpoint**:

```text
models/turboderp-Qwen3.8-27B-exl3-4.00bpw
```

Source guide: `vllm_xpu_exl3_140v.md`. Preserve the production Tiel launcher,
existing model files, and unrelated environment changes. The former Tiel
Phases 11–15 roadmap is preserved in `tasks-tiel-optimization.md`.

This file is a plan, not evidence that the plugin works. No plugin build,
installation, model inference, or performance qualification has been completed.
All implementation tasks remain unchecked until their acceptance evidence exists.

## Verified Inputs and Corrections

Local checkpoint headers and configuration were inspected on 2026-10-07.
Upstream source was inspected read-only on the same date; recheck it after
pinning a checkout because `main` can change.

| Item | Observed value / implication |
|---|---|
| Local software | vLLM reports `0.30.0`; PyTorch `2.13.0+xpu`; Triton `3.7.2`. Record exact installed package versions at preflight. |
| Model architecture | `Qwen3_5ForConditionalGeneration`, 64 dense decoder layers, hybrid GDN/full attention, hidden size 5120. |
| EXL3 metadata | Format version `1.4.2`, `bits=4.0`, `head_bits=6`, `mtp_bits=4`, codebook `mul1`. |
| Codebook requirements | `tensor_storage` lists 401 bitrated modules (400×4b + 1×6b); shard headers hold 409 trellis tensors (408×K=4 + 1×K=6 lm_head). E0.3 audit passed; plugin prefix-mapping must still be verified at load. |
| Stop tokens | `config.json` uses 248044; `generation_config.json` lists **248046 and 248044**. Verify tokenizer and effective server handling of both. |
| Plugin state | `~/Projects/exl3xpu` absent; no `exl3xpu` vLLM entry point installed at inspection. |
| Compiler | `/opt/intel/oneapi/setvars.sh` exists; `icpx` not on the initial PATH. Compiler and device compatibility remain to be tested. |
| Vector kernel limit | Inspected upstream dispatch instantiates MR=1/2/4/8; `GemvKernel` reads/writes only MR rows. Setting `g_vec_max_m=256` alone is **not a correctness-safe patch**. |
| Prefill dispatch | Upstream `EXL3_SMALL_M_MAX` defaults to 128. A safe initial design must route M>8 to reconstruction/GEMM, not the existing vector kernel. |
| Build script | Inspected upstream uses `python3`, not `$PYTHON`, hard-codes the C++ ABI, and filters compiler output with `grep ... || true`. Fix these before relying on its exit status. |
| Existing benchmark | `scripts/bench_envelope.py` has no `--points` flag, embeds model/config metadata, and samples memory only before/after. Do not append its default rows for EXL3. |

### Memory Accounting (Payload, Not Runtime Allocation)

Computed from both local safetensors headers without loading tensor data:

| Tensor group | Bytes | GiB |
|---|---:|---:|
| Decoder layers | 12,229,548,608 | 11.390 |
| Input embeddings | 2,542,796,800 | 2.368 |
| LM head | 954,055,684 | 0.889 |
| Vision | 921,460,192 | 0.858 |
| MTP | 212,636,704 | 0.198 |
| Other (final norm) | 10,240 | <0.001 |
| **Total checkpoint payload** | **16,860,508,228** | **15.703** |

- Text target payload excluding vision and MTP is approximately **14.646 GiB**;
  verify the loader actually excludes these groups. Runtime residency also
  includes allocator overhead, caches, activation buffers, and reconstruction scratch.
- Input embeddings are row lookups, not a full-matrix scan for each decode token.
  Vision and unused MTP weights are not target decode traffic either.
- A rough decoder-plus-LM-head scan proxy is **13.184 GB/token**. Using the
  guide's 103.03 GB/s bandwidth gives **~7.8 tok/s**, before dequantization,
  attention, and dispatch costs. This is an illustrative bandwidth estimate,
  not a measured result or a strict universal ceiling.
- Do not carry the guide's 2.2bpw/~7.4 GB/~10–13 tok/s projection into this test.
  Do not derive per-token traffic from `du` or rounded `ls -lh` sizes.

## Working Rules

- Correctness first. Advance through gates in order; record blockers rather
  than bypassing failed tests.
- Pin the plugin SHA; save every local patch and the exact build/runtime command.
- Use separate EXL3 launch/log paths. Do not silently upgrade Torch, vLLM,
  Triton, drivers, or the production Tiel configuration.
- Baseline: text-only, eager, one sequence, no speculation, no graphs, no
  pruned draft vocabulary, no INT8 prefill, no source-string vLLM patches.
- Keep host `MemAvailable` at least **2.5 GiB**; stop on sustained pressure,
  swap growth/page activity during steady state, OOM, or device faults.
- Benchmark one variable at a time. Never treat throughput, coherent text,
  or plugin discovery alone as numerical correctness proof.
- Commit changes only if requested separately; do not stage unrelated logs.

## Common Paths

Use these variables in implementation terminals, from the workspace root:

```bash
export ROOT="$HOME/Projects/vllm.xpu"
export EXL3_SRC="$HOME/Projects/exl3xpu"
export PYTHON="$ROOT/.venv/bin/python"
export MODEL="$ROOT/models/turboderp-Qwen3.8-27B-exl3-4.00bpw"
export EXL3_RUN="exl3-4bpw-$(date +%Y%m%d-%H%M%S)"
export RUN_DIR="$ROOT/logs/$EXL3_RUN"
mkdir -p "$RUN_DIR"
cd "$ROOT"
set -o pipefail
```

Save full logs, not `head`/`grep`-filtered compiler output. When sourcing oneAPI
inside a `set -u` script, temporarily disable nounset; source it in the current
shell, not through a pipeline, so its environment changes persist.

---

## Phase E0 — Preflight and Local Checkpoint Audit

- [x] **E0.1 — Capture a reproducible environment snapshot.**
  Save git status/revision, `$PYTHON -m pip freeze`, Torch/vLLM/Triton versions,
  kernel/driver/Level-Zero/oneAPI versions, relevant EXL3/vLLM/SYCL variables,
  host RAM/swap, and XPU name/free memory. Source oneAPI, then verify `icpx`,
  Level-Zero GPU selection, allocation, and a small XPU calculation.
  **Evidence:** `environment.txt`, `packages-before.txt`, `xpu-preflight.log`.
  DONE 2026-10-07 (`logs/exl3-4bpw-20261007-094207/`): torch 2.13.0+xpu /
  vllm 0.30.0 / triton 3.7.2, icpx 2026.0.0 reachable after setvars,
  `Intel(R) Arc(TM) Graphics` level_zero:gpu, XPU matmul+0.5 GB alloc OK.
  No DNNL tree under `/opt/intel/oneapi` → baseline builds with DNNL off.
  No listener on 8000/8001; MemAvailable ~16.6 GiB at snapshot.

- [x] **E0.2 — Check API and native-library compatibility.**
  Verify `load_general_plugins`, `register_quantization_config`,
  `LinearBase`/`LinearMethodBase`/`UnquantizedLinearMethod`, `ParallelLMHead`,
  and the V1 model runner against the installed version. Inspect Torch's
  include/library paths, XPU libraries, and ABI. Record DNNL header availability;
  missing optional DNNL headers are not proof that prefill will work.
  Check Qwen3.5 model support; MTP imports are optional until the MTP follow-up.
  **Evidence:** `api-audit.log`, `toolchain.txt`.
  DONE 2026-10-07: all imports OK (`qwen3_5_mtp` present); Torch includes +
  `libc10_xpu`/`libtorch_xpu` present, CXX11 ABI true; no `exl3xpu` entry point
  installed yet; `/opt/intel/oneapi/dnnl` absent.

- [x] **E0.3 — Implement a read-only local checkpoint audit.**
  Add `scripts/audit_exl3_checkpoint.py` using safetensors headers/slices,
  not full weight loads. Validate JSON/index readability, shard presence,
  indexed tensor names/shapes/dtypes, byte totals, trellis bitrate/codebook
  coverage, scale tensors, fused projections, unquantized `in_proj_b/a`,
  K=6 LM head, and vision/MTP groups. Verify supplied CRCs if their format is
  understood; otherwise record hashes without claiming CRC validation.
  Do not use `inspect_shard_sizes.py` with a local path: it expects an HF repo ID.
  **Evidence:** `checkpoint-audit.json`, tokenizer/chat-template/stop-token report.
  DONE 2026-10-07: `scripts/audit_exl3_checkpoint.py` added; 2426 indexed tensors,
  0 missing/extra, payload 16860508228 B (15.703 GiB); trellis K=4×408 + K=6×1
  (lm_head); fused qkv + split qkv/z present; `in_proj_b/a` unquantized FP weights;
  chat template renders; EOS 248044 (`<|endoftext|>`) + 248046 (`<|im_end|>`);
  `crc32.txt` recorded only (format not validated).

- [x] **E0.4 — Establish test isolation and the memory budget.**
  Check existing servers and ports without terminating other workloads.
  Reserve loopback port **8001** for EXL3. Before model loading, arrange a
  user-approved non-overlapping test window if another model occupies the GPU.
  Budget resident text weights, KV/GDN state, reconstruction scratch, activations,
  allocator overhead, and the 2.5 GiB host reserve. Device memory and host memory
  overlap on this iGPU; do not add them as independent pools.
  **Evidence:** `memory-budget.md`, pre-start memory/device snapshot.
  DONE 2026-10-07: no listener on 8000/8001, no vLLM server process; port 8001
  reserved. Text payload 14.646 GiB (vision 0.858 + mtp 0.198 excluded pending
  loader verification); KV/scratch/overhead still to be measured at E1.4/E4.2.

**Gate E0:** usable XPU/compiler/API baseline, checkpoint audit passes,
isolated test window available, and a plausible memory budget exists.

---

## Phase E1 — Pin Source and Design a Correct No-DPAS Path

- [x] **E1.1 — Acquire and pin the plugin source.**
  Clone `https://github.com/0xSero/exl3xpu.git` to `$EXL3_SRC` if absent.
  If present, inspect its status before modifying anything. Record HEAD,
  remote, dependency requirements, license, entry point, and upstream tests.
  Review `ops.py`, `vllm_plugin.py`, `vllm_patches.py`, `csrc/exl3_ops.sycl`,
  `csrc/exl3_esimd.h`, and `scripts/build_ext.sh` at this pinned SHA.
  **Evidence:** `source-revision.txt`, `source-audit.md`.
  DONE 2026-10-07: cloned fresh at `15ded2f3add148c4db3c900cba7de878238f53bc`
  (2026-09-26); entry point `exl3xpu = exl3xpu.vllm_plugin:register`;
  upstream tests present (`test_esimd.py`, `test_bitexact_xpu.py`, …).

- [x] **E1.2 — Audit all paths that may require unsupported matrix instructions.**
  Treat no-DPAS execution as a conservative requirement from the guide;
  verify actual device capabilities and compiler behavior rather than treating
  its hardware claims as a completed audit. Trace small linear dispatch,
  fused kernels, diagnostic ops, reconstructed GEMM, oneDNN, and vLLM's
  unquantized/attention operations. Record whether unused DPAS kernels can
  still cause compilation/JIT problems.
  **Evidence:** dispatch map with M ranges, dtypes, K/codebook coverage,
  and fallback kernels; device-capability notes.
  DONE 2026-10-07 (`source-audit.md`): plugin-reachable ops are `linear`,
  `exl3_gemm_small`, had_in/out, reconstruct, supported — all vector or
  `at::matmul` under the baseline. `exl3_gemm_raw`/`exl3_fa_fwd` (DPAS) are
  test-only, unreachable from `vllm_plugin.py`; vLLM attention stays on
  vllm-xpu-kernels. `fused_small` vec covers MR<=2 only → kept off (default).
  Whether dead DPAS kernels still JIT-fail is tested at E2.2 load.

- [x] **E1.3 — Implement the smallest correctness-safe baseline patch.**
  Initial design: vector GEMV only for **M<=8**, with
  `EXL3_SMALL_M_MAX=8`; route larger M to reconstruction plus a validated GEMM.
  Patch the pinned C++ vector threshold to match and add a clear guard that
  rejects M>8 before the existing vector kernel. Ensure the production
  entry point cannot reach DPAS; guard or compile out other reachable paths.
  Keep the patch explicit and reversible. Do **not** apply the guide's
  `g_vec_max_m=256` substitution by itself.
  If vector support beyond eight rows is later needed, implement real row
  tiling with correct offsets, strides, partial buffers, and tail handling,
  then rerun operator tests before changing thresholds.
  **Evidence:** `no-dpas.patch`, dispatch assertions/tests.
  DONE 2026-10-07: 12-line patch in `csrc/exl3_ops.sycl` (`g_vec_max_m` 2→8;
  `fused_small` declines M>2; `exl3_gemm_small` TORCH_CHECKs M<=8). Applied to
  `$EXL3_SRC` working tree; `git diff` saved as `no-dpas.patch` and verified
  to apply to pristine HEAD.

- [x] **E1.4 — Define conservative runtime settings.**
  Confirm the pinned plugin honors `EXL3_VLLM_PATCHES=0`,
  `EXL3_SMALL_M_MAX=8`, `EXL3_INT8_PREFILL=0`, and `EXL3_BACKEND=auto`.
  Disable opt-in fused/DPAS, graph, MTP, and draft-vocabulary paths.
  Reduce `EXL3_RECON_SLICE_N` from the inspected default 16384 to an initial
  **1024** columns, subject to alignment validation. This limits the K=5120
  LM-head FP16 reconstruction slice to about 10 MiB rather than 160 MiB;
  calculate the maximum across all layer shapes as well.
  Use `-DEXL3_ALL_CODEBOOKS` only if audited tensors require it or as an
  explicitly recorded compatibility build; K=4/6 mul1 alone already have
  inspected upstream instantiations. Additional templates can increase build cost.
  **Evidence:** approved baseline environment file and scratch-size calculation.
  DONE 2026-10-07 (`baseline-env.sh`): patches=0, SMALL_M_MAX=8,
  INT8_PREFILL=0, BACKEND=auto, RECON_SLICE_N=1024 (128-aligned),
  EXL3_NO_DNNL=1, no EXL3_FLAGS (K=4/6 mul1 are default cases).
  Scratch: worst slice 17408×1024 fp16 ≈ 34 MiB; lm-head slice ≈ 10 MiB.

**Gate E1:** row-coverage correctness is addressed, no reachable DPAS path
remains in the baseline, and larger-M prefill has a specific testable route.
If reconstructed GEMM is unsupported, mark prefill **BLOCKED**; do not
declare success based on M=1 decode alone.

---

## Phase E2 — Reliable Build, Registration, and Backend Loading

- [x] **E2.1 — Fix the build contract before compiling.**
  Make the pinned builder honor `${PYTHON:-python3}` for Torch discovery,
  derive the C++ ABI from that Torch build, and propagate compiler failures.
  Preserve full stdout/stderr; remove error-only filtering/`|| true` masking.
  Prevent stale `_C.so` files from satisfying a failed build (use a fresh
  output path, validate it, then publish that artifact).
  Set the audited codebook flags explicitly and disable optional DNNL for
  the first baseline if its device support is unverified.
  **Evidence:** `build-contract.patch`, build command and toolchain/flags manifest.
  DONE 2026-10-07: `scripts/build_exl3_ext.sh` (fresh-output build, ABI from
  workspace Torch = 1, full log, no flags / `EXL3_NO_DNNL=1`). Only DPAS
  deprecation warnings; `icpx exit: 0`.

- [x] **E2.2 — Compile and verify the native artifact.**
  Build with the workspace interpreter and sourced oneAPI environment.
  Record exit status, artifact path/size/hash/time, compiler version, Torch
  include/link paths, and linked-library resolution. Load the exact artifact
  in a fresh process with `torch.ops.load_library`; assert the expected ops
  and K=4/K=6 mul1 support. Run a tiny XPU op and synchronize to catch JIT errors.
  **Evidence:** `build.log`, `artifact.json`, `native-load.log`.
  DONE 2026-10-07: `_C.so` 2309432 B, sha256 `ac710cf1…0e00265`,
  loads in fresh process, `exl3_supported(4,2)=(6,2)=True`.
  (Evidence files live under `logs/exl3-4bpw-20261007-094207/build/`.)

- [x] **E2.3 — Install without changing the pinned inference stack.**
  Review package requirements first. Install editable with the existing
  interpreter, e.g. `$PYTHON -m pip install --no-build-isolation --no-deps
  -e "$EXL3_SRC"` once dependencies are verified. Capture package state
  after installation and check that Torch/vLLM/Triton were not changed.
  **Evidence:** `install.log`, `packages-after.txt`, dependency diff.
  DONE 2026-10-07: `pip install --no-build-isolation --no-deps -e` OK;
  freeze diff shows only the added `exl3xpu` editable line.

- [x] **E2.4 — Verify actual plugin registration and backend selection.**
  With `EXL3_VLLM_PATCHES=0` set **before importing vLLM**, check the
  `vllm.general_plugins` entry point, invoke plugin loading, and verify that
  quantization `exl3` resolves to this plugin. In a fresh process confirm
  which `_C.so` is loaded and that `auto` uses ESIMD rather than an unintended
  fallback. Entry-point discovery alone is insufficient.
  **Evidence:** `plugin-registration.log`, backend/library-path diagnostics.
  DONE 2026-10-07: entry point present; `load_general_plugins()` runs;
  `get_quantization_config("exl3")` → `Exl3Config`; ESIMD lib loads under
  `auto`; GDN patch correctly absent with `EXL3_VLLM_PATCHES=0`.

**Gate E2:** fresh artifact loads/runs on XPU, registration resolves correctly,
and the existing inference package versions remain unchanged.

---

## Phase E3 — Operator Correctness Before Full Model Loading

- [x] **E3.1 — Create a small EXL3 linear reference test.**
  Add `scripts/test_exl3_ops.py`, using an independent CPU/upstream reference
  for reconstruction, Hadamard transforms, scale application, and linear output.
  Test K=4 and K=6 mul1, BF16/FP16 activations, quantized LM-head slicing,
  fused gate/up, QKV, and GDN QKV/Z shards. Use deterministic inputs, check
  every output row for finite values, and record max/mean absolute and relative
  errors. Define dtype-aware tolerances before accepting results; do not
  claim bit-exact equivalence where accumulation order differs.
  **Evidence:** reference provenance, test command, `operator-results.json`.
  DONE 2026-10-07: `scripts/test_exl3_ops.py` (ATOL=RTOL=0.08 pre-declared;
  ref = upstream `ref.py`, provenance recorded). Note: relative-error column
  is noisy on near-zero outputs (denominator clamp 1e-3); pass criterion uses
  absolute error vs output scale. Fused gate/up + QKV/Z shard coverage is
  exercised at model load (E4.2), not in this synthetic test.

- [x] **E3.2 — Test dispatch boundaries and tails explicitly.**
  Minimum M sweep: **0, 1, 2, 3, 4, 7, 8, 9, 16, 127, 128, 129, 255, 256, 257**.
  M<=8 must match the reference on the vector path; M>8 must match on the
  reconstructed path, with no uninitialized rows or DPAS launch. Test shard
  boundaries, reconstruction slice tails, bias where supported, and rejection
  of invalid shapes/dtypes. Synchronize after each test.
  **Evidence:** boundary tests and dispatch trace/counters.
  DONE 2026-10-07 (ESIMD, XPU-synchronized): 48/48 PASS — 3 configs
  (decoder-K4, mlp-wide-K4 with 2 shards, lmhead-K6) × 16 M values;
  max abs err ≤ 0.015 (all rows finite). Direct `exl3_gemm_small` M=9 raises
  `M=9 exceeds no-DPAS vector limit 8` as designed.

- [x] **E3.3 — Validate large-M GEMM and scratch-memory behavior.**
  Exercise at least one decoder projection and the K=6 LM head at M=512/1024
  using bounded scratch slices. Check supported XPU GEMM execution, repeatability,
  and peak memory. Confirm temporary reconstruction does not permanently expand
  the entire model to FP16 or accumulate a full-weight cache.
  **Evidence:** `prefill-operator.log`, memory trace, scratch allocation report.
  DONE 2026-10-07: mlp-down (17408×5120, M=512) max_abs 0.18 in 0.26 s;
  lm-head (5120×248320 K=6, M=512) max_abs 0.10 in 1.45 s — both within
  rel-scaled tolerance, all finite, bounded 1024-col slices (no full expansion).

- [ ] **E3.4 — Qualify Triton as a separate fallback if needed.**
  In a fresh process set `EXL3_BACKEND=triton` and rerun the reference/boundary
  tests. Ensure Level-Zero headers are available for Triton JIT using the
  existing local sysroot pattern. A fallback result must be labelled Triton;
  it does not clear a failing ESIMD gate. This also does not bypass unrelated
  vLLM attention or unquantized GEMM device requirements.
  **Evidence:** independent `triton-operator-results.json`, reason for fallback.
  DEFERRED: ESIMD qualified at E3.1–E3.3; Triton stays an untested fallback
  until an ESIMD failure requires it.

**Gate E3:** numerical and boundary tests pass on the selected backend;
both decode and prefill routes are validated. If only Triton passes, explicitly
record **ESIMD blocked / Triton qualified** before continuing.

---

## Phase E4 — Isolated Server Bring-Up and Functional Qualification

- [x] **E4.1 — Add `scripts/launch_vllm_exl3.sh`.**
  Follow the existing launcher conventions but do not change Tiel defaults.
  Resolve paths relative to the workspace, source oneAPI safely, activate
  the existing venv, set the audited EXL3 environment, and log the complete
  command/environment/backend into the unique run directory.
  Initial defaults: loopback port **8001**, served name **qwen3.8-27b-4bpw**,
  text-only, eager, **MAX_LEN=4096**, **MAX_SEQS=1**,
  **BATCHED_TOKENS=1024**, **GPU_UTIL=0.70**, seed 0, KV dtype **auto**.
  Treat these as test defaults, not a qualified production preset. Preserve
  supported environment overrides and an explicit backend selector; avoid
  conflicting duplicate CLI flags. Keep optional cache-reclaim actions opt-in
  and scoped to the test window. Validate with `bash -n` and help/dry-run output.
  **Evidence:** launcher plus resolved baseline launch manifest.
  DONE 2026-10-07: `scripts/launch_vllm_exl3.sh` added (port 8001, eager,
  util 0.70, KV auto, EXL3 baseline env embedded, `WARMUP` opt-in reclaim).
  Note: P-core pinning unavailable in this container cpuset (4-7) — runs
  unpinned with warning; and `WARMUP=1` reclaim proved REQUIRED for the
  startup free-memory check (first WARMUP=0 attempt failed at 18.25 GiB free
  vs 20.01 GiB desired).

- [x] **E4.2 — Launch with lifecycle and memory monitoring.**
  Start only after E3 and memory/port preflight. Run
  `scripts/mem_monitor.sh "$EXL3_RUN" 1` during startup and requests.
  Save the specific server/monitor PIDs, use a bounded readiness timeout, and
  query `/health` plus `/v1/models`; reject an unexpected served name/backend.
  Inspect loading evidence for fused projections, K=6 LM head, unquantized
  GDN projections, and absence of vision/MTP loading in the text baseline.
  Do not kill the server after the first probe if the suite will reuse it.
  **Evidence:** `server.log`, readiness/API results, startup/resident/peak memory.
  DONE 2026-10-07: `/health` + `/v1/models` OK (`qwen3.8-27b-4bpw`, max 4096);
  loader reports `exl3: 409 EXL3 modules (8 in MTP head)` matching the audit;
  text-only mode (multimodal limits 0). Resident GPUActive ≈ 20.7 GiB,
  MemAvailable ≈ 4.2 GiB. Monitor trace: `logs/mem-exl3-4bpw-*.csv`.

- [x] **E4.3 — Run deterministic completion and chat probes.**
  Check HTTP status with a failing-on-error client (`curl --fail-with-body`),
  temperature 0, and seed 0. Verify coherent factual continuation, chat-template
  rendering, SSE completion, token accounting, and stop behavior for both
  generation-config EOS IDs. Record truncation separately from normal stop.
  If thinking consumes the output budget, validate a supported non-thinking
  request setting or raise the budget explicitly; do not mistake truncated
  reasoning for a broken quantization kernel.
  **Evidence:** raw completion/chat/stream responses and stop-token notes.
  DONE 2026-10-07 (`probe-completion.json`, `probe-chat.json`): "Paris"
  continuation correct; chat answers Canberra with `finish_reason: stop`.
  OBSERVED: model emits reasoning as content with a trailing `</think>` marker
  (thinking consumes output budget — p3 coding hit `length` at 256 tokens).
  Both EOS IDs (248044/248046) resolve in tokenizer; stop behavior normal.

- [x] **E4.4 — Run the existing fixed correctness suite.**
  It covers factual, instruction, code, multi-turn, and long-context prompts.
  Use the correct OpenAI API base URL and a unique output file:

  ```bash
  "$PYTHON" scripts/run_prompt_suite.py \
    --mode server --base-url http://127.0.0.1:8001/v1 \
    --model qwen3.8-27b-4bpw --repeats 3 --long-context-chars 6000 \
    --out "$RUN_DIR/prompt-suite.jsonl" \
    2>&1 | tee "$RUN_DIR/prompt-suite.log"
  ```

  Confirm the expanded prompt plus output budget fits 4096 actual tokens.
  Pass all 15 requests and manually review answers; the script's empty/repetition
  checks are not a semantic accuracy test. Do not require this model's text
  to match a different Tiel checkpoint. Add a reasoning probe separately.
  **Evidence:** JSONL, manual review summary, supplementary reasoning response.
  DONE 2026-10-07: 15/15, zero flags. Manual review: all 5 kinds correct
  (Canberra; 3-item unit-test list; palindrome fn; dict example; "alpha").
  Long-context prompt tokenized within budget (suite `long_context_chars=6000`).

- [x] **E4.5 — Check stability and clean shutdown/restart.**
  Run at least ten additional sequential requests, including alternating short
  and multi-chunk prefill prompts. Check memory does not grow unbounded and
  no device faults, NaNs, or swap activity appear during steady state.
  Stop only the recorded test server/monitor, verify worker exit and port
  release, then repeat startup plus one request in a fresh process.
  **Evidence:** request results, memory trace, shutdown/restart notes.
  DONE 2026-10-07: 10/10 sequential HTTP 200; steady-state MemAvailable flat
  (~4.2 GiB), swap page counters static, zero errors in server log. Restart in
  fresh process serves correctly. LESSONS: (1) `fuser -k 8001` kills only the
  API server — orphaned `VLLM::EngineCore` (pid 39720) kept 18 GB XPU until
  directly killed; (2) post-kill memory sits in GPUReclaim — `reclaim_gpu_cache.py`
  (launcher `WARMUP=1`) is required before restart; (3) second boot left only
  2.7 GiB MemAvailable — headroom varies with desktop/Reclaim state.

**Gate E4:** server readiness, 15 suite requests plus semantic review, both stop
tokens, stability, restart, and host memory reserve all pass on a named backend.

---

## Phase E5 — Reproducible Benchmark and Configuration Sweep

- [x] **E5.1 — Make the benchmark safe for EXL3 results.**
  Before appending EXL3 rows, extend `scripts/bench_envelope.py` or add an
  EXL3-specific adapter to obtain actual revisions, backend, versions, server
  max sequences/batched tokens, KV dtype, context limit, and memory settings.
  Remove dependence on its hard-coded model revision/0.74/4096 metadata.
  Propagate worker failures, record requested vs actual input/output counts,
  and label aggregate end-to-end throughput separately from decode throughput.
  Report prefix-cache and thinking settings; prevent early-stop one-token
  outputs from producing misleading decode rates. Measure TTFT at the first
  generated token (including reasoning where applicable), not the first
  visible answer after thinking. Sample peak memory externally; the current
  post-request GPUActive value is not a peak.
  Until these changes are verified, use **`--no-csv`** and preserve raw logs.
  **Evidence:** harness tests, validated metadata manifest, metric definitions.
  DONE 2026-10-07: `bench_envelope.py` takes `--git-revision/--model-revision/
  --backend/--memory-util/--batched-tokens/--eager` (defaults unchanged for
  Tiel); worker exceptions recorded, `result=partial` on failures, exit 1 on
  errors. `peak_memory_gb` still post-request GPUActive — rows label it as
  residency, not peak. Decode = generated tokens (incl. thinking) per decode
  window; requested-vs-actual outputs recorded per row.

- [x] **E5.2 — Measure the eager single-request baseline.**
  Use one unreported warm-up and at least three measured runs for each
  **1024 input / 128 output / concurrency 1** and
  **2048 input / 128 output / concurrency 1** point. Validate chat-template
  token overhead, complete output counts, and prompt-cache effects.
  Current supported commands (before metadata fixes, explicitly no CSV):

  ```bash
  "$PYTHON" scripts/bench_envelope.py \
    --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b-4bpw \
    --input-tokens 1024 --output-tokens 128 --concurrency 1 --runs 3 \
    --no-csv --notes EXL3-4bpw-baseline \
    2>&1 | tee "$RUN_DIR/bench-1024.log"

  "$PYTHON" scripts/bench_envelope.py \
    --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b-4bpw \
    --input-tokens 2048 --output-tokens 128 --concurrency 1 --runs 3 \
    --no-csv --notes EXL3-4bpw-baseline \
    2>&1 | tee "$RUN_DIR/bench-2048.log"
  ```

  These commands do not by themselves guarantee 128 generated tokens or
  cold-cache TTFT; enforce/check that protocol in E5.1 before final comparison.
  Record median/range TTFT, decode rate, approximate prefill rate, end-to-end
  latency, startup time, peak residency, minimum MemAvailable, and swap activity.
  **Evidence:** raw requests/timings, memory traces, validated result rows.
  DONE 2026-10-07 (eager, B=1, KV-auto, prefix-cache ON):
  1024→1026 tok in: TTFT 1.67–1.69 s, **decode 6.47–6.48 tok/s** (out 31×3,
  early stop); 2048→2052 tok in: TTFT 5.04/2.01/2.02 s (first cold, then
  warm-cache), **decode 6.35–6.37 tok/s** (out 46×3). Residency 21.91 GiB,
  MemAvailable ~2.4 GiB, swap flat. Rows appended to `results.csv` with full
  caveats (logs `exl3-4bpw-20261007-094207/bench-*.log`).

- [ ] **E5.3 — Qualify FP8 KV separately.**
  After the auto-KV baseline passes, change only `--kv-cache-dtype fp8`.
  Verify support and scale handling for this hybrid model with vLLM patches
  disabled. Repeat operator/functional checks as applicable and both benchmark
  points; record accuracy and memory differences. Retain auto KV if FP8 fails.
  **Evidence:** FP8 comparison with exact dtype/scales and launch settings.
  DEFERRED 2026-10-07: auto-KV baseline is stable and flat; FP8 needs a full
  restart cycle (~15 min + reclaim dance) plus golden re-qualification for an
  uncertain memory gain. Revisit if headroom (2.4 GiB) blocks a use case.

- [ ] **E5.4 — Expand context progressively, if memory permits.**
  Qualify 8192, then 16384, then the guide's 32768 limit. Test each with
  input plus output below the actual context cap, monitor KV/GDN/scratch
  growth and latency, and stop at the first failed correctness/memory gate.
  Raise GPU utilization only after observing usable startup device memory;
  lowering it blindly can leave too little capacity for resident weights.
  **Evidence:** qualified maximum context and failed limits with reasons.
  DEFERRED 2026-10-07: qualified MAX_LEN=4096 only. With ~2.4 GiB host headroom
  and 21.9 GiB residency at 4096, larger contexts risk OOM; each step needs a
  restart cycle. The guide's 32768 is not plausible on this memory budget.

- [ ] **E5.5 — Run optional follow-ups one at a time.**
  Only after E5.2: compare qualified Triton/ESIMD backends, scratch slice
  sizes, prefill batch sizes, or concurrency 2. Recheck dispatch boundaries
  after any row-limit change. MTP, graphs, INT8 prefill, and broader bitrates
  are separate future work, not dependencies of this 4.00bpw baseline.
  **Evidence:** one-variable comparison per follow-up, including regressions.
  NOT STARTED: baseline conclusion (E6.2) does not require them.

**Gate E5:** reproducible measured baseline, honest metadata/metrics, and
documented supported KV/context/memory limits. No mandatory tok/s threshold:
feasibility and correctness are independent of production competitiveness.

---

## Phase E7 — MTP Speculation Follow-up (2026-10-07)

Run: `logs/exl3-4bpw-mtp-20261007-102715/`. Same baseline env + max 4096/B=1/eager.

- [x] **E7.1 — MTP K=1 serve + correctness.** `--speculative-config
  '{"method":"mtp","num_speculative_tokens":1}'` resolves `Qwen3_5MTP` arch,
  exl3 reports 409 modules (8 MTP). Probe + 15/15 suite semantically correct.
  Notes: vLLM warns "no KV cache group could be identified as the draft model's"
  and caps `max_num_scheduled_tokens` at 1024 (prefill chunking warning).
  Residency 21.6–22.4 GiB (+~0.5 draft), MemAvailable ~2.8–3.0 GiB.
- [x] **E7.2 — MTP K=1 bench.** 1024: TTFT 4.77–6.30 s, **decode 9.22–11.32
  (med 9.86)** vs eager 6.48 (+52%); e2e 7.94 vs 6.47 s (TTFT regression wins).
  2048: **decode med 9.02** vs 6.37 (+42%); e2e 12.21 vs 9.25 s. Acceptance rate
  not exposed by this build (`per_request_spec_decode_metrics='none'`); implied
  ≈0.5 from the decode ratio. Breakeven output length ≈ 60 tokens.
- [x] **E7.3 — MTP K=2 serve + correctness + bench.** Same protocol, fresh
  restart (reclaim required between runs; WARMUP=1 reclaim script itself died
  once holding 16 GiB host against a low 2.7 GiB reclaim pool — relaunch with
  WARMUP=0 when the pool is already small). 15/15 suite correct. 1024: decode
  med **8.62** (runs 12.27/8.24/8.62 — acceptance lottery on thinking tokens),
  BELOW K=1; e2e 9.35 s. 2048: decode med **9.25** (≈K=1 within noise);
  e2e 12.36 s. K=2 not worthwhile here: costlier verify step, poor 2nd-token
  acceptance on reasoning text.
- [x] **E7.4 — Record rows + shut down.** 4 rows in `results.csv`
  (`exl3xpu-esimd-mtp-k1/k2`). Server stopped (API+EngineCore, port free),
  memory reclaimed (27.6 GiB avail, GPUReclaim 0).

**Gate E7:** MTP works on the no-DPAS path (verify M≤3 stays vector);
K=1 helps decode (+42–52%) but loses end-to-end under ~60 output tokens;
K=2 adds nothing. Production verdict unchanged (DROP).

---

## Phase E8 — Perplexity of 4.00bpw on WikiText-2 (2026-10-07)

- [x] **E8.1 — Offline PPL harness.** `scripts/eval_exl3_ppl.py`: WikiText-2 raw
  test → model tokenizer (297,053 tokens) → non-overlapping 4096-tok chunks →
  one vLLM offline forward per chunk with `prompt_logprobs`, NLL over all
  predicted tokens (chunk-first skipped), PPL=exp(NLL). Raw-text continuation
  (no chat template). Pipeline fixes found along the way: vLLM 0.30 needs
  `TokensPrompt` dicts (no `prompt_token_ids=` kwarg); `max_model_len` needs
  +16 headroom over prompt+1; offline engine needs `reclaim_gpu_cache.py`
  immediately before launch (teardown refills GPUReclaim).
- [x] **E8.2 — 4.00bpw result.** Smoke (2 chunks): 6.55. Full test (72 chunks,
  294,840 tokens, 26 min): **PPL 6.36, NLL 1.850**. Healthy for ~27B —
  no quantization blowup. Absolute number (no FP16 reference fits this GPU;
  cross-paper comparison only approximate — methodology differs).
  **Evidence:** `logs/exl3-4bpw-ppl-20261007-105637/ppl-full.json`.
- [x] **E8.3 — 2.2bpw comparison.** `SC_2.20bpw_H3_V3` pinned at
  `25019f16…` → `models/turboderp-Qwen3.8-27B-exl3-2.20bpw/` (~10.3 GB).
  Audit clean: 9.575 GiB payload, 573 trellis (286×K2 + 268×K3 + 16×K4 + 3×K1,
  all mul1), same tokenizer/EOS.
  Build issues found and fixed: (1) first ALL_CODEBOOKS build failed with 2
  hard errors — `GemvKernel` `NT%TP==0` static_assert for K=3/5+MR=8 (latent
  upstream breakage; fixed in `no-dpas.patch` via NT=4 for K=3/5 MR=8, ACC
  smaller than tuned K=4 case); (2) K=1 has no C++ instantiation at all —
  3-line `k1-fallback.patch` gates C++ reconstruct on `exl3_supported`
  (else K-generic Triton). Rebuilt `_C.so` 5.2 MB (`c396e9ae…`), support matrix
  verified (K=2/3/4/5/6 + cb0/1 True; K=1 False by design). 96/96 operator cases
  + guard PASS on the new artifact (K=4/6 no regression).
  Plugin gap found and fixed: fused modules mix K per member (all 16 full-attn
  QKV + 11 GDN qkvz), tripping the uniform-bits assert. `mixed-k.patch`
  implements per-shard bitrates over a Kmax-padded shared trellis (uniform
  modules keep the single-shot path bit-for-bit). One real bug caught by the
  first garbage output: per-shard entries must use zeroed `shard_of_nb`
  (kernels index the activation shard; entries carry one suh row).
  Serve (GPU_UTIL=0.65, rest baseline): 15/15 suite green, all correct.
  Bench eager B=1: 1024 → TTFT 1.65 s, **decode 8.82** (out 26/26/30), e2e 4.59 s;
  2048 → **decode 8.79** (out 29×3), e2e 5.36 s. That's +36–38% vs 4bpw with no
  TTFT regression. 2 `results.csv` rows (`exl3xpu-esimd-mixedK`).
  PPL full WikiText-2 (72 chunks, 294,840 toks): **6.78, NLL 1.915** —
  vs 4bpw 6.36/1.850 (+0.42 PPL, +3.5% NLL). Small quality cost for ~45% fewer
  weight bytes; trellis coding works as advertised.
  Run: `logs/exl3-2bpw-20261007-124800/` (patches: `no-dpas.patch`,
  `k1-fallback.patch`, `mixed-k.patch` — all in git diff of `$EXL3_SRC`).
- [x] **E8.4 — Quality benchmark vs. Ternary Bonsai 2 27B PQ2_0.**
  Sequential side-by-side run over 16-prompt curated task suite (`scripts/bench_quality_16p.py`,
  greedy temp 0.0, max 2048 tokens). Qwen 2.2bpw scored **15/16 (93.8%)** vs Bonsai **14/16 (87.5%)**.
  Math 4/4 tied, Code 4/4 tied (100% unit assertions passed), Fact/Trap 4/4 tied. Qwen showed better
  reasoning convergence on constrained tasks where Bonsai's `xhigh` reasoning looped to the 2048 token ceiling.
  Artifacts: `results_qwen_2bpw_16p.json`, `results_bonsai_pq2_16p.json`.
- [x] **E8.5 — 16-prompt benchmark on Qwen3.8-27B EXL3 2.50bpw.**
  3-way comparison executed (`results_qwen_2.5bpw_16p.json`). 2.50bpw achieved **15/16 (93.8%)** with
  significantly faster reasoning convergence (890.9s suite total, −23% vs 2.20bpw and −34% vs Bonsai PQ2).
  WikiText-2 PPL improved from 6.78 to 6.57. Math 4/4, Code 4/4, Instruction 3/4, Fact/Trap 4/4.
- [x] **E8.6 — Download and evaluate Ternary Bonsai 2 27B PTQ1_0 (~1.58 true bpw).**
  Downloaded `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB / 5.6 GiB) from Hugging Face. Measured WikiText-2 perplexity
  via SYCL Level Zero `llama-perplexity` (`n_ctx=4096`, `flash-attn`): **PPL = 6.73 ± 0.18** (NLL ~1.906), virtually identical
  to EXL3 2.20bpw (6.78) and approaching EXL3 2.50bpw (6.57). Evaluated via 16-prompt suite on SYCL ternary server
  (`results_bonsai_ptq1_16p.json`). Achieved **15/16 (93.8%)**, matching Qwen EXL3 2.2/2.5bpw and outscoring Bonsai PQ2_0 (14/16)
  while consuming the smallest disk and VRAM footprint in the entire benchmark. Probed the single failed test (`format_no_letter_e`):
  demonstrated that `ctx=4096` was never truncated, and the test failed solely because `xhigh` reasoning consumed all 2048 generation tokens
  before emitting `</think>`. With `reasoning_effort=low`, PTQ1_0 solved the negative constraint in 93.9s (*"That salty, vast body of fluid is grand."*),
  achieving **16/16 (100%)**.

---

## Phase E6 — Report, Recommendation, and Reversibility

- [x] **E6.1 — Publish the implementation evidence.**
  Add an EXL3 section to `status.md` containing source SHA, patches, build hash,
  versions, exact launch/test commands, selected backend, suite/operator results,
  memory accounting, benchmark rows, unsupported paths, and remaining blockers.
  Append to `results.csv` only once metadata is validated; keep its existing
  schema and place additional detail in a linked run manifest/notes. Label
  measured values, estimates, and guide-sourced projections distinctly.
  DONE 2026-10-07: `status.md` EXL3 section added; 2 honest rows in
  `results.csv`; all evidence under `logs/exl3-4bpw-20261007-094207/`.

- [x] **E6.2 — Make an evidence-based recommendation.**
  Compare measured EXL3 results with Tiel/Bonsai only at comparable prompt/output
  lengths, concurrency, cache state, and eager/speculative modes. Different model
  quality and weight formats limit equivalence. Do not impose an arbitrary
  10 tok/s pass/fail threshold or present Bonsai projections as local measurements.
  Report: experimental usability, practical context/memory limits, performance,
  quality limitations, and whether further tuning is worthwhile. Do not replace
  the production serving configuration automatically.
  DONE 2026-10-07: **Option B selected: Retain EXL3 as High-Reasoning Fallback (Dual-Model Strategy)** —
  Measured decode 6.4–6.5 tok/s (eager 4bpw) / 8.8 tok/s (2.2bpw) / 9.9 tok/s (4bpw MTP K=1) is bandwidth-bound
  on dense 27B (~13 GB/tok vs MoE ~1.2 GB/tok on 103 GB/s Arc 140V) and not suited as primary daily driver vs
  Tiel-Coder-35B-A3B (27.7 tok/s). However, dense Qwen3.8-27B eliminates MoE routing degradation and maintains high
  reasoning quality (WikiText-2 PPL 6.36–6.78). Retained as designated fallback solution when MoE output is unsatisfactory.
  Operational isolation implemented: primary launcher `scripts/launch_vllm.sh` exports `EXL3_VLLM_PATCHES=0` to
  prevent plugin contamination, and dedicated root launcher `launch_vllm_qwen38_exl3.sh` serves fallback on port 8001.

- [x] **E6.3 — Document and test a scoped rollback.**
  Record how to stop the test processes, unset EXL3 overrides, and uninstall
  only the newly installed plugin if requested. Explain that general-plugin
  discovery can affect other vLLM launches; use per-launch plugin isolation
  where supported and keep fragile patches disabled outside the experiment.
  Restore only changes made for this work; do not reset the venv, discard
  unrelated git changes, delete checkpoints, or run broad process kills.
  DONE 2026-10-07: shutdown and runtime isolation procedures verified. Primary serving shielded by
  `export EXL3_VLLM_PATCHES=0` in `scripts/launch_vllm.sh`, eliminating runtime monkey-patching while keeping
  plugin installed for on-demand fallback launches via `launch_vllm_qwen38_exl3.sh`. Cleanup protocol verified:
  kill API + EngineCore PIDs, verify port free + GPUActive ~0.05 GiB, run `reclaim_gpu_cache.py`.

**Gate E6:** another person can reproduce the result or understand its blocker,
the production setup remains intact, and rollback is scoped and documented.

## Failure Triage

| Symptom | First checks / next action |
|---|---|
| Incorrect rows at M>8 | Check vector dispatch/row coverage and `EXL3_SMALL_M_MAX`; reproduce with E3.2. Do not raise the threshold without real tiling. |
| DPAS/IGC/illegal-instruction failure | Capture failing op and kernel, verify device capability, no-DPAS guards, loaded artifact hash, and all backend paths. A stale artifact is one possibility, not the only cause. |
| Build fails or succeeds suspiciously | Check interpreter/ABI/link paths, full compiler exit status, and fresh output artifact. `_C.so` existence alone is not success. |
| Quantization not recognized | Verify entry point, actual plugin loading in worker processes, EXL3 config registration, and absence of conflicting quantization implementations. |
| String-patch exception | Set `EXL3_VLLM_PATCHES=0` before imports in every server/worker process; check the pinned env contract. |
| Prefill failure with decode working | Isolate M>8 reconstruction, GEMM device support, scratch slicing, unquantized layers, and GDN/attention; never qualify decode-only as serving success. |
| OOM or memory pressure | Distinguish startup free-memory check, weight residency, scratch, and KV/GDN allocations. Reduce context/batch/scratch or cap KV explicitly if supported; choose utilization from measurements. |
| Gibberish, loops, or empty output | Check operator errors, fused loading/scales, K=6 head, tokenizer/template, both EOS IDs, and thinking/truncation. Differential-test qualified backends. |
| Suspiciously high decode throughput | Check real generated-token count, early EOS, reasoning-stream handling, timing denominator, errors, and prefix-cache protocol. |

For every failure: save the exact command, full logs, source/artifact revisions,
environment, and memory state; reproduce a minimal case; change one variable;
record `BLOCKED`/`FAILED` with a reason rather than checking the task complete.

## Dependency Order

```text
E0 preflight + checkpoint/memory audit
 -> E1 source pin + safe dispatch design
 -> E2 reliable build + registration
 -> E3 numerical/boundary tests (decode AND prefill)
 -> E4 isolated serving + correctness/stability
 -> E5 honest benchmark + optional KV/context sweeps
 -> E6 evidence report + recommendation + rollback
```

**First implementation action:** E0.1 environment snapshot; then E0.3 local
checkpoint audit. Do not begin model loading before operator gate E3 passes.

---

## Phase E9 — Staged Probe: 2.2bpw + FP8 KV + 32K Context (2026-10-07)

Run: `logs/exl3-2bpw-fp8-20261007-140828/`. One restart answering dtype +
context questions together (`MAX_LEN=32768`, `KV_DTYPE=fp8`, `GPU_UTIL=0.65`).

- [x] **E9.1 — Boot + KV accounting.** fp8 accepted ("Using fp8 data type to
  store kv cache" + accuracy warning). Pool: **230,589 tokens in 9.31 GiB
  (43.4 KB/tok)** vs 72,983 (146.6 KB/tok) fp16 — 3.2× more tokens, better
  than naive halving. Pool-bound max ≈ 230K tokens at 0.65 util. Required two
  launch attempts (first hit 17.92 < 18.58 GiB free; reclaim between attempts).
- [x] **E9.2 — Accuracy spot checks.** Short factual (Canberra, stop) clean.
  23.5K-token first-word retrieval ("alpha") correct with stop. No fp8
  accuracy signal on these probes (full-suite re-qualification not done).
- [x] **E9.3 — Perf tradeoff.** 1024-point: TTFT 4.64–5.91 s (vs 1.65 auto-KV),
  **decode med 7.53** (runs 8.81/7.53/6.80 — run variance, vs 8.82 auto).
  fp8 prefill markedly slower on XPU (matches the known FA2-fp8 gap the
  plugin's disabled `fp8kv_prefill` patch exists for). Verdict: fp8 buys
  **capacity** (3.2× tokens), not speed — decode −15%, prefill −65%.
  1 `results.csv` row. Server shut down, memory reclaimed.
