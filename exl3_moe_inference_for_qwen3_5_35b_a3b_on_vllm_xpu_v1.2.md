# Plan: EXL3 MoE Inference for Qwen3.5-35B-A3B on vLLM XPU

**Version:** 1.2  
**Prepared:** October 23, 2026  
**Primary prototype hardware:** user-confirmed Core Ultra 7 258V, Arc 140V integrated GPU (8 Xe2 cores, LPDDR5X-8533), 32 GB shared system memory.  
**Starting implementation:** fork of `sybil-solutions/exl3xpu`, not a new inference engine.  
**Input:** existing EXL3 checkpoint of Qwen3.5-35B-A3B; quantizer porting is out of scope.  
**Status:** proposed engineering plan. No kernels, model loads, or benchmarks were executed for this plan.  
**Revision note (1.2):** review pass. Fixes a stale dense-model reference (P0-06), a self-inconsistent materialization estimate, an overstated small-$M$ claim, and an unverified `profile_run` assumption; adds missing roofline terms (GDN state traffic, compute ceiling), OS/allocation-limit gates, prior-art survey, bandwidth microbenchmark (P0-12), and unit/memory-budget discipline. Architecture numbers (hidden 2048, 40 layers, 256 experts, top-8, MoE/shared intermediate 512, 3 GDN : 1 full-attention pattern) were re-checked against Transformers documentation and mirrored configs on October 23, 2026; the checkpoint hash, repository size, and exl3xpu archive date were not independently re-verified.

## 1. Objective and scope

Deliver a pinned vLLM XPU integration that loads an existing EXL3 Qwen3.5-35B-A3B checkpoint, executes routed and shared experts correctly, and serves text requests on the 140V/32 GB platform within a measured memory envelope.

The attached `EXL3_Intel_Arc_Infer.md` supplies the useful principles: reference reconstruction before fusion, resident experts before offload, explicit synchronization, measured capacity, and separate numerical/performance gates. Its generic CUDA-to-Intel rewrite roadmap is superseded here by reuse of the existing Intel plugin. [A1][S1]

### Initial supported configuration, proposed

- Two pinned checkpoint revisions: a lower-bitrate branch (3.00bpw/3.50bpw) used for bring-up through Gate 2 and first full-model load, and the published 4.00bpw branch as the quality target, qualified at Gate 3 only if the measured memory budget allows (see §7). Verify every tensor's actual codebook/bitrate for both; a branch label is not a per-tensor guarantee.
- Single device, tensor/expert/pipeline parallel sizes of one, one request, short text-only context.
- Existing dense EXL3 linear operations reused for attention projections, shared-expert matrices, and output head where compatible.
- Dedicated EXL3 expert loading and execution adapter at the pinned vLLM MoE boundary.
- Correctness-first eager execution (`--enforce-eager`); optional compilation/graph capture added only after the baseline passes.
- Standard supported attention and recurrent-state paths; no inherited long-context, speculative-decoding, or multimodal promises.

**Deferred:** quantizer changes, MTP/speculative decoding, vision/video, LoRA, distributed expert execution, expert load balancing, long-context optimization, alternate codebooks/recipes, CPU expert computation, and expert streaming/offload. Deferred does not mean silently ignored: requested unsupported features must fail clearly.

## 2. Verified starting points and unresolved questions

### Source-backed observations

1. `exl3xpu` is an Intel ESIMD/vLLM plugin with reference reconstruction, dense matrix operations, and a published dense-model validation on Arc Pro B70. vLLM supplies scheduling, paged KV cache, and GDN/attention operations. The repository was archived October 12, 2026; development needs an owned fork. [S1]
2. The plugin exposes linear-method integration, fused projection loading, and per-tensor storage metadata. Its documented small-$M$ and large-$M$ paths differ; `ops.py` includes a reusable reconstructed-weight buffer and B70-tuned defaults. These paths need auditing for 140V (8 Xe2 cores, smaller L2, LPDDR5X bandwidth) and expert concurrency, not wholesale replacement. [S1][S2][S6]
3. The official target config has hidden size 2048, 40 layers, 256 routed experts, top-8 routing, routed/shared intermediate size 512, SiLU activation, and a mix of linear and full attention. It also declares recurrent-state dtype settings. The selected EXL3 checkpoint's own config remains authoritative for loading. [S3]
4. Current upstream vLLM Qwen3.5 code selects `Qwen3NextSparseMoeBlock`; current Qwen3Next code constructs a `FusedMoEFactory`, a router, and a shared-expert gate, with routing normalization controlled by configuration/defaults. These are discovery leads, not guaranteed APIs in the plugin's tested vLLM build. [S4][S5]
5. Existing EXL3 weights are published. The inspected 4.00bpw branch lists about 19.4 GB of repository content and three weight shards; its observed revision is `8a68a1aca688b768853bad4d926c0136ff62286c`. Repository size is not runtime peak memory, and a branch name is not proof of uniform per-tensor bitrate. [S7][S8]
6. Prior art exists and is not yet reviewed in this plan: a third-party `vllm-exl3` plugin reports routed-expert EXL3 serving with batched/grouped expert kernels on NVIDIA (GLM-5.3-Flash), and the same ecosystem reports a B70 "expert server" companion and NVMe expert offload. These are leads, unverified for Qwen3.5 or for Xe2; they may supply a vLLM MoE seam, checkpoint-layout handling for experts, or a grouped-kernel design to copy before writing new code. [S9][S10]

### Questions Gate 0 must resolve

- Which vLLM XPU build both supports the target hybrid architecture and exposes a usable custom MoE quantization boundary?
- What are the exact Level Zero / SYCL total memory (`globalMemSize`) and single-allocation (`maxMemAllocSize`) ceilings on the 140V OS/driver stack, and do they require OS/kernel tuning (on Linux, e.g. the TTM page limit `ttm.pages_limit` and related `xe`/`i915` settings; confirm which apply to the installed kernel) to expose $\ge 24\text{ GB}$ to XPU?
- Which OS is the target? vLLM's XPU backend is, to my knowledge, Linux-only (container/native); Windows WDDM and WSL paths should be treated as unsupported until proven. Fix the OS, kernel, and driver stack at Gate 0 rather than keeping both options open.
- What is the default single-allocation limit on this stack (commonly 4 GiB on Intel GPUs unless relaxed-allocation-limit options are enabled, e.g. `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` for the SYCL/PyTorch XPU path; confirm the exact variable for the pinned oneAPI/PyTorch), and does anything in the plan exceed it (large KV tensors at longer context, embedding/LM-head tensors of ~1 GB at FP16)?
- Does the selected checkpoint store expert matrices individually, fused, or in another layout, and are `gate_proj` and `up_proj` quantized with independent EXL3 input/output Hadamard sign vectors ($s_u, s_v$) per expert or as a pre-fused `gate_up` matrix?
- Can all required 140V ESIMD paths compile without register spill and pass reconstruction/linear tests on Lunar Lake Xe2?
- Does the pinned MoE API accept a custom method, or is a small maintained factory/loader patch necessary?
- Can compressed weights, model states, vLLM `profile_run` workspaces, runtime, and the OS coexist in 32 GB without sustained paging?

Do not claim a complete native audit, MoE support, 140V qualification, or performance parity before these questions are tested.

## 3. Proposed architecture

```text
Pinned EXL3 checkpoint + config + tensor-storage metadata
    -> shard-by-shard streaming loader (explicit mmap/cache reclamation) and coverage validator
    -> ordinary linear modules: existing EXL3 linear method
    -> MoE expert module: new EXL3 expert adapter
         router/top-k from pinned model semantics
         token-to-expert dispatch (device-side indexing/sorting in optimized path)
         per-expert EXL3 gate/up operations (respecting per-matrix su/sv Hadamard boundaries)
         SiLU gate/up activation product (in post-Hadamard reconstructed domain)
         per-expert EXL3 down operation
         weighted scatter/reduction
         shared-expert contribution, gated exactly once
    -> existing vLLM hybrid attention/recurrent-state execution
    -> existing scheduler / cache management / text API
```

The exact ownership of router execution, top-$k$ selection, weighting, shared-expert execution, and reduction must follow the pinned vLLM interface. Some interfaces pass logits; others pass selected IDs/weights or delegate selection internally. The diagram is conceptual, not a new API signature. [S4][S5]

### Integration rules

- Preserve router precision, score function, top-$k$ normalization, weight application point, and shared-gate semantics from the pinned implementation. Do not assume softmax/sigmoid placement from the config alone.
- Keep router and shared-gate weights in their checkpoint-prescribed representation. Never accidentally route them through the EXL3 matrix loader because their module names look similar.
- Apply EXL3 input/output randomized Hadamard transforms ($s_u, s_v$ sign flips + FWHT) and scales at each matrix's documented boundary. Gate and up projections may have distinct $s_u$ metadata; do not share input transforms without proving $s_{u,\text{gate}} == s_{u,\text{up}}$, and never fuse across the output transform before the nonlinear $\text{SiLU}(\text{gate}) \odot \text{up}$ product.
- Preserve expert identity across checkpoint names, runtime slots, fused gate/up order, and output aggregation. Disable expert remapping/load balancing initially.
- If the MoE factory rejects EXL3, fail explicitly rather than constructing an unquantized expert allocation or ignoring packed tensors.
- Prefer a registered custom MoE method where supported. If unavailable, isolate a minimal version-pinned factory/model-loader patch with tests. Avoid global monkey-patches and copied whole model implementations.

All rules above are proposed design constraints derived from the identified integration boundary. [S2][S4][S5]

## 4. Phase 0: Pin, audit, and prove the platform

**Goal:** isolate model integration, hardware portability, and checkpoint compatibility before combining them.

- [ ] **P0-01** Fork `exl3xpu`; pin its commit, vLLM/XPU build or image digest, PyTorch, compiler, driver/runtime, Transformers, and tokenizer versions.
- [ ] **P0-02** Inspect the installed vLLM code, not only upstream main; document the actual MoE factory/base class, create/load/post-load/apply hooks, router ownership, shared-expert contract, and expert mapping.
- [ ] **P0-03** Pin the primary 4.00bpw EXL3 checkpoint commit (`8a68a1a...`) AND a lower-bitrate branch (e.g., 3.00bpw or 3.50bpw; record its revision hash too). Use the lower-bitrate branch for bring-up through Gate 2 (faster loads, less paging risk); the 32 GB shared-memory budget and OS carve-out limits decide at Gate 3 whether 4.00bpw is viable. Inventory tensor keys, shapes, dtypes, codebooks, integer $K$ values, per-tensor storage metadata ($s_u, s_v$ Hadamard vectors/scales), nonquantized weights, whether the LM head and embeddings are stored above the nominal bitrate, and whether expert `gate`/`up` projections share input transforms. Record the actual tensor names used by the reference runtime (to my understanding `trellis`, `suh`, `svh`, and codebook flags such as `mcg`/`mul1`; verify against the pinned ExLlamaV3 source) and the Hadamard block size (EXL3 is understood to use 128-wide blocked Hadamard transforms rather than a full-dimension FWHT; if so, the "FWHT" notation in this plan means that blocked transform and the register/SLM fusion in §8 is cheaper than it looks).
- [ ] **P0-04** Confirm the selected checkpoint is compatible with the reference ExLlamaV3 version. The published model card names a minimum version; verify against the exact selected checkpoint. [S7]
- [ ] **P0-05** Query 140V Level Zero capabilities (`maxMemAllocSize`, `globalMemSize`, sub-slice/Xe-core count = 8, L2 size); verify the OS allows pinning $\ge 24\text{ GB}$ for XPU allocations. Compile/load the existing ESIMD extension with Xe2-LNL target flags and verify register-spill-free compilation rather than copying B70 tuning.
- [ ] **P0-06** Run existing reconstruction/linear tests on representative dense and expert shapes ($N, K \in \{512, 2048\}$); loading the full 35B checkpoint is not required for this hardware gate. (The B70 validation cited in [S1] concerns a different, dense model; it is not evidence for 140V or for MoE.)
- [ ] **P0-07** Probe the target vLLM hybrid attention/recurrent-state paths using bounded fixtures or a small architecture-matching test. Avoid loading the full floating-point 35B model on this machine.
- [ ] **P0-08** Establish a deterministic reference oracle: use a standalone PyTorch/CPU exact EXL3 trellis dequantizer on the 140V host for single-expert/single-layer fixtures, and generate a static artifact of end-to-end teacher-forced logits/routing traces from reference CUDA `ExLlamaV3` (on an external GPU node) so Gate 2/3 validation does not depend on local CUDA hardware.
- [ ] **P0-09** Define exact reconstruction checks and floating-point/quality tolerances before tuning. Record reference hardware/software and checksums.
- [ ] **P0-10** Write `docs/compatibility_manifest.md` (including memory/bandwidth roofline sanity checks), `docs/moe_api_audit.md`, and `docs/checkpoint_schema.md`; record unresolved blockers.
- [ ] **P0-11** Survey prior art ([S9], [S10], and the ExLlamaV3 reference MoE path) for: the vLLM MoE seam used, expert checkpoint layout handling, grouped/multi-matrix expert GEMM design, and licensing. Decide reuse vs. reimplementation before Phase 1.
- [ ] **P0-12** Measure the platform roofline: achievable read/copy bandwidth on the 140V (large and L2-sized working sets), achievable ALU throughput for a trellis-decode + MAC microkernel, and the achieved bandwidth of the existing EXL3 linear kernel on a dense $2048\times 512$ matrix at $M=1$. These replace the theoretical 136 GB/s figure in later claims.

**Gate 0:** existing EXL3 matrix paths pass on 140V, Level Zero memory limits (`globalMemSize`, `maxMemAllocSize`) are verified sufficient, the pinned target model backend is viable, a checkpoint mapping is specified, a supported MoE extension strategy is chosen, the host OS/kernel configuration is fixed, and achievable bandwidth/compute (P0-12) is measured. If any fails, stop and scope that blocker independently.

## 5. Phase 1: Expert loading and one expert MLP

**Goal:** verify expert storage and computation without routing complexity.

- [ ] **P1-01** Extend quantization dispatch to recognize the pinned MoE module type; preserve the ordinary linear path and explicitly reject unsupported modes.
- [ ] **P1-02** Implement expert-aware parameter/buffer registration without allocating full FP16 expert banks as placeholders. Keep per-layer expert storage below `maxMemAllocSize` (avoid single giant multi-gigabyte contiguous device buffers across layers). Sizing: 256 experts $\times$ 3 matrices $\times$ $2048\times 512$ $\times$ 0.5 B ≈ 0.40 GB per layer at 4.00bpw, ≈ 16 GB over 40 layers (roughly 80% of the checkpoint), so per-layer arenas are comfortably under a 4 GiB single-allocation limit while a whole-model buffer is not.
- [ ] **P1-03** Map each layer/expert/projection checkpoint entry into a packed descriptor: storage, offsets, shape, $K$/codebook, $s_u/s_v$ transforms/scales, and projection order.
- [ ] **P1-04** Preserve per-expert metadata even if projections are grouped in a packed arena; never concatenate codes as ordinary dense rows without a validated layout conversion.
- [ ] **P1-05** Stream checkpoint tensors shard-by-shard with bounded host staging; explicitly release `safetensors` mmap handles, run `gc.collect()`, and flush XPU caching allocators (`torch.xpu.empty_cache()`) between shards to avoid transient host+device double-buffering on unified memory. Reject missing, duplicated, unknown, and mismatched expert entries, and maintain a complete consumed/ignored-key report.
- [ ] **P1-06** Reconstruct `gate`, `up`, and `down` matrices for selected first/middle/last expert IDs; compare exactly with the trusted decoder where applicable.
- [ ] **P1-07** Execute one expert with existing linear kernels; validate `gate`/`up` order, input/output Hadamard transform boundaries, $\text{SiLU}(\text{gate}) \odot \text{up}$, `down` projection, and dtype against a dequantized reference.
- [ ] **P1-08** Test $M=0, 1, 2, 4, 8, 16, 32, 64, 128, 256$ (hot experts and the dequant+XMX crossover), nonuniform token counts, and the existing `ops.py` small/large-$M$ boundary; handle empty work without invalid dispatches. Note that with 256 experts and top-8 routing, decode is strictly $M=1$ per active expert and a 256-token prefill averages $M=8$ per active expert (512 tokens averages 16, and real routing is skewed so hot experts see far more); calibrate the small/large-$M$ crossover specifically for 8 Xe2 cores on 140V.
- [ ] **P1-09** Audit shared reconstruction workspace for reentrancy. Serialize initial expert calls or provide event-safe per-inflight workspaces before overlap. Ensure small-$M$ execution does not materialize full FP16 expert weights to system DRAM unnecessarily.
- [ ] **P1-10** Preserve dense-model regression tests; add `tests/test_expert_loader.py` and `tests/test_expert_mlp.py`.

**Gate 1:** one expert MLP and expert metadata mapping are correct across representative expert IDs and batch sizes, respecting Level Zero allocation caps and avoiding persistent expanded-weight banks.

## 6. Phase 2: Correct complete MoE block

**Goal:** add routing and shared-expert semantics before optimizing dispatch.

- [ ] **P2-01** Implement a simple eager baseline that gathers token rows by selected expert, calls the verified expert MLP, and scatters weighted outputs back to original token positions.
- [ ] **P2-02** Reuse the pinned model's routing implementation where possible. If the adapter receives top-$k$ results, do not recompute routing; if it receives logits, match the exact reference selection/normalization policy.
- [ ] **P2-03** Apply routing weights exactly once at the specified location; verify input-versus-output weighting semantics for the nonlinear expert path.
- [ ] **P2-04** Execute the shared expert and its learned gate exactly once, respecting whether the pinned framework already owns their execution/addition.
- [ ] **P2-05** Compare fixed-routing outputs first to isolate expert math; then compare free routing to detect router precision/top-$k$ differences.
- [ ] **P2-06** Test top-$k$ ties/near-ties, zero-token experts, one hot expert, uniform routing, skewed routing, repeated expert use across tokens, token permutation, and empty input.
- [ ] **P2-07** Verify all eight selected experts contribute for each normal token; no expert capacity/drop policy may be introduced silently.
- [ ] **P2-08** Compare router outputs, each expert result, routed sum, shared output, and final MoE output separately.
- [ ] **P2-09** Integrate one complete hybrid decoder layer; compare residuals/norms, attention or recurrent-state outputs, and MoE output.
- [ ] **P2-10** Add `tests/test_moe_routing.py`, `tests/test_moe_block.py`, and a layer fixture report.

**Gate 2:** the complete MoE block and representative linear/full-attention decoder layers pass agreed tolerances. A Python-loop baseline is acceptable here; it is not the performance release.

## 7. Phase 3: Bounded full-model serving on 140V

**Goal:** establish correct, memory-bounded text inference before grouped-kernel work.

### Memory and bandwidth policy

Use actual checkpoint tensor bytes, not 3B active parameters or nominal bpw, to budget resident storage. The inspected 4.00bpw repository is about 19.4 GB, but loading duplicates, parsed metadata, workspaces, KV cache, recurrent states, logits, runtime allocations, and OS memory are additional. The published model uses both attention types, so a KV-only capacity estimate is incomplete. [S3][S8]

**Proposed memory accounting:**

```text
peak physical demand = simultaneously resident distinct backing for:
  compressed weights + nonquantized components + shard loading staging
  + expert/dense workspaces (especially during vLLM profile_run) + activations/logits
  + full-attention KV cache + linear-attention recurrent/convolution (GDN) states
  + Level Zero / SYCL runtime + OS/background demand
```

CPU and XPU allocations share the 32 GB physical pool on the user's target. Track framework live/reserved memory, Level Zero heap limits (`globalMemSize` / `maxMemAllocSize`), OS availability/RSS/PSS where meaningful, and paging pressure; do not sum overlapping views blindly. Shared memory does not guarantee zero-copy tensors or eliminate driver allocation limits.

**Units.** Hugging Face sizes are decimal GB (19.4 GB ≈ 18.1 GiB); installed RAM and Level Zero limits are binary. Convert before comparing, and report budgets in GiB.

**Budget sketch (to be measured, not assumed):** 32 GiB physical, minus OS/desktop (≈2–4 GiB; less if headless) and the Python/PyTorch/oneAPI/vLLM runtime (≈2–4 GiB), leaves roughly 24–28 GiB. The 4.00bpw weights take ≈18.1 GiB, leaving single-digit GiB for shard staging, workspaces, KV, GDN state, and logits. If the iGPU-addressable limit defaults to a fraction of RAM (some integrated-GPU stacks default to about half; verify), 4.00bpw cannot be resident until that limit is raised, which is why bring-up uses the lower-bitrate branch.

**Roofline sanity check (decode, $M=1$)** — derived from config dimensions; estimates only:
- **Routed + shared experts:** 40 layers $\times$ (8 routed + 1 shared) $\times$ 3 matrices $\times$ $(2048 \times 512)$ ≈ 1.13 B params/token ≈ 566 MB at 4.00bpw.
- **Non-expert weights:** ≈ 1.8 B params ≈ 0.9 GB at 4.00bpw, made up of roughly 1.3 B in attention/GDN projections (10 full-attention layers ≈ 28 M each, 30 GDN layers ≈ 34 M each), the LM head ≈ 0.51 B (248,320 $\times$ 2048, read in full every token and possibly stored above 4 bpw), and the router ≈ 21 M. Embedding lookup reads one row, not the table.
- **GDN recurrent state (omitted in v1.1):** 30 layers $\times$ (32 value heads $\times$ 128 $\times$ 128) $\times$ 4 B (fp32 state) ≈ 63 MB resident, read and written each token ≈ 125 MB/token. Full-attention KV reads at 2K context add only ≈ 40 MB.
- **Total ≈ 1.6 GB/token.** At 95–110 GB/s effective, that is 14.5–17 ms, a bandwidth-only ceiling of ≈ 60–70 tok/s.
- **Compute check:** trellis decode plus MAC costs several ALU ops per weight over ≈ 3 B weights/token. On 8 Xe2 cores this may be ALU-bound before it is bandwidth-bound, so treat 60–70 tok/s as an upper bound, not a target; P0-12 measures the real ceiling.
- **Materialization penalty:** writing reconstructed FP16 weights to DRAM costs ≈ 0.5 B read + 2 B write + 2 B read ≈ 4.5 B/param, about 9$\times$ the fused path (v1.1 said 4$\times$, which contradicts its own 10–18 tok/s estimate). The penalty shrinks only if a 2 MB FP16 expert matrix stays L2-resident; measure rather than assume.

- [ ] **P3-01** Audit optional vision/MTP tensors and model construction. Use a supported text-only loading mode where available; otherwise count retained components. Do not silently discard required weights.
- [ ] **P3-02** Bound checkpoint shard staging, parsed metadata duplication, and workspace caches; verify that shard-by-shard mmap release and allocator flushes prevent loading peaks from exceeding physical RAM.
- [ ] **P3-03** Start with one sequence, proposed 2K maximum context, explicit `--max-num-batched-tokens 256` (or `512`), `--enforce-eager`, no MTP, no prefix caching, and conservative `--gpu-memory-utilization` (or explicit `--num-gpu-blocks-override`). Bounding `max_num_batched_tokens` is mandatory because vLLM's startup `profile_run` pushes that many dummy tokens through the whole model. Do not assume which experts the dummy run touches: depending on the dummy inputs it may concentrate on a few experts (large $M$ per expert, large reconstruction workspace) or spread across many (many launches); test both regimes. Also pass `--max-num-seqs 1` explicitly, and note that `--gpu-memory-utilization` is computed against the device-reported memory on a unified-memory part and can double-count host usage, so prefer `--num-gpu-blocks-override` once the budget is measured.
- [ ] **P3-04** Audit both full-attention KV cache allocation and `Qwen3Next` linear-attention/GDN recurrent state allocation during vLLM engine initialization. Disable optional INT8 prefill, attention overrides, compilation/graph capture, and shared-expert fusion initially. Retain the model's native gated/shared semantics.
- [ ] **P3-05** Build `models/qwen3.5-35b-a3b-exl3-4bpw-140v/model.yaml` using only flags verified against the pinned server; do not copy the B70 262K-context/16-sequence recipe.
- [ ] **P3-06** Measure cold-load, `profile_run`, and warm-serving peaks with a configurable system reserve; reduce prefill/context/workspace when needed and abort cleanly under sustained paging.
- [ ] **P3-07** Compare teacher-forced logits and NLL/perplexity with the same EXL3 checkpoint in the trusted runtime artifact from P0-08. Use the original floating-point model only as a secondary quantization-quality baseline.
- [ ] **P3-08** Test prompt prefill followed by cached decode, chunked versus unchunked prefill, request reset, sequential independent requests, and KV + recurrent state/cache isolation.
- [ ] **P3-09** Verify tokenizer/chat-template/EOS behavior and a minimal text API request; generated plausible text alone does not pass correctness.
- [ ] **P3-10** Record load time, TTFT, per-token latency, prefill throughput, sustained decode, memory, thermal/power conditions, and known limitations.
- [ ] **P3-11** Set expectations and timeouts for the Python-loop baseline: a 256-token chunk can touch most of the 256 experts in each of 40 layers, i.e. up to ≈ 30 k expert-matrix calls (each several kernels plus host syncs), and decode issues ≈ 1 k expert-matrix calls per token. It will be launch-bound and far below the roofline; size startup/health-check timeouts and test durations accordingly, and do not treat baseline speed as a Phase 4 regression reference.

**Gate 3:** one pinned full checkpoint serves text correctly on 140V within the measured envelope. If the 4.00bpw resident model cannot fit due to OS/driver carve-outs, test the pre-audited 3.00bpw/3.50bpw checkpoint from P0-03 or use larger validation hardware, documenting the exact memory boundary.

## 8. Phase 4: Expert-aware performance optimization

**Depends on Gate 3 or an explicitly documented larger-hardware functional gate.** Optimize only measured bottlenecks.

1. **Eliminate host syncs in dispatch (Critical bottleneck):** A Python loop over active experts that calls `.item()` or `nonzero()` on the CPU forces at least one host–device stall per MoE layer (≥ 40 per token, and many more if done per expert). Profile routing, token sorting/gather, per-expert calls, transforms, matrix work, and scatter separately. Replace host syncs with device-side token sorting/binning (`argsort`/histogram) and either a fixed top-$k$ device-indexed loop or indirect dispatch.
2. **Fast path for $M=1$ Decode (8 active experts/layer):** During single-sequence decode, exactly 8 distinct routed experts execute with $M=1$ per layer. Instead of general 256-expert histogram binning, (reuse the grouped design found in P0-11 if one exists) implement a dedicated $M=1$ path that gathers the 8 selected expert descriptors and runs a batched/grouped $M=1$ fused-dequant GEMV across the 8 experts in a single ESIMD kernel launch for `gate`/`up`, followed by gated SiLU, and a single grouped launch for `down`.
3. **Chunked Prefill & Small-$M$ Dominance:** Recognize that under uniform routing, 256 tokens average $M=8$ and 512 tokens average $M=16$ per expert, so for chunks up to 256 most active experts have $M \le 16$, while at 512 about half exceed it and real (skewed) routing puts hot experts far higher. Group by actual expert token count, not total request token count. Prioritize grouped small-$M$ fused kernels over full-matrix dequantize-to-FP16 + XMX GEMM, unless an individual hot expert exceeds the empirically measured dequant+XMX crossover point on 140V.
4. **Transform Algebra & Gate/Up Handling:** In EXL3, each matrix applies an input randomized Hadamard transform ($x \leftarrow \text{FWHT}(x \odot s_u)$) and an output transform ($y \leftarrow \text{FWHT}(y) \odot s_v$). Only share the input FWHT between `gate` and `up` if P0-03 proves $s_{u,\text{gate}} == s_{u,\text{up}}$ (or fuse the two independent sign-flips + FWHT into a single pre-pass kernel). Never fuse across the output transform, as $\text{SiLU}(\text{gate}) \odot \text{up}$ must be evaluated in the post-Hadamard reconstructed domain.
5. **LPDDR5X Bandwidth, L2 Residency, & Workspace Discipline:** On 140V ($\approx 136\text{ GB/s}$ peak bandwidth), reading 8 routed + 1 shared expert across 40 layers transfers $\approx 566\text{ MB/token}$ in expert weights alone. Ensure ESIMD kernels fuse trellis unpack + scale/Hadamard + dot-product in registers/SLM without writing intermediate FP16 expert weights out to LPDDR5X. Bound sorting, activation, reconstruction, and output buffers with explicit lifetimes.
6. **Compiler integration:** Provide correct custom-op schema/fake shape implementations and capture-safe allocation/dispatch where supported. Test eager versus compiled output/state equivalence.
7. **Optional INT8:** Re-enable only after separate quality evaluation. Existing dense-model INT8 measurements are not evidence of MoE quality parity. [S1]

**Benchmark proposal:** prompts 128, 512, 2K, then 8K only if admitted by memory; concurrency 1 first, then 2/4 if feasible; synthetic skew/uniform expert loads plus real text. Report median/tail latency, active-expert counts, tokens per expert, KV/recurrent-state footprint, effective memory bandwidth utilization (GB/s), and warm/cold conditions. No fixed tok/s or NVIDIA-parity target is asserted. Before Phase 4 starts, declare go/no-go thresholds as a fraction of the *measured* (P0-12) bandwidth/compute roofline, not the theoretical figure.

**Gate 4:** an optimized path improves a declared workload over the correctness baseline without numerical, memory, or reliability regression. Keep the baseline as an oracle/fallback, not a silently selected slow path.

## 9. Proposed repository changes

Paths below are proposed additions unless already present; exact class/API names follow Gate 0.

| Area | Planned change |
| --- | --- |
| `exl3xpu/vllm_plugin.py` | Recognize the audited MoE boundary, preserve linear dispatch, reject unsupported modes. |
| `exl3xpu/moe.py` | New expert descriptors, shard-safe streaming loader, reference dispatch, and adapter lifecycle. |
| `exl3xpu/moe_ops.py` | New expert-aware operation dispatch, device-side token binning, $M=1$ fast path, and workspace policy. |
| `csrc/` | Reuse existing primitives first; add grouped $M=1$ / small-$M$ dispatch/gather/scatter or expert-aware ESIMD kernels after profiling. |
| `exl3xpu/ops.py` | Audit/restrict workspace sharing, tune small/large-$M$ threshold for 8 Xe2 cores (140V), and expose reusable linear functionality without dense regressions. |
| `models/.../model.yaml` | New bounded text-only 140V recipe (`max_num_batched_tokens`, memory bounds); exact checkpoint and environment manifest. |
| `tests/` | Expert loader, reconstruction, Hadamard boundaries, routing, shared gate, block/layer/logit/state and dense regression tests. |
| `bench/moe_profile.py` | Proposed per-stage, bandwidth-efficiency, and expert-distribution profiler. |
| `bench/roofline.py` | P0-12 achievable bandwidth / decode-ALU microbenchmarks. |
| `docs/` | API audit, checkpoint schema, roofline/memory report, gate reports, measured support matrix. |

Do not implement a new quantizer, attention engine, scheduler, or general checkpoint conversion system as part of this plan.

## 10. Acceptance, risks, and decisions

### Acceptance hierarchy

- **Discrete representation:** exact key coverage, expert IDs, bit extraction, permutation, and packing interpretation.
- **Weight reconstruction:** exact equality to the trusted decoder where the same reconstruction contract permits it.
- **Linear/expert/block numerics:** justified dtype-aware absolute/relative errors and distortion distributions across $s_u/s_v$ Hadamard and trellis boundaries, not blanket bit-exact floating-point outputs.
- **Routing:** fixed-route isolation, score/normalization checks, tie handling, shared-gate accounting, and end-to-end free-routing comparison.
- **Model quality:** teacher-forced logits/KLD, NLL/perplexity, selected tasks, and KV + recurrent-state equivalence.
- **Runtime:** repeated loads/requests, stream safety, memory pressure across `profile_run` and serving, clear unsupported-mode errors, and reproducible recipe.

### Main risks and responses

| Risk | Response / decision point |
| --- | --- |
| Pinned vLLM lacks an EXL3 MoE extension seam | Gate 0: isolate minimal factory/loader patch or choose a compatible build; do not assume main matches installed code. |
| Level Zero `maxMemAllocSize` or OS shared-RAM carve-out blocks 19.4 GB load | Gate 0 / Gate 3: tune OS driver limits (`xe`/`i915` or WDDM), keep buffers per-layer below `maxMemAllocSize`, flush shard mmaps, or switch to the pre-audited 3.00bpw/3.50bpw branch. |
| vLLM `profile_run` OOMs or times out in the eager expert loop (routing of dummy tokens is input-dependent) | Gate 3: bound `--max-num-batched-tokens` (256/512), `--max-num-seqs 1`, serialize in-place expert workspaces, test concentrated and spread routing, raise timeouts (P3-11). |
| Unsupported or unfixed OS stack (vLLM XPU is Linux-oriented; iGPU memory cap unknown) | Gate 0: fix OS/kernel/driver, measure addressable memory, and decide the bring-up bitrate before any MoE work. |
| Duplicating existing EXL3 MoE work | P0-11: review prior art before Phase 1; reuse or consciously diverge. |
| 140V kernel or hybrid-state incompatibility | Separate platform tests before MoE integration; verify Xe2-LNL register pressure and preserve trusted fixtures. |
| Expert Hadamard transforms ($s_u, s_v$) or `gate`/`up` order corrupted | Gate 1: exact reconstruction and one-expert fixtures; never fuse across output FWHT before $\text{SiLU}(\text{gate}) \odot \text{up}$. |
| Routing/shared expert applied twice | Gate 2: intermediate-level ownership and contribution tests. |
| Host `.item()` syncs and per-expert launches dominate decode | Gate 4: device-side indexing and grouped $M=1$ / small-$M$ ESIMD kernel launches without DRAM weight materialization. |
| Shared scratch races | Serialize baseline, then event-safe per-inflight buffers and concurrency tests. |
| Archived plugin and moving vLLM API | Owned fork, exact pins, CI, documented patches and maintenance owner. |

### Planning allowances, not delivery estimates

Proposed bounded investigation budgets: Phase 0, 1–2 person-weeks; Phase 1, 1–2 additional person-weeks; Phase 2, 1–3 additional person-weeks. These are discretionary experiment caps, not evidence-backed forecasts. Re-estimate full-model integration and optimization at each gate. Stop/re-scope if a platform/API blocker consumes the budget. Track person-effort separately from calendar duration.

**Priority order:** platform/API/Level-Zero proof -> expert loader -> one expert -> one MoE block -> complete layer -> bounded full model (`profile_run` safe) -> grouped small-$M$ optimization. This keeps the project smaller than a new engine while testing the genuinely new MoE work early.

## 11. References and evidence provenance

Reviewed October 20, 2026 (Updated October 23, 2026). Default branches may change. Pin a coherent runtime/checkpoint combination before coding. The source observations above do not establish a completed port; all phases, paths, budgets, and operating settings are proposals.

- **[A1] User attachment:** `EXL3_Intel_Arc_Infer.md`, reviewed October 20, 2026. General inference feasibility, validation, memory, and staged-implementation principles.
- **[S1] exl3xpu README/repository:** scope, archive status, B70 validation, kernel layout, runtime ownership, and published feature restrictions. https://github.com/sybil-solutions/exl3xpu
- **[S2] exl3xpu plugin:** existing linear integration and checkpoint-storage handling. https://github.com/sybil-solutions/exl3xpu/blob/main/exl3xpu/vllm_plugin.py
- **[S3] Official target config:** model architecture, expert counts, dimensions, hybrid layer pattern, and state settings. https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
- **[S4] vLLM Qwen3.5 model:** current upstream model/loader and MoE-block selection; verify against pinned installed build. https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py
- **[S5] vLLM Qwen3Next model:** current upstream router/shared-expert/MoE factory integration; not a stable API guarantee. https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py
- **[S6] exl3xpu ops:** small/large-M dispatch, reconstruction workspace, and backend controls. https://github.com/sybil-solutions/exl3xpu/blob/main/exl3xpu/ops.py
- **[S7] Published EXL3 target model card:** existing bitrate branches and reference-runtime requirement. https://huggingface.co/turboderp/Qwen3.5-35B-A3B-exl3
- **[S8] Published 4.00bpw checkpoint tree:** observed shard inventory and revision; runtime peak must be measured. https://huggingface.co/turboderp/Qwen3.5-35B-A3B-exl3/tree/4.00bpw
- **[S9] Third-party `vllm-exl3` plugin (releases):** reported routed-expert EXL3 serving and batched expert kernels on NVIDIA; lead only, not reviewed for Qwen3.5 or Xe2. https://github.com/vcruz305/vllm-exl3/releases
- **[S10] GLM-5.3-Flash EXL3 offload recipes (sybil-solutions wiki):** reported expert offload and B70 expert-server companion; lead only. https://github.com/sybil-solutions/glm-flash-lite/wiki
