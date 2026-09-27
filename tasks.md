# Implementation Tasks: vLLM XPU Optimization for 258V / Tiel-Coder-35B-A3B

## Objective

Systematically close the decode throughput gap between vLLM XPU (19.15 tok/s baseline)
and the hardware roofline (~35 tok/s) on the Intel Core Ultra 7 258V, while preserving
correctness, stability, and serving usability. Every task changes **one variable**,
measures before and after, and records evidence in `logs/` and `results.csv`.

## Guiding Principles

- Correctness before performance — every change verified against the 5-prompt golden suite.
- One variable at a time — never stack untested changes.
- Evidence-driven — no claimed improvement without `results.csv` rows + server logs.
- Reversible — every experimental change must be env-gated or behind a flag.
- Memory-aware — host `MemAvailable` must stay ≥ 2.5 GiB; swap must not grow during steady state.

## Reference Numbers

| Metric | Baseline Value | Source |
|---|---|---|
| Eager decode (P=1024, B=1) | **19.15 tok/s** | `results.csv` row `Tiel compare A` |
| MTP decode (K=1, BF16 draft, P=1024) | **25.50 tok/s** | `results.csv` row `Tiel MTP (unfused draft)` |
| MTP decode (K=1, Qwen3.6, P=1024) | **26.83 tok/s** | `results.csv` row `MTP test` |
| Concurrency-2 aggregate | **22.51 tok/s** | `results.csv` row `Phase8 sweep` |
| Long-context decode (P=6660, B=1) | **18.81 tok/s** | `results.csv` row `Tiel compare D` |
| Resident weights (text-only, MXFP4) | **19.24 GiB** | `status.md` Gate 5 |
| Resident weights (+ MTP BF16 draft) | **21.57 GiB** | `status.md` Tiel MTP section |
| Host MemAvailable while serving | **2.1–3.5 GiB** | varies by config |
| AInfer greedy decode (reference) | **35.06–35.91 tok/s** | `../AInfer/STATUS.md` |
| AInfer dual-token MTP (reference) | **45.62–51.72 tok/s** | `../AInfer/improvement.md` I3.1 |
| llama.cpp Vulkan decode (reference) | **28.69 tok/s** | `fallback-comparison.md` |
| LPDDR5X-8533 measured stream BW | **103.03 GB/s** | `../AInfer/target_machine_identity.json` |

---

## Phase 11: MTP Draft Model Optimization (→ Gate 1 / Milestone M1)

### 11.1 Quantize MTP Draft Expert Weights to MXFP4

- [ ] **11.1.1** Audit the current Tiel MTP draft block composition.
      The `quantization_config.ignore` list in the Tiel checkpoint contains **785 `mtp.*` entries**,
      meaning all MTP weights are currently BF16. The MTP block is 1.69 GiB (785 tensors:
      19 expert layers × (gate_up_proj [256,1024,2048] + down_proj [256,2048,512]) + norms + router).
      - File: `models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4/config.json`
      - Evidence: `scripts/inspect_shard_sizes.py` → shard 32–33 sizes

- [ ] **11.1.2** Extend `scripts/quantize_tiel_mxfp4.py` to cover MTP expert weights.
      The unfused MTP experts (created by `scripts/unfuse_tiel_mtp.py`) are now per-expert
      gate/up/down BF16 tensors in shards 32–33. Apply the same MXFP4 quantization pipeline
      (group-32 symmetric E2M1 + E8M0 scales) using `torch.ops.vllm.xpu_mxfp4_quantize`.
      - Input: BF16 per-expert `mtp.0.mlp.experts.{i}.{gate,up,down}_proj.weight`
      - Output: `weight_packed` (uint8) + `weight_scale` (uint8 E8M0) tensors
      - Constraint: Must match the schema `CompressedTensorsW4A4Mxfp4.create_weights()` expects

- [ ] **11.1.3** Update `config.json` quantization ignore list.
      Remove the 768 MTP expert entries from the ignore list (keep MTP norms, router, and
      non-expert layers in the ignore list as BF16).

- [ ] **11.1.4** Re-index safetensors and verify with `scripts/audit_tiel_mxfp4.py`.
      Run SHA256SUMS verification, tensor count check, and spot-check dequantization
      (max rel-err < 0.15).

- [ ] **11.1.5** Smoke test the quantized-MTP checkpoint.
      ```bash
      MODEL=models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4 \
      MAX_SEQS=1 GPU_UTIL=0.74 \
        scripts/launch_vllm.sh tiel-mtp-mxfp4-smoke \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        --kv-cache-memory-bytes 500000000
      ```
      - Pass criteria: Server starts, `XPUExpertsMxFp4` selected for draft model,
        weights < 20.6 GiB, coherent generation on "The capital of France is" probe.
      - Log: `logs/server-tiel-mtp-mxfp4-smoke.log`

- [ ] **11.1.6** Benchmark quantized-MTP vs BF16-MTP draft.
      Run calibrated bench points: 1024/128/1 and 2048/128/1.
      - Expected: decode ≥ 26 tok/s (at least matching BF16 draft), MemAvailable +1 GiB.
      - Append rows to `results.csv` with notes `Tiel MTP MXFP4 draft`.

### 11.2 Speculative Depth Sweep (K=1 vs K=2)

- [ ] **11.2.1** Test `num_speculative_tokens=2` with BF16 draft (control).
      ```bash
      --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
      ```
      - If vLLM 0.30.0 supports K=2 for MTP: bench 1024/128/1, record acceptance rate
        from server metrics (if exposed) or estimate from throughput vs single-token baseline.
      - If unsupported (error): record the error, note that K=2 requires a future vLLM version,
        and close this task as BLOCKED.
      - Log: `logs/bench-tiel-mtp-k2.log`

- [ ] **11.2.2** If K=2 works: test with MXFP4 draft from 11.1.
      Compare: K=1 MXFP4 draft vs K=2 MXFP4 draft vs K=1 BF16 draft.
      - Success criterion: K=2 reaches **≥ 30 tok/s** on code completion prompts.

- [ ] **11.2.3** Measure acceptance rate across prompt domains.
      Run 5 representative prompts (factual, code, reasoning, instruction, long-context)
      and record per-prompt acceptance rate α and realized throughput.
      - AInfer reference: α ranges from 55% (arithmetic) to 94% (logic puzzle).
      - Log: `logs/bench-tiel-mtp-acceptance.log`

### 11.3 EOS Token Audit

- [ ] **11.3.1** Verify `eos_token_id` in the Tiel checkpoint `config.json`.
      Currently reads `eos_token_id: 248044` (text_config). Confirm this matches
      `<|endoftext|>` in the tokenizer vocabulary (248K vocab model).

- [ ] **11.3.2** Verify vLLM server uses the correct EOS tokens at runtime.
      Check server log for `eos_token_id` or `stop_token_ids` at startup.
      Confirm no stale Qwen2.5 tokens (151643, 151645) are referenced.
      ```bash
      grep -i "eos\|stop_token" logs/server-tiel-*.log | head -20
      ```

- [ ] **11.3.3** Verify `<|im_end|>` (248046) is used as a chat stop token.
      Send a chat completion request and confirm `finish_reason: "stop"` is triggered
      by `<|im_end|>`, not by a stale token.

**Gate 1 Exit Criteria:**
- [ ] Draft model quantized to MXFP4, resident weight memory reduced by ≥ 1.0 GiB.
- [ ] MTP decode throughput ≥ 28.0 tok/s on 1024-token prompt.
- [ ] 0 errors, golden suite passes, correct EOS handling.

---

## Phase 12: CPU & Driver Dispatch Optimization (→ Gate 2 / Milestone M2)

### 12.1 CPU P-Core Pinning

- [ ] **12.1.1** Identify the Lion Cove P-core IDs on this specific 258V.
      ```bash
      lscpu --extended | head -20
      # or: cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
      ```
      Verify cores 0–3 are Lion Cove (4.8 GHz max) and 4–7 are Skymont (3.7 GHz max).

- [ ] **12.1.2** Baseline inter-token jitter without pinning.
      Run `scripts/bench_envelope.py` point 1024/128/1 three times.
      Record p50, p95, p99, stddev of inter-token latency from the bench log.

- [ ] **12.1.3** Add `taskset` to `scripts/launch_vllm.sh`.
      ```bash
      # In launch_vllm.sh, change the exec line:
      exec taskset -c 0-3 python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >>"$LOG" 2>&1
      ```
      Gate with env var: `PINNED=${PINNED:-1}`, skip `taskset` when `PINNED=0`.

- [ ] **12.1.4** Benchmark with P-core pinning.
      Re-run the same 3× bench point. Record p50, p95, p99, stddev.
      - Success: p95 jitter reduced by ≥ 20% and/or TTFT reduced by ≥ 50 ms.
      - Log: `logs/bench-tiel-pinned.log`

- [ ] **12.1.5** Verify correctness with pinning.
      Run the 5-prompt golden suite under pinned config. All outputs must match
      non-pinned outputs (modulo the existing bit-nondeterminism caveat).

### 12.2 Single-Batch XPU Graph Capture

- [ ] **12.2.1** Investigate XPU graph capture configuration.
      Read vLLM source for XPU graph batch size selection:
      ```bash
      grep -rn "capture_sizes\|cudagraph_sizes\|xpu_graph" .venv/lib/python3.12/site-packages/vllm/ | head -30
      ```
      Determine if `VLLM_XPU_GRAPH_BATCH_SIZES` or similar env var controls capture sizes.

- [ ] **12.2.2** Test XPU graph with batch=1 only.
      Start server with:
      ```bash
      EAGER=0 VLLM_XPU_ENABLE_XPU_GRAPH=1 MAX_SEQS=1 \
        scripts/launch_vllm.sh tiel-graph-b1 \
        # Add any flag to restrict capture sizes to [1] if available
      ```
      Record: startup time, MemAvailable after init, GPUActive, GPUReclaim.
      - Pass: MemAvailable > 2.5 GiB (vs 1.48 GiB observed with batch [1,2,4]).

- [ ] **12.2.3** Benchmark XPU graph batch=1 decode.
      Run calibrated points 1024/128/1 and 2048/128/1.
      - Expected: 5–10% decode improvement over eager baseline (19.15 → 20–21 tok/s).
      - Append rows to `results.csv`.

- [ ] **12.2.4** Test XPU graph + MTP combined.
      If both batch-1 graph and MTP are compatible, benchmark the combination.
      This targets the multiplicative effect: graph dispatch savings + MTP bandwidth savings.
      - Expected: ≥ 28 tok/s.

- [ ] **12.2.5** If single-batch restriction is not configurable:
      File this as a vLLM feature request. Record the investigation in `status.md` and
      close this task as BLOCKED with the specific code path documented.

### 12.3 SoC Power & Thermal Profile

- [ ] **12.3.1** Read current power limits and GPU frequency.
      ```bash
      cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_power_limit_uw 2>/dev/null
      cat /sys/class/drm/card0/gt_cur_freq_mhz 2>/dev/null
      cat /sys/class/drm/card0/gt_max_freq_mhz 2>/dev/null
      cat /sys/class/drm/card0/gt_min_freq_mhz 2>/dev/null
      ```
      If files exist, record the values. If not (sysfs path differs on `xe` driver),
      search for the correct path:
      ```bash
      find /sys -name "*freq*" -path "*gt*" 2>/dev/null
      ```

- [ ] **12.3.2** Lock GPU frequency to max during benchmarking.
      If writable without sudo:
      ```bash
      echo 1950 > /sys/class/drm/card0/gt_min_freq_mhz
      ```
      If sudo required: document the command and expected effect, defer to user.

- [ ] **12.3.3** Benchmark with locked frequency vs default.
      Compare decode tok/s and inter-token jitter at locked 1.95 GHz vs default dynamic.
      Record thermal readings if available (`/sys/class/thermal/thermal_zone*/temp`).

**Gate 2 Exit Criteria:**
- [ ] P-core pinning active in launcher, jitter stddev reduced ≥ 20%.
- [ ] XPU graph investigation complete (either working with < 250 MiB overhead, or documented as blocked).
- [ ] GPU frequency behavior documented.

---

## Phase 13: Kernel & Attention Micro-Architecture (→ Gate 3 / Milestone M3)

### 13.1 Decode Attention EU Saturation Audit

- [ ] **13.1.1** Profile attention layer contribution to decode latency.
      AInfer found that at P=6720, 10 full-attention layers consume ~65 ms of 81 ms
      step time (~80% of decode). vLLM's FP8-KV long-context shows 18.81 tok/s (53 ms/step).
      Estimate attention fraction from the long-context vs short-context delta:
      - Short (P=1024): 19.15 tok/s → 52.2 ms/step
      - Long (P=6660): 18.81 tok/s → 53.2 ms/step
      - Delta: ~1.0 ms → attention is NOT the current bottleneck in vLLM at 7K
        (unlike AInfer where it dominated). Record this finding.

- [ ] **13.1.2** Inspect `vllm-xpu-kernels` decode attention dispatch.
      ```bash
      # Check if attention kernel parallelizes across KV tiles (not just heads)
      strings .venv/lib/python3.12/site-packages/vllm_xpu_kernels/*.so | grep -i "split\|tile\|chunk\|workgroup" | head -20
      # Check the Python-side dispatch:
      grep -rn "num_kv_splits\|split_k\|gqa.*decode" .venv/lib/python3.12/site-packages/vllm_xpu_kernels/ | head -20
      ```

- [ ] **13.1.3** If attention is NOT split across KV tiles:
      File observation that 16 query heads underutilize 64 VEs. Note this as a future
      upstream contribution opportunity. Document expected impact at 16K+ contexts.

- [ ] **13.1.4** If attention IS already split:
      Record the split factor and document that EU saturation is already addressed.

### 13.2 MoE Grouped GEMM Tile Investigation

- [ ] **13.2.1** Profile per-layer latency breakdown.
      Use `torch.profiler` or a simple timing wrapper around a single offline inference:
      ```python
      # scripts/profile_layers.py
      import torch
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
          # run one forward pass
      prof.export_chrome_trace("logs/trace-tiel-decode.json")
      ```
      Identify: embed, per-layer MoE, per-layer attention, per-layer GDN, LM head.

- [ ] **13.2.2** Compare per-layer MoE latency with AInfer's measured 5.92 ms/layer.
      AInfer measured: MoE triad (GateUp + SwiGLU + Down) = 5.92 ms/layer at B=256.
      vLLM's `cutlass_grouped_gemm_interface` profile should show similar or higher.

- [ ] **13.2.3** Document findings and determine if tile tuning is actionable.
      If per-layer MoE > 8 ms (significantly worse than AInfer), the tile parameters in
      `vllm-xpu-kernels` are suboptimal for 8 Xe-cores. File upstream issue with evidence.
      If ≈ 5–7 ms, tiles are already well-calibrated; close this task.

### 13.3 Prefill Workspace & Batch Token Sweep

- [ ] **13.3.1** Baseline: current `max_num_batched_tokens=4096`.
      Record prefill tok/s at P=1024 and P=2048 from existing data.

- [ ] **13.3.2** Test `max_num_batched_tokens=1024`.
      Start server, bench P=1024/128/1. Record TTFT, prefill tok/s, MemAvailable.
      Hypothesis: lower workspace, faster TTFT on short prompts, slower on long prompts.

- [ ] **13.3.3** Test `max_num_batched_tokens=2048`.
      Same bench points. Record.

- [ ] **13.3.4** Test `max_num_batched_tokens=8192` (if memory permits).
      Same bench points. Record. Watch MemAvailable carefully.

- [ ] **13.3.5** Select optimal `max_num_batched_tokens` and update `launch_vllm.sh` default.
      Decision criteria: best TTFT at P=1024 with MemAvailable > 2.5 GiB.

### 13.4 Long-Context Decode Qualification (8K+)

- [ ] **13.4.1** Benchmark 8192-token context with FP8 KV.
      ```bash
      MAX_LEN=8192 GPU_UTIL=0.78 \
        scripts/launch_vllm.sh tiel-8k-fp8 --kv-cache-dtype fp8
      ```
      Bench point: 8000/128/1. Record decode tok/s, MemAvailable, swap.

- [ ] **13.4.2** Benchmark 16384-token context with FP8 KV.
      Same setup with `MAX_LEN=16384`. Bench point: 15000/128/1.
      - Expected: decode stays > 17 tok/s (flat, based on existing 6.6K data).

- [ ] **13.4.3** Benchmark concurrency=2 at 4096-token context.
      Verify aggregate throughput ≥ 22.5 tok/s matches Phase 8 baseline.

**Gate 3 Exit Criteria:**
- [ ] Attention EU utilization documented (split or not-split).
- [ ] Per-layer MoE latency profiled and compared to AInfer reference.
- [ ] Optimal `max_num_batched_tokens` selected.
- [ ] 8K+ context sustained at ≥ 18 tok/s decode without swap growth.

---

## Phase 14: Serving & Agent Usability (→ Gate 4 / Milestone M4)

### 14.1 Client Disconnect Behavior

- [ ] **14.1.1** Test disconnect abort behavior.
      Start server. Send a streaming chat request with `max_tokens=512`.
      Disconnect the client (ctrl-C curl) after 5 tokens.
      - Check: Does the server log show the request was cancelled promptly?
      - Check: Is the next request serviced without delay?
      ```bash
      # Terminal 1: start server
      # Terminal 2:
      timeout 2 curl -N http://127.0.0.1:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Tiel","messages":[{"role":"user","content":"Write a long essay about recursion"}],"max_tokens":512,"stream":true}'
      # Terminal 3: immediately after disconnect
      time curl http://127.0.0.1:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Tiel","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}'
      ```
      Record: time-to-response of the second request.

- [ ] **14.1.2** If disconnect abort is slow (> 2s for next request):
      Investigate vLLM's async cancellation path for XPU. Document findings.

### 14.2 SSE Streaming & Long-Prefill Headers

- [ ] **14.2.1** Test SSE header timing with a long system prompt.
      Construct a request with ~4000 input tokens. Measure time from HTTP request
      to first SSE byte received.
      ```bash
      time curl -w "TTFB: %{time_starttransfer}\n" -N \
        http://127.0.0.1:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"Tiel","messages":[{"role":"system","content":"<4000 token system prompt>"},{"role":"user","content":"Hi"}],"max_tokens":10,"stream":true}'
      ```
      - Pass: TTFB < 5 seconds (headers sent before prefill completes).
      - If TTFB > 5s: document as a vLLM upstream limitation.

### 14.3 Tool Calling & Reasoning

- [ ] **14.3.1** Verify tool-calling server configuration.
      The LAN server was previously launched with:
      ```bash
      --enable-auto-tool-choice --tool-call-parser qwen3_xml
      ```
      Confirm this configuration still works after any optimization changes.
      Send a tool-calling request and verify `tool_calls` in the response.

- [ ] **14.3.2** Test `<think>` block handling.
      Send a reasoning-heavy prompt (e.g. math word problem). Check whether:
      - `<think>...</think>` appears in the streamed output
      - Token count includes thinking tokens (expected: yes, inline with `content`)
      - Document the current behavior and any `reasoning_content` field support.

- [ ] **14.3.3** Measure effective generation budget with thinking.
      If thinking tokens consume `max_tokens`, measure: for a request with
      `max_tokens=256`, how many visible (non-thinking) content tokens are produced?
      Document the recommended `max_tokens` sizing for agentic use.

### 14.4 Prefix Caching Effectiveness

- [ ] **14.4.1** Measure multi-turn prefix cache hit rate.
      Send 3 consecutive requests with the same system prompt but different user messages.
      Record TTFT for each request.
      - Expected: Request 2 and 3 have significantly lower TTFT than request 1
        due to prefix cache hits on the system prompt.

- [ ] **14.4.2** Document prefix caching behavior in `status.md`.
      Record: hit rate observed, TTFT improvement, any limitations.

**Gate 4 Exit Criteria:**
- [ ] Disconnect abort verified (next request within 2s).
- [ ] SSE header timing documented.
- [ ] Tool calling confirmed working post-optimization.
- [ ] Prefix caching hit rate measured and documented.

---

## Phase 15: Integration & Final Comparison

### 15.1 Best-Configuration Selection

- [ ] **15.1.1** Compile all optimization results into a comparison table.
      Columns: configuration, decode tok/s, MTP tok/s, MemAvailable, startup time, caveats.

- [ ] **15.1.2** Select the production configuration.
      Decision criteria (priority order):
      1. Zero errors / OOM over 10+ consecutive requests.
      2. Highest decode throughput.
      3. MemAvailable ≥ 2.5 GiB.
      4. Startup time < 60 seconds.

- [ ] **15.1.3** Encode the selected config as defaults in `scripts/launch_vllm.sh`.

### 15.2 Updated Fallback Comparison

- [ ] **15.2.1** Re-run the Phase 10 comparison points against AInfer and llama.cpp.
      Use the optimized vLLM configuration. Same points: 1024/128/1 and 6660/128/1.
      Update `fallback-comparison.md` with the new vLLM numbers.

- [ ] **15.2.2** Update `results.csv` with final optimized rows.

### 15.3 Documentation

- [ ] **15.3.1** Update `status.md` with Phase 11–15 outcomes.
- [ ] **15.3.2** Update `plan.md` milestone table with measured results.
- [ ] **15.3.3** Commit all changes with a descriptive message.

---

## Failure Triage Checklist (unchanged from original)

When a test fails, capture evidence before changing the environment.

1. Save the exact command and complete stdout/stderr.
2. Save `pip freeze` and relevant environment variables.
3. Save free memory, swap usage, GPU status, and kernel messages.
4. Reproduce once with the same configuration.
5. Reduce to the smallest failing model-load or operator test.
6. Search upstream issues using the exact exception and pinned versions.
7. Change only one component or flag.
8. Record the result, including failed attempts, in `status.md`.

## results.csv Schema (unchanged)

```text
timestamp,git_revision,model_revision,backend,torch_version,vllm_version,kernel_version,driver_version,oneapi_version,context_tokens,input_tokens,output_tokens,max_num_seqs,max_num_batched_tokens,memory_utilization,eager,ttft_s,prefill_tps,decode_tps,total_latency_s,peak_memory_gb,swap_used_gb,result,notes
```

---

## Task Summary & Dependency Graph

```
Phase 11: MTP Draft Optimization ─────────────────────── Gate 1 (M1: >32 tok/s MTP)
  11.1 Quantize MTP draft to MXFP4 ──┐
  11.2 Speculative depth sweep (K=2) ─┤── Gate 1
  11.3 EOS token audit ──────────────┘

Phase 12: CPU & Dispatch ──────────────────────────────── Gate 2 (M2: >23 tok/s eager)
  12.1 P-core pinning ───────────────┐
  12.2 XPU graph batch=1 ────────────┤── Gate 2
  12.3 Power/thermal profile ────────┘

Phase 13: Kernel & Attention ──────────────────────────── Gate 3 (M3: >27 tok/s eager)
  13.1 Attention EU saturation audit ┐
  13.2 MoE GEMM tile investigation ──┤── Gate 3
  13.3 Prefill batch token sweep ────┤
  13.4 Long-context qualification ───┘

Phase 14: Serving & Agent ─────────────────────────────── Gate 4 (M4: Production ready)
  14.1 Disconnect abort ─────────────┐
  14.2 SSE streaming headers ────────┤── Gate 4
  14.3 Tool calling & reasoning ─────┤
  14.4 Prefix caching audit ─────────┘

Phase 15: Integration & Comparison ────────────────────── Final
  15.1 Best-config selection ────────┐
  15.2 Updated fallback comparison ──┤── Release
  15.3 Documentation ────────────────┘
```

**Critical path:** 11.1 → 11.2 → 12.1 → 12.2 → 15.1
**Parallelizable:** 11.3 ∥ 12.3 ∥ 13.1–13.4 ∥ 14.1–14.4
