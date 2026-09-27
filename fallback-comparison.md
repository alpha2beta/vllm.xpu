# Fallback comparison — vLLM XPU vs AInfer vs llama.cpp (2026-09-27)

Phase 10 deliverable. Same box (Core Ultra 7 258V, Arc 140V, 32 GB),
same model family (Tiel-Coder-35B-A3B, Qwen3.5-MoE 40L / 256 experts / 8 active),
matched workload where possible: **1024-token prompt, 128 generated tokens,
batch 1**. Backends ran serialized (one resident at a time).

## Headline (decode throughput, higher is better)

| backend | quant | prompt 1024 → gen 128 | decode tok/s | prefill tok/s |
|---|---|---|---|---|
| **vLLM XPU** (0.30.0+xpu, eager) | MXFP4 (ours) | server, conc 1 | **19.15** | ~950 |
| **AInfer** (`bench_258v`, L0 recorded) | INT4-g128 (`.binfer`) | CLI harness | **27.34** | 294.7 |
| **llama.cpp Vulkan** (`llama-bench`, ngl 99) | Q4_K_M (APEX-Compact GGUF) | `-p 1024 -n 128` | **28.69** | 296.3 |
| OpenVINO GenAI | INT4 IR (Ornith, different model) | server :8080 | **not measurable** | — |

Raw logs: `logs/bench-tiel-01.log` (vLLM), `/tmp/opencode/ainfer_tiel_1024.{log,json}`,
`/tmp/opencode/llama_tiel_1024.{log,json}`.

## Long context (~6.7–7.2K prompt → 128 tokens, batch 1)

| backend | prompt | decode tok/s | prefill tok/s | notes |
|---|---|---|---|---|
| **vLLM XPU** | 6660 | **18.81** (−2% vs 1K) | 2028 | util 0.78, maxlen 7168 (`logs/bench-tiel-04.log`) |
| **llama.cpp Vulkan** | 7168 | **29.12** (+1.5% vs 1K) | 204.0 | `/tmp/opencode/llama_tiel_7k.json` |
| **AInfer** | 7168 | **11.54** (−58% vs 1K) | 63.3 | `/tmp/opencode/ainfer_tiel_7k.{log,json}` |

The long-context ranking **inverts the short-context story**: vLLM and llama
hold their decode rates flat into 7K (18.8 / 29.1), while AInfer's recorded
decode loop falls off a cliff (27.3 → 11.5) — its full-attention KV traffic
over 7K positions dominates, and prefill drops 295 → 63 for the same reason.
vLLM's prefill actually *improves* with length (950 → 2028 tok/s: chunked
prefill amortizes launch overhead). Operational cost of vLLM at 7K: needs
util 0.78 + maxlen 7168 (8192-len config OOMs the KV block planner at 0.74),
GPUActive 22.7 GiB, MemAvailable dipping to 2.3 GiB — workable but tighter
than the 4K envelope.

## Reading the table

- **llama.cpp Vulkan edges out AInfer on decode here (28.69 vs 27.34, +4.9%)**
  at P=1024 — the reverse of AInfer's committed short-prompt comparison
  (35.54 vs 29.33 at P≈21). Both are the same class: decode is MoE-weight
  traffic-bound and both move ~17–19 GB per token at similar rates.
- **vLLM is ~30% slower on decode (19.15)** — the generic paged-attention +
  Python-volunteer MoE path costs vs the two hand-rolled GPU loops. Its
  advantage is the serving stack (continuous batching: conc 2 → 20.12 tok/s
  aggregate; prefix caching; OpenAI API), not single-stream speed.
- **Prefill flips the ranking**: vLLM ~950 tok/s vs ~295 for both AInfer and
  llama at P=1024 (vLLM's chunked prefill + Flash Attention scales far better
  with prompt length; AInfer/llama numbers are steady-state loop rates).
- Memory resident: AInfer 18.22 GiB static; vLLM 21.35 GiB GPUActive (weights
  19.24 + KV/prefill workspace); llama Vulkan ~17.4 GiB weights + transient.

## Caveats (do not quote one number without these)

1. Different quantizations per backend (MXFP4 vs INT4-g128 vs Q4_K_M) — same
   architecture and active params, but quality/speed are quant-sensitive.
2. Different harnesses (vLLM server + template vs raw CLI prompts) — TTFT and
   prefill include different overheads; decode tok/s is the comparable column.
3. Short-prompt orderings do not transfer to P=1024 (shown above).
4. OpenVINO: server code exists (`~/Projects/openvino`, GPU, :8080) but the
   Ornith INT4 IR was never downloaded to this box (no `~/.cache/ornith-openvino`;
   17.7 GiB fetch), and it is a *different fine-tune* — any future number is
   arch-family-level only. AInfer's own assessment
   (`openvino_reference_assessment.md`) also deferred an OV export of Tiel.
