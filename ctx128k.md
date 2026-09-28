# Feasibility & Strategy: 100k–128k Context with vLLM & Tiel-Coder-35B on Intel Core Ultra 7 258V

> **Target Platform:** Intel Core Ultra 7 258V (Lunar Lake, Arc 140V Xe2 iGPU, 32 GB on-package LPDDR5X-8533 unified memory)  
> **Target Model:** `symrex/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4` (Sparse MoE Hybrid: 40 layers = 30 GDN linear attention + 10 full attention, 256 routed experts / 8 active, ~35.95B total / ~3B active params)  
> **Software Stack:** CachyOS Linux, Intel Compute Runtime 26.35, oneAPI Level-Zero 1.17+, vLLM 0.30.0+xpu, vllm-xpu-kernels 0.1.14.1

---

## Executive Summary

The potential to maximize context to **100k–128k tokens** with vLLM and Tiel Coder 35B on this 32 GB machine is **exceptionally high**—unusually high for a 35B-class model on an integrated mobile SoC.

While standard dense or MoE architectures (LLaMA-3, DeepSeek, Mixtral) require 15 to 30+ GB of VRAM solely for their KV cache at 100k tokens, Tiel Coder's **hybrid GDN (Gated DeltaNet) linear-attention architecture** reduces the KV cache footprint by **$4\times$ to $5\times$**, fitting a full 100k–128k KV cache into **~1.0 to 1.34 GB of VRAM** when using FP8 quantization.

---

## 1. The Architectural Secret: Why 100k KV Fits in ~1 GB

### Hybrid Attention Decomposition
In standard transformers, every layer computes full causal self-attention, storing key and value tensors for every sequence position across all layers.

In Tiel Coder 35B / Qwen3.6 35B:
1. **30 layers are Linear Attention (GDN / SSM):** Their recurrent internal state size is **$O(1)$** with respect to sequence length. The state is allocated once at startup (~60–80 MB total) and **does not grow whether processing 1,000 or 100,000 tokens**.
2. **Only 10 layers are Full Attention:** Specified by [`full_attention_interval: 4`](file:///home/yanchun/Projects/vllm.xpu/models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4/config.json).
3. **Low Head Count:** Each full-attention layer uses Grouped-Query Attention (GQA) with only **2 KV heads** of dimension **256**.

### Exact KV Cache Arithmetic

$$\text{KV per token (FP8)} = 10 \text{ layers} \times 2\text{ (K and V)} \times 2\text{ KV heads} \times 256\text{ dim} \times 1\text{ byte} = \mathbf{10,240\text{ bytes} \approx 10.0\text{ KB / token}}$$

$$\text{KV per token (BF16)} = 10 \text{ layers} \times 2 \times 2 \times 256 \times 2\text{ bytes} = \mathbf{20,480\text{ bytes} \approx 20.0\text{ KB / token}}$$

| Context Length | Standard 30B–70B Transformer (FP8) | **Tiel Coder 35B KV (FP8)** | Tiel Coder KV (BF16) |
|---|---|---|---|
| **32,768 (32k)** | ~1.6–2.5 GB | **~0.33 GB** | ~0.66 GB |
| **65,536 (65k)** | ~3.2–5.0 GB | **~0.67 GB** | ~1.34 GB |
| **100,000 (100k)** | ~5.0–8.0 GB | **~1.02 GB** | ~2.05 GB |
| **131,072 (128k)** | ~6.5–10.5 GB | **~1.34 GB** | ~2.68 GB |

### Empirical Validation in this Repository
* **vLLM Level-Zero Reservation ([`server-server-06-fp8kv.log`](file:///home/yanchun/Projects/vllm.xpu/logs/server-server-06-fp8kv.log#L83)):**  
  Reserving a 2.0 GiB FP8 KV cache allocated **170,738 tokens** ($2.61\times$ concurrency for 65k).
* **Bare-Metal Level-Zero Validation ([`../AInfer/STATUS.md`](file:///home/yanchun/Projects/AInfer/STATUS.md#L138)):**  
  A 128k context arena in KV8 format was verified on this exact hardware at **1,282.5 MiB (~1.25 GiB)**.

---

## 2. Physical Memory Budget on the 32 GB Package

The Core Ultra 7 258V features 32 GB of unified LPDDR5X-8533 memory shared across the CPU, GPU, OS, and desktop session:

```
[=========================== Physical System RAM (32.0 GB) ===========================]
[ OS + Desktop (~6.0-8.0 GB) ][ Model Weights (19.24 GB) ][ 128k KV (~1.34 GB) ][ Work (~0.8 GB) ]
```

### Breakdown of Inference Footprint (Text-Only, No MTP)
* **Model Weights (MXFP4):** **19.24 GiB** resident in VRAM.
* **100k–128k KV Cache (FP8):** **~1.05 to 1.34 GiB**.
* **Intermediate Activation / Workspace:** **~0.80 GiB** (bounded by chunked prefill).
* **Total Inference Commitment:** $19.24 + 1.34 + 0.80 \approx \mathbf{21.38\text{ GiB}}$.

**Verdict:** On a quiet host session where desktop applications (browser tabs, Steam) leave $\ge 23$ GiB free memory before startup, **128k context fits entirely in physical RAM without triggering host swapping**.

---

## 3. Performance Projections at 100k–128k

### A. Sustained Decode Throughput (~15 to 17 tok/s)
In standard architectures, decode speed drops drastically at long contexts because reading a 15–20 GB KV cache saturates memory bandwidth.

In Tiel Coder 35B at 100k context:
* Active MoE weights read per token: **~1.17 GB**.
* 100k FP8 KV cache read per token (across 10 layers): **~1.02 GB**.
* Total DRAM bandwidth consumed per decode token: $1.17 + 1.02 \approx \mathbf{2.19\text{ GB/token}}$.
* On the 258V's measured **103 GB/s** memory bus:
  $$\text{DRAM transfer time} = \frac{2.19\text{ GB}}{103\text{ GB/s}} \approx \mathbf{21.3\text{ ms}}$$
  Adding software dispatch overhead (~25–30 ms), decode throughput will only gently drop from **19.1 tok/s** at short context to **~15.0–16.5 tok/s at 100k–128k context**. (Measured Phase 13.4 decode was **18.76 tok/s at 15k context**, remaining essentially flat).

### B. Cold Prefill Latency (~75 to 110 seconds)
Prefill on 100k tokens on a 17–30W integrated GPU is compute-intensive:
* **The 30 Linear Layers:** Scale strictly linearly ($O(N)$). At ~1,700 tok/s measured prefill throughput, these layers require ~45 seconds.
* **The 10 Full Attention Layers:** Scale quadratically ($O(N^2)$). Across 10 layers, attention dot-products add ~35–55 seconds.
* **Cold TTFT:** A full cold 100k prompt will take **~75 to 110 seconds** (1.2 to 1.8 minutes).
* **Multi-Turn with Prefix Caching:** If 80k–90k tokens represent a static codebase or document, vLLM's block-level prefix caching skips those tokens on subsequent turns, dropping TTFT down to **< 3 seconds**.

### C. RoPE & Retrieval Integrity
In [`models/Tiel-Coder-35B.../config.json`](file:///home/yanchun/Projects/vllm.xpu/models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4/config.json):
* `max_position_embeddings: 262144` (**256k native context limit**).
* `rope_parameters.rope_theta: 10000000` (**10 million base frequency**).
* `partial_rotary_factor: 0.25`.

The model was natively pre-trained/fine-tuned for long contexts with a 10M base frequency. No external context window extension (such as YaRN interpolation) is needed to maintain mathematical coherence out to 128k.

---

## 4. Required Configuration & Trade-Offs

To maintain stability within the 32 GB shared memory envelope at 100k–128k context, three specific trade-offs are mandatory:

1. **Disable MTP (Speculative Decoding):**
   * Speculative decoding requires draft weights (0.47–1.69 GiB) and speculative verification KV block allocations.
   * Disabling MTP frees ~1.1 GiB of VRAM to house the 100k KV blocks.
2. **Strict Concurrency = 1 (`--max-num-seqs 1`):**
   * All available KV memory blocks must be dedicated to a single active stream.
3. **Mandatory FP8 KV Cache (`--kv-cache-dtype fp8`):**
   * BF16 KV cache would require ~2.68 GB at 128k, risking Level-Zero startup rejection or OS swapping.
4. **Chunked Prefill Workspace Sizing (`--max-num-batched-tokens 2048`):**
   * Bounding prefill chunks prevents activation spikes from competing with the KV cache.

---

## 5. Production Launch Command for 100k–128k Context

Before launching, close background applications (browsers, Steam) to ensure host `MemAvailable` is $\ge 23$ GiB.

```bash
#!/usr/bin/env bash
# 128k Context Launch Script for Tiel Coder 35B on Intel Arc 140V (258V)

# 1. Drain driver reclaim pool
python scripts/reclaim_gpu_cache.py

# 2. Launch vLLM server
MODEL=models/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4 \
SERVED=Tiel-Coder-35B-A3B \
MAX_LEN=131072 \
MAX_SEQS=1 \
GPU_UTIL=0.74 \
BATCHED_TOKENS=2048 \
SPEC="" \
  ./scripts/launch_vllm.sh longctx-128k \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 1400000000 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

---

## 6. Summary Scorecard

| Dimension | 4k Context (Interactive MTP) | 128k Context (Document/Repo Analysis) |
|---|---|---|
| **Primary Goal** | Real-time chat & code autocompletion | Full repository / multi-document analysis |
| **Decode Speed** | **27.69 tok/s** (MTP $K=2$) | **~15.0–16.5 tok/s** (Eager) |
| **KV Cache Size** | ~0.04 GB (500 MB pin) | **~1.34 GB** (1.4 GB pin) |
| **Model Weights** | 19.71 GiB (with MXFP4 draft) | **19.24 GiB** (text-only, no draft) |
| **Cold TTFT** | ~1.0 second | **~75–110 seconds** |
| **Cached TTFT** | ~1.0 second | **< 3.0 seconds** (via prefix caching) |
| **Host Stability** | High (2.2 GiB free headroom) | Tight but stable ($\ge 1.5$ GiB free headroom) |
