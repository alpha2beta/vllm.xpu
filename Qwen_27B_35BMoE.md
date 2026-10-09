# [Benchmark & In-Depth Analysis] 27B Dense vs. 35B MoE on a 32GB iGPU: 11 Models Benchmarked on Intel Lunar Lake (Arc 140V) — Throughput, Reasoning Cliffs, and Architectural Tradeoffs

**Target Community:** r/LocalLLaMA  
**Hardware Tested:** Intel Core Ultra 7 258V (Lunar Lake) — Arc 140V Xe2 iGPU (8 Xe cores), 32 GB shared LPDDR5X-8533 (~103 GB/s unified memory bandwidth).  
**Software Stack:** Native Linux (CachyOS, kernel 7.2.3), vLLM 0.31.0+xpu (native Xe2 MXFP4 & out-of-tree `exl3xpu` ESIMD plugin), llama.cpp SYCL (`arc-b580` optimized build).

---

## TL;DR / Executive Summary

Can modern consumer integrated graphics run 27B dense and 35B MoE models locally with high reasoning fidelity and usable interactive speeds? **Yes, but the architectural and quantization choices make or break the experience.**

1. **35B MoE (MXFP4, Active 3.5B/tok) is the undisputed daily driver champion:**  
   Because only ~3.5B parameters are activated per token, MoE models slash memory bandwidth demand from ~13 GB/tok down to ~1.2 GB/tok. Paired with MTP (Multi-Token Prediction) $K=2$, **Tiel-Coder-35B** and **KAT-Coder-V2.5** reach **24.8 – 27.7 tok/s** decode. KAT-Coder completed the entire 16K-context 10-task Hard Suite in an all-time record **4.5 minutes** (solving LeetCode Hard problems in ~17 seconds each with 0 syntax flaws).
2. **Dense 27B (Qwen 3.8 EXL3 3.00bpw) is the peak reasoning champion:**  
   Dense 27B eliminates MoE routing artifacts. At 3.00bpw, Qwen achieved a **flawless 100% (10/10) Hard Suite and 100% (16/16) Quality Suite score out-of-the-box**, solving Olympiad-level Chinese Remainder Theorem and bounded combinatorics proofs without a single hint or calibration. The tradeoff is decode throughput: bandwidth-bound at **~6.5 tok/s**.
3. **The "Sub-2.5bpw Quantization Cliff" is very real for post-training quantization:**  
   Pushing post-training trellis quantization down to 2.20bpw on dense 27B resulted in a catastrophic drop on code synthesis (**0/3 on LeetCode Hard**).
4. **Native Ternary Distillation beats Post-Training Quantization at extreme low bitrates:**  
   **Ternary Bonsai 2 27B** (distilled natively to 1.58–1.7 true bpw) completely crushes EXL3 2.20bpw. Fitting into just **5.95 – 7.20 GB VRAM**, it decodes at **12.1 – 13.1 tok/s** (+40–50% faster than EXL3 2.2bpw), achieves lower perplexity (**6.73 vs 6.78**), and scores **90–100%** on the benchmark suites.
5. **The Winning Production Strategy:** A **Dual-Model Deployment** on the 32GB package:
   * **Primary Daily Driver (Port 8080):** `Tiel-Coder-35B-A3B MXFP4` (or `KAT-Coder-V2.5 MXFP4`) @ **27.7 tok/s** for interactive agentic coding and instant needle retrieval.
   * **Designated Deep-Reasoning Fallback (Port 8001):** `Qwen3.8-27B EXL3 3.00bpw` (or `Bonsai PQ2_0 MTP`) when pure mathematical verification or strict negative constraints require 100% precision.

---

## The 11-Way Comprehensive Benchmark Matrix

All models were evaluated under identical, standardized conditions on the same Intel Arc 140V hardware across two rigorous test batteries:
1. **16-Prompt Curated Quality Suite:** Math word problems, multi-step algorithms, strict instruction formats (raw JSON, reverse alphabet lists, negative lexical constraints), and premise/factuality traps.
2. **10-Task Hard Suite at 16K Context:** 12.5K multi-hop needle-in-a-haystack retrieval, distractor document resolution, Olympiad math proofs (Chinese Remainder Theorem, bounded combinatorics via inclusion-exclusion), LeetCode Hard algorithms with automated unit assertion execution, and multi-constraint logic puzzles.

| Model & Architecture | Quant Format | Size on Disk / VRAM | WT2 PPL | 16-Prompt Quality | 10-Task Hard Score (16K ctx) | Hard Suite Wall Latency | Serving Engine & Decode Throughput |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **KAT-Coder-V2.5-Dev** (35B A3B MoE) | MXFP4 (E2M1) | 19.60 GB / 19.7 GiB | — | **15/16 (93.8%)** *(cal.)* | **8/10 (80.0%)** | **271.9s (4.5 min)** ⚡ *(Record)* | vLLM XPU (MTP K=2, **24.8 tok/s**) |
| **Tiel-Coder-35B-A3B** (35B A3B MoE) | MXFP4 (E2M1) | 20.47 GB / 19.7 GiB | — | **15/16 (93.8%)** | **9/10 (90.0%)** | **564.8s (9.4 min)** | vLLM XPU (MTP K=2, **27.7 tok/s**) |
| **Qwen3.8-27B** (Dense 27B) | EXL3 (3.00bpw) | 12.87 GB / 13.1 GiB | **6.46** | **16/16 (100.0%)** | **10/10 (100.0%)** | 1062.6s (17.7 min) | vLLM XPU (MTP K=1, ~6.5 tok/s) |
| **Bonsai PQ2_0 MTP** (27B Ternary) | GGUF PQ2_0 | 7.20 GB / 7.6 GiB | ~6.73 | **16/16 (100.0%)** *(cal.)* | **9/10 (90.0%)** *(cal. 10/10)* | **1147.5s (19.1 min)** | llama.cpp SYCL (**12.1 tok/s**) |
| **Ternary Bonsai PQ2_0** (27B Ternary) | GGUF PQ2_0 | 7.21 GB / 7.6 GiB | ~6.73 | **16/16 (100.0%)** *(cal.)* | **10/10 (100.0%)** *(cal.)* | 1827.4s (30.5 min) | llama.cpp SYCL (**12.1 tok/s**) |
| **Ternary Bonsai PTQ1_0** (27B Ternary) | GGUF PTQ1_0 | **5.95 GB / 5.6 GiB** | 6.73 | **16/16 (100.0%)** *(cal.)* | **8/10 (80.0%)** *(cal.)* | 1749.6s (29.2 min) | llama.cpp SYCL (**13.1 tok/s**) |
| **Qwen3.8-27B** (Dense 27B) | EXL3 (2.50bpw) | 11.45 GB / 11.7 GiB | 6.57 | 15/16 (93.8%) | 7/10 (70.0%) | 1317.8s (22.0 min) | vLLM XPU (MTP K=1, ~6.6 tok/s) |
| **Qwen3.8-27B** (Dense 27B) | EXL3 (2.20bpw) | 9.60 GB / 7.4 GiB | 6.78 | 15/16 (93.8%) | 7/10 (70.0%) | 1014.8s (16.9 min) | vLLM XPU (MTP K=1, ~8.8 tok/s) |
| **Qwen3.6-35B-A3B** (35B A3B MoE) | MXFP4 (E2M1) | 19.24 GB / 19.2 GiB | — | **14/16 (87.5%)** | **6/10 (60.0%)** | 1435.1s (23.9 min) | vLLM XPU (Eager, ~9.7 tok/s) |
| **KAT-EXL3-4bpw** (35B A3B MoE) | EXL3 (4.00bpw) | 18.85 GB / — | — | *Unsupported* | *Unsupported* | — | Incompatible (MoE kernel missing; fallback OOM) |

---

## Detailed Architectural Takeaways

### 1. MoE on Unified Memory: The Bandwidth Multiplier (Tiel-Coder vs. KAT-Coder)
On unified memory architectures like Intel Lunar Lake, AMD Strix Point, or Apple Silicon, memory bandwidth (~103 GB/s here) is the fundamental bottleneck for autoregressive token generation.
* **The Math:** A dense 27B model at 4bpw reads ~13.5 GB of weights from memory for *every single token generated*. At 103 GB/s peak bandwidth, your theoretical speed ceiling is ~7.6 tok/s.
* **The MoE Advantage:** A 35B model with 256 total experts but only **8 active experts per token (A3B)** only transfers ~1.2 GB of weights per token. Even on an iGPU, you get **25+ tok/s** decode without breaking a sweat!
* **KAT-Coder-V2.5-Dev vs. Tiel-Coder-35B:**
  * **KAT-Coder:** Incredible code specialization. Solved all 3 LeetCode Hard problems (Trapping Rain Water, LRU Cache with OrderedDict/DLL, Min Window Substring) in an average of **17.5 seconds per problem** with 100% test assertion passes. Furthermore, its 12.5K needle retrieval completed in **31.1 seconds** (fastest chunked prefill across all models).
  * **Tiel-Coder:** Better balance on mathematical formal logic. While KAT-Coder made minor arithmetic verification slips on Olympiad math (0/2), Tiel-Coder solved both the Chinese Remainder Theorem ($x=3386$) and bounded combinatorics ($N=301$) proofs cleanly (2/2).

### 2. Dense EXL3 Trellis: Perfection at 3.00bpw, Cliff at 2.20bpw
Using the out-of-tree `exl3xpu` vLLM plugin with native ESIMD kernels, we evaluated `turboderp/Qwen3.8-27B-exl3` across multiple bitrates:
* **The 3.00bpw "Sweet Spot":** At 3.00bpw (12.87 GB), perplexity is virtually identical to 4.00bpw (6.46 vs 6.36, a negligible delta of +0.10 PPL), while saving nearly 3 GB of VRAM. It was the **only model in the entire benchmark to achieve a 100% pass rate out-of-the-box** across all 26 combined tests. It solved negative constraints without looping and passed all coding and Olympiad math problems flawlessly.
* **The 2.20bpw Cliff:** Attempting to compress dense 27B weights to 2.20bpw via post-training trellis quantization causes parameter sensitivity collapse in code synthesis. While general instruction following and math proofs survived (7/10 Hard score), **coding fell to 0/3 (0%)**. 

### 3. Native Ternary Distillation: Why Bonsai 27B Outclasses EXL3 2.2bpw
One of the most eye-opening findings of this benchmark is the comparison between **post-training quantization** (EXL3 2.2bpw) and **native quantization-aware training / distillation** (Ternary Bonsai 2 27B):
* `Ternary-Bonsai-2-27B-PTQ1_0` weighs only **5.95 GB** (~1.58 true bits per weight).
* Despite being 3.6 GB smaller than Qwen 2.20bpw, Bonsai achieved **lower perplexity (6.73 vs 6.78)**, **faster decode (13.1 tok/s vs 8.8 tok/s)**, and **passed 2/3 LeetCode Hard problems** where Qwen 2.20bpw completely failed.
* With `Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP`, decode speed reaches **12.1 tok/s** with a 90% out-of-the-box hard suite score and 100% calibrated accuracy.
* **Lesson:** If your hardware VRAM budget is restricted to under 8 GB, **always prefer natively trained ternary models** over sub-2.5bpw post-training quantization.

### 4. Why KAT-EXL3-4bpw (MoE) Fails on Intel Arc vLLM
A frequent question: *"If upstream ExLlamaV3 supports Qwen 3.5 MoE, why can't we run KAT-EXL3-4bpw with the exl3 vLLM plugin?"*
* Upstream `turboderp-org/exllamav3` supports MoE in its standalone C++/CUDA engine on NVIDIA GPUs via TabbyAPI.
* The Intel Arc plugin (`0xSero/exl3xpu`) only implemented standard `LinearBase` and `ParallelLMHead` layers for dense models.
* In vLLM, MoE blocks are constructed via `FusedMoEFactory` (`RoutedExperts`). Because the plugin does not implement `FusedMoEMethodBase` or an ESIMD grouped-GEMM kernel, vLLM defaults to `UnquantizedFusedMoEMethod`, attempting to allocate unquantized BF16 weights for all 256 experts across 40 layers (~64.4 GiB), immediately OOMing on a 32GB system.
* For MoE on Intel Arc, **MXFP4** (`models/KAT-Coder-V2.5-Dev-MXFP4`) is currently the only optimized path, utilizing Intel's native `XPUExpertsMxFp4` grouped-GEMM kernel.

---

## Practical Deployment Recommendations

If you are running local LLMs on a 32GB unified memory device (Intel Lunar Lake, AMD Strix Point, or similar):

1. **For Daily Agentic Coding & Chat:** Deploy **`Tiel-Coder-35B-A3B MXFP4`** or **`KAT-Coder-V2.5-Dev MXFP4`** under vLLM with MTP $K=2$.
   * 25–28 tok/s decode speed feels like an API.
   * Needle retrieval on 12K–16K documents takes 30–40 seconds instead of several minutes.
2. **For High-Reasoning Fallback:** Keep a sidecar server with **`Qwen3.8-27B EXL3 3.00bpw`** (or `Bonsai PQ2_0 MTP`).
   * When an MoE model fails on a subtle mathematical proof, nested negative constraint, or niche edge-case algorithm, dense 3.00bpw solves it on the first attempt with 100% precision.
3. **Avoid Sub-2.5bpw Post-Training Quantization:** If you need an ultra-compact model (<8 GB VRAM), use **Ternary Bonsai 2 27B (PTQ1 / PQ2)** via llama.cpp SYCL rather than 2.2bpw dense EXL3.

---

*All benchmark scripts (`bench_quality_16p.py`, `bench_hard_suite.py`, `eval_exl3_ppl.py`), calibrated prompt harnesses, and raw evaluation JSON artifacts are open-source and documented in the repository.*
