# [Benchmarking] W4A16 GPTQ vs. MXFP4 on a 35B MoE: HumanEval+, 128k Context, and 25 tok/s on an Intel Arc 140V (Lunar Lake 17W Handheld)

**Subreddit:** r/LocalLLaMA  
**Author:** Pair programming on Lunar Lake (Intel Core Ultra 7 258V / Arc 140V)  
**Model Under Test:** [`Kwaipilot/KAT-Coder-V2.5-Dev`](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev) (Qwen 3.5 MoE architecture: 40 decoder layers, 256 routed experts / 8 active, 2,048 hidden dimension, linear attention GDN + full attention hybrid, MTP speculative draft block).  
**Hardware:** MSI Claw 8 AI+ (Intel Core Ultra 7 258V "Lunar Lake", Arc 140V Xe2-LPG iGPU, 64 Xe Execution Units, 32 GB shared LPDDR5x-8533 RAM, 30.9 GiB zram0 swap, running CachyOS native Linux kernel 7.2).  
**Runtime:** `vllm==0.31.0+xpu` with native Level-Zero XPU kernels, MTP $K=2$ speculative decoding, FP8 KV cache, and 128k (`131072`) context ceiling.

---

## TL;DR

* **Speed Winner:** **W4A16 GPTQ** decodes at **23.4–25.2 tok/s** (peaking at **29.1 tok/s** with MTP speculative decoding), completing the 164-problem HumanEval benchmark in **42m 12s** (vs. 47m 33s for MXFP4, **~11.3% faster** overall, and **24% faster** on our 16-prompt quality suite).
* **Coding Accuracy Winner:** **MXFP4** edges out GPTQ on full EvalPlus coding tests by ~3% (**89.63% Base / 86.59% Plus** vs. **86.59% Base / 82.93% Plus**), but both handily beat Tiel-Coder-35B GPTQ (84.15% / 81.10%).
* **Complex Reasoning & Edge-Case Math Winner:** **W4A16 GPTQ** wins on delicate math and strict constraints (**15/16 vs 14/16** on the quality suite; solved Chinese Remainder Theorem Olympiad math where MXFP4 failed), thanks to retaining unquantized BF16 activations instead of quantizing activations to FP4.
* **Prefill Speed (PP):** Both hit an identical theoretical peak of **1,247.1 tok/s** on 4k chunked prefill batches via FlashAttention-2.
* **The "MoE Dense Quantization Trap":** Standard dense GPTQ heuristics (like zeroing out unobserved Hessian directions) will catastrophically destroy 256-expert MoE models. Unvisited experts in calibration batches must fallback to RTN with optimal clip factors, never masked to zero.

---

## The Shootout: Head-to-Head Benchmark Matrix

Here are the exact numbers measured across both quantization formats on the same machine under identical sampling parameters (`temp=0.0` greedy, FP8 KV cache, MTP $K=2$ speculative decoding, batch size 1):

| Benchmark / Metric | KAT-Coder W4A16-GPTQ (Ours) | KAT-Coder MXFP4 (Official) | Tiel-Coder W4A16-GPTQ | Tiel-Coder MXFP4 |
| :--- | :---: | :---: | :---: | :---: |
| **Quantization Format** | **INT4 Group 128 (Sym)** | **MXFP4 E2M1 Group 32** | INT4 Group 128 (Sym) | MXFP4 E2M1 Group 32 |
| **Activation Dtype** | **BFloat16 (Unquantized)** | **FP4 (Microscaled)** | BFloat16 (Unquantized) | FP4 (Microscaled) |
| **Disk Footprint** | 20.0 GiB (42 shards) | 19.60 GiB (29 shards) | 20.53 GiB (18 shards) | 20.47 GiB (31 shards) |
| **VRAM Resident (Unified)** | 21.0–24.5 GiB / 28.58 GiB | 19.7 GiB | 23.0 GiB / 28.58 GiB | 19.7 GiB / 28.58 GiB |
| **HumanEval (Base)** | **86.59% (142 / 164)** | **89.63% (147 / 164)** | 84.15% (138 / 164) | 78.66% (129 / 164) |
| **HumanEval+ (Plus Tests)** | **82.93% (136 / 164)** | **86.59% (142 / 164)** | 81.10% (133 / 164) | 75.61% (124 / 164) |
| **HumanEval Total Runtime** | **42m 12s (Fastest)** ⚡ | 47m 33s | 46m 58s | 1h 30m 37s |
| **Avg Latency per Task** | **~15.4 s** | ~17.4 s | ~17.1 s | ~33.1 s |
| **16-Prompt Quality Score** | **15 / 16 (93.8%)** 🏆 | 14 / 16 (87.5%) | **16 / 16 (100.0%)** | 15 / 16 (93.8%) |
| **16-Prompt Suite Latency** | **165.52 s (24% faster)** ⚡| 217.83 s | ~180 s | ~200 s |
| **10-Task Hard Suite Score** | **8 / 10 (80.0%)** | **8 / 10 (80.0%)** | **10 / 10 (100.0%)** | 9 / 10 (90.0%) |
| **Hard Suite Output Speed** | **80.5 chars/s** | 60.2 chars/s | ~75 chars/s | ~55 chars/s |
| **Active Decode (TG)** | **23.4 – 25.2 tok/s** | 22.6 – 24.8 tok/s | 24 – 26 tok/s | 20 – 22 tok/s |
| **Peak Decode Throughput** | **29.10 tok/s** ⚡ | 28.40 tok/s | 27.8 tok/s | 24.2 tok/s |
| **Peak Chunked Prefill (PP)**| **1,247.1 tok/s** | **1,247.1 tok/s** | 1,247.1 tok/s | 1,247.1 tok/s |
| **MTP Draft Acceptance ($K=2$)**| **77.3% avg (up to 100%)** | **82.5% avg (up to 98%)** | 65%–75% | 60%–73% |
| **Mean Acceptance Length** | **2.55 tokens / step** | **2.65 tokens / step** | 2.25 tokens / step | 2.15 tokens / step |

---

## 1. Deep Dive: Prompt Processing (PP) vs. Token Generation (TG)

A big question in the community is how 4-bit weights with 16-bit activations (W4A16) compare against full microscaling (W4A4 MXFP4) on modern unified architectures. Here is what the telemetry shows on Lunar Lake's Xe2 architecture:

### Token Generation (TG / Decode)
* **W4A16 GPTQ is faster in practical serving:** vLLM's `CompressedTensorsWNA16MoEMethod` and `XPUExpertsWNA16` kernels decode at **23.4 tok/s average** (with bursts to **29.1 tok/s** on code templates).
* **Why? Unpacking Overhead vs. Quantized GEMM:** On an iGPU with shared memory (136 GB/s memory bandwidth), reading INT4 packed weights and dequantizing on the fly into BF16 registers is extremely fast. MXFP4 requires micro-scaling exponent alignment and 32-element group normalization across both weights and activations.
* Over 164 HumanEval problems, GPTQ finished in **42 minutes 12 seconds**, beating MXFP4 by **5 minutes 21 seconds** (15.4s vs 17.4s per coding task).
* On our 16-prompt quality benchmark, GPTQ was **24% faster** (165s vs 217s).

### Prompt Processing (PP / Prefill)
* **Short/Medium Prompts (50–500 tokens):** Prefill runs between **51 tok/s and 140 tok/s** (Time-To-First-Token is 1.4s to 3.7s, largely dominated by Python/HTTP scheduling overhead).
* **Long Prompts (2k–4k tokens):** Prefill scales rapidly to **398–500 tok/s** as the 64 Xe2 compute cores enter dense matrix mode.
* **Peak Chunked Prefill (12k+ tokens):** Both models hit an identical **1,247.1 tok/s ceiling** because both rely on the same FlashAttention-2 implementation with 4,096-token chunked prefill batches (`--max-num-batched-tokens 4096`).

---

## 2. Accuracy & Nuance: Where Each Format Wins

### Where W4A16 GPTQ Wins: Delicate Math & Fine Constraints
Because GPTQ leaves **activations in full BF16**, it avoids activation quantization noise. We observed this directly in high-precision tasks:
1. **Chinese Remainder Theorem (Olympiad Math):**  
   GPTQ solved a complex modular system (`math_chinese_remainder`) flawlessly in 53.6s. MXFP4 accumulated numeric rounding errors in its FP4 activation dot-products and failed to reach the correct congruence integer.
2. **Exact Word Count Formatting:**  
   Prompt: *"Write a summary in exactly 50 words."*  
   GPTQ followed the constraint and passed; MXFP4 drifted off by several words.

### Where MXFP4 Wins: Complex Algorithmic Contracts (HumanEval+)
1. **Contract Fuzzing & Edge Cases:**  
   On HumanEval, MXFP4 scored **89.63% Base** and **86.59% Plus**, outperforming GPTQ's **86.59% Base** and **82.93% Plus**.
2. **Why?** MXFP4's group size of 32 (versus GPTQ's group size of 128) provides much finer scale granularity across expert weights. For pure syntax and standard algorithmic puzzles where activation dynamic range is well-behaved, the smaller 32-element block size helps preserve subtle weight distributions.

Both models passed all 3 LeetCode Hard challenges in our 10-Task Hard Suite:
* `Trapping Rain Water` ($O(N)$ two-pointer)
* `LRU Cache` ($O(1)$ doubly-linked list + hash map)
* `Minimum Window Substring` ($O(N)$ sliding window)
* Multi-hop needle retrieval across $12,500$ tokens of synthetic enterprise audit logs.

---

## 3. The Big MoE Gotcha: Don't Apply Dense GPTQ to 256 Routed Experts!

If you are writing your own quantizer or adapting standard recipes (like the Arc Pro B60 guide) for massive MoE models (like KAT-Coder or Qwen 3.5 MoE with 256 experts), **watch out for this fatal trap**:

### The Bug
In dense model GPTQ scripts, you often see lines like this for singular Hessian matrices:
```python
# WARNING: DO NOT DO THIS ON MOE MODELS!
W.masked_fill_(dead.unsqueeze(1), 0.0)
```
In a dense model, unobserved or zero-variance input channels can be safely zeroed out. But in a **256-expert MoE architecture**:
1. During calibration (e.g., 64 samples × 1024 tokens), tokens are sparsely routed (only 8 active per token).
2. Many specialized experts might only see a few tokens—or zero tokens—in that specific slice.
3. They are **not dead**! They are just specialized experts waiting for specific prompts.
4. If you mask unvisited experts to zero, you literally erase their weights! In our first run, this silently zeroed out **248 out of 256 experts (96.88%)** across Layers 1–39. The model loaded fine, had valid shapes and scales, but generated repetitive looping garbage (`0/16` on quality tests).

### The Fix
1. **Fallback to RTN with Clip Optimization:** If an expert receives zero activation tokens (`diag <= 1e-8`), do **NOT** zero it. Simply quantize its real BF16 weights using Round-To-Nearest (RTN) with grid search over clip factors (`1.0, 0.96, 0.92, 0.88, 0.84`).
2. **Materialize Meta Parameters:** When loading layers on PyTorch's `meta` device, ensure you explicitly rebind expert weights to `layer.mlp.experts` before computing forward activations (`nxt_hs = layer(hs)`), otherwise hidden states will propagate through uninitialized random memory and corrupt the router for all subsequent layers.

Once fixed:
* Every layer reported **`Cold Experts: 0/256`** and relative error dropped from $0.9842$ to **$0.1498$**.
* 16-prompt quality jumped instantly from **0% to 93.8% (15/16)**!

---

## 4. Speculative Decoding: The Secret Sauce for 25 tok/s on 17W

Running a 35B MoE model at over 25 tok/s on a 17W handheld SoC sounds impossible without speculative decoding.

Here is how we did it:
* KAT-Coder includes a **1-layer Multi-Token Prediction (MTP) draft head** (19 tensors in BF16).
* We preserved the MTP block in BF16 in `model-mtp.safetensors` and configured vLLM with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
* **Telemetry:** The draft model achieved an average acceptance rate of **$77.3\%$ to $82.5\%$** (and up to $100\%$ on repetitive code structures).
* **Multiplier:** Instead of generating 1 token per forward step, the engine accepted **$2.55\text{--}2.71$ tokens per step**. This effectively doubled generation throughput from $\sim 11\text{ tok/s}$ up to **$23\text{--}29\text{ tok/s}$**.

---

## Conclusion & Recommendations

1. **If you want maximum decoding speed and lower latency:** Use **W4A16 GPTQ** (`CompressedTensorsWNA16MoEMethod`). It is 11–24% faster, lighter on the memory bus, and finishes large coding suites minutes ahead of MXFP4.
2. **If you want the highest possible unit-test pass rate on complex codebases:** Use **MXFP4** (`CompressedTensorsW4A4Mxfp4`). The 32-element micro-scaling groups give it a slight edge on complex fuzzing assertions (+3.6% on HumanEval+).
3. **If you are building an Intel Lunar Lake / Arc 140V local AI workstation:**
   - 35B MoE models with 8 active experts are the absolute sweet spot for 32 GB shared-memory handhelds.
   - Always enable MTP speculative decoding ($K=2$) and FP8 KV cache. It gives you 128k context support, zero swap thrashing, and desktop-class coding performance on battery power.

---
*All quantization scripts, preflight validators, and reproducible benchmark harnesses are open in our repo: [`alpha2beta/vllm.xpu`](https://github.com/alpha2beta/vllm.xpu).*
