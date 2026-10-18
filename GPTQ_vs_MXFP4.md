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
* **Cross-platform check — the ordering holds, the magnitude does not:** re-running the same format pair on a **discrete Arc Pro B60** with a **dense 27B** (Coder390) separates them by **5.05× on decode** (40.18 vs 7.96 tok/s) and **~2.9× on prefill**, not ~10 %. On *that* stack the cause is an **unfused W4A4 activation-quantization path**, not memory bandwidth — the two artifacts differ by only ~3 % in bits per weight. Full matrix, decomposition and caveats in **§5**.

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

## 5. Cross-Platform Check: The Same Two Formats on an Intel Arc Pro B60 (Battlemage, 24 GB)

Everything above is Lunar Lake / KAT-Coder. The same W4A16-vs-MXFP4 question was then measured independently on a **discrete** Arc Pro B60, with a different model *class* (dense, not MoE), a different Xe generation, and a different vLLM build. **The ordering survives. The magnitude does not** — it is **5×** here, not ~10 %.

**Model under test:** [`huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated) — a **dense 27B** hybrid: 64 layers (**48 GDN linear-attention + 16 full-attention**, `full_attention_interval 4`), hidden 5120, 24 heads / 4 KV heads / `head_dim` 256, a 1-layer MTP draft block (BF16), and a vision tower. Both builds below are quantizations **of this same BF16 checkpoint**, so the *format* is the only variable in the weights.

**Hardware:** Intel Arc Pro B60 — 24 GB GDDR6 at **456 GB/s** (192-bit, 19 Gbps effective), 20 Xe2-HPG cores / 160 XMX engines, negotiated PCIe Gen4 x8, **power-capped to 180 W**.
**Runtime:** `vllm-openai-xpu` v0.30.0 (engine reports `vllm 0.27.2rc1.dev77+gac7509e2b`), native Level-Zero XPU kernels, FP8 KV cache, MTP $K=4$, batch size 1.
**Harness:** identical for both builds — `bench_p512_g128.py` (p512 / g128, 1 warmup + 5 runs) for rate, `bench_quality_16p.py` (the same 16-prompt suite) for quality.

### 5.1 Head-to-Head Benchmark Matrix

| Benchmark / Metric | Coder390 **W4A16 GPTQ** | Coder390 **MXFP4** |
| :--- | :---: | :---: |
| **Quantization Format** | INT4 Group 128 (Sym), `desc_act=False` | MXFP4 E2M1 Group 32 |
| **Activation Dtype** | **BFloat16 (Unquantized)** | **FP4 (Microscaled, quantized per token)** |
| **Bits per weight** | **4.125** (int4 + fp16 scale / 128) | **4.25** (E2M1 + E8M0 scale / 32) |
| **Disk Footprint** | 19 GB (**5** shards) | 19 GB (**16** shards) |
| **Hub-resident size** | 18.23 GiB | 18.46 GiB |
| **Decode (TG), p512/g128** | **40.183 tok/s** 🏆 | 7.962 tok/s |
| **TTFT (512-token prompt)** | **0.460 s** 🏆 | 1.313 s |
| **Prefill (PP) plateau** | **1,372 tok/s** 🏆 | ~410 tok/s (implied by TTFT) |
| **Mean MTP acceptance length ($K=4$)** | **3.00** tok/step 🏆 | 2.83 tok/step |
| **MTP on, $K=4$ (same session)** | 40.183 tok/s — MTP-off leg **not measured** | **16.620 tok/s** (**2.081×** over its own MTP-off 7.987) |
| **Context ceiling** | 17,630-token KV pool (12,288 usable) | 81,920 (MTP off) / **40,960 (MTP on)** |
| **16-Prompt Quality Score** | **15 / 16 (93.8 %)** | **not measured** |
| **Decode vs context depth (512 → 12k)** | 40.200 → 37.315 tok/s (**−7.2 %**) | not swept |

**Headline: 5.05× on decode, ~2.9× on prefill.** Unlike the KAT-Coder matrix above, the MXFP4 build is not visibly buying anything for that price on this stack: it uses **more** bits per weight, has **no measured accuracy advantage here at all** (its quality was never measured), and its rate is ~5× behind.

### 5.2 Where the 5.05× Actually Comes From

Decomposed on *measured step times*, not asserted. `40.183 / 7.962 = 5.05×`:

| | MXFP4 | GPTQ |
| :--- | ---: | ---: |
| Decode | 7.962 tok/s | 40.183 tok/s |
| Tokens per forward step | **1.00** | **3.00** |
| ms per step | 125.6 | 74.7 |
| **ms per emitted token** | **125.6** | **24.9** |

MTP alone is therefore **at most** a `3.00×` factor — and the acceptance ratio is *not* a speedup, which was verified by turning MTP on for **both** builds so the term cancels:

| Leg (same session, same harness) | Decode median | TTFT median |
| :--- | ---: | ---: |
| MXFP4 — MTP off (control) | 7.987 tok/s | 1.310 s |
| MXFP4 — MTP on, $K=4$ | **16.620 tok/s** | 1.412 s |
| | **2.081×** | +7.8 % |

With MTP enabled on both sides the residual is `40.183 / 16.620 = **2.42×**`, and *that* residual is the stack. Acceptance is not quite equal either (2.83 vs 3.00); normalising to 3.00 gives a **2.28×** residual. So:

> **5.05× = 2.08× (MTP) × 2.42× (the W4A4 / stack term)** — and note the direction: **MTP is the *smaller* of the two terms.**

**Why the acceptance ratio overstates MTP.** A speculative step is itself more expensive — draft forward plus a 5-position verify costs **1.361×** a plain step here (125.2 ms → 170.5 ms), against 2.833 tokens accepted. The *net* is `2.833 / 1.361 = **2.081×**`, which is exactly the measured figure. Treat the acceptance ratio as an **upper bound** on MTP's gain, never the gain itself.

**Prefill isolates the stack cleanly**, because MTP contributes nothing there and it is compute-bound: 0.460 s vs 1.313 s for the same 512-token prompt ⇒ **~2.9× slower per prompt token** (≈3.1× once the ~60 ms fixed TTFT overhead is netted out). Decode (~2.3–2.4×) and prefill (~2.9×) therefore **agree in magnitude** — which is the main reason to trust the split.

### 5.3 The Mechanism: W4A4 Activation Quantization That Is Not Fused

This is a *different* complaint from the MoE trap in §3, and it is readable straight from the kernel rather than inferred:

```python
class XPUMxFp4LinearKernel(MxFp4LinearKernel):
    def apply_weights(self, layer, x, bias=None):
        x_fp4, x_blockscale = quant_mxfp4(x)          # <- every single call
        return torch.ops._xpu_C.fp4_gemm(x_fp4, layer.weight,
                                         x_blockscale, layer.weight_scale, ...)
```

and, inside `_xpu_mxfp4_quantize_impl`:

```python
x_q = torch.empty(M, N // 2,  ...)   # packed FP4,   allocated per call
x_s = torch.empty(M, N // 32, ...)   # block scales, allocated per call
torch.ops._C.per_token_group_quant_mxfp4(x, x_q, x_s, MXFP4_BLOCK_SIZE, eps)
x_s = x_s.to(dtype=torch.float8_e8m0fnu, ...)   # extra cast pass
```

So for **each of the 496 packed linear layers, on every forward pass**, MXFP4 pays: a group-32 activation-quantization kernel, an `fp32 → E8M0` scale cast, and two fresh allocations — *and then* a separate GEMM. Quantize and GEMM are **not fused**, so the floor is **~992 launches per step** before norms, attention or GDN. **W4A16 GPTQ pays none of it**: BF16 activations go straight into the fused dequant-GEMM.

Ranked:
1. **Per-token W4A4 activation quantization** (~3× per FLOP above) — the dominant stack term.
2. **Non-fused FP4 GEMM** (`torch.ops._xpu_C.fp4_gemm`) — a second launch per linear.
3. **`--enforce-eager`** — *amplifies* 1–2, since none of those launches can replay from a graph. It is inherited from the model README's measured config, not demonstrably required (`platforms/xpu.py` lists `mxfp4` under `supported_quantization` with no graph restriction). **Untested, not forced.**

**It is emphatically *not* a bandwidth story — and the arithmetic says so before any profiling:** **4.25 bits/weight for MXFP4 vs 4.125 for GPTQ is a ~3 % difference**, and the two artifacts are 18.46 vs 18.23 GiB. A bandwidth-bound model predicts near-parity, not 5×. Two further confirmations that it is compute: the five MXFP4 runs sat within **0.02 %** of each other — a fixed per-token cost, not noisy traffic — and prefill, the pure-compute regime, is ~2.9× off **on its own**.

### 5.4 Why This Is Not the §1 Answer, and What Would Reconcile Them

On Lunar Lake the two formats landed within ~10 % of each other on decode, and MXFP4 even won HumanEval+. Here MXFP4 is 5× behind. Three differences could carry that, and **none of them was isolated on this box**:

1. **vLLM build.** The Lunar Lake result is `vllm==0.31.0+xpu`; this box runs v0.30.0 (`vllm 0.27.2rc1.dev77`). Whether 0.31 fused the MXFP4 activation path was **not tested** — it is the most likely single explanation and the first thing worth re-measuring.
2. **Model class.** KAT-Coder is a 256-expert MoE (8 active per token) routed through `CompressedTensorsWNA16MoEMethod` / `XPUExpertsWNA16`; Coder390 is **dense** and hits the plain `XPUMxFp4LinearKernel` for all 496 linears. The kernel mix differs, so "MXFP4 vs GPTQ" is not one experiment across the two documents.
3. **Platform ratio.** 20 Xe2 cores over 456 GB/s of dedicated GDDR6 (discrete) vs 8 Xe2 cores over shared LPDDR5x (integrated). The W4A4 penalty is compute-side, so it should bite *harder* where FLOPs are scarcer relative to bytes — the direction of the observed gap.

**Caveats, stated because this is a cross-session comparison and the warning above about those is real.** Rates here are only comparable **within one session** — this host's decode is bimodal (**~40.2 or ~48.5 tok/s, a ~20 % band**, mechanism unidentified), and every rate in §5.1 was taken in the *slow* state, so all rows shift together and the **ratios** are the transferable result. The MXFP4 number is the one build *provably immune* to that band: five runs within 0.02 % and TTFT identical to three decimals on a same-session re-run (7.987 vs the 7.962 recorded the previous day, **+0.3 %**). No same-session A/B of the two formats was run here — each costs a server swap and a ~200 s cold start — so the 5.05× rests on same-harness, cross-session legs, with the MXFP4 leg's stability measured. And the **declared limitation: no accuracy comparison was made at all** — unlike the KAT-Coder matrix above, Coder390 MXFP4 has no 16-prompt and no PPL score, so the speed gap is **not yet known to be bought or paid for in quality**.

**Read-across to the recommendations below.** "W4A16 for speed, MXFP4 only where the finer 32-element groups buy accuracy" is consistent with what the B60 saw — but on this stack the speed penalty is ~5× rather than ~10 %. If you are targeting a **dense** model on **Intel discrete** silicon, check first whether your vLLM build fuses the MXFP4 activation quantization. If it does not, W4A16 is not a marginal win; it is a different tier.

---

## Conclusion & Recommendations

1. **If you want maximum decoding speed and lower latency:** Use **W4A16 GPTQ** (`CompressedTensorsWNA16MoEMethod`). It is 11–24% faster, lighter on the memory bus, and finishes large coding suites minutes ahead of MXFP4.
2. **If you want the highest possible unit-test pass rate on complex codebases:** Use **MXFP4** (`CompressedTensorsW4A4Mxfp4`). The 32-element micro-scaling groups give it a slight edge on complex fuzzing assertions (+3.6% on HumanEval+).
3. **If you are building an Intel Lunar Lake / Arc 140V local AI workstation:**
   - 35B MoE models with 8 active experts are the absolute sweet spot for 32 GB shared-memory handhelds.
   - Always enable MTP speculative decoding ($K=2$) and FP8 KV cache. It gives you 128k context support, zero swap thrashing, and desktop-class coding performance on battery power.
4. **If you are on *discrete* Intel silicon (Arc Pro B-series),** the same ordering holds but the stakes are much higher: W4A16 GPTQ decoded **5.05×** faster than MXFP4 on a dense 27B (§5), with ~2.9× on prefill. Before assuming MXFP4 is "within ~10 %" as it is on Lunar Lake, check whether your vLLM build **fuses** the MXFP4 per-token activation quantization — on the build tested here it does not, and that unfused path alone accounts for the ~2.4× residual left after MTP is equalised.

---
*All quantization scripts, preflight validators, and reproducible benchmark harnesses are open in our repo: [`alpha2beta/vllm.xpu`](https://github.com/alpha2beta/vllm.xpu).*

*§5 provenance: Coder390 numbers come from that repo's `REPRODUCIBILITY.md` §6–§8 (`bench_p512_g128.py`, `bench_prefill.py`, `bench_quality_16p.py`, `compare_mxfp4_mtp.sh`, at 180 W, MTP $K=4$); the GPTQ artifact is published as [`alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-GPTQ-Int4-sym-G128-MTP-BF16`](https://huggingface.co/alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-GPTQ-Int4-sym-G128-MTP-BF16), and the MXFP4 checkpoint as [`…-abliterated-MXFP4`](https://huggingface.co/alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4).*
