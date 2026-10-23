# vllm.xpu — vLLM on Intel Arc 140V (Lunar Lake 258V)

Experiments running large language models under [vLLM](https://github.com/vllm-project/vllm)
with the Intel XPU backend on a Core Ultra 7 258V (Arc 140V iGPU, 32 GB unified
memory). Two tracks: production MoE serving and EXL3 quantization bring-up —
plus a family of MXFP4 checkpoints quantized in-repo and published on the Hub.

## Hardware / software baseline

- Intel Core Ultra 7 258V, Arc 140V (Xe2-LPG, no XMX), ~103 GB/s measured bandwidth
- vLLM 0.30.0+xpu, PyTorch 2.13.0+xpu, Triton 3.7.2, oneAPI 2026.0.0
  (0.31.0+xpu for the Oct-15 Huihui smoke test)
- See `environment.md` (setup) and `compatibility.md` (version audit)

## Track 1 — Production: MoE coders (MXFP4 & W4A16-GPTQ)

Dense MoE serving with multi-token prediction (1024/128/1, MAX_LEN 4096 / up to 128k):

| Model | Quantization | MTP K=2 Speed | Quality (16p) | Hard Suite (10t) | HumanEval (Base) | HumanEval+ (Plus) |
|---|---|---|---|---|---|---|
| **KAT-Coder-V2.5-Dev** | W4A16-GPTQ | **25.20 tok/s** | **15/16 (93.8%)** | 8/10 (80.0%) | 86.59% (142/164) | 82.93% (136/164) |
| **KAT-Coder-V2.5-Dev** | MXFP4 (W4A4 MoE) | 24.82 tok/s | 14/16 (87.5%) | 8/10 (80.0%) | **89.63%** (147/164) | **86.59%** (142/164) |
| **Tiel-Coder-35B-A3B** | W4A16-GPTQ | 25.40 tok/s | **16/16 (100%)** | **10/10 (100%)** | 84.15% (138/164) | 81.10% (133/164) |
| **Tiel-Coder-35B-A3B** | MXFP4 (W4A4 MoE) | **27.69 tok/s** | 15/16 (93.8%) | 9/10 (90.0%) | 78.66% (129/164) | 75.61% (124/164) |

### HumanEval / HumanEval+ (EvalPlus 0.3.1) Benchmark Details

Full 164-task standardized evaluation under greedy decoding (`temp=0.0`, single sequence, FP8 KV cache):

| Benchmark Metric | KAT-Coder GPTQ | KAT-Coder MXFP4 | Tiel-Coder GPTQ | Tiel-Coder MXFP4 | Qwen3.8-27B EXL3 (3.0bpw) |
|---|---|---|---|---|---|
| **HumanEval (Base)** | 86.59% (142 / 164) | **89.63%** (147 / 164) | 84.15% (138 / 164) | 78.66% (129 / 164) | 69.51% (114 / 164) |
| **HumanEval+ (Base + Extra)** | 82.93% (136 / 164) | **86.59%** (142 / 164) | 81.10% (133 / 164) | 75.61% (124 / 164) | 67.68% (111 / 164) |
| **Total Codegen Duration** | **42m 12s (Fastest)** ⚡ | 47m 33s | 46m 58s | 1h 30m 37s | ~1h 15m (with CoT) |
| **Avg Latency per Task** | **~15.4s** | ~17.4s | ~17.1s | ~33.1s | ~27.4s |
| **Avg Output Chars / Task** | 1,235 chars | 1,244 chars | 1,231 chars | 2,135 chars (+60% verbosity) | 1,842 chars (incl. `<think>`) |
| **Active Throughput** | ~24–25 tok/s | ~24–25 tok/s | ~24–26 tok/s | ~20–22 tok/s | ~7.5–9.4 tok/s |
| **MTP Acceptance Rate** | **~77%–85%** | ~75%–83% | ~65%–75% | ~60%–73% | ~70%–88% (K=1) |
| **Max Context Served** | 128k (`131072`) | 128k (`131072`) | 96k (`98304`) | 96k (`98304`) | 16k (`16384`) |

> [!NOTE]
> **Qwen3.8-27B Truncation Analysis:** EvalPlus hardcodes `max_new_tokens=768` for OpenAI endpoints. Because Qwen3.8 emits 400–600 tokens of `<think>...</think>` reasoning before code generation, 41.5% of tasks (68/164) hit the 768-token ceiling, directly accounting for 29 of the 50 failures mid-function. Adjusting for token truncation yields **87.2% Base Pass@1** (143 / 164), matching KAT-Coder.

- **Launchers:**
  - `launch_vllm_kat_gptq.sh` (KAT W4A16-GPTQ, 128k default, FP8 KV, MTP K=2)
  - `launch_vllm_kat_coder.sh` (KAT MXFP4, 128k default, FP8 KV)
  - `launch_vllm_tiel_gptq.sh` (Tiel W4A16-GPTQ, 96k default, FP8 KV)
  - `launch_vllm_tiel_coder.sh` (Tiel MXFP4, 96k default, FP8 KV)
  - `scripts/run_evalplus_humaneval.sh` (Automated HumanEval/HumanEval+ benchmark runner)
- Status & Analysis: [**`GPTQ_vs_MXFP4.md`**](GPTQ_vs_MXFP4.md) (Reddit post & comprehensive shootout), `status.md`, `results.csv`

## Published Checkpoints (Hugging Face `alpha2beta/`)

Production `compressed-tensors` pack-quantized builds for the vLLM XPU backend
(`CompressedTensorsWNA16MoEMethod` and `CompressedTensorsW4A4Mxfp4`). All load-verified on this box.

| Checkpoint | Format & Arch | Size | Measured Performance |
|---|---|---|---|
| [KAT-Coder-V2.5-Dev-W4A16-GPTQ](https://huggingface.co/alpha2beta/KAT-Coder-V2.5-Dev-W4A16-GPTQ) | W4A16 INT4 MoE + BF16 MTP draft | 20.0 GiB / 42 shards | 25.20 tok/s (peak 29.1), HumanEval+ 82.93% |
| [KAT-Coder-V2.5-Dev-MXFP4](https://huggingface.co/alpha2beta/KAT-Coder-V2.5-Dev-MXFP4) | 35B-A3B MoE + MXFP4 MTP draft | 19.6 GiB / 29 shards | 24.82 tok/s MTP K=2, HumanEval+ 86.59% |
| [Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4](https://huggingface.co/alpha2beta/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4) | 35B-A3B MoE + MXFP4 MTP draft | 20.5 GiB / 32 shards | 27.69 tok/s MTP K=2 |
| [Qwen3.6-35B-A3B-occamy-1.0-MXFP4](https://huggingface.co/alpha2beta/Qwen3.6-35B-A3B-occamy-1.0-MXFP4) | 35B-A3B MoE + MXFP4 MTP draft | ~20.4 GiB / 21 shards | quality 13/16, hard 8/10 |
| [Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4](https://huggingface.co/alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4) | dense 27B + BF16 MTP | 18.44 GiB / 16 shards | 2.6 tok/s (dense wall) |
| [Qwen3.8-9B-Distill-EXL3-K2](https://huggingface.co/alpha2beta/Qwen3.8-9B-Distill-EXL3-K2) | EXL3 K=2 (`mul1`) + BF16 pass-through | 6.72 GiB / 35 shards | PPL 7.55 (+18.6% vs FP16), 52s load on 140V |

- Quantizers: `scripts/quantize_{tiel,kat,occamy}_mxfp4.py` (+ `*_mtp_mxfp4.py`,
  `finalize_*`, `audit_*`) for MoE; `scripts/quantize_huihui_qwen38_mxfp4.py`
  (dense recipe, reference `scripts/ref/`) for Qwen3.8. Local checkpoints live
  in `models/` (git-ignored); the Huihui build lives on external disk (see its
  model card for path).
- No other vLLM-ready MXFP4 builds of these models exist on the Hub (MLX/GGUF/
  NVFP4/GPTQ variants are not XPU-loadable) — checked per model before publishing.

## Track 2 — Experimental: EXL3 on Arc 140V (`0xSero/exl3xpu`)

Port of the EXL3 trellis-quantization vLLM plugin to XMX-less Xe2-LPG, tested on
Qwen3.8-27B at 4.00bpw and 2.20bpw. Local-only patches (source stays upstream).

Measured (eager, batch 1): 4.00bpw **6.5 tok/s** (PPL 6.36), 2.20bpw **8.8 tok/s**
(PPL 6.78); 3.00bpw hits the sweet spot — quality suite **16/16**, hard suite
**10/10** (incl. antileak variants), PPL **6.46** (+0.10 vs 4.00bpw), HumanEval
Base **69.51%** (raw 114/164; **87.2%** / 143/164 non-truncated, as 58% of failures were caused by EvalPlus 768-token CoT cutoff) and Plus **67.68%** (111/164);
MTP K=1 decode +42–52% but loses end-to-end under ~60 output tokens;
fp8 KV pools **230K tokens** at 15% slower decode. Verdict: works, not
production-competitive with the MoE track on this machine.

> **Scope note:** on dense Qwen3.8-27B, EXL3 (6.5–8.8 tok/s) outperforms MXFP4
> (~2.6–2.9 tok/s) — but the EXL3 plugin only implements the dense-27B path.
> The MoE coders above (24–28 tok/s) serve exclusively via MXFP4, which remains
> the only quantized path covering all four published checkpoints.

- Plan + evidence: `tasks.md` (Phases E0–E9), feasibility notes: `vllm_xpu_exl3_140v.md`
- Launch: `scripts/launch_vllm_exl3.sh` (port 8001, isolated from production)
- Key scripts: `scripts/audit_exl3_checkpoint.py`, `scripts/build_exl3_ext.sh`,
  `scripts/test_exl3_ops.py`, `scripts/eval_exl3_ppl.py`
- Patches live as diffs under `logs/exl3-*/` (`no-dpas.patch`, `k1-fallback.patch`,
  `mixed-k.patch`); plugin source itself is **not** vendored (see note below)

> **Reproducing the EXL3 track** requires the external plugin checkout beside this
> repo (`git clone https://github.com/0xSero/exl3xpu.git ~/Projects/exl3xpu`,
> pin `15ded2f3`, apply the `no-dpas`/`k1-fallback`/`mixed-k` patches from the
> corresponding `logs/` dir) and the model checkpoints in `models/` (git-ignored;
> see `scripts/download_model.sh` pattern — EXL3 weights came from
> `turboderp/Qwen3.8-27B-exl3` branches `SC_4.00bpw_H5*` / `SC_2.20bpw_H3_V3`).

## Repo map

| Path | Contents |
|---|---|
| `scripts/` | Launchers, benchmarks (`bench_envelope.py`), EXL3 audit/build/test/PPL tooling |
| `models/` | Checkpoints (git-ignored, local only) |
| `logs/` | Per-run evidence: server logs, mem traces, patches, result JSON |
| `results.csv` | All benchmark rows (schema documented in `tasks.md`/`tasks-tiel-optimization.md`) |
| `evalplus_results/` | HumanEval & HumanEval+ (EvalPlus 0.3.1) raw completions and eval JSONs |
| `tools/` | Level-Zero sysroot for Triton-XPU JIT |

`status.md` is the running lab notebook; `plan.md` / `ctx128k.md` hold older planning context.
