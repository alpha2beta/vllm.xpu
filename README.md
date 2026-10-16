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

## Track 1 — Production: MoE coders (MXFP4)

Dense MoE serving with multi-token prediction (1024/128/1, MAX_LEN 4096):

| Model | Eager | MTP K=2 | Quality / Hard |
|---|---|---|---|
| Tiel-Coder-35B MXFP4 | 19.15 tok/s | **27.69 tok/s** | 15/16 / 9/10 |
| KAT-Coder-V2.5-Dev MXFP4 | ~13–14 tok/s (E-cores) | **24.82 tok/s** | 15/16 / 8/10 |

- Plan: `tasks-tiel-optimization.md` (Phases 11–15 roadmap)
- Launch: `scripts/launch_vllm.sh` (default) / `scripts/launch_vllm_tiel_coder.sh`
  / `scripts/launch_vllm_kat_coder.sh` (KAT, 128k default, `KV_DTYPE` env for
  opt-in fp8 KV)
- Status: `status.md`, results: `results.csv`, fallback comparison: `fallback-comparison.md`

## Published MXFP4 checkpoints (Hugging Face `alpha2beta/`)

`compressed-tensors` `mxfp4-pack-quantized` builds for the vLLM XPU backend
(`CompressedTensorsW4A4Mxfp4` + `XPUExpertsMxFp4`), quantized with the inference
kernel's own quantizer. All load-verified on this box.

| Checkpoint | Arch | Size | Measured |
|---|---|---|---|
| [Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4](https://huggingface.co/alpha2beta/Tiel-Coder-35B-A3B-Genesis-Hermes-MXFP4) | 35B-A3B MoE + MXFP4 MTP draft | 20.5 GiB / 32 shards | 27.69 tok/s MTP K=2 |
| [KAT-Coder-V2.5-Dev-MXFP4](https://huggingface.co/alpha2beta/KAT-Coder-V2.5-Dev-MXFP4) | 35B-A3B MoE + MXFP4 MTP draft | 19.6 GiB / 29 shards | 24.82 tok/s MTP K=2 |
| [Qwen3.6-35B-A3B-occamy-1.0-MXFP4](https://huggingface.co/alpha2beta/Qwen3.6-35B-A3B-occamy-1.0-MXFP4) | 35B-A3B MoE + MXFP4 MTP draft | ~20.4 GiB / 21 shards | quality 13/16, hard 8/10 |
| [Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4](https://huggingface.co/alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-MXFP4) | dense 27B + BF16 MTP | 18.44 GiB / 16 shards | 2.6 tok/s (dense wall) |

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
**10/10** (incl. antileak variants), PPL **6.46** (+0.10 vs 4.00bpw);
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
| `tools/` | Level-Zero sysroot for Triton-XPU JIT |

`status.md` is the running lab notebook; `plan.md` / `ctx128k.md` hold older planning context.
