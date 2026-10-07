# vllm.xpu — vLLM on Intel Arc 140V (Lunar Lake 258V)

Experiments running large language models under [vLLM](https://github.com/vllm-project/vllm)
with the Intel XPU backend on a Core Ultra 7 258V (Arc 140V iGPU, 32 GB unified
memory). Two tracks: production MoE serving and EXL3 quantization bring-up.

## Hardware / software baseline

- Intel Core Ultra 7 258V, Arc 140V (Xe2-LPG, no XMX), ~103 GB/s measured bandwidth
- vLLM 0.30.0+xpu, PyTorch 2.13.0+xpu, Triton 3.7.2, oneAPI 2026.0.0
- See `environment.md` (setup) and `compatibility.md` (version audit)

## Track 1 — Production: Tiel-Coder-35B MoE (MXFP4)

Dense MoE serving with multi-token prediction. Best measured: **27.7 tok/s**
(MTP K=2) / 19.15 tok/s eager single-request.

- Plan: `tasks-tiel-optimization.md` (Phases 11–15 roadmap)
- Launch: `scripts/launch_vllm.sh` (default) / `scripts/launch_vllm_tiel_coder.sh`
- Status: `status.md`, results: `results.csv`, fallback comparison: `fallback-comparison.md`

## Track 2 — Experimental: EXL3 on Arc 140V (`0xSero/exl3xpu`)

Port of the EXL3 trellis-quantization vLLM plugin to XMX-less Xe2-LPG, tested on
Qwen3.8-27B at 4.00bpw and 2.20bpw. Local-only patches (source stays upstream).

Measured (eager, batch 1): 4.00bpw **6.5 tok/s** (PPL 6.36), 2.20bpw **8.8 tok/s**
(PPL 6.78); MTP K=1 decode +42–52% but loses end-to-end under ~60 output tokens;
fp8 KV pools **230K tokens** at 15% slower decode. Verdict: works, not
production-competitive with the MoE track on this machine.

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
