# MTP Status — Qwen3.5-35B-A3B EXL3 MoE on vLLM XPU (Arc 140V)

**Date:** October 25, 2026
**Scope:** Read-only status assessment. No code, launcher, checkpoint, or configuration changes were made; this document is the only output.
**Checkpoint under test:** `turboderp/Qwen3.5-35B-A3B-exl3` @ 3.09 bpw, rev `f599771c…`, local path `/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw`
**Serving stack:** vLLM XPU v0.31.0, `enforce-eager`, TP=1, `max_num_seqs=1`, 2048 context, port 8002

---

## 1. Bottom line

**MTP is not supported for this MoE deployment today, and it is not merely switched off — the shipped checkpoint does not contain an MTP head.**

`text_config.mtp_num_hidden_layers = 1` advertises an MTP layer, but the weight files contain **0 MTP/NextN tensors**. Everything else needed to run MTP (upstream model class, plugin key remapping, draft-head vocabulary pruning, dense-launcher plumbing) already exists and is exercised on the dense EXL3 path. The MoE path is missing exactly two things:

1. a checkpoint revision that actually ships the `mtp.*` weights, and
2. `SPEC` plumbing in `scripts/launch_vllm_moe_exl3.sh`.

MTP is explicitly out of scope for the current plan (`tasks_moe.md:28`) and is deferred until a validated checkpoint exists. No speculative decoding was enabled or benchmarked on the MoE model for this report.

---

## 2. Checkpoint evidence

Inspected `config.json` and `model.safetensors.index.json` directly:

| Property | Value |
|---|---|
| `architectures` | `Qwen3_5MoeForConditionalGeneration` |
| `text_config.num_hidden_layers` | 40 |
| `text_config.mtp_num_hidden_layers` | **1** (declared) |
| `speculative_config` in `config.json` | absent |
| `num_nextn_predict_layers` | absent |
| Total tensors in index | 124,579 |
| Tensors matching `mtp` / `nextn` | **0** |
| Shards | `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors` |

Runtime confirmation from the r7 Gate 4 server (`logs/moe-p3/server-p4-grouped-r7-p402.log`):

```
exl3: 30971 EXL3 modules (0 in MTP head)
... speculative_config=None ...
```

So the loader is not silently skipping MTP weights — there are none to load, and vLLM was started with no speculative config at all. Note that the declared layer count alone is misleading: the config field is inherited from the architecture, not evidence that the conversion wrote an MTP head.

---

## 3. Infrastructure that already exists

### Upstream vLLM (installed, v0.31.0)

`vllm/model_executor/models/qwen3_5_mtp.py` defines:

- `Qwen3_5MTP` (line 217) — shared dense MTP implementation,
- `Qwen3_5MoeMTP` (line 326) — MoE drafter wired through `QwenNextMixtureOfExperts`,
- MTP layer count from `mtp_num_hidden_layers` (line 83),
- standard `"mtp"` speculative method, so `--speculative-config '{"method":"mtp","num_speculative_tokens":k}'` is a supported CLI surface.

### EXL3 plugin (`exl3xpu`)

| Capability | Location | Status |
|---|---|---|
| `mtp.*` key canonicalization | `exl3xpu/vllm_plugin.py:36-46` (`_norm_key`) | present, exercised |
| `mtp_bits` config field | `exl3xpu/vllm_plugin.py:54-57` | present, defaults to base bits |
| Per-key default bitrate for `mtp.` prefix | `exl3xpu/vllm_plugin.py:117` | present |
| MTP-head module count in load log | `exl3xpu/vllm_plugin.py:121-122` | present |
| Pruned draft vocabulary | `EXL3_DRAFT_VOCAB` + `_spec_is_mtp()` + `_patch_mtp_draft_logits()` (`vllm_plugin.py:344, 416-480`) | present, gated on `"mtp"` in spec method |
| Draft-vocab generator | `exl3xpu/scripts/draft_vocab.py` | present |
| MTP identity / divergence tests | `exl3xpu/tests/test_mtp_identity.py`, `test_mtp_divergence.py` | present |

### Launchers

| Launcher | MTP support |
|---|---|
| `scripts/launch_vllm_exl3.sh` (dense) | `SPEC=${SPEC-mtp-k1}` default; builds `--speculative-config` for `mtp-k1`, `mtp-k2`, `ngram-k<N>` |
| `scripts/launch_vllm_moe_exl3.sh` (MoE) | **none** — no `SPEC` variable, no speculative flag, no `--speculative-config` |

This asymmetry is the entire plumbing gap on the serving side.

---

## 4. Dense MTP reference run (works, on a different model)

MTP has been run end-to-end on the dense EXL3 model, which proves the plugin + upstream stack is functional:

- Server log: `logs/server-exl3-4bpw-mtp-k1.log`
- Model: `models/turboderp-Qwen3.8-27B-exl3-4.00bpw`
- Loader: `409 EXL3 modules (8 in MTP head)`
- Engine config: `SpeculativeConfig(method='mtp', model='…Qwen3.8-27B-exl3-4.00bpw', num_spec_tokens=1)`
- Benchmark bundle: `logs/exl3-4bpw-mtp-20261007-102715/` (K=1 and K=2 at 1024 and 2048 input tokens, plus memory monitor and GPTQ-style probe JSONs)

Measured dense results (median, single request, 1024-token prompt, 128 max output, 31 tokens produced):

| Config | TTFT | Decode | End-to-end |
|---|---|---|---|
| K=1 | 4.79 s | 9.86 tok/s | 7.94 s |
| K=2 | 5.58 s | 8.62 tok/s | 9.35 s |

These numbers are **not** a speedup claim and are **not transferable** to the MoE model: they are dense Qwen3.8-27B at 4.00 bpw, and the run shows the classic MTP cost profile — the extra draft/verify pass is visible in TTFT and does not obviously pay for itself at these acceptance rates. The useful takeaway is mechanical: the `mtp` method loads a real MTP head, maps its tensors, and serves correctly on this XPU stack.

---

## 5. Why MTP is off for the MoE model

Four independent blockers, in order of hardness:

1. **No weights.** The pinned 3.09 bpw revision ships no `mtp.*` tensors. No launcher flag can fix this; a different checkpoint revision (or a re-conversion with MTP preserved) is required.
2. **No launcher plumbing.** Even with weights, `scripts/launch_vllm_moe_exl3.sh` cannot pass a speculative config.
3. **No MoE-specific validation.** Nothing has verified tensor mapping, bitrate assignment (`mtp_bits` vs base 3.09), draft-head vocabulary pruning, or acceptance rate for `Qwen3_5MoeMTP` + EXL3.
4. **Plan scope.** `tasks_moe.md:28` defers speculative decoding / MTP, to be evaluated separately after the core Gates. Gates 0–4 were certified without it; MTP is not a Gate 4 criterion.

Additional technical consideration: the P4-02 grouped decode kernel fast path is gated on **M=1**. Under MTP, the verify batch is `k+1` tokens, so those steps fall back to the P4-01 eager loop (fail-safe by design), and the draft forward adds another pass over a dense-equivalent decoder layer. On 8 Xe2 cores with decode already at ~25–28 tok/s against a 130.4 GB/s roofline (14.7–22.4 %), MTP must recover more time through accepted tokens than the draft+verify passes cost. That is an empirical question, not a safe assumption — the dense K=1/K=2 results above lean negative.

---

## 6. What enabling MTP would require

| Step | Work | Risk / notes |
|---|---|---|
| 1 | Obtain an MTP-bearing checkpoint revision (verify `mtp.*` keys in the index; 1 layer expected, dense chain) | Blocking; may require requesting a re-export from the checkpoint author |
| 2 | Confirm `quantization_config.json` handling for the MTP head | Plugin already defaults omitted MTP tensors to `mtp_bits`, falling back to base bits; decide whether the draft head should use a higher bitrate |
| 3 | Add `SPEC` support to `scripts/launch_vllm_moe_exl3.sh` mirroring the dense launcher (`mtp-k1`, `mtp-k2`, empty = off) | Small, mechanical |
| 4 | Memory budget check | Gate 4 peak was 18.36 GiB active at `GPU_UTIL=0.62`; an extra decoder layer of draft weights plus verify activations must fit the remaining headroom at 2048 context |
| 5 | Numerical parity | Speculative decoding with rejection sampling is lossless by construction, but confirm logits/PPL against the non-MTP baseline on the same prompts |
| 6 | Performance matrix | Decode tok/s, TTFT, acceptance rate, and end-to-end at 128/512/1024 input; compare against the certified Gate 4 baseline (25.27 / 27.90 / 26.19 tok/s) |
| 7 | Stress / regression | Reuse `bench/soak_moe.py` and the Phase 2 / dispatch / loader / reconstruction suites; verify the grouped path still latches off cleanly on verify batches |

**Decision:** not started. Per the standing instruction, MTP work waits for an explicit request; it also needs a checkpoint that does not currently exist locally.

---

## 7. Status summary

| Component | State |
|---|---|
| Upstream `Qwen3_5MoeMTP` model class | available |
| Upstream `"mtp"` speculative method | available |
| Plugin `mtp.*` tensor remapping | available, exercised on dense |
| Plugin `mtp_bits` / draft-vocab pruning | available |
| MoE checkpoint MTP weights | **missing (0 tensors)** |
| MoE launcher `SPEC` plumbing | **missing** |
| MoE MTP validation (parity, acceptance, perf, soak) | **not performed** |
| Plan scope | deferred (`tasks_moe.md:28`) |

**Conclusion:** MTP on Qwen3.5-35B-A3B EXL3 is infrastructure-ready but deployment-blocked on checkpoint contents. The certified Gate 4 configuration remains the supported operating point.

---

## 8. How to re-check status

```bash
# Is there an MTP head in a checkpoint?
python - <<'EOF'
import json
idx = json.load(open("model.safetensors.index.json"))["weight_map"]
m = [k for k in idx if "mtp" in k.lower() or "nextn" in k.lower()]
print(f"{len(m)} MTP tensors of {len(idx)}")
EOF

# Did the server actually enable it?
grep -E "EXL3 modules|speculative_config=" logs/moe-p3/server-*.log | tail -4
```

Expected current output: `0 MTP tensors`, `0 in MTP head`, `speculative_config=None`.
