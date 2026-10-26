# MTP Status — Qwen3.5-35B-A3B EXL3 MoE on vLLM XPU (Arc 140V)

**Date:** October 25, 2026 (addendum and validation October 26, 2026)
**Scope:** Original assessment was read-only. On October 26 an MTP-bearing checkpoint was retrieved, `SPEC` support was added to the MoE launcher (default off), and the MTP path was validated (§9). No serving default was changed: the certified Gate 4 configuration still runs without speculation.
**Checkpoint under test:** `turboderp/Qwen3.5-35B-A3B-exl3` @ 3.09 bpw, rev `f599771c…`, local path `/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw`
**Serving stack:** vLLM XPU v0.31.0, `enforce-eager`, TP=1, `max_num_seqs=1`, 2048 context, port 8002

---

## 1. Bottom line

**MTP is not supported for this MoE deployment today, and it is not merely switched off — the shipped checkpoint does not contain an MTP head.**

`text_config.mtp_num_hidden_layers = 1` advertises an MTP layer, but the weight files contain **0 MTP/NextN tensors**. Everything else needed to run MTP (upstream model class, plugin key remapping, draft-head vocabulary pruning, dense-launcher plumbing) already exists and is exercised on the dense EXL3 path. The MoE path is missing exactly two things:

1. a checkpoint revision that actually ships the `mtp.*` weights, and
2. `SPEC` plumbing in `scripts/launch_vllm_moe_exl3.sh`.

MTP is explicitly out of scope for the current plan (`tasks_moe.md:28`) and is deferred until a validated checkpoint exists. No speculative decoding was enabled or benchmarked on the MoE model for this report.

**Addendum (October 26, 2026):** both gaps have since been closed — an MTP-bearing candidate checkpoint was retrieved and `SPEC` plumbing was added to the MoE launcher — and a validation run was performed (§9). Result: MTP loads and decodes correctly, but K=1 is a **0.39× decode regression** on this stack, so `SPEC` remains off by default. The analysis below still describes the pinned 3.09 bpw Gate 4 artifact.

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
| `scripts/launch_vllm_moe_exl3.sh` (MoE) | supports `SPEC=mtp-k1|mtp-k2|ngram-k<N>`, default **empty (off)** as of Oct 26 |

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
2. **No launcher plumbing.** Even with weights, `scripts/launch_vllm_moe_exl3.sh` could not pass a speculative config. *(Resolved Oct 26: `SPEC` support added, §9.)*
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
| MoE checkpoint MTP weights | **available in candidate (`Sage-A1.5` 4.0 bpw, §9); absent in pinned 3.09 bpw** |
| MoE launcher `SPEC` plumbing | **available, default off** |
| MoE MTP validation (parity, acceptance, perf, soak) | **load + acceptance + A/B perf done (§9): K=1 = 0.39× decode; no soak** |
| Plan scope | deferred (`tasks_moe.md:28`) |

**Conclusion:** MTP is now fully plumbed and loads on the MoE path, but it is **not a deployment win**: on an MTP-bearing 4.0 bpw checkpoint, K=1 achieved 86.6 % acceptance yet decoded at 0.39× the non-MTP rate, because the M=2 verify batch bypasses the M=1 fast path. The certified Gate 4 configuration (no speculation) remains the supported operating point; MTP needs an M≥2 verify kernel before it can be reconsidered.

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

For the pinned 3.09 bpw checkpoint the expected output is `0 MTP tensors`, `0 in MTP head`, `speculative_config=None`.

---

## 9. Addendum — MTP-capable candidate checkpoint (October 26, 2026)

**Retrieved and verified:** `sam33339999/Sage-A1.5-Heretic-Abliterated-Uncensored-exl3-4.0bpw` @ `fc5e3f5581c6cf004c58013e1d7af9997ebd569a`
**Local path:** `/home/yanchun/models/Sage-A1.5-EXL3-4.0bpw-mtp`
**Size:** 18.38 GiB, 3 shards, 127,692 tensors
**Quantization:** EXL3 1.6.4, `bits=4.0`, `head_bits=6`, **`mtp_bits=4`**, `codebook=mul1`

This is an abliterated/uncensored derivative of the same architecture family (`Qwen3_5MoeForConditionalGeneration`, 40 layers, hidden 2048, 256 experts top-8, moe intermediate 512, `mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=false`). It is a **4.0 bpw** checkpoint, not the pinned 3.09 bpw Gate 4 artifact, and it is a different fine-tune — so it is suitable for MTP functional bring-up and performance measurement, not for Gate 3/4 numerical parity claims.

### MTP head inventory

| Item | Value |
|---|---|
| `mtp.*` tensors | **3,113** |
| Quantized linears in MTP head | 776 (`fc` + 256 routed experts × 3 projections + shared expert × 3 + attention × 4) |
| MTP head size | **0.398 GiB** |
| `mtp.fc` | Linear 4096 → 2048, EXL3-quantized, `trellis [256, 128, 64]` (4.0 bpw), no bf16 `.weight` |
| `mtp.layers.0` | full-attention layer + 256 routed experts + shared expert; router `mlp.gate.weight [256, 2048]` and `shared_expert_gate.weight` are bf16 |
| Norms | `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `norm` (bf16) |
| Embeddings | shared with the main model (no `mtp.embed_tokens`) |

Representative shapes (all consistent with 4 bpw, 16-column trellis blocks):

| Tensor | Shape | Dtype |
|---|---|---|
| `mtp.fc.trellis` | `[256, 128, 64]` | I16 |
| `mtp.layers.0.self_attn.q_proj.trellis` | `[128, 512, 64]` | I16 |
| `mtp.layers.0.mlp.experts.0.gate_proj.trellis` | `[128, 32, 64]` | I16 |
| `mtp.layers.0.mlp.experts.0.down_proj.trellis` | `[32, 128, 64]` | I16 |
| `mtp.layers.0.mlp.gate.weight` | `[256, 2048]` | F16 |

### Integrity check

- Index ↔ shard mapping: **0 missing, 0 extra, 0 duplicate, 0 shard mismatches** across all 127,692 tensors.
- Per-shard counts: 52,803 / 62,120 / 12,769; all three `safetensors` headers parse cleanly.
- `quantization_config.tensor_storage` omits the MTP head (0 `mtp.*` keys), which the plugin already handles by defaulting `mtp.` tensors to `mtp_bits=4`.
- Byte accounting: decoder 15.85 GiB, embeddings + LM head 0.95 GiB, vision tower 0.83 GiB (bf16, unused with `--language-model-only`), MTP 0.40 GiB, other 0.36 GiB.

### Validation run (October 26, 2026)

`SPEC` plumbing was added to `scripts/launch_vllm_moe_exl3.sh` (default off, so the certified Gate 4 path is unchanged) and the Sage checkpoint was served on port 8002 with `SPEC=mtp-k1`, `MAX_LEN=1024`, `BATCHED_TOKENS=256`, `GPU_UTIL=0.72`, `EAGER=1`.

**Loader — pass:**

```
Resolved architecture: Qwen3_5MoeMTP
exl3: 31747 EXL3 modules (776 in MTP head)
speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=1)
```

Startup 55 s; KV cache 2,150 tokens (3 padding layers added for the MTP drafter, 512 MiB budget); generation coherent.

**Acceptance and throughput — A/B on the same checkpoint, same six natural prompts, 96 output tokens each:**

| Metric | Baseline (no spec) | MTP K=1 | Delta |
|---|---|---|---|
| Aggregate decode | **24.29 tok/s** | **9.53 tok/s** | **0.39× (−61 %)** |
| Per-case range | 15.4–29.2 | 8.1–11.9 | slower in every case |
| Tokens per step | 1.000 | 1.866 | +87 % |
| Step time | 41.2 ms | 196.5 ms | 4.77× |
| Acceptance (`accepted/draft`) | — | 266/307 = **86.6 %** | — |
| TTFT | 0.62–2.25 s | 0.88–1.09 s | no clear effect |

Evidence: `logs/moe-p3/server-sage-mtp-k1.log`, `server-sage-base-nomtp.log`, `sage-mtp-k1-natural.json`, `sage-base-nomtp-natural.json`, `sage-mtp-k1-smoke.json`; probes in `scripts/mtp_smoke.py` and `scripts/mtp_acceptance.py`.

**Diagnosis:** acceptance is high (the model emits templated `<think>` reasoning, easy for the drafter), yet each MTP step costs 4.8× a baseline step. A step is a draft forward at M=1 plus a verify forward at M=2; only M=1 uses the P4-01 zero-copy fast path / P4-02 grouped kernel, so the verify batch falls into the slower eager multi-token path. Accepted tokens per step (1.87) cannot recover a 4.8× per-step cost.

**Verdict:** MTP is now functionally plumbed and loads correctly, but **K=1 is a net decode regression (0.39×) on this stack**. Do not enable it for deployment. The concrete lever is a grouped/verify kernel for M≥2 (the P4-03 grouped-prefill GEMM or a dedicated 2-token verify kernel); until that exists, `SPEC` stays empty.

### Implications

- **Memory:** language-only weights are ~17.15 GiB + 0.40 GiB MTP ≈ 17.55 GiB, about **+2.5 GiB** versus the 3.09 bpw artifact, whose Gate 4 peak was 18.36 GiB GPUActive at `GPU_UTIL=0.62` (2048 context). The observed bring-up used `GPU_UTIL=0.72` at 1024 context with a 512 MiB KV budget and loaded comfortably.
- **Loader:** the per-expert `mtp.` prefix is handled correctly by the plugin MoE stacking path — the load log reports 776 MTP modules, as predicted.
- **Kernel path:** verify batches are `k+1` tokens, so the M=1 grouped decode kernel is bypassed and the eager P4-01 loop serves MTP steps. This is the measured bottleneck.
- **Remaining work if MTP is pursued:** kernel support for M≥2 verify, then re-run this A/B; plus PPL/parity and a soak.

