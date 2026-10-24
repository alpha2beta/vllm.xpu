# Checkpoint Schema — `turboderp/Qwen3.5-35B-A3B-exl3` @ `3.09bpw`

**Task:** P0-03 (checkpoint pin + tensor schema catalog)
**Status:** verified against the remote checkpoint headers, no weights downloaded
**Evidence:**
- `scripts/audit_moe_checkpoint.py`
- `logs/moe-preflight/checkpoint-moe-audit.json` (machine readable)
- `logs/moe-preflight/checkpoint-moe-audit.txt` (summary)
- `logs/moe-preflight/checkpoint-moe-tensors.csv.gz` (all 124,579 tensors)
- `scripts/fetch_exl3_fixtures.py` → `fixtures/exl3-moe/` (77 real tensors, 9.7 MB)
- `scripts/check_exl3_backends.py` → `logs/moe-preflight/exl3-backend-parity.{log,json}`

---

## 1. Pinned revisions

| role | branch | commit | bytes | shards |
| --- | --- | --- | --- | --- |
| **bring-up (chosen)** | `3.09bpw` | `f599771cd8ca99ce9bb62d58a855ecfbac60e3c2` | 15.29 GB payload | 2 |
| quality reference | `4.09bpw` | `39e6392b9fb84ad2216fd3362793cf80f7d409fc` | 19.58 GB (repo) | 3 |

Corrections to the plan (`tasks_moe.md` §P0-03 and the reference design doc):

- The branches `4.00bpw` / `3.00bpw` **do not exist** for this model; the
  published set is `2.08 / 2.13 / 2.54 / 3.09 / 3.54 / 4.09 / 6.00 bpw`.
  The commit `8a68a1ac…` named in the task list is not a commit of this repo.
- The checkpoint is **not** "4.00 bpw experts". Measured bitrates are
  heterogeneous (see §4): routed experts are **3-bit**, shared expert and
  attention are **4-bit**, `lm_head` is **6-bit**.

## 2. Model configuration (from `config.json`)

| field | value |
| --- | --- |
| architectures | `Qwen3_5MoeForConditionalGeneration` |
| text model_type | `qwen3_5_moe_text` |
| layers | 40 |
| layer pattern | 10 × (3 × `linear_attention` → 1 × `full_attention`) |
| hidden size | 2048 |
| experts | 256 routed + 1 shared, top-8 routed |
| moe intermediate | 512 (routed **and** shared) |
| activation | SiLU (gated) |
| attention | GQA, 16 Q / 2 KV heads, head_dim 256, partial RoPE 0.25 |
| GDN | 16 K heads / 32 V heads, head_dim 128, conv kernel 4 |
| vocab | 248,320 (padded) |
| max positions | 262,144 |
| vision | present (`vision_config`: depth 27, hidden 1152, patch 16, merge 2) |
| MTP | `mtp_num_hidden_layers: 1` in config, **no MTP tensors shipped** |

## 3. EXL3 quantization metadata

`quantization_config.json` (identical block embedded in `config.json`):

| key | value | meaning |
| --- | --- | --- |
| `quant_method` | `exl3` | |
| `version` | `0.0.22` | quantizer/library version that wrote the file — **not** the `1.4.2` the dense fleet uses |
| `bits` | `3.09` | file-wide average, **not** a per-tensor bitrate |
| `head_bits` | `6` | `lm_head` only |
| `calibration` | `{rows: 250, cols: 2048}` | |
| `out_scales` | `always` | every module carries `svh` |
| `codebook` | `mcg` | module-level default; confirmed by a `mcg` tensor in all 30,971 modules |
| `tensor_storage` | 31,243 entries | canonical per-tensor inventory |

Per-module payload is a four-tensor group (naming as in `exl3xpu/ref.py`):

```
<module>.trellis   I16  [k/16, n/16, 16*K]
<module>.suh       F16  [k]        input  sign/scale vector
<module>.svh       F16  [n]        output sign/scale vector
<module>.mcg       I32  []         scalar; presence selects the MCG codebook
```

No `mul1` tensors exist in this checkpoint; `mcg_multiplier` in
`tensor_storage` is `3417055213 == 0xCBAC1FED`, matching `ref.MCG_MULT`.

## 4. Quantization map — every module family

`K` = trellis bits per weight (`trellis[..., -1] / 16`). All codebooks `mcg`.

| family | projections | modules | k | n | K | bpw |
| --- | --- | --- | --- | --- | --- | --- |
| **routed experts** | `gate_proj` | 10,240 | 2048 | 512 | **3** | 3.0 |
| | `up_proj` | 10,240 | 2048 | 512 | **3** | 3.0 |
| | `down_proj` | 10,240 | 512 | 2048 | **3** | 3.0 |
| **shared expert** | `gate_proj` / `up_proj` | 80 | 2048 | 512 | 4 | 4.0 |
| | `down_proj` | 40 | 512 | 2048 | 4 | 4.0 |
| **GDN linear attention** | `in_proj_qkv` | 30 | 2048 | 8192 | 4 | 4.0 |
| | `in_proj_z` | 30 | 2048 | 4096 | 4 | 4.0 |
| | `out_proj` | 30 | 4096 | 2048 | 4 | 4.0 |
| **full attention** | `q_proj` | 10 | 2048 | 8192 | 4 | 4.0 |
| | `k_proj` / `v_proj` | 20 | 2048 | 512 | 4 | 4.0 |
| | `o_proj` | 10 | 4096 | 2048 | 4 | 4.0 |
| **lm_head** | — | 1 | 2048 | 248320 | **6** | 6.0 |

Totals: **30,971** quantized modules. `K` histogram `{3: 30720, 4: 250, 6: 1}`;
`4` = 120 shared-expert + 90 GDN + 40 full-attention modules.

`bitrate_mismatches`: none. Every `tensor_storage[*].bits_per_weight` equals the
Trellis `K` of the corresponding module.

## 5. Unquantized tensors (695 total, 333 of them vision)

| kind | count | dtype | example |
| --- | --- | --- | --- |
| `embed_tokens` | 1 | F16 | `[248320, 2048]` |
| router `mlp.gate.weight` | 40 | F16 | `[256, 2048]` |
| `mlp.shared_expert_gate.weight` | 40 | F16 | `[1, 2048]` |
| RMSNorm weights | 132 | F32/BF16 | `[2048]`, GDN `[128]` |
| GDN `in_proj_a` / `in_proj_b` | 60 | F16 | `[32, 2048]` |
| GDN `conv1d.weight` | 30 | BF16 | `[8192, 1, 4]` |
| GDN `A_log`, `dt_bias` | 60 | F32 / BF16 | `[32]` |
| vision tower | 333 | BF16/F16 | `visual.blocks.*`, `visual.merger.*` |

There is **no** `.bias` on any quantized projection (`attention_bias: false`).

## 6. Payload budget (drives the resident-memory plan)

| family | bytes | GiB | share |
| --- | --- | --- | --- |
| routed experts | 12,237,004,800 | 11.40 | 80.0 % |
| embedding | 1,017,118,720 | 0.95 | 6.7 % |
| vision | 893,142,496 | 0.83 | 5.8 % |
| attention (GDN + full) | 651,275,400 | 0.61 | 4.3 % |
| lm_head | 381,920,260 | 0.36 | 2.5 % |
| shared expert | 63,693,280 | 0.06 | 0.4 % |
| router | 41,943,040 | 0.04 | 0.3 % |
| norms / misc | 331,776 | 0.00 | 0.0 % |
| **total** | **15,286,429,772** | **14.24** | |

A text-only deployment that skips the vision tower needs ≈ 13.4 GiB of weights.

## 7. Hadamard / sign-vector facts (the load-bearing finding)

EXL3 is reconstructed as

```
W_orig = diag(suh) · H128_blocks(rows) · W_inner · H128_blocks(cols) · diag(svh)
y      = H128(x ⊙ suh) @ W_inner  →  H128(·) ⊙ svh
```

with `H128` the normalised 128-wide Sylvester Hadamard, applied in **blocks of
128** (not a full-dimension FWHT). Confirmed by `suh.shape == [k]` and
`svh.shape == [n]` for every module, and independently by bit-exact agreement
between `exl3xpu.ref` and the XPU Triton kernels on real tensors:

```
logs/moe-preflight/exl3-backend-parity.log   recon_diff = 0 for all 18 modules
```

### 7.1 `gate_proj` and `up_proj` do **not** share `suh`

The plan (and risk `R-09`) allows sharing the input transform only if
`suh_gate == suh_up`. Measured on 120 expert pairs (3 experts × 40 layers),
byte-comparison of the raw fp16 vectors:

```
checked_pairs: 120   identical: 0
```

A closer look shows the relationship is **scale-only**:

| layer/expert | sign agreement | `suh_gate / suh_up` mean | ratio std |
| --- | --- | --- | --- |
| 0 / 0 | 1.0000 | 1.0755 | 0.391 |
| 0 / 5 | 1.0000 | 1.0227 | 0.060 |
| 17 / 33 | 1.0000 | 1.0343 | 0.056 |
| 39 / 255 | 1.0000 | 1.1333 | 0.100 |

So the ±1 sign pattern is shared, but the per-channel magnitudes differ, and a
diagonal scale does **not** commute with `H128`. Consequence:

- gate and up need **two separate input transforms** at the Hadamard boundary;
- they can still be stacked in one `w13` parameter (identical `K`, `k`, `n`,
  same codebook), exactly like the dense fused `gate_up_proj` path, which
  already carries one `suh` row per fused member
  (`exl3xpu/ops.py::exl3_linear_impl`, `xh = [G, M, k]`).
- the kernel must not assume `suh_gate is suh_up`; the cheap sign-sharing
  optimisation is a later, optional micro-optimisation.

### 7.2 `svh` also differs between gate and up

Cosine similarity 0.9947 on the same expert; they are independent output
vectors and must be applied per member.

## 8. Vision tower

333 tensors, unquantized, 0.83 GiB: `patch_embed.proj`, `pos_embed`,
27 × `blocks.*` (12 tensors each), `merger.linear_fc1/fc2/norm`.
No 2-D/3-D vision tensor uses EXL3. The `Qwen3_5MoeForConditionalGeneration`
wrapper will therefore build a fully unquantized BF16 vision encoder, which the
EXL3 plugin must leave alone (`_member_bits` already returns `None` for
`visual.`/`vision_tower.` prefixes).

## 9. Integrity checks performed

| check | result |
| --- | --- |
| index ↔ header tensor set | 124,579 == 124,579, zero mismatches |
| per-layer routed expert count | 256 for all 40 layers |
| gate/up/down shapes within a projection | 1 variant per projection, all layers |
| `suh` shape == `[k]`, `svh` shape == `[n]` | 30,971 / 30,971 |
| `k`, `n` 128-aligned | yes (required by the H128 blocks) |
| `trellis[..., -1] % 16` | 0 for every module |
| shared expert + router + shared gate present in all layers | yes |
| `K` vs `tensor_storage.bits_per_weight` | 0 mismatches |
| anomalies recorded by the auditor | **0** |

## 10. ESIMD support matrix for these bitrates

`exl3xpu._C.exl3_supported(K, cb)` at the start of Gate 0 (the "before"
state, kept because it explains the first roofline numbers):

| K | codebook | supported | modules affected |
| --- | --- | --- | --- |
| 3 | `mcg` (cb=1) | **NO** | **30,720** (all routed experts) |
| 4 | `mcg` (cb=1) | yes | 250 |
| 6 | `mcg` (cb=1) | yes | 1 |

`cb==1` was instantiated only for `K ∈ {4, 6}` in `csrc/exl3_ops.sycl`, while
the kernels themselves are templates on `<K, CB>` and `nt_of()` already
special-cased `K == 3`. The Triton fallback is K-generic and bit-exact, so the
gap was performance, not correctness.

**Now resolved** (`_C.so` sha256 `4003cc2b…` r3, built with
`-DEXL3_ALL_CODEBOOKS`; carried into the current `7642ee7f…` r4):

| K | codebook | supported | small-M limit | modules affected |
| --- | --- | --- | --- | --- |
| 3 | `mcg` (cb=1) | **yes** | **M ≤ 8** (vector-only) | 30,720 routed experts |
| 4 | `mcg` (cb=1) | yes | `SMALL_M_MAX` (DPAS) | 250 |
| 6 | `mcg` (cb=1) | yes | `SMALL_M_MAX` (DPAS) | 1 |

`(K=3, cb=1)` has **no DPAS instantiation** (the 140V has no XMX units), so
`exl3_small_m_max(3,1) == 8` and `ops.small_m_max()` routes M > 8 to
reconstruct+GEMM; the previous Python layer assumed `SMALL_M_MAX = 128` for
every supported format and tripped
`exl3_gemm_small: M=9 exceeds no-DPAS vector limit 8`. Evidence:
`logs/moe-preflight/k3-mr-sweep.{log,json}` (M ∈ 1…128, 32/32 pass) and
`logs/moe-preflight/parity-esimd-forced/` (18 modules, forced ESIMD).

## 11. Reproduction

```bash
# schema audit (headers only, ~2 min, no weights downloaded)
.venv/bin/python scripts/audit_moe_checkpoint.py \
    --repo turboderp/Qwen3.5-35B-A3B-exl3 --revision 3.09bpw

# 77 real tensors (9.7 MB) for tests
.venv/bin/python scripts/fetch_exl3_fixtures.py \
    --repo turboderp/Qwen3.5-35B-A3B-exl3 --revision 3.09bpw

# CPU-vs-XPU parity on those tensors
.venv/bin/python scripts/check_exl3_backends.py

# forced-ESIMD parity (fails loudly instead of degrading to Triton)
EXL3_BACKEND=esimd .venv/bin/python scripts/check_exl3_backends.py \
    --log-dir logs/moe-preflight/parity-esimd-forced

# M-bucket dispatch sweep for the vector-only routed format
.venv/bin/python scripts/check_k3_mr_sweep.py

# AOT register/spill diagnostic for the 140V (lnl-a1)
bash scripts/esimd_aot_diag.sh
```
