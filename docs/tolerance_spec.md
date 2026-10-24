# Numerical Tolerance Specification (P0-09)

This document fixes the error metrics and pass/fail thresholds **before** any
MoE kernel work. Every subsequent gate cites these definitions; changing them
requires a new revision of this file with a written justification and a
re-baselined oracle.

## 1. Reference of record

| Reference | Artifact | Precision | Use |
| --- | --- | --- | --- |
| Trellis dequantizer | `exl3xpu.ref.weight_orig` | bit-exact, CPU | weight reconstruction |
| Expert / router / block oracle | `fixtures/reference_moe_traces.npz` (built by `tests/ref_moe_oracle.py`) | fp32 CPU | P1–P3 unit and block tests |
| Full-layer logits | teacher-forced CPU run of the pinned checkpoint | fp32 CPU | Gate 2 / Gate 3 (KL) |
| External CUDA traces | *pending* — no CUDA node available | reference ExLlamaV3 | independent cross-check only, never the gate |

The CPU oracle is built from **real checkpoint tensors** (layer 0, experts 0 and
255, shared expert, router) fetched by `scripts/fetch_exl3_fixtures.py`. It is
deterministic: `tests/ref_moe_oracle.py --verify` rebuilds and compares all 26
arrays; a single changed element fails the check. The oracle is the authority
for Phase 1–3; the pending CUDA traces can only *contradict* it, in which case
the discrepancy is investigated before either is trusted.

Canonicalization rule: the reference always consumes the **same dtype-cast
activations** handed to the kernel. A bf16 kernel is compared against an fp32
matmul over bf16-cast inputs and bf16-cast dequantized weights. Comparing an
fp32-input reference against a bf16 kernel silently spends the bf16 rounding
budget on the harness, not the kernel.

## 2. Metric definitions

All metrics are computed in fp32 (or fp64 for KL accumulation) after casting
kernel outputs up. No metric is computed in bf16/fp16.

**Weight reconstruction error**

```
recon_diff(module) = number of elements where
                     bitcast_u16(W_kernel) != bitcast_u16(W_ref)
```

Bytes, not values: `+0.0` and `-0.0` must compare equal, so compare raw
exponent/mantissa payload after normalizing signed zero.

**Elementwise linear error**

```
abs_err = |y_kernel - y_ref|
rel_err = abs_err / max(|y_ref|, eps)        eps = 1e-3
max_abs = max over all elements and samples
p99_abs = 99th percentile over all elements and samples
```

**Router cosine**

```
cos_t = <logits_ref[t], logits_test[t]> / (||logits_ref[t]|| ||logits_test[t]||)
cos_min = min over tokens t
```

**Top-8 agreement**

```
agree_t = set(top8(logits_test[t])) == set(top8(logits_ref[t]))
```

Ties are broken by ascending expert index (PyTorch `topk` order) so the set
comparison is well defined. Selection *order* is not required to match; vLLM
sorts by weight before dispatch, not by logit.

**Full-layer KL**

```
p = softmax(logits_ref, dim=-1, dtype=fp64)
q = softmax(logits_test, dim=-1, dtype=fp64)
KL(p || q) = sum_v p_v * (log p_v - log q_v)
```

Reported as mean and max over evaluated token positions. Logits are compared
**after** the final RMSNorm and LM head, at temperature 1.0.

## 3. Thresholds

| ID | Check | Metric | Threshold | Applies at |
| --- | --- | --- | --- | --- |
| T-1 | Weight reconstruction | `recon_diff` | **== 0** (bit-exact) | every module, every gate |
| T-2 | Expert/dense GEMV | `max_abs` | **≤ 0.08** | M ∈ {1,2,4,8}, all projections |
| T-2b | Expert/dense GEMV (diagnostic) | `p99_abs`, `rel_err` | reported, non-gating | every T-2 run |
| T-3 | Router logits | `cos_min` | **≥ 0.999** | every token in the trace suite |
| T-3b | Router selection | top-8 agreement | **100 %** of tokens | every token in the trace suite |
| T-4 | Full-layer logits | mean `KL(p‖q)` | **≤ 1e-4** | Gate 2 / Gate 3 suite |
| T-4b | Full-layer logits (diagnostic) | max `KL(p‖q)` | reported, non-gating | Gate 2 / Gate 3 suite |
| T-5 | Determinism | repeat-run diff | **bit-identical** | every gate run |
| T-6 | Finiteness | NaN/Inf count | **== 0** | every gate run |

A gate passes only when every applicable threshold passes. Diagnostics (T-2b,
T-4b) are recorded to catch slow drift but never promote or block on their own.

## 4. Calibration anchors

Measured on the installed stack before this spec was frozen
(`logs/moe-preflight/exl3-backend-parity.log`):

| Case | Measured | Threshold | Headroom |
| --- | --- | --- | --- |
| Trellis reconstruction, 18 modules | `recon_diff = 0` | 0 | exact |
| K=3/mcg GEMV, worst `max_abs` (M=1..8) | 4.2e-5 | 0.08 | ~1900× |
| K=4/mcg GEMV, worst `max_abs` (M=1..8) | 8.7e-5 | 0.08 | ~920× |
| Triton vs CPU dequant weight equality | bit-exact | 0 | exact |

The 0.08 absolute bound is therefore ~3 orders of magnitude above the observed
bf16 matmul floor for these shapes. It is intentionally loose: it is a
*correctness* gate meant to catch routing, layout, or dequantization bugs (which
produce O(0.1–10) errors), not to rank kernel quality. Kernel quality is tracked
separately by the roofline numbers in `docs/compatibility_manifest.md`.

## 5. Test suites and sampling

| Suite | Inputs | Tolerances |
| --- | --- | --- |
| `tests/test_expert_loader.py` | all 256 experts × 3 projections, layer 0 | T-1, T-6 |
| `tests/test_expert_gemv.py` | fixtures experts {0, 255}, M ∈ {1,2,4,8}, seeded activations | T-1, T-2, T-5, T-6 |
| `tests/test_router.py` | `router/logits` trace, 4 tokens × 256 experts | T-3, T-3b, T-6 |
| `tests/test_moe_block.py` | `block/y` trace (restricted routing {0, 255}) | T-2 scaled to block output, T-5, T-6 |
| Gate 2 (single layer, eager) | 32 prompts × 128 positions, greedy | T-1…T-4b, T-5, T-6 |
| Gate 3 (full model, eager) | same suite as Gate 2 | T-1…T-4b, T-5, T-6 |

Seeds: activations use `torch.manual_seed(1234)` then a documented per-case
offset. The exact input tensors are stored in the oracle `.npz` for the unit
suites, so a rerun does not depend on RNG implementation details.

Input distribution for Gate 2/3: natural text prompts (no synthetic
adversarial inputs). “Normal inputs” in the plan is interpreted as
teacher-forced tokens from the evaluation prompts, 128 tokens each. Any token
that produces a top-8 tie is logged; ties are resolved by index as above.

## 6. Composition rules

1. **Per-kernel first.** A block/full-layer failure is bisected down to the
   smallest failing component before any tolerance is relaxed. Relaxing a
   downstream tolerance to hide an upstream T-1/T-2 failure is forbidden.
2. **No averaging away outliers.** T-2 is a max, not a mean. A single element
   above 0.08 fails the case even if `p99` is tiny.
3. **Determinism precedes accuracy.** A run whose repeat output differs is
   reported as T-5 failure and its accuracy numbers are discarded.
4. **Error budget is not additive.** Do not sum per-stage budgets; each stage
   must independently meet its threshold against the CPU oracle. This prevents
   a chain of “within tolerance” stages from producing an out-of-tolerance
   model.
5. **Routing differences invalidate token outputs.** If top-8 differs for a
   token, that token's logits are excluded from T-4 only after the T-3b failure
   has been recorded and investigated. Silent exclusion is not allowed.

## 7. Promotion and reporting

Each gate writes a JSON report with: git SHAs, environment pins, checkpoint
revision, dtype, seed, per-case metrics (T-1…T-6), and the raw worst-case
element index for any T-2 failure. Reports live under `logs/` next to the run
log. A gate is “passed” only if the report contains `"pass": true` and every
threshold above is satisfied; prose claims without a report are not evidence.

## 8. Known exceptions

* **External CUDA traces are not available.** T-4 is therefore measured against
  the CPU teacher-forced reference, which is stricter in precision but not
  independent of the local dequantizer. The CUDA cross-check remains an open
  P0-08 item and must be completed before any public performance claim; it does
  not block Gates 1–3 because those gates are defined against the CPU oracle.
* **K=3/MCG currently runs on Triton, not ESIMD.** Tolerances are path-agnostic;
  the same thresholds apply before and after the ESIMD instantiation is added,
  and the parity harness is re-run on both paths when it lands.
