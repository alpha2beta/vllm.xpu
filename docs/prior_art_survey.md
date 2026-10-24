# Prior Art Survey — EXL3 MoE kernels and checkpoint layouts (P0-11)

Reviewed 2026-10-23. The question this survey answers is narrow: *what should we
adopt, what should we re-derive, and what must we avoid?* It is not a feature
comparison of inference engines.

## 1. Sources reviewed

| # | Source | Version / commit reviewed | License | Access |
| --- | --- | --- | --- | --- |
| A | ExLlamaV3 (`turboderp-org/exllamav3`) | `master` tree as of 2026-10-23 | **MIT** | source read: `modules/block_sparse_mlp.py`, `moe_batch_recon.py`, `exllamav3_ext/quant/exl3_moe*` |
| B | exl3xpu (`sybil-solutions/exl3xpu`, archived; local fork) | `15ded2f` | **MIT** | full local checkout: `exl3xpu/`, `csrc/`, `docs/DESIGN.md` |
| C | vllm-exl3 (`vcruz305/vllm-exl3`) | `main`, 0.5.0-dev line | **AGPL-3.0-only** | README, `docs/VLLM_COMPATIBILITY.md`, release notes |
| D | glm-flash-lite + B70 expert server (`sybil-solutions/glm-flash-lite`) | `main` | **MIT** | README, `docs/b70-tier.md` |
| E | vLLM 0.31.0 installed tree | `0.31.0+xpu` | Apache-2.0 | local: `model_executor/layers/fused_moe/`, `models/qwen3_next.py` |

Nothing in C was copied. It is AGPL-3.0-only and this project is MIT/Apache-2.0;
only its *public interface choices* and its documented failure modes informed
this design. That boundary is restated in §7.

## 2. ExLlamaV3 — the reference MoE design (MIT)

ExLlamaV3 is the origin of the EXL3 format and the only implementation with a
mature routed-expert kernel family. Its Python layer is `BlockSparseMLP`
(`exllamav3/modules/block_sparse_mlp.py`). Structure:

* **Per-expert module lists.** `self.gates`, `self.ups`, `self.downs` each hold
  `num_local_experts` `Linear` objects. Each `Linear` owns its own
  `.trellis`, `.suh`, `.svh`, `K`, `mcg`/`mul1`.
* **Pointer tables.** `MultiLinear` builds device tensors of
  `ptrs_trellis`, `ptrs_suh`, `ptrs_svh` (one pointer per expert). The fused
  kernels take the pointer table plus `K`, `mcg`, `mul1` as launch parameters.
  This is the single most important structural idea to copy: **the expert
  dimension is data (a pointer table), not a Python loop.**
* **Kernel instantiations are compile-time `(K, codebook, N, rows)` tuples.**
  The CUDA extension has translation units
  `exl3_moe_inst_k{0..8}_cb{1,2}.cu` plus `_m32`/`_m64` row-tile variants and
  `_n128`/`_n256` width variants. `exl3_moe_inst_k3_cb1.cu` exists — **K=3 with
  the MCG codebook is a supported CUDA combination**, which de-risks adding the
  equivalent SYCL instantiation for our 30 720 routed modules.
* **Tiered execution, chosen by rows per expert.**
  1. `BC_BlockSparseMLP.run_bszN` — fully fused single launch for batch ≤ 8
     (`MAX_BSZN`), one slot per (token, expert) pick.
  2. `exl3_moe` fused kernel for moderate row counts (`TEMP_ROWS_FUSED`,
     widened to 256 rows with the 32/64-row tiles for `mul1`).
  3. Batched reconstruct tier (`moe_batch_recon.py`) for prefill: reconstruct
     expert weights in bounded groups and run dense GEMMs.
  4. Dense per-expert loop for configurations the fast paths do not support.
* **Deterministic accumulation is a first-class option.** `FUSED_DET`
  (default on) replaces atomic adds with slot + gather, so repeated runs are
  bit-identical. This matches our T-5 determinism gate; a nondeterministic
  atomic-sum MoE would fail Gate 2 regardless of accuracy.
* **Shared experts are not a routed expert.** `shared_experts` is a separate
  `MLP`/`GatedMLP`, optionally run as a one-expert fused launch (`sh_coop`) or
  through its own BC graph. This is the same ownership split vLLM uses on XPU
  (dense `Qwen3NextMLP` attached to `MoERunner`), and it is what we will keep.
* **Routing is a separate module** (`block_sparse_mlp_routing.py`): `std`,
  `std_bias`, `ds3`, `dots`, `sqrtsp`, `sqrtsp_hash`, each with a fixed-order
  top-k path. Qwen3.5 uses the plain `std` router (softmax, top-8,
  renormalize), which is the simplest case.
* **CPU/offload tiers** (`block_sparse_mlp_cpu.py`, `moe_cpu_host.py`) exist but
  are irrelevant at our memory budget.

**What to adopt (concepts, MIT-compatible):** pointer-table descriptors;
compile-time `(K, codebook)` instantiation matrix; row-tier dispatch; slot +
gather determinism; shared expert outside the routed path.

**What not to adopt:** the CUDA kernels themselves. They rely on `__hfma`
rounding, DPAS/`mma` shapes, and warp-level swizzles that do not map to Xe2
ESIMD. Porting them line-by-line would be slower than writing the SYCL
equivalent against our existing `exl3_ops.sycl` primitives.

## 3. exl3xpu — our starting point (MIT)

The fork already solves the dense half of the problem:

* `exl3xpu/vllm_plugin.py` registers `exl3` with
  `register_quantization_config` and resolves dense linears to
  `Exl3LinearMethod`. It returns `None` for `RoutedExperts` — that is exactly
  the seam P1-01 must fill.
* `exl3xpu/ops.py` dispatches fused small-M GEMMs (M ≤ 128) and
  reconstruct + oneDNN GEMM above that, with per-shard `suh` handling
  (`xh = [G, M, k]`) for vLLM's fused sibling projections.
* `csrc/exl3_ops.sycl` has the ESIMD `launch_reconstruct<K, cb>` and
  `dispatch_mr<K, cb>` tables. At survey time only `(K=4, mcg)` and
  `(K=4, mul1)` were instantiated and `exl3_supported(3, 1)` was `false`, so
  all K=3/MCG routed experts fell back to Triton at ~1.3 GB/s. **That gap is
  now closed** (`(K=3, cb=1)` built, 130 µs vs 311 µs per M=1 GEMV) — see
  `docs/compatibility_manifest.md` §5.
* `exl3xpu/ref.py` is a bit-exact CPU decoder validated against ExLlamaV3's CUDA
  reconstruct; our P0-04 parity run reproduces it with `recon_diff = 0`.

**Conclusion:** extend exl3xpu in place. The missing pieces are (a) a MoE quant
method and loader, (b) a grouped-expert op, (c) the K=3/cb=1 ESIMD
instantiation. No new engine and no fork of ExLlamaV3.

## 4. vllm-exl3 — the vLLM seam (AGPL-3.0, read-only reference)

This is the only public project that plugs EXL3 routed experts into vLLM's
modern MoE stack. Its documented integration surface answers the P0-02 question
“where does an EXL3 MoE method attach?”:

* `Exl3MoEMethod` subclasses vLLM's `FusedMoEMethodBase` and implements
  `create_weights` and `get_fused_moe_quant_config` against vLLM v0.30.0.
* It consumes `RoutedExperts` and the `w13_*` / `w2_*` packed payload naming
  (`w13_trellis`, `w13_suh`, `w13_svh`, `w2_*`).
* It has a mixed-K fused dispatch (`exl3_moe_mixedk`) and per-bit row caps
  (`VLLM_EXL3_NATIVE_MOE_MAX_ROWS_K2/_K3/_K4`), with **explicit fallbacks**
  when a shape is not eligible.
* The grouped-prefill planner is present but `execution_available` is false —
  the project says so itself. Do not mistake planner existence for a kernel.
* Its fast path is gated to **K4/MCG**; K2/K3 and distinct gate/up rotations
  keep fallback paths. That matches our own finding that K=3/MCG is the gap.
* An experimental UVA expert-placement guard exists for host-mapped expert
  payloads. Not relevant to us: our routed payload is 11.4 GiB and fits in
  device memory.

**What to adopt:** the *interface shape* — `Exl3MoEMethod` over
`FusedMoEMethodBase`, `RoutedExperts` as the weight owner, `w13`/`w2` grouping,
per-bit row caps, fail-closed eligibility. These are vLLM API facts, not
copyrightable implementation.

**What to avoid:** (1) copying any code — AGPL-3.0-only is incompatible with
shipping this work under the repo's current MIT/Apache-2.0 terms; (2) inheriting
its “prefer fallback” behavior. Our spec forbids
`UnquantizedFusedMoEMethod` fallbacks, so an ineligible configuration must
raise, not silently degrade. (3) Its TP4/grouped-prefill work: not on our
TP=1 path.

## 5. sybil-solutions B70 expert server — grouped launch over a pointer table (MIT)

`glm-flash-lite` runs GLM-5.3-Flash (12 096 routed experts) on one 24 GB GPU by
tiering experts across VRAM, RAM, NVMe, and optionally an **Arc Pro B70 expert
server**. The B70 tier is the closest existing thing to what we are building:

* The B70 container holds ~3 000 experts in its 32 GB and runs **exl3xpu's
  grouped MoE kernel** (`b70tier/kernels/moe-n128`, SwiGLU clamp 10) over **a
  pointer table of resident experts** (per `docs/b70-tier.md`).
* The host posts `(expert pick, input row)` pairs to a shared-memory ring; the
  card returns weighted partial rows. Cost: **~0.17 ms per layer call +
  0.021 ms per expert**; ~0.4 ms/token wall time because it overlaps with the
  CPU lane.
* Quality: B70 expert outputs vs an fp64 reference have **relative L2
  0.0013–0.0015** on 48 real experts; the AVX2 CPU tier gets 0.0010. The
  pointer-table + grouped-kernel path is therefore numerically sound, not just
  fast.
* Out-of-range expert picks are **masked inside the kernel** and contribute
  exact zeros. This is the pattern to reuse if we ever shard experts (TP/EP);
  it avoids host-side filtering and keeps the launch shape static.

**Relevance to the 140V:** we do not need tiering — 15.29 GB of checkpoint
fits in 28.58 GiB of GPU-visible memory with room for KV. But the *kernel
contract* (pointer table + grouped launch + in-kernel masking + SwiGLU clamp)
is directly reusable as a design, and the measured 0.021 ms/expert gives a
sanity bound for our Phase 4 expert-GEMV target.

## 6. Consolidated design decisions

| Decision | Adopt from | Rationale |
| --- | --- | --- |
| Attach via `FusedMoEMethodBase` + `RoutedExperts` | C (vLLM API) | only supported seam in vLLM 0.31; no core patch |
| Packed payloads `w13_*` / `w2_*`, no bf16 master copy | A, C | 11.4 GiB routed budget |
| Per-expert descriptor table (pointers + `K` + codebook + scales) | A, D | expert dim as data; enables one launch per group |
| Compile-time `(K, codebook)` instantiation matrix | A, B | K=3/MCG is a real combination (A ships it); we must add it to SYCL |
| Row-tier dispatch: fused small-M → grouped GEMV → batched reconstruct | A | decode is M=1; prefill needs dense GEMM |
| Deterministic slot + gather accumulation | A | T-5 bit-identical repeat runs |
| Shared expert stays a dense EXL3 linear | A, E | matches vLLM XPU ownership; 0.06 GiB |
| Fail-closed on unsupported configs | B, our spec | no unquantized fallback |
| In-kernel masking of out-of-range expert picks | D | future TP/EP without shape churn |
| No code from `vllm-exl3` | C license | AGPL-3.0-only vs MIT/Apache-2.0 |

## 7. Licensing matrix and compliance

| Component we ship | Origin | License | Obligation |
| --- | --- | --- | --- |
| `exl3xpu` core + `csrc/` (modified) | B | MIT | keep copyright + permission notice |
| New `exl3xpu/moe.py`, SYCL grouped kernels | clean-room, informed by A/D designs | MIT | ours; attribute inspiration in comments where non-obvious |
| vLLM plugin glue | E (API only) | Apache-2.0 | none beyond vLLM's own terms |
| `vllm-exl3` | C | AGPL-3.0-only | **no code, no derived code**; interface names only |

Practical rule for reviewers: a change is acceptable if it can be explained
from (i) the EXL3 format spec in `docs/checkpoint_schema.md`, (ii) the existing
`csrc/exl3_ops.sycl` primitives, and (iii) public vLLM API signatures. If a
diff only makes sense by having read `vllm-exl3` source, it is rejected.

## 8. What this means for Phase 1

1. **P1-01** fills exactly the `RoutedExperts → None` hole in
   `exl3xpu/vllm_plugin.py`, with `Exl3MoEMethod(FusedMoEMethodBase)`. Follow
   vLLM's contract, not vllm-exl3's file layout.
2. **P1-02/P1-03** implement the pointer-table descriptor design from A/D:
   per-layer slab of `[256, ...]` packed tensors plus a device int64 pointer
   table, no per-expert Python objects in the hot path.
3. **Checkpoint adapter** must map this checkpoint's
   `experts.{i}.{gate,up,down}_proj.{trellis,suh,svh,mcg}` into vLLM's expected
   `experts.w13_weight` / `experts.w2_weight` (+ sub-suffixes). vLLM's default
   `RoutedExperts` loader assumes a different suffix set, so a name/shape
   adapter is mandatory (documented in `docs/moe_api_audit.md`).
4. **Kernel work** starts from the now-published `(K=3, cb=1)` instantiation —
   the same combination ExLlamaV3 ships on CUDA — and the next step is a
   grouped variant over the descriptor table.
5. **Performance expectation** stays bounded by the roofline in
   `docs/compatibility_manifest.md`: ~70 tok/s single-stream ceiling. The
   Triton fallback penalty is gone, but a single M=1 expert GEMV is still
   latency-bound (~3 GB/s), so expert grouping is what unlocks the bandwidth.

## 9. References

* ExLlamaV3: <https://github.com/turboderp-org/exllamav3> (MIT)
* exl3xpu (archived upstream): <https://github.com/sybil-solutions/exl3xpu> (MIT)
* vllm-exl3: <https://github.com/vcruz305/vllm-exl3> (AGPL-3.0-only)
* glm-flash-lite / B70 tier: <https://github.com/sybil-solutions/glm-flash-lite>, `docs/b70-tier.md` (MIT)
* vLLM 0.31.0 XPU installed tree: `$ROOT/.venv/lib/python3.12/site-packages/vllm/` (Apache-2.0)
