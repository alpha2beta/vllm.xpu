# vLLM 0.31.0 MoE API Audit — extension boundary for EXL3

**Task:** P0-02 (audit installed vLLM MoE interfaces and hooks)
**Scope:** `vllm 0.31.0+xpu` as installed in `.venv/lib/python3.12/site-packages/vllm/`
**Status:** complete; all line numbers verified against the installed tree
**Companion evidence:** `docs/checkpoint_schema.md`, `logs/moe-preflight/environment-moe.json`

Paths below are relative to `.venv/lib/python3.12/site-packages/vllm/`.

---

## 1. Call chain for Qwen3.5-35B-A3B

```
Qwen3_5MoeForConditionalGeneration
└── Qwen3_5Model                      model_executor/models/qwen3_5.py:219
    └── Qwen3_5DecoderLayer           qwen3_5.py:122
        ├── linear_attn | self_attn   (GDN or full attention, per layer_types)
        └── self.mlp = Qwen3NextSparseMoeBlock
                                      qwen3_5.py:169   (only when
                                      config.model_type == "qwen3_5_moe_text")
            ├── self.gate               = GateLinear          qwen3_next.py:186
            ├── self.shared_expert_gate = ReplicatedLinear    qwen3_next.py:192
            ├── self.shared_expert      = Qwen3NextMLP        qwen3_next.py:204
            └── self.experts            = FusedMoEFactory     qwen3_next.py:218
                └── MoERunner                                 fused_moe/runner/moe_runner.py:228
                    ├── self.routed_experts  (RoutedExperts)  fused_moe/routed_experts.py:66
                    └── self._shared_experts (SharedExperts)  moe_runner.py:283-290
```

Notes that matter for us:

- `Qwen3_5DecoderLayer.__init__` calls `super(Qwen3NextDecoderLayer, self).__init__()`
  (`qwen3_5.py:129`), i.e. it deliberately skips `Qwen3NextDecoderLayer.__init__`
  and builds attention + MLP itself. There is **no** per-model MoE override.
- Both the routed experts and the shared expert are owned by the **same**
  `FusedMoEFactory` call (`qwen3_next.py:218-238`), not by the sparse block.
- `Qwen3NextSparseMoeBlock.forward` (`qwen3_next.py:240`) calls
  `self.experts(hidden_states=hidden_states, router_logits=hidden_states)`.
  The argument named `router_logits` is the **hidden state**; logits are computed
  inside the runner. Do not be misled by the name.

## 2. Routing and token dispatch

- `GateLinear` (`fused_moe/router/gate_linear.py`) owns the router weights
  (`mlp.gate.weight`, F16, `[256, 2048]` in this checkpoint). It is **not**
  quantized: `Exl3Config.get_quant_method` returns `None` for it and vLLM falls
  back to the unquantized linear method.
- `MoERunner` computes `topk_weights` / `topk_ids` and then calls
  `quant_method.apply(layer, x, topk_weights, topk_ids, shared_experts,
  shared_experts_input)` (`fused_moe_method_base.py:143-169`).
- The runner dispatches through one of two custom ops selected in
  `MoERunner._select_forward` (`moe_runner.py:307-322`):
  `torch.ops.vllm.moe_forward` or `torch.ops.vllm.moe_forward_shared`
  (the `_shared` variant is used when a shared expert is attached, which is our
  case — see §4).
- `renormalize` comes from `config.norm_topk_prob` (`qwen3_next.py:227`);
  Qwen3.5 sets it implicitly to `True`, matching the reference top-8 softmax
  normalisation used in `tests/ref_moe_oracle.py`.

## 3. The quantization extension contract

`FusedMoEMethodBase` (`fused_moe/fused_moe_method_base.py:28`) is the only
supported extension point for a custom expert format. Our `Exl3MoEMethod` must
implement:

| member | requirement |
| --- | --- |
| `create_weights(layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra)` | abstract (`:49-59`); must create **all** expert parameters on `layer` |
| `get_fused_moe_quant_config(layer)` | abstract (`:102-105`); may return `None` if we never use a vLLM kernel |
| `apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)` | the modular entry point (`:143-169`); default raises `NotImplementedError` |
| `apply_monolithic(layer, x, router_logits, input_ids=None)` | alternative entry point (`:166-190`); only used when `is_monolithic` is true |
| `supports_internal_mk` | `moe_kernel is not None` (`:35-39`); keep `False` |
| `is_monolithic` | `False` when `moe_kernel is None` and no `experts_cls` attr (`:129-135`) → the runner computes top-k and calls `apply` |

`RoutedExperts` construction (`routed_experts.py:145-196`):

```python
self.quant_method = self._get_quant_method(prefix, quant_config, moe_config)  # :145
self.quant_method.maybe_roundup_sizes(...)                                    # :154
self.quant_method.create_weights(layer=self, **moe_quant_params)              # :193
```

### 3.1 Silent unquantized fallback — the trap

```python
# routed_experts.py:207-222
quant_method = None
if quant_config is not None:
    quant_method = resolve_quant_method(quant_config, self, prefix)
if quant_method is None:
    quant_method = UnquantizedFusedMoEMethod(moe_config)      # :220
```

`Exl3Config.get_quant_method(layer, prefix)` used to handle only `LinearBase`
and `ParallelLMHead` and return `None` for everything else — including
`RoutedExperts`. Consequence: **without an `Exl3MoEMethod`, vLLM silently
allocates 256 × (2048×512 + 2048×512 + 512×2048) fp16 expert weights per layer
≈ 1.6 GiB/layer ≈ 64 GiB for the model**, and then fails on the missing
`.weight` tensors (or OOMs first). The spec forbids this path; the adapter must
therefore fail fast rather than fall back.

**Closed by P1-01.** `Exl3Config.get_quant_method` now returns
`exl3xpu.moe.Exl3MoEMethod` for `RoutedExperts`, and that constructor
*validates* before returning:

| Check | Behaviour on failure |
| --- | --- |
| checkpoint storage map non-empty | `Exl3MoEUnsupported` |
| expert tensors exist under this layer's normalized prefix | `Exl3MoEUnsupported` |
| every expert has gate/up/down, and the expert count matches `num_local_experts` | `Exl3MoEUnsupported` |
| one trellis K across all experts | `Exl3MoEUnsupported` |
| `(K, codebook)` has an ESIMD kernel | `Exl3MoEUnsupported` |
| `tp_size == ep_size == 1`, EP off | `Exl3MoEUnsupported` |

`create_weights` (P1-02) and `apply` (P1-03) raise `NotImplementedError`
stubs, so a model load stops with an actionable message instead of allocating
unquantized experts. Evidence: `tests/test_dispatch.py`,
`logs/moe-preflight/test-dispatch.{log,json}`.

### 3.2 Weight-loading contract

`RoutedExperts` exposes generic loaders (`routed_experts.py:509` `_load_w13`,
`:560` `_load_w2`, `:622` `weight_loader`) that copy checkpoint tensors into
`param.data` slices. The parameter names are fixed by the mapping built with
`fused_moe_make_expert_params_mapping(...)` →
`RoutedExperts.build_expert_params_mapping(...)` (`routed_experts.py:1027-1130`),
which resolves each checkpoint `experts.{id}.gate_proj...` into
`experts.w13_weight` / `experts.w2_weight` with `shard_id ∈ {w1, w3, w2}`.

Two consequences were identified for the EXL3 MoE adapter, both **closed by
P1-02** (`exl3xpu/moe.py`):

1. **Parameter names are not free — but the per-expert prefix is.** The
   mapping's per-expert entries carry `param_name = "experts.w13_"` /
   `"experts.w2_"` (not `w13_weight`); the stock loader substitutes that
   prefix into the checkpoint name, so `experts.{id}.gate_proj.trellis`
   becomes `experts.w13_trellis`. Registering `w13_trellis` / `w13_suh` /
   `w13_svh` / `w13_mcg` (and `w2_*`) therefore resolves through the real
   mapping. Verified against the installed vLLM: experts 0 and 255 produce
   24/24 `weight_loader` calls with the expected `(shard_id, expert_id)`.
2. **The checkpoint does not use `.weight` for quantized experts.** This is
   handled by the same prefix substitution: the suffix after the projection
   (`trellis|suh|svh|mcg`) is carried through verbatim.

The remaining trap is the stock loader's fused-tensor heuristic:
`RoutedExperts.load_weights` (`routed_experts.py:918`) sets
`is_fused = loaded_weight.dim() == 3` and then chunks `w1`/`w3` along dim 1.
A per-expert EXL3 trellis is also 3-D (`[k/16, n/16, 16K]`), so the stock path
silently splits one expert into 64 pseudo-experts. `create_weights` therefore
binds a per-layer loader (`moe.py::Exl3MoEMethod._layer_load_weights`) that:

- parses `<expert>.<projection>.<suffix>` relative to the module (the form
  `AutoWeightsLoader` passes after stripping the child prefix);
- maps `gate_proj→w1`, `up_proj→w3`, `down_proj→w2`, then selects
  `w13_*`/`w2_*` and slot `0/1`;
- validates expert range, per-slot shape, and exact dtype before
  `param.data[expert, slot].copy_(...)`;
- raises `Exl3MoEUnsupported` for any unexpected name/projection/suffix or a
  shard/parameter mismatch — nothing is silently ignored.

`process_weights_after_loading` then requires every `(param, expert, slot)`
slice to have been loaded (3,072 slices for 256 experts) and verifies that the
per-expert codebook scalar equals the configured id. `--load-format dummy`
never calls `load_weights`, so the completeness contract is skipped there
explicitly rather than passing an empty set off as complete.

### 3.3 Shard-by-shard streaming (P1-05)

`moe.py::stream_expert_shards(shard_paths, layers)` drives the whole model
load without holding all shards open. It:

- opens exactly one shard with `safe_open` (mmap) and iterates its keys;
- classifies routed-expert keys with `split_expert_key` (→ normalized
  `layers.{n}.mlp.experts` + `<expert>.<proj>.<suffix>`), and reports routed
  keys whose layer is absent as `out_of_scope` (not "unaccounted");
- reads each tensor lazily via `get_tensor` inside a generator, so besides the
  packed arenas only one checkpoint tensor's payload is resident;
- closes the mmap on `with`-exit, then `gc.collect()` + `torch.xpu.empty_cache()`
  before the next shard, which is what keeps host/device duplication down on
  the shared-memory Arc part.

On the pinned 3.09bpw shards (Arc 140V) this packs **122,880/122,880** routed
tensors into 40 layers (**11,673 MiB** of arenas) with `missing=0`,
`unaccounted=0`, `out_of_scope=0`. Host RSS stays bounded at **1.25–1.26 GiB**
across both shards (independent of the 7.9 / 6.3 GiB shard sizes) while
`xpu_allocated` stays flat at **11,673 MiB** — no transient duplication beyond
the packed arenas; XPU free ends at **3.87 GiB** after settle.

Evidence: `tests/test_expert_loader.py` (15/15 cases),
`logs/moe-preflight/test-expert-loader.{log,json}`; streaming
`logs/moe-p1/shard-loading-memory.csv`, `logs/moe-p1/shard-streaming-audit.json`
(`scripts/test_shard_streaming.py`).

### 3.4 Bit-exact reconstruction (P1-06)

The C++ dequantizer `exl3xpu_C::exl3_reconstruct(trellis, w, n0, K, cb)` is
bit-identical to the CPU oracle `exl3xpu.ref.reconstruct_inner` on real
checkpoint tensors: for experts `{0,1,127,254,255}` × `{gate,up,down}`
(15 matrices, both `[k/16,n/16,16K]` geometries, `K=3`, `cb=1`) the
reconstruction has **0 differing fp16 elements** (`max_abs=0.0`). The `n0`
column-slice path is exact too, and the reconstructed weights reproduce the
P0-08 trace linear outputs with `max_abs=0.0`. Evidence:
`tests/test_expert_reconstruction.py`,
`logs/moe-p1/reconstruction-audit.{log,json}`.

### 3.5 Single-expert MLP (P1-07)

The same per-linear op the MoE will use — `ops.exl3_linear` with
`shard_of_nb=zeros(n/128)` and `group_bounds=[0, n]` — composes into the routed
MLP (`y = down(silu(gate(x)) * up(x))`). Against the fp32 dequantized
reference MLP on real layer-0 checkpoint tensors, for expert 0 and
`M ∈ {1,2,4,8}` in fp16 and bf16:

| tensors | max_abs | T-2 (0.08) |
| --- | --- | --- |
| `down` output | 2.9e-6 … 3.6e-6 | pass |
| `gate` / `up` intermediates | 3.1e-5 … 2.5e-4 | pass |

Reruns are bit-identical (T-5), there are no NaN/Inf (T-6), and the P0-08 trace
replay for experts 0/255 matches `expert/{id}/down_proj/y` at `max_abs=2.9e-6`.
Evidence: `tests/test_expert_mlp.py`,
`logs/moe-p1/single-expert-eval.{log,json}`.

### 3.6 Batch sweep and empty work (P1-08)

`tests/test_batch_sweep.py` sweeps `M ∈ {0,1,2,4,8,16,32,64,128,256}`:

- **M=0 launches zero XPU kernels** (profiler-counted) and returns
  `[0, 2048]`; the C++ `exl3_linear` returns at `if (M == 0)` before touching
  any kernel.
- M=1…256 match the reference MLP with `max_abs ≤ 4.3e-6` (T-2).
- Crossover (gate, 2048→512), fused vector vs reconstruct+oneDNN fp16 GEMM:
  fused is 1.49×/1.41×/1.24× faster at M=1/2/4 and break-even at M=8;
  M>8 only has reconstruct+GEMM. This justifies `small_m_max(3,1)=8`.

**Production contract — `suh` is 2-D `[num_groups, k]`.** The C++ takes
`S = suh.size(0)` and computes one input Hadamard per group. Passing a 1-D
`suh` silently sets `S=k` and runs k redundant transforms (correct but
64 MiB at M=8, 2 GiB at M=256). The dense plugin stores `suh` as
`[num_groups, k]`; the MoE `apply` must pass `suh.unsqueeze(0)` per expert
projection. Evidence: `logs/moe-p1/batch-sweep.{log,json}`,
`logs/moe-p1/crossover-calibration.json`.

### 3.7 Reconstruction workspace and reentrancy (P1-09)

At `RECON_SLICE_N=1024` the large-M path reconstructs one `k×1024` fp16 slice
at a time (≤ 4 MiB at k=2048). It previously used a `static` per-device
`weight_scratch` tensor, which is shared by all streams and therefore unsafe
under concurrency (vLLM's future async scheduling). The slice is now allocated
per call (`at::empty({k, n1-n0})`); the caching allocator is stream-ordered, so
in-flight calls get distinct buffers and the block is reused only on the same
stream.

Audited facts (`tests/test_reentrancy.py`):

| M | kernels | peak delta | persistent |
| --- | --- | --- | --- |
| 8 | `HadIn` + `Gemv<3,1,8,4>` + `HadOut` (no Reconstruct/GEMM) | 2.04 MiB (GEMV `part`) | 0.00 MiB |
| 64 | `HadIn` + `Reconstruct`×`ceil(n/1024)` + `gemm` + `HadOut` | 2.38 / 1.56 MiB | 0.00 MiB |

Two XPU streams × 100 interleaved large/small trials are bit-identical to
serial. Evidence: `logs/moe-p1/reentrancy-audit.{log,json}`,
`logs/moe-p1/reentrancy-memory.csv`, build
`logs/moe-preflight/esimd-build-reentrancy-r4/`.

## 4. Shared-expert ownership on XPU (no AITER)

`Qwen3NextSparseMoeBlock.__init__` (`qwen3_next.py:147-238`) decides the
shared-expert execution mode:

```python
is_fused_shared_expert_enabled = resolve_layer_fused_shared_expert(...)   # :151
replicate_shared_expert = _should_replicate_misaligned_shared_expert(...) # :163
...
self.shared_expert = Qwen3NextMLP(..., expert_gate=self.shared_expert_gate,
                                  disable_tp=self.replicate_shared_expert) # :206
self.experts = FusedMoEFactory(shared_experts=None if replicate else self.shared_expert,
                               n_shared_experts=1 if self.shared_expert is None else None,
                               fuse_shared_experts=is_fused_shared_expert_enabled, ...)
```

On XPU:

- `resolve_layer_fused_shared_expert` is false: AITER is unavailable, and
  `rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()` returns `None`
  (`fused_moe/utils.py:74-93`).
- `get_quark_ocp_mx_group_size` returns `None` for a non-Quark config
  (`quantization/utils/config_utils.py:15-27`), so
  `_should_replicate_misaligned_shared_expert` returns `False`
  (`qwen3_next.py:99-128`).

Therefore the shared expert is a **dense `Qwen3NextMLP` attached to
`MoERunner`** (`shared_experts=self.shared_expert`, `n_shared_experts=None`,
`fuse_shared_experts=False`), executed inside `moe_forward_shared`. It is
*not* a 257th expert.

Implications:

- The shared expert is quantized in this checkpoint (4-bit MCG, three
  projections per layer) and is served by the existing dense
  `Exl3LinearMethod` — the MoE adapter does not need to touch it.
- Because it lives inside the custom op `moe_forward_shared`, the dense EXL3
  linear op must remain graph/custom-op safe (`torch.ops.exl3xpu::linear` is
  already registered as an opaque custom op).
- `self.shared_expert_gate` is a `ReplicatedLinear` (F16, `[1, 2048]`) applied
  as `sigmoid(gate · x)` inside `Qwen3NextMLP` via `expert_gate=`; no changes
  needed.

## 5. What `Exl3MoEMethod` must do (design contract derived from the audit)

1. `create_weights` — **implemented (P1-02)**
   - routed experts: per-expert/per-projection `trellis` (I16
     `[E, 2, hidden/16, intermediate/16, 16*K]` for `w13`,
     `[E, 1, intermediate/16, hidden/16, 16*K]` for `w2` — i.e. always
     `[k/16, n/16, 16*K]`, not transposed), `suh` (F16), `svh` (F16), `mcg`
     scalar. Gate and up keep **distinct** `suh` vectors
     (`docs/checkpoint_schema.md` §7.1).
   - no fp16 weight buffer for the 30,720 routed modules; measured
     291.8 MiB/layer, eight allocations, largest 192 MiB (< 4 GiB cap).
2. Weight loading — **implemented (P1-02, streaming P1-05)**
   - checkpoint `experts.{id}.{gate,up,down}.{trellis,suh,svh,mcg}` map into
     the parameters above through the per-layer loader described in §3.2;
     bit-exact for real checkpoint tensors of experts 0/255.
   - `stream_expert_shards` loads the full checkpoint one shard at a time and
     flushes the mmap/allocator between shards (§3.3): 122,880/122,880 tensors,
     40 layers, `missing=0`, `unaccounted=0`.
2b. Packed descriptors — **implemented (P1-03)**
   - `layer.exl3_moe_descriptors` is an int64 `[E, 3, 14]` buffer with
     `(allocation index, element offset)` for trellis/suh/svh/codebook plus
     `k`, `n`, `K`, `cb`, `family`, `slot`; `expert_views(layer, e)` returns
     the actual parameter slices. The per-expert `.mcg` scalar stores the
     codebook *multiplier* (`0xCBAC1FED` for `mcg`), which
     `process_weights_after_loading` verifies against the config.
2c. Packed-arena integrity — **implemented (P1-04)**
   - `verify_expert_integrity(layer)` runs on every settle path (including
     `--load-format dummy`, since it is structural) and asserts: distinct
     allocation per tensor kind; distinct storage per `(expert, slot)` slice
     (so `suh`/`svh` scales + Hadamard vectors are never shared between
     experts or gate/up); registered shapes equal the tiled plan; descriptor
     offsets tile each arena exactly (`(expert*slots+slot)*per_expert`, no
     dense-row concatenation); trellis geometry `[k/16, n/16, 16K]` for both
     families (w2 is **not** transposed); `suh`/`svh` lengths equal `k`/`n`;
     codebook multipliers match the config. Returns
     `{experts: 256, projections: 3, allocations: 8, tiled_slices: 3072, ...}`.
3. `apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)` — **implemented (P2-01)**
   - consumes the runner's precomputed `topk_ids`/`topk_weights` (no
     renormalization); loops over active experts, gathers token rows, runs the
     verified per-expert MLP (`gate`/`up` → SiLU·up → `down`, with
     `suh.unsqueeze(0)`), scales the **down-projection output** by the routing
     weight (never the input), and `index_add_`s into an fp32 accumulator;
   - zero-token experts are skipped; M=0 returns `[0, D]` with no kernels;
   - the runner owns the shared expert (`_maybe_combine` adds
     `shared + routed`), so `apply` ignores `shared_experts`;
   - `enable_debug_capture()` records per-expert `gate`/`up`/`act`/`down` and
     the routed accumulator for P2-08 (zero overhead when disabled).
4. Fail loudly (`RuntimeError`) if `quant_config` is EXL3 but the method is not
   installed, to close the silent-fallback hole. — **done in P1-01**
   (`Exl3MoEUnsupported`).
5. `apply_monolithic` is **not** needed; the runner routes the modular path
   when `is_monolithic` is false.

## 6. Risks confirmed by this audit

| id | risk | status |
| --- | --- | --- |
| R-04 | vLLM lacks an extensible MoE quant interface | **Mitigated** — `FusedMoEMethodBase` is sufficient; no core patch needed for dispatch. Weight-name mapping still needs an adapter. |
| — | Silent `UnquantizedFusedMoEMethod` fallback | **Confirmed real** (`routed_experts.py:219-220`); must be actively blocked. |
| — | Gate/up fused-member `suh` assumed equal | **False on this checkpoint** (0/120 identical); fused stacking is legal only with per-member `suh`. |
| — | Shared expert treated as a 257th expert | **False** on XPU; it is a runner-attached dense MLP. |
| — | MTP head | config declares `mtp_num_hidden_layers: 1` but the checkpoint ships **no** MTP tensors; the loader must tolerate their absence. |

## 7. Verification commands

```bash
# class/line references used above
.venv/bin/python - <<'PY'
import inspect
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
print(inspect.getsourcefile(RoutedExperts))
print(inspect.getsourcefile(FusedMoEMethodBase))
PY

# confirm the fallback class actually gets selected today
.venv/bin/python - <<'PY'
from vllm.model_executor.layers.fused_moe.routed_experts import UnquantizedFusedMoEMethod
print(UnquantizedFusedMoEMethod.__name__)
PY
```
