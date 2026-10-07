# EXL3 source audit — pinned `15ded2f3add148c4db3c900cba7de878238f53bc` (2026-09-26)

## Reachability in vLLM serving (`vllm_plugin.py` calls only these)

| Op | Caller | M routing (baseline `EXL3_SMALL_M_MAX=8`) |
|---|---|---|
| `linear` (whole C++ op) | `Exl3LinearMethod.apply` when `exl3_supported(K,cb)` | M<=8 → `exl3_gemm_small`; M>8 → had_in + reconstruct slices + `at::matmul` + had_out |
| `exl3_gemm_small` | python fallback `exl3_linear_impl` (M<=SMALL_M_MAX) | vec if M<=`g_vec_max_m` else **DPAS** → guarded by patch |
| `exl3_had_in_rm` / `exl3_had_out_h` / `exl3_reconstruct` | slice path (M>8) | ESIMD vector kernels, no DPAS |
| `exl3_supported` | backend selection | K=4/6 + cb=2 default; no flag needed for this checkpoint |

## NOT reachable from the plugin (tests/bench only)

- `exl3_gemm_raw` — only `tests/test_bitexact_xpu.py` calls it (path!=0 → DPAS).
- `exl3_fa_fwd` / oneDNN `exl3_sdpa` — FA in `fa_esimd.h` uses `xmx/dpas.hpp`;
  vLLM attention uses vllm-xpu-kernels, never this op. oneDNN headers are absent
  here, so `EXL3_DNNL` compiles out anyway.
- `fused_small` — only when `g_fused=1` (default off, no plugin caller sets it).
  Fused vec branch instantiates only MR=1/2 kernels: an MR=2 kernel unrolls
  exactly m=0,1, so M=3..8 would silently drop rows. Baseline keeps fused OFF
  and the patch makes `fused_small` decline M>2 back to the unfused path.
- INT8 prefill (`g_int8`) — default off; baseline `EXL3_INT8_PREFILL=0`.

## Unfused vector correctness (M<=8, `g_vec_max_m=8`)

`dispatch_mr` instantiates MR=1/2/4/8; MR selector is
`M<=1?1 : M<=2?2 : M<=4?4 : 8`, and `GemvKernel` guards tails
(`if (m < M)` on load, `if (m >= M) break` on store). MR=8 covers M=5..8
exactly; MR=4 covers M=3,4. No row tiling needed at/below 8 rows.

## vLLM string patches (`vllm_patches.py`)

`apply_all` = align_sync + gdn_mask_index + fp8kv_prefill + xpu_block_size.
Each checks expected source and skips on mismatch, but all target speculative/
prefill internals. Baseline runs with `EXL3_VLLM_PATCHES=0` (never imported).

## Build script notes (`scripts/build_ext.sh` at this SHA)

Uses `python3` (not `$PYTHON`), hard-codes `-D_GLIBCXX_USE_CXX11_ABI=1`,
masks failures with `grep -E "error" || true`. E2 uses a fixed local builder.
DNNL auto-skips (no `/opt/intel/oneapi/dnnl` tree); baseline also exports
`EXL3_NO_DNNL=1` explicitly.
