#!/usr/bin/env python3
"""Determine the operand layout that makes torch.ops._xpu_C.fp4_gemm agree
with a dequantized float32 reference matmul on XPU.

Layouts tried for the weight [N, K] and scale [N, K/32] operands, mirroring
XPUMxFp4LinearKernel.process_weights_after_loading() and its variants.
"""
import importlib.util
import pathlib

import torch
import vllm_xpu_kernels  # noqa: F401
import vllm._xpu_ops  # noqa: F401

spec = importlib.util.spec_from_file_location(
    "mkc", pathlib.Path(__file__).parent / "mxfp4_kernel_check.py")
mkc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mkc)

torch.manual_seed(0)
M, K, N = 8, 512, 384
x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
w_bf16 = torch.randn(N, K, device="xpu", dtype=torch.bfloat16)

x_q, x_s = torch.ops.vllm.xpu_mxfp4_quantize(x)
w_q, w_s = torch.ops.vllm.xpu_mxfp4_quantize(w_bf16)

x_hat = mkc._dequant_mx(x_q, x_s)
w_hat = mkc._dequant_mx(w_q, w_s)           # [N, K]
ref = x_hat @ w_hat.t().float()             # [M, N]
print(f"ref shape {tuple(ref.shape)} ref mean|.| {ref.abs().mean().item():.4f}")

w_v4 = w_q.view(torch.float4_e2m1fn_x2)      # [N, K]

candidates = {
    "A: w.t() view,        s.t().contig": (w_v4.t(), w_s.t().contiguous()),
    "B: w.t().contig,      s.t().contig": (w_v4.t().contiguous(), w_s.t().contiguous()),
    "C: w as-is [N,K],     s as-is": (w_v4, w_s),
    "D: w.t() view,        s as-is": (w_v4.t(), w_s),
    "E: w.t().contig,      s.t() view": (w_v4.t().contiguous(), w_s.t()),
}

for name, (wt, st) in candidates.items():
    try:
        out = torch.ops._xpu_C.fp4_gemm(x_q, wt, x_s, st, torch.bfloat16, None)
        o = out.float()
        rel = (o - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6)
        mx = (o - ref).abs().max().item()
        print(f"{name}: out={tuple(out.shape)} rel={float(rel):.5f} maxabs={mx:.4f} "
              f"mean|out|={o.abs().mean().item():.4f}")
    except Exception as e:
        print(f"{name}: ERROR {e!r}")

# sanity: is out perhaps out = x_hat @ w_hat (no transpose) or w_hat @ x_hat.t()?
alt1 = x_hat @ w_hat.float()                # only makes sense if shapes align
print("shape check: x_hat", tuple(x_hat.shape), "w_hat", tuple(w_hat.shape))
