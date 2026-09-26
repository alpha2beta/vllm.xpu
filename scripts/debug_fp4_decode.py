#!/usr/bin/env python3
"""Debug helper: is our FP4 decode / MX scale handling the same as
vllm_xpu_kernels' own dequant_mxfp4? Used to diagnose the fp4_gemm
reference mismatch in logs/mxfp4-kernel-check.txt."""
import torch
import vllm_xpu_kernels  # noqa: F401  (registers torch.ops._C.*)
import vllm._xpu_ops  # noqa: F401  (registers torch.ops.vllm.*)
from vllm_xpu_kernels.moe_utils import dequant_mxfp4

torch.manual_seed(0)
M, K = 4, 128
x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)

q, s = torch.ops.vllm.xpu_mxfp4_quantize(x)
print("q", q.shape, q.dtype, "s", s.shape, s.dtype)

theirs = dequant_mxfp4(q, s)          # reference from the kernel package
print("theirs mean abs", theirs.float().abs().mean().item())

# our decoder (same code as scripts/mxfp4_kernel_check.py)
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "mkc", pathlib.Path(__file__).parent / "mxfp4_kernel_check.py")
mkc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mkc)

ours = mkc._dequant_mx(q, s)
print("ours   mean abs", ours.abs().mean().item())

d = (ours - theirs.float()).abs().max().item()
print("max |ours - theirs| =", d)

# how close is dequantized x to the original?
orig = x.float()
print("max |theirs - orig| / max|orig| =",
      (theirs.float() - orig).abs().max().item() / orig.abs().max().item())
print("max |ours   - orig| / max|orig| =",
      (ours - orig).abs().max().item() / orig.abs().max().item())

# inspect a single block: raw scale bytes, decoded scale, element values
row = 0
blk = 0
s_bytes = s.view(torch.uint8)[row, blk] if s.dtype != torch.uint8 else s[row, blk]
print("scale byte =", int(s_bytes), "-> 2^(e-127) =", 2.0 ** (int(s_bytes) - 127))
print("orig block:", orig[row, blk * 32:blk * 32 + 8].tolist())
print("theirs blk:", theirs.float()[row, blk * 32:blk * 32 + 8].tolist())
print("ours   blk:", ours[row, blk * 32:blk * 32 + 8].tolist())
print("absmax orig block:", orig[row, blk * 32:(blk + 1) * 32].abs().max().item())
