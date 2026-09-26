#!/usr/bin/env python3
"""Phase 4 gate check: prove that an MXFP4 operation is registered AND executes
correctly on the XPU, rather than merely confirming that packages import.

Run with the experiment venv active:

    source .venv/bin/activate
    python scripts/mxfp4_kernel_check.py

Exit code 0 == all checks passed. Every check prints PASS/FAIL and the log is
meant to be captured to logs/mxfp4-kernel-check.txt.
"""
from __future__ import annotations

import sys

import torch

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import torch

    print(f"torch={torch.__version__}")
    check("torch.xpu.is_available()", torch.xpu.is_available())
    if not torch.xpu.is_available():
        return 1
    print(f"device={torch.xpu.get_device_name(0)}")

    # ---------------------------------------------------------------- import
    import vllm

    print(f"vllm={vllm.__version__}")
    try:
        import vllm_xpu_kernels  # noqa: F401

        check("import vllm_xpu_kernels", True)
    except Exception as e:  # pragma: no cover
        check("import vllm_xpu_kernels", False, repr(e))
        return 1

    # The `torch.ops.vllm.*` XPU ops are registered lazily by
    # `vllm._xpu_ops.xpu_ops.register_ops_once()` at module import (the engine
    # does this via its model runner), so import it explicitly here.
    try:
        import vllm._xpu_ops  # noqa: F401

        check("import vllm._xpu_ops (registers torch.ops.vllm.* ops)", True)
    except Exception as e:
        check("import vllm._xpu_ops (registers torch.ops.vllm.* ops)", False, repr(e))

    # ------------------------------------------------------ op registration
    def op_exists(namespace: str, name: str) -> bool:
        try:
            ops = getattr(torch.ops, namespace)
            getattr(ops, name)
            return True
        except (AttributeError, RuntimeError):
            return False

    check(
        "op torch.ops.vllm.xpu_mxfp4_quantize registered",
        op_exists("vllm", "xpu_mxfp4_quantize"),
    )
    check("op torch.ops._xpu_C.fp4_gemm registered", op_exists("_xpu_C", "fp4_gemm"))
    check("op torch.ops._xpu_C.is_xe2_arch registered", op_exists("_xpu_C", "is_xe2_arch"))

    # ------------------------------------------------------- arch detection
    try:
        xe2 = bool(torch.ops._xpu_C.is_xe2_arch())
        check("is_xe2_arch() returns True (Arc 140V = Xe2)", xe2)
    except Exception as e:
        check("is_xe2_arch()", False, repr(e))

    # ------------------------------------------------------ numeric MXFP4 test
    torch.manual_seed(0)
    M, K, N = 64, 512, 384  # deliberately non-square to expose layout bugs
    x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device="xpu", dtype=torch.bfloat16)

    try:
        x_q, x_s = torch.ops.vllm.xpu_mxfp4_quantize(x)
        w_q, w_s = torch.ops.vllm.xpu_mxfp4_quantize(w_bf16)
        ok = (
            tuple(x_q.shape) == (M, K // 2)
            and _as_u8(x_q).dtype == torch.uint8
            and tuple(x_s.shape) == (M, K // 32)
            and tuple(w_q.shape) == (N, K // 2)
            and tuple(w_s.shape) == (N, K // 32)
        )
        check(
            "xpu_mxfp4_quantize shapes/packing",
            ok,
            f"x_q={tuple(x_q.shape)}/{x_q.dtype} x_s={tuple(x_s.shape)}/{x_s.dtype} "
            f"w_q={tuple(w_q.shape)}/{w_q.dtype} w_s={tuple(w_s.shape)}/{w_s.dtype}",
        )
    except Exception as e:
        check("xpu_mxfp4_quantize executes", False, repr(e))
        return 1

    # -------------------------------------------------- numeric FP4 GEMM test
    try:
        # Mirror XPUMxFp4LinearKernel.process_weights_after_loading(): the
        # weight parameter is replaced by `weight.view(float4).t()` (a
        # NON-contiguous view — materialising it with .contiguous() gives wrong
        # results, see scripts/debug_fp4_gemm.py), and the scale by
        # `weight_scale.t().contiguous()`.
        w_packed = w_q.view(torch.float4_e2m1fn_x2).t()  # [K, N], view
        w_scale_t = w_s.t().contiguous()  # [K/32, N], keep float8_e8m0fnu
        out = torch.ops._xpu_C.fp4_gemm(
            x_q, w_packed, x_s, w_scale_t, torch.bfloat16, None
        )
        check(
            "fp4_gemm executes",
            tuple(out.shape) == (M, N),
            f"out={tuple(out.shape)}/{out.dtype} (expected {(M, N)})",
        )
    except Exception as e:
        check("fp4_gemm executes", False, repr(e))
        out = None

    if out is not None:
        # Reference: dequantize both operands from the packed FP4 + E8M0 scales
        # and multiply in float32. Agreement within bf16 rounding shows the
        # kernel is really computing MXFP4 matmul, not a placeholder.
        ref = _reference_fp4_matmul(x_q, x_s, w_q, w_s)
        out_f = out.float()
        denom = ref.abs().mean().clamp(min=1e-6)
        rel = (out_f - ref).abs().mean() / denom
        check(
            "fp4_gemm matches dequant reference (rel err < 2e-2)",
            bool(rel < 2e-2),
            f"rel_err={float(rel):.5f} ref_mean_abs={float(ref.abs().mean()):.4f}",
        )

    # -------------------------------------- MXFP4 MoE path (the model's path)
    try:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
            XPUExpertsMxFp4,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp4Dynamic,
            kMxfp4Static,
        )

        check(
            "XPUExpertsMxFp4._supports_current_device()",
            XPUExpertsMxFp4._supports_current_device(),
        )
        check(
            "XPUExpertsMxFp4 accepts (mxfp4-static weight, dynamic act)",
            XPUExpertsMxFp4._supports_quant_scheme(kMxfp4Static, kMxfp4Dynamic),
        )
        check(
            "XPUExpertsMxFp4 accepts (mxfp4-static weight, no act quant)",
            XPUExpertsMxFp4._supports_quant_scheme(kMxfp4Static, None),
        )
    except Exception as e:
        check("XPUExpertsMxFp4 import/registration", False, repr(e))

    # ------------------------------------------------- basic XPU op coverage
    try:
        a = torch.randn(128, 128, device="xpu", dtype=torch.bfloat16)
        b = torch.randn(128, 128, device="xpu", dtype=torch.bfloat16)
        c = (a @ b).float()
        torch.xpu.synchronize()
        check("bf16 matmul on XPU", torch.isfinite(c).all().item())
    except Exception as e:
        check("bf16 matmul on XPU", False, repr(e))

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} check(s) failed): {FAILURES}")
        return 1
    print("RESULT: PASS — MXFP4 ops registered and numerically correct on XPU")
    return 0


# E2M1 (OCP FP4) decode table: sign bit + 3-bit magnitude code.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _as_u8(t: "torch.Tensor") -> "torch.Tensor":
    """View a packed 1-byte-per-element tensor (float4_e2m1fn_x2 / float8_e8m0fnu)
    as uint8 so bitwise decoding works."""
    if t.dtype == torch.uint8:
        return t
    try:
        return t.view(torch.uint8)
    except Exception:
        return t.contiguous().view(torch.uint8)


def _decode_fp4_packed(packed: "torch.Tensor") -> "torch.Tensor":
    """packed: [..., K/2] one-byte elements -> float32 [..., K]
    (low nibble = even index).

    OCP E2M1 layout: bit3 = sign, bits[2:0] = magnitude code, where the
    positive magnitude table is [0, 0.5, 1, 1.5, 2, 3, 4, 6].
    """
    packed = _as_u8(packed)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    table = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)

    def dec(nib: "torch.Tensor") -> "torch.Tensor":
        # cast to int64: uint8 indices are interpreted as boolean masks
        idx = (nib & 0x07).to(torch.int64)
        mag = table[idx]
        return torch.where(nib.to(torch.int64) >= 8, -mag, mag)

    vals_lo = dec(lo)
    vals_hi = dec(hi)
    out = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.float32,
        device=packed.device,
    )
    out[..., 0::2] = vals_lo
    out[..., 1::2] = vals_hi
    return out


def _dequant_mx(packed: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    vals = _decode_fp4_packed(packed)  # [rows, K]
    e = _as_u8(scale).to(torch.int32).to(torch.float32) - 127.0  # E8M0 -> 2^e
    scales = torch.repeat_interleave(e, group, dim=1)  # [rows, K]
    return vals * (2.0**scales)


def _reference_fp4_matmul(x_q, x_s, w_q, w_s) -> "torch.Tensor":
    x_hat = _dequant_mx(x_q, x_s).float()
    w_hat = _dequant_mx(w_q, w_s).float()
    return x_hat @ w_hat.t()


if __name__ == "__main__":
    sys.exit(main())
