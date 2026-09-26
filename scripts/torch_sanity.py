#!/usr/bin/env python3
"""Phase 3 gate: PyTorch detects the Arc 140V and completes a simple XPU operation.

    source .venv/bin/activate
    python scripts/torch_sanity.py | tee logs/torch-sanity.txt
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"torch: {torch.__version__}")
    ok = True

    available = torch.xpu.is_available()
    print(f"xpu available: {available}")
    if not available:
        print("RESULT: FAIL — torch.xpu.is_available() is False")
        return 1

    print(f"device: {torch.xpu.get_device_name(0)}")
    props = torch.xpu.get_device_properties(0)
    print(f"properties: {props}")
    total = getattr(props, "total_memory", None)
    if total:
        print(f"device total memory: {total / (1024**3):.2f} GiB")

    # minimal tensor allocation
    a = torch.arange(1024, device="xpu", dtype=torch.float32)
    s = int(a.sum().item())
    print(f"arange sum = {s} (expected {1023 * 1024 // 2})")
    ok &= s == 1023 * 1024 // 2

    # matrix multiplication
    x = torch.randn(256, 256, device="xpu", dtype=torch.bfloat16)
    y = torch.randn(256, 256, device="xpu", dtype=torch.bfloat16)
    z = x @ y
    torch.xpu.synchronize()
    finite = bool(torch.isfinite(z.float()).all().item())
    print(f"bf16 matmul finite: {finite}, shape {tuple(z.shape)}")
    ok &= finite

    # host<->device round trip
    h = [1.0, 2.0, 3.0]
    d = torch.tensor(h, device="xpu")
    back = d.cpu().tolist()
    print(f"host->device->host roundtrip: {back}")
    ok &= back == h

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
