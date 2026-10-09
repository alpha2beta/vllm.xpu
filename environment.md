# Environment

All values below were captured on this host; raw output is in `logs/host-check.txt`
(regenerate with `scripts/host_check.sh`). Software-stack versions are appended after
the install phase completes.

## Hardware

| Item | Value |
|---|---|
| Machine | MSI **Claw 8 AI+ A2VM** (board MS-1T52), handheld |
| CPU | Intel® Core™ Ultra 7 258V (Lunar Lake), 8 cores / 8 threads, 400–4800 MHz, stepping 1, microcode 0x128 |
| GPU | Intel® Arc™ Graphics 130V/140V (PCI `8086:64A0`, subsystem `1462:146C`), Xe2-LPG iGPU |
| Memory | 30 GiB reported (`free -h`), 32 GB on-package LPDDR5x-8533, **shared** by CPU/GPU |
| Swap | 30.9 GB **zram0** (`zram0 → [SWAP]`) — any host-memory overflow compresses into zram, so "swap activity" must be watched explicitly |
| Storage | 473 GB NVMe `nvme0n1p5` mounted at `/`; **131 GB free** before the model download |
| BIOS | E1T52IMS.112, 2025-12-04 |

## Operating system

| Item | Value |
|---|---|
| Distro | CachyOS (Arch-based, rolling) — `BUILD_ID=rolling` |
| Kernel | `7.2.9-1-cachyos-deckify` (verified 2026-10-09; was `7.2.3-3` at setup) |
| Python (system) | 3.14.7 (not used); **3.12.14** at `~/.local/bin/python3.12` (used for `.venv`) |
| Tooling | `uv 0.12.12`, `hf` / `huggingface-cli` available |

Deviation from `plan.md`: no fresh Ubuntu 24.04 install; the existing CachyOS install is used
as-is. Native Linux only — WSL2 explicitly out of scope.

## GPU driver & runtime stack (system packages)

| Component | Version | State |
|---|---|---|
| Kernel driver | `xe` (module Live; `DRIVER=xe`) — not `i915` | correct driver for Lunar Lake / Xe2 |
| Device nodes | `/dev/dri/card0` (root:video 660), `/dev/dri/renderD128` (**666**) | non-root access OK |
| Level Zero loader | `level-zero-loader 1.32.0-1` | installed |
| Compute runtime | `intel-compute-runtime 26.35.39758.10-1.1` | ≥ the documented 26.18 minimum |
| Graphics compiler | `intel-graphics-compiler 1:2.41.5-1.1` | installed |
| OpenCL ICD | `ocl-icd 2.3.5-1.1` | installed |
| oneAPI (system) | `intel-oneapi-* 2026.0.0` incl. `dpcpp-cpp`, `mkl`, `tbb`, `umf` at `/opt/intel/oneapi` | matches `vllm-xpu-kernels` requirement "oneAPI 2026.0" |
| GPU monitor | `intel-gpu-tools` **not installed** (needs an interactive `sudo`) | gap — see tasks.md Phase 2 |

### GPU discovery evidence (non-root)

```
$ sycl-ls
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2,
                               Intel(R) Arc(TM) Graphics 20.4.4 [1.17.39758]
[opencl:cpu][opencl:0]         Intel(R) OpenCL, Intel(R) Core(TM) Ultra 7 258V OpenCL 3.0
[opencl:gpu][opencl:1]         rusticl, Mesa Intel(R) Graphics (LNL) OpenCL 3.1 [26.2.3-arch3.1]
[opencl:gpu][opencl:2]         Intel(R) OpenCL Graphics (integrated),
                               Intel(R) Arc(TM) Graphics OpenCL 3.0 NEO [26.35.39758]

$ clinfo | grep "Device Name"
  Mesa Intel(R) Graphics (LNL)          (rusticl — not used by vLLM)
  Intel(R) Arc(TM) Graphics             (Intel NEO compute runtime — used by Level Zero/ZES)
```

Note: a `rusticl` (Mesa) OpenCL platform is also present. The SYCL/Level-Zero path used by
PyTorch-XPU and `vllm-xpu-kernels` resolves to the Level-Zero device `[1.17.39758]`, i.e.
compute-runtime 26.35.

**Gate 2: PASS** — device visible to the OS, Level Zero and the SYCL runtime as the unprivileged
user, no permission errors.

---

## Software stack (installed in `.venv`, Python 3.12.14)

Captured from `logs/pip-freeze.txt` (192 packages) after the 0.31.0 upgrade (2026-10-09).

| Component | Installed version | Source / note |
|---|---|---|
| Python | 3.12.14 | `~/.local/bin/python3.12 -m venv .venv` |
| vLLM | **`0.31.0+xpu`** | `https://wheels.vllm.ai/0.31.0/xpu` (33.2 MB wheel) |
| `vllm-xpu-kernels` | **0.1.15.4** | pinned by `vllm==0.31.0+xpu` (`vllm_xpu_kernels==0.1.15.4`) |
| PyTorch | **`2.14.0+xpu`** | `https://download.pytorch.org/whl/xpu` (vLLM 0.31.0 pins `torch==2.14.0`) |
| torchvision / torchaudio | `0.29.1+xpu` / `2.11.0+xpu` | same index (torchvision follows torch; torchaudio has no newer +xpu build) |
| Triton | `3.8.0+xpu` shim + `triton-xpu 3.8.0` | `https://wheels.vllm.ai/xpu` (pinned by vLLM 0.31.0: `triton==3.8.0+xpu`) |
| compressed-tensors | 0.17.0 | pinned by vLLM 0.30.0 — the version that registers `mxfp4-pack-quantized` |
| transformers | 5.17.0 | ≥ 5.10.4 required for Qwen3.5/3.6 architectures |
| oneAPI (pip runtime) | `intel-sycl-rt`, `intel-cmplr-lib-rt`, `dpcpp-cpp-rt`, `intel-opencl-rt`, `oneccl`, `onemkl-sycl-*`, `mkl`, `intel-openmp` — `2026.1.x` (were `2026.0.0` under vLLM 0.30.0) | pulled by the torch 2.14.0 XPU build |
| oneAPI (system) | 2026.0.0 at `/opt/intel/oneapi` (`setvars.sh` sourced for runs) | pacman |
| Level Zero driver | `1.17.39758` (as reported by `torch.xpu` properties) | via compute-runtime 26.35 |
| Device | `Intel(R) Arc(TM) Graphics`, `device_id=0x64A0`, **total_memory 29270 MB (28.58 GiB)**, `gpu_eu_count=64`, `gpu_subslice_count=8`, `is_integrated_gpu=1` | `scripts/torch_sanity.py` → `logs/torch-sanity.txt` |

Gate 3 evidence (`logs/torch-sanity.txt`): `xpu available: True`, BF16 matmul finite,
arange-sum and host↔device round-trip correct → **PASS**.

Gate 4 evidence (`logs/mxfp4-kernel-check.txt`): `torch.ops.vllm.xpu_mxfp4_quantize`,
`torch.ops._xpu_C.fp4_gemm`, `torch.ops._xpu_C.is_xe2_arch` all registered;
`is_xe2_arch() == True`; `fp4_gemm` result matches an independently dequantized
float32 reference to **rel. err 1.17e-3** (BF16 output rounding); `XPUExpertsMxFp4`
reports `_supports_current_device()` and accepts the checkpoint's
(static mxfp4 weight, dynamic mxfp4 activation) scheme → **PASS**.

Install command (reproducible):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
uv pip install "vllm==0.31.0+xpu" \
  --extra-index-url https://wheels.vllm.ai/0.31.0/xpu \
  --extra-index-url https://wheels.vllm.ai/xpu \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  --index-strategy unsafe-best-match
```

> The explicit `+xpu` local-version pin is **required**: with plain `vllm==0.31.0`, uv/pip
> resolve the CUDA wheel from PyPI instead of the 33.2 MB XPU wheel.
