# Plan: vLLM XPU Runtime for Qwen3.6-35B-A3B (MXFP4) on Intel Core Ultra 7 258V

## 0. Reality Check Before Starting

- **Hardware**: Core Ultra 7 258V = 8-core Xe2 Arc 140V iGPU + 32 GB on-package
  LPDDR5x-8533, unified/shared across CPU, GPU, and NPU. No dedicated VRAM.
- **Model footprint**: 35B total params (MoE, ~3B active) at MXFP4 (~4.2 bit/weight)
  ≈ 18–19 GB just for weights. Leaves roughly 10–13 GB for OS, driver overhead,
  and KV cache — workable, but long-context (262K) is not realistic on this box.
- **vLLM's XPU backend and `vllm-xpu-kernels` are primarily validated against
  Arc Pro / Flex / Max (datacenter/workstation) GPUs**, not thin-laptop iGPUs.
  Expect rough edges: driver quirks, missing kernel paths, possible fallback to
  slower ops. If vLLM proves unworkable, llama.cpp (SYCL backend) or IPEX-LLM
  are the fallback options for this specific hardware.
- Treat this as an experimental/best-effort setup, not a production deployment.

## 1. OS & Driver Prerequisites

- [ ] Confirm OS: Linux (Ubuntu 24.04+ recommended) — vLLM XPU support is
      Linux-first; Windows/WSL2 support is far less mature.
- [ ] Install/verify the Intel GPU kernel driver stack (`xe` in-kernel driver
      for Lunar Lake, or latest Intel compute runtime).
- [ ] Install Level Zero runtime and `intel-opencl-icd` / compute-runtime
      packages so the GPU is visible to oneAPI tools.
- [ ] Verify GPU visibility:
      ```
      clinfo | grep -i "arc"
      ls /dev/dri
      ```
- [ ] Add your user to the `render` and `video` groups (needed for `/dev/dri`
      access without root).

## 2. Install Intel oneAPI Base Toolkit

- [ ] Download and install oneAPI Base Toolkit (version matching your target
      `vllm-xpu-kernels` release — check current requirement, historically
      2025.3–2026.0).
- [ ] Source the environment each session (or add to shell profile):
      ```
      source /opt/intel/oneapi/setvars.sh
      ```

## 3. Python Environment

- [ ] Use Python 3.9–3.12 (per `vllm-xpu-kernels` requirements).
- [ ] Create an isolated environment:
      ```
      python -m venv .venv
      source .venv/bin/activate
      ```

## 4. Install PyTorch XPU Build

- [ ] Install the XPU-enabled PyTorch wheel matching the oneAPI version:
      ```
      pip install torch --index-url https://download.pytorch.org/whl/xpu
      ```
- [ ] Sanity check:
      ```
      python -c "import torch; print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0))"
      ```

## 5. Install vLLM + vllm-xpu-kernels

- [ ] Install latest vLLM built for XPU (pulls `vllm-xpu-kernels` automatically
      as a wheel dependency) — or build from source using vLLM's
      `Dockerfile.xpu` / XPU install instructions if a prebuilt wheel isn't
      current enough for MXFP4 support.
- [ ] Confirm the kernel package registers correctly on import:
      ```
      python -c "import vllm_xpu_kernels; print('ok')"
      ```
- [ ] Verify MXFP4 quantization kernels are present in this build (check
      release notes / changelog for `vllm-xpu-kernels`, since MXFP4 support on
      XPU is relatively new and version-dependent).

## 6. Obtain the Model Weights

- [ ] Pull the MXFP4 weights (from Hugging Face or wherever Qwen3.6-35B-A3B
      MXFP4 is hosted) — confirm the specific repo publishes a native MXFP4
      checkpoint, not just FP8/INT4 alternatives.
- [ ] Verify local disk space: weights ≈ 18–19 GB, so budget 25+ GB free to
      allow for tokenizer/config files and any temp conversion artifacts.

## 7. Launch vLLM

- [ ] Start with conservative settings tuned to the 32 GB shared-memory
      budget — small `max-model-len`, small `max-num-seqs`, eager mode first
      (skip CUDA-graph-style compilation) to isolate correctness issues from
      performance issues:
      ```
      python -m vllm.entrypoints.openai.api_server \
        --model /path/to/qwen3.6-35b-a3b-mxfp4 \
        --quantization mxfp4 \
        --dtype float16 \
        --gpu-memory-utilization 0.80 \
        --max-model-len 4096 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 4 \
        --block-size 64 \
        --enforce-eager \
        --trust-remote-code
      ```
- [ ] If it fails to launch, first suspect: (a) quantization flag/name
      mismatch for MXFP4, (b) insufficient shared memory after OS overhead,
      (c) missing MXFP4 kernel support in the installed `vllm-xpu-kernels`
      version.

## 8. Validate & Tune

- [ ] Confirm generation correctness on a few prompts before trusting output.
- [ ] Monitor memory pressure (`intel_gpu_top` or similar) — watch for OOM or
      swapping, since this is unified memory shared with the OS.
- [ ] Gradually raise `max-model-len` / `max-num-seqs` until you find the
      ceiling before hitting OOM or unacceptable slowdown.
- [ ] Benchmark tokens/sec to decide if throughput is acceptable for your use
      case — Arc 140V is an efficiency-class iGPU, not a throughput part.

## 9. Fallback Plan

- [ ] If vLLM XPU proves unstable or MXFP4 support is incomplete for this
      hardware/version combo, fall back to:
      - **llama.cpp** with SYCL backend (broader, more mature Intel iGPU
        support, GGUF quantization formats), or
      - **IPEX-LLM**'s vLLM integration (separate patched vLLM branch, more
        laptop-iGPU-tested, but a narrower supported model list).

## Open Questions to Confirm Before Starting

- Exact `--quantization` flag name vLLM expects for MXFP4 in your installed
  version (naming may differ from `mxfp4`).
- Whether the specific Qwen3.6-35B-A3B MXFP4 checkpoint you plan to use has
  been tested with vLLM's loader, or needs a conversion step.
- Current `vllm-xpu-kernels` release notes for MXFP4-on-XPU maturity — this
  is a fast-moving area and worth rechecking immediately before setup.
