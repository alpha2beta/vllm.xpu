# GPTQ-Int4 Quantization on Intel Arc 140V (Lunar Lake) — Operating Guide

How to produce a **GPTQ-Int4 (symmetric, group 128) W4A16** checkpoint of the 35B MoE
model [Kwaipilot/KAT-Coder-V2.5-Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev) on this Intel Arc 140V host,
verify the resulting artifact, and serve it via vLLM with MTP speculative decoding.

Adapted from the Arc Pro B60 operating guide (`GPTQ_quantization_b60.md`) to the unified-memory
architecture of the **MSI Claw 8 AI+ A2VM (Intel Core Ultra 7 258V / Arc 140V)**.

---

## 0. TL;DR — The Commands That Work on This Box

```bash
cd /home/yanchun/Projects/vllm.xpu

# 0. Ensure no running vLLM servers hold RAM or the Arc 140V iGPU
pkill -f "vllm serve" || true

# 1. Run the preflight probe (checks XPU, memory, HF repo, tokenizer, meta layer, MTP block)
./.venv/bin/python scripts/preflight_probe_kat.py

# 2. Check inputs and storage headroom before starting
./scripts/quantize_kat_gptq.sh --check

# 3. Quantize (streams logs to terminal & logs/quantize-kat-gptq.log)
#    Calibration: 64 chunks x 1024 tokens = 65,536 tokens (proven-safe memory budget)
./scripts/quantize_kat_gptq.sh --calib-samples 64 --calib-length 1024

# 4. Verify the artifact
./.venv/bin/python - <<'PY'
import json
O = "models/KAT-Coder-V2.5-Dev-W4A16-GPTQ"
wm = json.load(open(f"{O}/model.safetensors.index.json"))["weight_map"]
packed = sum(1 for k in wm if k.endswith(".weight_packed"))
scales = sum(1 for k in wm if k.endswith(".weight_scale"))
mtp = sum(1 for k in wm if k.startswith("mtp."))
print(f"Index tensors: {len(wm)} | Packed linears: {packed} | Scales: {scales} | MTP: {mtp}")
PY

# 5. Serve the quantized model with MTP K=2 speculative decoding on port 8000
./launch_vllm_kat_gptq.sh
```

Expected artifact: **~20.5 GiB resident weights across 42 safetensors shards**, `config.json` carrying
`compressed-tensors` `pack-quantized` W4A16 configuration, 128k context support, and MTP K=2 speculative draft head.

---

## 1. Architectural Adaptation: Arc Pro B60 vs. Arc 140V

The B60 reference guide targeted a discrete GPU setup with Docker. The Arc 140V environment requires
adaptation across hardware, software stack, and model architecture:

| Dimension | Arc Pro B60 Reference | Arc 140V (This Host) | Why It Matters |
|---|---|---|---|
| **Host System** | Desktop i5-12400F, 45 GB RAM, 4 GB swap | Handheld MSI Claw 8 AI+, Core Ultra 7 258V | Portable Lunar Lake SoC |
| **GPU / Memory** | Arc Pro B60 (24 GB discrete VRAM) | Arc 140V (30 GiB unified LPDDR5x-8533) | CPU and GPU physically share the same memory bus |
| **Swap Subsystem** | 4 GB disk swap | 30.9 GB **zram0** (compressed in-RAM swap) | Memory overflow compresses in RAM; thrashing kills throughput |
| **Runtime Stack** | Docker (`vllm-quantizer:gptqmodel7.6.0`) | Native Python 3.12 (`.venv`), PyTorch 2.14.0+xpu | Docker not used; all operations native in `.venv` |
| **Source Model** | Dense `Qwen3.8-27B` (64 layers, 400 linears) | MoE `KAT-Coder-V2.5-Dev` (40 layers, 256 routed experts/layer) | MoE architecture requires expert-batched Hessian & quantization |
| **Quantization Format** | AutoGPTQ format (`.qweight`, `.scales`, `.qzeros`) | `compressed-tensors` (`.weight_packed`, `.weight_scale`, `.weight_shape`) | Matches vLLM's `CompressedTensorsWNA16MoEMethod` and `XPUwNa16LinearKernel` |
| **MTP Draft Block** | 15 BF16 draft tensors (`mtp-bf16.safetensors`) | 19 BF16 draft tensors (`model-00014-of-mtp.safetensors`) | Speculative decoding draft head (k=2) preserved unquantized |

---

## 2. The One Memory Budget on Unified Memory

On Intel Arc 140V, the "one memory budget" principle from the B60 guide is **physically literal**:
the GPU does not have separate VRAM; it shares the 32 GB on-package LPDDR5x memory directly with the CPU.
The Linux `xe` driver maps GPU memory allocations directly from system RAM.

Key realities on this host:
1. **Physical Budget:** 30.88 GiB total reported RAM. An idle system with desktop and services leaves **~24.5 GiB available**.
2. **zram0 Swap:** When allocations exceed physical RAM, the kernel compresses memory pages into `zram0`. If GPU buffers spill into compressed zram, memory bandwidth collapses and Level-Zero engine timeouts (`UR_RESULT_ERROR_DEVICE_LOST`) occur.
3. **Weight Streaming:** The 69.3 GB BF16 source model **cannot be loaded into RAM at once**. The quantizer streams **one decoder layer at a time** into XPU memory, computes Hessians, runs GPTQ, saves the packed layer shard, and immediately calls `gc.collect()` and `torch.xpu.empty_cache()`.
4. **Calibration Budget:** Just as discovered on B60, calibration activation buffers are the primary memory driver:
   - `128 x 2048` (262,144 tokens): Exceeds memory budget, forces zram compression, causes `DEVICE_LOST`.
   - `64 x 1024` (65,536 tokens): **Proven-safe operating point**. Resident activation tensors remain under 1.2 GiB, memory stays flat throughout all 40 layers.

---

## 3. Storage Headroom & Shard Streaming

The host root filesystem (`/dev/nvme0n1p5`) has **~75 GiB free**.
Storing all 13 unquantized BF16 shards (69.3 GB) simultaneously with the output quantized checkpoint (~20.5 GB)
would require ~90 GB and trigger an `ENOSPC` disk exhaustion.

The adapted workflow handles this safely:
- **Shared non-expert shard:** Shard 12 (`model-00012-of-00013.safetensors`, 4.56 GB) contains attention, router, and shared expert tensors for all 40 layers. It is downloaded once and kept staged.
- **Expert shards:** Shards 0 through 11 (5.00 GB each) are downloaded on-demand layer by layer.
- **On-demand eviction:** With `--clean-shards`, once all layers residing in a source shard are processed, that staging shard can be removed, keeping peak staging disk usage under **15 GiB**.

---

## 4. W4A16 Quantization Recipe for MoE (Qwen3.5 MoE)

The quantization parameters match the proven reference configuration of `Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP`:

| Parameter | Value | Rationale |
|---|---|---|
| `num_bits` | 4 | W4A16 precision (INT4 weights, BF16/FP16 activations) |
| `group_size` | 128 | Symmetric group quantization with per-group scales |
| `symmetric` | True | Required by vLLM XPU WNA16 int4 kernel |
| `observer` | `gptq` | Activation-aware second-order Hessian calibration |
| `pack_format` | `pack-quantized` | `compressed-tensors` int32 column nibble packing |
| `damp_percent` | 0.01 (1%) | Regularizes near-singular Hessians; boosted automatically if needed |
| `clip_factors` | `(1.0, 0.96, 0.92, 0.88, 0.84)` | Grid search minimizing RTN roundoff error |
| `targets` | `["Linear"]` | All routed experts, linear attention, and shared experts |
| `ignore` | `lm_head`, `embed_tokens`, `norm`, `conv1d`, `mlp.gate`, `mtp.*` | Critical modules preserved in unquantized BF16 |

### Layout in Safetensors:
For each quantized projection (e.g. `experts.0.gate_proj`):
- `weight_packed`: `torch.int32` tensor `[rows, cols // 8]` (8 4-bit nibbles per int32)
- `weight_scale`: `torch.bfloat16` tensor `[rows, cols // 128]`
- `weight_shape`: `torch.int64` tensor `[2]` storing `[rows, cols]`

---

## 5. MTP Speculative Decoding Draft Model

KAT-Coder-V2.5-Dev supports Multi-Token Prediction (MTP) for speculative decoding.
The MTP draft layer is stored as a standalone 19-tensor block in `models/_staging_kat_mtp/model-00014-of-mtp.safetensors`
(1.6 GB, BF16).

In the quantized artifact:
1. The 19 draft tensors are saved as `model-mtp.safetensors` in BF16.
2. In `config.json`, `text_config.mtp_num_hidden_layers` is set from `0` to `1` so vLLM initializes `Qwen3_5MoeMTP`.
3. All `mtp.*` tensors are registered in `model.safetensors.index.json` and excluded in `quantization_config.ignore`.
4. The server launches with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`, yielding up to **2.08x decode acceleration** without quality degradation.

---

## 6. Preflight Probe

Before starting a ~50-minute quantization run, execute the cheap preflight probe:

```bash
./.venv/bin/python scripts/preflight_probe_kat.py
```

The probe verifies:
1. **XPU Device & Driver:** `torch.xpu.is_available() == True`, Level-Zero NEO runtime active, primary device `Intel(R) Arc(TM) Graphics` (28.58 GiB addressable), matmul test passes.
2. **Memory & Storage:** ≥ 20 GiB available RAM, swap space active, ≥ 25 GiB free disk on `/`.
3. **Tokenizer:** `Qwen2Tokenizer` loads, vocab 248044, pad ID 248044, eos ID 248046, round-trip encoding succeeds.
4. **AutoConfig & Meta Layer:** Architecture resolves to `Qwen3_5MoeForConditionalGeneration`, 40 decoder layers, 256 routed experts, meta-device layer builds with **0 unmaterialized buffers**.
5. **MTP Draft Block:** `model-00014-of-mtp.safetensors` exists and carries exactly 19 valid tensors.

---

## 7. Running Quantization

Execute via the wrapper script:

```bash
# Check configuration and print planned command:
./scripts/quantize_kat_gptq.sh --check

# Full quantization run:
./scripts/quantize_kat_gptq.sh --calib-samples 64 --calib-length 1024
```

### Telemetry & Progress
Expected progress per layer on Arc 140V:
- **Hessian accumulation:** ~40–45 s
- **Expert GPTQ solver:** ~25–30 s
- **Layer forward pass:** ~10–12 s
- **Total per layer:** ~80 s
- **40 layers total:** ~50–55 minutes

Memory telemetry remains flat:
- `hostAvail`: ~22,000–24,000 MiB (stable)
- `swapFree`: ~30,880 MiB (0 swap thrashing)
- `xpuAlloc`: ~1,200–1,800 MiB (flat)

If interrupted, re-running the command automatically detects `models/KAT-Coder-V2.5-Dev-W4A16-GPTQ/hidden-after.pt`
and resumes from the last completed layer.

---

## 8. Verifying the Artifact

Verify the generated checkpoint before serving:

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

O = Path("models/KAT-Coder-V2.5-Dev-W4A16-GPTQ")
idx = json.loads((O / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]

packed = [k for k in wm if k.endswith(".weight_packed")]
scales = [k for k in wm if k.endswith(".weight_scale")]
shapes = [k for k in wm if k.endswith(".weight_shape")]
mtp = [k for k in wm if k.startswith("mtp.")]

print("=== Checkpoint Verification ===")
print(f"Total Tensors: {len(wm)}")
print(f"Packed W4 Projections: {len(packed)} (Expected: 31,160)")
print(f"Weight Scales:         {len(scales)} (Expected: 31,160)")
print(f"Weight Shapes:         {len(shapes)} (Expected: 31,160)")
print(f"MTP Draft Tensors:     {len(mtp)} (Expected: 19)")

assert len(packed) == len(scales) == len(shapes), "Mismatch in packed/scale/shape tensor counts!"
assert len(mtp) == 19, f"Expected 19 MTP tensors, got {len(mtp)}"
print("ALL ARTIFACT CHECKS PASSED.")
PY
```

Expected shard layout:
- `model-layer-00.safetensors` to `model-layer-39.safetensors` (~435 MiB each, containing packed expert and attention weights)
- `model-base.safetensors` (~2.9 GiB, containing `embed_tokens`, `lm_head`, norms)
- `model-mtp.safetensors` (~1.6 GiB, containing the 19 BF16 draft model tensors)
- Total size on disk: **~21 GiB across 42 safetensors files**.

---

## 9. Serving on Intel Arc 140V

Launch using the production launcher:

```bash
./launch_vllm_kat_gptq.sh
```

Or with custom parameters:
```bash
MAX_LEN=98304 ./launch_vllm_kat_gptq.sh 96k
PORT=8001 ./launch_vllm_kat_gptq.sh custom-port
```

### Production Serving Configuration:
- **Serving Script:** `scripts/launch_vllm_kat_gptq.sh` (symlinked as `launch_vllm_kat_gptq.sh`)
- **Model Path:** `models/KAT-Coder-V2.5-Dev-W4A16-GPTQ`
- **Served Model Name:** `KAT-Coder-V2.5-Dev-W4A16-GPTQ`
- **Context Length:** 131,072 tokens (128k context; ceiling 262,144)
- **KV Cache Budget:** FP8 KV cache with 2.5 GB allocated (`--kv-cache-memory-bytes 2500000000`), accommodating >200k tokens
- **Total Resident Memory:** ~23.0 GiB (safe 5.5 GiB headroom on 28.58 GiB addressable memory)
- **Speculative Decoding:** MTP K=2 (`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`)
- **Anti-Loop Sampling Parameters:** `--override-generation-config '{"temperature":0.6,"repetition_penalty":1.08,"presence_penalty":0.2,"top_p":0.95}'`
- **Tool Calling:** `--enable-auto-tool-choice --tool-call-parser qwen3_xml`

---

## 10. Failure Catalogue & Diagnostics (Arc 140V)

| Symptom | Underlying Cause | Solution |
|---|---|---|
| `UR_RESULT_ERROR_DEVICE_LOST` or Level-Zero timeout | GPU allocation exceeded physical RAM, triggering zram swap thrashing | Keep `--calib-samples 64 --calib-length 1024`. Do not exceed 64 samples on 30 GiB unified RAM. |
| Host process SIGKILL during layer 0 | Existing vLLM server process holding XPU memory | Run `pkill -f 'vllm serve'` before launching quantization. |
| `No space left on device (ENOSPC)` | Shards downloaded simultaneously without cleanup | Use `--clean-shards` flag in `quantize_kat_gptq.sh` to purge source shards after processing. |
| `RuntimeError: Hessian not positive definite` | Zero activations routed to certain experts during calibration | Automatic diagonal damping boost handles this; if persistent, increase damping with `DAMP=0.05`. |
| Cold-start KV cache allocation failure | First vLLM startup compiles kernels and exhausts memory | Restart launcher; compile cache warming succeeds on subsequent launch. |
| Detokenizer suppresses reasoning traces | Missing tool/reasoning parser | Always pass `--enable-auto-tool-choice --tool-call-parser qwen3_xml`. |

---

## 11. File Index

| File | Purpose |
|---|---|
| `GPTQ_quantization.md` | This operating guide (KAT-Coder-V2.5-Dev on Arc 140V) |
| `GPTQ_quantization_b60.md` | Upstream B60 reference guide (Coder390 on Arc Pro B60) |
| `scripts/preflight_probe_kat.py` | Cheap preflight validator for XPU, RAM, tokenizer, and layers |
| `scripts/quantize_kat_gptq.py` | Core W4A16 GPTQ quantizer engine with expert-batched Hessian solver |
| `scripts/quantize_kat_gptq.sh` | Orchestration shell script with server guard, memory logs, and verification |
| `scripts/launch_vllm_kat_gptq.sh` | Production vLLM launcher for Arc 140V with FP8 KV cache and MTP K=2 |
| `launch_vllm_kat_gptq.sh` | Root workspace symlink to the production serving launcher |
