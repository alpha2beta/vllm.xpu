# Qwen3.5-35B-A3B EXL3 MoE Lifecycle Memory Report

**Model:** `turboderp/Qwen3.5-35B-A3B-exl3` @ `3.09bpw`  
**Platform:** Intel Core Ultra 7 258V (Arc 140V integrated GPU, 8 Xe2 cores, 32 GiB shared LPDDR5X-8533)  
**Software:** PyTorch 2.14.0+xpu, vLLM 0.31.0+xpu, Intel oneAPI 2026.0 (Level Zero 1.32.0)  
**Configuration:** `gpu_memory_utilization=0.62`, `max_model_len=2048`, `max_num_batched_tokens=256`, `max_num_seqs=1`, `--enforce-eager`  
**Evidence Trace:** [`logs/moe-p3/mem-trace.csv`](file:///home/yanchun/Projects/vllm.xpu/logs/moe-p3/mem-trace.csv) (713 samples @ 1 Hz)

---

## 1. Executive Summary

During the complete lifecycle of the live vLLM MoE API server (cold boot, streaming weight ingestion, vLLM dummy `profile_run`, KV-cache allocation, warmup, and live request serving), memory remained strictly bounded within the 32 GiB unified architecture:
- **Baseline Free Memory (Idle):** 22.91–23.52 GiB `MemAvailable`.
- **Peak Device Usage (Weights + KV Cache):** 18.08 GiB `GPUActive`.
- **Minimum Host `MemAvailable` Across Run:** **2.71 GiB** (strictly exceeding the $\ge 2.5\text{ GiB}$ Gate 3 safety boundary).
- **Swap Activity During Active Serving:** Zero swap thrashing ($\Delta \text{swap} = 0.00\text{ GiB}$ across benchmark iterations).
- **Post-Run Reclaim:** Full recovery back to 22.67 GiB `MemAvailable` via `scripts/reclaim_gpu_cache.py`.

---

## 2. Memory State Progression

| Lifecycle Phase | Duration | Host `MemAvailable` (GiB) | `GPUActive` (GiB) | `GPUReclaim` (GiB) | Host RSS (GiB) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Idle Pre-flight** | — | 23.51 | 0.29 | 0.24 | 0.05 | Reclaim script verified clean baseline |
| **Shard 1/2 Load** | 12.8s | 14.80 | 7.90 | 0.00 | 1.25 | Lazy generator streaming; no duplication |
| **Shard 2/2 Load** | 9.8s | 8.20 | 13.54 | 0.00 | 1.26 | 122,880/122,880 tensors loaded in 20.49s |
| **Engine `profile_run`** | 16.2s | 3.35 | 17.72 | 0.00 | 1.67 | Bounded dummy batch ($M=256$) prevents OOM |
| **KV Cache Init** | ~2s | 3.11 | 17.96 | 0.00 | 1.67 | 3.31 GiB KV cache (40,704 tokens) |
| **Warm Serving (Idle)** | — | 2.73 | 18.08 | 0.00 | 1.67 | Stable memory floor |
| **Active Decode (128 tok)** | 9.8s | 2.72 | 18.08 | 0.00 | 1.67 | Zero memory expansion during decode |
| **Active Decode (512 tok)** | 12.4s | 2.72 | 18.08 | 0.00 | 1.67 | Chunked prefill ($M=256$) executed smoothly |
| **Active Decode (1024 tok)**| 18.2s | 2.71 | 18.08 | 0.00 | 1.67 | Lowest observed memory point: **2.71 GiB** |
| **Post-Shutdown Reclaim**| 12s | **22.67** | 0.29 | 0.00 | 0.05 | Xe GPUReclaim pool fully released |

---

## 3. Shard Streaming & Allocation Validation (P3-02)

1. **Host RSS Ceiling:** Host RSS remained flat at $\le 1.67\text{ GiB}$ after loading the 14.25 GiB checkpoint, confirming that `safe_open` handles were properly closed and unmapped.
2. **Unified Memory Duplication:** `torch.xpu.empty_cache()` and `gc.collect()` executed after each shard eliminated transient host/device duplication.
3. **Headroom Compliance:** Host `MemAvailable` never breached the 2.5 GiB threshold, ensuring zero process termination by the Linux OOM killer.
