#!/usr/bin/env python3
"""Phase 3: Offline load and forward generation smoke test for Qwen3.5-35B-A3B EXL3 MoE.

Per tasks_moe.md P3-02, P3-03, P3-04:
- Language model only (excludes vision tower)
- max_model_len=2048, max_num_seqs=1, max_num_batched_tokens=256
- gpu_memory_utilization=0.70, enforce_eager=True
- Logs peak RSS, XPU memory allocation, generation time and output text.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = str(ROOT / "models/turboderp-Qwen3.5-35B-A3B-exl3-3.09bpw")
GB = 1024**3


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def get_gpu_meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("GPUActive", "GPUReclaim", "MemAvailable", "MemFree")):
                    k, v = line.split(":", 1)
                    out[k.strip()] = int(v.split()[0]) * 1024
    except Exception:
        pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.62)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--json-out", default=str(ROOT / "logs/moe-p3/smoke-offline.json"))
    args = parser.parse_args()

    # Ensure baseline EXL3 environment is set
    os.environ["EXL3_VLLM_PATCHES"] = "0"
    os.environ["EXL3_SMALL_M_MAX"] = "8"
    os.environ["EXL3_INT8_PREFILL"] = "0"
    os.environ["EXL3_BACKEND"] = "auto"
    os.environ["EXL3_RECON_SLICE_N"] = "1024"
    os.environ["EXL3_NO_DNNL"] = "1"
    os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"

    import torch
    if not torch.xpu.is_available():
        print("ERROR: torch.xpu is not available")
        return 1

    print(f"torch={torch.__version__} xpu=True device={torch.xpu.get_device_name(0)}")
    print(f"Initial host RSS: {rss_gib():.2f} GiB")
    gm_init = get_gpu_meminfo()
    print(f"Initial MemAvailable: {gm_init.get('MemAvailable', 0) / GB:.2f} GiB, "
          f"GPUActive: {gm_init.get('GPUActive', 0) / GB:.2f} GiB, "
          f"GPUReclaim: {gm_init.get('GPUReclaim', 0) / GB:.2f} GiB")

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
        disable_log_stats=False,
        seed=0,
    )
    print(f"Initializing LLM with kwargs:\n{json.dumps(llm_kwargs, indent=2, default=str)}")

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    load_time = time.time() - t0
    print(f"Model loaded and engine initialized in {load_time:.2f}s")
    print(f"Post-load host RSS: {rss_gib():.2f} GiB")
    if torch.xpu.is_available():
        print(f"XPU allocated: {torch.xpu.memory_allocated() / GB:.2f} GiB, "
              f"reserved: {torch.xpu.memory_reserved() / GB:.2f} GiB, "
              f"max: {torch.xpu.max_memory_allocated() / GB:.2f} GiB")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=0,
    )

    print(f"Generating completion for prompt: {args.prompt!r}")
    t1 = time.time()
    outputs = llm.generate([args.prompt], sampling_params)
    gen_time = time.time() - t1

    output_record = outputs[0]
    generated_text = output_record.outputs[0].text
    prompt_tokens = len(output_record.prompt_token_ids)
    output_tokens = len(output_record.outputs[0].token_ids)

    print(f"Generation finished in {gen_time:.2f}s ({output_tokens / gen_time:.2f} tok/s)")
    print(f"Prompt tokens: {prompt_tokens}, Generated tokens: {output_tokens}")
    print("----- GENERATED OUTPUT -----")
    print(generated_text)
    print("----------------------------")

    gm_end = get_gpu_meminfo()
    print(f"Final host RSS: {rss_gib():.2f} GiB")
    print(f"Final MemAvailable: {gm_end.get('MemAvailable', 0) / GB:.2f} GiB, "
          f"GPUActive: {gm_end.get('GPUActive', 0) / GB:.2f} GiB, "
          f"GPUReclaim: {gm_end.get('GPUReclaim', 0) / GB:.2f} GiB")

    res = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "load_time_s": load_time,
        "gen_time_s": gen_time,
        "tokens_per_s": output_tokens / gen_time if gen_time > 0 else 0,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "peak_rss_gib": rss_gib(),
        "xpu_allocated_gib": torch.xpu.memory_allocated() / GB if torch.xpu.is_available() else 0,
        "mem_available_gib": gm_end.get("MemAvailable", 0) / GB,
        "status": "PASS",
    }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Results written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
