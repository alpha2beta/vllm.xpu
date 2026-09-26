#!/usr/bin/env python3
"""Phase 5: smallest possible offline load + deterministic completion test.

Conservative defaults per tasks.md: one sequence, short context, eager execution,
no speculative decoding, no compilation. Prints load time, peak host RSS, peak XPU
memory and the generated text.

    source .venv/bin/activate
    python scripts/smoke_offline.py --model models/Qwen3.6-35B-A3B-MXFP4 \
        | tee logs/smoke-offline-01.log
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

GB = 1024**3


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--hf-overrides", default=None, help='e.g. {"architectures": ["Qwen3_5MoeForCausalLM"]}')
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--language-model-only",
        action="store_true",
        help="skip the vision tower (saves ~0.9 GB); rejects image inputs",
    )
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    import torch

    print(f"torch={torch.__version__} xpu={torch.xpu.is_available()} "
          f"device={torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'n/a'}")
    print(f"rss before import: {rss_gib():.2f} GiB")

    from vllm import LLM, SamplingParams

    kwargs: dict = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=False,
    )
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if args.hf_overrides:
        kwargs["hf_overrides"] = json.loads(args.hf_overrides)

    print(f"LLM kwargs: {json.dumps(kwargs, default=str)}")
    t0 = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t0
    print(f"LOAD_SECONDS={load_s:.1f}  peak_rss={rss_gib():.2f} GiB")
    if torch.xpu.is_available():
        print(f"xpu_mem_allocated={torch.xpu.memory_allocated() / GB:.2f} GiB "
              f"max={torch.xpu.max_memory_allocated() / GB:.2f} GiB "
              f"reserved={torch.xpu.memory_reserved() / GB:.2f} GiB")

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=0)
    t1 = time.time()
    out = llm.generate([args.prompt], sp)
    gen_s = time.time() - t1

    rec = out[0]
    text = rec.outputs[0].text
    print(f"GENERATE_SECONDS={gen_s:.1f}")
    print(f"prompt_tokens={len(rec.prompt_token_ids)} "
          f"output_tokens={len(rec.outputs[0].token_ids)} "
          f"finish_reason={rec.outputs[0].finish_reason}")
    print("---- output ----")
    print(text)
    print("---- end ----")
    print(f"peak_rss_end={rss_gib():.2f} GiB")
    if torch.xpu.is_available():
        print(f"xpu_max_allocated_end={torch.xpu.max_memory_allocated() / GB:.2f} GiB")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model": args.model,
            "prompt": args.prompt,
            "output": text,
            "load_s": load_s,
            "generate_s": gen_s,
            "peak_rss_gib": rss_gib(),
            "xpu_max_alloc_gib": torch.xpu.max_memory_allocated() / GB if torch.xpu.is_available() else None,
            "kwargs": kwargs,
        }, indent=2, default=str))

    bad = not text.strip() or text.count("\n\n") > 3
    print("RESULT:", "SUSPECT" if bad else "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
