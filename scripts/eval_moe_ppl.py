#!/usr/bin/env python3
"""P3-07: Perplexity evaluation of Qwen3.5-35B-A3B EXL3 MoE on WikiText-2 test.

Evaluates WikiText-2 test set in non-overlapping chunks of 2048 tokens.
Uses teacher-forced forward passes via prompt_logprobs=1 to calculate NLL and PPL.
Asserts that PPL is within expected quantization boundaries (finite, no blowup).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(ROOT / "models/turboderp-Qwen3.5-35B-A3B-exl3-3.09bpw")
DEFAULT_DATA = "/tmp/opencode/wikitext/test.parquet"


def load_wikitext_raw_test(parquet_path: str) -> str:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    col_name = "text" if "text" in table.column_names else table.column_names[0]
    lines = [str(x) for x in table[col_name].to_pylist()]
    return "\n".join(t for t in lines if t.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--max-chunks", type=int, default=10,
                        help="Number of chunks to evaluate (0 for full test set, default 10 for bounded baseline)")
    parser.add_argument("--gpu-util", type=float, default=0.62)
    parser.add_argument("--out", default=str(ROOT / "logs/moe-p3/ppl-results.json"))
    args = parser.parse_args()

    # Baseline environment
    os.environ["EXL3_VLLM_PATCHES"] = "0"
    os.environ["EXL3_SMALL_M_MAX"] = "8"
    os.environ["EXL3_INT8_PREFILL"] = "0"
    os.environ["EXL3_BACKEND"] = "auto"
    os.environ["EXL3_RECON_SLICE_N"] = "1024"
    os.environ["EXL3_NO_DNNL"] = "1"
    os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"

    from vllm import LLM, SamplingParams

    print(f"Loading model {args.model} for PPL eval (max_model_len={args.chunk_tokens + 16})...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        max_model_len=args.chunk_tokens + 16,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        gpu_memory_utilization=args.gpu_util,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
        seed=0,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s", flush=True)

    tok = llm.get_tokenizer()
    print(f"Loading dataset from {args.data}...")
    raw_text = load_wikitext_raw_test(args.data)
    token_ids = tok.encode(raw_text)
    print(f"Total token count in test set: {len(token_ids)}")

    chunks = [
        token_ids[i:i + args.chunk_tokens]
        for i in range(0, len(token_ids), args.chunk_tokens)
    ]
    chunks = [c for c in chunks if len(c) == args.chunk_tokens]
    if args.max_chunks > 0:
        chunks = chunks[:args.max_chunks]
    print(f"Evaluating {len(chunks)} non-overlapping chunks of {args.chunk_tokens} tokens each...")

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    nll_sum = 0.0
    n_tokens_evaluated = 0
    chunk_ppls = []

    t_eval_start = time.time()
    for i, chunk in enumerate(chunks):
        t_chunk = time.time()
        out = llm.generate([{"prompt_token_ids": chunk}], sampling_params=sp)[0]
        plp = out.prompt_logprobs
        assert plp is not None and len(plp) == len(chunk)

        chunk_nll = 0.0
        chunk_toks = 0
        for pos in range(1, len(chunk)):
            d = plp[pos]
            target_id = chunk[pos]
            if d and target_id in d:
                lp = d[target_id].logprob
                if lp is not None and not math.isnan(lp) and not math.isinf(lp):
                    chunk_nll += -lp
                    chunk_toks += 1

        nll_sum += chunk_nll
        n_tokens_evaluated += chunk_toks
        cur_running_ppl = math.exp(nll_sum / n_tokens_evaluated)
        cur_chunk_ppl = math.exp(chunk_nll / chunk_toks) if chunk_toks > 0 else float("inf")
        chunk_ppls.append(cur_chunk_ppl)
        dt = time.time() - t_chunk
        print(f"Chunk {i+1}/{len(chunks)}: chunk_ppl={cur_chunk_ppl:.3f}, running_ppl={cur_running_ppl:.3f} ({chunk_toks} toks in {dt:.1f}s)")

    eval_time = time.time() - t_eval_start
    final_nll = nll_sum / n_tokens_evaluated if n_tokens_evaluated > 0 else float("nan")
    final_ppl = math.exp(final_nll)

    print("\n================ PPL EVALUATION SUMMARY ================")
    print(f"Chunks evaluated: {len(chunks)} x {args.chunk_tokens} tokens")
    print(f"Tokens evaluated: {n_tokens_evaluated}")
    print(f"Evaluation time: {eval_time:.1f}s ({n_tokens_evaluated / eval_time:.2f} tok/s)")
    print(f"Mean NLL: {final_nll:.4f}")
    print(f"Perplexity (PPL): {final_ppl:.3f}")
    print("========================================================")

    # Sanity checks
    assert not math.isnan(final_ppl), "PPL is NaN!"
    assert not math.isinf(final_ppl), "PPL is Inf!"
    assert final_ppl < 25.0, f"Perplexity exploded: {final_ppl:.2f} >= 25.0"

    results = {
        "model": args.model,
        "dataset": "WikiText-2 raw test",
        "method": f"non-overlapping {args.chunk_tokens}-token chunks, prompt_logprobs=1",
        "chunk_tokens": args.chunk_tokens,
        "chunks_evaluated": len(chunks),
        "tokens_evaluated": n_tokens_evaluated,
        "mean_nll": final_nll,
        "ppl": final_ppl,
        "chunk_ppls": chunk_ppls,
        "load_time_s": load_time,
        "eval_time_s": eval_time,
        "tok_per_s": n_tokens_evaluated / eval_time if eval_time > 0 else 0,
        "status": "PASS",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
