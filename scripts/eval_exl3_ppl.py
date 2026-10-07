#!/usr/bin/env python3
"""Perplexity of the local EXL3 4.00bpw checkpoint on WikiText-2 raw test.

Method (standard non-overlapping chunks, reported as such):
  - concatenate WikiText-2 raw test, tokenize with the model tokenizer,
  - split into non-overlapping chunks of --chunk-tokens (default 4096),
  - one forward per chunk via vLLM offline `prompt_logprobs` (no generation),
  - NLL = -mean(log p(tok_i | tok_<i])) over all predicted tokens (skip the
    first token of each chunk, which has empty context),
  - PPL = exp(NLL).

Environment (same baseline as serving; set before import):
  EXL3_VLLM_PATCHES=0 EXL3_SMALL_M_MAX=8 EXL3_INT8_PREFILL=0 EXL3_BACKEND=auto
  EXL3_RECON_SLICE_N=1024 EXL3_NO_DNNL=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu

Usage:
  source logs/<run>/baseline-env.sh
  EXL3_SRC=~/Projects/exl3xpu .venv/bin/python scripts/eval_exl3_ppl.py \
      --model models/turboderp-Qwen3.8-27B-exl3-4.00bpw \
      --data /tmp/opencode/wt2-test.parquet \
      --chunk-tokens 4096 --max-chunks 0 --out logs/<run>/ppl.json
  (--max-chunks 0 = full test set.)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def load_wikitext_raw_test(parquet: str) -> str:
    import pandas as pd
    df = pd.read_parquet(parquet)
    col = "text" if "text" in df.columns else df.columns[0]
    # WikiText-2 raw keeps one article line per row plus "=" headers; join with
    # newlines, drop empty rows. Standard evaluators use the raw token stream.
    lines = [str(t) for t in df[col].tolist()]
    return "\n".join(t for t in lines if t.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/turboderp-Qwen3.8-27B-exl3-4.00bpw")
    ap.add_argument("--data", default="/tmp/opencode/wt2-test.parquet")
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.70)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for var, want in (("EXL3_VLLM_PATCHES", "0"), ("EXL3_SMALL_M_MAX", "8")):
        if str(__import__("os").environ.get(var)) != want:
            print(f"refusing: {var}={__import__('os').environ.get(var)!r}, "
                  f"baseline requires {want!r}")
            return 2

    from vllm import LLM, SamplingParams

    t0 = time.time()
    # +16 headroom: prompt (chunk) + requested output token must fit max_model_len.
    llm = LLM(model=args.model, enforce_eager=True,
              max_model_len=args.chunk_tokens + 16, max_num_seqs=1,
              max_num_batched_tokens=1024,
              gpu_memory_utilization=args.gpu_util,
              language_model_only=True, seed=0)
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    tok = llm.get_tokenizer()
    text = load_wikitext_raw_test(args.data)
    ids = tok.encode(text)
    print(f"test tokens: {len(ids)}", flush=True)
    chunks = [ids[i:i + args.chunk_tokens]
              for i in range(0, len(ids), args.chunk_tokens)]
    chunks = [c for c in chunks if len(c) == args.chunk_tokens]
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"chunks: {len(chunks)} x {args.chunk_tokens}", flush=True)

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    nll_sum = 0.0
    n_tok = 0
    for i, c in enumerate(chunks):
        out = llm.generate([{"prompt_token_ids": c}],
                           sampling_params=sp)[0]
        plp = out.prompt_logprobs  # [None, {tok: logprob}, ...], len == chunk
        assert plp is not None and len(plp) == len(c), (len(plp or []), len(c))
        for pos in range(1, len(c)):
            d = plp[pos]
            lp = d[c[pos]].logprob if c[pos] in d else None
            if lp is None:  # tokenizer/offset edge: skip, count it
                continue
            nll_sum += -lp
            n_tok += 1
        if (i + 1) % 10 == 0 or i + 1 == len(chunks):
            cur = math.exp(nll_sum / n_tok)
            print(f"  chunk {i+1}/{len(chunks)} running PPL={cur:.3f} "
                  f"({n_tok} toks, {time.time()-t0:.0f}s)", flush=True)

    ppl = math.exp(nll_sum / n_tok)
    res = {"model": args.model, "dataset": "wikitext-2-raw-v1 test",
           "method": f"non-overlapping {args.chunk_tokens}-tok chunks, "
                     "prompt_logprobs, skip chunk-first tokens",
           "tokens": n_tok, "nll": nll_sum / n_tok, "ppl": ppl,
           "chunks": len(chunks), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=2))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
