#!/usr/bin/env python3
"""Phase 6: correctness baseline over the fixed prompt suite.

Runs every prompt in scripts/prompts.json three times with fixed sampling,
records prompt/output/token counts/stop reason/errors, and flags obvious
decoding failures (empty output, NaN-ish text, repetition loops).

Modes:
  --mode server   (default) talks to a running OpenAI-compatible server
  --mode offline  builds an offline vLLM engine (slow: reloads the model)

    python scripts/run_prompt_suite.py --mode server --base-url http://127.0.0.1:8000/v1 \
        --model Qwen3.6-35B-A3B-MXFP4 --out logs/prompt-suite/run-01.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = json.loads((ROOT / "scripts" / "prompts.json").read_text())


def expand_long_prompt(prompt: dict, target_chars: int) -> str:
    text = prompt["chat"][-1]["content"]
    unit = prompt["repeat_unit"]
    if prompt.get("repeat_placeholder") in text:
        reps = max(1, target_chars // len(unit))
        text = text.replace(prompt["repeat_placeholder"], unit * reps)
    return text


def repetition_ratio(text: str, n: int = 40) -> float:
    """Fraction of the last `n`-char window that is a repeat of the preceding window."""
    if len(text) < 2 * n:
        return 0.0
    tail = text[-n:]
    prev = text[-2 * n : -n]
    return 1.0 if tail == prev else 0.0


def judge(text: str) -> list[str]:
    flags = []
    if not text.strip():
        flags.append("empty")
    if repetition_ratio(text) >= 1.0:
        flags.append("repetition_loop")
    if text.count("�") > 0:
        flags.append("replacement_chars")
    if len(re.findall(r"(.)\1{20,}", text)) > 0:
        flags.append("char_run")
    return flags


def run_server(args) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    results = []
    for p in SUITE["prompts"]:
        messages = [dict(m) for m in p["chat"]]
        if p.get("repeat_placeholder"):
            messages[-1]["content"] = expand_long_prompt(p, args.long_context_chars)
        for attempt in range(1, args.repeats + 1):
            t0 = time.time()
            rec = {
                "prompt_id": p["id"],
                "kind": p["kind"],
                "attempt": attempt,
                "messages": messages,
                "max_tokens": p.get("max_tokens", 256),
            }
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=SUITE["sampling"]["temperature"],
                    top_p=SUITE["sampling"]["top_p"],
                    max_tokens=rec["max_tokens"],
                    seed=SUITE["sampling"]["seed"],
                )
                ch = resp.choices[0]
                rec.update(
                    output=ch.message.content or "",
                    finish_reason=ch.finish_reason,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    latency_s=round(time.time() - t0, 3),
                    flags=judge(ch.message.content or ""),
                    error=None,
                )
            except Exception as e:
                rec.update(
                    output="", finish_reason=None, prompt_tokens=None,
                    completion_tokens=None, latency_s=round(time.time() - t0, 3),
                    flags=["exception"], error=repr(e),
                )
            results.append(rec)
            print(
                f"{p['id']} #{attempt}: {rec.get('completion_tokens')} tok, "
                f"{rec['latency_s']}s, flags={rec['flags']}"
                + (f", err={rec['error']}" if rec.get("error") else "")
            )
    return results


def run_offline(args) -> list[dict]:
    from vllm import LLM, SamplingParams

    kwargs: dict = dict(
        model=args.model_path,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        language_model_only=True,
    )
    if args.kv_cache_memory_bytes:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    print(f"LLM kwargs: {json.dumps(kwargs, default=str)}", flush=True)
    llm = LLM(**kwargs)
    results = []
    sp = SamplingParams(
        temperature=SUITE["sampling"]["temperature"],
        top_p=SUITE["sampling"]["top_p"],
        seed=SUITE["sampling"]["seed"],
    )
    for p in SUITE["prompts"]:
        messages = [dict(m) for m in p["chat"]]
        if p.get("repeat_placeholder"):
            messages[-1]["content"] = expand_long_prompt(p, args.long_context_chars)
        sp.max_tokens = p.get("max_tokens", 256)
        prompt_text = llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for attempt in range(1, args.repeats + 1):
            t0 = time.time()
            rec = {"prompt_id": p["id"], "kind": p["kind"], "attempt": attempt}
            try:
                out = llm.generate([prompt_text], sp)[0].outputs[0]
                rec.update(
                    output=out.text,
                    finish_reason=out.finish_reason,
                    prompt_tokens=len(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else None,
                    completion_tokens=len(out.token_ids),
                    latency_s=round(time.time() - t0, 3),
                    flags=judge(out.text),
                    error=None,
                )
            except Exception as e:
                rec.update(output="", flags=["exception"], error=repr(e),
                           latency_s=round(time.time() - t0, 3))
            results.append(rec)
            print(f"{p['id']} #{attempt}: {rec.get('completion_tokens')} tok "
                  f"{rec['latency_s']}s flags={rec['flags']}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["server", "offline"], default="server")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-MXFP4")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.74)
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=0,
                    help="pin KV cache size so capacity does not vary run to run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--long-context-chars", type=int, default=12000)
    ap.add_argument("--out", default=str(ROOT / "logs" / "prompt-suite" / "run.jsonl"))
    args = ap.parse_args()

    results = run_server(args) if args.mode == "server" else run_offline(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    flagged = [r for r in results if r.get("flags")]
    print(f"\n{len(results)} completions written to {out}")
    if flagged:
        print(f"{len(flagged)} flagged:")
        for r in flagged:
            print(f"  {r['prompt_id']} #{r['attempt']}: {r['flags']} {r.get('error') or ''}")
        return 1
    print("no empty/repetition/malformed outputs detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
