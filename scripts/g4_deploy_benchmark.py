#!/usr/bin/env python3
"""G4 Deployment Validation Benchmark for Qwen3.5-35B-A3B MoE (Arc 140V).

Runs 12 diverse prompts at max_tokens=480 (~5,760 tokens total) to satisfy the
G4 requirement of >= 3,000 drafted tokens on the certified MAX_LEN=2048 configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8002"

PROMPTS = [
    (
        "explain_refrigerator",
        "Explain in plain language how a refrigerator keeps food cold, detailing the thermodynamic cycle, compressor, condenser coils, and expansion valve step-by-step.\n\n",
    ),
    (
        "history_french_rev",
        "Write a comprehensive multi-paragraph overview of the social, economic, and political causes of the French Revolution, analyzing the Three Estates and the fiscal crisis under Louis XVI.\n\n",
    ),
    (
        "howto_derailleur",
        "Describe in detail how a bicycle derailleur moves the chain between gears, how cable tension and limit screws work, and how index shifting achieves precise gear selection.\n\n",
    ),
    (
        "story_lighthouse",
        "A detective arrives at an abandoned offshore lighthouse on a stormy night. Write a detailed atmospheric scene describing her investigation and what she discovers inside the keeper's quarters.\n\n",
    ),
    (
        "code_merge_sort",
        "Write an efficient Python implementation of Merge Sort and Quick Sort, including docstrings, type annotations, edge case handling for duplicates, and unit test assertions.\n\n",
    ),
    (
        "code_lru_cache",
        "Implement a thread-safe LRU Cache in Python using a doubly linked list and a hash map from scratch without using collections.OrderedDict. Provide get, put, evict, and clear methods.\n\n",
    ),
    (
        "math_bridge_logic",
        "Four hikers reach a bridge at night with one flashlight. Crossing times: 1, 2, 5, and 10 minutes. At most two people can cross together, moving at the slower hiker's pace. The flashlight must be carried back. Find the minimum total time for all four to cross, explaining every forward and return trip step-by-step.\n\n",
    ),
    (
        "math_compound_interest",
        "A person invests $15,000 at an annual interest rate of 6.5% compounded monthly. Calculate the future value after 10 years, showing all mathematical steps and the compounding formula, then compare it with continuous compounding.\n\n",
    ),
    (
        "physics_relativity",
        "Explain Einstein's special theory of relativity, focusing on time dilation and length contraction, using the classic light clock thought experiment with clear geometric derivations.\n\n",
    ),
    (
        "system_design",
        "Design a scalable distributed URL shortening service like TinyURL or Bitly. Detail the API endpoints, database schema, base62 encoding strategy, collision resolution, and multi-tier caching architecture.\n\n",
    ),
    (
        "code_parser",
        "Write a recursive descent expression parser in Python for arithmetic expressions supporting addition, subtraction, multiplication, division, unary minus, and parentheses, producing an AST.\n\n",
    ),
    (
        "philosophy_ethics",
        "Compare and contrast utilitarianism and deontological ethics regarding autonomous vehicle decision-making in unavoidable accident scenarios (the trolley problem applied to self-driving cars).\n\n",
    ),
]

def fetch_metrics() -> dict[str, float]:
    try:
        txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_") and "_total{" in line:
            name = line.split("{")[0].split(":")[1]
            out[name] = float(line.rsplit(" ", 1)[1])
    return out

def run_completion(prompt: str, max_tokens: int = 480, model: str = "sage-a1.5-exl3-mtp") -> dict:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    chunks: list[str] = []
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = ch.get("text") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                chunks.append(piece)
    total = time.time() - t0
    text = "".join(chunks)
    n = (usage or {}).get("completion_tokens") or len(chunks)
    decode_time = total - (ttft or 0.0)
    decode_tps = (n - 1) / decode_time if (ttft and decode_time > 0 and n > 1) else 0.0
    return {
        "ttft": ttft,
        "total": total,
        "decode_time": decode_time,
        "tokens": n,
        "decode_tps": decode_tps,
        "sample_text": text[:80].replace("\n", " "),
    }

def main():
    parser = argparse.ArgumentParser(description="G4 Deploy Benchmark")
    parser.add_argument("--tag", default=os.environ.get("G4_TAG", "run"), help="Tag for run (e.g. baseline, mtp-k1)")
    parser.add_argument("--model", default="sage-a1.5-exl3-mtp", help="Served model name")
    parser.add_argument("--tokens", type=int, default=480, help="Max tokens per prompt")
    parser.add_argument("--out-dir", default="logs/moe-p3", help="Output directory")
    args = parser.parse_args()

    print(f"=== G4 Deploy Benchmark: tag={args.tag}, model={args.model}, tokens={args.tokens} ===")
    os.makedirs(args.out_dir, exist_ok=True)

    m0 = fetch_metrics()
    rows = []
    print(f"{'Case':<25} | {'Tok':>5} | {'TTFT':>7} | {'Decode TPS':>11} | {'Sample Output'}")
    print("-" * 80)

    t_wall_start = time.time()
    for name, prompt in PROMPTS:
        r = run_completion(prompt, max_tokens=args.tokens, model=args.model)
        r["case"] = name
        rows.append(r)
        ttft_str = f"{r['ttft']:.2f}s" if r["ttft"] is not None else "N/A"
        print(f"{name:<25} | {r['tokens']:5d} | {ttft_str:>7} | {r['decode_tps']:10.2f}/s | {r['sample_text']}")

    t_wall_total = time.time() - t_wall_start
    m1 = fetch_metrics()
    spec_delta = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in set(m0) | set(m1)}

    tot_tokens = sum(r["tokens"] for r in rows)
    tot_decode_time = sum(r["decode_time"] for r in rows)
    agg_decode_tps = (tot_tokens - len(rows)) / tot_decode_time if tot_decode_time > 0 else 0.0

    drafts = spec_delta.get("spec_decode_num_draft_tokens_total", 0.0)
    accepted = spec_delta.get("spec_decode_num_accepted_tokens_total", 0.0)
    steps = spec_delta.get("spec_decode_num_drafts_total", 0.0)
    acc_rate = (accepted / drafts) if drafts > 0 else 0.0

    print("-" * 80)
    print(f"Total Output Tokens  : {tot_tokens}")
    print(f"Total Wall Time      : {t_wall_total:.1f} s")
    print(f"Aggregate Decode TPS : {agg_decode_tps:.2f} tok/s")
    if drafts > 0:
        print(f"Spec Drafts Total    : {drafts:.0f} (Target >= 3,000)")
        print(f"Spec Accepted Total  : {accepted:.0f}")
        print(f"Acceptance Rate      : {acc_rate * 100:.2f}% ({accepted:.0f}/{drafts:.0f})")
        print(f"Tokens / Step        : {1.0 + (accepted / steps):.3f}")

    out_file = os.path.join(args.out_dir, f"g4-{args.tag}-natural.json")
    summary = {
        "tag": args.tag,
        "tot_tokens": tot_tokens,
        "wall_time": t_wall_total,
        "aggregate_decode_tps": agg_decode_tps,
        "spec_delta": spec_delta,
        "drafts": drafts,
        "accepted": accepted,
        "steps": steps,
        "acceptance_rate": acc_rate,
        "rows": rows,
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote results to {out_file}")

if __name__ == "__main__":
    main()
