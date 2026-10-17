#!/usr/bin/env python3
"""Prefill (prompt-processing) rate sweep.

The decode rate is recorded by Phase 8's harness; this measures the *other* half.

Two independent measurements, at each prompt length:

  * client  -- wall-clock TTFT from the request send to the first streamed token
               (includes tokenize + scheduling + network, so it has a fixed offset)
  * server  -- deltas of vLLM's own `request_prefill_time_seconds` and
               `request_prefill_kv_computed_tokens` histograms, snapshotted around
               the requests. This is pure prefill on the engine and is the figure
               that should be quoted as "the prefill rate".

A least-squares fit of TTFT against prompt length gives the *marginal* rate
(1 / slope) -- i.e. what an extra token of prompt costs -- with the intercept
absorbing fixed per-request overhead.

Usage:
    python3 bench_prefill.py [--lengths 256,512,1024,2048,4096,6144]
                             [--reps 3] [--out prefill.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from typing import Any

from bench_p512_g128 import build_exact_prompt, stream_completion

SUM_FMT = r"^vllm:%s_sum\{[^}]*\}\s+([0-9.eE+]+)$"


# --------------------------------------------------------------------------- #
# Server-side counters
# --------------------------------------------------------------------------- #
def fetch_metrics(base_url: str, timeout: float = 10.0) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def prefill_stats(base_url: str) -> tuple[float, float]:
    """(cumulative prefill seconds, cumulative prefill KV-computed tokens)."""
    text = fetch_metrics(base_url)
    mt = re.search(SUM_FMT % "request_prefill_time_seconds", text, re.M)
    mk = re.search(SUM_FMT % "request_prefill_kv_computed_tokens", text, re.M)
    if not mt or not mk:
        raise RuntimeError("prefill metrics not found on /metrics")
    return float(mt.group(1)), float(mk.group(1))


# --------------------------------------------------------------------------- #
# Stats / fitting
# --------------------------------------------------------------------------- #
def med(v: list[float]) -> float:
    return statistics.median(v) if v else float("nan")


def linfit(xs: list[float], ys: list[float]) -> dict[str, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return {"slope": b, "intercept": a, "r2": r2}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--lengths", default="256,512,1024,2048,4096,6144",
                    help="comma-separated prompt lengths (keep <= 8192: that is "
                         "--max-num-batched-tokens, above which prefill chunks)")
    ap.add_argument("--reps", type=int, default=3, help="requests per length")
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="short decode; prefill rate comes from TTFT + server counters")
    ap.add_argument("--out", default="prefill.json")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    # one warmup so JIT/graph compilation never lands in a measurement
    print("[setup] building warmup prompt ...", file=sys.stderr)
    warm, _ = build_exact_prompt(args.base_url, args.model, min(lengths))
    stream_completion(args.base_url, args.model, warm, args.max_tokens, args.timeout)
    print("[setup] warmup done (not measured)", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for L in lengths:
        print(f"[setup] building exact {L}-token prompt ...", file=sys.stderr)
        prompt, _ = build_exact_prompt(args.base_url, args.model, L)

        t0, k0 = prefill_stats(args.base_url)
        ttfts: list[float] = []
        for i in range(args.reps):
            r = stream_completion(args.base_url, args.model, prompt,
                                  args.max_tokens, args.timeout)
            ttfts.append(r["ttft_s"])
            print(f"  L={L:<5} rep {i + 1}/{args.reps}  "
                  f"ttft {r['ttft_s']:.4f}s  prompt {r['prompt_tokens']}",
                  file=sys.stderr)
        t1, k1 = prefill_stats(args.base_url)

        d_t, d_k = t1 - t0, k1 - k0
        row = {
            "length": L,
            "ttft_s": med(ttfts),
            "ttft_all_s": ttfts,
            "server_prefill_s": d_t,
            "server_tokens": d_k,
            "server_tok_s": (d_k / d_t) if d_t > 0 else float("nan"),
        }
        rows.append(row)
        print(f"[{L:>5}] ttft median {row['ttft_s'] * 1000:7.1f} ms   "
              f"server {d_k:8.0f} tok / {d_t:6.3f} s = "
              f"{row['server_tok_s']:8.0f} tok/s")

    # ---- fits ----------------------------------------------------------- #
    # Marginal cost per prompt token = slope of *time* against *tokens*.
    # (Regressing inverse-rate on length is wrong: rate rises with length as the
    # fixed per-request cost amortises, which drives the inverse-rate slope
    # negative and the reported rate to nonsense.)
    xs = [float(r["length"]) for r in rows]
    fit_ttft = linfit(xs, [r["ttft_s"] for r in rows])
    fit_srv = linfit([r["server_tokens"] for r in rows],
                     [r["server_prefill_s"] for r in rows])

    srv_tok = sum(r["server_tokens"] for r in rows)
    srv_sec = sum(r["server_prefill_s"] for r in rows)
    overall = srv_tok / srv_sec if srv_sec > 0 else float("nan")

    marginal_client = 1.0 / fit_ttft["slope"] if fit_ttft["slope"] else float("nan")
    marginal_server = 1.0 / fit_srv["slope"] if fit_srv["slope"] else float("nan")
    plateau = max(r["server_tok_s"] for r in rows)
    n90 = next((r["length"] for r in rows if r["server_tok_s"] >= 0.90 * plateau), None)

    print()
    print("=" * 74)
    print(f"{'length':>8} {'TTFT median':>13} {'server prefill':>16} {'rate':>12}")
    for r in rows:
        print(f"{r['length']:>8} {r['ttft_s'] * 1000:>11.1f} ms "
              f"{r['server_tokens']:>7.0f}/{r['server_prefill_s']:>6.3f}s "
              f"{r['server_tok_s']:>10.0f} t/s")
    print("-" * 74)
    print(f"aggregate server prefill rate   : {overall:,.0f} tok/s "
          f"({srv_tok:,.0f} tok / {srv_sec:.3f} s)")
    print(f"marginal server rate (time/tok) : {marginal_server:,.0f} tok/s "
          f"({fit_srv['slope'] * 1000:.3f} ms/token, "
          f"{fit_srv['intercept'] * 1000:+.0f} ms/req, R2={fit_srv['r2']:.4f})")
    print(f"marginal client rate (TTFT slope): {marginal_client:,.0f} tok/s "
          f"({fit_ttft['slope'] * 1000:.3f} ms/token, "
          f"{fit_ttft['intercept'] * 1000:+.0f} ms/req, R2={fit_ttft['r2']:.4f})")
    print(f"plateau (best single length)    : {plateau:,.0f} tok/s; "
          f"90% of it is reached by ~{n90} tokens")
    print("=" * 74)

    with open(args.out, "w") as fh:
        json.dump({
            "rows": rows,
            "aggregate_server_tok_s": overall,
            "marginal_server_tok_s": marginal_server,
            "marginal_server_ms_per_token": fit_srv["slope"] * 1000,
            "marginal_server_intercept_s": fit_srv["intercept"],
            "server_fit_r2": fit_srv["r2"],
            "marginal_client_tok_s": marginal_client,
            "ttft_intercept_s": fit_ttft["intercept"],
            "ttft_r2": fit_ttft["r2"],
            "plateau_tok_s": plateau,
            "tokens_to_90pct_plateau": n90,
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "note": "server rates are deltas of vllm:request_prefill_* counters; "
                    "prefix caching is OFF so every prompt token is computed; "
                    "everything tested is <= 8192 tokens, i.e. a single "
                    "prefill chunk (max-num-batched-tokens).",
        }, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
