#!/usr/bin/env python3
"""p512/g128 decode benchmark harness for vLLM XPU (tasks.md Phase 8).

Implements plan section 6.4 as described verbatim in tasks.md:

  * build an exact 512-token prompt via /tokenize + /detokenize
  * run one same-shape warmup (JIT-compile penalty, not measured)
  * then 5 measured completions with max_tokens=128, temperature=0,
    ignore_eos=true, streaming enabled
  * rate per run = (completion_tokens - 1) / (t_end - t_first_token)
      - completion_tokens from the API's `usage` block (not SSE chunk counts)
      - t_end / t_first_token from a monotonic client clock (not the
        server's periodic logger)
  * record median, mean, min/max decode tok/s and median TTFT

Stdlib only: no pip dependencies required on the host.

Usage:
    python3 bench_p512_g128.py --base-url http://127.0.0.1:8000 \
        --model qwen38 --out /mnt/usb/vllm.xpu/bench_p512_g128.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import time
import urllib.parse
from typing import Any

# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only, streaming-capable)
# --------------------------------------------------------------------------- #


def _connect(base_url: str, timeout: float):
    u = urllib.parse.urlsplit(base_url)
    if u.scheme not in ("http", "https"):
        raise SystemExit(f"unsupported scheme in --base-url: {base_url}")
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(host, port, timeout=timeout), (u.path or "").rstrip("/")


def post_json(base_url: str, path: str, payload: dict[str, Any],
              timeout: float = 120.0) -> dict[str, Any]:
    conn, root = _connect(base_url, timeout)
    try:
        body = json.dumps(payload)
        conn.request("POST", f"{root}{path}", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        if resp.status >= 400:
            raise RuntimeError(f"{path} -> HTTP {resp.status}: {raw[:800]}")
        return json.loads(raw)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# /tokenize + /detokenize
# --------------------------------------------------------------------------- #


def tokenize(base_url: str, model: str, text: str) -> list[int]:
    """POST /tokenize (vLLM also exposes /v1/tokenize; both accepted).

    Contract verified against /openapi.json on the running server: the request
    is `TokenizeCompletionRequest` = {model, prompt} (a `text` field is rejected
    as "Field required ... prompt"), and the reply carries `tokens`.
    """
    payload = {"model": model, "prompt": text}
    last: Exception | None = None
    for path in ("/tokenize", "/v1/tokenize"):
        try:
            r = post_json(base_url, path, payload)
            toks = r.get("tokens")
            if toks is None:
                raise RuntimeError(f"{path} returned no 'tokens': {r}")
            return list(toks)
        except Exception as e:  # noqa: BLE001 - try the alias once
            last = e
    raise RuntimeError(f"tokenize failed on both endpoints: {last}")


def detokenize(base_url: str, model: str, ids: list[int]) -> str:
    """POST /detokenize -- reply key is `prompt` (verified against /openapi.json
    and a live call; some vLLM builds call it `text`, so accept either)."""
    payload = {"model": model, "tokens": ids}
    last: Exception | None = None
    for path in ("/detokenize", "/v1/detokenize"):
        try:
            r = post_json(base_url, path, payload)
            txt = r.get("prompt") if "prompt" in r else r.get("text")
            if txt is None:
                raise RuntimeError(f"{path} returned neither 'prompt' nor 'text': {r}")
            return str(txt)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"detokenize failed on both endpoints: {last}")


def build_exact_prompt(base_url: str, model: str, target: int) -> tuple[str, list[int]]:
    """Build text whose tokenization is *exactly* `target` tokens.

    Strategy (uses /tokenize + /detokenize as required by tasks.md):
      1. tokenize a base corpus; if long enough, take the first `target`
         ids, detokenize them, and verify the round trip.
      2. otherwise binary-search how many filler words to append so the
         total lands on `target`.
      3. always verify the final text re-tokenizes to exactly `target`
         tokens before returning.
    """
    base = (
        "You are a careful assistant. Summarize the following technical "
        "passage in plain language, then list its key trade-offs."
    )
    ids = tokenize(base_url, model, base)

    candidates: list[str] = []

    if len(ids) >= target:
        candidates.append(detokenize(base_url, model, ids[:target]))

    # Filler-word search: monotonic-ish token count as a function of count.
    filler = " Furthermore"
    lo, hi = 0, target * 4 + 16
    while lo <= hi:
        mid = (lo + hi) // 2
        text = base + filler * mid
        n = len(tokenize(base_url, model, text))
        if n == target:
            candidates.append(text)
            break
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1

    # Linear touch-up around the binary-search landing point.
    if not candidates:
        for delta in range(-8, 9):
            mid = max(0, (lo + hi) // 2 + delta)
            text = base + filler * mid
            if len(tokenize(base_url, model, text)) == target:
                candidates.append(text)
                break

    for text in candidates:
        got = tokenize(base_url, model, text)
        if len(got) == target:
            # Round-trip through /detokenize as the plan requires.
            rt = detokenize(base_url, model, got)
            final = rt if len(tokenize(base_url, model, rt)) == target else text
            if len(tokenize(base_url, model, final)) == target:
                return final, tokenize(base_url, model, final)

    raise SystemExit(
        f"could not construct an exact {target}-token prompt "
        f"(last attempts: {[len(candidates)]})"
    )


# --------------------------------------------------------------------------- #
# Streaming completion measurement
# --------------------------------------------------------------------------- #


def _iter_sse_lines(conn: http.client.HTTPConnection):
    while True:
        line = conn.fp.readline() if conn.fp is not None else b""
        if not line:
            return
        yield line.decode("utf-8", "replace").rstrip("\n").rstrip("\r")


def stream_completion(base_url: str, model: str, prompt: str, max_tokens: int,
                      timeout: float) -> dict[str, Any]:
    """One streaming completion; returns client-clock timings + usage."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    conn, root = _connect(base_url, timeout)
    t_request = time.monotonic()
    t_first: float | None = None
    usage: dict[str, Any] | None = None
    text_len = 0
    try:
        body = json.dumps(payload)
        conn.request("POST", f"{root}/v1/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status >= 400:
            raw = resp.read().decode("utf-8", "replace")
            raise RuntimeError(f"/v1/completions -> HTTP {resp.status}: {raw[:800]}")

        buf = b""
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode("utf-8", "replace").strip()
                if not s.startswith("data:"):
                    continue
                data = s[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    piece = (ch.get("text") or "")
                    if piece:
                        text_len += len(piece)
                        if t_first is None:
                            t_first = time.monotonic()
    finally:
        conn.close()

    t_end = time.monotonic()
    if t_first is None:
        raise RuntimeError("no streamed content arrived (t_first_token undefined)")
    if not usage:
        raise RuntimeError("no `usage` block received (stream_options unsupported?)")

    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    if completion_tokens <= 1:
        raise RuntimeError(f"completion_tokens={completion_tokens}; cannot compute rate")

    return {
        "t_request": t_request,
        "t_first": t_first,
        "t_end": t_end,
        "ttft_s": t_first - t_request,
        "decode_s": t_end - t_first,
        # tasks.md Phase 8 formula, verbatim:
        "decode_tok_s": (completion_tokens - 1) / (t_end - t_first),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "text_chars": text_len,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1, help="same-shape warmup runs (not measured)")
    ap.add_argument("--runs", type=int, default=5, help="measured completions")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="", help="write JSON report here")
    args = ap.parse_args()

    print(f"[setup] building exact {args.prompt_tokens}-token prompt via /tokenize+/detokenize ...",
          file=sys.stderr)
    prompt, ids = build_exact_prompt(args.base_url, args.model, args.prompt_tokens)
    print(f"[setup] prompt ready: {len(ids)} tokens, {len(prompt)} chars", file=sys.stderr)

    for i in range(args.warmup):
        print(f"[warmup {i + 1}/{args.warmup}] (JIT compile penalty, not measured)",
              file=sys.stderr)
        w = stream_completion(args.base_url, args.model, prompt, args.max_tokens,
                              args.timeout)
        print(f"[warmup {i + 1}] {w['decode_tok_s']:.2f} tok/s "
              f"(ttft {w['ttft_s']:.2f}s)", file=sys.stderr)

    runs: list[dict[str, Any]] = []
    for i in range(args.runs):
        r = stream_completion(args.base_url, args.model, prompt, args.max_tokens,
                              args.timeout)
        r["run"] = i + 1
        runs.append(r)
        print(f"[run {i + 1}/{args.runs}] decode {r['decode_tok_s']:.3f} tok/s  "
              f"ttft {r['ttft_s']:.3f}s  "
              f"tokens {r['completion_tokens']}  "
              f"prompt {r['prompt_tokens']}", file=sys.stderr)

    rates = [r["decode_tok_s"] for r in runs]
    ttfts = [r["ttft_s"] for r in runs]
    rate_stats = _stats(rates)
    ttft_stats = _stats(ttfts)

    report = {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "warmup": args.warmup,
            "measured_runs": args.runs,
            "rate_formula": "(completion_tokens - 1) / (t_end - t_first_token)",
            "clock": "time.monotonic() client-side",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "runs": runs,
        "decode_tok_s": rate_stats,
        "ttft_s": {
            "median": ttft_stats["median"],
            "mean": ttft_stats["mean"],
            "min": ttft_stats["min"],
            "max": ttft_stats["max"],
            "n": ttft_stats["n"],
        },
        "prompt_tokens_actual": runs[0]["prompt_tokens"] if runs else None,
    }

    print()
    print(f"decode tok/s : median {rate_stats['median']:.3f}  "
          f"mean {rate_stats['mean']:.3f}  "
          f"min {rate_stats['min']:.3f}  max {rate_stats['max']:.3f}  "
          f"(n={rate_stats['n']})")
    print(f"TTFT (s)     : median {ttft_stats['median']:.3f}  "
          f"mean {ttft_stats['mean']:.3f}  "
          f"min {ttft_stats['min']:.3f}  max {ttft_stats['max']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
