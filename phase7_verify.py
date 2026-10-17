#!/usr/bin/env python3
"""Startup + correctness verification for vLLM XPU (tasks.md Phases 6 & 7).

HTTP-side checks only (the log greps need `docker`, which the operator runs).

  Phase 6:
    * GET /health
    * GET /v1/models  -> confirm the served-model-name is listed
  Phase 7:
    * one throwaway request (accept the JIT-compile warmup penalty)
    * a second same-shape request -> latency must drop substantially
    * greedy determinism: temperature=0 prompt twice -> byte-identical output
      (a corrupting spec-decode path would diverge)

Stdlib only.

Usage:
    python3 phase7_verify.py --base-url http://127.0.0.1:8000 --model qwen38
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.parse


def _connect(base_url: str, timeout: float):
    u = urllib.parse.urlsplit(base_url)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(host, port, timeout=timeout), (u.path or "").rstrip("/")


def request(base_url: str, path: str, payload: dict | None, timeout: float,
            method: str = "GET"):
    conn, root = _connect(base_url, timeout)
    try:
        headers = {"Content-Type": "application/json"} if payload else {}
        body = json.dumps(payload) if payload is not None else None
        conn.request(method, f"{root}{path}", body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", "replace")
        return resp.status, raw
    finally:
        conn.close()


def chat_once(base_url: str, model: str, prompt: str, temperature: float,
              max_tokens: int, timeout: float) -> tuple[float, str]:
    """Returns (wall_seconds, assistant_text)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.monotonic()
    status, raw = request(base_url, "/v1/chat/completions", payload, timeout, "POST")
    dt = time.monotonic() - t0
    if status >= 400:
        raise RuntimeError(f"HTTP {status}: {raw[:600]}")
    obj = json.loads(raw)
    msg = obj["choices"][0]["message"]
    txt = (msg["content"] if isinstance(msg, dict) else msg.content) or ""
    return dt, txt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--skip", action="store_true", help="only /health + /v1/models")
    args = ap.parse_args()

    ok = True

    # ---- Phase 6 -------------------------------------------------------- #
    try:
        st, raw = request(args.base_url, "/health", None, 30)
        if st == 200:
            print(f"[phase6] /health -> 200 OK")
        else:
            print(f"[phase6] /health -> HTTP {st}: {raw[:200]}")
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f"[phase6] /health FAILED: {e}")
        print("         wait for `Application startup complete` in `docker logs -f vllm-xpu-b60`")
        return 2

    try:
        st, raw = request(args.base_url, "/v1/models", None, 30)
        obj = json.loads(raw)
        ids = [m.get("id") for m in obj.get("data", [])]
        print(f"[phase6] /v1/models -> {ids}")
        if args.model not in ids:
            print(f"[phase6] ERROR: expected model id {args.model!r} not listed")
            ok = False
        else:
            print(f"[phase6] model id {args.model!r} confirmed")
    except Exception as e:  # noqa: BLE001
        print(f"[phase6] /v1/models FAILED: {e}")
        ok = False

    if args.skip:
        return 0 if ok else 1

    # ---- Phase 7 -------------------------------------------------------- #
    prompt = "In one short sentence, explain what a speculative decode head does."

    print("\n[phase7] throwaway request 1 (JIT-compile warmup, penalty expected) ...")
    try:
        t1, a1 = chat_once(args.base_url, args.model, prompt, 0.0,
                           args.max_tokens, args.timeout)
        print(f"[phase7]   wall {t1:.2f}s  ({len(a1)} chars)")
    except Exception as e:  # noqa: BLE001
        print(f"[phase7]   request 1 FAILED: {e}")
        return 3

    print("[phase7] same-shape request 2 ...")
    try:
        t2, a2 = chat_once(args.base_url, args.model, prompt, 0.0,
                           args.max_tokens, args.timeout)
        print(f"[phase7]   wall {t2:.2f}s  ({len(a2)} chars)")
    except Exception as e:  # noqa: BLE001
        print(f"[phase7]   request 2 FAILED: {e}")
        return 3

    if t2 < t1:
        drop = (1.0 - t2 / t1) * 100.0
        print(f"[phase7] latency dropped {drop:.1f}% on run 2 "
              f"({'JIT compiled OK' if drop > 10 else 'marginal — check JIT logs'})")
    else:
        print(f"[phase7] WARNING: latency did NOT drop ({t1:.2f}s -> {t2:.2f}s)")

    # Greedy determinism spot-check.
    print("\n[phase7] greedy determinism (temperature=0, same prompt twice) ...")
    g_prompt = "Name the capital of France. Answer with one word only."
    try:
        _, g1 = chat_once(args.base_url, args.model, g_prompt, 0.0, 16, args.timeout)
        _, g2 = chat_once(args.base_url, args.model, g_prompt, 0.0, 16, args.timeout)
        same = g1 == g2
        print(f"[phase7]   run1: {g1!r}")
        print(f"[phase7]   run2: {g2!r}")
        if same:
            print("[phase7] byte-identical: PASS (spec-decode path not corrupting output)")
        else:
            print("[phase7] DIVERGED: FAIL — corrupting spec-decode path suspected")
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f"[phase7] determinism check FAILED: {e}")
        ok = False

    print("\nReminder — the log-side checks still need docker:")
    print("  docker logs vllm-xpu-b60 2>&1 | tail -40     # SpecDecoding metrics block")
    print("  docker logs vllm-xpu-b60 2>&1 | grep -iE "
          "'device memory|XCCL|GPTQ|drafter|MTP' ")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
