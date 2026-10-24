#!/usr/bin/env python3
"""P3-08: Request isolation and KV/GDN state reset functional test.

Tests against the live vLLM MoE API server (port 8002):
1. Prompt prefill followed by single-token decode steps.
2. State isolation: Prompt A -> Prompt B -> Prompt A gives identical outputs,
   confirming KV cache and GDN recurrent states reset cleanly without cross-contamination.
3. Chunked prefill: Prompt > 256 tokens (exceeding max_num_batched_tokens=256) completes cleanly.
4. Stop conditions: verifies finish_reason == 'stop' and no stop tokens leaked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
MODEL_NAME = "qwen3.5-35b-a3b-exl3"


def call_chat_completion(base_url: str, messages: list[dict], max_tokens: int = 16, temperature: float = 0.0) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_completion(base_url: str, prompt: str, max_tokens: int = 16, temperature: float = 0.0) -> dict:
    url = f"{base_url.rstrip('/')}/completions"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_sequential_decode(base_url: str) -> dict:
    print("\n--- Test 1: Sequential decode ---")
    prompt = "The capital of France is"
    t0 = time.time()
    res = call_completion(base_url, prompt, max_tokens=8, temperature=0.0)
    elapsed = time.time() - t0
    choice = res["choices"][0]
    text = choice.get("text", "")
    print(f"Prompt: {prompt!r}")
    print(f"Generated ({elapsed:.2f}s): {text!r}")
    assert "Paris" in text, f"Expected 'Paris' in output, got: {text!r}"
    return {"name": "sequential_decode", "elapsed_s": elapsed, "output": text, "status": "PASS"}


def test_request_isolation(base_url: str) -> dict:
    print("\n--- Test 2: Request isolation (A -> B -> A) ---")
    prompt_a = "The capital city of Germany is"
    prompt_b = "The chemical element with atomic number 79 is"

    print("Step 2.1: Run Prompt A (first run)")
    t0 = time.time()
    res_a1 = call_completion(base_url, prompt_a, max_tokens=10, temperature=0.0)
    dt_a1 = time.time() - t0
    text_a1 = res_a1["choices"][0]["text"]
    print(f"Prompt A1 output ({dt_a1:.2f}s): {text_a1!r}")

    print("Step 2.2: Run Prompt B (intermediate run)")
    t0 = time.time()
    res_b = call_completion(base_url, prompt_b, max_tokens=10, temperature=0.0)
    dt_b = time.time() - t0
    text_b = res_b["choices"][0]["text"]
    print(f"Prompt B output ({dt_b:.2f}s): {text_b!r}")

    print("Step 2.3: Run Prompt A (second run after B)")
    t0 = time.time()
    res_a2 = call_completion(base_url, prompt_a, max_tokens=10, temperature=0.0)
    dt_a2 = time.time() - t0
    text_a2 = res_a2["choices"][0]["text"]
    print(f"Prompt A2 output ({dt_a2:.2f}s): {text_a2!r}")

    assert text_a1 == text_a2, (
        f"State leakage detected! Prompt A produced differing outputs:\n"
        f"Run 1: {text_a1!r}\nRun 2: {text_a2!r}"
    )
    print("MATCH: Prompt A1 and Prompt A2 are identical! KV & GDN states reset cleanly.")
    return {
        "name": "request_isolation",
        "prompt_a": prompt_a,
        "text_a1": text_a1,
        "prompt_b": prompt_b,
        "text_b": text_b,
        "text_a2": text_a2,
        "identical": (text_a1 == text_a2),
        "status": "PASS",
    }


def test_chunked_prefill(base_url: str) -> dict:
    print("\n--- Test 3: Chunked prefill (> 256 tokens) ---")
    # max_num_batched_tokens is 256, so a ~300-token prompt exercises multiple prefill chunks
    repeat_text = "The quick brown fox jumps over the lazy dog. " * 35  # ~350 tokens
    prompt = f"Background text: {repeat_text}\nQuestion: What animal is mentioned? Answer in one word:"
    t0 = time.time()
    res = call_completion(base_url, prompt, max_tokens=8, temperature=0.0)
    elapsed = time.time() - t0
    text = res["choices"][0]["text"]
    usage = res.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    print(f"Prompt tokens: {prompt_tokens} (exceeds batched token bound 256: {prompt_tokens > 256})")
    print(f"Generated text ({elapsed:.2f}s): {text!r}")
    assert any(w in text.lower() for w in ["fox", "dog"]), f"Expected fox/dog, got: {text!r}"
    return {
        "name": "chunked_prefill",
        "prompt_tokens": prompt_tokens,
        "output": text,
        "elapsed_s": elapsed,
        "status": "PASS",
    }


def test_stop_token_behavior(base_url: str) -> dict:
    print("\n--- Test 4: Stop condition & EOS behavior ---")
    messages = [
        {"role": "system", "content": "You are a concise factual assistant."},
        {"role": "user", "content": "What is 2 + 2? Answer with only the number and nothing else."},
    ]
    t0 = time.time()
    res = call_chat_completion(base_url, messages, max_tokens=256, temperature=0.0)
    elapsed = time.time() - t0
    choice = res["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")
    print(f"Chat response ({elapsed:.2f}s): {content!r}")
    print(f"Finish reason: {finish_reason!r}")
    assert finish_reason == "stop", f"Expected finish_reason 'stop', got {finish_reason!r}"
    assert "<|im_end|>" not in content, "EOS token <|im_end|> leaked into text"
    assert "<|endoftext|>" not in content, "EOS token <|endoftext|> leaked into text"
    assert "4" in content, f"Expected '4' in response, got: {content!r}"
    return {
        "name": "stop_token_behavior",
        "content": content,
        "finish_reason": finish_reason,
        "elapsed_s": elapsed,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-json", default="logs/moe-p3/request-isolation.json")
    args = parser.parse_args()

    results = []
    print(f"Running request isolation suite against {args.base_url}")
    results.append(test_sequential_decode(args.base_url))
    results.append(test_request_isolation(args.base_url))
    results.append(test_chunked_prefill(args.base_url))
    results.append(test_stop_token_behavior(args.base_url))

    all_pass = all(r.get("status") == "PASS" for r in results)
    summary = {
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.get("status") == "PASS"),
        "status": "PASS" if all_pass else "FAIL",
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll tests passed ({summary['passed']}/{summary['total_tests']})! Written to {args.out_json}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
