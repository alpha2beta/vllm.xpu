#!/usr/bin/env python3
"""P3-09: Probe OpenAI API endpoints, chat template, EOS stop tokens on port 8002.

Validates:
1. GET /v1/models (verifies served model name)
2. POST /v1/chat/completions with system + user message (chat template rendering)
3. Factual coherence and stop token termination (<|im_end|>, <|endoftext|>)
4. Saves full response record to logs/moe-p3/probe-chat-completion.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://127.0.0.1:8002"
MODEL_NAME = "qwen3.5-35b-a3b-exl3"


def http_get(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--out", default=str(ROOT / "logs/moe-p3/probe-chat-completion.json"))
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print(f"Probing vLLM MoE API at {base}...")

    # 1. Health check
    print("Step 1: Check /health")
    health_url = f"{base}/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"Health check status: {resp.status}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return 1

    # 2. Query /v1/models
    print("\nStep 2: Query /v1/models")
    models_res = http_get(f"{base}/v1/models")
    model_ids = [m["id"] for m in models_res.get("data", [])]
    print(f"Available models: {model_ids}")
    assert args.model in model_ids, f"Model {args.model} not found in {model_ids}"

    # 3. Query /v1/chat/completions
    print("\nStep 3: Query /v1/chat/completions (Chat Template test)")
    messages = [
        {"role": "system", "content": "You are an accurate, helpful assistant."},
        {"role": "user", "content": "What is the capital of Japan? Answer in one sentence."},
    ]
    chat_payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
        "seed": 0,
    }
    t0 = time.time()
    chat_res = http_post(f"{base}/v1/chat/completions", chat_payload)
    chat_time = time.time() - t0

    choice = chat_res["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason")
    usage = chat_res.get("usage", {})
    prompt_toks = usage.get("prompt_tokens", 0)
    compl_toks = usage.get("completion_tokens", 0)

    print(f"Chat response latency: {chat_time:.2f}s ({compl_toks / chat_time:.2f} tok/s)")
    print(f"Prompt tokens: {prompt_toks}, Completion tokens: {compl_toks}")
    print(f"Finish reason: {finish_reason}")
    print("Content:")
    print("-" * 40)
    print(content)
    print("-" * 40)

    # Validations
    assert finish_reason == "stop", f"Expected finish_reason == 'stop', got {finish_reason}"
    assert "Tokyo" in content or "tokyo" in content.lower(), f"Expected 'Tokyo' in answer, got: {content}"
    assert "<|im_end|>" not in content, "Leaked <|im_end|>"
    assert "<|endoftext|>" not in content, "Leaked <|endoftext|>"

    # 4. Query /v1/completions
    print("\nStep 4: Query /v1/completions (Text completion test)")
    comp_payload = {
        "model": args.model,
        "prompt": "The largest planet in our solar system is",
        "max_tokens": 16,
        "temperature": 0.0,
        "seed": 0,
    }
    t0 = time.time()
    comp_res = http_post(f"{base}/v1/completions", comp_payload)
    comp_time = time.time() - t0
    comp_text = comp_res["choices"][0]["text"]
    print(f"Text completion ({comp_time:.2f}s): {comp_text!r}")
    assert "Jupiter" in comp_text, f"Expected 'Jupiter' in output, got: {comp_text}"

    # Save output artifact
    out_record = {
        "timestamp": time.time(),
        "model": args.model,
        "endpoint": f"{base}/v1/chat/completions",
        "chat_response": chat_res,
        "completion_response": comp_res,
        "chat_time_s": chat_time,
        "comp_time_s": comp_time,
        "status": "PASS",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_record, f, indent=2)
    print(f"\nAudit record saved to {args.out}")
    print("ALL API PROBES PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
