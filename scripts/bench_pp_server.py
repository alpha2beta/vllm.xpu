#!/usr/bin/env python3
"""Benchmark prompt-processing (prefill) TTFT and throughput against vLLM server on port 8002."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from transformers import AutoTokenizer

BASE = "http://127.0.0.1:8002"

def wait_for_server(timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"Server is healthy ({time.time() - start:.1f}s)")
                    return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("Server failed to respond to /health within timeout")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/yanchun/models/Sage-A1.5-EXL3-4.0bpw-mtp")
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--tag", default="prefill")
    args = parser.parse_args()

    wait_for_server()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Base prompt with varying repetition to get exact token counts without caching
    print("\n" + "=" * 80)
    print(f"Prefill Benchmark: Tag={args.tag}")
    print(f"{'Target Tokens':<15} | {'Actual Tokens':<15} | {'TTFT (s)':<12} | {'PP Speed (tok/s)':<18}")
    print("-" * 80)

    results = []
    # Vocabulary sample for pseudo-random text to prevent prefix-cache hits
    import random
    words = [
        "quantum", "entropy", "algorithm", "compiler", "sycl", "matrix", "vector",
        "tensor", "register", "pipeline", "bandwidth", "latency", "memory", "cache",
        "coherence", "atomic", "barrier", "distributed", "scheduler", "parallel"
    ]

    for target_len in args.lengths:
        # Generate random text
        rng = random.Random(target_len + 42)
        text = " ".join(rng.choice(words) for _ in range(int(target_len * 1.3)))
        tokens = tokenizer.encode(text)
        if len(tokens) > target_len:
            tokens = tokens[:target_len]
            text = tokenizer.decode(tokens)
        actual_len = len(tokens)

        # Run completion with max_tokens=1
        body = json.dumps({
            "model": "sage-a1.5-exl3",
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{BASE}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"}
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        t1 = time.perf_counter()

        ttft = t1 - t0
        pp_speed = actual_len / ttft if ttft > 0 else 0

        print(f"{target_len:<15} | {actual_len:<15} | {ttft:<12.3f} | {pp_speed:<18.1f}")
        results.append({
            "target_len": target_len,
            "actual_len": actual_len,
            "ttft_s": ttft,
            "pp_speed_tps": pp_speed,
        })

    print("=" * 80)
    out_file = f"logs/moe-p4/pp-{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {out_file}\n")

if __name__ == "__main__":
    main()
