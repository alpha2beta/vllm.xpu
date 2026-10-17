#!/usr/bin/env python3
"""watch_stats.py — llama.cpp-style live pp/tg lines from vLLM Engine stats ticks.

vLLM already prints, every VLLM_LOG_STATS_INTERVAL (default 10 s), a line like:
  Engine 000: Avg prompt throughput: 358.2 tokens/s,
  Avg generation throughput: 24.7 tokens/s, Running: 0 reqs, Waiting: 0 reqs,
  GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%

This tails `docker logs -f` and reprints each tick as one short line:
  [14:02:11] pp 358.2 t/s | tg 24.7 t/s | run 0 wait 0 | kv 0.0% | hit 0.0%

Coarser than llama.cpp by design: these are per-engine aggregates over the
tick window, not per-request live counters — idle ticks read 0. For exact
per-request pp/TTFT/tg, use bench_p512_g128.py.

Usage:  python3 watch_stats.py [container]   (default: vllm-exl3-b60)
"""
import re
import subprocess
import sys

TICK = re.compile(
    r"(\d{2}:\d{2}:\d{2}).*?"
    r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*([\d.]+)\s*tokens/s,\s*"
    r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs,\s*"
    r"GPU KV cache usage:\s*([\d.]+%),\s*Prefix cache hit rate:\s*([\d.]+%)"
)

name = sys.argv[1] if len(sys.argv) > 1 else "vllm-exl3-b60"
proc = subprocess.Popen(
    ["sg", "docker", "-c", f"docker logs -f --tail 3 {name}"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
try:
    for line in proc.stdout:
        m = TICK.search(line)
        if m:
            ts, pp, tg, run, wait, kv, hit = m.groups()
            print(f"[{ts}] pp {pp:>8} t/s | tg {tg:>7} t/s | "
                  f"run {run} wait {wait} | kv {kv:>6} | hit {hit}", flush=True)
except KeyboardInterrupt:
    pass
finally:
    proc.terminate()
