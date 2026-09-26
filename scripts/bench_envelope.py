#!/usr/bin/env python3
"""Phase 8: memory/context envelope benchmark against the running server.

For a given input/output length and concurrency it reports TTFT, prefill
throughput, decode throughput and end-to-end latency, samples host memory and
GPUActive/GPUReclaim, and appends a row to results.csv.

    python scripts/bench_envelope.py --input-tokens 1024 --output-tokens 128 \
        --concurrency 1 --runs 2
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results.csv"
UNIT = "alpha beta gamma delta epsilon zeta eta theta "  # ~9-10 tokens per repeat
COLS = ("timestamp,git_revision,model_revision,backend,torch_version,vllm_version,"
        "kernel_version,driver_version,oneapi_version,context_tokens,input_tokens,"
        "output_tokens,max_num_seqs,max_num_batched_tokens,memory_utilization,eager,"
        "ttft_s,prefill_tps,decode_tps,total_latency_s,peak_memory_gb,swap_used_gb,"
        "result,notes").split(",")


def meminfo() -> dict[str, int]:
    d: dict[str, int] = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0])  # kB
    return d


def gpu_mem() -> dict[str, int]:
    out = {}
    for line in open("/proc/meminfo"):
        if line.startswith(("GPUActive", "GPUReclaim")):
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024  # bytes
    return out


def _tokenize(client, root: str, model: str, text: str) -> int | None:
    """POST /tokenize (note: NOT /v1/tokenize on this build).

    Uses urllib directly: the OpenAI client resolves paths against base_url,
    so passing an absolute URL here gets mangled and silently 404s.
    """
    import json as _json
    import urllib.request

    try:
        req = urllib.request.Request(
            f"{root.rstrip('/')}/tokenize",
            data=_json.dumps({"model": model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return len(_json.loads(resp.read()).get("tokens", []))
    except Exception:
        return None


def make_prompt(client, root: str, model: str, target_tokens: int) -> tuple[str, int]:
    """Build a prompt of ~`target_tokens` tokens, calibrated against the server tokenizer."""
    unit = UNIT

    def build(k: int) -> str:
        return ("Read the following block and then answer the question.\n\n"
                + unit * max(1, k)
                + "\n\nQuestion: what is the first word of the block? One word only.")

    k0 = max(1, target_tokens // 9)
    text, n0 = build(k0), _tokenize(client, root, model, build(k0))
    if n0 is None:  # no tokenizer endpoint: fall back to chars/4
        return text, len(text) // 4

    k1 = k0 * 2
    n1 = _tokenize(client, root, model, build(k1)) or n0
    per_repeat = max(1, (n1 - n0) / max(1, k1 - k0))
    base = n0 - per_repeat * k0
    k = max(1, int(round((target_tokens - base) / per_repeat)))
    text = build(k)
    n = _tokenize(client, root, model, text) or (base + per_repeat * k)
    return text, int(n)


def one_request(client, model: str, prompt: str, max_tokens: int) -> dict:
    from openai import OpenAI  # noqa: F401  (client passed in)
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    chunks = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    usage_out = None
    for ev in stream:
        if getattr(ev, "usage", None):
            usage_out = ev.usage
        choices = getattr(ev, "choices", None)
        if not choices:
            continue
        delta = choices[0].delta
        if getattr(delta, "content", None):
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
    total = time.perf_counter() - t0
    if usage_out is not None:
        out_tokens = usage_out.completion_tokens or 0
    ttft = ttft or total
    return {
        "ttft": ttft,
        "total": total,
        "out_tokens": out_tokens,
        "decode_tps": out_tokens / max(total - ttft, 1e-6),
        "chunks": chunks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen3.6-35B-A3B-MXFP4")
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600.0)
    root = args.base_url[:-3].rstrip("/") if args.base_url.endswith("/v1") else args.base_url

    prompt, in_tok = make_prompt(client, root, args.model, args.input_tokens)
    print(f"prompt ~{in_tok} tokens (target {args.input_tokens}), "
          f"max_tokens={args.output_tokens}, concurrency={args.concurrency}, "
          f"runs={args.runs}", flush=True)

    g0, m0 = gpu_mem(), meminfo()
    results: list[dict] = []
    lock = threading.Lock()

    def worker():
        r = one_request(client, args.model, prompt, args.output_tokens)
        with lock:
            results.append(r)
            print(f"  req: ttft={r['ttft']:.2f}s total={r['total']:.2f}s "
                  f"out_tok={r['out_tokens']} decode={r['decode_tps']:.2f} tok/s",
                  flush=True)

    t_start = time.perf_counter()
    for _ in range(args.runs):
        threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.perf_counter() - t_start

    g1, m1 = gpu_mem(), meminfo()
    if not results:
        print("no successful requests — nothing to report")
        return 1
    ttfts = [r["ttft"] for r in results]
    totals = [r["total"] for r in results]
    decodes = [r["decode_tps"] for r in results]
    out_toks = sum(r["out_tokens"] for r in results)
    prefill_tps = in_tok / statistics.median(ttfts) if ttfts else 0.0

    summary = (
        f"\nTTFT median={statistics.median(ttfts):.2f}s (min {min(ttfts):.2f} / "
        f"max {max(ttfts):.2f})  | prefill ~{prefill_tps:.1f} tok/s\n"
        f"decode: median={statistics.median(decodes):.2f} tok/s per request; "
        f"aggregate={out_toks / wall:.2f} tok/s over {wall:.1f}s wall\n"
        f"end-to-end: median={statistics.median(totals):.2f}s\n"
        f"GPUActive {g0.get('GPUActive', 0)/2**30:.2f} -> {g1.get('GPUActive', 0)/2**30:.2f} GiB | "
        f"GPUReclaim {g0.get('GPUReclaim', 0)/2**30:.2f} -> {g1.get('GPUReclaim', 0)/2**30:.2f} GiB\n"
        f"MemAvailable {m0['MemAvailable']/1048576:.2f} -> {m1['MemAvailable']/1048576:.2f} GiB | "
        f"swap used {(m0['SwapTotal']-m0['SwapFree'])/1048576:.2f} -> "
        f"{(m1['SwapTotal']-m1['SwapFree'])/1048576:.2f} GiB\n"
    )
    print(summary)

    if not args.no_csv:
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_revision": "none (not a git repo)",
            "model_revision": "7eceff3a9f7e6f916c824d197266d86676bce695",
            "backend": "vllm-0.30.0+xpu (Intel XPU / Arc 140V, Xe2)",
            "torch_version": "2.13.0+xpu",
            "vllm_version": "0.30.0+xpu",
            "kernel_version": "vllm-xpu-kernels 0.1.14.1",
            "driver_version": "xe + Level-Zero 1.17.39758",
            "oneapi_version": "2026.0.0",
            "context_tokens": args.input_tokens,
            "input_tokens": in_tok,
            "output_tokens": out_toks,
            "max_num_seqs": args.concurrency,
            "max_num_batched_tokens": 4096,
            "memory_utilization": 0.74,
            "eager": True,
            "ttft_s": round(statistics.median(ttfts), 3),
            "prefill_tps": round(prefill_tps, 2),
            "decode_tps": round(statistics.median(decodes), 2),
            "total_latency_s": round(statistics.median(totals), 3),
            "peak_memory_gb": round(g1.get("GPUActive", 0) / 2**30, 2),
            "swap_used_gb": round((m1["SwapTotal"] - m1["SwapFree"]) / 1048576, 2),
            "result": "ok",
            "notes": args.notes or (f"server bench: concurrency={args.concurrency} "
                                    f"runs={args.runs}; streamed, temperature=0"),
        }
        with CSV.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLS).writerow(row)
        print(f"appended row to {CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
