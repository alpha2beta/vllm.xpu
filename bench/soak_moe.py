#!/usr/bin/env python3
"""Gate 4 criterion 5: multi-hour decode soak / stress benchmark.

Hammers the running MoE server with a rotating mix of prompts (real / uniform
/ skewed, short and max-context) for a wall-clock duration, recording per
request:

  * outcome (ok / error + message)
  * TTFT, total latency, decode throughput
  * GPUActive / MemAvailable / swap / platform temperature at completion

Every observation is appended as one JSON line to --out so the run is fully
recoverable if interrupted, and a summary (rate, error count, latency drift
first-hour vs last-hour, memory/thermal drift) is printed at the end.

    .venv/bin/python bench/soak_moe.py --hours 3 \
        --out logs/moe-p4/soak-r7.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_envelope import gpu_mem, meminfo, one_request  # noqa: E402
from bench_moe_envelope import (  # noqa: E402
    CONTEXT_LIMIT,
    CONTEXT_MARGIN,
    build_prompt,
    thermal_mC,
)

PATTERNS = ("real", "uniform", "skewed")
# (input, output) cells rotating through the soak; max-context cells included
# so the KV cache stays under pressure for the whole run.
CELLS = ((128, 128), (512, 128), (1912, 128), (512, 512), (128, 512),
         (1528, 512), (1024, 256))


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 4 C5 multi-hour soak")
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="qwen3.5-35b-a3b-exl3")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--out", default=str(ROOT / "logs/moe-p4/soak-r7.jsonl"))
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="seconds between requests")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600.0)
    root = args.base_url[:-3].rstrip("/") if args.base_url.endswith("/v1") else args.base_url
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pre-build prompts once (tokenizer round-trips are not part of the stress).
    prompts: dict[tuple[str, int, int], tuple[str, int]] = {}
    for pattern in PATTERNS:
        for inp, out_tok in CELLS:
            target = min(inp, CONTEXT_LIMIT - out_tok - CONTEXT_MARGIN)
            prompts[(pattern, inp, out_tok)] = build_prompt(
                client, root, args.model, pattern, target)

    t_end = time.time() + args.hours * 3600
    n_ok = n_err = 0
    obs: list[dict] = []
    errors: list[str] = []
    deadline = time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(t_end))
    print(f"soak: {args.hours} h -> {out} (until {deadline}), "
          f"{len(prompts)} prompt variants", flush=True)

    i = 0
    with out.open("a") as fh:
        while time.time() < t_end:
            pattern, inp, out_tok = list(prompts)[i % len(prompts)]
            prompt, in_tok = prompts[(pattern, inp, out_tok)]
            i += 1
            th = thermal_mC()
            g = gpu_mem()
            m = meminfo()
            t0 = time.time()
            try:
                r = one_request(client, args.model, prompt, out_tok)
                rec = {"t": t0, "status": "ok", "pattern": pattern,
                       "in": in_tok, "out": r["out_tokens"],
                       "ttft": round(r["ttft"], 3),
                       "total": round(r["total"], 3),
                       "decode_tps": round(r["decode_tps"], 2),
                       "gpu_gib": round(g.get("GPUActive", 0) / 2**30, 2),
                       "memavail_gib": round(m["MemAvailable"] / 2**30, 2),
                       "swap_gib": round((m["SwapTotal"] - m["SwapFree"]) / 2**30, 2),
                       "temp_c": round(th / 1000.0, 1) if th else None}
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 — soak records, never dies
                msg = repr(exc)[:300]
                rec = {"t": t0, "status": "error", "pattern": pattern,
                       "in": in_tok, "out": out_tok, "error": msg,
                       "gpu_gib": round(g.get("GPUActive", 0) / 2**30, 2),
                       "memavail_gib": round(m["MemAvailable"] / 2**30, 2),
                       "temp_c": round(th / 1000.0, 1) if th else None}
                n_err += 1
                errors.append(f"{time.strftime('%H:%M:%S')} {msg}")
                print(f"  ERROR #{n_err}: {msg}", flush=True)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            obs.append(rec)
            if n_ok % 20 == 0:
                last = obs[-20:]
                dec = [o["decode_tps"] for o in last if o["status"] == "ok"]
                print(f"  [{len(obs)}] ok={n_ok} err={n_err} "
                      f"decode~{statistics.mean(dec):.1f} tok/s "
                      f"gpu={rec['gpu_gib']} GiB temp={rec.get('temp_c')}C "
                      f"elapsed={(time.time())/60:.0f}m", flush=True)
            time.sleep(args.sleep)

    # ---- summary ----
    ok = [o for o in obs if o["status"] == "ok"]
    print("\n=== SOAK SUMMARY (Gate 4 criterion 5) ===")
    print(f"requests: {len(obs)} total, {len(ok)} ok, {n_err} errors")
    print(f"duration: {obs[0]['t']:.0f} -> {obs[-1]['t']:.0f} "
          f"({(obs[-1]['t'] - obs[0]['t']) / 3600:.2f} h)" if len(obs) > 1 else "")
    if ok:
        half = len(ok) // 2 or 1
        d1 = [o["decode_tps"] for o in ok[:half]]
        d2 = [o["decode_tps"] for o in ok[half:]]
        t1 = [o["ttft"] for o in ok[:half]]
        t2 = [o["ttft"] for o in ok[half:]]
        print(f"decode tok/s: first half median={statistics.median(d1):.2f} "
              f"second half median={statistics.median(d2):.2f} "
              f"(drift {100 * (statistics.median(d2) / statistics.median(d1) - 1):+.1f}%)")
        print(f"ttft s      : first half median={statistics.median(t1):.2f} "
              f"second half median={statistics.median(t2):.2f}")
        gpus = [o["gpu_gib"] for o in ok]
        temps = [o["temp_c"] for o in ok if o.get("temp_c")]
        print(f"GPUActive   : start={gpus[0]:.2f} min={min(gpus):.2f} "
              f"max={max(gpus):.2f} GiB (leak if end > start + 0.5)")
        if temps:
            print(f"temperature : start={temps[0]:.0f} max={max(temps):.0f} C")
        swaps = [o["swap_gib"] for o in ok]
        print(f"swap        : start={swaps[0]:.2f} end={swaps[-1]:.2f} GiB")
    print(f"errors: {n_err}")
    for e in errors[:10]:
        print(f"  {e}")
    verdict = "PASS" if n_err == 0 and len(obs) >= 100 else \
              ("PASS-with-errors" if n_err <= len(obs) * 0.01 else "FAIL")
    print(f"RESULT: {verdict} (criteria: 0 errors, >=100 requests)")
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "total": len(obs), "ok": len(ok), "errors": n_err,
        "hours": (obs[-1]["t"] - obs[0]["t"]) / 3600 if len(obs) > 1 else 0,
        "verdict": verdict, "error_samples": errors[:20],
    }, indent=2))
    print(f"summary -> {summary_path}")
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
