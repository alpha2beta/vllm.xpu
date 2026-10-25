#!/usr/bin/env python3
"""P4-07: comprehensive MoE benchmark sweep for Gate 4.

Executes the full evaluation matrix from tasks_moe.md against the running
EXL3 MoE server (loopback :8002, single device, TP=1, max 2048 context):

  * input lengths  : 128, 512, 2048 tokens
  * output lengths : 128, 512 tokens
  * routing patterns: real text, synthetic uniform, synthetic skewed
    (prompt-construction proxies for the routing distribution — routed-expert
    telemetry is disabled in this build, aux_output_config returns no routing)

Per cell it records TTFT, prefill/decode throughput, end-to-end latency,
effective memory-bandwidth efficiency (% of the P0-12 130.4 GB/s roofline,
using the P4-05 measured 26.50 MiB/token/layer x 40 layers trellis read
volume), peak GPU/host memory and the platform thermal sensor before/after.
Cells whose input+output exceed the server's 2048-token context are attempted
and recorded honestly (skip/truncated rows carry the observed reason).

Rows are published to results_moe.csv.

    .venv/bin/python bench/bench_moe_envelope.py \
        --base-url http://127.0.0.1:8002/v1 --model qwen3.5-35b-a3b-exl3 \
        --git-revision 54fedb2 --artifact 13c235df
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_envelope import (  # noqa: E402  (shared client/tokenize helpers)
    _tokenize,
    make_prompt,
    meminfo,
    gpu_mem,
    one_request,
)

CSV = ROOT / "results_moe.csv"
COLS = (
    "timestamp,git_revision,artifact_sha,model_revision,backend,context_limit,"
    "routing_pattern,input_pattern,input_tokens,output_tokens,ttft_s,prefill_tps,"
    "decode_tps,total_latency_s,bandwidth_gbps,bandwidth_eff_pct,peak_gpu_gb,"
    "swap_used_gb,thermal_start_c,thermal_end_c,result,notes"
).split(",")

# P4-05 measured trellis read volume (logs/moe-p4/bandwidth-efficiency.log).
MIB_PER_TOKEN_PER_LAYER = 26.50
NUM_LAYERS = 40
ROOFLINE_GBPS = 130.4          # P0-12 copy roofline, Arc 140V
BYTES_PER_TOKEN = MIB_PER_TOKEN_PER_LAYER * NUM_LAYERS * 2**20
CONTEXT_LIMIT = 2048           # server --max-model-len
CONTEXT_MARGIN = 8             # tokenizer-calibration slack for prompt + output

REAL_TEXT = (
    "The Franco-Prussian War of 1870 reshaped the balance of power in Europe "
    "far more decisively than its nine-month duration might suggest. When the "
    "Prussian armies encircled Paris, the conflict had already answered the "
    "question that had haunted diplomats since 1815: whether a unified Germany "
    "could exist without dominating the continent. Historians still debate "
    "whether Bismarck provoked the war deliberately or merely exploited the "
    "mistakes of Napoleon III, but the outcome was never really in doubt once "
    "the mobilization timetables began to run. The siege of Paris itself "
    "produced its own literature, from the balloon mail dispatched nightly "
    "over the Prussian lines to the bitter municipal politics of the "
    "Government of National Defence. What mattered most, in the long run, was "
    "not the Treaty of Frankfurt but the habit of mind it installed: the "
    "conviction, on both sides of the Rhine, that the next war would have to "
    "be finished quickly, before the enemy's industry could be brought fully "
    "to bear. That conviction shaped planning for the next four decades."
)


def uniform_words(n: int, seed: int = 1234) -> str:
    """Synthetic 'uniform' routing proxy: widely mixed everyday vocabulary."""
    import random

    rng = random.Random(seed)
    vocab = (
        "river stone window garden copper amber forest letter bridge mirror "
        "candle harvest thunder valley quarry lantern orchard granite meadow "
        "harbor pebble spiral compass anvil willow marble glacier prairie "
        "cinder thistle falcon marina basalt canyon dune terrace vineyard "
        "summit pendant cobbler lighthouse anemone barricade chandelier"
    ).split()
    return " ".join(rng.choice(vocab) for _ in range(n))


def skewed_words(n: int) -> str:
    """Synthetic 'skewed' routing proxy: one rare token repeated (hot experts)."""
    return " defenestration" * n


def thermal_mC() -> float | None:
    """Platform temperature (acpitz) in millidegC, or None if unavailable."""
    for p in sorted(Path("/sys/class/hwmon").glob("hwmon*/temp1_input")):
        try:
            if (p.parent / "name").read_text().strip() in ("acpitz", "coretemp"):
                return float(p.read_text().strip())
        except OSError:
            continue
    return None


def _shrink_to_fit(client, root: str, model: str, build, k: int,
                   target: int) -> tuple[str, int]:
    """Rebuild the prompt with fewer repeats until it fits `target` tokens.

    Calibration rounds the repeat count, which can land a few tokens over —
    fatal when prompt + output must stay under the 2048-token context.
    """
    text = build(max(1, k))
    n = _tokenize(client, root, model, text)
    per = None
    while n is not None and n > target and k > 1:
        if per is None:
            per = max(1.0, n / max(1, k))
        k = max(1, k - max(1, int((n - target) / per)))
        text = build(k)
        n = _tokenize(client, root, model, text)
    return text, int(n or 0)


def build_prompt(client, root: str, model: str, pattern: str,
                 target: int) -> tuple[str, int]:
    """Build a ~target-token prompt for one routing pattern."""
    if pattern == "real":
        # Calibrate the repeat count with tokenizer feedback (the paragraph is
        # ~130 tokens, so a naive target//90 overshoots small targets).
        def build(k: int) -> str:
            return REAL_TEXT * max(1, k)
        k0 = 1
        n0 = _tokenize(client, root, model, build(k0))
        if n0 is None:
            text = build(max(1, target // 130))
            return text, len(text) // 4
        if n0 >= target:
            return build(1), n0
        k1 = 3
        n1 = _tokenize(client, root, model, build(k1)) or (n0 * 3)
        per = max(1e-6, (n1 - n0) / (k1 - 1))
        base = n0 - per
        k = max(1, int(round((target - base) / per)))
        text, n = _shrink_to_fit(client, root, model, build, k, target)
        return text, n
    # synthetic patterns: calibrate with the same tokenizer feedback loop
    def build(k: int) -> str:
        unit = uniform_words(12) if pattern == "uniform" else skewed_words(4)
        return unit * max(1, k)
    k0 = max(1, target // 12)
    n0 = _tokenize(client, root, model, build(k0))
    if n0 is None:
        text = build(k0)
        return text, len(text) // 4
    k1 = k0 * 2
    n1 = _tokenize(client, root, model, build(k1)) or n0
    per = max(1e-6, (n1 - n0) / max(1, k1 - k0))
    base = n0 - per * k0
    k = max(1, int(round((target - base) / per)))
    text, n = _shrink_to_fit(client, root, model, build, k, target)
    return text, n


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-07 MoE benchmark matrix")
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="qwen3.5-35b-a3b-exl3")
    ap.add_argument("--inputs", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--outputs", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--patterns", nargs="+",
                    default=["real", "uniform", "skewed"],
                    choices=["real", "uniform", "skewed"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--git-revision", default=None)
    ap.add_argument("--artifact", default=None,
                    help="_C.so sha256 prefix (r6/r7 attribution)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600.0)
    root = args.base_url[:-3].rstrip("/") if args.base_url.endswith("/v1") else args.base_url

    if args.git_revision is None:
        import subprocess
        try:
            args.git_revision = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                text=True).strip()
        except Exception:
            args.git_revision = "unknown"
    if args.artifact is None:
        sha = ROOT / "logs/moe-preflight/esimd-build-moegrouped-r7/artifact.sha256"
        args.artifact = sha.read_text().split()[0][:8] if sha.is_file() else "unknown"

    cells = [(i, o, p) for p in args.patterns for i in args.inputs for o in args.outputs]
    print(f"P4-07 matrix: {len(cells)} cells "
          f"(inputs={args.inputs}, outputs={args.outputs}, patterns={args.patterns})")
    print(f"artifact={args.artifact} git={args.git_revision} label={args.label!r}")

    rows: list[dict] = []
    write_header = not CSV.exists() or CSV.stat().st_size == 0
    for idx, (inp, out, pattern) in enumerate(cells, 1):
        tag = f"[{idx}/{len(cells)}] in={inp} out={out} pattern={pattern}"
        note = ""
        result = "ok"
        # Context envelope: prompt + output must fit max_model_len, so a
        # nominal 2048-input cell runs at (2048 - out) input tokens — the
        # largest prompt that can still generate `out` tokens (max-context
        # decode). A small margin absorbs tokenizer-calibration error (the
        # calibrated prompt can land a token or two over target).
        target = min(inp, CONTEXT_LIMIT - out - CONTEXT_MARGIN)
        if target < inp:
            note = (f"input target {inp} capped to {target} "
                    f"(context {CONTEXT_LIMIT} - output {out} - margin "
                    f"{CONTEXT_MARGIN})")
        prompt, in_tok = build_prompt(client, root, args.model, pattern, target)
        t_start_cell = time.perf_counter()
        th0 = thermal_mC()
        g0, m0 = gpu_mem(), meminfo()

        if in_tok + out > CONTEXT_LIMIT:
            # Defensive: calibration overshoot. Trim is not attempted here;
            # the server's 400 is recorded as a skip row.
            note = (note + "; " if note else "") + \
                (f"input+output ({in_tok}+{out}) exceeds context "
                 f"{CONTEXT_LIMIT}; server rejected")
        try:
            results = [one_request(client, args.model, prompt, out)
                       for _ in range(args.runs)]
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
            print(f"{tag}: SKIP ({exc!r})")
            rows.append(_row(args, pattern, inp, in_tok, out, None, note or "error",
                             "skip", th0, th0, g0, m0, time.perf_counter() - t_start_cell))
            continue

        th1 = thermal_mC()
        g1, m1 = gpu_mem(), meminfo()
        ttfts = [r["ttft"] for r in results]
        totals = [r["total"] for r in results]
        decodes = [r["decode_tps"] for r in results]
        out_toks = sum(r["out_tokens"] for r in results)
        wall = time.perf_counter() - t_start_cell
        prefill = in_tok / statistics.median(ttfts) if ttfts else 0.0
        decode_med = statistics.median(decodes)
        bw_gbps = decode_med * BYTES_PER_TOKEN / 1e9
        eff = 100.0 * bw_gbps / ROOFLINE_GBPS
        if out_toks and out_toks < out * len(results):
            result = "truncated"
            note = (note + "; " if note else "") + \
                f"generated {out_toks}/{out * len(results)} tokens (context limit)"
        elif note:
            result = "ok-caveat"

        row = _row(args, pattern, inp, in_tok, out, {
            "ttft_s": statistics.median(ttfts),
            "prefill_tps": prefill,
            "decode_tps": decode_med,
            "total_latency_s": statistics.median(totals),
            "bandwidth_gbps": bw_gbps,
            "bandwidth_eff_pct": eff,
            "peak_gpu_gb": g1.get("GPUActive", 0) / 2**30,
            "swap_used_gb": (m1["SwapTotal"] - m1["SwapFree"]) / 1048576,
        }, note, result, th0, th1, g0, m0, wall)
        rows.append(row)
        print(f"{tag}: ttft={row['ttft_s']:.2f}s decode={row['decode_tps']:.2f} tok/s "
              f"prefill={row['prefill_tps']:.1f} bw={row['bandwidth_gbps']:.1f} GB/s "
              f"({row['bandwidth_eff_pct']:.1f}%) result={result}"
              + (f" note={note}" if note else ""), flush=True)

    if not rows:
        print("no cells executed")
        return 1
    with CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended {len(rows)} rows to {CSV}")

    ok = [r for r in rows if r["result"].startswith("ok")]
    if ok:
        print(f"decode tok/s by cell (ok rows): {len(ok)}/{len(rows)}")
        best = max(ok, key=lambda r: r["decode_tps"])
        print(f"  best decode: {best['decode_tps']:.2f} tok/s "
              f"(in={best['input_tokens']} out={best['output_tokens']} "
              f"pattern={best['routing_pattern']})")
    return 0


def _row(args, pattern, inp, in_tok, out, metrics, note, result,
         th0, th1, g0, m0, wall) -> dict:
    m = metrics or {}
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": args.git_revision,
        "artifact_sha": args.artifact,
        "model_revision": "3.09bpw",
        "backend": "exl3 grouped (P4-02) Arc 140V",
        "context_limit": CONTEXT_LIMIT,
        "routing_pattern": pattern,
        "input_pattern": "calibrated prompt",
        "input_tokens": in_tok,
        "output_tokens": out,
        "ttft_s": round(m["ttft_s"], 3) if m else "",
        "prefill_tps": round(m["prefill_tps"], 2) if m else "",
        "decode_tps": round(m["decode_tps"], 2) if m else "",
        "total_latency_s": round(m["total_latency_s"], 3) if m else "",
        "bandwidth_gbps": round(m["bandwidth_gbps"], 2) if m else "",
        "bandwidth_eff_pct": round(m["bandwidth_eff_pct"], 2) if m else "",
        "peak_gpu_gb": round(m["peak_gpu_gb"], 2) if m else
                       round(g0.get("GPUActive", 0) / 2**30, 2),
        "swap_used_gb": round(m["swap_used_gb"], 2) if m else
                        round((m0["SwapTotal"] - m0["SwapFree"]) / 1048576, 2),
        "thermal_start_c": round(th0 / 1000.0, 1) if th0 else "",
        "thermal_end_c": round(th1 / 1000.0, 1) if th1 else "",
        "result": result,
        "notes": (note + ("; " if note else "") +
                  (f"label={args.label}; " if args.label else "") +
                  f"wall={wall:.1f}s; streamed, temperature=0, runs={args.runs}"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
