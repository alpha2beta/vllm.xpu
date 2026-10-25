#!/usr/bin/env python3
"""P2-10: standardized Phase 2 test runner.

Runs every Phase 2 suite in a subprocess, records wall time, pass/fail, case
counts and the worst absolute deviation reported by each suite, and writes
``logs/moe-p2/phase2-summary.json``.

    .venv/bin/python scripts/run_phase2_tests.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, script, json artifact it writes)
SUITES = [
    ("P2-01/04/07 MoE block", "tests/test_moe_block.py",
     "logs/moe-p2/moe-block-eval.json"),
    ("P2-02/03/06 routing + edge cases", "tests/test_moe_routing.py",
     "logs/moe-p2/routing-eval.json"),
    ("P2-05a fixed routing", "tests/test_fixed_routing.py",
     "logs/moe-p2/fixed-routing.json"),
    ("P2-05b free routing", "tests/test_free_routing.py",
     "logs/moe-p2/free-routing.json"),
    ("P2-08 intermediate stages", "tests/test_intermediate_probe.py",
     "logs/moe-p2/intermediate-stage-validation.json"),
    ("P2-09 hybrid decoder layer", "tests/test_hybrid_layer.py",
     "logs/moe-p2/full-layer-eval.json"),
]


def collect_metrics(path: Path) -> dict:
    if not path.is_file():
        return {"available": False}
    data = json.loads(path.read_text())
    worst = None
    n_cases = 0

    def walk(obj):
        nonlocal worst, n_cases
        if isinstance(obj, dict):
            if isinstance(obj.get("max_abs"), (int, float)):
                worst = obj["max_abs"] if worst is None else max(worst, obj["max_abs"])
            if "case" in obj and "pass" in obj:
                n_cases += 1
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    return {
        "available": True,
        "task": data.get("task"),
        "pass": data.get("pass"),
        "worst_max_abs": worst,
        "case_records": n_cases,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "logs/moe-p2/phase2-summary.json"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filter on suite labels")
    args = ap.parse_args()

    results = []
    t_all = time.perf_counter()
    for label, script, artifact in SUITES:
        if args.only and not any(s in label for s in args.only):
            continue
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT, capture_output=True, text=True,
        )
        seconds = time.perf_counter() - t0
        metrics = collect_metrics(ROOT / artifact)
        rec = {
            "suite": label,
            "script": script,
            "artifact": artifact,
            "exit_code": proc.returncode,
            "seconds": round(seconds, 2),
            "pass": proc.returncode == 0 and metrics.get("pass") is not False,
            **metrics,
        }
        if proc.returncode != 0:
            rec["stderr_tail"] = proc.stderr[-1500:]
        results.append(rec)
        print(f"{'PASS' if rec['pass'] else 'FAIL'}  {seconds:7.2f}s  {label}")

    summary = {
        "task": "P2-10",
        "pass": all(r["pass"] for r in results),
        "suites_run": len(results),
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "worst_max_abs": max(
            (r["worst_max_abs"] for r in results
             if isinstance(r.get("worst_max_abs"), (int, float))),
            default=None,
        ),
        "tolerance_T2_max_abs": 0.08,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\npass={summary['pass']} worst_max_abs={summary['worst_max_abs']} "
          f"total={summary['total_seconds']}s")
    print(f"wrote {out}")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
