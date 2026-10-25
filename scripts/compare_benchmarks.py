#!/usr/bin/env python3
"""Compare two EvalPlus HumanEval evaluation result JSON files."""

import json
import sys
from pathlib import Path


def load_results(path_str):
    p = Path(path_str)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    eval_dict = data.get("eval", {})
    base_passes = set()
    plus_passes = set()
    total = len(eval_dict)
    for task_id, samples in eval_dict.items():
        if samples and samples[0].get("base_status") == "pass":
            base_passes.add(task_id)
        if samples and samples[0].get("plus_status") == "pass":
            plus_passes.add(task_id)
    return {
        "path": p.name,
        "total": total,
        "base_passes": base_passes,
        "plus_passes": plus_passes,
        "base_score": (len(base_passes) / total * 100.0) if total else 0.0,
        "plus_score": (len(plus_passes) / total * 100.0) if total else 0.0,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_benchmarks.py <eval_results_a.json> <eval_results_b.json>")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    res_a = load_results(file_a)
    res_b = load_results(file_b)

    if not res_a:
        print(f"Error: Could not read {file_a}")
        sys.exit(1)
    if not res_b:
        print(f"Error: Could not read {file_b}")
        sys.exit(1)

    name_a = sys.argv[3] if len(sys.argv) > 3 else "Option A (Thinking OFF)"
    name_b = sys.argv[4] if len(sys.argv) > 4 else "Option B (Thinking ON - 4096)"

    print("=" * 80)
    print("                     EVALPLUS BENCHMARK COMPARISON                     ")
    print("=" * 80)
    print(f"{'Metric':<35} | {name_a:<20} | {name_b:<20}")
    print("-" * 80)
    print(f"{'Total Evaluated Tasks':<35} | {res_a['total']:<20} | {res_b['total']:<20}")
    print(f"{'HumanEval (Base) Pass@1 (%)':<35} | {res_a['base_score']:<20.2f} | {res_b['base_score']:<20.2f}")
    print(f"{'HumanEval (Base) Passed Count':<35} | {len(res_a['base_passes']):<20} | {len(res_b['base_passes']):<20}")
    print(f"{'HumanEval+ (Plus) Pass@1 (%)':<35} | {res_a['plus_score']:<20.2f} | {res_b['plus_score']:<20.2f}")
    print(f"{'HumanEval+ (Plus) Passed Count':<35} | {len(res_a['plus_passes']):<20} | {len(res_b['plus_passes']):<20}")
    print("=" * 80)

    # Discrepancies
    only_a_base = sorted(res_a["base_passes"] - res_b["base_passes"], key=lambda x: int(x.split("/")[1]))
    only_b_base = sorted(res_b["base_passes"] - res_a["base_passes"], key=lambda x: int(x.split("/")[1]))

    print(f"\nTasks passed ONLY in {name_a} ({len(only_a_base)}):")
    print(", ".join(only_a_base) if only_a_base else "None")

    print(f"\nTasks passed ONLY in {name_b} ({len(only_b_base)}):")
    print(", ".join(only_b_base) if only_b_base else "None")
    print("=" * 80)


if __name__ == "__main__":
    main()
