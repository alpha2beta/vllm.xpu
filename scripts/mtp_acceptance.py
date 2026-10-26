#!/usr/bin/env python3
"""Acceptance + throughput on natural prompts (MTP K=1)."""
import json, time, os
import mtp_smoke as S

TAG = os.environ.get("MTP_TAG", "mtp-k1")

PROMPTS = [
    ("explain",  "Explain in plain language how a refrigerator keeps food cold.\n\n"),
    ("history",  "Write a paragraph about the main causes of the French Revolution.\n\n"),
    ("howto",    "Describe how a bicycle derailleur moves the chain between gears.\n\n"),
    ("story",    "A detective arrives at a lighthouse on a stormy night. Summarize what she finds.\n\n"),
    ("code",     "Here is a Python function that merges two sorted lists:\n\ndef merge(a, b):\n"),
    ("math",     "A train leaves at 09:15 and travels 240 km at 80 km/h. Show the arrival time.\n\n"),
]

def main():
    m0 = S.metrics()
    rows = []
    for name, p in PROMPTS:
        r = S.complete(p, 96)
        r["case"] = name
        rows.append(r)
        print(f"{name:8s} tok={r['tokens']:3d} ttft={r['ttft']:.2f}s decode={r['decode_tps'] or 0:5.2f} tok/s  {r['text'][:70]!r}")
    m1 = S.metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1)}
    drafts = d.get("spec_decode_num_draft_tokens_total", 0)
    acc = d.get("spec_decode_num_accepted_tokens_total", 0)
    steps = d.get("spec_decode_num_drafts_total", 0)
    print(f"\ndrafts={drafts:.0f} steps={steps:.0f} accepted={acc:.0f}")
    if drafts:
        print(f"acceptance={acc/drafts:.3f}  mean accepted/step={acc/steps:.3f}  tok/step={1+acc/steps:.3f}")
    tot = sum(r["tokens"] for r in rows); wall = sum(r["total"] for r in rows)
    dec = sum((r["tokens"] - 1) for r in rows) / sum(r["total"] - r["ttft"] for r in rows)
    print(f"aggregate: {tot} tokens / {wall:.1f}s wall, decode {dec:.2f} tok/s")
    json.dump({"rows": rows, "spec_delta": d, "decode_tps": dec},
              open(f"logs/moe-p3/sage-{TAG}-natural.json", "w"), indent=2)
    print(f"wrote logs/moe-p3/sage-{TAG}-natural.json")

if __name__ == "__main__":
    main()
