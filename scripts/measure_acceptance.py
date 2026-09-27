#!/usr/bin/env python3
"""11.2.3: per-prompt MTP acceptance rate via /metrics deltas (K=2)."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.6-35B-A3B-MXFP4"
UNIT = "alpha beta gamma delta epsilon zeta eta theta "

PROMPTS = [
    ("factual", "What is the capital of Australia? Answer in one sentence.", 64),
    ("code", "Write a Python function is_palindrome(s: str) -> bool that ignores case "
     "and non-alphanumeric characters. Code only.", 128),
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
     "the ball. How much does the ball cost? Think step by step, then give the answer.", 128),
    ("instruction", "List exactly three benefits of unit testing as a numbered list. "
     "No introduction or closing remarks.", 128),
    ("long-context", None, 64),  # built below (~3000 tok filler + retrieval Q)
]


def metrics():
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            line = line.strip()
            if not line.startswith("vllm:spec_decode") or not line.endswith("} 0.0") is False:
                pass
            if line.startswith("vllm:spec_decode") and line.split("{")[0].endswith("_total"):
                name = line.split("{")[0]
                val = float(line.rsplit(None, 1)[-1])
                if "per_pos" in name:
                    pos = int(line.split('position="')[1].split('"')[0])
                    out.setdefault("per_pos", {})[pos] = out.get("per_pos", {}).get(pos, 0) + val
                elif "num_drafts" in name:
                    out["drafts"] = out.get("drafts", 0) + val
                elif "num_draft_tokens" in name:
                    out["draft_tokens"] = out.get("draft_tokens", 0) + val
                elif "num_accepted_tokens" in name and "per_pos" not in name:
                    out["accepted"] = out.get("accepted", 0) + val
    return out


def chat(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    c = d["choices"][0]
    txt = c["message"]["content"] or ""
    return txt, c["finish_reason"], dt


def main():
    filler = ("Read the following block and then answer the question.\n\n" + UNIT * 330
              + "\n\nQuestion: what is the first word of the block? One word only.")
    rows = []
    for kind, prompt, mt in PROMPTS:
        if prompt is None:
            prompt = filler
        m0 = metrics()
        txt, fin, dt = chat(prompt, mt)
        m1 = metrics()
        dd = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("drafts", "draft_tokens", "accepted")}
        p0 = m0.get("per_pos", {})
        p1 = m1.get("per_pos", {})
        per_pos = {p: p1.get(p, 0) - p0.get(p, 0) for p in set(p0) | set(p1)}
        alpha = dd["accepted"] / dd["draft_tokens"] if dd["draft_tokens"] else float("nan")
        # per-position acceptance: pos1 cond. on pos0 accepted (K=2 chain)
        a0 = per_pos.get(0, 0)
        n1_given = dd["draft_tokens"] / 2 if dd["draft_tokens"] else 0  # ~2 drafts/step
        rows.append((kind, dd, per_pos, alpha, dt, len(txt), fin))
        print(f"{kind}: drafts={dd['drafts']:.0f} draft_tok={dd['draft_tokens']:.0f} "
              f"acc={dd['accepted']:.0f} per_pos={per_pos} alpha={alpha:.3f} "
              f"t={dt:.1f}s out_chars={len(txt)} finish={fin}", flush=True)
    with open("logs/bench-tiel-mtp-acceptance.log", "w") as f:
        for kind, dd, per_pos, alpha, dt, nch, fin in rows:
            f.write(f"{kind}: {dd} per_pos={per_pos} alpha={alpha:.4f} t={dt:.2f}s "
                    f"chars={nch} finish={fin}\n")
    print("wrote logs/bench-tiel-mtp-acceptance.log")


if __name__ == "__main__":
    main()
