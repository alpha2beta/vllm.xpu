#!/usr/bin/env python3
"""MTP smoke + acceptance probe for the EXL3 MoE server on :8002."""
import json, sys, time
import urllib.request

BASE = "http://127.0.0.1:8002"

def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_") and "_total{" in line:
            name = line.split("{")[0].split(":")[1]
            out[name] = float(line.rsplit(" ", 1)[1])
    return out

def complete(prompt, max_tokens, model="sage-a1.5-exl3-mtp"):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "seed": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; chunks = []; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = ch.get("text") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                chunks.append(piece)
    total = time.time() - t0
    text = "".join(chunks)
    n = (usage or {}).get("completion_tokens") or len(chunks)
    return {"ttft": ttft, "total": total, "tokens": n,
            "decode_tps": (n - 1) / (total - ttft) if ttft and total > ttft and n > 1 else None,
            "text": text}

FILLER = ("The history of computing spans mechanical calculators, vacuum-tube machines, "
          "transistors, integrated circuits, and parallel accelerators. ")

def prompt_of(n_tokens):
    # ~16 tokens per sentence; good enough for a length sweep
    reps = max(1, n_tokens // 16)
    return (FILLER * reps)[: n_tokens * 4]

def main():
    m0 = metrics()
    cases = [("short", prompt_of(32), 64), ("mid", prompt_of(256), 64), ("long", prompt_of(896), 64)]
    results = []
    for name, prompt, mt in cases:
        try:
            r = complete(prompt, mt)
        except Exception as e:
            print(f"{name:6s} ERROR {type(e).__name__}: {e}")
            return 1
        r["case"] = name
        results.append(r)
        print(f"{name:6s} tokens={r['tokens']:3d} ttft={r['ttft']:.2f}s total={r['total']:.2f}s "
              f"decode={r['decode_tps'] or 0:.2f} tok/s")
        print(f"       first 120 chars: {r['text'][:120]!r}")
    m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1)}
    drafts = d.get("spec_decode_num_draft_tokens_total", 0)
    accepted = d.get("spec_decode_num_accepted_tokens_total", 0)
    print("\nspec metrics delta:")
    for k in sorted(d):
        print(f"  {k}: {d[k]:.0f}")
    if drafts:
        print(f"acceptance: {accepted:.0f}/{drafts:.0f} = {accepted/drafts:.3f} "
              f"(mean accepted/step = {accepted/d.get('spec_decode_num_drafts_total',0):.3f})")
    out = {"results": results, "spec_delta": d}
    with open("logs/moe-p3/sage-mtp-k1-smoke.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote logs/moe-p3/sage-mtp-k1-smoke.json")

if __name__ == "__main__":
    sys.exit(main())
