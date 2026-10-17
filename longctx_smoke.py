#!/usr/bin/env python3
"""longctx_smoke.py — near-limit context fit + retrieval check on vllm-exl3-b60.

Builds a ~70k-token document (repo's own record files, repeated), plants a
needle code at ~60% depth, asks for it back. Two passes: cold, then warm
(prefix caching on). Verdict = code present in reply + prefill/decode times.
Stdlib only.
"""
import json, time, urllib.request

BASE = "http://127.0.0.1:8001"
MODEL = "qwen3.8-27b-exl3"
NEEDLE = "BLUEBERRY-7429"
TARGET = 85000
HARD_CAP = 88000  # stay clear of max_model_len=90000

def post(path, obj, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return time.monotonic() - t0, json.loads(body)

def ntok(text):
    _, r = post("/tokenize", {"model": MODEL, "prompt": text})
    return r["count"]

parts = []
for f in ("REPRODUCIBILITY.md", "tasks.md", "serve_exl3.sh", "compare_tp2.sh"):
    try:
        parts.append(open("/mnt/usb/vllm.xpu/" + f).read())
    except OSError:
        pass
base = "\n\n".join(parts)
n1 = ntok(base)
reps = max(1, int(TARGET / n1) + 1)
doc = ("\n\n" + base) * reps
while ntok(doc) > HARD_CAP and reps > 1:
    reps -= 1
    doc = ("\n\n" + base) * reps
insert_at = int(len(doc) * 0.6)
doc = doc[:insert_at] + f"\n\nMEMO: The secret code is {NEEDLE}.\n\n" + doc[insert_at:]
n = ntok(doc)
print(f"doc: {len(doc)/1024:.0f} KiB chars -> {n} tokens (target ~{TARGET})")
question = "\n\nQuestion: What is the secret code in the memo above? Reply with only the code."
prompt = doc + question

for tag in ("cold", "warm"):
    dt, r = post("/v1/completions", {"model": MODEL, "prompt": prompt,
                                     "max_tokens": 256, "temperature": 0})
    ch = r["choices"][0]
    text = ch.get("text") or ""
    ok = NEEDLE in text or NEEDLE in json.dumps(r)
    prefill_rate = n / dt
    print(f"[{tag}] total {dt:.1f}s for {n}tok prompt "
          f"(~{prefill_rate:.0f} tok/s incl. prefill) needle_found={ok}")
    print(f"[{tag}] finish={ch.get('finish_reason')} "
          f"usage={r.get('usage')} reply={text.strip()[:120]!r}")
    rc = ch.get("reasoning_content") or r["choices"][0].get("reasoning")
    if rc:
        print(f"[{tag}] reasoning[:200]={str(rc)[:200]!r}")
