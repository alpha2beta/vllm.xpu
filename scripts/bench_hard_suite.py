#!/usr/bin/env python3
"""Challenging 10-Task Hard Benchmark Suite for distinguishing sub-4bpw models.

Evaluates 4 core categories:
  1. Long-Context Deep Needle & Multi-Hop Synthesis (~12,000 - 14,000 tokens)
  2. Olympiad & Hard Quantitative Logic (Modular Arithmetic CRT, Bounded Combinatorics)
  3. Algorithmic Implementation (LeetCode Hard: Trapping Rain Water, LRU Cache, Min Window Substring)
  4. Complex Multi-Constraint Satisfaction & Deductive Logic (5-Floor Logic, 4-Rule Multi-Constraint, Epistemic Trap)

Compatible with OpenAI API endpoints (vLLM and llama-server).
Evaluates correctness via exact numeric verification, executable Python unit tests, and constraint parsers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Dict, List, Tuple
from openai import OpenAI


def generate_long_context_documents() -> Tuple[str, str, str]:
    """Generates ~12,000 tokens of realistic synthetic enterprise infrastructure logs

    with two planted needles at ~25% depth and ~75% depth for multi-hop synthesis.
    """
    records = []
    
    # Needle 1: Planted at record ~35 (~25% depth)
    # Server phoenix-7 migrated to hyperion-9 by operator code-name OSPREY
    # Needle 2: Planted at record ~115 (~75% depth)
    # Operator OSPREY holds cryptographic key-hash 9f8a-c4e1-22b0
    
    total_records = 120
    for i in range(1, total_records + 1):
        hour = (i // 5) % 24
        minute = (i * 3) % 60
        sec = (i * 7) % 60
        ms = (i * 13) % 1000
        ts = f"2026-10-08T{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}Z"
        
        if i == 30:
            records.append(
                f"[{ts}] [CRITICAL-AUDIT] [infra-migrator] Event-ID: EVT-94821 | Source: dc-west-zone-2 | "
                f"ACTION: Server phoenix-7 was migrated to cluster hyperion-9 by operator code-name OSPREY. "
                f"Verification checksum: 0x44a1f9e; status: COMMITTED; authorization: DIRECTIVE-ALPHA-7."
            )
        elif i == 90:
            records.append(
                f"[{ts}] [SECURITY-KEYRING] [auth-vault-primary] Event-ID: EVT-39912 | Subsystem: credential-ledger | "
                f"IDENTITY: Operator code-name OSPREY holds cryptographic key-hash 9f8a-c4e1-22b0. "
                f"Clearance level: OMEGA; revoked: FALSE; last-rotated: 2026-09-15."
            )
        elif i == 60:
            # Distractor amendment needle
            records.append(
                f"[{ts}] [FINANCE-GOVERNANCE] [budget-controller] Event-ID: EVT-77215 | Scope: project-chimera | "
                f"NOTICE: Project Chimera-Alpha initial budget of $4,500,000 was officially amended and capped at $3,180,000 "
                f"by executive order. All subsequent disbursements must reflect final approved budget of $3,180,000."
            )
        else:
            # Background enterprise logs
            node_id = f"worker-node-{((i * 17) % 32) + 1:02d}"
            subsystem = ["memory_arbiter", "storage_tier", "net_mesh", "compute_worker", "replication_v2"][i % 5]
            lat = 4.2 + (i % 15) * 1.8
            tput = 400 + (i % 23) * 35
            records.append(
                f"[{ts}] [TELEMETRY] [{subsystem}] Node: {node_id} | Session-ID: sess-{i:05d}-xyz | "
                f"METRIC: latency={lat:.1f}ms, throughput={tput}req/s, cache_hit_ratio={(85 + (i % 14)):.1f}%, "
                f"gc_pause={(0.8 + (i % 7)*0.4):.2f}ms. Heartbeat acknowledged by orchestrator."
            )
            
    full_text = "\n".join(records)
    needle1_key = "9f8a-c4e1-22b0"
    needle2_val = "$3,180,000"
    return full_text, needle1_key, needle2_val


HARD_BENCHMARK_PROMPTS = [
    # -------------------------------------------------------------
    # Category 1: Long-Context Deep Synthesis (12K+ tokens)
    # -------------------------------------------------------------
    {
        "id": "long_ctx_multihop_needle",
        "category": "long_ctx",
        "is_long_ctx": True,
        "eval_type": "contains_word",
        "target": "9f8a-c4e1-22b0",
        "query_instruction": (
            "You are an auditor reviewing the enterprise infrastructure logs above.\n"
            "Question: What was the cryptographic key-hash held by the operator who migrated server phoenix-7?\n"
            "Trace the migration event to find the operator, then locate the operator's cryptographic key-hash in the keyring records.\n"
            "State the final key-hash clearly on the last line as 'Key-hash: X'."
        )
    },
    {
        "id": "long_ctx_distractor_amendment",
        "category": "long_ctx",
        "is_long_ctx": True,
        "eval_type": "contains_word",
        "target": "3,180,000",
        "query_instruction": (
            "You are a financial compliance auditor reviewing the enterprise infrastructure logs above.\n"
            "Question: What was the final approved budget for Project Chimera-Alpha after all executive amendments?\n"
            "Make sure to state the final amended amount (not any superseded initial budget).\n"
            "State the final amount clearly on the last line as 'Final budget: X'."
        )
    },

    # -------------------------------------------------------------
    # Category 2: Olympiad & Hard Quantitative Logic
    # -------------------------------------------------------------
    {
        "id": "math_chinese_remainder",
        "category": "math",
        "prompt": (
            "Find the smallest positive integer x such that:\n"
            "  x = 3 (mod 17)\n"
            "  x = 4 (mod 19)\n"
            "  x = 5 (mod 23)\n"
            "Show your mathematical derivation step-by-step. On the last line, state the final integer clearly as 'x = [number]'."
        ),
        "eval_type": "regex",
        "pattern": r"(?:x\s*=\s*)?3386",
        "expected": "3386"
    },
    {
        "id": "math_bounded_combinatorics",
        "category": "math",
        "prompt": (
            "How many integer solutions exist to the equation:\n"
            "  a + b + c + d = 24\n"
            "subject to the strict constraints:\n"
            "  1 <= a <= 6\n"
            "  2 <= b <= 8\n"
            "  0 <= c <= 10\n"
            "  3 <= d <= 12\n"
            "Compute the exact number of valid integer combinations (a, b, c, d). State the final integer on the last line as 'Total solutions: [number]'."
        ),
        "eval_type": "regex",
        "pattern": r"(?:total solutions:\s*)?301",
        "expected": "301"
    },

    # -------------------------------------------------------------
    # Category 3: Algorithmic Implementation (LeetCode Hard)
    # -------------------------------------------------------------
    {
        "id": "code_trapping_rain_water",
        "category": "code",
        "prompt": (
            "Write a Python function `trap(height: list[int]) -> int` that computes how much water it can trap after raining "
            "over an elevation map. Optimize for O(N) time and O(1) auxiliary space (using two pointers).\n"
            "Wrap your code in a ```python ... ``` block. Do not include example usage or tests."
        ),
        "eval_type": "python_assert",
        "test_code": """
assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6, "Standard elevation test 1"
assert trap([4,2,0,3,2,5]) == 9, "Standard elevation test 2"
assert trap([]) == 0, "Empty list"
assert trap([3]) == 0, "Single element"
assert trap([2, 0, 2]) == 2, "Simple bowl"
assert trap([1, 2, 3, 4, 5]) == 0, "Ascending slopes trap no water"
assert trap([5, 4, 3, 2, 1]) == 0, "Descending slopes trap no water"
assert trap([5, 1, 1, 1, 5]) == 12, "Wide flat bottom"
"""
    },
    {
        "id": "code_lru_cache",
        "category": "code",
        "prompt": (
            "Design and implement an LRU (Least Recently Used) cache in Python as class `LRUCache`.\n"
            "It must support:\n"
            "  - `__init__(self, capacity: int)`\n"
            "  - `get(self, key: int) -> int`: Returns value of key if key exists, otherwise -1. Runs in O(1) average time.\n"
            "  - `put(self, key: int, value: int) -> None`: Updates or inserts value. If capacity is reached, evicts the least recently used key. Runs in O(1) average time.\n"
            "Wrap your code in a ```python ... ``` block."
        ),
        "eval_type": "python_assert",
        "test_code": """
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1, "Failed to retrieve key 1"
cache.put(3, 3) # evicts key 2
assert cache.get(2) == -1, "Key 2 should have been evicted"
cache.put(4, 4) # evicts key 1
assert cache.get(1) == -1, "Key 1 should have been evicted"
assert cache.get(3) == 3, "Key 3 should exist"
assert cache.get(4) == 4, "Key 4 should exist"

# Test updating existing key
c2 = LRUCache(2)
c2.put(2, 1)
c2.put(2, 2)
assert c2.get(2) == 2, "Failed to update existing key"
c2.put(1, 1)
assert c2.get(2) == 2, "Failed to retrieve key 2"
c2.put(4, 1) # evicts key 1 (since key 2 was accessed by get(2))
assert c2.get(2) == 2, "Key 2 was recently accessed, should remain"
assert c2.get(1) == -1, "Key 1 should have been evicted"

# Test capacity 1
c3 = LRUCache(1)
c3.put(10, 100)
assert c3.get(10) == 100
c3.put(20, 200)
assert c3.get(10) == -1
assert c3.get(20) == 200
"""
    },
    {
        "id": "code_min_window_substring",
        "category": "code",
        "prompt": (
            "Write a Python function `min_window(s: str, t: str) -> str` that finds the minimum window substring of `s` "
            "such that every character in `t` (including duplicate occurrences) is included in the window. "
            "If no such substring exists, return empty string `''`. Optimize for O(|s| + |t|) time.\n"
            "Wrap your code in a ```python ... ``` block."
        ),
        "eval_type": "python_assert",
        "test_code": """
assert min_window("ADOBECODEBANC", "ABC") == "BANC", "Classic min window test"
assert min_window("a", "a") == "a", "Single char match"
assert min_window("a", "aa") == "", "Target has more chars than source"
assert min_window("bbaac", "aba") == "baa", "Duplicate letters in target"
assert min_window("aaflslflsldkalskaaa", "aaa") == "aaa", "Multiple duplicates"
assert min_window("xyz", "a") == "", "Disjoint strings"
"""
    },

    # -------------------------------------------------------------
    # Category 4: Complex Multi-Constraint & Logical Deduction
    # -------------------------------------------------------------
    {
        "id": "logic_five_floors",
        "category": "logic_constraint",
        "prompt": (
            "Five colleagues (Alice, Bob, Carol, David, Elena) work on five different floors (floors 1, 2, 3, 4, 5) of an office building.\n"
            "Rules:\n"
            "1. Alice works on a higher floor than Bob.\n"
            "2. Carol works on neither the first floor nor the fifth floor.\n"
            "3. David works exactly two floors above Carol (Floor_David = Floor_Carol + 2).\n"
            "4. Elena does NOT work on floor 4 or floor 5.\n"
            "5. Bob works on floor 3.\n"
            "Determine the exact floor number for all five colleagues (1 to 5). On the last line, state your final answer clearly as:\n"
            "'Floors: Alice=[A], Bob=[B], Carol=[C], David=[D], Elena=[E]'"
        ),
        "eval_type": "regex",
        "pattern": r"Alice\s*=\s*\[?5\]?.*Bob\s*=\s*\[?3\]?.*Carol\s*=\s*\[?2\]?.*David\s*=\s*\[?4\]?.*Elena\s*=\s*\[?1\]?",
        "expected": "Alice=5, Bob=3, Carol=2, David=4, Elena=1"
    },
    {
        "id": "format_multi_constraint_4rules",
        "category": "logic_constraint",
        "prompt": (
            "Write a short response satisfying ALL 4 rules simultaneously:\n"
            "Rule 1: Exactly 3 sentences long.\n"
            "Rule 2: Every sentence must start with either 'Although' or 'Because'.\n"
            "Rule 3: You must NOT use the letter 'z' or 'Z' anywhere in your response.\n"
            "Rule 4: The second sentence must contain the exact phrase 'deep blue ocean'.\n"
            "Output ONLY the 3 sentences, nothing else."
        ),
        "eval_type": "custom_4rules",
        "expected": "3 sentences starting with Although/Because, no letter z, second sentence contains 'deep blue ocean'"
    },
    {
        "id": "trap_sheep_all_but_nine",
        "category": "logic_constraint",
        "prompt": (
            "A farmer has 17 sheep. A sudden illness strikes, and all but 9 die. "
            "How many sheep are still alive? Answer with a single integer on the last line as 'Alive: [number]'."
        ),
        "eval_type": "regex",
        "pattern": r"(?:alive:\s*)?9\b",
        "expected": "9 (all but 9 die means 9 remain alive)"
    }
]


def extract_clean_answer(content: str) -> tuple[str, str]:
    """Separates thinking trace (<think>...</think>) from final answer."""
    think = ""
    answer = content
    if "<think>" in content and "</think>" in content:
        parts = content.split("</think>")
        think = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
    elif "</think>" in content:
        parts = content.split("</think>")
        think = parts[0].strip()
        answer = parts[1].strip()
    return think, answer


def extract_python_code(answer: str) -> str:
    """Extracts python code from an answer, handling markdown blocks if present."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", answer)
    if match:
        return match.group(1).strip()
    return answer.strip()


def evaluate_custom_4rules(answer: str) -> Tuple[bool, str]:
    text = answer.strip()
    # Check no z/Z
    if re.search(r"[zZ]", text):
        return False, "Failed Rule 3: Found letter 'z' or 'Z'"
    
    # Split sentences (by . ! ?)
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(raw_sentences) != 3:
        return False, f"Failed Rule 1: Expected 3 sentences, got {len(raw_sentences)}"
    
    # Check each sentence begins with Although or Because
    for idx, s in enumerate(raw_sentences, 1):
        if not re.match(r"^(?:Although|Because)\b", s, re.IGNORECASE):
            return False, f"Failed Rule 2: Sentence {idx} does not start with Although or Because: {s[:30]!r}"
            
    # Check second sentence contains 'deep blue ocean'
    if "deep blue ocean" not in raw_sentences[1].lower():
        return False, f"Failed Rule 4: Second sentence does not contain 'deep blue ocean': {raw_sentences[1]!r}"
        
    return True, "Passed all 4 rules simultaneously"


def evaluate_hard_item(item: dict, answer: str) -> tuple[bool, str]:
    eval_type = item["eval_type"]
    
    if eval_type == "contains_word":
        target = item["target"].lower()
        if target in answer.lower():
            return True, f"Found target: {item['target']}"
        return False, f"Target {item['target']!r} not found in: {answer[-200:]!r}"

    elif eval_type == "regex":
        lines = [line.strip() for line in answer.strip().split("\n") if line.strip()]
        last_chunk = " ".join(lines[-4:]) if lines else answer
        if re.search(item["pattern"], last_chunk, re.IGNORECASE):
            return True, f"Matched expected pattern: {item['expected']}"
        if re.search(item["pattern"], answer, re.IGNORECASE):
            return True, f"Matched expected pattern in answer: {item['expected']}"
        return False, f"Did not find pattern for {item['expected']} in: {answer[-200:]!r}"

    elif eval_type == "python_assert":
        code = extract_python_code(answer)
        full_code = code + "\n\n" + item["test_code"]
        scope: Dict[str, Any] = {}
        try:
            exec(full_code, scope, scope)
            return True, "All unit tests passed"
        except AssertionError as e:
            return False, f"Assertion failed: {e}"
        except Exception as e:
            return False, f"Execution error: {type(e).__name__}: {e}"

    elif eval_type == "custom_4rules":
        return evaluate_custom_4rules(answer)

    return False, "Unknown evaluation type"


def run_hard_benchmark(base_url: str, model_name: str, out_file: str, max_tokens: int = 2048, skip_long_ctx: bool = False, start_task: int = 1):
    client = OpenAI(base_url=base_url, api_key="dummy")
    print(f"=== Starting Challenging 10-Task Hard Benchmark ===")
    print(f"Target: {model_name} @ {base_url}")
    print(f"Sampling: greedy (temperature=0.0), max_tokens={max_tokens}")
    if skip_long_ctx:
        print("Note: --skip-long-ctx enabled (skipping 12K token prompts)")
    if start_task > 1:
        print(f"Note: Resuming from task {start_task}/10")
    print()

    # Pre-generate long context once
    long_ctx_text, n1_key, n2_val = generate_long_context_documents()
    print(f"Pre-generated synthetic audit log: {len(long_ctx_text)} chars (~12,500 tokens)\n")

    results = []
    category_stats = {"long_ctx": [0, 0], "math": [0, 0], "code": [0, 0], "logic_constraint": [0, 0]}

    # If resuming and out_file exists, load previous successful results
    import os
    if start_task > 1 and os.path.exists(out_file):
        try:
            with open(out_file) as f:
                prev_data = json.load(f)
            prev_results = prev_data.get("results", [])
            for r in prev_results[:start_task - 1]:
                results.append(r)
                cat = r["category"]
                category_stats[cat][1] += 1
                if r.get("passed", False):
                    category_stats[cat][0] += 1
            print(f"Loaded {len(results)} previous task results from {out_file}")
        except Exception as e:
            print(f"Warning: could not load existing {out_file}: {e}")

    for i, item in enumerate(HARD_BENCHMARK_PROMPTS, 1):
        if i < start_task:
            continue

        pid = item["id"]
        cat = item["category"]
        is_long = item.get("is_long_ctx", False)

        if is_long and skip_long_ctx:
            print(f"[{i:02d}/10] ({cat.upper()}) {pid}... SKIPPED (--skip-long-ctx)")
            continue

        if is_long:
            prompt_content = f"{long_ctx_text}\n\n---\n{item['query_instruction']}"
        else:
            prompt_content = item["prompt"]

        print(f"[{i:02d}/10] ({cat.upper()}) {pid}...", end=" ", flush=True)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            dur = time.time() - t0
            msg = resp.choices[0].message
            raw_content = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", "") or ""
            
            # Combine or clean answer
            if raw_content:
                think, answer = extract_clean_answer(raw_content)
                if not think and reasoning_content:
                    think = reasoning_content
            else:
                # If content is empty but reasoning is present
                think = reasoning_content
                answer = raw_content

            passed, reason = evaluate_hard_item(item, answer)
        except Exception as e:
            dur = time.time() - t0
            raw_content = ""
            think = ""
            answer = ""
            passed = False
            reason = f"API Error: {e}"

        status_str = "PASS" if passed else "FAIL"
        print(f"{status_str} ({dur:.1f}s, think_chars={len(think)}, ans_chars={len(answer)})")
        if not passed:
            print(f"       Reason: {reason}")

        category_stats[cat][1] += 1
        if passed:
            category_stats[cat][0] += 1

        results.append({
            "id": pid,
            "category": cat,
            "latency_s": round(dur, 2),
            "think_chars": len(think),
            "answer_chars": len(answer),
            "passed": passed,
            "reason": reason,
            "answer_preview": answer[:300]
        })

    # Summary
    total_passed = sum(v[0] for v in category_stats.values())
    total_items = sum(v[1] for v in category_stats.values())
    
    print("\n" + "="*50)
    print(f"HARD BENCHMARK SUMMARY FOR {model_name}")
    print("="*50)
    for cat, (p, t) in category_stats.items():
        if t > 0:
            print(f"  {cat.capitalize():18s}: {p}/{t} ({p/t*100:.1f}%)")
    if total_items > 0:
        print(f"  Total Score       : {total_passed}/{total_items} ({total_passed/total_items*100:.1f}%)")
    print("="*50)

    summary_data = {
        "model": model_name,
        "base_url": base_url,
        "total_score": f"{total_passed}/{total_items}",
        "pass_rate_pct": round(total_passed / total_items * 100, 1) if total_items > 0 else 0.0,
        "category_stats": {c: f"{p}/{t}" for c, (p, t) in category_stats.items() if t > 0},
        "results": results
    }

    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Detailed output written to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3.8-27b-2.5bpw")
    parser.add_argument("--out", default="results_hard_bench.json")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--skip-long-ctx", action="store_true")
    parser.add_argument("--start-task", type=int, default=1, help="Start from task index (1-based)")
    args = parser.parse_args()

    run_hard_benchmark(args.base_url, args.model, args.out, args.max_tokens, args.skip_long_ctx, args.start_task)

