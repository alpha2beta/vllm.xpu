#!/usr/bin/env python3
"""16-Prompt Task Benchmark for comparing ~2-bit quantized models.

Evaluates 4 core categories:
  1. Multi-Step Math & Quantitative Logic (4 prompts)
  2. Coding & Algorithmic Implementation (4 prompts with unit-test assertions)
  3. Strict Instruction Following & Constraints (4 prompts)
  4. Factual Knowledge & Trap Detection (4 prompts)

Separates <think>...</think> reasoning traces from the final answer and evaluates
correctness automatically.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Dict, List
from openai import OpenAI

BENCHMARK_PROMPTS = [
    # -------------------------------------------------------------
    # Category 1: Multi-Step Math & Quantitative Logic
    # -------------------------------------------------------------
    {
        "id": "math_bridge",
        "category": "math",
        "prompt": "Four hikers reach a bridge at night with one flashlight. Crossing times: 1, 2, 5, and 10 minutes. At most two people can cross together, moving at the slower hiker's pace. The flashlight must be carried back. What is the minimum total time (in minutes) for all four to cross? State the final number clearly on the last line as 'Total time: X minutes'.",
        "eval_type": "regex",
        "pattern": r"(?:total time:\s*|minimum(?:\s+total)?\s+time:\s*)?17(?:\s+minutes)?",
        "expected": "17 minutes"
    },
    {
        "id": "math_lcm",
        "category": "math",
        "prompt": "A store sells apples in packs of 6 and oranges in packs of 8. A customer wants to buy the exact same number of apples and oranges. What is the smallest positive total number of fruit (apples + oranges combined) the customer must buy? State the final answer on the last line as 'Total fruit: X'.",
        "eval_type": "regex",
        "pattern": r"(?:total fruit:\s*)?48",
        "expected": "48 (24 apples + 24 oranges)"
    },
    {
        "id": "math_speed",
        "category": "math",
        "prompt": "Train A leaves a station traveling at 60 mph. Train B leaves the same station 1 hour later traveling in the same direction on a parallel track at 80 mph. How many hours after Train B departs will Train B catch up to Train A? State the final answer on the last line as 'Hours: X'.",
        "eval_type": "regex",
        "pattern": r"(?:hours:\s*)?3(?:\s+hours)?",
        "expected": "3 hours"
    },
    {
        "id": "math_probability",
        "category": "math",
        "prompt": "A bag contains 3 red balls and 5 blue balls. If two balls are drawn at random without replacement, what is the exact probability (in fraction form in lowest terms, like P/Q) that both balls are red? State the final fraction on the last line as 'Probability: P/Q'.",
        "eval_type": "regex",
        "pattern": r"(?:probability:\s*)?3/28",
        "expected": "3/28"
    },

    # -------------------------------------------------------------
    # Category 2: Coding & Algorithmic Implementation (Assertion-Checked)
    # -------------------------------------------------------------
    {
        "id": "code_palindrome",
        "category": "code",
        "prompt": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is a palindrome, ignoring non-alphanumeric characters and case. Output ONLY the python function implementation. Do not include markdown fences or usage examples.",
        "eval_type": "python_assert",
        "test_code": """
assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("race a car") is False
assert is_palindrome("") is True
assert is_palindrome("0P") is False
assert is_palindrome("Was it a car or a cat I saw?") is True
"""
    },
    {
        "id": "code_merge_intervals",
        "category": "code",
        "prompt": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals and returns the merged list sorted by start time. Output ONLY the python function implementation without markdown code fences or explanatory text.",
        "eval_type": "python_assert",
        "test_code": """
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([]) == []
assert merge_intervals([[1,4],[0,4]]) == [[0,4]]
assert merge_intervals([[1,4],[2,3]]) == [[1,4]]
"""
    },
    {
        "id": "code_two_sum",
        "category": "code",
        "prompt": "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that finds two distinct indices i and j such that nums[i] + nums[j] == target. Output ONLY the python function implementation without markdown code fences or commentary.",
        "eval_type": "python_assert",
        "test_code": """
assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]
assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]
assert sorted(two_sum([3, 3], 6)) == [0, 1]
"""
    },
    {
        "id": "code_flatten_dict",
        "category": "code",
        "prompt": "Write a Python function `flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict` that flattens a nested dictionary. Keys should be concatenated with `sep`. Output ONLY the python function implementation without markdown code fences or commentary.",
        "eval_type": "python_assert",
        "test_code": """
d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
expected = {'a': 1, 'b.c': 2, 'b.d.e': 3}
assert flatten_dict(d) == expected
assert flatten_dict({}) == {}
"""
    },

    # -------------------------------------------------------------
    # Category 3: Strict Instruction Following & Formatting Constraints
    # -------------------------------------------------------------
    {
        "id": "format_json_only",
        "category": "instruction",
        "prompt": 'Extract the user information from this sentence: "Marcus Aurelius was born on April 26, 121 in Rome." Output ONLY a raw valid JSON object with keys "name", "birth_date", and "birth_place". Absolutely NO markdown code fences (no ```json), NO introductory words, NO commentary.',
        "eval_type": "json_schema",
        "required_keys": ["name", "birth_date", "birth_place"]
    },
    {
        "id": "format_reverse_capitals",
        "category": "instruction",
        "prompt": "List exactly 5 distinct European capital cities in strict reverse alphabetical order (Z to A). Format your output strictly as a numbered list (1. to 5.) with only the city names. No other text, no explanations.",
        "eval_type": "reverse_capitals",
        "expected_count": 5
    },
    {
        "id": "format_no_letter_e",
        "category": "instruction",
        "prompt": "Write a complete English sentence describing the ocean without using the letter 'e' (or 'E') anywhere in the sentence. Output only the sentence.",
        "eval_type": "no_letter_e"
    },
    {
        "id": "format_word_count",
        "category": "instruction",
        "prompt": "Summarize what photosynthesis is in exactly between 15 and 25 words (inclusive). Do not include any title, prefix, or explanation. Output only the summary sentence.",
        "eval_type": "word_count",
        "min_words": 15,
        "max_words": 25
    },

    # -------------------------------------------------------------
    # Category 4: Factual Knowledge & Premise Trap Detection
    # -------------------------------------------------------------
    {
        "id": "fact_capital_australia",
        "category": "fact_trap",
        "prompt": "What is the capital of Australia? Answer in one short sentence.",
        "eval_type": "contains_word",
        "target": "Canberra"
    },
    {
        "id": "trap_us_president_1650",
        "category": "fact_trap",
        "prompt": "Who was the President of the United States in the year 1650?",
        "eval_type": "trap_president",
        "expected": "Rejects premise (United States did not exist in 1650 / office did not exist)"
    },
    {
        "id": "trap_steel_vs_feathers",
        "category": "fact_trap",
        "prompt": "Which weighs more: two kilograms of steel or two kilograms of feathers? Answer in one sentence.",
        "eval_type": "trap_equal_weight",
        "expected": "Both weigh the same / equal"
    },
    {
        "id": "reasoning_shortest_person",
        "category": "fact_trap",
        "prompt": "Alice is taller than Bob. Charlie is shorter than Bob. David is taller than Alice. Who is the shortest person? Give only the name of the shortest person.",
        "eval_type": "contains_word",
        "target": "Charlie"
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


def evaluate_item(item: dict, answer: str) -> tuple[bool, str]:
    eval_type = item["eval_type"]
    
    if eval_type == "regex":
        # Search the last 3 lines or entire answer
        lines = [line.strip() for line in answer.strip().split("\n") if line.strip()]
        last_chunk = " ".join(lines[-3:]) if lines else answer
        if re.search(item["pattern"], last_chunk, re.IGNORECASE):
            return True, f"Matched expected pattern: {item['expected']}"
        # Also check whole answer as fallback
        if re.search(item["pattern"], answer, re.IGNORECASE):
            return True, f"Matched expected pattern in answer: {item['expected']}"
        return False, f"Did not find pattern for {item['expected']} in: {answer[-150:]!r}"

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

    elif eval_type == "json_schema":
        # Check if raw JSON or markdown wrapped
        clean = answer.strip()
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean)
        markdown_wrapped = False
        if match:
            clean = match.group(1).strip()
            markdown_wrapped = True
        try:
            data = json.loads(clean)
            if not isinstance(data, dict):
                return False, f"Parsed JSON is not a dict: {type(data).__name__}"
            for k in item["required_keys"]:
                if k not in data:
                    return False, f"Missing required key: {k}"
            if markdown_wrapped:
                return True, "Valid JSON (warning: wrapped in markdown code fence despite constraint)"
            return True, "Strict valid raw JSON with all required keys"
        except Exception as e:
            return False, f"JSON parse error: {e}"

    elif eval_type == "reverse_capitals":
        lines = [re.sub(r"^\d+[\.\)]\s*", "", line).strip() for line in answer.split("\n") if line.strip()]
        # Filter lines that look like city names
        cities = [l for l in lines if l and not l.lower().startswith("here")]
        if len(cities) != item["expected_count"]:
            return False, f"Expected {item['expected_count']} cities, got {len(cities)}: {cities}"
        # Check reverse alphabetical
        is_rev = all(cities[i].lower() >= cities[i+1].lower() for i in range(len(cities)-1))
        if not is_rev:
            return False, f"Not strictly reverse alphabetical: {cities}"
        return True, f"Passed reverse alphabetical list: {cities}"

    elif eval_type == "no_letter_e":
        # Check the answer
        clean = answer.strip()
        if not clean:
            return False, "Empty answer"
        has_e = "e" in clean.lower()
        if has_e:
            count = clean.lower().count("e")
            return False, f"Failed constraint: contains {count} 'e'/'E' letters"
        return True, "Successfully avoided letter 'e'"

    elif eval_type == "word_count":
        clean = answer.strip()
        words = clean.split()
        count = len(words)
        if item["min_words"] <= count <= item["max_words"]:
            return True, f"Word count {count} is within [{item['min_words']}, {item['max_words']}]"
        return False, f"Word count {count} outside [{item['min_words']}, {item['max_words']}]"

    elif eval_type == "contains_word":
        target = item["target"].lower()
        if target in answer.lower():
            return True, f"Found target keyword '{item['target']}'"
        return False, f"Target keyword '{item['target']}' not found in: {answer!r}"

    elif eval_type == "trap_president":
        text = answer.lower()
        rejections = ["did not exist", "was not established", "no president", "before the united states", "1789", "wasn't a country"]
        if any(r in text for r in rejections):
            return True, "Correctly rejected premise (US/president did not exist in 1650)"
        return False, f"Did not clearly reject false premise: {answer[-200:]!r}"

    elif eval_type == "trap_equal_weight":
        text = answer.lower()
        if any(w in text for w in ["same", "equal", "neither", "both weigh"]):
            return True, "Correctly recognized equal weight"
        return False, f"Failed trap: {answer!r}"

    return False, "Unknown evaluation type"


def run_benchmark(base_url: str, model_name: str, out_file: str, max_tokens: int = 2048):
    client = OpenAI(base_url=base_url, api_key="dummy")
    print(f"=== Starting 16-Prompt Quality Benchmark ===")
    print(f"Target: {model_name} @ {base_url}")
    print(f"Sampling: greedy (temperature=0.0), max_tokens={max_tokens}\n")

    results = []
    category_stats = {"math": [0, 0], "code": [0, 0], "instruction": [0, 0], "fact_trap": [0, 0]}

    for i, item in enumerate(BENCHMARK_PROMPTS, 1):
        pid = item["id"]
        cat = item["category"]
        print(f"[{i:02d}/16] ({cat.upper()}) {pid}...", end=" ", flush=True)
        
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": item["prompt"]}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            dur = time.time() - t0
            msg = resp.choices[0].message
            raw_content = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", "") or ""
            think, answer = extract_clean_answer(raw_content)
            if not think and reasoning_content:
                think = reasoning_content
            passed, reason = evaluate_item(item, answer)
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
            "prompt": item["prompt"],
            "think": think,
            "answer": answer,
            "latency_s": round(dur, 2),
            "passed": passed,
            "reason": reason
        })

    # Summary
    total_passed = sum(v[0] for v in category_stats.values())
    total_items = sum(v[1] for v in category_stats.values())
    
    print("\n" + "="*50)
    print(f"BENCHMARK SUMMARY FOR {model_name}")
    print("="*50)
    for cat, (p, t) in category_stats.items():
        print(f"  {cat.capitalize():15s}: {p}/{t} ({p/t*100:.1f}%)")
    print(f"  Total Score    : {total_passed}/{total_items} ({total_passed/total_items*100:.1f}%)")
    print("="*50)

    summary_data = {
        "model": model_name,
        "base_url": base_url,
        "total_score": f"{total_passed}/{total_items}",
        "pass_rate_pct": round(total_passed / total_items * 100, 1),
        "category_stats": {c: f"{p}/{t}" for c, (p, t) in category_stats.items()},
        "results": results
    }

    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Detailed output written to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3.8-27b-2.2bpw")
    parser.add_argument("--out", default="results_bench_16p.json")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    run_benchmark(args.base_url, args.model, args.out, args.max_tokens)
