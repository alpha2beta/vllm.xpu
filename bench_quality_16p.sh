#!/usr/bin/env bash
# bench_quality_16p.sh - run the upstream 16-prompt quality benchmark against a
# local vLLM server.
#
#   ./bench_quality_16p.sh <host_port> <served_model_name> <out.json>
#   ./bench_quality_16p.sh 8003 coder390-gptq quality_coder390_gptq_16p.json
#
# `bench_quality_16p.py` is upstream's, fetched verbatim from
#   https://github.com/alpha2beta/vllm.xpu/blob/main/scripts/bench_quality_16p.py
# (394 lines, sha256 798663c3fe8852fd..., last touched 9ac2d383b560 2026-10-08).
# Only the invocation lives here. It needs the `openai` client, which this host
# does not have installed; `uv run --with openai` supplies it ephemerally.
#
# PRECONDITION, and it is not optional: the server must separate the reasoning
# trace (serve_coder390_gptq.sh passes `--reasoning-parser qwen3`). vLLM emits
# `</think>` as a token but suppresses it from `content`, so without the parser
# the reasoning arrives concatenated in front of the answer with no delimiter to
# split on. `extract_clean_answer()` then returns the whole thing as the answer,
# and the CODE category fails with SyntaxError while the INSTRUCTION category
# fails on constraints like "no letter e" that the *reasoning* violates. MATH
# and FACT_TRAP still pass, so the run looks plausible while ~8/16 items are
# being scored on the wrong text.
set -eu

PORT="${1:?usage: $0 <port> <served-name> <out.json>}"
MODEL="${2:?usage: $0 <port> <served-name> <out.json>}"
OUT="${3:?usage: $0 <port> <served-name> <out.json>}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== 16-prompt quality benchmark — model '$MODEL' on :$PORT ==="
echo "=== reasoning split check (must have '</think>' absent AND reasoning populated) ==="

exec uv run --quiet --with openai python3 "$HERE/bench_quality_16p.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL" \
  --out "$OUT"
