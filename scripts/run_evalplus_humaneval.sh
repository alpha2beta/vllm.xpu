#!/usr/bin/env bash
# Run HumanEval / HumanEval+ benchmark against the running vLLM server
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${1:-$(curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")}"
LOG="logs/evalplus-humaneval-${MODEL_NAME}.log"
mkdir -p logs evalplus_results

echo "=== HumanEval & HumanEval+ Benchmark for ${MODEL_NAME} Started: $(date) ===" | tee "$LOG"

# Clean any existing partial sample file for a fresh full run
rm -f "evalplus_results/humaneval/${MODEL_NAME}_openai_temp_0.0"*.jsonl

echo "--- Generating 164 solutions via evalplus.codegen ---" | tee -a "$LOG"
OPENAI_API_KEY="none" .venv/bin/evalplus.codegen \
  --model "${MODEL_NAME}" \
  --dataset humaneval \
  --backend openai \
  --base_url http://127.0.0.1:8000/v1 \
  --greedy \
  --bs 1 \
  --root evalplus_results \
  2>&1 | tee -a "$LOG"

echo "--- Evaluating solutions with EvalPlus (Base + Plus test suites) ---" | tee -a "$LOG"
.venv/bin/evalplus.evaluate \
  --dataset humaneval \
  --samples "evalplus_results/humaneval/${MODEL_NAME}_openai_temp_0.0.jsonl" \
  2>&1 | tee -a "$LOG"

echo "=== HumanEval & HumanEval+ Benchmark for ${MODEL_NAME} Completed: $(date) ===" | tee -a "$LOG"
