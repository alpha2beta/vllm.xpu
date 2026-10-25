#!/usr/bin/env bash
# Resume HumanEval / HumanEval+ benchmark against running vLLM server without deleting existing progress
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"

RAW_MODEL_NAME="${1:-$(curl -s "${BASE_URL}/models" | python3 -c "import sys, json; data=json.load(sys.stdin).get('data', []); print(data[0]['id'] if data else 'unknown')")}"
# Strip leading path and .gguf extension if model ID is a file path
MODEL_NAME="$(basename "${RAW_MODEL_NAME}" .gguf)"

LOG="logs/evalplus-humaneval-${MODEL_NAME}.log"
mkdir -p logs evalplus_results

echo "=== Resuming HumanEval & HumanEval+ Benchmark for ${MODEL_NAME} at $(date) ===" | tee -a "$LOG"
echo "Target Endpoint: ${BASE_URL} (Raw Model: ${RAW_MODEL_NAME})" | tee -a "$LOG"

export PYTHONUNBUFFERED=1
export EVALPLUS_TIMEOUT_PER_TASK="${EVALPLUS_TIMEOUT_PER_TASK:-1200}"
export EVALPLUS_MAX_NEW_TOKENS="${EVALPLUS_MAX_NEW_TOKENS:-2048}"
export EVALPLUS_ENABLE_THINKING="${EVALPLUS_ENABLE_THINKING:-1}"

echo "Settings: EVALPLUS_TIMEOUT_PER_TASK=${EVALPLUS_TIMEOUT_PER_TASK}s, EVALPLUS_MAX_NEW_TOKENS=${EVALPLUS_MAX_NEW_TOKENS}, EVALPLUS_ENABLE_THINKING=${EVALPLUS_ENABLE_THINKING}" | tee -a "$LOG"

echo "--- Generating remaining solutions via evalplus.codegen (resuming from existing samples) ---" | tee -a "$LOG"
OPENAI_API_KEY="none" .venv/bin/evalplus.codegen \
  --model "${MODEL_NAME}" \
  --dataset humaneval \
  --backend openai \
  --base_url "${BASE_URL}" \
  --greedy \
  --bs 1 \
  --resume True \
  --root evalplus_results \
  2>&1 | tee -a "$LOG"

echo "--- Evaluating solutions with EvalPlus (Base + Plus test suites) ---" | tee -a "$LOG"
.venv/bin/evalplus.evaluate \
  --dataset humaneval \
  --samples "evalplus_results/humaneval/${MODEL_NAME}_openai_temp_0.0.jsonl" \
  2>&1 | tee -a "$LOG"

echo "=== HumanEval & HumanEval+ Benchmark for ${MODEL_NAME} Completed: $(date) ===" | tee -a "$LOG"
