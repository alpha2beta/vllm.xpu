#!/usr/bin/env bash
# Run Option B benchmark: Thinking ON with Reasoning Budget 1024 and Token Limit 2048
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
BASE_MODEL="Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP"
MODEL_NAME="${BASE_MODEL}-think-budget-1024"

LOG_DIR="logs"
RESULTS_DIR="evalplus_results/humaneval"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

LOG="${LOG_DIR}/evalplus-humaneval-${MODEL_NAME}.log"

echo "========================================================================" | tee "$LOG"
echo "STARTING OPTION B BENCHMARK (Thinking Budget: 1024, Token Limit: 2048)" | tee -a "$LOG"
echo "Start Time: $(date)" | tee -a "$LOG"
echo "Target Endpoint: ${BASE_URL}" | tee -a "$LOG"
echo "Model Identifier: ${MODEL_NAME}" | tee -a "$LOG"
echo "========================================================================" | tee -a "$LOG"

# Health check endpoint
echo "Checking if llama-server is healthy at ${BASE_URL}..." | tee -a "$LOG"
if ! curl -sf "${BASE_URL}/health" >/dev/null; then
    echo "ERROR: Server at ${BASE_URL} is not responding." | tee -a "$LOG"
    exit 1
fi
echo "Server is ready." | tee -a "$LOG"

# Clean any existing partial run for this specific model identifier
rm -f "${RESULTS_DIR}/${MODEL_NAME}_openai_temp_0.0"*.jsonl

export PYTHONUNBUFFERED=1
export EVALPLUS_ENABLE_THINKING=1
export EVALPLUS_MAX_NEW_TOKENS=2048
export EVALPLUS_TIMEOUT_PER_TASK=1200

echo "Settings: EVALPLUS_ENABLE_THINKING=1, EVALPLUS_MAX_NEW_TOKENS=2048, EVALPLUS_TIMEOUT_PER_TASK=1200s" | tee -a "$LOG"

echo "--- Generating 164 solutions via evalplus.codegen ---" | tee -a "$LOG"
OPENAI_API_KEY="none" .venv/bin/evalplus.codegen \
  --model "${MODEL_NAME}" \
  --dataset humaneval \
  --backend openai \
  --base_url "${BASE_URL}" \
  --greedy \
  --bs 1 \
  --root evalplus_results \
  2>&1 | tee -a "${LOG}"

echo "--- Evaluating solutions with EvalPlus (Base + Plus test suites) ---" | tee -a "$LOG"
.venv/bin/evalplus.evaluate \
  --dataset humaneval \
  --samples "${RESULTS_DIR}/${MODEL_NAME}_openai_temp_0.0.jsonl" \
  2>&1 | tee -a "${LOG}"

echo "========================================================================" | tee -a "$LOG"
echo "OPTION B COMPLETED: $(date)" | tee -a "$LOG"
echo "========================================================================" | tee -a "$LOG"

# Generate side-by-side comparison with Option A
FILE_A="${RESULTS_DIR}/${BASE_MODEL}-think-off_openai_temp_0.0_eval_results.json"
FILE_B="${RESULTS_DIR}/${MODEL_NAME}_openai_temp_0.0_eval_results.json"

if [[ -f "${FILE_A}" && -f "${FILE_B}" ]]; then
    echo "" | tee -a "$LOG"
    echo "========================================================================" | tee -a "$LOG"
    echo "COMPARISON: Option A (Thinking OFF) vs. Option B (Thinking Budget 1024)" | tee -a "$LOG"
    echo "========================================================================" | tee -a "$LOG"
    python3 scripts/compare_benchmarks.py "${FILE_A}" "${FILE_B}" "Option A (Thinking OFF)" "Option B (Budget 1024)" | tee -a "$LOG"
fi
