#!/usr/bin/env bash
# Sequentially run HumanEval benchmark with Option A (Thinking OFF) followed by Option B (Thinking ON with 4096 budget)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
BASE_MODEL="Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP"

MODEL_A="${BASE_MODEL}-think-off"
MODEL_B="${BASE_MODEL}-think-on-4096"

LOG_DIR="logs"
RESULTS_DIR="evalplus_results/humaneval"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

LOG_A="${LOG_DIR}/evalplus-humaneval-${MODEL_A}.log"
LOG_B="${LOG_DIR}/evalplus-humaneval-${MODEL_B}.log"
LOG_PIPELINE="${LOG_DIR}/benchmark-pipeline-comparison.log"

exec > >(tee -a "${LOG_PIPELINE}") 2>&1

echo "========================================================================"
echo "STARTING BENCHMARK COMPARISON PIPELINE"
echo "Start Time: $(date)"
echo "Target Endpoint: ${BASE_URL}"
echo "Phase 1: ${MODEL_A} (Thinking OFF, 2048 tokens)"
echo "Phase 2: ${MODEL_B} (Thinking ON, 4096 tokens)"
echo "========================================================================"

# Health check endpoint
echo "Checking if llama-server is healthy at ${BASE_URL}..."
if ! curl -sf "${BASE_URL}/models" >/dev/null; then
    echo "ERROR: Server at ${BASE_URL} is not responding."
    exit 1
fi
echo "Server is ready."

# ----------------------------------------------------------------------
# PHASE 1: Option A (Thinking OFF)
# ----------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [PHASE 1] Starting Option A: Thinking OFF (${MODEL_A})"
echo ">>> Time: $(date)"
echo "========================================================================"

if [[ -s "${RESULTS_DIR}/${MODEL_A}_openai_temp_0.0_eval_results.json" ]]; then
    echo ">>> [PHASE 1 ALREADY COMPLETE] Found existing evaluation results for ${MODEL_A}."
    echo ">>> Skipping Phase 1 codegen and using evaluated results."
else
    # Clean any previous partial run for Option A
    rm -f "${RESULTS_DIR}/${MODEL_A}_openai_temp_0.0"*.jsonl

    export PYTHONUNBUFFERED=1
    export EVALPLUS_ENABLE_THINKING=0
    export EVALPLUS_MAX_NEW_TOKENS=2048
    export EVALPLUS_TIMEOUT_PER_TASK=600

    echo "Phase 1 Codegen: Generating 164 solutions with thinking OFF..."
    OPENAI_API_KEY="none" .venv/bin/evalplus.codegen \
      --model "${MODEL_A}" \
      --dataset humaneval \
      --backend openai \
      --base_url "${BASE_URL}" \
      --greedy \
      --bs 1 \
      --root evalplus_results \
      2>&1 | tee "${LOG_A}"

    echo "Phase 1 Evaluate: Running HumanEval / HumanEval+ test suites for ${MODEL_A}..."
    .venv/bin/evalplus.evaluate \
      --dataset humaneval \
      --samples "${RESULTS_DIR}/${MODEL_A}_openai_temp_0.0.jsonl" \
      2>&1 | tee -a "${LOG_A}"
fi

echo ">>> [PHASE 1 COMPLETE] Option A finished at $(date)!"

# ----------------------------------------------------------------------
# PHASE 2: Option B (Thinking ON, 4096 tokens)
# ----------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [PHASE 2] Starting Option B: Thinking ON with 4096 budget (${MODEL_B})"
echo ">>> Time: $(date)"
echo "========================================================================"

# Clean any previous partial run for Option B
rm -f "${RESULTS_DIR}/${MODEL_B}_openai_temp_0.0"*.jsonl

export EVALPLUS_ENABLE_THINKING=1
export EVALPLUS_MAX_NEW_TOKENS=4096
export EVALPLUS_TIMEOUT_PER_TASK=1200

echo "Phase 2 Codegen: Generating 164 solutions with thinking ON (4096 max tokens)..."
OPENAI_API_KEY="none" .venv/bin/evalplus.codegen \
  --model "${MODEL_B}" \
  --dataset humaneval \
  --backend openai \
  --base_url "${BASE_URL}" \
  --greedy \
  --bs 1 \
  --root evalplus_results \
  2>&1 | tee "${LOG_B}"

echo "Phase 2 Evaluate: Running HumanEval / HumanEval+ test suites for ${MODEL_B}..."
.venv/bin/evalplus.evaluate \
  --dataset humaneval \
  --samples "${RESULTS_DIR}/${MODEL_B}_openai_temp_0.0.jsonl" \
  2>&1 | tee -a "${LOG_B}"

echo ">>> [PHASE 2 COMPLETE] Option B finished at $(date)!"

# ----------------------------------------------------------------------
# PHASE 3: Comparison
# ----------------------------------------------------------------------
echo ""
echo "========================================================================"
echo ">>> [PHASE 3] Final Comparison Report"
echo "========================================================================"

FILE_A="${RESULTS_DIR}/${MODEL_A}_openai_temp_0.0_eval_results.json"
FILE_B="${RESULTS_DIR}/${MODEL_B}_openai_temp_0.0_eval_results.json"

if [[ -f "${FILE_A}" && -f "${FILE_B}" ]]; then
    python3 scripts/compare_benchmarks.py "${FILE_A}" "${FILE_B}" "Option A (Thinking OFF)" "Option B (Thinking ON - 4096)"
else
    echo "Warning: One or both evaluation result files missing."
fi

echo "========================================================================"
echo "PIPELINE COMPLETED: $(date)"
echo "========================================================================"
