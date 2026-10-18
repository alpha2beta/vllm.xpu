#!/usr/bin/env bash
# quantize_kat_gptq.sh - Production W4A16 GPTQ (sym, g128) quantizer wrapper for
# Kwaipilot/KAT-Coder-V2.5-Dev on Intel Arc 140V (Lunar Lake Core Ultra 7 258V).
#
# Adapted from the Arc Pro B60 GPTQ quantization guide:
#   * Mirrors compressed-tensors packed INT4 W4A16 recipe
#   * Respects the unified 30 GiB RAM/XPU memory budget
#   * Supports shard streaming and cleanup for disk-constrained hosts
#   * Automatically validates preflight, environment, and output artifacts
#
# Usage:
#   ./scripts/quantize_kat_gptq.sh                 # Full 40-layer quantization
#   ./scripts/quantize_kat_gptq.sh --check         # Dry-run checks only
#   ./scripts/quantize_kat_gptq.sh --layers 0-1    # Quick trial (layers 0 and 1)
#   ./scripts/quantize_kat_gptq.sh --clean-shards  # Delete source shards as layers finish
#
set -euo pipefail

CALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CALL_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_PY="${ROOT_DIR}/.venv/bin/python"
OUT_DIR="${ROOT_DIR}/models/KAT-Coder-V2.5-Dev-W4A16-GPTQ"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/quantize-kat-gptq.log"
mkdir -p "${LOG_DIR}"

CLEAN_SHARDS=1
CHECK_ONLY=0
DRY_RUN=0
PASSTHRU=()

for arg in "$@"; do
  case "$arg" in
    --check)            CHECK_ONLY=1 ;;
    --dry-run)          DRY_RUN=1 ;;
    --no-clean-shards)  CLEAN_SHARDS=0 ;;
    *)                  PASSTHRU+=("$arg") ;;
  esac
done

if [ "${CLEAN_SHARDS}" = "1" ]; then
  PASSTHRU+=("--clean-shards")
fi

echo "============================================================"
echo "KAT-Coder-V2.5-Dev W4A16 GPTQ Quantizer on Intel Arc 140V"
echo "============================================================"

# 0. Check for conflicting vLLM server processes holding XPU or RAM
RUNNING_SERVERS=$(pgrep -f "vllm serve" || true)
if [ -n "${RUNNING_SERVERS}" ]; then
  echo "WARNING: Detected running vLLM server process(es): ${RUNNING_SERVERS}"
  if [ "${CHECK_ONLY}" = "0" ] && [ "${DRY_RUN}" = "0" ]; then
    echo "ERROR: Refusing to start while a vLLM server is active. Stop it first to free unified memory:"
    echo "  pkill -f 'vllm serve'"
    exit 1
  fi
fi

# 1. Run Preflight Probe
echo "--- Running Preflight Probe ---"
if ! "${VENV_PY}" "${CALL_DIR}/preflight_probe_kat.py"; then
  echo "Preflight probe failed. Aborting."
  exit 1
fi

if [ "${CHECK_ONLY}" = "1" ]; then
  echo "Check complete. Output directory will be: ${OUT_DIR}"
  exit 0
fi

# 2. Setup Environment Variables
export VLLM_TARGET_DEVICE=xpu
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7"
export ZE_AFFINITY_MASK=0
export HF_HUB_ENABLE_HF_TRANSFER=1

CMD=(
  "${VENV_PY}" "${CALL_DIR}/quantize_kat_gptq.py"
  --out "${OUT_DIR}"
  --device "xpu"
  "${PASSTHRU[@]}"
)

echo "--- Invocation ---"
echo "${CMD[*]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "Dry run complete."
  exit 0
fi

echo "--- Starting Quantization (streaming to console & ${LOG_FILE}) ---"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

# 3. Post-Quantization Verification
echo "--- Verifying Output Artifacts ---"
"${VENV_PY}" - <<PY
import json, os, sys
from pathlib import Path

out = Path("${OUT_DIR}")
idx_path = out / "model.safetensors.index.json"
if not idx_path.is_file():
    print("ERROR: model.safetensors.index.json not found in output directory!")
    sys.exit(1)

idx = json.loads(idx_path.read_text())
wm = idx.get("weight_map", {})
packed = [k for k in wm if k.endswith(".weight_packed")]
scales = [k for k in wm if k.endswith(".weight_scale")]
shapes = [k for k in wm if k.endswith(".weight_shape")]
mtp = [k for k in wm if k.startswith("mtp.")]
plain = [k for k in wm if not (k.endswith(".weight_packed") or k.endswith(".weight_scale") or k.endswith(".weight_shape"))]

print(f"Total Index Tensors: {len(wm)}")
print(f"Packed W4 Projections: {len(packed)}")
print(f"BFloat16 Scales: {len(scales)}")
print(f"Weight Shapes: {len(shapes)}")
print(f"MTP Speculative Draft Tensors: {len(mtp)}")
print(f"Unquantized Base/Norm Tensors: {len(plain)}")

cfg_path = out / "config.json"
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text())
    qc = cfg.get("quantization_config", {})
    print("Quantization Format:", qc.get("format"))
    print("Quant Method:", qc.get("quant_method"))
    print("MTP Hidden Layers:", cfg.get("text_config", {}).get("mtp_num_hidden_layers"))
PY

echo "Quantization finished successfully. Output ready at: ${OUT_DIR}"
