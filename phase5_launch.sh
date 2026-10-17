#!/bin/bash
# Phase 5 - first launch attempt.
#
# Source: cookbook Quick Start Step 3 (plan section 4.3 is missing from the plan
# file, so the cookbook is the substitute source, per the tasks.md preamble).
#
# ADAPTATIONS FOR THE 22.71 GiB B60 (cookbook numbers assume a 32 GiB B70):
#   --gpu-memory-utilization 0.93   (cookbook 0.88)
#   --max-model-len 12288           (cookbook 100000)
#
# Why: weights alone are 18.216 GiB (5 safetensors shards) out of 22.711 GiB
# total, leaving a 4.495 GiB ceiling for KV + activations + graphs. With KV at
# fp8 and 128 KiB/token (64 layers x 4 kv heads x head_dim 256 x K+V):
#     32768 tok -> 4.00 GiB KV -> needs utilisation >= 0.978 BEFORE activations
#                                    => not viable on this card
#     12288 tok -> 1.50 GiB KV -> fits at 0.93 with ~1.4 GiB left for acts
# Context can be raised later (Phase 10) once the real KV/token figure is read
# back off vLLM's own startup log.
set -u

IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL_DIR='/mnt/usb/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16'
COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
NAME='vllm-xpu-b60'
B60_RENDER_NODE='/dev/dri/renderD128'
RENDER_GID="$(stat -c '%g' "$B60_RENDER_NODE")"

echo "B60_RENDER_NODE=$B60_RENDER_NODE"
echo "RENDER_GID=$RENDER_GID"
echo "ZE_AFFINITY_MASK=0  (Phase 4: index 0 = Arc Pro B60)"
echo "MODEL_DIR=$MODEL_DIR"
echo

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" -p 8000:8000 \
  --restart unless-stopped \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 12288 --gpu-memory-utilization 0.93 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-prefix-caching --served-model-name qwen38 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'

rc=$?
echo
echo "docker run exit=$rc"
echo "container: $NAME"
exit $rc
