#!/usr/bin/env bash
# Phase 7: health check for the local OpenAI-compatible server.
# Usage: scripts/health_check.sh [base_url] [model_name] [expected_repeats]
set -uo pipefail

BASE=${1:-http://127.0.0.1:8000}
MODEL=${2:-Qwen3.6-35B-A3B-MXFP4}
REPEATS=${3:-3}

echo "== /health =="
curl -sS -m 30 "$BASE/health" -o /dev/null -w "http=%{http_code} time=%{time_total}s\n" || exit 1

echo "== /v1/models =="
MODELS=$(curl -sS -m 30 "$BASE/v1/models")
echo "$MODELS" | head -c 500
echo
echo "$MODELS" | grep -q "$MODEL" && echo "model listed: OK" || { echo "model NOT listed: FAIL"; exit 1; }

fail=0
for i in $(seq 1 "$REPEATS"); do
  echo "== chat/completions run $i =="
  t0=$(date +%s.%N)
  RESP=$(curl -sS -m 300 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK and nothing else.\"}],\"max_tokens\":16,\"temperature\":0}")
  t1=$(date +%s.%N)
  echo "$RESP" | head -c 400
  echo
  if echo "$RESP" | grep -q '"content"'; then
    # `bc` is not installed on this host; use awk for the elapsed time
    ELAPSED=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    printf "run %s: OK (%ss)\n" "$i" "$ELAPSED"
  else
    printf "run %s: FAIL\n" "$i"
    fail=1
  fi
done

exit $fail
