#!/usr/bin/env bash
# compare_mxfp4_mtp.sh — experiment 1: does MTP close the MXFP4 gap?
#
# Runs BOTH legs back-to-back in ONE session, because known issue 11 says a
# rate recorded on a different day is not comparable (decode is bimodal ~40.2 /
# ~48.5 tok/s, ~20 %). The recorded MXFP4 7.962 tok/s is from 2026-10-16, so
# leg A re-measures it here as the control; leg B changes exactly one thing.
#
#   leg A : serve_mxfp4.sh      (MTP off — the recorded configuration)
#   leg B : serve_mxfp4_mtp.sh  (MTP on, k=4; everything else identical)
#
# Both legs: bench_p512_g128.py, p512/g128, 1 warmup + 5 measured runs.
# Leg B also captures the engine's SpecDecoding metrics so the gain can be
# checked against the measured acceptance rate rather than assumed.
#
# Restores serve_coder390_gptq.sh (the live :8003 server) on exit, success or
# failure — the trap fires even if a leg fails to come up.
#
# Usage: ./compare_mxfp4_mtp.sh          (~35-50 min: two ~15 min cold starts)
set -u
cd "$(dirname "$0")"

URL='http://127.0.0.1:8002'
MODEL='coder390-mxfp4'
LEG_A='bench_mxfp4_rerun_off.json'
LEG_B='bench_mxfp4_mtp_on.json'
SUMMARY='exp1_mxfp4_mtp_summary.json'
ACCEPT='exp1_mxfp4_mtp_acceptance.txt'
LOG='exp1_mxfp4_mtp_run.log'

restore_gptq=0
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --leg-b-only reuses this session's already-measured MTP-off control
# (bench_mxfp4_rerun_off.json) instead of re-booting leg A. Added after the
# first leg-B attempt died in init on KV memory: leg A was already valid and
# re-running it would just cost another ~10 min cold start.
LEG_B_ONLY=0
[ "${1:-}" = "--leg-b-only" ] && LEG_B_ONLY=1

cleanup() {
  rc=$?
  log "cleanup (rc=$rc): stopping MXFP4 legs"
  sg docker -c "docker rm -f vllm-mxfp4-b60 vllm-mxfp4-mtp-b60" >/dev/null 2>&1 || true
  if [ "$restore_gptq" = 1 ]; then
    log "restoring the live server: serve_coder390_gptq.sh (:8003)"
    ./serve_coder390_gptq.sh >>"$LOG" 2>&1 || log "WARN: restore failed — see $LOG"
  fi
  exit $rc
}
trap cleanup EXIT

: >"$LOG"
log "=== experiment 1: coder390 MXFP4 + MTP ==="

# --- who is running now? -----------------------------------------------
running="$(sg docker -c "docker ps --format '{{.Names}}'" 2>/dev/null)"
log "containers up: $(echo $running | tr '\n' ' ')"
for c in $running; do
  case "$c" in
    vllm-coder390-gptq|vllm-xpu-b60|vllm-exl3-b60|vllm-mxfp4-b60|vllm-mxfp4-mtp-b60)
      log "will stop $c to free the B60"
      sg docker -c "docker stop $c" >/dev/null 2>&1 || true
      [ "$c" = vllm-coder390-gptq ] && restore_gptq=1 ;;
    *) log "NOTE: unrelated container running: $c" ;;
  esac
done

# --- leg A: MTP off (control, same session) ----------------------------
if [ "$LEG_B_ONLY" = 1 ]; then
  [ -s "$LEG_A" ] || { log "FATAL: --leg-b-only but $LEG_A is missing/empty"; exit 1; }
  log "--- LEG A: SKIPPED (--leg-b-only); reusing $LEG_A measured earlier this session ---"
else
log "--- LEG A: MXFP4 with MTP OFF (control) ---"
./serve_mxfp4.sh >>"$LOG" 2>&1
if [ $? -ne 0 ]; then log "FATAL: leg A never became healthy"; exit 1; fi

sg docker -c "docker logs vllm-mxfp4-b60 2>&1" \
  | grep -iE "GPU KV cache size|Available KV|graphs|enforce_eager" | tail -5 | tee -a "$LOG"

python3 bench_p512_g128.py --base-url "$URL" --model "$MODEL" --out "$LEG_A" 2>&1 | tee -a "$LOG"
[ -s "$LEG_A" ] || { log "FATAL: leg A bench produced no JSON"; exit 1; }
sg docker -c "docker rm -f vllm-mxfp4-b60" >/dev/null 2>&1 || true
fi

# --- leg B: MTP on -----------------------------------------------------
log "--- LEG B: MXFP4 with MTP ON (k=4) ---"
./serve_mxfp4_mtp.sh >>"$LOG" 2>&1
if [ $? -ne 0 ]; then
  log "FATAL: leg B never became healthy — MTP did not load on the stock image"
  sg docker -c "docker logs vllm-mxfp4-mtp-b60 2>&1" | tail -40 | tee -a "$LOG"
  exit 1
fi

# Did the draft actually build unquantized (no /patch_mtp.py needed)?
sg docker -c "docker logs vllm-mxfp4-mtp-b60 2>&1" \
  | grep -iE "mtp|draft|speculat|GPU KV cache size" | head -25 | tee -a "$LOG"

python3 bench_p512_g128.py --base-url "$URL" --model "$MODEL" --out "$LEG_B" 2>&1 | tee -a "$LOG"
[ -s "$LEG_B" ] || { log "FATAL: leg B bench produced no JSON"; exit 1; }

# Acceptance metrics actually observed on the engine side.
sg docker -c "docker logs vllm-mxfp4-mtp-b60 2>&1" \
  | grep -oE "Mean acceptance length: [0-9.]+,.*Per-position acceptance rate: [0-9., ]+" \
  | tail -5 >"$ACCEPT"
log "SpecDecoding metrics -> $ACCEPT"; cat "$ACCEPT" | tee -a "$LOG"

# --- summary -----------------------------------------------------------
python3 - "$LEG_A" "$LEG_B" "$SUMMARY" <<'PY' | tee -a "$LOG"
import json, sys, statistics
a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
A = json.load(open(a)); B = json.load(open(b))
def med(d): return d["decode_tok_s"]["median"]
def ttft(d): return d["ttft_s"]["median"]
ra, rb = med(A), med(B)
rec = {"same_session": True, "harness": "bench_p512_g128.py p512/g128, 1 warmup + 5 runs",
       "prompt_tokens": 512, "max_tokens": 128,
       "leg_a_config": "ctx 81920, MTP off, eager, max-num-batched-tokens 4096",
       "leg_b_config": "MTP k=4, ctx 40960 (forced: the draft's BF16 weights leave only "
                       "2.49 GiB for KV at util 0.95, and 81920 needs 3.67 GiB; the engine's "
                       "own estimate of the max is 46592). ctx does not affect the p512 decode rate.",
       "leg_a_mtp_off": {"decode_median_tok_s": ra, "ttft_median_s": ttft(A),
                          "runs": [r["decode_tok_s"] for r in A["runs"]]},
       "leg_b_mtp_on": {"decode_median_tok_s": rb, "ttft_median_s": ttft(B),
                         "runs": [r["decode_tok_s"] for r in B["runs"]]},
       "speedup": rb / ra if ra else None,
       "recorded_cross_session_2026_10_16": 7.96245735913954}
json.dump(rec, open(out, "w"), indent=2)
print()
print("=" * 74)
print(f"{'leg':<34}{'decode median':>16}{'TTFT':>10}")
print("-" * 74)
print(f"{'A: MXFP4, MTP off (control, today)':<34}{ra:>13.3f} t/s{ttft(A):>9.3f}s")
print(f"{'B: MXFP4, MTP on k=4 (today)':<34}{rb:>13.3f} t/s{ttft(B):>9.3f}s")
print("-" * 74)
print(f"MTP gain, same session: {rb/ra:.3f}x")
print(f"recorded 2026-10-16 (cross-session control): 7.962 t/s -> {rb/7.96245735913954:.3f}x")
print("=" * 74)
PY
log "wrote $SUMMARY"
log "=== experiment 1 done ==="
