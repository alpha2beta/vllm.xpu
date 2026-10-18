#!/usr/bin/env bash
# publish_gptq_hf.sh - resilient publication of the coder390 GPTQ-Int4 artifact
# to the Hugging Face Hub.
#
# Why this exists instead of a single `hf upload`:
#   On 2026-10-18 a plain directory upload ran for 1h53m, pushed 5.75 GB of xorbs
#   into the Xet CAS, then died on a WAN blip (NetworkUnreachable / Connection
#   timed out) *without committing anything* -- the Hub still showed 0/5 shards.
#   A ~19 GB upload over a ~10 Mbps upstream is a multi-hour operation, so it
#   must tolerate the link going away.
#
# Design:
#   * ONE SHARD AT A TIME. Each shard is ~3.2-4.3 GB (~1-1.5 h), so the blast
#     radius of a drop is one shard rather than the whole model, and every
#     finished shard is committed (durable) instead of sitting in CAS
#     unreferenced, where it risks garbage collection.
#   * Retry each shard until the HUB ITSELF confirms it (not until the command
#     exits 0). Xet is content-addressed, so a retry re-queries chunk hashes and
#     skips whatever is already in CAS -- retries are cheap, not restarts.
#   * Idempotent and resumable: re-running after any failure, or after this
#     script dies, picks up where it left off.
#
# Usage:
#   ./publish_gptq_hf.sh                 # publish (long; run detached)
#   ./publish_gptq_hf.sh --check         # status only, no uploads
#   ./publish_gptq_hf.sh --shard 3       # only shard 3 (debug)
set -u

REPO='alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-GPTQ-Int4-sym-G128-MTP-BF16'
DIR='/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-GPTQ-Int4-sym-G128-MTP-BF16'
MARK="$DIR/.hf_publish_done"
SHARDS=(
  model-00001-of-00005.safetensors
  model-00002-of-00005.safetensors
  model-00003-of-00005.safetensors
  model-00004-of-00005.safetensors
  model-00005-of-00005.safetensors
)
SIDECARS=(README.md SHA256SUMS preprocessor_config.json video_preprocessor_config.json)

say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

# Is <file> present in the remote repo? (the Hub is the source of truth)
# Exact match on the JSON "path" field: a plain grep would let
# preprocessor_config.json match inside video_preprocessor_config.json.
have() { hf models ls "$REPO" --format json 2>/dev/null | grep -qF "\"path\": \"$1\""; }

status() {
  echo "=== $REPO ==="
  hf models ls "$REPO" --format json 2>/dev/null | python3 -c "
import json, sys
try: files = json.load(sys.stdin)
except Exception: files = []
for f in files: print('  %12d  %s' % (f.get('size', 0), f.get('path', '?')))
print('  (%d files)' % len(files))
"
  printf -- '--- shards present:'
  for s in "${SHARDS[@]}"; do have "$s" && printf ' %s' "${s:10:5}"; done
  echo
}

if [ "${1:-}" = "--check" ]; then status; exit 0; fi

only=''
[ "${1:-}" = "--shard" ] && only="${2:?shard number}"

say "publishing $DIR -> $REPO"
available_kb=$(df -Pk "$DIR" | awk 'NR==2{print $4}')
say "disk free: $((available_kb/1048576)) GiB"

fail_total=0
for s in "${SHARDS[@]}"; do
  [ -n "$only" ] && case "$s" in *-0000"$only"-*) : ;; *) continue ;; esac
  if have "$s"; then say "SKIP  $s (already on the Hub)"; continue; fi
  sz=$(stat -c %s "$DIR/$s" 2>/dev/null || echo 0)
  say "START $s ($((sz/1048576)) MiB)"
  ok=0
  for attempt in $(seq 1 60); do
    # generous per-attempt retries inside hf/xet, then our own backoff
    if hf upload "$REPO" "$DIR" . --include "$s" \
         --commit-message "Add $s"; then
      if have "$s"; then ok=1; say "DONE  $s (attempt $attempt)"; break; fi
      say "WARN  $s: command ok but Hub does not list it; retrying"
    else
      say "WARN  $s: upload failed (attempt $attempt)"
    fi
    sleep 30
  done
  if [ "$ok" != 1 ]; then
    say "GIVE UP on $s after 60 attempts -- re-run this script later to resume"
    fail_total=$((fail_total+1))
  fi
done

for f in "${SIDECARS[@]}"; do
  [ -f "$DIR/$f" ] || continue
  have "$f" && { say "SKIP  $f"; continue; }
  say "START $f"
  hf upload "$REPO" "$DIR" . --include "$f" --commit-message "Add $f" || true
done

echo
status
missing=0
for s in "${SHARDS[@]}"; do have "$s" || missing=$((missing+1)); done
if [ "$missing" = 0 ] && [ "$fail_total" = 0 ]; then
  say "ALL 5 SHARDS PUBLISHED"
  date -u +%FT%TZ > "$MARK"
else
  say "INCOMPLETE: $missing shard(s) missing -- re-run to resume"
fi
exit 0
