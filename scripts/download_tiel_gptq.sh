#!/usr/bin/env bash
# Download and verify arcitech-psp/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP
set -euo pipefail

REPO="arcitech-psp/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP"
DEST="models/Tiel-Coder-35B-A3B-W4A16-GPTQ-XPU-MTP"
LOG="logs/download-tiel-gptq.log"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "$DEST" logs

echo "=== Downloading $REPO to $DEST ===" | tee "$LOG"
hf download "$REPO" \
  --local-dir "$DEST" \
  --exclude "assets/*" \
  2>&1 | tee -a "$LOG"

echo "=== Verifying SHA256 checksums ===" | tee -a "$LOG"
cd "$DEST"
if [ -f SHA256SUMS ]; then
  # Only check files that exist in the directory (skipping excluded assets)
  grep -v "assets/" SHA256SUMS | sha256sum -c --status && echo "All checksums verified OK!" | tee -a "../../$LOG" || echo "Warning: Checksum verification had issues" | tee -a "../../$LOG"
fi
echo "=== Done ===" | tee -a "../../$LOG"
