#!/usr/bin/env bash
# Download the pinned model revision used by this experiment.
# Usage: scripts/download_model.sh
set -euo pipefail

REPO="pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4"
REV="7eceff3a9f7e6f916c824d197266d86676bce695"
DEST="models/Qwen3.6-35B-A3B-MXFP4"

mkdir -p "$DEST" logs/model
echo "downloading $REPO @ $REV -> $DEST"
hf download "$REPO" --revision "$REV" --local-dir "$DEST" 2>&1 | tee -a logs/model/download.log

echo
echo "verifying file count / size against logs/model/repo-filelist.txt"
python3 - <<'PY'
import pathlib
expected = {}
for line in pathlib.Path("logs/model/repo-filelist.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) == 2 and parts[0].isdigit():
        expected[parts[1]] = int(parts[0])
dest = pathlib.Path("models/Qwen3.6-35B-A3B-MXFP4")
missing = [f for f in expected if not (dest / f).exists()]
bad = [f for f, sz in expected.items()
       if (dest / f).exists() and (dest / f).stat().st_size != sz]
total = sum((dest / f).stat().st_size for f in expected if (dest / f).exists())
print(f"expected {len(expected)} files, total {total/1e9:.3f} GB")
print(f"missing: {missing}")
print(f"size mismatch: {bad}")
print("OK" if not missing and not bad else "FAILED")
PY
