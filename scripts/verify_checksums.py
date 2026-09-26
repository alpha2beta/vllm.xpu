#!/usr/bin/env python3
"""Verify the downloaded checkpoint against the Hugging Face LFS SHA-256 OIDs
for the pinned revision (independent of the downloader's own integrity checks).

    python scripts/verify_checksums.py [--model-dir models/Qwen3.6-35B-A3B-MXFP4]

Writes logs/model/checksums.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

REPO = "pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4"
REV = "7eceff3a9f7e6f916c824d197266d86676bce695"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models/Qwen3.6-35B-A3B-MXFP4"))
    ap.add_argument("--skip-lfs", action="store_true",
                    help="also check small non-LFS files by git blob oid")
    args = ap.parse_args()
    dest = pathlib.Path(args.model_dir)

    url = f"https://huggingface.co/api/models/{REPO}?blobs=true&revision={REV}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        meta = json.load(fh)

    expected: dict[str, str | None] = {}
    for s in meta["siblings"]:
        lfs = s.get("lfs") or {}
        oid = lfs.get("oid")  # sha256 for LFS-tracked files
        expected[s["rfilename"]] = oid

    out: list[str] = []
    out.append(f"repo={REPO} revision={REV}")
    ok = bad = skipped = 0
    for name, oid in sorted(expected.items()):
        path = dest / name
        if not path.exists():
            out.append(f"MISSING   {name}")
            bad += 1
            continue
        if oid is None:
            skipped += 1
            continue
        digest = sha256(path)
        if digest == oid:
            ok += 1
            out.append(f"OK        {name}  sha256={digest[:16]}…")
        else:
            bad += 1
            out.append(f"MISMATCH  {name}  local={digest} expected={oid}")

    out.append("")
    out.append(f"verified={ok} mismatched={bad} non-lfs-skipped={skipped}")
    out.append("RESULT: " + ("OK" if bad == 0 else "FAILED"))

    report = "\n".join(out) + "\n"
    (ROOT / "logs/model/checksums.txt").write_text(report)
    print(report)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
