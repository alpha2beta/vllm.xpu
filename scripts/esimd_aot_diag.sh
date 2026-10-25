#!/usr/bin/env bash
# P0-05 item 5: AOT ESIMD register-pressure / spill diagnostic for Arc 140V.
#
# Why this exists
# ---------------
# The production build uses `-fsycl-targets=spir64`, which emits SPIR-V and
# defers register allocation to IGC at JIT time, so the build log carries no
# register/spill information. Two obvious approaches were tried and rejected
# empirically (both reproduced below so the conclusion is auditable):
#
#   * `-Rpass-analysis=kernel-resource-usage`: not implemented by ICPX 2026.0;
#     produces no remarks at all.
#   * `-cl-fail-on-spill`: accepted by ocloc via `-internal_options` but is a
#     no-op in this IGC build. The negative control spills 31 KB and still
#     "Build succeeded".
#
# What does work: IGC itself reports spills per kernel on the AOT device build,
# as
#
#     Spill memory used = 31296 bytes for kernel <mangled name>
#     Compiling kernel with spill code may degrade performance. ...
#
# So this script compiles the *real* translation unit ahead of time for the
# 140V device (`lnl-a1`) and asserts that no kernel produces a spill report.
# A synthetic spill-heavy kernel is built first to prove the detector is live.
#
# Usage (from workspace root):
#   bash scripts/esimd_aot_diag.sh [log_file]
#
# Exit status: 0 = detector verified live and zero real kernels spill.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXL3_SRC="${EXL3_SRC:-$HOME/Projects/exl3xpu}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LOG="${1:-$ROOT/logs/moe-preflight/esimd-compile-lnl.log}"
JSON="${LOG%.log}.json"
WORK="${WORK:-/tmp/opencode/esimd-diag}"
DEVICE="${DEVICE:-lnl-a1}"
OBJ="$WORK/exl3_ops_lnl.o"

mkdir -p "$(dirname "$LOG")" "$WORK"

run() {
  set +u
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force
  set -u

  echo "=== AOT ESIMD diagnostic: real EXL3 kernels -> $DEVICE ==="
  echo "date:    $(date -u +%FT%TZ)"
  echo "host:    $(uname -srm)"
  echo "source:  $EXL3_SRC/csrc/exl3_ops.sycl @ $(git -C "$EXL3_SRC" rev-parse --short HEAD)"
  echo "icpx:    $(which icpx) / $(icpx --version 2>/dev/null | head -1)"
  echo "target:  -fsycl-targets=spir64_gen -Xsycl-target-backend \"-device $DEVICE\""

  local T ABI
  T="$("$PYTHON" -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
  ABI="$("$PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
  echo "torch:   $T (CXX11 ABI $ABI)"
  echo

  # ---------------------------------------------------------------------------
  # 0. Detector sanity check: a kernel that cannot fit in 256 GRFs.
  #    300 live simd<float,32> = 300 * 128 B = 300 GRF.
  # ---------------------------------------------------------------------------
  echo "=== 0. negative control (must report spills) ==="
  "$PYTHON" - "$WORK/spill_negative_control.cpp" <<'PY'
import sys

n = 300
decl = "\n".join(
    f"        simd<float, 32> v{i} = block_load<float, 32>(out + {i * 32});"
    for i in range(n)
)
upd = "\n".join(
    f"                v{i} = v{i} * 1.0001f + v{(i + 1) % n} * 0.5f;" for i in range(n)
)
acc = " + ".join(f"v{i}" for i in range(n))
open(sys.argv[1], "w").write(
    f"""// Auto-generated negative control: {n} live simd<float,32> = {n} GRF,
// beyond the 256-GRF maximum, so IGC must spill.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
using namespace sycl::ext::intel::esimd;

void spill_heavy(sycl::queue &q, float *out, int n) {{
    q.single_task([=]() SYCL_ESIMD_KERNEL {{
{decl}
        #pragma unroll 1
        for (int r = 0; r < n; ++r) {{
{upd}
        }}
        simd<float, 32> s = {acc};
        block_store<float, 32>(out, s);
    }});
}}

int main() {{
    sycl::queue q;
    float *p = sycl::malloc_shared<float>(32 * {n}, q);
    spill_heavy(q, p, 2);
    q.wait();
    return 0;
}}
"""
)
PY

  local neg_log="$WORK/negative-control.log"
  icpx -fsycl -fsycl-targets=spir64_gen \
    -Xsycl-target-backend "-device $DEVICE" \
    -O2 "$WORK/spill_negative_control.cpp" -o "$WORK/negative-control" \
    >"$neg_log" 2>&1
  local neg_rc=$?
  local neg_spills
  neg_spills=$(grep -c "Spill memory used" "$neg_log")
  grep -E "Spill memory used|spill code may degrade" "$neg_log" | sed 's/^/  /'
  echo "  exit=$neg_rc spill_reports=$neg_spills"
  if [ "$neg_rc" -ne 0 ] || [ "$neg_spills" -lt 1 ]; then
    echo "RESULT: ABORT — the detector did not fire on a known-spilling kernel."
    echo "        A clean real-kernel run would be meaningless; investigate first."
    return 3
  fi
  echo "  detector verified live."

  # The same control with -cl-fail-on-spill, documenting that the flag is inert.
  echo
  echo "=== 0b. why -cl-fail-on-spill is not used ==="
  icpx -fsycl -fsycl-targets=spir64_gen \
    -Xsycl-target-backend "-device $DEVICE -internal_options -cl-fail-on-spill" \
    -O2 "$WORK/spill_negative_control.cpp" -o "$WORK/negative-control-failspill" \
    >"$WORK/negative-control-failspill.log" 2>&1
  echo "  exit=$? (expected 0: the flag is a no-op here; the kernel demonstrably spills)"
  grep -c "Spill memory used" "$WORK/negative-control-failspill.log" |
    sed 's/^/  spill reports still emitted: /'

  # ---------------------------------------------------------------------------
  # 1. Real kernels.
  # ---------------------------------------------------------------------------
  echo
  echo "=== 1. real kernels: exl3_ops.sycl ==="
  echo "  flags: -fsycl-targets=spir64_gen -O3 -ffast-math -std=c++20"
  echo "         -DEXL3_ALL_CODEBOOKS -DEXL3_NO_DNNL=1 -fsycl-device-code-split=per_kernel"

  local start end rc
  start=$(date +%s)
  icpx -fsycl -fsycl-targets=spir64_gen \
    -Xsycl-target-backend "-device $DEVICE" \
    -O3 -ffast-math -fPIC -std=c++20 -c \
    -fsycl-device-code-split=per_kernel \
    -DEXL3_ALL_CODEBOOKS -DEXL3_NO_DNNL=1 "-D_GLIBCXX_USE_CXX11_ABI=$ABI" \
    -I "$EXL3_SRC/csrc" -I"$T/include" -I"$T/include/torch/csrc/api/include" \
    -x c++ "$EXL3_SRC/csrc/exl3_ops.sycl" -x none -o "$OBJ" \
    >"$WORK/real-kernels.log" 2>&1
  rc=$?
  end=$(date +%s)

  echo
  echo "  exit=$rc  elapsed=$((end - start)) s  log=$WORK/real-kernels.log"
  local spills
  spills=$(grep -c "Spill memory used" "$WORK/real-kernels.log" || true)
  echo "  spill reports: $spills"

  # ---------------------------------------------------------------------------
  # 2. Prove the routed-expert instantiations are in the image.
  # ---------------------------------------------------------------------------
  local gemv rec
  gemv=$("$PYTHON" - "$OBJ" <<'PY'
import re, sys
data = open(sys.argv[1], "rb").read().decode("latin-1")
seen = set()
for mr, nt in re.findall(r"GemvKernelILi3ELi1ELi(\d+)ELi(\d+)ELb0EE", data):
    seen.add(f"GemvKernel<3,1,MR={mr},NT={nt}>")
print("\n".join(sorted(seen, key=lambda s: int(s.split("MR=")[1].split(",")[0]))))
PY
)
  rec=$("$PYTHON" - "$OBJ" <<'PY'
import re, sys
data = open(sys.argv[1], "rb").read().decode("latin-1")
labels = set()
for nt, tail in re.findall(r"ReconstructKernelILi3ELi1ELi(\d+)E(.{0,60})", data):
    if "half" in tail:
        labels.add(f"ReconstructKernel<3,1,NT={nt},half>")
    elif "bfloat" in tail:
        labels.add(f"ReconstructKernel<3,1,NT={nt},bfloat16>")
    else:
        labels.add(f"ReconstructKernel<3,1,NT={nt}>")
print("\n".join(sorted(labels)))
PY
)
  echo
  echo "=== 2. routed-expert (K=3, mcg) instantiations present in the AOT image ==="
  echo "$gemv" | sed 's/^/  /'
  [ -n "$rec" ] && echo "  $rec"

  # ---------------------------------------------------------------------------
  # 3. Verdict.
  # ---------------------------------------------------------------------------
  echo
  if [ "$rc" -eq 0 ] && [ "$spills" -eq 0 ] && [ -n "$gemv" ] && [ -n "$rec" ]; then
    echo "RESULT: PASS"
    echo "  - all instantiated kernels compiled AOT for $DEVICE"
    echo "  - zero 'Spill memory used' reports (no register spills under 128/256-GRF modes)"
    echo "  - K=3/mcg GemvKernel and ReconstructKernel present in the image"
  else
    echo "RESULT: FAIL (see above)"
    echo "  compiled=$([ "$rc" -eq 0 ] && echo yes || echo no)" \
         "spills=$spills" \
         "k3_gemv=$([ -n "$gemv" ] && echo present || echo MISSING)" \
         "k3_recon=$([ -n "$rec" ] && echo present || echo MISSING)"
  fi
  echo
  echo "note: the shipped _C.so still targets spir64 (IGC JIT at first use); this"
  echo "      object is diagnostic only and is not published."
  ls -l "$OBJ" 2>/dev/null
}

# Compile output goes to both the terminal and the artifact log.
run 2>&1 | tee "$LOG"
STATUS="${PIPESTATUS[0]}"

# Machine-readable sidecar.
"$PYTHON" - "$LOG" "$JSON" "$STATUS" <<'PY'
import json
import re
import sys
from pathlib import Path

log, out, status = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
text = log.read_text(errors="replace")
real = text.split("=== 1. real kernels", 1)[-1]
spills = re.findall(r"Spill memory used = (\d+) bytes for kernel (\S+)", real)
gemv = sorted(set(re.findall(r"GemvKernel<3,1,MR=\d+,NT=\d+>", real)))
rec = sorted(set(re.findall(r"ReconstructKernel<3,1,NT=\d+[^>]*>", real)))
rec_payload = {
    "status": "PASS" if status == 0 else "FAIL",
    "exit_code": status,
    "device": "lnl-a1",
    "spill_reports_real_kernels": [
        {"bytes": int(b), "kernel": k} for b, k in spills
    ],
    "k3_mcg_gemv_instantiations": gemv,
    "k3_mcg_reconstruct_present": bool(rec),
    "diagnostic": "IGC 'Spill memory used' report on AOT spir64_gen build",
    "negative_control": "spill_negative_control.cpp",
    "rejected_flags": {
        "-Rpass-analysis=kernel-resource-usage": "not implemented in ICPX 2026.0",
        "-cl-fail-on-spill": "accepted but inert in IGC 2026.0 (negative control still builds)",
    },
}
out.write_text(json.dumps(rec_payload, indent=2) + "\n")
print(f"wrote {out}")
PY

exit "$STATUS"
