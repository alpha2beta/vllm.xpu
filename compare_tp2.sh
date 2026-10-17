#!/bin/bash
# ---------------------------------------------------------------------------
# compare_tp2.sh - B60-only vs B60+B580, measured so the second card is the
#                  ONLY thing that varies.
#
#   ./compare_tp2.sh --step control        # B60 only, XPU graph OFF
#   ./compare_tp2.sh --step tp2            # B60 + B580, TP=2, graph OFF
#   ./compare_tp2.sh --step restore        # back to the recorded config
#   ./compare_tp2.sh --summary             # print the comparison table
#   ./compare_tp2.sh --step tp2 --dry-run  # print the command only
#   ./compare_tp2.sh --step tp2 --util 0.85  # if the B580 rank OOMs
#   ./compare_tp2.sh --step tp2 --no-bench
#
# WHY THREE STEPS AND NOT TWO
#   TP2 cannot use XPU graphs: "XPU graph capture has been refused for
#   multi-GPU execution on every tested image generation" (cookbook
#   DUAL-B70-TP2.md), so TP2 must run VLLM_XPU_ENABLE_XPU_GRAPH=0. The
#   recorded baseline runs with it =1. Comparing those directly would charge
#   the graph-mode loss to the second card:
#       bench_final_180w.json   B60 only, graph ON   (recorded: 48.490 tok/s)
#       bench_cmp_control.json  B60 only, graph OFF  <- the control
#       bench_cmp_tp2.json      B60+B580, graph OFF  <- card is the only delta
#
# The docker command is extracted from serve_8081.sh's template -- never
# re-typed -- so this script cannot drift from the record of launch. Every
# value it changes is printed as an "intentional deviation" before it runs.
#
# HETEROGENEOUS DEVIATION, STATED UP FRONT: the cookbook's dual-GPU recipe is
# for TWO IDENTICAL B70s. This host pairs a 22.71 GiB B60 (Gen4 x8, 13.85
# GB/s measured) with an 11.33 GiB B580 (Gen3 x4, 3.58 GB/s measured). TP2
# assumes symmetric memory per rank, and 0.93 x 11.33 = 10.54 GiB has to hold
# half the weights (~8.69 GiB) + half the KV + activations. That is the
# expected failure point -- which is what --util is for. The PCIe asymmetry is
# the expected *performance* failure point: without P2P aperture on a consumer
# topology, every all_reduce is host-staged across a x4 link.
# ---------------------------------------------------------------------------
set -u

# bash 5.2+ defaults patsub_replacement ON, which rewrites '&' in the
# replacement of ${var//pat/repl} into the matched text -- that silently turns
# `2>&1` into `2>@HOOK@1`. Every replacement here is meant literally.
shopt -u patsub_replacement 2>/dev/null || true

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/serve_8081.sh"

IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL_DIR='/mnt/usb/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16'
COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
NAME='vllm-xpu-b60'
B60_RENDER_NODE='/dev/dri/renderD128'
RENDER_GID="$(stat -c '%g' "$B60_RENDER_NODE")"
B60_HW="$(echo /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*)"
B580_HW="$(echo /sys/bus/pci/devices/0000:0b:00.0/hwmon/hwmon*)"

STEP=''
DRY=0
NOBENCH=0
UTIL=''

usage() { sed -n '2,32p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --step)     STEP="$2"; shift 2 ;;
    --util)     UTIL="$2"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    --no-bench) NOBENCH=1; shift ;;
    --summary)  STEP='summary'; shift ;;
    -h|--help)  usage ;;
    *) echo "unknown arg: $1  (try --help)" >&2; exit 2 ;;
  esac
done
[ -n "$STEP" ] || { echo "need --step control|tp2|restore  (or --summary)" >&2; exit 2; }

# --- summary mode: read-only, no docker -------------------------------------
if [ "$STEP" = summary ]; then
  printf '%-42s %10s %10s %8s %8s\n' "configuration" "decode_med" "ttft_med" "peak_W" "max_C"
  line() { # $1=json  $2=label  $3=b60 csv  $4=b580 csv (optional)
    local dec tt pk tc
    if [ -f "$1" ]; then
      dec=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"{d['decode_tok_s']['median']:.3f}\")" "$1")
      tt=$(python3 -c  "import json,sys;d=json.load(open(sys.argv[1]));print(f\"{d['ttft_s']['median']:.3f}\")" "$1")
    else dec='-'; tt='-'; fi
    pk='-'; tc='-'
    for c in "$3" "$4"; do
      [ -n "$c" ] && [ -f "$c" ] || continue
      # t_s > 1.0: the sampler's first row divides by a ~0 s interval and
      # prints a phantom ~700 W spike; it is not card power.
      v=$(awk -F, 'NR>1 && $1>1.0 && $2+0>mx {mx=$2+0} END{if(mx) printf "%.1f", mx}' "$c")
      if [ -n "$v" ]; then
        if [ "$pk" = '-' ]; then pk="$v"
        else pk=$(awk -v a="$v" -v b="$pk" 'BEGIN{print (a>b)?a:b}'); fi
      fi
      v=$(awk -F, 'NR>1 && $4+0>mx {mx=$4+0} END{if(mx) printf "%.0f", mx/1000}' "$c")
      if [ -n "$v" ]; then
        if [ "$tc" = '-' ]; then tc="$v"
        else tc=$(awk -v a="$v" -v b="$tc" 'BEGIN{print (a>b)?a:b}'); fi
      fi
    done
    printf '%-42s %10s %10s %8s %8s\n' "$2" "$dec" "$tt" "$pk" "$tc"
  }
  line "$HERE/bench_final_180w.json"    "B60 only, graph ON  (recorded)"      "$HERE/phase9_final_180w.csv" ""
  line "$HERE/bench_cmp_control.json"   "B60 only, graph OFF (control)"       "$HERE/phase9_cmp_control_b60.csv" ""
  line "$HERE/bench_cmp_tp2.json"       "B60 + B580, TP2, graph OFF"          "$HERE/phase9_cmp_tp2_b60.csv"     "$HERE/phase9_cmp_tp2_b580.csv"
  exit 0
fi

# --- restore delegates to the launcher of record ----------------------------
if [ "$STEP" = restore ]; then
  echo "restoring the recorded configuration via serve_8081.sh ..."
  exec "$SRC"
fi

case "$STEP" in control|tp2) ;; *) echo "bad --step: $STEP" >&2; exit 2 ;; esac

# --- docker access ----------------------------------------------------------
if ! docker info >/dev/null 2>&1 && [ -z "${_CMP_REEXEC:-}" ]; then
  printf -v _q '%q ' "$0" "$@"
  exec env _CMP_REEXEC=1 sg docker -c "$_q"
fi

[ -f "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
TMPL=$(sed -n "/<<'TMPL'/,/^TMPL\$/{/<<'TMPL'/d;/^TMPL\$/d;p}" "$SRC")
[ -n "$TMPL" ] || { echo "failed to extract the template from serve_8081.sh" >&2; exit 1; }

# --- knob values per step ---------------------------------------------------
PORT_FLAGS='-p 8081:8000 -p 8000:8000'
API_ARG=''
HOOK=''
EXTRA_DOCKER=''
EXTRA_ENV=''
EXTRA_MOUNTS=''
SERVE_ARGS=''
[ -n "$UTIL" ] || UTIL='0.93'

if [ "$STEP" = control ]; then
  ZE_MASK='0'          # B60 only
  XPU_GRAPH='0'        # match TP2's compile-only mode
else
  ZE_MASK='0,1'        # both cards
  XPU_GRAPH='0'        # capture is refused on multi-GPU (cookbook)
  EXTRA_DOCKER='--ipc=host --cap-add SYS_PTRACE'
  EXTRA_MOUNTS=' -v "$COOKBOOK/patches/patch_vllm_worker_affinity.py:/patch_affinity.py:ro"'
  EXTRA_ENV=' -e B70_WORKER_AFFINITY=1 -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 -e CCL_SYCL_ALLTOALL_TMP_BUF=1'
  SERVE_ARGS='--tensor-parallel-size 2 '
  # fail closed on the patch's own success line, per DUAL-B70-TP2.md
  HOOK='python /patch_affinity.py >/tmp/aff.log 2>&1 || { echo "[patch_affinity] FAILED"; cat /tmp/aff.log; exit 1; }; grep -q "installed successfully" /tmp/aff.log || { echo "[patch_affinity] NO SUCCESS LINE"; cat /tmp/aff.log; exit 1; }; '
fi

CMD="$TMPL"
CMD="${CMD//@PORTS@/$PORT_FLAGS}"
CMD="${CMD//@ZE_MASK@/$ZE_MASK}"
CMD="${CMD//@XPU_GRAPH@/$XPU_GRAPH}"
CMD="${CMD//@EXTRA_DOCKER@/$EXTRA_DOCKER}"
CMD="${CMD//@EXTRA_ENV@/$EXTRA_ENV}"
CMD="${CMD//@EXTRA_MOUNTS@/$EXTRA_MOUNTS}"
CMD="${CMD//@HOOK@/$HOOK}"
CMD="${CMD//@SERVE_ARGS@/$SERVE_ARGS}"
CMD="${CMD//@UTIL@/$UTIL}"
CMD="${CMD//@API_KEY_ARG@/$API_ARG}"

echo "=== step: $STEP ==="
echo "intentional deviations from the recorded launch (phase5_launch.sh):"
echo "  -e ZE_AFFINITY_MASK              0   -> $ZE_MASK"
echo "  -e VLLM_XPU_ENABLE_XPU_GRAPH     1   -> $XPU_GRAPH"
echo "  --gpu-memory-utilization       0.93   -> $UTIL"
if [ "$STEP" = tp2 ]; then
  echo "  + --ipc=host --cap-add SYS_PTRACE"
  echo "  + -e B70_WORKER_AFFINITY=1  and  -e CCL_SYCL_* (4 vars)"
  echo "  + -v patch_vllm_worker_affinity.py, applied with fail-closed check"
  echo "  + --tensor-parallel-size 2"
fi
echo "  (ports unchanged: $PORT_FLAGS)"
echo

if [ "$DRY" = 1 ]; then
  echo "=== docker run (dry run, not executed) ==="
  printf '%s\n' "$CMD"
  exit 0
fi

# --- launch -----------------------------------------------------------------
echo "recreating container $NAME ..."
docker rm -f "$NAME" >/dev/null 2>&1 || true
if ! eval "$CMD"; then
  echo "ERROR: docker run failed" >&2
  exit 1
fi

DEADLINE=300
[ "$STEP" = tp2 ] && DEADLINE=600      # two ranks loading, compile-only
echo "waiting for /health on :8000 (deadline ${DEADLINE}s) ..."
t=0
while [ "$t" -lt "$DEADLINE" ]; do
  if curl -fsS -o /dev/null --max-time 3 "http://127.0.0.1:8000/health" 2>/dev/null; then
    echo "  healthy after ${t}s"
    break
  fi
  sleep 3; t=$((t + 3))
done
if [ "$t" -ge "$DEADLINE" ]; then
  echo "  NOT healthy after ${DEADLINE}s -- last log lines:" >&2
  docker logs --tail 60 "$NAME" 2>&1 | sed 's/^/    /' >&2
  exit 1
fi

docker ps --filter "name=$NAME" --format '  {{.Names}}  {{.Status}}  {{.Ports}}'

# --- benchmark --------------------------------------------------------------
if [ "$NOBENCH" = 1 ]; then
  echo "(--no-bench)"
  exit 0
fi

OUT="$HERE/bench_cmp_${STEP}.json"
CSVB60="$HERE/phase9_cmp_${STEP}_b60.csv"
CSV580="$HERE/phase9_cmp_${STEP}_b580.csv"

echo
echo "benchmarking (identical parameters to the recorded baseline: p512, g128, 5 runs) ..."
python3 "$HERE/phase9_sampler.py" --hwmon "$B60_HW" --out "$CSVB60" --duration 45 --interval 0.25 &
P60=$!
if [ "$STEP" = tp2 ]; then
  python3 "$HERE/phase9_sampler.py" --hwmon "$B580_HW" --out "$CSV580" --duration 45 --interval 0.25 &
  P580=$!
fi
python3 "$HERE/bench_p512_g128.py" --base-url http://127.0.0.1:8000 --out "$OUT"
BENCH_RC=$?
wait $P60
[ "${P580:-}" ] && wait $P580

if [ "$BENCH_RC" -ne 0 ]; then
  echo "benchmark failed (rc=$BENCH_RC)" >&2
  exit "$BENCH_RC"
fi

echo
python3 - "$OUT" "$CSVB60" "$CSV580" <<'PY'
import csv, json, sys
j, c60, c580 = sys.argv[1:4]
d = json.load(open(j))
print(f"  decode  median {d['decode_tok_s']['median']:.3f}  "
      f"(min {d['decode_tok_s']['min']:.3f} / max {d['decode_tok_s']['max']:.3f}) tok/s")
print(f"  TTFT    median {d['ttft_s']['median']:.3f} s")
for path, label in ((c60, "B60"), (c580, "B580")):
    try:
        # t_s > 1.0: skip the sampler's first-row phantom spike (~700 W from a
        # ~0 s divisor interval); it is not card power.
        rows = [r for r in csv.DictReader(open(path))
                if r.get("card_W") and float(r.get("t_s") or 0) > 1.0]
    except FileNotFoundError:
        continue
    if not rows:
        continue
    w = sorted(float(r["card_W"]) for r in rows)
    t = max(int(r["temp_pkg_mC"]) for r in rows if r.get("temp_pkg_mC")) / 1000
    print(f"  {label:4s} power min/med/max = {w[0]:.1f} / {w[len(w)//2]:.1f} / {w[-1]:.1f} W"
          f"   peak temp {t:.0f} C")
PY

echo
echo "recorded: $OUT"
echo "next:     ./compare_tp2.sh --summary        (all three configurations)"
