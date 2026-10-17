#!/bin/bash
# ---------------------------------------------------------------------------
# serve_8081.sh - run the same vLLM/B60 server, additionally published on a
#                  LAN port so opencode on another machine can use it.
#
#   ./serve_8081.sh                 # relaunch with 8081 (and 8000) published
#   ./serve_8081.sh --check         # no docker changes: verify state + print URLs
#   ./serve_8081.sh --dry-run       # print the exact command, don't run it
#   ./serve_8081.sh --api-key SECRET  # require Authorization: Bearer SECRET
#   ./serve_8081.sh --port 9000 --no-local
#
# WHY A SEPARATE SCRIPT AT ALL
#   phase5_launch.sh is the launch command of record (REPRODUCIBILITY.md §5) and
#   only publishes 8000. Rather than edit that record, this file embeds a *copy*
#   of the identical `docker run` text with @PLACEHOLDERS@ where a knob lives:
#       @PORTS@          -p flags
#       @ZE_MASK@        ZE_AFFINITY_MASK            (recorded value: 0)
#       @XPU_GRAPH@      VLLM_XPU_ENABLE_XPU_GRAPH   (recorded value: 1)
#       @EXTRA_DOCKER@   extra docker-run flags       (recorded value: none)
#       @EXTRA_ENV@      extra -e flags               (recorded value: none)
#       @EXTRA_MOUNTS@   extra -v flags               (recorded value: none)
#       @HOOK@           commands between the patches and `vllm serve`
#       @SERVE_ARGS@     extra `vllm serve` args       (recorded value: none)
#       @UTIL@           --gpu-memory-utilization      (recorded value: 0.93)
#       @API_KEY_ARG@    trailing --api-key (rec. value: none)
#   `parity()` strips every @UPPER@ token and the -p flags from both files and
#   diffs them, so the two can never silently diverge. Defaults below are the
#   recorded configuration; compare_tp2.sh reuses this same template to derive
#   the control and TP2 variants, so there is only ever ONE copy of the command.
#
# NOTE: 0.0.0.0:8000 is *already* published by phase5_launch.sh, so the server
#       is likely already LAN-reachable on 8000. This exists to give 8081 a
#       stable, dedicated, easy-to-remember endpoint -- and to do the firewall /
#       health / URL plumbing properly.
# ---------------------------------------------------------------------------
set -u

# bash 5.2+ defaults patsub_replacement ON, which rewrites '&' in the
# replacement of ${var//pat/repl} into the matched text -- that silently turns
# `2>&1` into `2>@HOOK@1`. Every replacement here is meant literally.
shopt -u patsub_replacement 2>/dev/null || true

# --- fail fast, and quietly put ourselves into the docker group if we can ----
if ! docker info >/dev/null 2>&1 && [ -z "${_SERVE_REEXEC:-}" ]; then
  if command -v sg >/dev/null 2>&1 && getent group docker >/dev/null 2>&1; then
    printf -v _q '%q ' "$0" "$@"
    exec env _SERVE_REEXEC=1 sg docker -c "$_q"
  fi
  echo "ERROR: cannot talk to the docker API (need docker group or sudo)" >&2
  exit 1
fi

IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL_DIR='/mnt/usb/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16'
COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"
NAME='vllm-xpu-b60'
B60_RENDER_NODE='/dev/dri/renderD128'
RENDER_GID="$(stat -c '%g' "$B60_RENDER_NODE")"
HERE="$(cd "$(dirname "$0")" && pwd)"
RECORD="$HERE/phase5_launch.sh"

HOST_PORT=8081
LOCAL_PORT=8000
API_KEY=''
DRY=0
CHECK=0

usage() { sed -n '2,24p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --port)      HOST_PORT="$2"; shift 2 ;;
    --no-local)  LOCAL_PORT=0; shift ;;
    --api-key)   API_KEY="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    --check)     CHECK=1; shift ;;
    -h|--help)   usage ;;
    *) echo "unknown arg: $1  (try --help)" >&2; exit 2 ;;
  esac
done

# --- the exact `docker run` from phase5_launch.sh, plus @PLACEHOLDERS@ -------
read -r -d '' TMPL <<'TMPL' || true
docker run -d --name "$NAME" @PORTS@ \
  --restart unless-stopped @EXTRA_DOCKER@ \
  --device /dev/dri \
  --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" @EXTRA_MOUNTS@ \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=@ZE_MASK@ \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=@XPU_GRAPH@ \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True @EXTRA_ENV@ \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; @HOOK@exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 12288 --gpu-memory-utilization @UTIL@ --kv-cache-dtype fp8 --port 8000 @SERVE_ARGS@--max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-prefix-caching --served-model-name qwen38 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_xml --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"@API_KEY_ARG@'
TMPL

# --- parity with the launch command of record -------------------------------
collapse() { tr -s '[:space:]' ' ' | sed -e 's/^ //' -e 's/ $//' ; }

parity() {
  [ -f "$RECORD" ] || { echo "  (no phase5_launch.sh to compare against)"; return 0; }
  local a b
  a=$(awk '/^docker run /{f=1} f{print} /--speculative-config/{if(f) exit}' "$RECORD" \
      | sed -E 's/-p [^ ]+//g' | collapse)
  # The knobs that carry their value *inside* the token get the recorded default
  # back (@ZE_MASK@=0, @XPU_GRAPH@=1 are literal values in phase5_launch.sh);
  # everything else is a purely additive knob and is deleted.
  b=$(printf '%s\n' "$TMPL" \
      | sed -E -e 's/@ZE_MASK@/0/g' -e 's/@XPU_GRAPH@/1/g' -e 's/@UTIL@/0.93/g' \
              -e 's/@[A-Z_]+@//g' \
      | collapse)
  if [ "$a" = "$b" ]; then
    echo "  parity with phase5_launch.sh : OK (identical apart from ports/extra args)"
  else
    echo "  !! DIVERGED from phase5_launch.sh:" >&2
    diff <(printf '%s\n' "$a" | fold -s -w 100) \
         <(printf '%s\n' "$b" | fold -s -w 100) >&2
    [ "$DRY" -eq 1 ] || { echo "aborting -- fix the drift, don't launch a different config" >&2; exit 1; }
  fi
}

# --- helpers -----------------------------------------------------------------
# Only real host interfaces: docker0 / br-* / veth* are bridge-local and would
# be a dead end to hand to the other PC.
lan_ips() {
  ip -4 -o addr show scope global \
    | awk '$2 !~ /^(docker|br-|veth|virbr|cni|flannel)/ {split($4,a,"/"); print a[1]}'
}

# The address the kernel would actually use to reach the internet -- the best
# single guess for "which URL should the other PC use".
primary_ip() {
  ip -4 route get 1.1.1.1 2>/dev/null | grep -o 'src [0-9.]*' | awk '{print $2}'
}

cap_watts() {
  local f
  for f in /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap; do
    [ -r "$f" ] && { awk '{printf "%d", $1/1000000}' "$f"; return 0; }
  done
  echo "?"
}

wait_health() {
  local port=$1 deadline=$(( ${2:-300} )) t=0
  echo "waiting for /health on :$port (cold start ~160 s) ..."
  while [ "$t" -lt "$deadline" ]; do
    if curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "  healthy after ${t}s"
      return 0
    fi
    sleep 3; t=$((t + 3))
  done
  echo "  NOT healthy after ${deadline}s -- see: docker logs $NAME" >&2
  return 1
}

print_urls() {
  local port=$1 ip other
  ip=$(primary_ip); [ -z "$ip" ] && ip=$(lan_ips | head -1)
  echo
  echo "endpoint(s):"
  echo "  local : http://127.0.0.1:$port/v1"
  echo "  PREFERRED: http://$ip:$port/v1     <- use this from the other PC"
  other=$(lan_ips | grep -v "^$ip$" | tr '\n' ' ')
  [ -n "${other// /}" ] && echo "  also on : $(echo $other | sed "s|[^ ]*|http://&:$port/v1|g")"
  echo
  echo "smoke test (run on the OTHER pc):"
  echo "  curl -fsS http://$ip:$port/v1/models"
  echo
  echo "opencode config on the other PC (~/.config/opencode/opencode.json):"
  cat <<JSON
  {
    "\$schema": "https://opencode.ai/config.json",
    "provider": {
      "vllm-b60": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Qwen3.8-27B on Arc Pro B60",
        "options": {
          "baseURL": "http://$ip:$port/v1",
          "apiKey": "${API_KEY:-sk-not-needed}",
          "timeout": 600000,
          "headerTimeout": 600000,
          "chunkTimeout": 600000
        },
        "models": {
          "qwen38": {
            "name": "Qwen3.8-27B GPTQ-Int4 + MTP",
            "limit": { "context": 122880, "output": 4086 },
            "reasoning": true
          }
        }
      }
    }
  }
JSON
  echo
  echo "then select it with:  /models  ->  vllm-b60/qwen38"
}

firewall_note() {
  if systemctl is-active --quiet ufw 2>/dev/null; then
    echo
    echo "!! ufw is ACTIVE. Docker-published ports normally bypass ufw (they go"
    echo "   through the DOCKER chain, not INPUT), so this often already works --"
    echo "   but if the other PC cannot connect, open it explicitly:"
    echo "     sudo ufw allow from 192.168.5.0/24 to any port $HOST_PORT proto tcp"
    echo "     sudo ufw status"
  fi
}

cap_note() {
  local w; w=$(cap_watts)
  echo
  if [ "$w" = "180" ]; then
    echo "power cap   : $w W  (matches the recorded configuration)"
  else
    echo "power cap   : $w W  -- recorded baseline was taken at 180 W"
    echo "              restore with: sudo $HERE/phase10_setcap.sh 180"
  fi
}

# --- --check: report only ----------------------------------------------------
if [ "$CHECK" -eq 1 ]; then
  echo "=== parity ==="
  parity
  echo
  echo "=== container ==="
  docker ps -a --filter "name=$NAME" --format '  {{.Names}}  {{.Status}}  {{.Ports}}' \
    || echo "  (none)"
  echo
  echo "=== listening ==="
  ss -ltn | awk 'NR==1 || /:8000|:'"$HOST_PORT"'/'
  echo
  echo "=== health ==="
  for p in 8000 "$HOST_PORT"; do
    if curl -fsS -o /dev/null --max-time 3 "http://127.0.0.1:$p/health" 2>/dev/null; then
      echo "  :$p  HTTP 200"
    else
      echo "  :$p  not responding"
    fi
  done
  cap_note
  print_urls "$HOST_PORT"
  firewall_note
  exit 0
fi

# --- build and show the command ---------------------------------------------
PORT_FLAGS="-p $HOST_PORT:8000"
[ "$LOCAL_PORT" -gt 0 ] && [ "$LOCAL_PORT" != "$HOST_PORT" ] && PORT_FLAGS="$PORT_FLAGS -p $LOCAL_PORT:8000"
API_ARG=''
[ -n "$API_KEY" ] && API_ARG=" --api-key $API_KEY"

CMD="$TMPL"
CMD="${CMD//@PORTS@/$PORT_FLAGS}"
CMD="${CMD//@ZE_MASK@/0}"                 # recorded: index 0 = Arc Pro B60
CMD="${CMD//@XPU_GRAPH@/1}"               # recorded: XPU graphs on
CMD="${CMD//@EXTRA_DOCKER@/}"             # recorded: none
CMD="${CMD//@EXTRA_ENV@/}"                # recorded: none
CMD="${CMD//@EXTRA_MOUNTS@/}"             # recorded: none
CMD="${CMD//@HOOK@/}"                     # recorded: none
CMD="${CMD//@SERVE_ARGS@/}"               # recorded: none (tp=1)
CMD="${CMD//@UTIL@/0.93}"                 # recorded: 0.93
CMD="${CMD//@API_KEY_ARG@/$API_ARG}"

echo "=== parity ==="
parity
echo
echo "=== planned ports ==="
echo "  $PORT_FLAGS   (container always listens on 8000)"
[ -n "$API_KEY" ] && echo "  --api-key: yes (clients must send Authorization: Bearer ...)"
echo
cap_note

if [ "$DRY" -eq 1 ]; then
  echo
  echo "=== docker run (dry run, not executed) ==="
  printf '%s\n' "$CMD"
  exit 0
fi

# --- launch ------------------------------------------------------------------
echo
echo "recreating container $NAME ..."
docker rm -f "$NAME" >/dev/null 2>&1 || true

if ! eval "$CMD"; then
  echo "ERROR: docker run failed" >&2
  exit 1
fi

echo
wait_health "$HOST_PORT" 300 || exit 1
[ "$LOCAL_PORT" -gt 0 ] && [ "$LOCAL_PORT" != "$HOST_PORT" ] && wait_health "$LOCAL_PORT" 60 || true

echo
docker ps --filter "name=$NAME" --format '  {{.Names}}  {{.Status}}  {{.Ports}}'
print_urls "$HOST_PORT"
firewall_note
echo
echo "note: first request after a cold start costs ~2.6 s (JIT); the second ~1.9 s."
