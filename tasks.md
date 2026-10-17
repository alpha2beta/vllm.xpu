# tasks.md — vLLM XPU / Qwen3.8-27B MTP on Intel Arc Pro B60

Step-by-step implementation checklist for `vllm\_xpu\_b60\_deployment\_plan.md`. Work top to
bottom — later phases assume earlier ones passed. Check off each box as you go; don't skip
Phase 0 even if you're confident about your hardware, since the whole plan's VRAM math depends
on it.

\---

## Phase 0 — Hardware truth-check (do this first, no exceptions)

* \[x] Run `lspci -nn | grep -Ei 'VGA|Display|Intel'` and record both GPUs' PCI addresses.
  **Recorded:** B60 = `0000:03:00.0` (`8086:e211`, Arc Pro B60, subsystem `172F:0300`);
  B580 = `0000:0b:00.0` (`8086:e20b`, Arc B580).
* \[x] Run `ls -l /dev/dri` and `ls -l /dev/dri/by-path` and map each `cardX`/`renderDXXX` to a
PCI address from the previous step.
  **Recorded:** B60 = `card0` / `renderD128` (GID 990 `render`); B580 = `card1` / `renderD129`.
* \[x] For the card you believe is the B60, read its actual VRAM:
`cat /sys/class/drm/cardX/device/mem\_info\_vram\_total` (or `xpu-smi discovery`).
**Record this number** — the plan's two VRAM scenarios (24 GB vs 16 GB) both exist in
the wild; don't assume either.
  **Recorded: 24,385,683,456 B = 22.71 GiB → the 24 GB variant.** (`mem_info_vram_total`
  does not exist under the `xe` driver, and `xpu-smi discovery` reports "No device
  discovered" — that tool only supports datacenter GPUs. Figure taken from OpenCL
  `CL_DEVICE_GLOBAL_MEM_SIZE` via `clinfo`, which equals total VRAM. Device UUID
  `868011e2-0000-0000-0300-000000000000`.)
* \[x] Repeat for the B580 and confirm it reports a distinct, smaller VRAM figure so you can
tell the two apart later from vLLM's startup logs.
  **Recorded: 12,168,933,376 B = 11.33 GiB** (12 GB variant) — distinct and smaller ✓
  (UUID `86800be2-…-0b00-…`).
* \[x] Check PCIe link health for the B60:
`sudo lspci -vv -s <b60\_pci\_addr> | grep -E 'LnkCap|LnkSta|LnkCap2|LnkCtl2'`.
Flag anything downgraded (e.g. negotiated at x1 or x4 when the slot supports x16) before
trusting any throughput comparison later.
  **RESOLVED BY MEASUREMENT — no downgrade.** Register reads alone were a dead end
  (PCIe capability sits at config offset `0x40`, and non-root reads stop at 64 bytes
  both via sysfs *and* `/proc/bus/pci/*`). Every readable register — kernel sysfs for
  both cards, and the operator's root `lspci -vv` — reports **Gen1 x1** on the card's
  internal `e2ff → e2f0 → endpoint` hops, with `LnkCap2: Supported Link Speeds:
  2.5GT/s`. That cannot be the real link, so it was measured instead with
  `l0_bw.cpp` (Level Zero pinned-host ↔ device copies, 4 GiB per trial, client
  monotonic clock):

  | Device | H2D | D2H | Conclusion |
  |---|---:|---:|---|
  | **B60** (`card0`) | **13.85 GB/s** | **11.88 GB/s** | practical **PCIe Gen4 x8** (~15.75 GB/s theoretical) |
  | B580 (`card1`) | **3.58 GB/s** | **2.97 GB/s** | practical **PCIe Gen3 x4** (~3.94 GB/s theoretical) |

  The B580 contrast is what validates the method: each card's measured bandwidth
  matches *its own* upstream-port registers (B60 → Gen4 x8, B580 → Gen3 x4) to
  ~90 % of theoretical, while both show the identical Gen1 x1 artifact below their
  switch — so it is a property of the `e2ff/e2f0` topology, not of either slot.

  Gen1 x1 would cap at ~0.25 GB/s — measured throughput is **~55× higher**, so the
  Gen1 x1 register reading describes an internal fabric hop, **not** the effective
  link. **Flagged, resolved, not a problem.**

  **The real thing to record:** the B60's CPU root port (`00:01.0`) and card upstream
  port (`01:00.0`) both negotiated **Gen4 x8, not x16**. The B580 is on the PCH at
  **Gen3 x4**. So the two cards differ by ~4× in link width/generation — note this
  before any transfer-sensitive (as opposed to VRAM-resident decode) comparison.
* \[x] **Decision gate:** if confirmed VRAM is ≥ \~20 GiB, proceed as planned. If it's \~16 GiB,
re-read plan §1.2 — the 27B checkpoint may not fit with FP8 KV + MTP scratch at any
usable context. Consider a smaller model or dropping MTP before continuing.
  **GATE PASSES: 22.71 GiB ≥ ~20 GiB → proceed as planned.**

## Phase 1 — Host prerequisites

* \[x] `sudo usermod -aG render,video $USER \&\& newgrp render`
  **No-op — already a member of both:** `id` shows `4(video),990(render)`. Skip the
  `usermod`; a fresh login would only be needed if the group were newly added.
* \[x] Confirm your user can read `/dev/dri/renderDXXX` for the B60 without `sudo`.
  **Confirmed:** `renderD128` (B60) is `crw-rw---- root render` and is both readable and
  writable as `yanchun` without sudo.
* \[x] Confirm Docker is installed and can pass `/dev/dri` into a container
(`docker run --rm --device /dev/dri:/dev/dri busybox ls /dev/dri`).
  **PASS (verified against the real image, not busybox).** `docker.io` installed via
  apt; daemon on `Docker Root Dir: /var/lib/docker`.

  **Deviation recorded:** the plan's `data-root` on `/mnt/usb/docker` was **reverted to
  the default**. `/mnt/usb` is **exFAT**, and `chown` on exFAT returns `EPERM`, so the
  daemon died with `could not create or set daemon root permissions: /mnt/usb/docker:
  chown ...: operation not permitted` (and then hit systemd's start-rate limit). exFAT
  cannot be a Docker `data-root` — alternatives would be a loopback ext4 image on
  `/mnt/usb`, which was not needed: the image is only 4.33 GiB compressed / 18.4 GB
  unpacked, so root `/` went from 31 G free to **55 G free (75 % used)**.

  `docker run --rm --device /dev/dri --entrypoint bash <pinned image> -c 'ls -l /dev/dri'`
  lists `card0 card1 renderD128 renderD129` (see Phase 3). **Operator still needed:**
  `docker` group membership requires `sudo usermod -aG docker $USER` — group lookups via
  `sg docker -c '…'` work, but the *current* shell's group set is fixed at login, so
  plain `docker …` needs `newgrp docker` or a fresh login.
* \[x] Confirm free disk space for: the model (\~16 GiB+), the container image, and HF cache.
  **Confirmed — layout changed (operator decision):** root `/` is 225 G with only **31 G
  free (86 % full)**, so both the ~18.2 GiB model and the Docker image live on
  `/mnt/usb` (**668 G free**). Model path is therefore
  `/mnt/usb/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` instead of the
  `$HOME/models/...` shown in Phase 2.

## Phase 2 — Model and patches

* \[x] `pip install huggingface\_hub`
  **Done via `uv`** (host Python 3.14 has no `pip`/`ensurepip`): `uv tool install
  huggingface_hub` → `huggingface-hub 2.0.0`, CLI binary is now `hf`
  (`~/.local/bin/hf`); `huggingface-cli` no longer exists in v2.x.
* \[x] Download the pinned checkpoint:
  **DONE (19 GB, 16/16 files) at `/mnt/usb/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`**
  — path deliberately differs from the `$HOME/models/...` below (Phase 1 note: root `/`
  only had 31 G free). Used `hf download … --revision 9d189a60… --local-dir …`
  (the `--local-dir-use-symlinks` flag no longer exists in `huggingface_hub` 2.x).

```bash
  huggingface-cli download \\
    SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \\
    --revision 9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e \\
    --local-dir "$HOME/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16" \\
    --local-dir-use-symlinks False
  ```

* \[x] Confirm the downloaded revision matches the pinned hash (HF CLI prints this; verify it).
  **VERIFIED from the downloaded `.metadata` files:** each records commit
  `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e` = the pinned revision ✓.
  **Contract also verified** against the cookbook's expectations from
  `quantize_config.json`: `method: gptq`, `bits: 4`, `sym: true`,
  `group_size: 128`, `desc_act: false`, `lm_head: false`,
  `dynamic: {"-:.*mtp.*": {}}` (the `mtp.*` tensors stay BF16) ✓.
  *Note: `preprocessor_config.json` and `video_preprocessor_config.json` are absent
  from the pinned revision — harmless because we serve `--language-model-only`, but
  dropping that flag would fail image-processor load per cookbook §6b.*
* \[x] Clone the cookbook:

```bash
  git clone https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook.git \~/intel-arc-pro-b70-inference-cookbook
  ```

* \[x] Verify patch hashes:
  **Both match the pinned values exactly** (`git` HEAD `88271830`, 2026-09-26):
  `patch_mtp_nightly.py` = `4d7a02c4…3c53e9c14` ✓,
  `patch_mtp_boundary.py` = `41d2f74e…7ce4cc9b50` ✓.

```bash
  cd \~/intel-arc-pro-b70-inference-cookbook
  sha256sum patches/patch\_mtp\_nightly.py   # expect 4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14
  sha256sum patches/patch\_mtp\_boundary.py  # expect 41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50
  ```

* \[x] **Stop-if-mismatch:** if either hash differs, do not proceed with this plan's exact
commands — the cookbook has moved on. Diff the new patch against what this plan expects
before using it.
  **PASSED — no diff needed.** Both hashes match, so the cookbook has not moved on
  relative to this plan.

## Phase 3 — Container image

* \[x] Pull the pinned digest:
  **PULLED** (4.33 GiB compressed, 18 layers, unpacked 18.4 GB) by the operator.

```bash
  docker pull vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
  ```

* \[x] Confirm the pulled image's digest matches (avoid a registry substituting a moved tag).
  **MATCH — stop-if-mismatch PASSED.**

```
$ docker image inspect vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
    --format '{{json .RepoDigests}}'
["vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"]
```

  Registry was also checked independently (`curl -sIL …/manifests/sha256:f01e24f6…`
  → `HTTP/2 200`, `docker-content-digest: sha256:f01e24f6…`), so it is the digest the
  plan pinned, not a moved tag.

* \[x] **Go/no-go:** can the container enumerate `/dev/dri` devices?
  (`docker run --rm --device /dev/dri --entrypoint bash <image> -c 'ls -l /dev/dri'`;
  script kept at `/mnt/usb/vllm.xpu/phase3_dri_test.sh`)
  **PASS, exit 0.** Inside the container: `card0`, `card1`, `renderD128`, `renderD129`
  all present at mode `660`; **`renderD128`/`renderD129` are RW OK**; `libze_loader.so.1`
  and `libze_intel_gpu.so.1` resolve; `intel-opencl-icd 26.27.39122.11-0` installed.

  **Two deviations recorded:**
  1. `--device /dev/kfd` from the plan is **omitted — this Intel host has no `/dev/kfd`**
     (that node is AMD/ROCm). Passing a non-existent node aborts the run with
     `error gathering device information … no such file or directory`.
  2. The image's entrypoint is `vllm serve`, so any inspection command needs
     `--entrypoint bash` (otherwise the probe text is parsed as `vllm serve` args).

## Phase 4 — Device isolation setup

* \[x] Export the confirmed B60 render node from Phase 0:

```bash
  export B60\_RENDER\_NODE="/dev/dri/renderDXXX"
  export RENDER\_GID="$(stat -c '%g' "$B60\_RENDER\_NODE")"
  ```

  **Resolved:** `B60_RENDER_NODE=/dev/dri/renderD128`, `RENDER_GID=990`.
  (Both cards share render GID 990, so `stat /dev/dri/render*` is unambiguous here —
  but pin it to `renderD128` explicitly so a future card can't shift it.)
* \[x] Decide the `ZE\_AFFINITY\_MASK` value that selects the B60 given your Level Zero device
order (may not be `0` — check with `sycl-ls` or `clinfo` if ambiguous).
  **DECIDED: `ZE_AFFINITY_MASK=0` selects the B60.** Verified empirically against
  Level Zero itself (host has no `sycl-ls`; used `/mnt/usb/AInfer/tools/l0probe`,
  which calls `zeInit`/`zeDriverGet`/`zeDeviceGet` — the same enumeration the container
  uses):

  | `ZE_AFFINITY_MASK` | Level Zero devices enumerated |
  |---|---|
  | *(unset)* | 2 → **B60**, B580 |
  | `0` | 1 → **Intel Arc Pro B60** ✓ |
  | `1` | 1 → Intel Arc B580 (i.e. `1` is a *device index*, not a bitmask) |
  | `0,1` | 2 → both |

  So the value is a comma-separated device **index** list, and the B60 is index 0.
  Confirm again in Phase 6 by grepping the reported device memory (22.71 GiB, not
  11.33 GiB).

## Phase 5 — First launch attempt (standard device mapping)

* \[x] Run the launch command from plan §4.3, with your confirmed `RENDER\_GID` and
`ZE\_AFFINITY\_MASK` substituted in.
  **DONE via `/mnt/usb/vllm.xpu/phase5_launch.sh`.** Plan §4.3 is missing from the plan
  file, so per the preamble the source is the **cookbook Quick Start Step 3**, adapted:

  | | cookbook (32 GiB B70) | **this run (22.71 GiB B60)** |
  |---|---|---|
  | `--gpu-memory-utilization` | 0.88 | **0.93** |
  | `--max-model-len` | 100000 | **12288** |

  **Why the deviation (VRAM math):** weights are **18.216 GiB** of file / **17.38 GiB**
  resident, on a 22.711 GiB card → only **4.495 GiB** left for KV + activations + graphs.
  KV is fp8 at ~128 KiB/token (64 layers × 4 kv heads × head_dim 256 × K+V), so:

  ```
  ctx  8192 -> 1.00 GiB   needs utilisation >= 0.846 before activations
  ctx 12288 -> 1.50 GiB   needs >= 0.868
  ctx 32768 -> 4.00 GiB   needs >= 0.978   <-- not viable on this card
  ctx100000 -> 12.21 GiB  needs >= 1.340   <-- cookbook's own value, impossible here
  ```

  Confirmed by vLLM itself: `GPU KV cache size: 17,630 tokens` at utilisation 0.93, so
  `--max-model-len` cannot exceed ~17.6k here (12288 leaves headroom; Phase 10 box 1).

  Pinned inputs substituted: `RENDER_GID=990`, `ZE_AFFINITY_MASK=0`, container name
  `vllm-xpu-b60`, `--restart unless-stopped`, model bind-mounted read-only at `/model`,
  both verified patches mounted at `/patch_mtp.py` + `/patch_boundary.py`.
  **`--device /dev/kfd` from the plan is omitted — this Intel host has no such node.**
* \[x] Tail logs: `docker logs -f vllm-xpu-b60`
  **Done** (note: `docker logs` is cumulative across restarts — use
  `docker logs --since <RFC3339-UTC>` to isolate a run, and the container logs in **UTC**,
  one hour behind local CEST).
* \[x] **If you see `ze\_fd\_manager`, DRM-directory-access, or other oneCCL/XCCL errors:**
stop the container, re-launch with the fallback flags from plan §4.4
(`--cap-add SYS\_PTRACE --security-opt seccomp=unconfined --ipc=host`), and note in your
reproducibility record that the fallback was required.
  **FALLBACK FLAGS NOT REQUIRED.** Only informational CCL warnings appear
  (`|CCL_WARN| pidfd is not supported, fallbacks to drmfd exchange mode`,
  `no membind support for NUMA node 0, skip thread membind`) — no `ze_fd_manager`, no
  DRM-directory-access failure, no XCCL error. Stock flags from the cookbook suffice.

  **A different failure *was* hit, and matters more:** on a **cold container** (no
  `/root/.cache/vllm`), `torch.compile` takes **66.86 s + 17.46 s**, and the first engine
  init then dies:
  ```
  ValueError: To serve at least one request with the model's max seq len (12288),
  (1.24 GiB KV cache is needed, which is larger than the available KV cache memory (0.17 GiB)
  ```
  The compile-time peak inflates the measured activation budget, leaving 0.17 GiB instead
  of the 1.82 GiB seen on a warm start. **vLLM's APIServer auto-retries and the second
  attempt succeeds** (compile cache now warm: 1.12 s). Reproduced twice — it is
  deterministic, not a flake. Warm restarts are clean. Recorded, not fatal; if a clean
  cold log is ever required, the cookbook's compile-OOM remedy
  (`-e TORCHINDUCTOR_COMPILE_THREADS=1`) is the documented lever.
* \[x] Wait for `Application startup complete` before doing anything else.
  **Observed:** cold start **~160 s** to `Application startup complete`; warm restart
  **~45 s**.

## Phase 6 — Startup verification

* \[x] `curl -fsS http://127.0.0.1:8000/health \&\& echo "HEALTH OK"`
  **`HEALTH OK`** (HTTP 200).
* \[x] `curl -fsS http://127.0.0.1:8000/v1/models | jq` — confirm `qwen38` (or your
`--served-model-name`) is listed.
  **`qwen38` listed**, `root: /model`, `max_model_len: 12288`.
* \[x] Grep startup logs for the reported device memory figure and confirm it matches the B60's
Phase-0 VRAM number, not the B580's.
  **MATCH:** `Free memory on device (21.56/22.71 GiB)` → **22.71 GiB = the B60's
  Phase-0 VRAM**, not the B580's 11.33 GiB. Budget: `Desired GPU memory utilization is
  (0.93, 21.12 GiB)`, `Actual usage is 18.09 GiB for consumed memory (weights +
  non-torch), 1.22 GiB for peak activation, and 0.11 GiB for CUDAGraph memory`.
* \[x] Grep startup logs for successful XCCL init, GPTQ-on-XPU path selection, and MTP drafter
initialization messages.
  All three present:
  - **XCCL:** `distributed_init_method=file:///tmp/vllm_dist_… backend=xccl`
  - **GPTQ on XPU:** `Using XPUwNa16LinearKernel for AutoGPTQLinearMethod`
  - **MTP drafter:** `Resolved architecture: Qwen3_5MTP` +
    `Loading drafter model...` + `speculative_config(method='mtp', num_spec_tokens=4)`
  - KV: `GPU KV cache size: 17,630 tokens, Maximum concurrency for 12,288 tokens per
    request: 1.43x`

## Phase 7 — Correctness and MTP verification

* \[x] Send one throwaway request (accept the JIT-compile warmup penalty — don't measure it).
  Sent. On the **genuinely cold** engine (compile cache wiped), request 1 = **2.623 s**;
  recorded here only as the baseline for the next box, not as a benchmark number.
* \[x] Send a second, same-shape request and confirm latency dropped substantially (JIT
compiled successfully).
  **2.623 s → 1.892 s = −27.9 %**, then stable (run 3 = 1.892 s; 1.909 s across a warm
  restart). The penalty is **mostly a startup cost, not a per-request one**:
  `torch.compile` runs at engine init (66.86 s cold, 1.12 s with cache), so a server that
  has been up for a while shows only the residual — the first pre-warm pass of an
  already-running server measured just 1.84 → 1.91 s (no drop).
  Script: `/mnt/usb/vllm.xpu/phase7_verify.py`.
* \[x] `docker logs vllm-xpu-b60 2>\&1 | tail -40` and confirm a `SpecDecoding metrics` block
appears with nonzero drafted/accepted tokens and a mean acceptance length. If this block
is absent or all-zero, MTP is not engaging — check that both patches actually ran
(`python /patch\_mtp.py` / `patch\_boundary.py` should log something in container stdout
before `vllm serve` starts).
  **MTP IS ENGAGING.** Both patches logged to stdout *before* `vllm serve`:
  ```
  patched /opt/venv/.../vllm/model_executor/models/qwen3_5_mtp.py
  patched /opt/venv/.../vllm/v1/attention/backends/gdn_attn.py
  ```
  and the block is nonzero:
  ```
  SpecDecoding metrics: Mean acceptance length: 3.05, Accepted: 43 tokens,
  Drafted: 84 tokens, Per-position acceptance rate: 0.667, 0.619, 0.476, 0.286,
  Avg Draft acceptance rate: 51.2%
  ```
  (re-confirmed after restarts: `Accepted: 129, Drafted: 252`, same 3.05 acceptance).
  Acceptance falls off sharply past position 4 — consistent with the cookbook's finding
  that **MTP depth 4 is optimal**, which is what `num_speculative_tokens: 4` sets.
* \[x] Optional correctness spot-check: send a greedy (`temperature=0`) factual prompt twice and
confirm byte-identical output (a corrupting spec-decode path would diverge).
  **PASS — byte-identical** across two greedy runs.

## Phase 8 — Reproducible benchmark

* \[x] Implement or copy the p512/g128 harness (plan §6.4): build an exact 512-token prompt via
`/tokenize` + `/detokenize`, run one same-shape warmup, then 5 measured completions with
`max\_tokens=128`, `temperature=0`, `ignore\_eos=true`, streaming enabled.
  **Written: `/mnt/usb/vllm.xpu/bench_p512_g128.py`** (stdlib only — host Python 3.14 has
  no pip, so a dependency-free harness was the only option). Verified: prompt is exactly
  **512 tokens / 5975 chars**, round-tripped through `/tokenize` + `/detokenize`.

  Two endpoint-contract surprises fixed against the live `/openapi.json`: `/tokenize`
  requires `{"model","prompt"}` (not `text`, and this build has **no** `/v1/tokenize`),
  and `/detokenize` replies with `{"prompt": ...}` (not `{"text": ...}`).
* \[x] Compute rate per run as `(completion\_tokens - 1) / (t\_end - t\_first\_token)` using the
API's `usage` block and a monotonic client clock — **not** the server's periodic logger
and **not** SSE chunk counts.
  Implemented exactly that way: `completion_tokens` from the API `usage` block
  (`{'prompt_tokens': 512, 'completion_tokens': 128, ...}`), `t_first_token`/`t_end` from
  `time.monotonic()`. The server's `Avg generation throughput` lines were used only as a
  cross-check, never as the measurement.
* \[x] Record median, mean, min/max decode tok/s and median TTFT across the 5 runs.

  **Baseline at the default 220 W cap** (`bench_p512_g128.json`, `bench_power_default.json`):

  | run | median decode tok/s | mean | min | max | median TTFT |
  |---|---:|---:|---:|---:|---:|
  | 1 | **46.791** | 45.063 | 38.026 | 46.859 | **0.478 s** |
  | 2 (during power sampling) | **46.781** | 46.784 | 46.759 | 46.812 | **0.478 s** |

  Run 1's 5th sample dipped to 38.026 tok/s (one-off); run 2 was tight to **0.05 tok/s**
  across all 5 samples, so **46.78–46.79 tok/s is the stable figure**. Warmup pass:
  46.76 tok/s (TTFT 0.58 s) — no material JIT penalty remains at benchmark time.

  *Reference: the cookbook quotes 84.65 tok/s, but on a **32 GiB B70**. This is a
  22.71 GiB **B60** on Gen4 x8, `--max-num-seqs 1`, prefix caching disabled — per
  Phase 9 the cap is to be chosen from **this** card's own curve, not B70's number.*
* \[x] Log this result plus every item in plan §8's reproducibility checklist.
  Plan §8 is missing from the plan file, so the checklist was sourced from the cookbook's
  equivalent. **Record: `/mnt/usb/vllm.xpu/REPRODUCIBILITY.md`** (hardware, image digest,
  model revision, exact launch command, flags, results, deviations).

## Phase 9 — Power/thermal pass

* \[x] Enumerate hwmon entries for the B60's `cardX` and record `power1\_cap` / `power1\_cap\_max`.
  **card0 (B60) → `/sys/class/drm/card0/device/hwmon/hwmon8`, `name=xe`.**
  (Verified the mapping resolves through `0000:00:01.0 → 01:00.0 → 02:01.0 → 03:00.0`;
  card1/B580 → `hwmon9`, `power1_cap = 190 W`.)

  | attribute | value |
  |---|---|
  | `power1_cap` | **220 000 000 µW = 220 W** (the current default cap) |
  | `power1_cap_max` | **not exposed by the `xe` driver** — no such attribute exists |
  | `power1_crit` | 440 W (closest published upper bound) |
  | `power1_label` / `power1_cap_interval` | `card` / 15 |
  | `energy1` / `energy2` | `card` / `pkg` counters — **readable without sudo** |
  | `fan1_input` | 1846 RPM idle |

  Sensors (`temp1` does not exist; they start at `temp2`): `pkg`, `mctrl`, `pcie`
  (crit 100 °C) and `vram` + `vram_ch_0..11` (crit 105 °C).

  *Gotcha worth remembering: `power1_cap` is mode `664 root:root` — **not writable by
  the operator account**, so every cap change needs `sudo`.*
* \[x] Try the benchmark at your current default cap, then at 150W and 180W (dense-model
guidance), comparing median decode tok/s and peak temperature at each.
  **DONE — all three legs measured with `phase9_sampler.py` running alongside the
  benchmark, so power *and* temperature are recorded at every cap.** Cap writes still
  needed `sudo`, so the legs were batched into two root scripts that trap-restore the
  original cap on `EXIT/INT/TERM` (`phase9_caps.sh`, `phase9_final.sh`).

  | cap | runs | median decode tok/s (per run) | mean of medians | Δ vs 220 W | TTFT median | peak card | peak pkg | peak `pkg` |
  |---:|---:|---|---:|---:|---:|---:|---:|---:|
  | 220 W (default) | 4 | 46.791 / 46.781 / 46.677 / 46.616 | **46.72** | — | 0.478 s | 195.9 W (89 %) | 144.1 W | **62.0 °C** |
  | 180 W | 3 | 47.923 / 48.481 / 48.490 | **48.30** | **+3.4 %** | **0.469 s** | 190.2 W (106 %) | 141.4 W | 59.0 °C |
  | 150 W | 2 | 48.285 / 47.822 | **48.05** | +2.9 % | 0.488–0.501 s | 173.5 W (116 %) | 128.2 W | 60.0 °C |

  Supporting measurements: at 220 W the card idles at 37.8 W and hits 195.9 W = **89 % of
  the cap**, `vram` reaches 62 °C, `mctrl`/`pcie` 55–57 °C, fan 2400–2640 RPM. **Nothing is
  thermally limited at any cap** — every reading sits 40+ °C below the 100 °C crit.

  Three things the curve established:

  1. **The cap effect is real, not run-to-run noise.** All four 220 W runs cluster inside
     a 0.18 tok/s band (46.62–46.79), while every capped run sits 1.1–1.5 tok/s above it.
  2. **180 W and 150 W are a statistical tie on decode** (48.30 vs 48.05 — a 0.25 gap
     against a within-cap spread of ~0.46), so decode alone cannot pick between them.
  3. **TTFT breaks the tie: 150 W costs prefill.** 0.488–0.501 s vs 0.469 s at 180 W,
     ~+6 %, because prefill is the compute-bound, power-hungry half. Decode is unaffected.

  *Correction to an earlier note in this file.* A preliminary reading here predicted that
  "since 220 W never binds and 62 °C is far from the trip point, lowering the cap can only
  ever cost performance." **That was wrong.** Capping *raised* throughput by ~3 % at both
  180 W and 150 W. Most plausible mechanism: at 220 W the `xe` driver lets the card boost
  high and then ride the limit, whereas a tighter cap pins a lower, steadier clock — and a
  single-stream decode (`--max-num-seqs 1`) is latency-bound, so consistent clocks beat
  peak clocks. Not proven, but the measurement is replicated across 9 runs.

  *Also observed:* the cap is not an instantaneous hard ceiling — peaks overshoot to 106 %
  (180 W) and 116 % (150 W), consistent with a limit averaged over `power1_cap_interval`
  = 15 s.
* \[x] Pick a cap based on your own measured curve, not the B70 numbers verbatim — B60 thermals
and power delivery may differ.
  **DECIDED — 180 W.** It wins on both axes measured: highest mean decode (48.30 tok/s,
  +3.4 % over default) *and* best TTFT (0.469 s), while running 3 °C cooler than 220 W.

  | candidate | decode | TTFT | verdict |
  |---|---:|---:|---|
  | 220 W (default) | 46.72 | 0.478 s | rejected — 3.4 % slower, hottest |
  | **180 W** | **48.30** | **0.469 s** | **chosen — best on both** |
  | 150 W | 48.05 | 0.488–0.501 s | rejected — tied on decode but +6 % TTFT |

  Not chosen on peak power: 150 W does draw less (173.5 W vs 190.2 W peak), but this
  workload never approaches the 180 W limit anyway, and the TTFT regression is a real cost
  for an interactive endpoint. **If** power later becomes the binding constraint and
  TTFT doesn't matter, 150 W is the fallback — but re-test it, because these are 128-token
  bursts at `--max-num-seqs 1`, not sustained multi-user load.
* \[x] Re-run Phase 8's benchmark once more at the chosen cap as the "final" recorded baseline.
  **DONE — `bench_final_180w.json` at 180 W: 48.490 tok/s median (mean 49.057), TTFT
  0.469 s, peak 190.2 W, `pkg` 59 °C.** This supersedes the 220 W reference of
  46.781 tok/s / 0.478 s (`bench_power_default.json`) — **+3.7 % median, −1.9 % TTFT.**
  Run with `phase9_final.sh`, which sets 180 W, waits 16 s for `power1_cap_interval`
  (15 s) to elapse so the limit actually takes effect, then restores 220 W on exit.

  *One observation to keep honest:* runs 1, 3 and 5 landed at ~48.48 tok/s while runs 2 and
  4 landed at ~49.92 — a bimodal split, not a smooth distribution, which is why the mean
  (49.057) exceeds the median (48.490). The median is the figure carried forward.

  **Caveat for whoever deploys this:** `power1_cap` is plain sysfs and does **not** survive
  a reboot, so 180 W must be re-applied at boot — see Phase 10.

## Phase 10 — Production hardening (after the baseline is recorded)

* \[x] Decide context length: only raise `--max-model-len` above `32768` after confirming
headroom against your Phase-0 VRAM figure at the chosen `--gpu-memory-utilization`.
  **DECIDED — raising above 32768 is impossible on this card; practical ceiling is 16384.**
  Phase-0 VRAM is 22.711 GiB and weights are 17.38 GiB resident (18.216 GiB on disk),
  leaving 4.495 GiB for KV + activations + graphs at the theoretical maximum:

  | `--max-model-len` | fp8 KV needed | utilisation needed (weights+KV only) | verdict @ 0.93 |
  |---:|---:|---:|---|
  | 12288 (current) | 1.50 GiB | 0.868 | **fits**, ~0.5 GiB spare after acts |
  | 16384 | 2.00 GiB | 0.890 | **fits** (vLLM reports a 17,630-token pool) |
  | 32768 | 4.00 GiB | **0.978** | not viable — leaves <0.5 GiB for acts/graphs |
  | 100000 (cookbook) | 12.21 GiB | **1.340** | physically impossible |

  So the answer to "raise above 32768" is **no**, by arithmetic, not by choice. Keeping
  **12288** for the recorded baseline (it covers the p512/g128 benchmark with margin);
  16384 is the max that can be offered if a longer context is wanted, and it needs a
  relaunch + a fresh benchmark because it is a different configuration.
* \[x] Decide prefix caching: keep `--enable-prefix-caching` on for production/RAG use; if you
want a number directly comparable to the published 84.65 tok/s reference, re-run Phase 8
once with `--no-enable-prefix-caching` and record that separately.
  **DECIDED.** The launch already runs **`--no-enable-prefix-caching`** (the cookbook's own
  benchmark setting), so the comparable-to-reference condition is satisfied by
  construction — **46.78 tok/s was measured with prefix caching off**, which is the right
  baseline. For production/RAG, flip to `--enable-prefix-caching`; that is a *different*
  configuration and gets its own Phase 8 re-run rather than being folded into this number.
* \[ ] Set `--restart unless-stopped` (already in the launch command) and confirm the container
survives a host reboot with the B60 still correctly selected (repeat Phase 6's device
memory check after reboot — Level Zero ordering can change across reboots).
  **Partial — `--restart unless-stopped` is confirmed present in `phase5_launch.sh`, and
  two in-place `docker restart`s already re-selected the B60 correctly**
  (`Free memory on device (21.56/22.71 GiB)` both times). **The actual host reboot is
  still outstanding** (operator action), and that is also the only way to check the
  cold-start failure in issue 1 of `REPRODUCIBILITY.md` under real conditions.

  **Two things the reboot will silently undo, both prepared already:**

  1. **The 180 W cap.** `power1_cap` is volatile sysfs state — it reverts to the 220 W
     firmware default on boot, so the Phase 9 configuration would stop applying without
     anyone noticing. `vllm-b60-power-cap.service` re-applies it via `phase10_setcap.sh`,
     which resolves the hwmon dir from the B60's PCI address (`0000:03:00.0`) instead of
     a hardcoded `hwmon8`, since hwmon numbering is not stable across boots.
  2. **Level Zero device ordering.** `ZE_AFFINITY_MASK=0` in `phase5_launch.sh` is what
     pins the B60, and the mask is an *index*, not a fixed handle — so the Phase 6 memory
     check (`21.56/22.71 GiB`, not `11.xx`) must be re-read after the reboot rather than
     assumed.

  Reboot command block for the operator — `phase10_install.sh` does the unit install,
  applies 180 W immediately, and arms Phase 10's telemetry timer, so it is the single
  "put this into production" command:
  ```bash
  sudo bash /mnt/usb/vllm.xpu/phase10_install.sh   # units + 180 W cap + monitor timer
  sudo reboot
  # after boot:
  cat /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon*/power1_cap   # expect 180000000
  docker ps --filter name=vllm-xpu-b60                            # expect Up (restart policy)
  grep -m1 'Free memory on device' <(docker logs vllm-xpu-b60 2>&1)  # expect .../22.71 GiB
  ```
* \[x] Wire up ongoing monitoring (`xpu-smi dump` on a cron/systemd timer, or scrape into
whatever metrics stack you already run) for VRAM, temperature, and throttle state.
  **DONE — `sudo bash phase10_install.sh` has been run.** Both units report `enabled`,
  the timer fired at 18:31:50 CEST with the next run due in 60 s, and `b60_monitor.csv`
  is accumulating rows, e.g.
  `2026-09-26T16:31:50Z,35.9,21.3,47.0,48.0,48.0,49.0,1725,180,20,0.000,0,0,1` — note
  **`cap=180`**, confirming the same command also applied the chosen power cap (so the
  running system now matches `bench_final_180w.json`).
  Notes that made this easier than the plan assumes: `xpu-smi` is
  unusable on this host (datacenter-only, "No device discovered") and `intel_gpu_top` is
  not installed — but **hwmon is fully readable without sudo**:
  `/sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon8/` exposes `energy1`/`energy2` (card/pkg,
  µJ counters → sample for watts), `temp2..temp17` (`pkg`, `vram`, `mctrl`, `pcie`,
  `vram_ch_0..11`), `fan1_input` and `power1_cap`. `/sys/class/drm/card0/metrics/` exists
  but is **empty** (the `xe` driver has no throttle attribute), so **throttle state is
  derived as `at_cap_pct = 100 × card_W / cap_W`** — sustained ~100 % means power-limited.

  Four files, all in this directory:

  | file | role |
  |---|---|
  | `phase10_monitor.sh` | appends one CSV row; unprivileged (hwmon + localhost `/metrics`) |
  | `vllm-b60-monitor.service` | `Type=oneshot`, `User=yanchun`, one sample per call |
  | `vllm-b60-monitor.timer` | `OnBootSec=2min` + `OnUnitActiveSec=1min`, `Persistent=true` |
  | `phase10_setcap.sh` + `vllm-b60-power-cap.service` | re-apply 180 W after reboot (Phase 10 box 3) |

  Columns cover all three things this box asks for — **VRAM** via `cache_pct`
  (`vllm:kv_cache_usage_perc`, the KV pool in use; `xpu-smi` can't report VRAM here),
  **temperature** (`pkg`/`vram`/`mctrl`/`pcie`), and **throttle state** (`at_cap_pct`) —
  plus `running`/`waiting` queue depth from `/metrics`, which doubles as the liveness
  probe (`vllm_ok`). Verified against the live server, e.g. a real row:

  ```
  ts_iso,card_W,pkg_W,temp_pkg_C,temp_vram_C,temp_mctrl_C,temp_pcie_C,fan_rpm,cap_W,at_cap_pct,cache_pct,running,waiting,vllm_ok
  2026-09-26T15:59:36Z,38.2,23.6,47.0,48.0,48.0,49.0,1731,220,17,0.000,0,0,1
  ```

  The output rotates at 20 MB, so ~1 minute cadence is safe indefinitely.

  ```bash
  sudo bash /mnt/usb/vllm.xpu/phase10_install.sh        # installs + arms everything
  systemctl list-timers vllm-b60-monitor.timer           # verify it is armed
  tail -3 /mnt/usb/vllm.xpu/b60_monitor.csv              # verify rows are arriving
  journalctl -u vllm-b60-monitor -n 20 --no-pager        # if not
  ```
  The installer uses `install -m 644` rather than `cp` — `/mnt/usb` is exFAT, which stores
  no unix permissions and hands these unit files a 755 mode that systemd warns about
  (`systemd-analyze verify` passes clean once they are off exFAT).

  *Manual smoke test needs no sudo:* `bash /mnt/usb/vllm.xpu/phase10_monitor.sh`.
* \[x] Store the final reproducibility record (Phase 8 + Phase 9 results, all checklist items
from plan §8) somewhere durable before considering further optimization (draft-INT4,
larger context, etc.) — each future change is a new configuration to benchmark, not a
patch to this one.
  **WRITTEN: `/mnt/usb/vllm.xpu/REPRODUCIBILITY.md`** — hardware + PCIe evidence, image
  digest, model revision + quant contract, patch hashes, exact launch command, correctness
  and MTP metrics, both benchmark runs, power/thermal trace, all seven deviations, five
  known issues, and an artifact index.

---

## Appendix — post-plan work (after the Phase 0–10 checklist)

The checklist above is the plan as written. Everything below was done *after* it,
on the same box; each entry is a **separate configuration** with its own record in
`REPRODUCIBILITY.md` §6 (never folded into the Phase 8 baseline).

* \[x] **Port 8081 for LAN / opencode** — `serve_8081.sh` (same container as
  `phase5_launch.sh`, publishes 8000+8081, and parity-guards its `docker run`
  against the launch command of record). Container `vllm-xpu-b60`.
* \[x] **B60-only vs B60+B580 comparison** — B60-only control (XPU graphs off)
  **39.458 tok/s**; TP2 never initialized (`zeMemOpenIpcHandle …
  ZE_RESULT_ERROR_INVALID_ARGUMENT`, evidence `tp2_failure_excerpt.txt`) — no P2P
  aperture on this consumer topology. PP2 deliberately not attempted.
* \[x] **EXL3 4.00bpw alternative server** — image
  `ghcr.io/0xsero/exl3xpu@sha256:21412bdd…`, model
  `turboderp/Qwen3.8-27B-exl3 @ 113cf7ab`; `serve_exl3.sh` (host :8001, ctx 90000,
  `max_num_seqs 4`, util 0.95, MTP k=3). A/B vs GPTQ: **41.526 tok/s** vs
  **48.490** (−14.4 %), TTFT 0.363 s vs 0.469 s. Long-context smoke passed
  (84,853-token doc, needle `BLUEBERRY-7429` at 60 % depth).
* \[x] **MXFP4 coder390 alternative server** — stock `vllm-openai-xpu:v0.30.0`,
  `serve_mxfp4.sh` (host :8002, served `coder390-mxfp4`). Demonstrated max context
  **81,920**; decode **7.962 tok/s**, TTFT 1.313 s (no MTP, eager, W4A4 dynamic
  acts). MTP block present for a later spec-decode experiment.
* \[x] **Why MXFP4 is ~5× slower than GPTQ — decomposed (2026-10-17)** —
  `40.183 / 7.962 = 5.05×`, first split on the measured acceptance:
  `5.05× = 3.00× (acceptance) × 1.68× (residual)`. **⚠️ Revised by experiment 1:**
  the acceptance ratio is an *upper bound*, not a speedup — measured, MTP nets only
  `2.081×`, so the honest split is `5.05× = 2.08× (MTP) × 2.42× (the W4A4/eager
  stack)`, which makes the **stack** the larger term, not MTP. The residual stack
  penalty is ~2.42× on decode (~2.28× acceptance-corrected) against ~2.9× on
  prefill, so the two now agree in magnitude. The dominated-by-MTP claim is retracted. The 1.68× *understates* the stack penalty (a GPTQ step verifies
  5 positions vs MXFP4's 1), so prefill was used as the clean view: TTFT 0.460 s vs
  1.313 s at 512 tokens ⇒ **~2.9× slower per prompt token**, compute-bound and
  MTP-independent. Cause read from the image: `XPUMxFp4LinearKernel.apply_weights`
  runs `per_token_group_quant_mxfp4` + two fresh allocations + an fp32→E8M0 cast
  **per packed linear per forward** (496 of them) *before* a **non-fused**
  `torch.ops._xpu_C.fp4_gemm` — W4A16 GPTQ pays none of it. Not bandwidth:
  4.25 vs 4.125 bits/weight (~3 %). Also `--enforce-eager` is inherited from the
  model README, **not** forced by any XPU graph restriction. §6 records this and
  corrects the earlier "graphs second" ordering. Reordered levers: MTP ≫ W4A4 path.
* \[x] **MXFP4 + MTP (experiment 1) — done, and it revises the decomposition** —
  `serve_mxfp4_mtp.sh` vs a same-session MTP-off control (`compare_mxfp4_mtp.sh`):
  **7.987 → 16.620 tok/s = 2.081×**, TTFT 1.310 → 1.412 s (stable across 5 runs).
  **MTP is worth 2.08×, not the 3.00× acceptance ratio** — a spec step costs 1.361×
  more (125.2 → 170.5 ms: draft forward + 5-position verify), so
  `2.833 / 1.361 = 2.081×`. The acceptance ratio is an upper bound, never the gain.
  Residual with MTP on both sides: `40.183 / 16.620 = 2.42×`, so MXFP4+MTP is still
  2.42× behind GPTQ. Secondary: the draft is **less accurate under W4A4**
  (per-position `1.000, 0.833, 0.000, 0.000` = 2.83, vs GPTQ's `1.000, 1.000, …` =
  3.00) — both drafts are BF16, so the difference is the *target*, a quality symptom
  of the A4 path visible in a rate metric. Leg A also shows MXFP4 is **not** subject
  to the bimodal state (7.987 vs 7.962 recorded, +0.3 %, TTFT identical to 3 dp).
  Fit note: ctx had to drop 81920 → **40960** (the draft's BF16 weights take ~0.9 GiB
  from the same budget; 81920 needed 3.67 GiB vs 2.49 available, engine max 46592).
  Next levers are all stack-side: `--enforce-eager` off, then batched-tokens 8192.
* \[x] **coder390 GPTQ-Int4 build** — `quantize_coder390.sh` +
  `quantize_coder390_gptq.py` in `vllm-quantizer:gptqmodel7.6.0` (gptqmodel 7.6.0
  / torch 2.13.0+xpu / transformers 5.16.1), same recipe as the reference artifact,
  52 GB BF16 source streamed from disk (pileval calibration). **Four failed attempts
  and one probe** (`REPRODUCIBILITY.md` §6): 128 × 2048 dies 3/3 with
  `UR_RESULT_ERROR_DEVICE_LOST` (layers 2, 2, then 4 — so it is **stochastic**, not a
  fixed layer), `expandable_segments:False` dies of a **host OOM-kill**, and a
  **16 × 2048 probe survives**. Root cause: the `xe` driver bills GPU memory into the
  system zone (`gpu_active ≈ 35 GB` = both cards, `Free swap = 112 kB`), so VRAM and
  the 45 GB of RAM are **one budget**; a 128 × 2048 calibration (~16 GiB device +
  ~10 GiB host) overruns both. **Attempt 5 (`64 × 1024` = 65,536 tokens, ¼ budget)
  completed: `EXIT=0`, 4255 s quantize + 99 s save, 19 GB / 5 shards, 2399 tensors,
  400 `qweight`, 333 `model.visual.*` F16, 15 `mtp.*` BF16** — a field-for-field match
  to the reference. `quant_merge_mtp.py` proved unnecessary (GPTQModel 7.6.0 kept the
  MTP block). Wrapper hardened to B60-only device nodes, with `QDEV_ALLOC_CONF` /
  `QDEV_*` overrides and a `--mem-log` GPU+host+swap telemetry flag. **Served** via
  `serve_coder390_gptq.sh` (host :8003, `coder390-gptq`): healthy after ~200 s, engine
  reports **1.82 GiB KV → 17,630 tokens** (identical to the reference launch), MTP draft
  resolved, and a 256-token chat smoke test passes. Bench p512/g128 @180 W:
  **40.183 tok/s, TTFT 0.460 s, MTP k=4 mean acceptance 3.00**. Re-benched back-to-back in
  one session, the reference build gives **40.239** → **parity at −0.14 %**. The recorded
  48.490 was a **fast machine state**, not a property of the reference: this host is
  **bimodal (~40.2 / ~48.5 tok/s)** — see known issue 11. The same control exercise
  **reversed** the recorded EXL3 comparison (EXL3 41.617 vs reference GPTQ 40.239,
  same session). Consolidated same-session table in §6.
* \[x] **Long-context pp / tg on the coder390 GPTQ build** — `bench_prefill.py`
  (1024→12000 tokens, 3 reps) + `bench_p512_g128.py --prompt-tokens D --max-tokens 128`
  (512/2048/4096/8192/12000, 5 runs each), 180 W, prefix caching off, MTP k=4, on the
  live :8003 server. **pp is linear to the 12288 ceiling**: plateau **1372 tok/s** @4096,
  marginal **1268 tok/s** (R²=0.9986), aggregate 1306; going past the 8192
  `--max-num-batched-tokens` boundary is only 1372 → 1266 tok/s (chunked prefill, no
  cliff). **tg degrades monotonically with depth: 40.200 @512 → 37.315 @12000 tok/s
  (−7.2 %)**, TTFT 0.463 → 9.539 s — expected, since only 16 of 64 layers are
  `full_attention`. pp shape agrees with the recorded `prefill_180w.json` to ~3 %.
  Live llama.cpp-style `pp … | tg … | run … | kv …` console proven with
  `watch_stats.py vllm-coder390-gptq` (`coder390_gptq_pp_tg_live.log`). Caveat: this
  session ran in the **slow** bimodal state (ctx-512 = 40.200, known issue 11), so the
  curve *shape* is the transferable result. §6 + 3 new artifacts.
* \[ ] **180 W cap re-application after a reboot** — the boot race is fixed in the
  repo (`ConditionPathExistsGlob` → poll up to 90 s for the hwmon node), and the fixed
  unit is now **installed and working**: `vllm-b60-power-cap.service` reports
  `is-enabled=enabled`, `is-active=active`, `ConditionResult=yes`, `ExecMainStatus=0`,
  and `power1_cap` reads **180000000** (applied 2026-10-17 13:36:58, unit file dated
  the same minute). What is still unproven is the **boot path**: the last boot was
  2026-10-07 10:41:29, i.e. *before* this unit existed, so the cap has not yet been
  observed to come up on its own. A further reboot and a `power1_cap` re-read is the
  last open box of the plan itself.
* \[x] **Quality: 16-prompt task benchmark on the coder390 GPTQ build (2026-10-17)** —
  upstream's `bench_quality_16p.py` (fetched verbatim) run against the live `:8003` server.
  **First quality number for any build here: 15/16 (93.8 %)** — math 4/4, **code 4/4**,
  instruction 3/4, fact/trap 4/4. The lone FAIL (`format_no_letter_e`, "no letter e") is a
  **flaky item, not a stable weakness**: it spent all 2048 tokens at one budget and all
  **4096** at another (`finish_reason=length`, empty `content` both times), but the *as-is*
  run had converged on the same prompt in **17.2 s with a correct answer**, and a later
  no-parser boot looped **3/3** — so it is deterministic *within* an instance and not
  *across* instances, and the parser is not the variable (known issue 13). The number is
  therefore **15/16 ±1 item (±6.25 %)**. Getting here needed a **launch fix, not
  just a benchmark run** (next bullet). Throughput control after the fix: **40.208 tok/s /
  0.462 s TTFT** vs 40.183 / 0.460 recorded → **+0.06 %**, so the recorded rate rows stand.
  §6 + 5 artifacts.
* \[x] **Reasoning-parser parity gap found and fixed (2026-10-17)** — the engine emits
  `</think>` (token **248069**, visible in `logprobs`) but **suppresses it from `content`** and
  leaves `reasoning` null, so with no `--reasoning-parser` the trace is concatenated onto the
  answer with no delimiter. The quality harness scored **8/16** that way (code **0/4** on
  `SyntaxError` from exec-ing the reasoning; instruction **0/4** on "no letter e": 249 hits,
  word count 144 of ≤25) versus **15/16** with `--reasoning-parser qwen3`, whose
  `vllm/parser/qwen3.py` matches this checkpoint's `<think>`/`</think>` tags **by token id**.
  Added to `serve_coder390_gptq.sh`. This is a **parity gap, not a deviation** —
  `serve_exl3.sh` has always set it; `phase5_launch.sh` and `serve_8081.sh` still do not.
  Recorded as known issue 12.
* \[ ] **Quality / PPL A/B across builds** — the coder390 GPTQ build now has a task-quality
  number (15/16), but there is **no like-for-like comparison**: the reference, EXL3 and MXFP4
  have not been through the same harness in the same session, and **no perplexity has been
  measured at all**. Natural next steps: `bench_hard_suite.py` (10 harder tasks —
  long_ctx/math/code/logic_constraint, same CLI) and the **offline** `eval_exl3_ppl.py` (vLLM
  `prompt_logprobs`, no generation → one process per model, not the HTTP server). Also resolve
  the generation-config caveat: upstream sets
  `--override-generation-config '{"temperature":0.6,"repetition_penalty":1.08,"presence_penalty":0.2,"top_p":0.95}'`
  server-wide and the script does **not** override the penalties, which we deliberately did not
  adopt (it would change production sampling and invalidate the recorded rate rows).
* \[x] **`GPTQ_quantization.md` written — an operating guide for future quantizations
  (2026-10-17)** — recipe field-by-field, the one-memory-budget rule (§2, with the
  telemetry table that separates a healthy run from a dying one), the toolchain image and
  how to rebuild it, source-layout requirements (the MTP block is a *separate* file absent
  from the index), preflight, the run, artifact verification (2399/400/333/15), the MTP
  repair path, a failure catalogue of every real failure on this host, and the serving
  recipe with its three traps. Companion change: `quantize_coder390_gptq.py`'s argparse
  defaults moved from the **128 × 2048 that never completed here** to the proven-safe
  **64 × 1024**, so a bare `./quantize_coder390.sh` now reproduces the successful
  artifact (deviation 8 updated). Also corrected a stale §6 line that still described the
  build as 128 × 2048.

