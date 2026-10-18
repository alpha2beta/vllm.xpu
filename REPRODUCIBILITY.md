# Reproducibility record — vLLM XPU, Qwen3.8-27B GPTQ-Int4 + MTP, Intel Arc Pro B60

Date: 2026-09-26. Host `UideaCentre`, Ubuntu, `docker.io` (apt), no sudo available to the
agent — privileged steps were run by the operator.

Plan file `vllm_xpu_b60_deployment_plan.md` is **missing**; its sections §1.2, §4.3, §4.4,
§6.4 and §8 were sourced from the cookbook
`SergiioB/intel-arc-pro-b70-inference-cookbook` @ `88271830`,
`docs/qwen38-27b/QWEN38-VLLM-XPU.md`.

## 1. Hardware

| | B60 (target) | B580 (not used) |
|---|---|---|
| PCI address | `0000:03:00.0` (`8086:e211`, sub `172F:0300`) | `0000:0b:00.0` (`8086:e20b`) |
| DRM nodes | `card0` / `renderD128` (GID 990 `render`) | `card1` / `renderD129` |
| VRAM | **24,385,683,456 B = 22.71 GiB** (24 GB variant) | 12,168,933,376 B = 11.33 GiB |
| UUID | `868011e2-0000-0000-0300-000000000000` | `86800be2-…-0b00-…` |
| Topology | CPU root port `00:01.0` → `01:00.0` → `02:01.0` → endpoint | PCH `00:1c.4` → `09:00.0` → `0a:01.0` → endpoint |
| Link (negotiated) | **PCIe Gen4 x8** | **PCIe Gen3 x4** |
| Measured H2D / D2H | **13.85 / 11.88 GB/s** | 3.58 / 2.97 GB/s |
| hwmon | `hwmon8` (`xe`), `power1_cap` **220 W** | `hwmon9`, `power1_cap` 190 W |

VRAM source: OpenCL `CL_DEVICE_GLOBAL_MEM_SIZE` via `clinfo` — `mem_info_vram_total` does
not exist under the `xe` driver and `xpu-smi` only supports datacenter GPUs.

**PCIe note:** every readable register (kernel sysfs for both cards, and the operator's
root `lspci -vv`) reports **Gen1 x1** on the cards' internal `e2ff → e2f0 → endpoint`
hops, with `LnkCap2: Supported Link Speeds: 2.5GT/s`. Measured bandwidth is ~55× the
Gen1 x1 ceiling (~0.25 GB/s), and each card's measurement matches *its own* upstream
registers (x8 Gen4 / x4 Gen3) to ~90 % of theoretical — so the x1 reading describes an
internal fabric hop, **not** the effective link. Probe: `l0_bw.cpp` → `l0_bw`.

## 2. Container image

```
vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
```

* Registry: `HTTP/2 200`, `docker-content-digest` identical before pull.
* Local: `docker image inspect … --format '{{json .RepoDigests}}'` →
  `["vllm/vllm-openai-xpu@sha256:f01e24f6…"]` — **stop-if-mismatch passed.**
* 18 layers, 4.33 GiB compressed, 18.4 GB unpacked, amd64.
* Inside: `vllm 0.27.2rc1.dev77+gac7509e2b`, `libze_loader.so.1`, `libze_intel_gpu.so.1`,
  `intel-opencl-icd 26.27.39122.11-0`.
* Entry point is `vllm serve` — inspection commands need `--entrypoint bash`.

## 3. Model

```
SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
revision 9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e
```

* 16 files, 19 GB (18.216 GiB of safetensors), downloaded to
  `/mnt/usb/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
  (`hf download --revision … --local-dir …`; `huggingface_hub` 2.0.0, CLI is `hf`).
* Revision verified from the per-file `.cache/huggingface/download/*.metadata` commit hash.
* `quantize_config.json`: `method gptq`, `bits 4`, `sym true`, `group_size 128`,
  `desc_act false`, `lm_head false`, `dynamic {"-:.*mtp.*": {}}` → **the `mtp.*` tensors
  stay BF16**, as the cookbook requires.
* Absent from this revision: `preprocessor_config.json`, `video_preprocessor_config.json`
  — harmless because we serve `--language-model-only`.
* Architecture `Qwen3_5ForConditionalGeneration`; `text_config`: 64 layers, hidden 5120,
  24 heads, **4 KV heads, head_dim 256**, `full_attention_interval 4` (hybrid GDN +
  full attention), `mtp_num_hidden_layers 1`, `max_position_embeddings 262144`.

## 4. Patches (both verified before use)

```
4d7a02c4ea10ca7c00dc89ad927fa3dafa747dbf0553d2adf24e30a3c53e9c14  patches/patch_mtp_nightly.py
41d2f74e5fef1f074b76b5a90dd1016de437228431802cfb1fa7bd7ce4cc9b50  patches/patch_mtp_boundary.py
```

Both matched the pinned values → stop-if-mismatch passed; the cookbook had not moved on.

## 5. Exact launch command

`/mnt/usb/vllm.xpu/phase5_launch.sh` runs:

```bash
docker run -d --name vllm-xpu-b60 -p 8000:8000 --restart unless-stopped \
  --device /dev/dri --group-add 990 \
  -v /dev/dri:/dev/dri:ro \
  -v /mnt/usb/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro \
  -v …/patches/patch_mtp_nightly.py:/patch_mtp.py:ro \
  -v …/patches/patch_mtp_boundary.py:/patch_boundary.py:ro \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model \
   --quantization gptq --dtype float16 --max-model-len 12288 --gpu-memory-utilization 0.93 \
   --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 8192 \
   --no-enable-prefix-caching --served-model-name qwen38 --language-model-only \
   --enable-auto-tool-choice --tool-call-parser qwen3_xml \
   --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'
```

`ZE_AFFINITY_MASK=0` = **device index**, not a bitmask: verified with a Level Zero probe
(`0`→B60 only, `1`→B580 only, `0,1`→both, unset→both).

**Host-side prerequisite not visible in this command:** the final baseline was measured at
a **180 W** power cap, which lives outside the container in sysfs. To reproduce
`bench_final_180w.json`, set it first:

```bash
sudo /mnt/usb/vllm.xpu/phase10_setcap.sh 180   # or: systemctl enable --now vllm-b60-power-cap
```

(The 46.78 tok/s Phase 8 figures were all taken at the default 220 W.)

### LAN variant: `serve_8081.sh`

Same container, additionally published on a LAN port so opencode on another machine can
use it (`-p 8081:8000` **plus** the original `-p 8000:8000`, so every local tool that
hardcodes `127.0.0.1:8000` keeps working). It embeds a copy of the `docker run` text above
and runs a **parity check** that strips the `-p` flags from both files and diffs them,
aborting before any `docker rm` if they have drifted — so this file and `phase5_launch.sh`
cannot silently disagree about configuration.

```bash
./serve_8081.sh                 # recreate + wait for /health + print URLs
./serve_8081.sh --check         # report only
./serve_8081.sh --dry-run       # print the command
./serve_8081.sh --api-key K     # require Authorization: Bearer
```

Because it *recreates* the container, a relaunch costs a ~195 s cold start (measured).

## 6. Results

### Correctness / MTP
* `/health` 200; `/v1/models` lists `qwen38`, `max_model_len 12288`.
* Device figure in logs: **21.56/22.71 GiB → the B60**, not the B580's 11.33 GiB.
* XCCL: `backend=xccl`; GPTQ: `XPUwNa16LinearKernel`; drafter: `Qwen3_5MTP`, 4 spec tokens.
* **`SpecDecoding metrics: Mean acceptance length 3.05, Accepted 43–129, Drafted 84–252,
  per-position 0.667 / 0.619 / 0.476 / 0.286, avg draft acceptance 51.2 %`.**
* Greedy determinism: two `temperature=0` runs byte-identical.
* Warmup: first cold request 2.623 s → 1.892 s (**−27.9 %**), then stable.
* **Re-check after the 8081 relaunch** (identical config, new container): B60 re-selected
  (`Free memory on device (21.53/22.71 GiB)`), MTP architecture resolved, fp8 KV active,
  chat completion correct, cap still 180 W. Two figures moved slightly:
  | | Phase 6 | after relaunch |
  |---|---:|---:|
  | device free at startup | 21.56 / 22.71 GiB | 21.53 / 22.71 GiB |
  | **KV cache pool** | **17,630 tokens** | **18,164 tokens** |
  The KV pool varies **±3 % launch to launch** (extra 66 MiB for activations/graphs).
  Neither the fp8 conclusion nor the Phase 10 context ceiling changes: 18,164 × 128 KiB
  = 2.21 GiB, and 32768 tokens would still need 4.00 GiB — still impossible. **The
  17,630 figure quoted elsewhere in this record should be read as "≈17.6–18.2k".**

### Benchmark (plan §6.4: p512 / g128, prefix caching off, `--max-num-seqs 1`)
Rate = `(completion_tokens - 1) / (t_end - t_first_token)`, `completion_tokens` from the
API `usage` block, times from `time.monotonic()`.

**Final recorded baseline — 180 W cap, `bench_final_180w.json`:**

| metric | value |
|---|---:|
| median decode tok/s | **48.490** |
| mean | 49.057 (runs 2/4 bimodal at ~49.9 — median carried forward) |
| min / max | 48.466 / 49.929 |
| median TTFT | **0.469 s** |
| peak card / pkg power | 190.2 W / 141.4 W |
| peak `pkg` / `vram` temperature | 59 °C / 60 °C |

Superseded 220 W reference (`bench_power_default.json`): 46.781 tok/s, TTFT 0.478 s.
**Δ = +3.7 % decode, −1.9 % TTFT.** Earlier 220 W runs for the record: 46.791 (run 1),
46.781 (run 2), 46.677, 46.616 — a 0.18 tok/s band. Run 1's last sample (38.026) was a
one-off. Warmup pass 46.76 tok/s / TTFT 0.58 s.

### Prefill rate (separate sweep, `bench_prefill.py`, at the chosen 180 W cap)

Phase 8 measured only decode. Prefill was measured with a 6-point prompt-length sweep
(256 → 6144 tokens, 3 reps each), taking server-side deltas of
`vllm:request_prefill_time_seconds` / `request_prefill_kv_computed_tokens` alongside
client TTFT:

| prompt | TTFT (median) | engine rate |
|---:|---:|---:|
| 256 | 308.8 ms | 840 tok/s |
| 512 | 475.5 ms | 1,085 tok/s |
| 1,024 | 826.3 ms | 1,245 tok/s |
| 2,048 | 1,570.8 ms | 1,301 tok/s |
| 4,096 | 3,105.7 ms | **1,323 tok/s** (peak) |
| 6,144 | 4,732.8 ms | 1,303 tok/s |

* **Steady-state prefill ≈ 1,300 tok/s**, plateauing once prompts exceed ~1,024 tokens.
* **Marginal cost 0.749 ms/token → 1,335 tok/s.** Two independent fits agree: engine
  counters 1,335 tok/s and client TTFT slope 1,331 tok/s, both **R² = 0.9995**.
* **Aggregate 1,282 tok/s** (42,240 tokens / 32.956 s of engine prefill time).
* Fixed per-request overhead ≈ **74 ms** (client-observed), which is why a 256-token
  prompt only reaches 840 tok/s — short prompts pay overhead they can't amortise.
* Everything tested is ≤ 8192 tokens, i.e. a **single prefill chunk**
  (`--max-num-batched-tokens 8192`); longer prompts chunk and were not measured.
* TTFT at 512 tokens here (475.5 ms) matches Phase 8's 0.469–0.478 s, so the two
  records cross-validate.

### Power & thermal at the default cap (180 samples @ 4 Hz, `phase9_samples.csv`)

| | idle | load |
|---|---:|---:|
| card power (`energy1`) | 37.8 W | **195.9 W peak** |
| pkg power (`energy2`) | — | 144.1 W |
| temp `pkg` / `vram` | 48 / 48 °C | **62 / 62 °C** |
| temp `mctrl` / `pcie` | 49 / 51 °C | 57 / 57 °C |
| fan | 1846 RPM | 2630 RPM |

`power1_cap` = **220 W**, `power1_crit` = 440 W, `power1_cap_max` **not exposed** by the
`xe` driver, `power1_cap_interval` = 15 s. Peak = **89 % of cap** → the default cap does
not bind, which is what made the sweep below worth running.

### Power-cap curve (Phase 9) — the cap is a *performance* knob here, not just a thermal one

Every point is a full 5-run benchmark with `phase9_sampler.py` sampling power/thermals
alongside; 220 W has four independent runs, 180 W three, 150 W two.

| cap | runs | per-run median tok/s | mean of medians | Δ vs 220 W | TTFT | peak card | peak `pkg` temp |
|---:|---:|---|---:|---:|---:|---:|---:|
| 220 W (default) | 4 | 46.791 / 46.781 / 46.677 / 46.616 | **46.72** | — | 0.478 s | 195.9 W (89 %) | 62 °C |
| **180 W (chosen)** | 3 | 47.923 / 48.481 / 48.490 | **48.30** | **+3.4 %** | **0.469 s** | 190.2 W (106 %) | 59 °C |
| 150 W | 2 | 48.285 / 47.822 | 48.05 | +2.9 % | 0.488–0.501 s | 173.5 W (116 %) | 60 °C |

**Chosen cap: 180 W** — best on *both* axes (highest decode *and* lowest TTFT), 3 °C
cooler than default. 150 W ties on decode but costs ~6 % TTFT because prefill is
compute-bound and power-hungry; it remains the fallback if power draw becomes the
binding constraint.

Four findings worth carrying forward:

1. **Lowering the cap raised throughput by ~3 %**, replicated across 9 runs. The working
   hypothesis is that at 220 W the `xe` driver boost-then-rides-the-limit, while a tighter
   cap pins a steadier clock — and single-stream decode is latency-bound, so consistent
   clocks beat peak clocks. Not proven; the measurement is.
2. **Nothing is thermally limited at any cap**: 59–62 °C against a 100 °C crit
   (vram crit 105 °C), fan ≤ 2640 RPM. This is purely a power/clock effect.
3. **The cap is not an instantaneous hard ceiling** — peaks reach 106 % (180 W) and
   116 % (150 W), consistent with an average enforced over `power1_cap_interval` = 15 s.
4. **Caveat:** these are 128-token bursts at `--max-num-seqs 1`. Sustained multi-user
   load could bind differently at 150/180 W and must be re-tested before assuming the
   curve holds.

`power1_cap` (mode 664 `root:root`) needs sudo to write and is **volatile — it reverts to
the 220 W firmware default on reboot**, hence `vllm-b60-power-cap.service`.

### Dual-GPU: B60-only vs B60+B580 (2026-09-27 — control measured, TP2 impossible)

`compare_tp2.sh` derives both variants from `serve_8081.sh`'s template (never
re-typed) and prints every deviation before launching. TP2 must run with XPU graphs
off (capture is refused on multi-GPU, cookbook DUAL-B70-TP2.md), so the control is
**B60-only with graphs off** — the card is the only delta:

| configuration | decode median | TTFT median | peak card | max temp |
|---|---:|---:|---:|---:|
| B60 only, graph ON (recorded §6) | 48.490 tok/s | 0.469 s | 190.2 W | 59 °C |
| B60 only, graph OFF (control, 180 W) | 39.458 tok/s | 0.428 s | 188.9 W | 59 °C |
| B60+B580, TP2, graph OFF | **never initialized** | — | — | — |

Two results, one negative and one instructive:

1. **TP2 died in engine init on both ranks** with the exact failure the cookbook
   documents for P2P-less topologies (`zeMemOpenIpcHandle`,
   `ZE_RESULT_ERROR_INVALID_ARGUMENT`; see §8.8, evidence
   `tp2_failure_excerpt.txt`). There is no B60+B580 number to compare — the
   comparison is impossible, and that *is* the answer: the second card cannot join
   this model on this PCIe topology.
2. **Graphs off costs −18.6 % decode at 180 W (48.490 → 39.458) but *improves*
   TTFT (0.469 → 0.428 s).** Same pattern as the cap curve: prefill is
   compute-bound and likes the untethered burst; decode is latency-bound and the
   graph-less execution is peaky enough to ride the 180 W cap (peak 188.9 W =
   105 %). Corroborating detail: with graphs off the MTP draft positions 3–4
   accepted **0.000** across all snapshots (mean acceptance 2.92–2.99 vs 3.05
   recorded) — the last two of four draft tokens were pure waste. Whether that is
   graph-off numerics or a small-sample artefact is not separated; do not quote
   MTP acceptance from a graph-off run.

### EXL3 4.00bpw context estimate for B60 (2026-10-05 — analysis only, not deployed)

Checked `0xSero/exl3xpu` (`models/qwen3.8-27b-exl3-4.00bpw/model.yaml`,
`recipe.json`): B70 holds 262,144 ctx with weights 14.91 GiB + fp8 KV
~8.4 GiB for 256K (pool 272,570 in 1600-token blocks, `util 0.965`,
`max_num_seqs 16`, MTP k=3, vision included). Our own checkpoint's
`config.json` confirms the architecture this implies: 64 layers =
**48 `linear_attention` (GDN, fixed-size state) + 16 `full_attention`**,
so fp8 KV/token = 16×4×256×2×1 B = **32 KiB** (256K × 32 KiB = 8.0 GiB,
+~5 % block overhead = repo's 8.4 GiB ✓). `max_position_embeddings`
is natively 262144, so the limit is memory, not RoPE.
**Correction to §7.3:** the "128 KiB/token (64 layers)" geometry there is
wrong — only 16 layers store paged KV. The measured pool (17,630 tokens
from 1.82 GiB ⇒ ~110 KiB/token effective) still stands; the gap is vLLM
0.27's unified-KV padding/GDN state on our stack, which is exactly why the
EXL3 plugin's leaner 32 KiB/token matters. Our 16,384 ceiling is unchanged
for the GPTQ stack.
Scaling to B60 (22.71 GiB, same 14.91 GiB weights):

| config | budget | non-KV overhead | KV left | pool ≈ | max ctx ≈ |
|---|---:|---:|---:|---:|---:|
| lean, seqs 1–2, text-only, 0.93 | 21.12 | ~2.0 | 4.21 GiB | ~138k | **~128k** |
| typical, seqs 4–8, text-only, 0.95 | 21.57 | ~3.2 | 3.46 GiB | ~113k | **~109k** |
| typical + vision tower (~0.9 GiB) | 21.57 | ~4.1 | 2.56 GiB | ~84k | **~81k** |
| recipe-like, seqs 16 + vision, 0.965 | 21.92 | ~5.5 | 1.51 GiB | ~49k | **~48k** |

So: **~48k pessimistic / ~80–110k realistic / ~128k lean** — roughly
3–8× our current 16k. 262,144 is impossible (needs 8.4 GiB KV + 14.91
weights = 23.3 GiB > 22.71 before any overhead; zero-overhead ceiling is
~230k pool). Caveats: different stack (their vLLM 0.26.1 + llm-scaler +
exl3xpu plugin image), validated on B70/BMG-G31 only, and vision costs
~1 GiB unless `--language-model-only` equivalent is used. Verify empirically:
launch with a generous `--max-model-len` and read back the engine's
reported KV pool, as in §6.
EXL3 vs MXFP4 (2026-10-06, analysis): EXL3 is a QTIP-family trellis codebook
VQ (non-uniform, targetable fractional bpw); MXFP4 is the OCP standard
fixed format (E2M1, block-32, E8M0 scale = 4.25 bits/el). Literature and
publisher numbers put codebook VQ ahead on PPL-per-bit (e.g. Qwen3-235B
EXL3-4.0 KL 0.046 vs orig; MXFP4 RTN ≈ +0.3 PPL vs uniform INT4 ≈ +0.8 on
Llama-class). Decisive on B60: EXL3 has ESIMD XPU kernels + a ready
Qwen3.8-27B checkpoint (weight-only decode); MXFP4 has no XPU kernel path
in vLLM (Marlin MXFP4 is CUDA-only, Intel-GPU column ❌; Intel docs list
only W4A16/W8A16 as deployable on Intel) — a paper format on Battlemage.

### EXL3 A/B vs GPTQ on B60 (2026-10-05 — `serve_exl3.sh`, port 8001, both @180 W)

Deployed `ghcr.io/0xsero/exl3xpu@sha256:21412bdd` (31.8 GB image) +
`turboderp/Qwen3.8-27B-exl3 @ 113cf7ab` with B60 `--set` overrides
(ctx 90000, seqs 4, util 0.95). Healthy after ~260 s. Engine reports
**pool 91,282 tokens from 4.09 GiB** on first boot (97,500 on the next —
±3 % launch-to-launch allocator variance, same as the GPTQ pool's
17,630→18,164; ⇒ ~47 KiB/token effective vs 32 KiB
theory — hybrid padding/GDN pages; estimate band §6 confirmed, slightly
under the 113k "typical" guess because weights resident are **17.34 GiB**,
not 14.91). Same `bench_p512_g128.py`:

| build | decode median | TTFT median | ctx / pool | MTP |
|---|---:|---:|---|---|
| GPTQ-Int4 (recorded §6) | 48.490 tok/s | 0.469 s | 12288 / 17,630 | k=4, accept 3.05 |
| **EXL3 4.00bpw** | **41.526 tok/s (−14.4 %)** | **0.363 s (−22.6 %)** | 90000 / ~91–98k | k=3, accept ~2.92 |

Read: EXL3 trades ~14 % decode for ~23 % faster prefill (int8 prefill +
oneDNN SDPA working) and **5.3× the context** (65,536 usable, pool would
allow ~90k). > **⚠️ Corrected 2026-10-17 (known issue 11) — the decode ordering inverts.** These two rows
> were recorded on different days with no same-session control, and GPTQ decode throughput on
> this host is **bimodal (~40.2 / ~48.5 tok/s)**. Re-measured back-to-back today
> (`bench_exl3_rerun.json` vs `bench_reference_rerun.json`):
>
> | build | decode median | TTFT median |
> |---|---:|---:|
> | GPTQ (reference `qwen38`) | 40.239 tok/s | 0.461 s |
> | **EXL3 4.00bpw** | **41.617 tok/s** | **0.363 s** |
>
> In the prevailing state EXL3 is **+3.4 % on decode**, not −14.4 %, *and* 21 % faster to first
> token. The 48.490/41.526 pairing had compared a fast-state GPTQ run against a normal-state
> EXL3 run. (GPTQ did reach 48.5 in one of five runs today; EXL3 held 41.6 across all five,
> with a single 37.4 excursion.) The **context** advantage (90000 vs 12288) was never in
> question, so EXL3's case is stronger than originally recorded, not weaker.

Confounders, stated not hidden: different stacks
(vLLM 0.27.2 vs 0.26.1+plugin), different MTP k (4 vs 3), draft vocab
pruned vs full. Quality (PPL/side-by-side) A/B not run — rate only.
Note: their `qwen3_5_mtp` name logs a deprecation warning (resolves to
`mtp` internally); harmless. Only one server runs at a time (resident
~18.1 vs ~17.3+4.1 GiB); `serve_exl3.sh` refuses while `vllm-xpu-b60` is up.
Long-context smoke (`longctx_smoke.py`, 2026-10-05): 84,853-token doc
(repo records repeated, `BLUEBERRY-7429` needle at 60 % depth) → retrieved
exactly, `finish=stop`, 10 completion tokens. Cold 85.3 s (~995 tok/s
prefill-incl.), prefix-cached rerun 2.9 s (~29.5k tok/s) — caching works at
85k. (First attempt at 56k returned empty text: max_tokens 64 starved the
thinking trace; 256 fixed it.)
Live pp/tg: both servers already log `Engine 000: Avg prompt/generation
throughput` ticks; `watch_stats.py [container]` reprints them
llama.cpp-style (`[14:59:25] pp 0.0 t/s | tg 0.0 t/s | run 0 wait 0 |
kv 0.0% | hit 0.0%`). Tick interval = `VLLM_LOG_STATS_INTERVAL` (default
10 s; `serve_exl3.sh` sets 2 s — logging only, no perf effect; GPTQ
scripts untouched so the recorded launch stays byte-identical). Ticks are
per-engine window aggregates, not per-request — exact per-request numbers
still come from `bench_p512_g128.py`.

### MXFP4 coder390 on B60 (2026-10-16 — `serve_mxfp4.sh`, port 8002, @180 W)

Checkpoint `/mnt/stk/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-
abliterated-MXFP4` (compressed-tensors `mxfp4-pack-quantized`, 496 packed
linears, bf16 MTP block) served by stock `vllm-openai-xpu:v0.30.0`
(`XPUMxFp4LinearKernel`, W4A4 dynamic-activation fp4_gemm — no plugin;
README's `torch.ops.vllm.xpu_mxfp4_quantize` name is slightly off, the
real path is `mxfp4_utils` + `torch.ops._xpu_C.fp4_gemm`). Scheme matches
the kernel gate exactly (static group-32 weights + dynamic group-32 acts).
First boot (ctx 32768): **pool 83,740 tokens from 3.4 GiB** (⇒ ~42.6
KiB/token), weights+non-torch 17.58 GiB, eager (0.0 graphs). Ceiling run
(ctx 81920): healthy, concurrency 1.19x, probe OK → **81,920 is the
demonstrated max**. Pool varies boot-to-boot (83,740 → 97,456, ±8 %),
so 93k (fits only high-roll boots) would be fragile; 81920 fits even the
low roll with ~1,800 tokens margin. MTP off (README config); bf16 mtp.*
present for a later spec-decode experiment. Rate A/B, same harness
(`bench_mxfp4_p512_g128.json`, 2026-10-16):

> **⚠️ Cross-session caveat (2026-10-17, known issue 11).** The GPTQ and EXL3 rows were
> recorded on different days; re-measuring them same-session reversed their decode ordering
> (see the EXL3 section above). Only the MXFP4 row is unaffected — its ~5× gap is far wider
> than the ~20 % bimodal band. See the coder390 subsection in §6.

| build | decode median | TTFT median | ctx | MTP |
|---|---:|---:|---|---|
| GPTQ-Int4 | 48.490 tok/s | 0.469 s | 12288 | k=4 |
| EXL3 4.00bpw | 41.526 tok/s | 0.363 s | 90000 | k=3 |
| **MXFP4 coder390** | **7.962 tok/s** | **1.313 s** | 81920 | off |

Runs 1–5 identical to 3 decimals (7.962–7.964, TTFT 1.313 ×5) — a fixed
per-token cost dominates, not kernel noise. ~6× behind GPTQ decode is the
stack of three handicaps, all stated: **no MTP** (worth ~2.5–3× alone at
these acceptance rates), **eager** (no graphs), and W4A4 dynamic-act
quantization per step. Biggest lever is clearly MTP (bf16 draft present);
graphs second. Quality comparison (coder390 abliterated vs base Qwen3.8)
not run — rate only, and the weights differ anyway.

> **⚠️ Superseded in part (2026-10-17):** "graphs second" is wrong — see the decomposition
> below. The second-largest term is the W4A4 activation-quantisation path itself; `eager`
> is only a modifier on it, and it is inherited from the model README rather than forced.

### Why MXFP4 decode is 5× slower than GPTQ — decomposition (2026-10-17, analysis + kernel source)

Asked to justify the ~5× gap, so it was decomposed rather than restated. Two separate
things are called "4-bit" here and they are **different compute schemes**:
GPTQ is **W4A16** (4-bit weights, fp16 activations); MXFP4 is **W4A4** (4-bit weights *and*
4-bit activations, quantised dynamically per step).

**The factorisation.** `40.183 / 7.962 = 5.05×`, and it splits exactly on the measured
acceptance length:

| | MXFP4 | coder390 GPTQ |
|---|---:|---:|
| decode | 7.962 tok/s | 40.183 tok/s |
| tokens per forward step | 1.00 | **3.00** (`Mean acceptance length`) |
| ms per step | 125.6 | 74.7 |
| **ms per emitted token** | **125.6** | **24.9** |

> **5.05× = 3.00× (MTP, as an *upper bound*) × 1.68× (residual)** — see the ⚠️ below;
> the 3.00× is the **acceptance ratio**, and the acceptance ratio is not a speedup.

MTP alone is therefore **at most ~60 % of the entire gap** — and the checkpoint already ships
the draft: every `mtp.*` tensor is in the compressed-tensors `ignore` list
(`mtp.fc`, `mtp.layers.0.*`, `mtp.norm`, `mtp.pre_fc_norm_*`), so it was never quantised.
That is the basis of experiment 1 below.

**The 1.68× understates the stack penalty.** A GPTQ step verifies up to 5 positions while
MXFP4's verifies 1, so comparing step-to-step makes MXFP4 look *better* than it is. Prefill
is the clean view — MTP contributes nothing there, and it is compute-bound:

| 512-token prompt | TTFT | implied rate |
|---|---:|---:|
| GPTQ | 0.460 s | ~1,285 tok/s (from the §6 sweep: 0.789 ms/token marginal) |
| MXFP4 | 1.313 s | ~410 tok/s |

⇒ **~2.9× slower per prompt token** (≈3.1× once the ~60 ms fixed TTFT overhead is netted
out). A slower *per-FLOP* path, not memory traffic.

**Where the per-FLOP cost comes from — read from the image, not inferred.**
`vllm/model_executor/kernels/linear/mxfp4/xpu.py`:

```python
class XPUMxFp4LinearKernel(MxFp4LinearKernel):
    def apply_weights(self, layer, x, bias=None):
        out_dtype = x.dtype
        x_fp4, x_blockscale = quant_mxfp4(x)        # <- every single call
        return torch.ops._xpu_C.fp4_gemm(x_fp4, layer.weight,
                                         x_blockscale, layer.weight_scale,
                                         out_dtype, bias)
```

and `_xpu_mxfp4_quantize_impl` in `vllm/_xpu_ops.py`:

```python
x_q = torch.empty(M, N // 2, ...)                    # packed FP4, allocated per call
x_s = torch.empty(M, N // 32, ...)                   # block scales,  allocated per call
torch.ops._C.per_token_group_quant_mxfp4(x, x_q, x_s, MXFP4_BLOCK_SIZE, eps)
x_s = x_s.to(dtype=torch.float8_e8m0fnu, ...)        # extra cast pass
```

So for each of the **496 packed linears, every forward**, MXFP4 pays: a group-32 activation
quantisation kernel, an fp32→E8M0 scale cast, and two fresh allocations — *then* a separate
GEMM. Quantise and GEMM are **not fused**, so the floor is ~992 launches per step before
norms/attention/GDN. **GPTQ pays none of this**: fp16 activations go straight into the fused
dequant-GEMM. Ranked:

1. **W4A4 dynamic activation quantisation** (~3× per FLOP above) — the dominant stack term.
2. **Non-fused FP4 GEMM** (`torch.ops._xpu_C.fp4_gemm`) as a second launch per linear.
3. **`--enforce-eager`** — amplifies 1–2, since none of those launches replay from a graph.
   Not demonstrably *required*: `platforms/xpu.py` lists `mxfp4` under `supported_quantization`
   with no graph restriction, and `serve_mxfp4.sh`'s own comment says the flag came from the
   model README's measured config. Untested, not forced.

**What it is *not*: bandwidth.** E2M1 4-bit + E8M0 scale per 32 = **4.25 bits/weight**; GPTQ
Int4 + fp16 scale per 128 = **4.125 bits/weight** — a ~3 % difference. A bandwidth story
predicts near-parity, not 5×. Two further confirmations that it is compute: the five runs sat
within **0.02 %** of each other (a fixed per-token cost, not noisy memory), and prefill — the
pure-compute regime — is ~2.9× off on its own. The gap is also far outside the ~20 % bimodal
band (known issue 11), so unlike the EXL3/GPTQ ordering it is real and build-attributable.

**Consequence for the MXFP4 follow-ups** (all three need no rebuild): enable MTP (predicted
~24 tok/s at the same acceptance), then drop `--enforce-eager`, then
`--max-num-batched-tokens 8192` (4096 currently splits a 12k prompt into 3 chunks vs 2).

### Experiment 1 — MTP on MXFP4, and why the acceptance ratio is not a speedup (2026-10-17)

`serve_mxfp4_mtp.sh` (MTP k=4, everything else identical to `serve_mxfp4.sh`) vs a
**same-session** re-measured MTP-off control, `compare_mxfp4_mtp.sh`, `bench_p512_g128.py`
p512/g128, 1 warmup + 5 runs each:

| leg (same session, 2026-10-17) | decode median | TTFT median | runs |
|---|---:|---:|---|
| A — MTP off (control) | **7.987 tok/s** | 1.310 s | 7.985 / 7.987 / 7.985 / 7.987 / 7.987 |
| B — MTP on, k=4 | **16.620 tok/s** | 1.412 s | 16.620 / 16.617 / **16.279** / 16.625 / 16.623 |
| | **2.081×** | +7.8 % | |

Leg A also settles an open question: 7.987 vs the recorded 7.962 (2026-10-16) is **+0.3 %**, and
TTFT was identical to 3 decimals on all five runs — so **MXFP4's rate is not subject to the
bimodal state** (known issue 11), which retroactively validates that cross-session row (unlike
the EXL3/GPTQ ordering, which really did need re-measuring).

> **⚠️ Correction to the decomposition above.** MTP buys **2.08×, not 3.00×.** The recorded
> 3.00× was the *acceptance ratio*, and reading it as a speedup was wrong: a speculative step
> is itself more expensive. Measured — leg A is 1 token per 125.2 ms step; leg B accepts 2.833
> tokens per 170.5 ms step, i.e. the spec step costs **1.361×** more (draft forward + 5-position
> verify). Net = 2.833 / 1.361 = **2.081×**, exactly the measured figure.
> **The acceptance ratio is an upper bound on MTP's net gain, never the gain itself.**

**Corrected factorisation.** With MTP enabled on both sides the residual is
`40.183 / 16.620 = 2.418×`:

> **5.05× = 2.08× (MTP, now measured) × 2.42× (the MXFP4 W4A4/eager stack)**

which moves the *stack*, not MTP, into first place — the opposite of what this section said
an hour ago. Acceptance is not quite equal either (2.83 vs GPTQ's 3.00); normalising to
acceptance 3.00 gives 17.60 tok/s and a **2.28×** stack residual. So on decode the stack costs
**~2.3–2.4×**, against **~2.9×** on prefill — the two now agree in magnitude instead of
disagreeing, which is the main reason to trust the corrected split. The "MTP ≈ 60 % of the gap"
claim is retracted; by factor MTP is the *smaller* of the two terms.

Two secondary findings worth keeping:

* **The draft is less accurate under MXFP4.** Per-position acceptance is
  `1.000, 0.833, 0.000, 0.000` (2.83) where the GPTQ build gives `1.000, 1.000, 0.000, 0.000`
  (3.00) — the second position slips. Both drafts are BF16; the difference is the **target**
  (W4A4 vs W4A16), i.e. noisier hidden states reduce draft/target agreement. That is a quality
  symptom of the A4 path showing up in a *rate* metric, and a second reason the open PPL A/B
  matters.
* **TTFT regresses 7.8 %** (1.310 → 1.412 s, stable across runs) — MTP drafting costs something
  before the first token is emitted.

**Fit note:** the first leg-B attempt failed to initialise at ctx 81920 —
`3.67 GiB KV cache is needed, but only 2.49 GiB available … estimated maximum model length is
46592`. Not an MTP or quantization failure: the draft loaded correctly (no `/patch_mtp.py`
needed, as predicted from the `ignore` list) but its BF16 weights come out of the same budget
(known issue 10), so leg A's 3.4 GiB of KV became 2.49 GiB. ctx was therefore reduced to
**40960**; ctx does not affect the p512 decode rate being measured. **MTP costs ~0.9 GiB of
weights and therefore context on this card.**

**Conclusion: MTP recovers 2.08× and leaves MXFP4 still 2.42× behind GPTQ** (16.620 vs
40.183 tok/s). The remaining levers are all on the stack, in the order recorded above:
drop `--enforce-eager`, then `--max-num-batched-tokens 8192`.

### coder390 GPTQ-Int4 — quantization + B60-isolation fix (2026-10-17, in progress)

A **same-recipe** GPTQ-Int4 (sym, g128, `desc_act=False`, `lm_head=False`,
pack `int32`, `dynamic={"-:.*mtp.*": {}}`) copy of the **Huihui Qwen3.8-27B
Coder390 EfficientThink abliterated** BF16 model, built with
`quantize_coder390.sh` / `quantize_coder390_gptq.py` inside
`vllm-quantizer:gptqmodel7.6.0` (gptqmodel 7.6.0, torch 2.13.0+xpu,
transformers 5.16.1). The 52 GB BF16 source (`…-bf16`, 2 shards + a separate
`mtp-bf16.safetensors`) is streamed module-by-module from disk
(`offload_to_disk=True` + LazyTurtle) because it does **not** fit in the 45 GB
host RAM. Calibration: pileval (`mit-han-lab/pile-val-backup`,
`split=validation`, streaming, shuffle buffer 10000), **64 chunks × 1024 tokens**
(65,536) — the reference's calibration set is unknown *and* the size is cut 4× from the
classic 262,144 budget; see deviation 8. `mtp.*` is re-injected afterwards by `quant_merge_mtp.py`, since
transformers' `Qwen3_5ForConditionalGeneration` ignores `^mtp.*`.

**Attempt 1 (local 10:11 / UTC 08:11) died at layer 2 of 63** with
`RuntimeError: level_zero backend failed with error: 20
(UR_RESULT_ERROR_DEVICE_LOST)`, raised in
`gptqmodel/looper/forward_executor.py:282 → utils/model.py:265 move_to`. The
root cause is in the kernel log, not Python:

```
10:14:33 xe 0000:0b:00.0: cannot be used for peer-to-peer DMA ... (0000:03:00.0)
10:18:46 xe 0000:0b:00.0: Timed out wait for G2H, fence 30295, action 0509
10:18:46 xe 0000:0b:00.0: failed to enable GuC scheduling policies: -ETIME
10:19:05 xe 0000:03:00.0: Suspend fence, guc_id=15, failed to respond
10:19:05 xe 0000:03:00.0: TLB invalidation fence timeout, seqno=192189 recv=192188
10:19:05 iwlwifi 0000:00:14.3: Device error - SW reset
10:22:01 xe 0000:03:00.0: Engine reset: engine_class=ccs ... Timedout job ... in python3
10:22:01 xe 0000:03:00.0: Xe device coredump created
```

An initial reading blamed the B580 (the wrapper originally exposed all of
`/dev/dri`, so level-zero enumerated both cards and probed B60↔B580 P2P — the
`cannot be used for peer-to-peer DMA` line above). **That reading was wrong.**
Attempt 2 was hardened to B60-only device nodes
(`--device renderD128 --device card0`, no `/dev/dri` bind; a health probe then
confirms `device_count() == 1`) and it died in the *same* place with a kernel log
containing the B60 reset **only** — no B580 involvement at all.

### Attempt log — stochastic, and memory-driven

| attempt | config | outcome |
|---|---|---|
| 1 | 128×2048 | `DEVICE_LOST` at layer 2, 10:18 elapsed |
| 2 | 128×2048, B60-only nodes | `DEVICE_LOST` at layer 2, 9:05 elapsed |
| 3 | 128×2048, + VRAM logger | `DEVICE_LOST` at **layer 4**, 15:33 elapsed |
| probe | **16**×2048 | ran 17 min / 18+ layers, **no reset** |
| 4 | 128×2048, `expandable_segments:False` | **host OOM-kill** (SIGKILL), 2.6 min |
| 5 | **64×1024**, default allocator | **SUCCESS** — `EXIT=0`, 4255 s quantize + 99 s save |

Ruled out along the way: **thermal/power** (~100 W, 54–62 °C at the crash;
`power1_cap` reads 0/unset after the reboot), **the B580** (attempt 2), **a fixed
layer** (2, 2, then 4), **elapsed time alone** (the 16-sample probe outlived both
128-sample runs), and **the desktop** (with the crashed container killed, an idle
B60 reports 24,214/24,480 MiB free — GNOME costs ~260 MiB).

**Root cause — VRAM and host RAM are one budget on this host.** Attempt 4's
kernel OOM report is explicit:

```
Free swap = 112kB / Total swap = 4194300kB
active_anon:3661584kB inactive_anon:7208328kB ... all_unreclaimable? yes
gpu_active:35480432kB gpu_reclaim:0kB
Out of memory: Killed process 160687 (python3) anon-rss:9794776kB
```

`gpu_active ≈ 35.4 GB` ≈ the B60's 24 GB + the B580's 12 GB: **the `xe` driver
bills GPU allocations into the system memory zone**, so a full GPU exhausts host
RAM and swap with it. The 128-sample calibration needs ~16 GiB of device-side
activation/Hessian buffers *and* ~10 GiB of host-side captured calibration data
(`calibration_data_device` defaults to CPU), which is enough to push this 45 GB
host over. The GPU `ccs` engine timeouts in attempts 1–3 are that same pressure
surfacing as a GPU page-fault/stall instead of an OOM-kill. Cutting calibration
rows relieves both sides at once — which is exactly why 16 samples survived and
128 did not.

Also relevant: a crashed run **leaves a zombie container holding the whole GPU**
(it prints the traceback but never exits; `docker stats` showed `Up` while
`free=6 MiB`). The wrapper `docker rm -f`s it before each start, so runs do begin
clean — but any manual VRAM reading taken before that kill is meaningless.

**Attempt 5 succeeded.** It keeps every recipe field and cuts only the calibration
to 64 chunks × 1024 tokens (65,536 tokens — ¼ of the classic 262,144 budget), a 4×
cut to the footprint on both sides; deviation 8 records it. Telemetry stayed
**flat** for the whole run, which is what the 128-sample attempts never did:

```
                      128×2048 (att. 1-4)     64×1024 (att. 5)
GPU free              ~0-18 MiB               13,031-16,717 MiB (flat)
GPU reserved          33-50 GiB               10,236-10,756 MiB (flat)
host available        swap exhausted          35,200-37,672 MiB (flat)
```

```
[11:24:17] quantize finished in 4255 s (70.9 min)
[11:25:57] saved in 99 s
[11:25:57] TOTAL 4356 s
EXIT=0
```

**Artifact** — `/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-GPTQ-Int4-sym-G128-MTP-BF16`
(19 GB, 5 shards):

| group | dtype | count |
|---|---|---:|
| `*.qweight` (quantized linears) | I32 | **400** |
| `model.visual.*` (vision tower) | F16 | **333** |
| `mtp.*` (draft block, `dynamic`-excluded) | **BF16** | **15** |
| other (qzeros/scales/norms/embed) | I32 / F16 | 1651 |
| **index total** | | **2399** |

That is a field-for-field match to the reference checkpoint (2399 tensors, 400
`qweight`, 333 visual, 15 BF16 `mtp.*`), and `quantize_config.json` carries the
reference recipe (`bits 4`, `group_size 128`, `sym true`, `desc_act false`,
`lm_head false`, `pack_dtype int32`, `pack_impl cpu`, `true_sequential true`,
`act_group_aware true`, `damp_percent 0.05`, `damp_auto_increment 0.01`,
`static_groups false`, `mse 0.0`, `fallback rtn/"0.5%"`,
`hessian.staging_dtype float32`, `dynamic {"-:.*mtp.*": {}}`) with
`meta.quantizer = gptqmodel:7.6.0`. **`quant_merge_mtp.py` was not needed** —
GPTQModel 7.6.0 preserved the 15 `mtp.*` tensors in BF16 rather than dropping
them, so the MTP draft is already index-addressable.

Served as `serve_coder390_gptq.sh` (host :8003, `coder390-gptq`).

**Evidence** (verbatim, in this directory): `quant_attempt1_device_lost.txt`,
`quant_attempt2_device_lost_b60only.txt`, `quant_attempt3_diag128.log`,
`quant_attempt4_host_oom.log`, `quant_probe16.log`, `quant_attempt5_success.log`.

**Serving gotcha.** The first cold-start engine init of this model fails exactly as
known issue 1 describes (`1.24 GiB KV cache is needed … available 0.18 GiB`) and the
container exits (1). Recovery depends on `--restart unless-stopped` — Docker restarts
the *same* container, whose warm `torch.compile` cache then fits the KV budget. The
initial version of `serve_coder390_gptq.sh` omitted the flag and the server simply
never came up; it is now carried explicitly.

**Serving result (`bench_coder390_gptq_p512_g128.json`, 180 W cap, MTP k=4).** Healthy
after ~200 s; the engine reports **1.82 GiB KV → 17,630 tokens**, exactly the recorded
reference GPTQ figure; MTP draft resolved (`B70_MTP_BF16_DRAFT=1`, sharing the target's
embedding and `lm_head` weights); `GDN decode kernel: triton` (the same fallback the
reference records in known issue 5), `quantization=auto_gptq`, `enforce_eager=False`.

| metric | value |
|---|---:|
| decode median (5 runs) | **40.183 tok/s** (40.175–40.187, ±0.02 %) |
| TTFT median | **0.460 s** |
| ctx / KV pool | 12288 / 17,630 |
| MTP | k=4, `Mean acceptance length 3.00`, per-position `1.000/1.000/0.000/0.000` |

Smoke test (`/v1/chat/completions`, `max_tokens` 256, `temperature` 0) returns a coherent
answer with `finish_reason=length`, i.e. the budget — not a fault — ended it.

That 40.183 sits **17 % below the 48.490** recorded for the reference build, and per-token
kernel cost does not depend on weight *values*, so a same-session control was run instead
of assuming: the **reference** model (`qwen38`) was restarted on :8000 and benched with the
same harness at the same 180 W cap, minutes apart.

| bench (same session, 2026-10-17) | run1 | run2 | run3 | run4 | run5 | median |
|---|---:|---:|---:|---:|---:|---:|
| recorded baseline (2026-09-26) | 48.466 | 49.913 | 48.486 | 49.929 | 48.490 | **48.490** |
| **reference, re-run now** | 40.239 | 40.228 | 40.224 | **48.496** | 40.245 | 40.239 |
| **coder390 GPTQ (ours)** | 40.187 | 40.183 | 40.184 | 40.175 | 40.179 | **40.183** |

Three conclusions:

1. **Our build is at parity with the reference** — 40.183 vs 40.239 = **−0.14 %**, measured
   back-to-back. The 64 × 1024 calibration deviation costs nothing in throughput, and the
   KV pool is identical (17,630 tokens).
2. **The host is bimodal, ~40.2 or ~48.5 tok/s, and the recorded 48.490 was captured
   entirely in the fast state.** The reference emits *both* states inside a single bench
   session (runs 1–3 and 5 slow, run 4 fast), and acceptance cannot explain a 20 % step-time
   swing — `Mean acceptance length` moved only 3.00 → 3.18, worth ~6 %. The card is also
   6–12 °C hotter than in September (52–60 °C vs 46–48 °C; fan ~1,800–2,260 rpm vs ~1,700),
   which points at a clock/power state, though a graph-vs-eager execution path is not ruled
   out. **The mechanism is not identified** — only that it is a step-time effect, real, and
   reproducible in-session. See known issue 11.
3. **All rates in this record are only comparable within one session** (known issue 11).
   The GPTQ 48.490 leg was recorded 2026-09-26 in the fast state; the EXL3 and MXFP4 legs
   were recorded 2026-10-05 and 2026-10-16 **without** a same-session control. That is not
   hypothetical — it had already **inverted a published conclusion**. Re-measured
   back-to-back today, `bench_exl3_rerun.json` gives EXL3 **41.617 tok/s / 0.363 s** against
   this session's reference GPTQ **40.239 / 0.461 s**: EXL3 is ~3.4 % **faster** on decode
   and 21 % faster to first token, where the cross-session pairing had shown it 14.4 %
   *slower*. The MXFP4 gap (7.962, ~5×) is far too wide for a 20 % state to explain and
   stands.

**Consolidated same-session comparison (2026-10-17, all 180 W, `bench_p512_g128.py`, 5 runs):**

| build | decode median | TTFT median | ctx | MTP | measured |
|---|---:|---:|---|---|---|
| **coder390 GPTQ-Int4 (this build)** | **40.183 tok/s** | **0.460 s** | 12288 | k=4, acc 3.00 | today |
| reference GPTQ-Int4 (`qwen38`) | 40.239 tok/s | 0.461 s | 12288 | k=4, acc 3.00–3.18 | today |
| EXL3 4.00bpw | 41.617 tok/s | 0.363 s | 90000 | k=3, acc 2.6–3.0 | today |
| MXFP4 coder390 | 7.962 tok/s | 1.313 s | 81920 | off | 2026-10-16; **re-run today: 7.987** |
| MXFP4 coder390 **+ MTP k=4** | **16.620 tok/s** | 1.412 s | 40960 | k=4, acc 2.83 | today (experiment 1) |

Read: the new coder390 GPTQ build matches the reference build to **−0.14 %** and inherits its
17,630-token pool; EXL3 leads decode by ~3.4 % and TTFT by 21 % while offering 7.3× the
context; MXFP4 remains ~5× behind. Its gap **is** now re-measured (experiment 1): 7.987 today
confirms the recorded 7.962, enabling MTP lifts it to 16.620 (2.081×), and it is still **2.42×
behind GPTQ with MTP on both sides** — the residual is the W4A4/eager stack, not MTP. It is
also the one build whose rate is provably **not** subject to the bimodal band. Quality
(PPL / side-by-side) is still **rate-only** for all of them.

### Long-context pp / tg on the coder390 GPTQ build (2026-10-17 — `bench_prefill.py`, `bench_p512_g128.py`)

Both halves of the rate measured on the new build, on the live :8003 server, 180 W, MTP k=4,
prefix caching off (so every prompt token is really computed). "Long context" here means **up to
12000 tokens**, because 12288 is this build's structural ceiling: the engine has a 17,630-token KV
pool, so a single request can only reach `12288` (`12000 + 128` fits; `12288 + 8` does not), and the
`--max-num-batched-tokens 8192` setting splits any prompt above 8192 into two forwards — noted
below rather than left to look like a cliff. The EXL3 server is the one that goes to 90k.

**pp (prompt processing) — server-side prefill counters, 3 reps each:**

| prompt tokens | TTFT median | server prefill | rate |
|---:|---:|---|---:|
| 1024 | 801.2 ms | 3072 tok / 2.391 s | 1285 tok/s |
| 2048 | 1520.6 ms | 6144 tok / 4.574 s | 1343 tok/s |
| 4096 | 2996.8 ms | 12288 tok / 8.955 s | **1372 tok/s** (plateau) |
| 8192 | 6196.1 ms | 24576 tok / 18.520 s | 1327 tok/s |
| 12000 | 9500.9 ms | 36000 tok / 28.431 s | 1266 tok/s |

Aggregate 1,306 tok/s (82,080 tok / 62.872 s). Least-squares fits: marginal **1,268 tok/s**
(0.789 ms/token, R² = 0.9986) server-side and 1,264 tok/s client-side (0.791 ms/token, intercept
−125 ms); the plateau is 1,372 tok/s and 90 % of it is reached by ~1024 tokens. Prefill is linear
in length across the whole range — the 12000 point is **not** a cliff: chunked prefill (8192 +
3808) costs only the usual 2-chunk amortisation, 1372 → 1266 tok/s. Against the recorded
`prefill_180w.json` (reference GPTQ, same cap) the shapes agree to ~3 % (that sweep: plateau 1323,
marginal 1335, 256→6144 only), so long-context prefill is **at parity** too.

**tg (token generation) vs context depth** — `bench_p512_g128.py --prompt-tokens D --max-tokens 128`,
warmup + 5 measured runs per depth (all five runs within ±0.02 % at every depth):

| context tokens | decode median | min–max | TTFT median | KV usage |
|---:|---:|---|---:|---:|
| 512 | **40.200 tok/s** | 40.199–40.208 | 0.463 s | 2.9 % |
| 2048 | 39.531 tok/s | 39.528–39.539 | 1.535 s | 11.6 % |
| 4096 | 38.994 tok/s | 38.991–38.999 | 3.029 s | 23.2 % |
| 8192 | 38.060 tok/s | 38.051–38.067 | 6.240 s | 46.5 % |
| 12000 | 37.315 tok/s | 37.296–37.325 | 9.539 s | 71.9 % |

Read: decode degrades **monotonically and mildly with depth — 40.200 → 37.315 tok/s, i.e. −7.2 %
from 512 to 12000 tokens** (~0.25 %/1k on average, front-loaded), while TTFT grows strictly
linearly (prefill rate is depth-independent, as above). The cost is attention over a growing KV
pool; it is small because this architecture is 48 GDN (`linear_attention`) + only 16
`full_attention` layers, so most layers never look past the recurrent state.

**Live pp/tg in the launch console.** `watch_stats.py vllm-coder390-gptq` tails the engine and
reprints each stats tick llama.cpp-style, e.g. during the 12000-token prefill:

```
[13:38:48] pp   6000.2 t/s | tg    36.5 t/s | run 1 wait 0 | kv  71.9% | hit 0.0%
[13:38:50] pp      0.0 t/s | tg    27.5 t/s | run 0 wait 0 | kv   0.0% | hit 0.0%
```

(`pp 6000.2` is the *instantaneous* rate over that 2 s tick — 12000 tokens in one window — not the
sustained 1266 tok/s of the table; the `tg` column keeps showing the decode rate after the prefill
ends. `watch_stats.py`'s own docstring says the same: coarse per-tick aggregates, exact
per-request numbers come from `bench_p512_g128.py`.) Captured as
`coder390_gptq_pp_tg_live.log` (112 ticks).

**Caveat, per known issue 11:** this session's ctx-512 decode is **40.200 tok/s — the slow state**,
the same one as the parity measurement (40.183 / 40.239), so these curves are a slow-state
measurement. If the fast state applies to prefill as well as decode, all pp *and* tg rows shift
together; the **shape** (linear TTFT, −7.2 % decode to 12k) is the transferable result, and the
absolute levels must not be compared against another session's table (e.g. the 2026-09-26
48.490 with no depth sweep). No same-session control build was run for this sweep — it costs a
server swap — so treat the pp-vs-recorded-reference ~3 % as shape agreement, not a controlled A/B.

### Quality: 16-prompt task benchmark — and the reasoning trace that had to be separated first (2026-10-17 — `bench_quality_16p.py`, `bench_quality_16p.sh`)

Upstream (`alpha2beta/vllm.xpu`) ships `scripts/bench_quality_16p.py`: 16 prompts in 4 categories
(math; code, assertion-checked; strict instruction-following; factual/premise-trap),
auto-scored, driven through the OpenAI client. Fetched **verbatim** (394 lines,
`sha256 798663c3fe8852fd…`, last touched `9ac2d383b560` 2026-10-08); only the invocation is
local, in `bench_quality_16p.sh`. This is the **first quality number for any build here** — the
eval box had been rate-only.

**It runs unmodified against `:8003`. It first returned 8/16 — for the wrong reason.**

| category | as-is | after `--reasoning-parser qwen3` |
|---|---:|---:|
| Math | 4/4 | 4/4 |
| Code | **0/4** | **4/4** |
| Instruction | **0/4** | **3/4** |
| Fact / trap | 4/4 | 4/4 |
| **Total** | **8/16 (50.0 %)** | **15/16 (93.8 %)** |

The harness separates the trace with `extract_clean_answer()`, which needs `</think>` **in
`content`**. This engine instead emits `</think>` as a real token — **id 248069**, visible at
index 146 of the `logprobs` token stream — and then **suppresses it from the decoded text**,
leaving `reasoning` null. The reasoning therefore arrived concatenated in front of the answer
with nothing to split on (`think_chars=0` on all 16 items), and the scorer graded the model's
private musings. The failure texts are self-incriminating:

```
format_reverse_capitals  Expected 5 cities, got 6: ["We need answer user's request. Need ..."]
format_no_letter_e       Failed constraint: contains 249 'e'/'E' letters
format_word_count        Word count 144 outside [15, 25]
code_palindrome          Execution error: SyntaxError: unterminated string literal
```

Math and fact/trap still passed (regex over the last lines and substring search survive
contamination), so the run *looks* plausible at 50 % while half the suite is scored on the wrong
text. **Never run a quality harness against this stack without the parser.**

**The fix is `--reasoning-parser qwen3`, and it was a parity gap, not a deviation.**
`serve_exl3.sh` has always set it; the GPTQ launches never did. `vllm/parser/qwen3.py` matches
this checkpoint exactly — `THINK_START="<think>"`, `THINK_END="</think>"`, the tags the chat
template itself uses, matched **by token id**, so the suppressed tag is still found. Both
`phase5_launch.sh` (launch of record) and `serve_8081.sh` still lack it; they were left alone
deliberately, but any quality harness aimed at them hits the same wall.

Upstream's own artifacts confirm they run with the split: `results_bonsai_pq2_mtp_16p.json`
carries `think_chars=1454, ans_chars=829` with a clean answer — something this engine cannot
produce without the flag.

**Verified clean after the change** (single probe): `content='\n\nCharlie'`, `reasoning` length
197, no `</think>` in `content`. The trace moves to the **`reasoning`** field — vLLM 0.30 spells
it `reasoning` while the script reads `reasoning_content`, so our `think_chars` column stays 0.
Cosmetic only: scoring uses `content`, which is now correct.

**The single remaining failure is neither an artifact nor stable — it is a flaky item.**
`format_no_letter_e` (a sentence with no letter "e") is recorded as `Empty answer` after 43.5 s: at
`max_tokens 2048` it spent all 2048 tokens, and at **4096 it spent all 4096** —
`finish_reason=length` with `content_len=0` in both cases (4,486 → 10,547 chars of reasoning, no
exit). But the **as-is** run had scored this same prompt in **17.2 s with a correct answer** —
`"A vast, calm, dark body of salt is grand."`, which contains no "e"; it failed there only because
the reasoning was concatenated in front of it.

So the item **converges on some server instances and loops on others, and the flag is not the
variable**: booting the *same* server **without** `--reasoning-parser` looped **3/3** (42–43 s, cap
hit, `finish_reason=length`). Within one instance it is deterministic (4/4 with the flag, 3/3
without); across instances it is not. `--reasoning-parser` is therefore confirmed **purely
output-side** — it changes *where the trace is reported*, never what is generated.

Consequence: **15/16 carries a ±1-item (±6.25 %) uncertainty** — the honest figure is 15–16/16, and
a single-item delta between builds should not be read as real. Mechanism unidentified; it is in the
same family as this host's unexplained decode bimodality (known issue 11), so a shared cause in
per-instance kernel autotuning is plausible but unproven (known issue 13).

**Throughput is unaffected, so the recorded rate rows stay valid.** Same shape as the recorded
run, re-measured after the change: **40.208 tok/s, TTFT 0.462 s** (3 runs, spread ≤0.003) against
**40.183 / 0.460** recorded — **+0.06 %** — and still in the **slow bimodal state** (known issue
11), the same band as the 40.183/40.239 parity pair. A pure output post-processor, as expected.

**Comparability caveats before reading 15/16 against anything else:**

* Upstream sets
  `--override-generation-config '{"temperature":0.6,"repetition_penalty":1.08,"presence_penalty":0.2,"top_p":0.95}'`
  server-wide. The script sends `temperature=0.0` but **not** the penalties, so those apply in
  *their* runs and not in ours. We did **not** adopt it — it would change this server's
  production sampling and invalidate the recorded rate rows. Resolve this before any cross-repo
  comparison.
* 16 prompts is a **tiny** sample: one item swings the total by 6.25 %.
* Upstream's published 16p artifacts are for **other checkpoints** (`ternary-bonsai-2-27b`,
  13–16/16 across variants). 15/16 is the same ballpark but is **not** a like-for-like
  comparison, and the generation-config caveat above is unresolved.
* Sibling harnesses, not yet run here: `bench_hard_suite.py` (10 harder tasks —
  long_ctx/math/code/logic_constraint, same CLI) and `eval_exl3_ppl.py`, the real PPL tool — but
  the latter is **offline** (vLLM `prompt_logprobs`, no generation), so it needs one process per
  model rather than the HTTP server.

## 7. Deviations from the plan / cookbook

1. **`data-root` on `/mnt/usb/docker` reverted to `/var/lib/docker`.** `/mnt/usb` is
   **exFAT**; `chown` returns `EPERM`, so the daemon died with
   `could not create or set daemon root permissions`. Not fixable on exFAT (a loopback
   ext4 image would be the alternative). Image is only 4.33 GiB → root went 31 G → 55 G free.
2. **Model path** `/mnt/usb/models/…` instead of `$HOME/models/…` (root `/` had 31 G free).
3. **`--max-model-len 12288`** instead of the cookbook's 100000, **`--gpu-memory-utilization 0.93`**
   instead of 0.88 — the card has 22.71 GiB, not 32 GiB. 100000 tokens needs
   `utilisation >= 1.340`, i.e. physically impossible; 32768 needs >= 0.978 (also not viable).
   vLLM confirms 17,630 tokens of KV is all that exists at 0.93.
4. **`--device /dev/kfd` omitted** — no such node on this Intel host (AMD/ROCm); passing a
   missing node aborts `docker run`.
5. **No `--ipc=host` / `--network=host` / `SYS_PTRACE` / `seccomp=unconfined`** — not needed;
   only informational CCL warnings appeared. These remain the documented fallback.
6. **`huggingface_hub` installed with `uv tool install`** (host Python 3.14 has no pip);
   CLI is `hf`, and `--local-dir-use-symlinks` no longer exists in v2.x.
7. **Endpoint contracts** differed from assumptions: `/tokenize` wants `prompt` (and there is
   no `/v1/tokenize`), `/detokenize` replies with `prompt` not `text`.
8. **The coder390 GPTQ build uses pileval calibration, not the reference's set — and at a
   smaller size than the classic budget.** The reference
   `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` does not record how it was calibrated, so
   `quantize_coder390_gptq.py` uses the GPTQ/AWQ standard — `mit-han-lab/pile-val-backup`,
   `split=validation`, streaming with a 10000-row shuffle buffer. **Size is 64 chunks ×
   1024 tokens (65,536 tokens), ¼ of the classic 128 × 2048 = 262,144** because on this
   host 128 × 2048 reliably kills the job (GPU `cs` engine timeout, or a host OOM-kill:
   `xe` bills GPU memory into the system zone, so VRAM and the 45 GB of RAM are one
   budget — see §6). Every other recipe field (bits, group_size, sym, desc_act, lm_head,
   pack_dtype/impl, true_sequential, act_group_aware, damp_percent, damp_auto_increment,
   fallback, hessian.staging_dtype, dynamic) mirrors the reference's
   `quantize_config.json` exactly. A `--calib-dataset`/`--calib-samples`/`--calib-length`
   override re-runs any other calibration (e.g. if the host gains RAM, or the desktop is
   freed and the budget can go back up). **`quantize_coder390_gptq.py`'s argparse defaults
   were changed to 64 × 1024 on 2026-10-17** (they were 128 × 2048 — the configuration that
   never completed on this host), so a bare `./quantize_coder390.sh` now reproduces the
   successful artifact; the classic budget is one override away.

## 8. Known issues / gotchas

1. **Cold-container start fails its first engine init, then auto-recovers — and the
   recovery *requires* `--restart unless-stopped`.** `torch.compile` = 66.86 s + 17.46 s,
   then `ValueError: … 1.24 GiB KV cache is needed, which is larger than the available KV
   cache memory (0.17 GiB)` → `Engine core initialization failed`. `torch.compile`'s
   transient pool eats the KV budget on the first pass; the *same* container restarted
   against a warm compile cache succeeds (1.12 s). Reproduced twice; deterministic. Total
   cold start ~160 s, warm restart ~45 s. Persisted container state means host reboots are
   warm. **The mechanism is Docker, not vLLM:** the APIServer does *not* retry in-process —
   it exits after the single failure, and only the container's restart policy brings it
   back with its filesystem (and thus the compile cache) intact. On 2026-10-17 a
   hand-written `serve_coder390_gptq.sh` omitted the flag, so the cold start died
   `Exited (1)` with the identical `0.18 GiB` message and never came up; adding
   `--restart unless-stopped` (as `phase5_launch.sh` and `serve_8081.sh` already had)
   fixed it. **Symptom to recognise: an `Exited (1)` container whose log ends in
   `Engine core initialization failed … (0.18 GiB)` is a missing restart policy, not a
   real memory shortfall — do not go tuning `gpu_memory_utilization`.**
2. **`docker logs` is cumulative across restarts** and the container logs in **UTC**
   (one hour behind local CEST) — use `docker logs --since <RFC3339-UTC>`.
3. **`power1_cap` is `664 root:root`** → cap changes need sudo, **and the value is
   volatile**: it resets to the 220 W firmware default on every reboot, so the chosen
   180 W must be re-applied (`vllm-b60-power-cap.service`) or the recorded baseline no
   longer describes the running system. **2026-09-27 proved the failure mode is real,
   not theoretical:** the host rebooted (Phase 10's action), the unit evaluated its
   `ConditionPathExistsGlob` at 10:55:33 — **1.3 s before `hwmon8` appeared
   (10:55:34.328)** — reported `skipped, unmet condition check`, and the card ran at
   220 W all session (`b60_monitor.csv` `cap_W` 180 → 220 at 09:03:42Z). Fix: the
   condition is removed from `vllm-b60-power-cap.service` and `phase10_setcap.sh` now
   **polls up to 90 s** for the hwmon node and fails loudly instead of passing
   silently. **Installed 2026-10-17 13:36:58** — the unit now reports `is-enabled`,
   `is-active`, `ConditionResult=yes`, `ExecMainStatus=0` and `power1_cap` =
   `180000000`, so the *runtime* half is confirmed; only the boot path is unproven
   (last boot 2026-10-07, before the unit existed). Lesson: a systemd path-condition
   on a driver-created node is a race, not a guard.
4. **`xpu-smi` finds no device** (datacenter-only tool) and `intel_gpu_top` is not installed;
   telemetry comes from hwmon `xe` (`energy*`, `temp*`) instead.
5. **GDN decode fell back to Triton** (`the fused CUDA kernel requires a BF16 GDN model …`) —
   worth revisiting if decode throughput is chased later.

6. **`/v1/completions` bypasses the chat template — short bare prompts give garbage.**
   The model ships a real `chat_template` (it injects
   `<system>\nReasoning effort is set to xhigh ...`) and `/v1/completions` skips it, so a
   5-token prompt reaches the model without any template context and deterministically
   returns `!!!!!!!!` (10/10 identical at `temperature=0`). With a longer bare passage it
   behaves normally, and **`/v1/chat/completions` is correct** ("...is Paris."). Use the
   chat endpoint; opencode does. Not a deployment fault, but it looks like one if you
   probe with `curl .../v1/completions -d '{"prompt":"The capital of France is"}'`.
   Related: the reasoning preamble consumes output tokens, so an undersized `max_tokens`
   can truncate the answer — 16 tokens produced only the preamble, 32 produced the answer.
7. **The first completion after a cold start can differ from every later one.** Right
   after the 8081 relaunch, the first `temperature=0` call returned
   `!ysorthussy.validator翁'o Lamb`; the same prompt then returned `!!!!!!!!` ten times
   in a row. Most likely eager-vs-XPU-graph numerics while `VLLM_XPU_ENABLE_XPU_GRAPH=1`
   finishes capturing — argmax on a near-flat degenerate distribution flips easily.
   Phase 7's byte-identical determinism result still stands (it ran after warmup), but
   **the first request of a fresh container should not be treated as representative.**
8. **TP2 across B60+B580 is impossible on this topology — `zeMemOpenIpcHandle`.**
   Attempted 2026-09-27 (`compare_tp2.sh --step tp2`, full cookbook §12 contract:
   affinity patch + 4 CCL vars + `--ipc=host --cap-add SYS_PTRACE` + TP=2 + graphs
   off). Both ranks died in engine init with
   `RuntimeError: oneCCL: ze_call.cpp:28 do_call: EXCEPTION: ze error at
   zeMemOpenIpcHandle, code: ZE_RESULT_ERROR_INVALID_ARGUMENT`
   → `RuntimeError: Engine core initialization failed` (evidence:
   `tp2_failure_excerpt.txt`). Consumer PCIe has no P2P aperture routing (no ACS),
   so one device context cannot import the other's buffers; the cookbook's own
   references admit the same failure for **2× identical B60** (intel/llm-scaler#594)
   — a heterogeneous B60+B580 pair never had a chance. PP2 was deliberately *not*
   attempted: at `--max-num-seqs 1` pipeline bubbles + stage transfer can only lose
   to a single card, and sustained multi-GPU is already marked fragile.
9. **The affinity patch is not restart-safe, and `--restart unless-stopped` turns
   that into a permanent crash loop that hides the real error.** `ok2`'s marker
   (`"end B70_WORKER_AFFINITY"`) never occurs in its own inject, so after attempt 1
   patches the file (verified via `docker cp`: ok2 + ok3 injections present),
   every subsequent restart prints `FATAL: anchor count 0 != 1` and exits before
   vLLM runs — the `zeMemOpenIpcHandle` root cause above was only visible in the
   *head* of the logs. Any retry of multi-GPU work needs `--restart no` (or
   `docker rm -f` between attempts) so engine errors stay visible.
10. **VRAM and host RAM are one budget on this host: the `xe` driver bills GPU
    allocations into the system memory zone.** A kernel OOM report during the coder390
    quantization showed `gpu_active:35480432kB` (≈ B60 24 GB + B580 12 GB) inside
    `Node 0 Normal` with `Free swap = 112kB` and `gpu_reclaim:0` — i.e. filling the GPU
    can exhaust the 45 GB of RAM and swap, and the two failure modes (GPU
    `UR_RESULT_ERROR_DEVICE_LOST` / `ccs` engine timeout, and a host OOM-kill of the
    container) are the **same** pressure seen from two sides. Budget GPU *and* host
    memory together for any heavy job here; on 2026-10-17 a 128 × 2048 pileval GPTQ
    calibration (~16 GiB device + ~10 GiB host) overran both, while 16 samples was fine.
    Corollary: a crashed run **keeps its container alive holding the whole GPU** (it
    prints the traceback but never exits) — kill it before measuring free VRAM.
    Keep expose-surface tidy too: `--device /dev/dri` makes level-zero enumerate the
    unused B580 (probe B60↔B580 P2P); the quantizer wrapper now passes
    `--device renderD128 --device card0` only. (An earlier revision of this entry blamed
    the B580 outright for the `DEVICE_LOST`; attempt 2 disproved that — see §6.)
11. **Decode throughput is bimodal on this host (~40.2 vs ~48.5 tok/s, ~20 %), so rate
    comparisons are only valid within one session.** Established 2026-10-17 by benching
    the *reference* model twice: the all-fast 2026-09-26 baseline (`bench_final_180w.json`,
    median 48.490, all five runs 48.5–49.9) versus the same model now
    (`bench_reference_rerun.json`, 40.239/40.228/40.224/**48.496**/40.245) — the fast state
    appeared once in five runs, in the middle of the session. It is **step time**, not
    acceptance (`Mean acceptance length` only moved 3.00 → 3.18, ~6 %), so it is a
    clock/power state; the card reads 52–60 °C now versus 46–48 °C in September, with fan
    ~1,800–2,260 rpm versus ~1,700. Practical rules: (a) never compare a rate against a
    number recorded on a different day — re-run the control in the same session; (b) do not
    read a ~20 % decode delta between two builds as a property of the builds (the
    coder390-vs-reference control measured **−0.14 %** back-to-back); (c) a single bench
    run can land in either state, so keep all five and report the median *and* the spread.
    The recorded §6 EXL3 leg (41.526, 2026-10-05) sits inside this band, and re-measuring it
    same-session **reversed** the recorded decode ranking (EXL3 41.617 vs reference GPTQ
    40.239) — see §6.

12. **vLLM emits `</think>` but suppresses it from `content` — so quality harnesses silently
    score the reasoning trace.** This checkpoint's `</think>` is added token **248069** —
    `special: False` in `tokenizer.json`, so it survives into the token stream and *is* visible
    in `logprobs` — yet the decoded `content` omits it and `reasoning` stays `null`. With no
    `--reasoning-parser` the trace is therefore *concatenated* onto the answer with no
    delimiter. Consequence: `bench_quality_16p.py` scored **8/16** instead of **15/16**,
    failing Code **0/4** on `SyntaxError` from `exec`-ing prose and Instruction **0/4** on
    constraints the reasoning violates ("no letter e": 249 hits; word count 144 of ≤25) — while
    Math and Fact/trap passed, so the run looks plausible at 50 % while it scores the wrong
    text. Fix: `--reasoning-parser qwen3` (§6). `serve_exl3.sh` already sets it;
    `phase5_launch.sh` and `serve_8081.sh` do not.

13. **A generation outcome can differ between server instances even at `temperature=0`.** Same
    launcher, same model dir, same prompt, same greedy sampler — `format_no_letter_e` converged in
    **17.2 s** with a correct answer on one `:8003` instance and **looped to the `max_tokens` cap**
    on others (**4/4** with `--reasoning-parser`, **3/3** after booting the identical server
    without it, all `finish_reason=length`). *Within* an instance the result is deterministic;
    *across* instances it is not. This is **not** the reasoning parser — the loop reproduces with
    and without it, confirming the flag is output-side only. The cause is unidentified and sits
    in the same family as the unexplained decode bimodality (issue 11); per-instance kernel
    autotuning selecting slightly different numerics at a decision boundary is the hypothesis, not
    a finding. **Practical rule: a 16-prompt suite has ±1-item (±6.25 %) granularity — re-run an
    item before calling it a model weakness, and never read a one-item delta between builds as
    real.**

14. **GPTQModel's save silently drops the image-processor configs.** `gptqmodel` 7.6.0 writes
    the quantized shards, `config.json`, `quantize_config.json` and the tokenizers, but **not**
    `preprocessor_config.json` or `video_preprocessor_config.json` — even though the build keeps
    the 333-tensor visual tower in F16. A text-only launch (`--language-model-only`) never
    notices; the omission surfaces only when the file list is compared against the BF16 source
    or the reference checkpoint (both ship the two files), or when vision is actually used. Fix
    is a straight copy — they describe the *unquantized* tower. Caught on 2026-10-18 while
    preparing the HF publication, i.e. **after** the artifact had been verified field-for-field,
    benchmarked and served for a day: the `2399 / 400 / 333 / 15` counts prove the *weights* are
    right and say nothing about the *sidecar* files.

15. **Publishing ~18 GiB over this uplink is a multi-hour, failure-prone operation: supervise it
    per shard, never fire it once.** On 2026-10-18 the first, naive attempt — a single
    `hf upload` of the whole directory, launched 08:29 UTC — ran **1 h 53 m**, pushed **5.75 GB
    into the Xet CAS, and then died silently at 10:22:54 UTC** after a WAN blip (`code 101
    NetworkUnreachable` at 10:14:43, `code 110 Connection timed out` at 10:20:02), **having
    committed nothing**: at 12:50 UTC the Hub still reported `0/5` shards and `0.02 GiB`. The
    process was not at fault — it had survived ~2 h past the session that spawned it, with no
    OOM and no signal in the journal — the *uplink* is ~10 Mbps, so 18.24 GiB is a ~4 h floor
    even when nothing goes wrong, and a drop anywhere in a 4 h window is near-certain. Two
    lessons, both now encoded in `publish_gptq_hf.sh`: (a) **one shard per invocation**, so a
    drop costs at most the shard in flight *and* every finished shard is **durably committed**
    rather than sitting in CAS unreferenced (unreferenced xorbs are garbage-collectable, so an
    uncommitted multi-hour upload is a real GC risk, not just wasted time); (b) **retry until
    the Hub itself lists the file**, not until the command exits 0. Reassurance on the scare the
    retry produced: **the 5.75 GB was not lost.** When the supervised re-run (12:51:55 UTC)
    reached shard 1 it dedup'd almost entirely and **committed 3.35 GiB in 78 s** (`d5e5bcf8`).
    So content-addressed uploads are genuinely resumable — it is *commitment*, not the bytes,
    that a mid-flight drop destroys. (A `not_found` pair early in the retry looked like the CAS
    had been wiped; it actually belonged to the *next* shard, whose xorbs had never been sent.
    Reading dedup results requires knowing which file's chunks the session is on.)

## 9. Artifacts

| file | contents |
|---|---|
| `tasks.md` | the phase-by-phase checklist with results annotated into each box |
| `GPTQ_quantization.md` | **operating guide** for GPTQ-Int4 quantization on this host: recipe, the one-memory-budget rule, toolchain image, verification, failure catalogue, serving |
| **`github.com/alpha2beta/vllm.xpu` @ branch `b60`** | **public snapshot of this whole directory** — orphan branch, 77 files / 9.3 MB including every attempt log; initial commit `c4dc5b8`, verified byte-identical to local (`main` untouched, still the default branch) |
| **`github.com/alpha2beta/vllm.xpu` @ `main` → `GPTQ_vs_MXFP4.md` §5** | **cross-platform MXFP4-vs-GPTQ comparison** contributed upstream (`0bb6378`, +105 lines): the B60/dense-27B result beside the KAT-Coder/Lunar-Lake one — 5.05× vs ~10 %, with the decomposition, the unfused-W4A4 mechanism and the explicit caveats |
| **`huggingface.co/alpha2beta/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-GPTQ-Int4-sym-G128-MTP-BF16`** | the published GPTQ-Int4 artifact — 2,399 tensors, 18.24 GiB / 5 shards, `SHA256SUMS`, model card and both repaired preprocessor configs; **the only GPTQ of Coder390 on the Hub** as of 2026-10-18. Shard upload is bandwidth-bound (~10 Mbps up); see known issue 15 and `publish_gptq_hf.sh` |
| `publish_gptq_hf.sh` | **resilient** HF publisher: uploads one shard per `hf upload` invocation, retries until the *Hub* lists the file (not until the command exits 0), then the sidecars; idempotent/resumable, `--check` for status only, `--shard N` for one shard. Born from known issue 15 |
| `phase5_launch.sh` | exact launch command (recreate with `docker rm -f` first) |
| `serve_8081.sh` | same container published on :8081 for LAN/opencode use; parity-checks itself against `phase5_launch.sh` |
| `compare_tp2.sh` | B60-vs-B60+B580 harness: derives control/TP2/restore from `serve_8081.sh`'s template, benches + samples both cards, `--summary` prints the table |
| `phase3_dri_test.sh` | container `/dev/dri` go/no-go probe |
| `phase7_verify.py` | Phase 6/7 HTTP checks (health, models, warmup, determinism) |
| `bench_p512_g128.py` | Phase 8 p512/g128 harness (stdlib only) |
| `bench_p512_g128.json`, `bench_power_default.json` | benchmark results at the default 220 W cap |
| `bench_180w.json`, `bench_150w.json` | first cap-curve legs (no sampling) |
| `bench_180w_therm.json`, `bench_150w_therm.json` | cap-curve legs with power/thermal sampling |
| `bench_final_180w.json` | **the final recorded baseline** (180 W) |
| `bench_reference_rerun.json`, `bench_coder390_gptq_p512_g128.json`, `bench_exl3_rerun.json` | same-session rate controls proving the ~40.2 / ~48.5 bimodal state (§6, known issue 11) |
| `bench_prefill.py`, `prefill_180w.json` | prefill-rate sweep (256→6144 tokens, engine + client fits) |
| `bench_220w_a.json`, `bench_220w_b.json` | control runs proving the cap effect is real |
| `bench_cmp_control.json`, `phase9_cmp_control_b60.csv` | B60-only graph-OFF control at 180 W (39.458 tok/s) + sampled trace |
| `tp2_failure_excerpt.txt` | verbatim engine root cause of the failed B60+B580 TP2 attempt |
| `phase9_sampler.py` | unprivileged power/temperature/fan sampler (hwmon-based) |
| `phase9_caps.sh`, `phase9_final.sh` | root cap-sweep + final-baseline runners (trap-restore the cap) |
| `phase9_samples.csv`, `phase9_180w.csv`, `phase9_150w.csv`, `phase9_final_180w.csv` | sampled traces |
| `phase10_monitor.sh` | one-row-per-call CSV sampler (VRAM/KV, temps, throttle, queue) |
| `vllm-b60-monitor.service`, `vllm-b60-monitor.timer` | 1-minute systemd sampler (unprivileged) |
| `phase10_setcap.sh`, `vllm-b60-power-cap.service` | re-apply the 180 W cap after reboot |
| `phase10_install.sh` | one-shot installer: units + 180 W cap + telemetry timer |
| `l0_bw.cpp`, `l0_bw` | PCIe link bandwidth probe (Phase 0) |
| `phase9_samples.csv` | 4 Hz power/temperature/fan trace during a benchmark pass |
| `quantize_coder390.sh`, `quantize_coder390_gptq.py` | GPTQ-Int4 builder for the coder390 abliterated model (streaming, MTP-aware, B60-only) |
| `quant_merge_mtp.py` | re-injects `mtp-bf16.safetensors` into the GPTQ output + fixes the index/config |
| `preflight_probe.py` | validates tokenizer/config/meta shell on the real model before a long run |
| `quant_attempt1_device_lost.txt`, `quant_attempt2_device_lost_b60only.txt` | attempts 1–2 `DEVICE_LOST` container stdout + the kernel-log story (§6) |
| `quant_attempt3_diag128.log` | attempt 3 (128×2048 + VRAM logger): crash at layer 4, plus the `free≈0` occupancy curve |
| `quant_attempt4_host_oom.log` | attempt 4 (`expandable_segments:False`): the host OOM-kill |
| `quant_probe16.log` | the 16-sample probe that survived — the key negative control |
| `quant_attempt5_success.log` | **the successful run** (64×1024): occupancy curve, ETA, final totals |
| `serve_exl3.sh`, `serve_mxfp4.sh` | the EXL3 (host :8001) and MXFP4 (host :8002) alternative-server scripts (§6) |
| `longctx_smoke.py` | long-context needle-in-a-haystack smoke test (§6) |
| `serve_coder390_gptq.sh` | serves the coder390 GPTQ build (host :8003, `coder390-gptq`) |
| `prefill_coder390_gptq.json` | long-context prefill (pp) sweep for that build, 1024→12000 tokens, 3 reps (§6) |
| `bench_coder390_gptq_tg_vs_ctx.json` | decode (tg) vs context depth, 512→12000 tokens, 5 runs per depth (§6) |
| `coder390_gptq_pp_tg_live.log` | the live llama.cpp-style `pp … / tg …` console captured during that sweep |
| `serve_mxfp4_mtp.sh` | experiment 1: MXFP4 server with MTP k=4, exactly one variable changed vs `serve_mxfp4.sh` (§6) |
| `compare_mxfp4_mtp.sh` | experiment 1 A/B driver: re-measures MXFP4 MTP-off in the same session as MTP-on, then restores :8003 |
| `bench_mxfp4_rerun_off.json`, `bench_mxfp4_mtp_on.json` | experiment 1 legs: same-session MTP-off control (7.987) and MTP-on (16.620) (§6) |
| `exp1_mxfp4_mtp_summary.json` | experiment 1 verdict: 2.081×, config of both legs, forced ctx 40960 |
| `exp1_mxfp4_mtp_acceptance.txt` | the engine's SpecDecoding metrics for leg B — the 2.83 acceptance that bounds the gain |
| `exp1_mxfp4_mtp_run.log` | full experiment 1 transcript, including the 81920 KV-memory init failure |
| `bench_quality_16p.py` | upstream 16-prompt quality harness, **verbatim** (394 lines, `sha256 798663c3fe8852fd…`, `alpha2beta/vllm.xpu` @ `9ac2d383b560`) |
| `bench_quality_16p.sh` | its invocation (`uv run --with openai` — this host has no `openai` client) plus the reasoning-split precondition |
| `quality_coder390_gptq_16p_asis.json`, `.log` | the **contaminated** run: 8/16, Code 0/4, Instruction 0/4 — the evidence for known issue 12 (§6) |
| `quality_coder390_gptq_16p.json` | the corrected run after `--reasoning-parser qwen3`: **15/16 (93.8 %)** (§6) |
| `bench_coder390_gptq_after_reasoning_parser.json` | throughput control after the flag: 40.208 tok/s / 0.462 s — unchanged (§6) |
