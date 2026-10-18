# GPTQ-Int4 quantization on the Arc Pro B60 — operating guide

How to produce a **GPTQ-Int4 (symmetric, group 128) W4A16** checkpoint of a large BF16
Qwen3.8 model on this host, then verify and serve it. Written from the 2026-10-17
`coder390` build, which is the one full success to date.

This is the *how-to*. The *why*, the evidence, and the failure narrative live in
`REPRODUCIBILITY.md` — §6 (`coder390 GPTQ-Int4`), §7 deviation 8 (calibration size),
§8 known issue 10 (one memory budget). Read §8 known issue 10 before your first run;
it is the reason four attempts died.

Host assumptions: **i5-12400F, 45 GB RAM, 4 GB swap, one Arc Pro B60 22.7 GiB**
(`renderD128` / `card0`), a second B580 that must stay out of the container.

---

## 0. TL;DR — the commands that worked

```bash
cd /mnt/usb/vllm.xpu

# 0. nothing else may hold the GPU or the RAM
sg docker -c 'docker ps'                      # no vllm-* server may be running
./quantize_coder390.sh --check                # inputs, handles, sizes

# 1. quantize.  THE CALIBRATION OVERRIDE IS NOT OPTIONAL — see §2.
#    ~73 min on the B60.  Streams logs to the terminal.
./quantize_coder390.sh --calib-samples 64 --calib-length 1024

# 2. verify before you trust it (expects 2399 / 400 / 333 / 15 — see §7)
python3 - <<'PY'
import json
O="/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-GPTQ-Int4-sym-G128-MTP-BF16"
wm=json.load(open(f"{O}/model.safetensors.index.json"))["weight_map"]
print("tensors", len(wm),
      "| qweight", sum(k.endswith(".qweight") for k in wm),
      "| visual", sum(k.startswith("model.visual.") for k in wm),
      "| mtp", sum(k.startswith("mtp.") for k in wm))
PY

# 3. serve it
./serve_coder390_gptq.sh
```

Expected artifact: **19 GB, 5 shards**, `quantize_config.json` carrying
`meta.quantizer = "gptqmodel:7.6.0"`.

---

## 1. The recipe, and why every field is frozen

`quantize_coder390_gptq.py` mirrors the reference checkpoint
`Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` field-for-field, so the two are directly
comparable and a new model inherits a known-good configuration:

| field | value | why |
|---|---|---|
| `bits` / `group_size` / `sym` | 4 / 128 / True | reference parity; W4A16 |
| `desc_act` | False | reference; also avoids activation reordering cost |
| `lm_head` | False | reference — head stays BF16 |
| `true_sequential` | True | reference |
| `pack_dtype` / `pack_impl` | `int32` / `cpu` | reference; the packing backend hint |
| `damp_percent` / `damp_auto_increment` | 0.05 / 0.01 | reference |
| `fallback` | `Fallback(rtn, "0.5%")` | reference — 0.5 % of modules may fall back to RTN |
| `hessian.staging_dtype` | `float32` | reference; avoids fp16 Hessian overflow |
| `method` / `format` | `gptq` / `gptq` | auto-gptq layout, what vLLM expects |
| `dynamic` | `{"-:.*mtp.*": {}}` | **keeps the MTP draft block out of quantization** — §8 |
| `offload_to_disk` | True | **the only reason this fits at all** — §2 |

`dynamic={"-:.*mtp.*": {}}` is a name-matching override that applies an *empty* config to
anything matching `mtp.*`, i.e. "leave it alone". Every `mtp.*` tensor therefore stays
**BF16** and — under gptqmodel 7.6.0 — is *written out with the rest of the model*. That
last part is version-dependent; §8 covers the case where it is not.

Do not "improve" a field to match some blog post. If you change one, change it
deliberately and record it in §7 of `REPRODUCIBILITY.md`, as deviation 8 does for the
calibration size.

---

## 2. The one memory budget — and why calibration size is *the* knob

**On this host VRAM and system RAM are a single budget.** The `xe` driver bills GPU
allocations into the system memory zone, so filling the GPU also consumes RAM and swap.
The kernel OOM report from attempt 4 makes it explicit:

```
Free swap = 112kB / Total swap = 4194300kB
gpu_active:35480432kB   gpu_reclaim:0kB       <- ≈ B60 24 GB + B580 12 GB
Out of memory: Killed process (python3) anon-rss:9794776kB
```

The same pressure surfaces two ways, and they look like different bugs:

* **GPU side** — `RuntimeError: level_zero backend failed with error: 20
  (UR_RESULT_ERROR_DEVICE_LOST)`, plus a kernel-log `ccs` engine reset. This is a
  GPU page-fault/stall, *not* a driver bug.
* **Host side** — the container is OOM-killed outright (SIGKILL at 2.6 min).

The 52–56 GB source cannot be held in RAM, so loading uses gptqmodel's
`offload_to_disk=True` path: a **meta-device shell** model plus a `LazyTurtle` that
streams one module at a time from the checkpoint. Peak resident RAM stays far below the
checkpoint size — the *weights* are not the problem. The **calibration activations are**:

| calibration | device-side | host-side | outcome |
|---|---|---|---|
| 128 × 2048 (262,144 tok) | ~16 GiB | ~10 GiB | **dies every time** |
| 64 × 1024 (65,536 tok) | — | — | **succeeds, telemetry flat** |
| 16 × 2048 | — | — | survived 17 min / 18+ layers (probe only) |

So the working point is **64 chunks × 1024 tokens**: the same token *count* class as the
probe, cut 4× from the classic budget, relieving both sides at once.

> ⚠️ **`quantize_coder390_gptq.py` historically defaulted to `--calib-samples 128
> --calib-length 2048` — the configuration that never once completed on this host.** The
> defaults are now the proven-safe `64 / 1024`; pass overrides explicitly if you want a
> different budget. Changing the size changes the Hessians and therefore the quantized
> weights, so a re-run at a different size is a *different artifact* — record it.

Watch the telemetry to know which regime you are in. A healthy run is **flat**:

| signal | 128×2048 (fails) | 64×1024 (succeeds) |
|---|---|---|
| GPU free | ~0–18 MiB | **13,031–16,717 MiB, flat** |
| GPU reserved | 33–50 GiB | **10,236–10,756 MiB, flat** |
| host available | swap exhausted | **35,200–37,672 MiB, flat** |

Enable it with `--mem-log 30` (adds an XPU + host-memory line every 30 s). If free VRAM
trends toward zero, kill it — you are in the failing regime and it will not recover.

---

## 3. The toolchain image

`vllm-quantizer:gptqmodel7.6.0` (local ID `b541e1d25e23`, 17.3 GB, 21 layers):

```dockerfile
FROM vllm/vllm-openai-xpu:v0.30.0     # 17 GB, 18 layers — carries its own toolchain
USER root
RUN pip install --no-cache-dir gptqmodel==7.6.0
RUN python3 -c "import torch; assert torch.__version__.endswith('+xpu')" \
 && python3 -c "import transformers, gptqmodel, importlib.metadata as m; \
                print(transformers.__version__, m.version('gptqmodel'))"
RUN pip install --no-cache-dir zstandard
```

Verified contents: **vLLM 0.30.0 · torch 2.13.0+xpu · transformers 5.16.1 · gptqmodel
7.6.0**. The base already exports `PIP_EXTRA_INDEX_URL`/`UV_EXTRA_INDEX_URL` pointing at
`download.pytorch.org/whl/xpu` and sets `VLLM_TARGET_DEVICE=xpu`, so the XPU torch is
never accidentally replaced by a CPU wheel. `zstandard` is needed because the calibration
dataset is fetched as `zstd`-compressed parquet.

Recreate with:

```bash
sg docker -c 'docker build -t vllm-quantizer:gptqmodel7.6.0 -' <<'EOF'
FROM vllm/vllm-openai-xpu:v0.30.0
USER root
RUN pip install --no-cache-dir gptqmodel==7.6.0
RUN python3 -c "import torch; assert torch.__version__.endswith('+xpu'), torch.__version__"
RUN pip install --no-cache-dir zstandard
EOF
```

Assert the `+xpu` suffix. A silently-CPU torch will appear to work and then produce
garbage or die on the first layer.

---

## 4. Source model requirements

The wrapper expects a directory (`SRC_HOST` in `quantize_coder390.sh`) that looks like a
normal HF checkpoint, **plus** one non-standard file:

```
config.json  chat_template.jinja  tokenizer.json  tokenizer_config.json
generation_config.json  preprocessor_config.json  video_preprocessor_config.json
model-00001-of-00002.safetensors     49.8 GB
model-00002-of-00002.safetensors      4.9 GB
model.safetensors.index.json
mtp-bf16.safetensors                  849 MB   <- NOT in the index
```

Two things to know:

* **The MTP block arrives as a separate file and is not referenced by
  `model.safetensors.index.json`.** transformers' `Qwen3_5ForConditionalGeneration`
  declares `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, so the MTP draft layer is
  not a parameter of the HF model class at all. This is why it needs the `dynamic`
  exclusion (§1) and why §8 has a repair path.
* Confirm the architecture resolves before committing an hour:
  `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`.
  64 layers = 48 GDN `linear_attention` + 16 `full_attention`
  (`full_attention_interval=4`), `num_attention_heads 24`, `num_key_value_heads 4`,
  `head_dim 256`. `in_proj_a`/`in_proj_b` are skipped → **400** quantized linears.

---

## 5. Preflight

Two cheap checks before an hour of GPU time.

```bash
./quantize_coder390.sh --check     # image? source? target? servers running? GID?
```

Then validate the three *risky* load steps against the real model dir, without
quantizing (`preflight_probe.py`):

```bash
sg docker -c "docker run --rm -v /mnt/usb/models/<src>:/work/src:ro \
  -v /mnt/usb/vllm.xpu:/work/scripts:ro --entrypoint python3 \
  vllm-quantizer:gptqmodel7.6.0 /work/scripts/preflight_probe.py /work/src"
```

It checks, in order, and exits non-zero with a traceback on the first failure:

1. **tokenizer loads through GPTQModel's own loader** (`Tokenicer`) — this is what
   catches pathological `pad_token_id` values before they surface deep inside the run;
2. `AutoConfig` resolves the architecture;
3. the **meta-device shell model builds** — the mechanism that makes a 52 GB checkpoint
   quantizable on a 45 GB host. Expect `params=N meta=N`; a non-zero count of
   non-meta params means the streaming path will stage real weights in RAM.

---

## 6. The run

```bash
./quantize_coder390.sh --calib-samples 64 --calib-length 1024
```

The wrapper (`quantize_coder390.sh`) is deliberately stricter than a raw `docker run`:

* **Refuses to start while any vLLM server container is live** — the GPU *and* the RAM
  are needed in full.
* **`docker rm -f`s any previous quantizer container** before starting. A crashed run
  prints its traceback and then **hangs forever holding the entire GPU**; any VRAM
  reading taken before that kill is meaningless.
* Exposes **only `renderD128` + `card0`**, never all of `/dev/dri`. Exposing the whole
  node makes level-zero enumerate the B580 and probe B60↔B580 P2P, which logs
  `cannot be used for peer-to-peer DMA`. (This turned out *not* to be the cause of the
  crashes — attempt 2 died identically B60-only — but it is still correct hygiene.)
* Mounts the HF cache from `/mnt/usb/models/.hf-cache` so the calibration set is
  downloaded once.

The container runs **foreground**, streaming logs. Useful knobs (env, before the command):

| env | default | use |
|---|---|---|
| `QDEV_RENDER_NODE` / `QDEV_CARD` / `QDEV_ZE_MASK` / `QDEV_DEVICE` | B60 | retarget the accelerator |
| `QDEV_ALLOC_CONF` | unset (gptqmodel's own) | `expandable_segments:False,max_split_size_mb:256` — see below |

On `PYTORCH_ALLOC_CONF`: gptqmodel sets
`expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`, which
can grow VMM *reserved* far past physical VRAM (40 GiB on a 24 GiB card) and fill the
device. It was a suspect for the stochastic `DEVICE_LOST`; attempt 4 tried
`expandable_segments:False` and moved the failure to a host OOM-kill instead. **The
allocator is not the fix — the calibration size is.** Leave it alone unless you are
actively diagnosing.

Pass-through args reach `quantize_coder390_gptq.py`:

| flag | default | note |
|---|---|---|
| `--calib-samples` / `--calib-length` | 64 / 1024 | §2 — the important one |
| `--calib-dataset` / `--calib-split` | `mit-han-lab/pile-val-backup` / `validation` | `<name>[:<config>]`, streamed, 10 000-row shuffle buffer |
| `--mem-log` | 0 | seconds between telemetry lines; use 30 |
| `--no-dynamic-mtp` | off | omit the `dynamic` rule (for models with no MTP block) |
| `--offload-path` | `<out>/../_quant_offload` | scratch for the streaming path |
| `--pack-impl` | `cpu` | reference parity |

### What a good run looks like

```
[10:11:12] src=... out=... device=xpu:0
[10:11:12] calibration: 64 chunks (>= 1024 tokens each) from N rows
[10:11:30] loading model (meta shell + lazy checkpoint streaming) ...
[10:14:02] loaded
[10:14:02] quantizing (64 calibration chunks) ...
[11:24:17] quantize finished in 4255 s (70.9 min)
[11:25:57] saved in 99 s
[11:25:57] TOTAL 4356 s
```

**Budget ~73 minutes** (70.9 quantize + 1.7 save) and expect `EXIT=0`. Anything that
ends without a `TOTAL` line did not finish.

---

## 7. Verify the artifact — do not skip this

A run can exit 0 and still be wrong. Check all of it, every time:

```bash
O=/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-GPTQ-Int4-sym-G128-MTP-BF16
du -sh "$O"                       # 19 GB
ls "$O"                           # 5 shards + quantize_config.json + quant_log.json
                                  # + BOTH preprocessor configs — see the gap below

python3 - "$O" <<'PY'
import json, sys, os
O = sys.argv[1]
wm = json.load(open(f"{O}/model.safetensors.index.json"))["weight_map"]
print("index tensors :", len(wm))                                   # 2399
print("qweight (I32) :", sum(1 for k in wm if k.endswith(".qweight")))  # 400
print("visual (F16)  :", sum(1 for k in wm if k.startswith("model.visual.")))  # 333
print("mtp tensors   :", sum(1 for k in wm if k.startswith("mtp.")))   # 15
qc = json.load(open(f"{O}/quantize_config.json"))
print("quantizer     :", qc["meta"]["quantizer"])                   # gptqmodel:7.6.0
print("dynamic       :", qc.get("dynamic"))                         # {'-:.*mtp.*': {}}
PY
```

| check | expected | if wrong |
|---|---|---|
| index tensors | **2399** | shards incomplete — re-run |
| `*.qweight` (int32) | **400** | the linear set changed; confirm `in_proj_a/b` skipping and the layer count |
| `model.visual.*` | **333**, dtype F16 | vision tower must stay BF16, not quantized |
| `mtp.*` | **15**, dtype **BF16** | §8 repair path |
| `meta.quantizer` | `gptqmodel:7.6.0` | records *which* toolchain made it |
| `quantize_config.json` | bits 4 / group 128 / sym True / desc_act False | recipe drifted — record it as a deviation |
| `preprocessor_config.json` **and** `video_preprocessor_config.json` | both present, byte-equal to the source's | **GPTQModel does not write these** — see below |

### The gap GPTQModel leaves: it drops the image-processor configs

**gptqmodel 7.6.0 does not carry `preprocessor_config.json` or
`video_preprocessor_config.json` into the output directory.** The saved model is
otherwise complete — it keeps the 333-tensor visual tower in F16 — so you end up with a
vision tower that has no image-preprocessor config beside it. Nothing fails loudly: a
text-only launch (`--language-model-only`) works, and the omission only shows up when
something tries to use vision, or when you compare your output against a reference
checkpoint's file list.

The reference `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` ships **both** files
(alongside `processor_config.json`), and so does the BF16 source. Repair by copying them
straight across — they describe the *unquantized* tower, so they are bit-identical to the
source's:

```bash
S=/mnt/usb/models/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-bf16
for f in preprocessor_config.json video_preprocessor_config.json; do cp -p "$S/$f" "$O/$f"; done
```

`quant_log.json` is written by our build but not by the reference (which ships only
`quant_log.csv`); it is harmless extra provenance.

### Structural identity against the reference

The strongest single check is per-dtype **element** totals against the reference
checkpoint — it catches a wrong group size, a missed layer set, or a dtype slip in one
number. Our coder390 build and the reference are identical:

```
tensors=2399  total=6,712,046,320
F16 3,219,819,760   I32 3,067,527,168   BF16 424,699,392
```

(read the safetensors headers directly — 8-byte little-endian header length, then the
JSON — rather than summing `torch` tensors, so it costs no RAM):

```bash
python3 - "$O" <<'PY'
import json, struct, os, sys, collections
t = collections.Counter(); n = 0
for f in sorted(os.listdir(sys.argv[1])):
    if not f.endswith(".safetensors"): continue
    with open(os.path.join(sys.argv[1], f), "rb") as fh:
        (hl,) = struct.unpack("<Q", fh.read(8)); hdr = json.loads(fh.read(hl))
    for k, v in hdr.items():
        if k == "__metadata__": continue
        e = 1
        for s in v["shape"]: e *= s
        t[v["dtype"]] += e; n += 1
print("tensors", n, dict(t))
PY
```

Record the output size, the five counts, and the quantize/save timings in
`REPRODUCIBILITY.md` §9 alongside a new artifact row.

---

## 8. MTP: the draft block must survive

The MTP draft layer is what gives the served model its ~2× speculative decoding, so a
build that silently drops it is a serious regression that still "works".

Two paths, in order of preference:

1. **Verified working — gptqmodel 7.6.0 + `dynamic={"-:.*mtp.*": {}}`.** The exclusion
   keeps `mtp.*` out of quantization and 7.6.0 writes the BF16 block out with the rest of
   the model. `quant_merge_mtp.py` was **not needed** for the coder390 build. Confirm with
   the `mtp tensors : 15` check in §7.
2. **Repair path — `quant_merge_mtp.py`.** If the block is missing from the output (older
   gptqmodel, or `--no-dynamic-mtp` was passed), re-inject it from the source:

   ```bash
   python3 quant_merge_mtp.py \
     --src-mtp /mnt/usb/models/<src>/mtp-bf16.safetensors \
     --out     /mnt/usb/models/<out>
   ```

   It copies the block into the output dir and adds its 15 keys to
   `model.safetensors.index.json` so index-based loaders (vLLM, transformers) pick it up.
   It is idempotent, and it warns if the source file contains non-`mtp.*` keys.

A useful sanity check on the source block before quantizing: 15 tensors, all BF16, and
`config.json` reporting `mtp_num_hidden_layers = 1`.

---

## 9. Failure catalogue

Every entry below is a real failure from this host. Ordered by how likely you are to hit
it.

| symptom | cause | fix |
|---|---|---|
| `UR_RESULT_ERROR_DEVICE_LOST` at layer 2–4 + kernel `ccs` engine reset | GPU pressure spilling into host RAM | **cut the calibration** (§2). Not thermal, not the B580, not a fixed layer |
| container killed (SIGKILL) at ~2.6 min, no traceback | host OOM — same budget, other edge | cut the calibration; do not bother with swap |
| run "finishes" with a traceback but the container stays `Up` holding the GPU | crashed runs do not exit | `sg docker -c 'docker rm -f vllm-quant-coder390'` — the wrapper does this on the next start, but do it before reading `free` VRAM |
| only 1 GPU visible expected, more devices enumerated | whole `/dev/dri` exposed | use `--device renderD128 --device card0` (the wrapper already does) |
| `DEVICE_LOST` reproduced with the B60 alone | — | confirms the memory diagnosis; the B580 is **not** implicated |
| `expandable_segments:False` moves the failure to an OOM-kill | allocator is not the cause | revert; fix the calibration size |
| tokenizer/`pad_token_id` errors during load | model metadata | `preflight_probe.py` (§5) catches these before the long run |

**Things already ruled out — do not re-litigate them:** thermal/power (~100 W, 54–62 °C
at the crash), the second GPU (attempt 2), a fixed layer (2, 2, then 4), elapsed time
(the 16-sample probe outlived both 128-sample runs), and the desktop (an idle B60 reports
24,214/24,480 MiB free with the container killed).

---

## 10. Serving the result

```bash
./serve_coder390_gptq.sh              # start (refuses if another server runs)
./serve_coder390_gptq.sh --check      # state + LAN URL, no docker changes
./serve_coder390_gptq.sh --dry-run    # print the docker command
```

The serving recipe mirrors `phase5_launch.sh` (the launch command of record) — GPTQ
dispatch, `--dtype float16`, `--kv-cache-dtype fp8`, `--language-model-only`, MTP k=4,
and the two engine patches. Differences: model dir, host port **8003**, served name
**`coder390-gptq`**, and `--reasoning-parser qwen3`.

Three things that will waste your afternoon if you skip them:

* **`--restart unless-stopped` is load-bearing.** The *first* engine init on a cold
  container deterministically fails with `ValueError: … 1.24 GiB KV cache is needed,
  which is larger than the available KV cache memory (0.18 GiB)`, because
  `torch.compile`'s transient pool eats the KV budget. The APIServer does **not** retry
  in-process; only Docker restarting the *same* container brings it back with the warm
  compile cache. Expect **~220 s** to healthy.
* **`--reasoning-parser qwen3` is not cosmetic.** The engine emits `</think>` (added
  token **248069**) but the detokenizer **suppresses it from `content`**, leaving
  `reasoning` null. Without the parser the whole reasoning trace arrives concatenated in
  front of the answer with no delimiter, and any harness that wants the answer alone
  scores the model on its private musings — vLLM's own 16-prompt quality benchmark goes
  from **15/16 to 8/16** that way (code 4/4 → 0/4 on `exec`-ing prose). `serve_exl3.sh`
  has always set it; `phase5_launch.sh` and `serve_8081.sh` do not. See `REPRODUCIBILITY.md`
  §6 and known issue 12.
* **A single quality item is not reproducible across server instances — and it is not the
  parser.** `format_no_letter_e` converged with a correct answer on one `:8003` instance
  and looped to the `max_tokens` cap on others, **with and without** the flag (3/3 and 4/4
  respectively, all `finish_reason=length`). Deterministic within an instance, not across.
  So treat a 16-prompt suite as **±1 item (±6.25 %)** and re-run a lone failure before
  believing it (known issue 13). Do not "fix" it with the parser — that is output-side only.
* **MTP needs no patch here.** The draft resolves via `B70_MTP_BF16_DRAFT=1` and the
  compressed-tensors `ignore` list; the log should show `Resolved architecture:
  Qwen3_5MTP`. Note it costs ~0.9 GiB of weights, i.e. **context**: at
  `--gpu-memory-utilization 0.93` a 12288 ctx is comfortable, but raising ctx with MTP on
  is a memory negotiation (see the MXFP4 experiment in §6).

---

## 11. Adapting to another model or a bigger host

* **Another model.** Change `SRC_HOST`/`OUT_HOST` in `quantize_coder390.sh`. The python
  builder is model-agnostic; only the `dynamic` rule is Qwen3.8-specific. Verify the
  expected tensor/qweight counts change *predictably* from the new architecture
  (`qweight` ≈ quantized linear count; skip layers named in the model's config).
* **A model without MTP.** Pass `--no-dynamic-mtp`; skip §8.
* **A host with more RAM.** Raise the calibration back toward the classic
  `128 × 2048` and watch telemetry (§2). This changes the Hessians, so it is a new
  artifact, not a re-run — record it as a deviation.
* **More samples at the same memory cost.** Halving `--calib-length` and doubling
  `--calib-samples` keeps the token count but changes the attention/Hessian profile;
  prefer the recorded shape unless you are deliberately exploring.
* **A different dataset.** `--calib-dataset <hf-name>[:<config>] --calib-split <split>`.
  Pileval is the GPTQ/AWQ standard; the reference checkpoint does not record its set, so
  this is already a documented deviation (deviation 8).

---

## 12. File index

| file | role |
|---|---|
| `quantize_coder390.sh` | wrapper: device isolation, server guard, zombie cleanup, mounts |
| `quantize_coder390_gptq.py` | the builder: calibration, recipe, load→quantize→save |
| `preflight_probe.py` | validates tokenizer/config/shell before the long run (§5) |
| `quant_merge_mtp.py` | repair path: re-inject `mtp-bf16.safetensors` (§8) |
| `serve_coder390_gptq.sh` | serves the result on :8003 as `coder390-gptq` (§10) |
| `publish_gptq_hf.sh` | resilient Hub publisher: one shard per invocation, retries until the Hub lists it; resumable, `--check` / `--shard N` (known issue 15) |
| `bench_p512_g128.py` | decode/TTFT benchmark, 1 warmup + N runs |
| `bench_prefill.py` | prefill (pp) sweep |
| `bench_quality_16p.sh`, `bench_quality_16p.py` | 16-prompt quality benchmark |
| `quant_attempt{1..5}_*.log`, `quant_probe16.log` | the attempt-by-attempt evidence (§9) |
| `REPRODUCIBILITY.md` §6/§7/§8 | narrative, deviations, known issues |

### The numbers to expect, in one place

| quantity | value |
|---|---|
| source size | ≈ 55.6 GB (2 shards + a separate 849 MB MTP block) |
| calibration | pileval, 64 × 1024 tokens = 65,536 |
| quantize / save / total | 4255 s / 99 s / **4356 s (~73 min)** |
| output | **19 GB, 5 shards** |
| indices | 2399 tensors · 400 `qweight` · 333 `visual` F16 · 15 `mtp` BF16 |
| serving cold start | ~220 s (one failed init, then the Docker restart) |
| decode @ p512/g128, MTP k=4, 180 W | **40.2 tok/s**, TTFT 0.46 s |
| quality, 16-prompt task benchmark | **15/16 (93.8 %)**, ±1 item — see §10 |
