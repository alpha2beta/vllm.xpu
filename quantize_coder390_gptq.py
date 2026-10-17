#!/usr/bin/env python3
"""
quantize_coder390_gptq.py -- GPTQ INT4 (symmetric, group 128) quantization of
the Huihui Qwen3.8-27B Coder390 EfficientThink (abliterated) BF16 model,
mirroring the recipe of `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`.

Recipe parity (from the reference checkpoint's quantize_config.json / meta):
    method=gptq  bits=4  group_size=128  sym=True  desc_act=False
    lm_head=False  pack_dtype=int32  pack_impl=cpu
    act_group_aware=True  true_sequential=True  static_groups=False  mse=0.0
    damp_percent=0.05  damp_auto_increment=0.01
    fallback=Fallback(rtn, "0.5%")  hessian.staging_dtype=float32
    dynamic={"-:.*mtp.*": {}}   # MTP draft block kept unquantized (BF16)

Design notes for THIS host (i5-12400F, 45 GB RAM, Arc Pro B60 22.7 GiB):
  * The 52 GB BF16 source does NOT fit in RAM. We rely on GPTQModel's
    `offload_to_disk=True` path, which builds a meta-device *shell* model plus a
    checkpoint-backed `LazyTurtle` that streams one module at a time, so peak
    host RAM stays far below the checkpoint size.
  * The B60 (XPU) is the quantization accelerator (`device="xpu:0"`).
  * `mtp.*` is not a parameter of transformers' `Qwen3_5ForConditionalGeneration`
    (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so it is re-injected
    afterwards by `quant_merge_mtp.py` -- exactly as the reference artifact and
    the MXFP4 build did.

Run inside `vllm-quantizer:gptqmodel7.6.0` (gptqmodel 7.6.0, torch 2.13.0+xpu,
transformers 5.16.1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def start_mem_logger(device: str, interval: float) -> None:
    """Diagnostics: daemon thread printing XPU memory every `interval` seconds.

    Used to tell a VRAM-exhaustion hang apart from a driver/submission hang
    (the 128-sample runs died at layer 2 with UR_RESULT_ERROR_DEVICE_LOST).
    """
    import threading
    import torch as _torch

    def loop() -> None:
        mib = 1024 * 1024
        while True:
            try:
                free, total = _torch.xpu.mem_get_info(device)
                alloc = _torch.xpu.memory_allocated(device)
                res = _torch.xpu.memory_reserved(device)
                print(f"[mem {time.strftime('%H:%M:%S')}] free={free / mib:.0f} "
                      f"total={total / mib:.0f} alloc={alloc / mib:.0f} "
                      f"reserved={res / mib:.0f} MiB | {_host_mem()}", flush=True)
            except Exception as e:  # pragma: no cover
                print(f"[mem {time.strftime('%H:%M:%S')}] error: {e}", flush=True)
            time.sleep(interval)

    threading.Thread(target=loop, name="mem-logger", daemon=True).start()


def _host_mem() -> str:
    """Host MemAvailable / SwapFree, in MiB. The xe driver bills GPU memory to
    the system zone, so host OOM and device loss are two faces of one limit."""
    try:
        vals: dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                vals[key.strip()] = int(rest.split()[0])
        return (f"hostAvail={vals.get('MemAvailable', 0) / 1024:.0f}MiB "
                f"swapFree={vals.get('SwapFree', 0) / 1024:.0f}MiB")
    except Exception:  # pragma: no cover
        return "hostAvail=? swapFree=?"


def build_calibration(dataset: str, split: str, samples: int, seq_len: int,
                      tokenizer, seed: int = 42) -> list[str]:
    """Pack `dataset` text into `samples` chunks of >= seq_len tokens each.

    `dataset` is either `<hf-name>` or `<hf-name>:<config>` (e.g.
    `wikitext:wikitext-2-raw-v1`).

    Returns a list of raw-text strings; GPTQModel tokenizes them itself (the
    well-trodden raw-text calibration path). Chunking (rather than passing short
    lines) keeps the token count near the classic GPTQ budget of
    `samples * seq_len` tokens, which drives the Hessian quality.
    """
    from datasets import load_dataset

    name, _, cfg = dataset.partition(":")
    log(f"calibration: loading {name}" + (f" [{cfg}]" if cfg else "") + f" split={split} (streaming)")
    load_kwargs = dict(split=split, streaming=True)
    if cfg:
        ds = load_dataset(name, cfg, **load_kwargs)
    else:
        ds = load_dataset(name, **load_kwargs)
    if seed is not None:
        try:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        except TypeError:  # non-streaming fallback
            ds = ds.shuffle(seed=seed)

    text_col = "text"
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    scanned = 0
    for row in ds:
        scanned += 1
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        buf.append(text)
        buf_tokens += len(tokenizer.encode(text, add_special_tokens=False))
        if buf_tokens >= seq_len:
            chunks.append("\n".join(buf))
            buf, buf_tokens = [], 0
            if len(chunks) >= samples:
                break
        if scanned > 4_000_000:  # guard against pathological streams
            break
    if buf and len(chunks) < samples:
        chunks.append("\n".join(buf))

    if len(chunks) < samples:
        log(f"WARNING: only {len(chunks)} of {samples} calibration chunks built "
            f"from {dataset} (scanned {scanned} rows)")
    else:
        log(f"calibration: {len(chunks)} chunks (>= {seq_len} tokens each) "
            f"from {scanned} rows")
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="BF16 source model dir")
    ap.add_argument("--out", required=True, help="output dir for the GPTQ model")
    ap.add_argument("--device", default="xpu:0", help="quantization accelerator")
    ap.add_argument("--offload-path", default=None,
                    help="dir for offload_to_disk (default: <out>/../_quant_offload)")
    ap.add_argument("--calib-dataset", default="mit-han-lab/pile-val-backup",
                    help="'<hf-name>[:<config>]' (default: pileval, the GPTQ/AWQ standard)")
    ap.add_argument("--calib-split", default="validation")
    # Calibration size is THE memory knob on this host (VRAM and system RAM are one
    # budget -- see REPRODUCIBILITY.md known issue 10 / GPTQ_quantization.md §2).
    # 128 x 2048 = 262,144 tokens reliably kills the job here (GPU ccs timeout or a
    # host OOM-kill); 64 x 1024 = 65,536 completed. These defaults are therefore the
    # proven-safe values, NOT the classic GPTQ budget -- pass --calib-samples /
    # --calib-length explicitly to go back up (a bigger host, or a deliberate
    # deviation). Note a size change alters the Hessians, i.e. it is a new artifact.
    ap.add_argument("--calib-samples", type=int, default=64)
    ap.add_argument("--calib-length", type=int, default=1024)
    ap.add_argument("--no-dynamic-mtp", action="store_true",
                    help="omit dynamic={' -:.*mtp.*': {}} from the config")
    ap.add_argument("--pack-impl", default="cpu",
                    help="packing backend hint (reference artifact used 'cpu')")
    ap.add_argument("--mem-log", type=float, default=0.0,
                    help="if >0, log XPU memory every N seconds (diagnostics)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        log(f"ERROR: source not found: {src}")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    offload = Path(args.offload_path) if args.offload_path else (out.parent / "_quant_offload")
    offload.mkdir(parents=True, exist_ok=True)
    log(f"src={src}\nout={out}\noffload={offload}\ndevice={args.device}")

    dataset = args.calib_dataset

    # --- imports (after arg parsing so --help is instant) --------------------
    import torch
    from transformers import AutoTokenizer
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig
    from gptqmodel.quantization.config import GPTQConfig, HessianConfig  # noqa: F401

    log(f"torch {torch.__version__} | xpu available: {torch.xpu.is_available() if hasattr(torch, 'xpu') else 'n/a'}")
    if args.device.startswith("xpu") and hasattr(torch, "xpu") and not torch.xpu.is_available():
        log("ERROR: device=xpu requested but torch.xpu.is_available() is False")
        return 2

    if args.mem_log > 0:
        start_mem_logger(args.device, args.mem_log)
        log(f"mem logger enabled (every {args.mem_log}s)")

    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=False)
    calibration = build_calibration(dataset, args.calib_split, args.calib_samples,
                                    args.calib_length, tokenizer)

    # --- config: mirror the reference quantize_config.json -------------------
    kwargs = dict(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        lm_head=False,
        true_sequential=True,
        method="gptq",
        format="gptq",
        pack_dtype="int32",
        pack_impl=args.pack_impl,
        offload_to_disk=True,
        offload_to_disk_path=str(offload),
        device=args.device,
        hessian=HessianConfig(staging_dtype="float32"),
    )
    if not args.no_dynamic_mtp:
        kwargs["dynamic"] = {"-:.*mtp.*": {}}
    qcfg = QuantizeConfig(**kwargs)
    log("QuantizeConfig: " + json.dumps({
        "bits": qcfg.bits, "group_size": qcfg.group_size, "sym": qcfg.sym,
        "desc_act": qcfg.desc_act, "lm_head": qcfg.lm_head,
        "pack_impl": qcfg.pack_impl, "offload_to_disk": qcfg.offload_to_disk,
        "offload_to_disk_path": qcfg.offload_to_disk_path,
        "dynamic": qcfg.dynamic, "device": str(qcfg.device),
    }, default=str))

    # --- load (meta shell + LazyTurtle streaming) ---------------------------
    t0 = time.time()
    log("loading model (meta shell + lazy checkpoint streaming) ...")
    model = GPTQModel.load(str(src), qcfg)
    log(f"loaded in {time.time() - t0:.0f} s")

    # --- quantize -----------------------------------------------------------
    t1 = time.time()
    log(f"quantizing ({len(calibration)} calibration chunks) ...")
    result = model.quantize(calibration)
    dt = time.time() - t1
    log(f"quantize finished in {dt:.0f} s ({dt/60:.1f} min)")

    # --- save ---------------------------------------------------------------
    t2 = time.time()
    log(f"saving to {out} ...")
    model.save(str(out))
    log(f"saved in {time.time() - t2:.0f} s")

    log(f"TOTAL {time.time() - t0:.0f} s")
    if result:
        try:
            (out / "quant_log.json").write_text(json.dumps(result, indent=2))
        except Exception as e:  # pragma: no cover
            log(f"note: could not write quant_log.json: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
