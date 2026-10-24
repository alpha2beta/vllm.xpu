#!/usr/bin/env python3
"""P1-05: shard-by-shard streaming loader test.

Drives ``exl3xpu.moe.stream_expert_shards`` over the *real* 3.09bpw checkpoint
and records memory/telemetry per shard, then audits that every routed-expert
key was consumed and that each packed layer passes the P1-02..P1-04 checks.

    .venv/bin/python scripts/test_shard_streaming.py \
        --model-dir /home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw

Artifacts (defaults):

* ``logs/moe-p1/shard-loading-memory.csv``  — one row per shard
* ``logs/moe-p1/shard-streaming-audit.json`` — key audit + per-layer status

Why a dedicated harness: on the Arc unified-memory part, keeping several
8 GiB shard mmaps open *while* touching the packed arenas doubles the
transient traffic. The loader opens one shard at a time, reads each tensor
lazily, and releases the mmap + ``torch.xpu.empty_cache()`` before the next.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from exl3xpu.moe import (
    Exl3MoEMethod,
    split_expert_key,
    stream_expert_shards,
)
from exl3xpu.vllm_plugin import Exl3Config

NUM_EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 512
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/home/yanchun/models/Qwen3.5-35B-A3B-exl3-3.09bpw"
_PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_mib() -> float:
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * _PAGE / 2**20


def xpu_stats() -> dict[str, float]:
    try:
        free, total = torch.xpu.mem_get_info()
        return {
            "xpu_allocated_mib": torch.xpu.memory_allocated() / 2**20,
            "xpu_reserved_mib": torch.xpu.memory_reserved() / 2**20,
            "xpu_free_mib": free / 2**20,
            "xpu_total_mib": total / 2**20,
        }
    except Exception:
        return {
            "xpu_allocated_mib": -1.0,
            "xpu_reserved_mib": -1.0,
            "xpu_free_mib": -1.0,
            "xpu_total_mib": -1.0,
        }


def make_layer(layer_key: str, prefix: str) -> RoutedExperts:
    layer = object.__new__(RoutedExperts)
    nn.Module.__init__(layer)
    layer.layer_name = prefix
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.lora_base_layer_prefix = ""
    layer.is_fused_checkpoint_transposed = False
    layer.moe_config = types.SimpleNamespace(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_dim=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        moe_parallel_config=types.SimpleNamespace(
            tp_size=1, ep_size=1, use_ep=False, enable_eplb=False
        ),
    )
    layer.expert_map_manager = types.SimpleNamespace(num_fused_shared_experts=0)
    return layer


def build_config(model_dir: str) -> Exl3Config:
    with open(Path(model_dir) / "quantization_config.json") as f:
        qc = json.load(f)
    config = Exl3Config(
        bits=float(qc.get("bits", 3)),
        head_bits=int(qc.get("head_bits", 6)),
        codebook=qc.get("codebook", "mcg"),
    )
    config.maybe_update_config(model_dir)
    return config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="layer ids to pack (default: all found in the index)")
    ap.add_argument("--device", default=None, help="xpu (default) or cpu")
    ap.add_argument("--csv-out", default=str(ROOT / "logs/moe-p1/shard-loading-memory.csv"))
    ap.add_argument("--audit-out", default=str(ROOT / "logs/moe-p1/shard-streaming-audit.json"))
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="only process the first N shards (smoke test)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    device = args.device or ("xpu" if torch.xpu.is_available() else "cpu")

    # Expected routed-expert keys, grouped by their shard.
    expected_by_shard: dict[str, list[str]] = {}
    expected_keys: set[str] = set()
    layer_ids: set[int] = set()
    for key, shard in weight_map.items():
        layer_key, _ = split_expert_key(key)
        if layer_key is None:
            continue
        expected_keys.add(key)
        expected_by_shard.setdefault(shard, []).append(key)
        layer_ids.add(int(layer_key.split(".")[1]))

    selected = sorted(layer_ids if args.layers is None else set(args.layers))
    shard_files = sorted(expected_by_shard)
    if args.limit_shards is not None:
        shard_files = shard_files[: args.limit_shards]

    print(f"model_dir      : {model_dir}")
    print(f"device         : {device}")
    print(f"shards         : {len(expected_by_shard)} ({', '.join(shard_files)})")
    print(f"layer ids      : {selected[0]}..{selected[-1]} ({len(selected)})")
    print(f"expected keys  : {len(expected_keys)} routed-expert tensors")

    config = build_config(str(model_dir))

    print(f"building {len(selected)} packed layers ...")
    t_build = time.perf_counter()
    layers: dict[str, RoutedExperts] = {}
    with torch.device(device):
        for layer_id in selected:
            prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
            layer_key = f"layers.{layer_id}.mlp.experts"
            layer = make_layer(layer_key, prefix)
            method = Exl3MoEMethod(config, layer, prefix)
            method.create_weights(
                layer, NUM_EXPERTS, HIDDEN, INTERMEDIATE, torch.bfloat16
            )
            # Real ``RoutedExperts.__init__`` assigns ``self.quant_method``
            # (routed_experts.py:145); mirror it so the settle path matches.
            layer.quant_method = method
            layers[layer_key] = layer
    print(f"  built in {time.perf_counter() - t_build:.1f}s; "
          f"{xpu_stats()['xpu_allocated_mib']:.0f} MiB allocated")

    csv_rows: list[dict] = []

    def on_shard(row: dict) -> None:
        stats = {**xpu_stats(), "rss_mib": rss_mib()}
        row.update(stats)
        csv_rows.append(row)
        print(f"  shard {row['shard_index']}: {row['expert_tensors']} tensors in "
              f"{row['seconds']:.1f}s | rss {row['rss_mib']:.0f} MiB | "
              f"xpu alloc {row['xpu_allocated_mib']:.0f} MiB")

    print("streaming shards ...")
    t_stream = time.perf_counter()
    report = stream_expert_shards(
        [str(model_dir / s) for s in shard_files], layers, on_shard=on_shard
    )
    stream_seconds = time.perf_counter() - t_stream
    peak = {**xpu_stats(), "rss_mib": rss_mib()}

    consumed = set(report["consumed"])
    selected_set = set(selected)
    scope = {k for k in expected_keys
             if int(split_expert_key(k)[0].split(".")[1]) in selected_set}
    missing = sorted(scope - consumed)
    unaccounted = sorted(consumed - expected_keys)
    print(f"streamed {report['expert_tensors']} tensors in {stream_seconds:.1f}s; "
          f"missing={len(missing)} unaccounted={len(unaccounted)} "
          f"out_of_scope={len(report['out_of_scope'])}")

    # Per-layer settle: completeness + P1-04 integrity.
    t_settle = time.perf_counter()
    layer_status = {}
    for layer_key, layer in layers.items():
        method = layer.quant_method
        method.process_weights_after_loading(layer)
        summary = method.verify_expert_integrity(layer)
        layer_status[layer_key] = summary
    print(f"settled {len(layers)} layers in {time.perf_counter() - t_settle:.1f}s")

    # -- artifacts ---------------------------------------------------------
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "shard_index", "shard", "expert_tensors", "seconds", "rss_mib",
        "xpu_allocated_mib", "xpu_reserved_mib", "xpu_free_mib", "xpu_total_mib",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: row.get(k) for k in fields})

    audit = {
        "task": "P1-05",
        "model_dir": str(model_dir),
        "device": device,
        "device_name": torch.xpu.get_device_properties(0).name
        if device == "xpu" else "cpu",
        "shards": shard_files,
        "shards_total": len(expected_by_shard),
        "limit_shards": args.limit_shards,
        "layers_selected": selected,
        "expected_expert_keys": len(expected_keys),
        "expected_in_scope": len(scope),
        "consumed": len(consumed),
        "expert_tensors_streamed": report["expert_tensors"],
        "missing": missing,
        "unaccounted": unaccounted,
        "out_of_scope": len(report["out_of_scope"]),
        "stream_seconds": stream_seconds,
        "per_shard": report["shards"],
        "peak": peak,
        "final": {**xpu_stats(), "rss_mib": rss_mib()},
        "layer_status": layer_status,
        "pass": not missing and not unaccounted,
    }
    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {audit_path}")
    print(f"PASS={audit['pass']} peak xpu allocated {peak['xpu_allocated_mib']:.0f} MiB, "
          f"peak rss {peak['rss_mib']:.0f} MiB")
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
