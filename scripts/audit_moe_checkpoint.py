#!/usr/bin/env python3
"""Read-only EXL3 MoE checkpoint auditor (P0-03).

Inspects safetensors *headers* only (via HTTP range requests for Hub repos, or
local files) so a 15-20 GB checkpoint can be audited without downloading it.
Cross-checks the tensor payload against ``quantization_config.json`` and
verifies the invariants the MoE adapter will rely on:

  * routed expert count / top-k / shared expert presence,
  * per-module Trellis K, codebook, bitrate and matrix dimensions,
  * ``suh`` / ``svh`` Hadamard vectors,
  * whether ``gate_proj`` and ``up_proj`` share the same input ``suh``,
  * which tensors are unquantized (embeddings, router, norms, ...),
  * absence of vision / MTP tensors for text-only serving.

Outputs (default ``logs/moe-preflight/``):
  checkpoint-moe-audit.json          machine-readable audit
  checkpoint-moe-tensors.csv.gz      full tensor inventory (name, dtype, shape, file)
  checkpoint-moe-audit.txt           human-readable summary

Example:
  .venv/bin/python scripts/audit_moe_checkpoint.py \
      --repo turboderp/Qwen3.5-35B-A3B-exl3 --revision 3.09bpw
  .venv/bin/python scripts/audit_moe_checkpoint.py --path /path/to/checkpoint
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "logs" / "moe-preflight"
QUANT_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
LAYER_RE = re.compile(r"layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$")
HF_ENDPOINT = "https://huggingface.co"


def hf_token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001 - anonymous access is fine for public repos
        return None


def main_ref() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_DEFAULT_REVISION

        return HF_HUB_DEFAULT_REVISION
    except Exception:  # noqa: BLE001
        return "main"



# --------------------------------------------------------------------------------------
# Checkpoint access (local directory or Hugging Face repo, header-range only)
# --------------------------------------------------------------------------------------
class HeaderSource:
    """Read safetensors headers / small tensor slices without a full download.

    Remote access uses plain HTTP ``Range`` requests through a keep-alive
    session. ``HfFileSystem`` was measured at ~1.1 s per 4 KiB range read on
    this host (fsspec re-fetches a multi-MiB block on every ``seek``), which
    made a 240-vector audit take over 15 minutes; direct ranges are ~50 ms.
    """

    def __init__(self, repo: str | None, revision: str | None, path: Path | None):
        self.repo = repo
        self.revision = revision
        self.path = path
        self._session = None
        self._data_base: dict[str, int] = {}
        self._log(
            f"source: {path if path is not None else f'{repo}@{revision or main_ref()}'}"
        )

    @staticmethod
    def _log(message: str) -> None:
        print(f"  {message}", file=sys.stderr, flush=True)

    # -- transport ---------------------------------------------------------------
    def _http(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            token = hf_token()
            if token:
                self._session.headers["Authorization"] = f"Bearer {token}"
        return self._session

    def _url(self, filename: str) -> str:
        return f"{HF_ENDPOINT}/{self.repo}/resolve/{self.revision or main_ref()}/{filename}"

    def _range(self, filename: str, start: int, end: int) -> bytes:
        """Inclusive byte range, as HTTP semantics define it."""
        headers = {"Range": f"bytes={start}-{end}"}
        response = self._http().get(self._url(filename), headers=headers, timeout=120)
        if response.status_code == 206:
            return response.content
        if response.status_code == 200:
            # Server ignored Range; slice locally (small files only).
            return response.content[start : end + 1]
        raise RuntimeError(
            f"{filename}: HTTP {response.status_code} for range {start}-{end}: "
            f"{response.text[:200]}"
        )

    # -- API ---------------------------------------------------------------------
    def read_header(self, filename: str) -> dict:
        if self.path is not None:
            with open(self.path / filename, "rb") as handle:
                prefix = handle.read(8)
                header_len = struct.unpack("<Q", prefix)[0]
                header = json.loads(handle.read(header_len))
        else:
            prefix = self._range(filename, 0, 7)
            if len(prefix) != 8:
                raise RuntimeError(f"{filename}: short safetensors length prefix")
            header_len = struct.unpack("<Q", prefix)[0]
            if header_len > 256 * 1024 * 1024:
                raise RuntimeError(f"{filename}: implausible header length {header_len}")
            blob = self._range(filename, 8, 8 + header_len - 1)
            if len(blob) != header_len:
                raise RuntimeError(
                    f"{filename}: short header read ({len(blob)} of {header_len})"
                )
            header = json.loads(blob)
        header.pop("__metadata__", None)
        self._data_base[filename] = 8 + header_len
        return header

    def read_tensor_bytes(self, filename: str, offset: int, length: int) -> bytes:
        base = self._data_base[filename]
        if self.path is not None:
            with open(self.path / filename, "rb") as handle:
                handle.seek(base + offset)
                return handle.read(length)
        return self._range(filename, base + offset, base + offset + length - 1)

    def read_text(self, filename: str) -> str:
        if self.path is not None:
            return (self.path / filename).read_text()
        # Small metadata files only (index/quantization config, tens of MB at
        # most); a full GET avoids Range-truncation surprises.
        response = self._http().get(self._url(filename), timeout=300)
        response.raise_for_status()
        return response.text


def strip_prefix(name: str) -> str:
    for prefix in ("model.language_model.", "language_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def module_of(name: str) -> str:
    return name.rsplit(".", 1)[0]


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------
def audit(args) -> tuple[dict, list[tuple[str, str, list[int], str, str]]]:
    source = HeaderSource(args.repo, args.revision, Path(args.path) if args.path else None)

    index = None
    if args.path:
        index_path = Path(args.path) / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
    else:
        index = json.loads(source.read_text("model.safetensors.index.json"))
    if index is None:
        raise SystemExit("model.safetensors.index.json not found")

    weight_map: dict[str, str] = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    headers: dict[str, dict] = {}
    for shard in shard_files:
        headers[shard] = source.read_header(shard)
        HeaderSource._log(f"header {shard}: {len(headers[shard])} tensors")

    tensors: dict[str, tuple[dict, str]] = {}
    for shard, header in headers.items():
        for name, meta in header.items():
            tensors[name] = (meta, shard)

    missing_in_headers = sorted(set(weight_map) - set(tensors))
    missing_in_index = sorted(set(tensors) - set(weight_map))

    quant_cfg = json.loads(source.read_text("quantization_config.json"))
    storage: dict[str, dict] = quant_cfg.get("tensor_storage", {})
    norm_storage = {strip_prefix(k): v for k, v in storage.items()}

    rows: list[tuple[str, str, list[int], str, str]] = []
    for name, (meta, shard) in sorted(tensors.items()):
        rows.append(
            (strip_prefix(name), meta["dtype"], list(meta["shape"]), name, shard)
        )

    # ---- module-level quant metadata -------------------------------------------------
    modules: dict[str, dict] = defaultdict(dict)
    unquantized: list[tuple[str, str, list[int]]] = []
    for name, (meta, _shard) in tensors.items():
        key = strip_prefix(name)
        base, _, leaf = key.rpartition(".")
        if leaf in QUANT_SUFFIXES:
            modules[base][leaf] = {"shape": list(meta["shape"]), "dtype": meta["dtype"]}
        else:
            unquantized.append((key, meta["dtype"], list(meta["shape"])))

    tensor_names = {strip_prefix(name) for name in tensors}


    module_reports: dict[str, dict] = {}
    anomalies: list[str] = []
    k_counter: Counter[int] = Counter()
    codebook_counter: Counter[str] = Counter()
    for base, parts in sorted(modules.items()):
        if "trellis" not in parts:
            anomalies.append(f"{base}: quantized metadata without trellis: {sorted(parts)}")
            continue
        trellis = parts["trellis"]["shape"]
        if len(trellis) != 3:
            anomalies.append(f"{base}: trellis rank {len(trellis)} != 3")
            continue
        k_tiles, n_tiles, bits = trellis
        if bits % 16:
            anomalies.append(f"{base}: trellis last dim {bits} not a multiple of 16")
            continue
        K = bits // 16
        k, n = k_tiles * 16, n_tiles * 16
        codebook = "mcg" if "mcg" in parts else ("mul1" if "mul1" in parts else "3inst")
        suh = parts.get("suh", {}).get("shape")
        svh = parts.get("svh", {}).get("shape")
        if suh != [k]:
            anomalies.append(f"{base}: suh shape {suh} != expected [{k}]")
        if svh != [n]:
            anomalies.append(f"{base}: svh shape {svh} != expected [{n}]")
        if k % 128 or n % 128:
            anomalies.append(f"{base}: dimensions not 128-aligned (k={k}, n={n})")
        k_counter[K] += 1
        codebook_counter[codebook] += 1
        module_reports[base] = {
            "k": k,
            "n": n,
            "K": K,
            "bits_per_weight": K,
            "codebook": codebook,
            "trellis_shape": trellis,
        }

    HeaderSource._log(f"quantized modules: {len(module_reports)}, unquantized: {len(unquantized)}")

    # ---- structural expectations -----------------------------------------------------
    layer_ids = sorted(
        {int(m.group(1)) for name in tensors if (m := LAYER_RE.search(name))}
    )
    routed_layers: dict[int, dict] = {}
    expert_counts: dict[int, set[int]] = defaultdict(set)
    for name in tensors:
        key = strip_prefix(name)
        lm = LAYER_RE.search(key)
        em = EXPERT_RE.search(key)
        if lm and em and ".experts." in key:
            expert_counts[int(lm.group(1))].add(int(em.group(1)))

    # Group quantized modules by (layer, expert) in one pass; a nested
    # per-layer rescan of all 31k modules is quadratic and slow.
    expert_modules: dict[int, dict[int, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for key, entry in module_reports.items():
        lm = LAYER_RE.search(key)
        if not lm:
            continue
        match = re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$", key)
        if not match:
            continue
        expert_modules[int(lm.group(1))][int(match.group(1))][match.group(2)] = entry

    for layer in layer_ids:
        per_expert = expert_modules.get(layer)
        if not per_expert:
            continue
        routed_layers[layer] = {
            "expert_count": len(per_expert),
            "projections": {},
        }
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shapes = {
                (e[proj]["k"], e[proj]["n"], e[proj]["K"], e[proj]["codebook"])
                for e in per_expert.values()
                if proj in e
            }
            routed_layers[layer]["projections"][proj] = sorted(
                [{"k": s[0], "n": s[1], "K": s[2], "codebook": s[3]} for s in shapes],
                key=lambda x: (x["K"], x["k"], x["n"]),
            )
            if len(shapes) != 1:
                anomalies.append(
                    f"layer {layer}: {proj} has heterogeneous expert shapes/K: {shapes}"
                )

    # Shared expert and router presence per layer.
    for layer in layer_ids:
        base = f"layers.{layer}.mlp"
        if f"{base}.shared_expert.down_proj" not in module_reports:
            anomalies.append(f"layer {layer}: shared expert not found")
        for required in (f"{base}.gate.weight", f"{base}.shared_expert_gate.weight"):
            if required not in tensor_names:
                anomalies.append(f"layer {layer}: missing {required}")

    # ---- Hadamard gate/up equality ---------------------------------------------------
    suh_equality = {"checked_pairs": 0, "identical": 0, "differing_examples": []}
    if args.suh_check:
        full_name = {strip_prefix(n): n for n in tensors}
        total = len(layer_ids) * len(args.suh_check)
        for done, (layer, expert) in enumerate(
            (l, e) for l in layer_ids for e in args.suh_check
        ):
            pair = []
            for proj in ("gate_proj", "up_proj"):
                key = f"layers.{layer}.mlp.experts.{expert}.{proj}.suh"
                original = full_name.get(key)
                if original is None or original not in weight_map:
                    pair = []
                    break
                meta = tensors[original][0]
                begin, end = meta["data_offsets"]
                pair.append(
                    source.read_tensor_bytes(weight_map[original], begin, end - begin)
                )
            if not pair:
                continue
            if (done + 1) % 20 == 0:
                HeaderSource._log(f"  suh check {done + 1}/{total}")
            suh_equality["checked_pairs"] += 1
            if pair[0] == pair[1]:
                suh_equality["identical"] += 1
            elif len(suh_equality["differing_examples"]) < 5:
                suh_equality["differing_examples"].append(
                    {"layer": layer, "expert": expert}
                )

    if suh_equality["checked_pairs"]:
        print(
            f"  gate/up suh: {suh_equality['identical']}/{suh_equality['checked_pairs']} "
            "pairs identical",
            file=sys.stderr,
            flush=True,
        )

    # ---- unquantized inventory -------------------------------------------------------
    unquant_by_kind: Counter[str] = Counter()
    unquant_examples: dict[str, dict] = {}
    for name, dtype, shape in unquantized:
        if name.endswith("embed_tokens.weight"):
            kind = "embedding"
        elif name.endswith("lm_head.weight"):
            kind = "lm_head"
        elif name.endswith("mlp.gate.weight"):
            kind = "router_gate"
        elif name.endswith("shared_expert_gate.weight"):
            kind = "shared_expert_gate"
        elif "norm.weight" in name or name.endswith(".norm.weight"):
            kind = "rmsnorm"
        elif name.endswith("linear_attn.conv1d.weight"):
            kind = "gdn_conv1d"
        elif name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
            kind = "gdn_in_proj_ab"
        elif name.endswith(("A_log", "dt_bias")):
            kind = "gdn_state_param"
        elif "visual" in name or "vision" in name:
            kind = "vision"
        else:
            kind = "other"
        unquant_by_kind[kind] += 1
        unquant_examples.setdefault(kind, {"name": name, "dtype": dtype, "shape": shape})

    has_vision = any(("visual" in n or "vision" in n) for n in tensors)
    has_mtp = any(n.startswith(("mtp.", "model.mtp.")) or ".mtp." in n for n in tensors)

    # Payload bytes per family: drives the resident-memory plan for Gate 2/3.
    family_bytes: Counter[str] = Counter()
    for name, (meta, _shard) in tensors.items():
        nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        if ".mlp.experts." in name:
            family = "routed_experts"
        elif ".mlp.shared_expert." in name or ".shared_expert_gate." in name:
            family = "shared_expert"
        elif "visual" in name or "vision" in name:
            family = "vision"
        elif "lm_head" in name:
            family = "lm_head"
        elif "embed_tokens" in name:
            family = "embedding"
        elif "linear_attn" in name or "self_attn" in name:
            family = "attention"
        elif ".mlp.gate." in name:
            family = "router"
        else:
            family = "norms_and_misc"
        family_bytes[family] += nbytes
    total_payload = sum(family_bytes.values())

    # Cross-check tensor_storage bitrates.
    bitrate_mismatches = []
    for base, report in module_reports.items():
        stored = norm_storage.get(base) or norm_storage.get(base.replace("lm_head", "lm_head"))
        if not stored:
            continue
        bits = stored.get("bits_per_weight")
        if bits is not None and int(bits) != report["K"]:
            bitrate_mismatches.append(
                {"module": base, "quantization_config_bits": bits, "trellis_K": report["K"]}
            )
            if len(bitrate_mismatches) >= 20:
                break

    # Shared expert / lm_head / attention module summaries.
    def summarize(pattern: str) -> dict:
        selected = {k: v for k, v in module_reports.items() if re.search(pattern, k)}
        if not selected:
            return {}
        # Group by projection leaf so `down_proj` (k=512,n=2048) is not
        # reported as a "heterogeneous" outlier next to gate/up (k=2048,n=512).
        by_leaf: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for name, report in selected.items():
            leaf = name.rsplit(".", 1)[-1]
            by_leaf[leaf][
                (report["k"], report["n"], report["K"], report["codebook"])
            ] += 1
        projections = {}
        for leaf, shapes in sorted(by_leaf.items()):
            projections[leaf] = {
                "module_count": sum(shapes.values()),
                "variants": [
                    {
                        "count": count,
                        "k": s[0],
                        "n": s[1],
                        "K": s[2],
                        "codebook": s[3],
                    }
                    for s, count in sorted(shapes.items())
                ],
            }
        first = selected[sorted(selected)[0]]
        return {
            "module_count": len(selected),
            "example": sorted(selected)[0],
            "projections": projections,
            "shape_variants_across_projections": len(
                {(v["k"], v["n"], v["K"], v["codebook"]) for v in selected.values()}
            ),
            "note": (
                "shape_variants_across_projections counts the fused-projection "
                "shape spread; variants within one projection are listed above."
            ),
            "first_module": {
                "k": first["k"],
                "n": first["n"],
                "K": first["K"],
                "codebook": first["codebook"],
            },
        }

    audit = {
        "checkpoint": {
            "repo": args.repo,
            "revision": args.revision,
            "path": args.path,
            "quantization_config": {
                k: quant_cfg.get(k)
                for k in (
                    "quant_method",
                    "version",
                    "bits",
                    "head_bits",
                    "codebook",
                    "out_scales",
                    "calibration",
                )
            },
            "shards": shard_files,
            "tensor_count": len(tensors),
            "weight_index_entries": len(weight_map),
        },
        "integrity": {
            "index_entries_missing_from_headers": missing_in_headers[:20],
            "header_tensors_missing_from_index": missing_in_index[:20],
            "header_tensor_count": len(tensors),
        },
        "structure": {
            "layer_count": len(layer_ids),
            "layer_ids": [layer_ids[0], layer_ids[-1]] if layer_ids else [],
            "routed_experts_per_layer": sorted(
                {len(v) for v in expert_counts.values()}
            ),
            "has_vision_tensors": has_vision,
            "has_mtp_tensors": has_mtp,
            "payload_bytes_by_family": {
                k: {"bytes": v, "gib": round(v / 2**30, 3)}
                for k, v in family_bytes.most_common()
            },
            "payload_bytes_total": total_payload,
        },
        "quantization": {
            "quantized_module_count": len(module_reports),
            "K_histogram": dict(sorted(k_counter.items())),
            "codebook_histogram": dict(codebook_counter),
            "bitrate_mismatches": bitrate_mismatches,
            "routed_expert_modules": summarize(r"\.mlp\.experts\.\d+\."),
            "shared_expert_modules": summarize(r"\.mlp\.shared_expert\."),
            "linear_attention_modules": summarize(r"\.linear_attn\."),
            "full_attention_modules": summarize(r"\.self_attn\."),
            "lm_head": module_reports.get("lm_head"),
        },
        "hadamard": {
            "all_quantized_modules_have_suh_svh": all(
                ("suh" in modules[b] and "svh" in modules[b]) for b in module_reports
            ),
            "gate_up_suh": suh_equality,
            "blocked_fwht_width": 128,
            "note": (
                "EXL3 applies H128 in 128-wide blocks to input and output of every "
                "linear; suh/svh are full k/n-length vectors (see exl3xpu/ref.py)."
            ),
        },
        "unquantized": {
            "kind_histogram": dict(unquant_by_kind),
            "examples": unquant_examples,
            "tensor_count": len(unquantized),
        },
        "layers": routed_layers,
        "anomalies": anomalies,
    }
    return audit, rows


def write_outputs(audit: dict, rows, out_dir: Path, write_csv: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "checkpoint-moe-audit.json"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")

    if write_csv:
        csv_path = out_dir / "checkpoint-moe-tensors.csv.gz"
        with gzip.open(csv_path, "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "dtype", "shape", "checkpoint_name", "shard"])
            for name, dtype, shape, original, shard in rows:
                writer.writerow([name, dtype, json.dumps(shape), original, shard])

    q = audit["quantization"]
    s = audit["structure"]
    lines = [
        "# EXL3 MoE checkpoint audit (P0-03)",
        f"checkpoint: {audit['checkpoint']['repo'] or audit['checkpoint']['path']}"
        f" @ {audit['checkpoint']['revision'] or 'local'}",
        f"shards: {', '.join(audit['checkpoint']['shards'])}",
        f"tensors: {audit['checkpoint']['tensor_count']}",
        f"layers: {s['layer_count']}",
        f"routed experts/layer: {s['routed_experts_per_layer']}",
        f"quantized modules: {q['quantized_module_count']}",
        f"Trellis K histogram: {q['K_histogram']}",
        f"codebooks: {q['codebook_histogram']}",
        f"vision tensors: {s['has_vision_tensors']}, MTP tensors: {s['has_mtp_tensors']}",
        "",
        "## Routed experts (layer 0)",
        json.dumps(audit["layers"].get(0, {}), indent=2),
        "",
        "## Shared expert",
        json.dumps(q["shared_expert_modules"], indent=2),
        "",
        "## Linear attention (GDN)",
        json.dumps(q["linear_attention_modules"], indent=2),
        "",
        "## lm_head",
        json.dumps(q["lm_head"], indent=2),
        "",
        "## Unquantized tensors",
        json.dumps(audit["unquantized"], indent=2),
        "",
        "## Hadamard gate/up suh equality",
        json.dumps(audit["hadamard"]["gate_up_suh"], indent=2),
        "",
        f"## Anomalies ({len(audit['anomalies'])})",
        *[f"- {a}" for a in audit["anomalies"][:100]],
        "",
    ]
    (out_dir / "checkpoint-moe-audit.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {json_path}")
    if write_csv:
        print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'checkpoint-moe-audit.txt'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Hugging Face repo id")
    source.add_argument("--path", help="local checkpoint directory")
    parser.add_argument("--revision", default=None, help="Hub revision (branch/tag/sha)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--suh-check",
        type=int,
        nargs="*",
        default=[0, 127, 255],
        help="expert ids whose gate/up suh vectors are compared byte-for-byte",
    )
    parser.add_argument("--no-tensor-csv", action="store_true")
    args = parser.parse_args()
    audit_data, rows = audit(args)
    write_outputs(audit_data, rows, args.out_dir, not args.no_tensor_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
