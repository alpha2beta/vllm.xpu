#!/usr/bin/env python3
"""P0-07 probe: vLLM Qwen3.5 hybrid-attention / GDN path on Arc 140V.

Builds the *real* ``QwenGatedDeltaNetAttention`` layer from the pinned
checkpoint configuration (no checkpoint weights loaded; parameters are filled
with deterministic dummy data), allocates the recurrent-state tensors that
vLLM's ``MambaStateShapeCalculator`` prescribes, and runs ``forward_xpu`` for

  * a single-token decode step, and
  * an 8-token prefill chunk,

then checks that the state tensors live in XPU memory, are mutated by the
kernel, contain only finite values, and that repeat runs are deterministic.
This satisfies the Gate 0 criterion "pinned vLLM hybrid attention / GDN paths
validated with dummy fixtures".

Writes ``logs/moe-preflight/gdn-fixture-probe.log`` (+ ``.json``).

Usage:
  .venv/bin/python scripts/gdn_fixture_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "logs" / "moe-preflight"
CONFIG_DIR = Path("/tmp/opencode/qwen35-3.09bpw")


PROBE_CONFIG_DIR = ROOT / "fixtures" / "gdn-probe"


def prepare_probe_config(config_dir: Path) -> Path:
    """Copy the pinned config, dropping ``quantization_config``.

    The GDN probe runs with dummy weights, so there is nothing to dequantize;
    the copy keeps every architecture dimension identical to the checkpoint
    while letting vLLM build plain linear layers.
    """
    raw = json.loads((config_dir / "config.json").read_text())
    removed = raw.pop("quantization_config", None)
    if isinstance(raw.get("text_config"), dict):
        raw["text_config"].pop("quantization_config", None)
    PROBE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (PROBE_CONFIG_DIR / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    (PROBE_CONFIG_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "P0-07 GDN fixture probe: real Qwen3.5-35B-A3B dimensions, "
                "no quantization, dummy weights",
                "source_config": str(config_dir / "config.json"),
                "removed_keys": ["quantization_config"],
                "quantization_removed": removed is not None,
            },
            indent=2,
        )
        + "\n"
    )
    return PROBE_CONFIG_DIR


def make_vllm_config(config_dir: Path):
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

    model_config = ModelConfig(
        model=str(config_dir),
        tokenizer=str(config_dir),
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,
        # The probe builds the layer with dummy tensors instead of loading the
        # EXL3 weights, so `process_weights_after_loading` never runs and the
        # dense EXL3 linear method has nothing to slice. The probe config has
        # `quantization_config` stripped, so vLLM builds plain linear layers;
        # EXL3 dense parity is covered by P0-04/P0-06 and the MoE adapter by
        # Phase 1.
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.5),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
    )


@contextmanager
def standalone_vllm(config_dir: Path):
    """vLLM's test harness equivalent: process group + current config."""
    os.environ.setdefault("VLLM_TARGET_DEVICE", "xpu")
    # torch.distributed's env:// rendezvous wants a master address even at
    # world_size=1.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")

    import torch

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    vllm_config = make_vllm_config(config_dir)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield vllm_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--prefill-tokens", type=int, default=8)
    parser.add_argument("--device", default="xpu:0")
    args = parser.parse_args()

    lines: list[str] = []
    record: dict = {
        "probe": "gdn_fixture",
        "device": args.device,
        "kernel": platform.release(),
        "checks": {},
        "errors": [],
        "pass": False,
    }

    def log(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    try:
        import torch

        import vllm_xpu_kernels._xpu_C  # noqa: F401  registers the SYCL ops
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )
        from vllm.utils.torch_utils import set_random_seed
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        probe_dir = prepare_probe_config(args.config_dir)
        with standalone_vllm(probe_dir) as vllm_config:
            cfg = vllm_config.model_config.hf_text_config
            device = torch.device(args.device)

            log("# P0-07 GDN / hybrid-attention fixture probe")
            log(f"torch: {torch.__version__}")
            log(f"device: {torch.xpu.get_device_name(0)}")
            log(f"config: {probe_dir} (from {args.config_dir}, quantization_config removed)")
            log(
                "model dims: "
                f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
                f"full_attention_interval={cfg.full_attention_interval} "
                f"k_heads={cfg.linear_num_key_heads} v_heads={cfg.linear_num_value_heads} "
                f"head_k={cfg.linear_key_head_dim} head_v={cfg.linear_value_head_dim} "
                f"conv_kernel={cfg.linear_conv_kernel_dim}"
            )
            record["model_dims"] = {
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "full_attention_interval": cfg.full_attention_interval,
                "linear_num_key_heads": cfg.linear_num_key_heads,
                "linear_num_value_heads": cfg.linear_num_value_heads,
                "linear_key_head_dim": cfg.linear_key_head_dim,
                "linear_value_head_dim": cfg.linear_value_head_dim,
                "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
                "mamba_ssm_dtype": getattr(cfg, "mamba_ssm_dtype", None),
            }

            pattern = [
                "full_attention"
                if (i + 1) % cfg.full_attention_interval == 0
                else "linear_attention"
                for i in range(cfg.num_hidden_layers)
            ]
            gdn_count = pattern.count("linear_attention")
            full_count = pattern.count("full_attention")
            log(
                f"layer pattern: {gdn_count} x linear_attention + {full_count} x "
                f"full_attention ({cfg.full_attention_interval}:1)"
            )
            record["layer_pattern"] = {
                "linear_attention": gdn_count,
                "full_attention": full_count,
                "repeats_every": cfg.full_attention_interval,
            }

            # ---- state shapes straight from vLLM's own calculator -----------
            conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp_world_size=1,
                num_k_heads=cfg.linear_num_key_heads,
                num_v_heads=cfg.linear_num_value_heads,
                head_k_dim=cfg.linear_key_head_dim,
                head_v_dim=cfg.linear_value_head_dim,
                conv_kernel_size=cfg.linear_conv_kernel_dim,
            )
            log(f"conv_state per sequence: {conv_shape}")
            log(f"ssm_state per sequence:  {ssm_shape}")
            record["state_shapes"] = {
                "conv_state": list(conv_shape),
                "ssm_state": list(ssm_shape),
            }

            # ---- build the layer -------------------------------------------
            set_random_seed(1234)
            layer = QwenGatedDeltaNetAttention(
                config=cfg,
                vllm_config=vllm_config,
                prefix="model.language_model.layers.0.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=True,
            ).eval()

            numel = sum(p.numel() for p in layer.parameters())
            nbytes = sum(p.numel() * p.element_size() for p in layer.parameters())
            log(f"layer params: {numel:,} ({nbytes / 2**20:.1f} MiB)")
            record["layer_params"] = {"numel": numel, "bytes": nbytes}

            # Deterministic finite dummy weights; this probe exercises the GDN
            # execution/state path, not numeric parity with the checkpoint.
            with torch.no_grad():
                for name, p in layer.named_parameters():
                    if "A_log" in name:
                        p.fill_(0.5)
                    elif "dt_bias" in name:
                        p.fill_(0.1)
                    elif "conv1d" in name:
                        p.copy_(torch.randn(p.shape) * 0.05)
                    elif "norm" in name.lower():
                        p.fill_(1.0)
                    else:
                        p.copy_(torch.randn(p.shape) * 0.02)

                # Match the checkpoint dtypes exactly: only ``A_log`` is F32,
                # ``dt_bias`` and every activation-path weight are BF16. The
                # XPU GDN kernel asserts dt_bias.dtype == core_attn_out.dtype.
                for name, p in layer.named_parameters():
                    leaf = name.rsplit(".", 1)[-1]
                    if p.dtype == torch.float32 and leaf != "A_log":
                        p.data = p.data.to(torch.bfloat16)

            layer = layer.to(device)

            num_seqs = 1
            conv_state = torch.zeros(
                (num_seqs, *conv_shape), dtype=torch.bfloat16, device=device
            )
            ssm_state = torch.zeros(
                (num_seqs, *ssm_shape), dtype=torch.float32, device=device
            )
            log(
                f"allocated conv_state {tuple(conv_state.shape)} "
                f"{conv_state.numel() * conv_state.element_size() / 2**20:.2f} MiB, "
                f"ssm_state {tuple(ssm_state.shape)} "
                f"{ssm_state.numel() * ssm_state.element_size() / 2**20:.2f} MiB "
                f"on {conv_state.device.type}"
            )
            record["state_allocation"] = {
                "conv_state_shape": list(conv_state.shape),
                "conv_state_dtype": str(conv_state.dtype),
                "conv_state_mib": conv_state.numel()
                * conv_state.element_size()
                / 2**20,
                "ssm_state_shape": list(ssm_state.shape),
                "ssm_state_dtype": str(ssm_state.dtype),
                "ssm_state_mib": ssm_state.numel() * ssm_state.element_size() / 2**20,
                "device": str(conv_state.device),
                "is_xpu": conv_state.device.type == "xpu",
            }

            # The engine binds the mamba cache into ``layer.kv_cache``.
            layer.kv_cache = (conv_state, ssm_state)
            vllm_config.compilation_config.static_forward_context[layer.prefix] = layer

            def run(
                num_tokens: int,
                num_prefills: int,
                num_decodes: int,
                x: torch.Tensor | None = None,
            ):
                if x is None:
                    x = (torch.randn(num_tokens, cfg.hidden_size) * 0.05).to(
                        torch.bfloat16
                    )
                    x = x.to(device)
                qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
                state_idx = torch.tensor([0], dtype=torch.int32, device=device)
                has_initial = torch.tensor([False], dtype=torch.bool, device=device)
                metadata = GDNAttentionMetadata(
                    num_prefills=num_prefills,
                    num_prefill_tokens=num_tokens if num_prefills else 0,
                    num_decodes=num_decodes,
                    num_decode_tokens=num_tokens if num_decodes else 0,
                    num_spec_decodes=0,
                    num_spec_decode_tokens=0,
                    num_actual_tokens=num_tokens,
                    has_initial_state=has_initial if num_prefills else None,
                    non_spec_query_start_loc=qsl,
                    non_spec_state_indices_tensor=state_idx,
                )
                with set_forward_context(
                    attn_metadata={layer.prefix: metadata},
                    vllm_config=vllm_config,
                    num_tokens=num_tokens,
                ):
                    out = layer.forward_xpu(x)
                torch.xpu.synchronize()
                return out

            results: dict = {}

            # ---- decode -----------------------------------------------------
            conv_before = conv_state.clone()
            ssm_before = ssm_state.clone()
            decode_out = None
            for _ in range(args.decode_steps):
                decode_out = run(1, num_prefills=0, num_decodes=1)
            conv_moved = not torch.equal(conv_before, conv_state)
            ssm_moved = not torch.equal(ssm_before, ssm_state)
            log("")
            log(
                f"decode x{args.decode_steps}: out {tuple(decode_out.shape)} "
                f"{decode_out.dtype}"
            )
            log(f"  finite: {bool(torch.isfinite(decode_out.float()).all())}")
            log(
                f"  out abs mean {decode_out.float().abs().mean().item():.6f} "
                f"max {decode_out.float().abs().max().item():.6f}"
            )
            log(f"  conv_state updated: {conv_moved}")
            log(f"  ssm_state  updated: {ssm_moved}")
            results["decode"] = {
                "out_shape": list(decode_out.shape),
                "out_dtype": str(decode_out.dtype),
                "finite": bool(torch.isfinite(decode_out.float()).all()),
                "out_abs_mean": decode_out.float().abs().mean().item(),
                "conv_state_updated": conv_moved,
                "ssm_state_updated": ssm_moved,
            }

            # ---- prefill ----------------------------------------------------
            conv_state.zero_()
            ssm_state.zero_()
            prefill_out = run(args.prefill_tokens, num_prefills=1, num_decodes=0)
            log("")
            log(
                f"prefill {args.prefill_tokens} tokens: out {tuple(prefill_out.shape)} "
                f"{prefill_out.dtype}"
            )
            log(f"  finite: {bool(torch.isfinite(prefill_out.float()).all())}")
            log(
                f"  out abs mean {prefill_out.float().abs().mean().item():.6f} "
                f"max {prefill_out.float().abs().max().item():.6f}"
            )
            log(f"  conv_state written: {bool(conv_state.abs().sum() > 0)}")
            log(f"  ssm_state  written: {bool(ssm_state.abs().sum() > 0)}")
            results["prefill"] = {
                "tokens": args.prefill_tokens,
                "out_shape": list(prefill_out.shape),
                "finite": bool(torch.isfinite(prefill_out.float()).all()),
                "out_abs_mean": prefill_out.float().abs().mean().item(),
                "conv_state_written": bool(conv_state.abs().sum() > 0),
                "ssm_state_written": bool(ssm_state.abs().sum() > 0),
            }

            # ---- determinism ------------------------------------------------
            conv0 = conv_state.clone()
            ssm0 = ssm_state.clone()
            fixed_x = (
                (torch.randn(args.prefill_tokens, cfg.hidden_size) * 0.05)
                .to(torch.bfloat16)
                .to(device)
            )
            out_a = run(
                args.prefill_tokens, num_prefills=1, num_decodes=0, x=fixed_x.clone()
            )
            conv_a, ssm_a = conv_state.clone(), ssm_state.clone()
            conv_state.copy_(conv0)
            ssm_state.copy_(ssm0)
            out_b = run(
                args.prefill_tokens, num_prefills=1, num_decodes=0, x=fixed_x.clone()
            )
            same_out = torch.equal(out_a, out_b)
            same_conv = torch.equal(conv_a, conv_state)
            same_ssm = torch.equal(ssm_a, ssm_state)
            log("")
            log(
                f"determinism (2 identical prefill runs): out={same_out} "
                f"conv={same_conv} ssm={same_ssm}"
            )
            results["determinism"] = {
                "out_identical": same_out,
                "conv_state_identical": same_conv,
                "ssm_state_identical": same_ssm,
            }

            # ---- allocation headroom ----------------------------------------
            props = torch.xpu.get_device_properties(0)
            free, total = torch.xpu.mem_get_info()
            mem_available = None
            try:
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1]) * 1024
                        break
            except OSError:
                pass
            log("")
            mem_note = (
                f", host MemAvailable {mem_available / 2**30:.2f} GiB"
                if mem_available
                else ""
            )
            log(
                f"XPU memory: free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB "
                "(mem_get_info; integrated GPU shares system RAM)"
                + mem_note
            )
            results["xpu_memory"] = {
                "free_bytes": free,
                "total_bytes": total,
                "host_mem_available_bytes": mem_available,
            }

            record["checks"] = results
            record["pass"] = bool(
                results["decode"]["finite"]
                and results["decode"]["conv_state_updated"]
                and results["decode"]["ssm_state_updated"]
                and results["prefill"]["finite"]
                and results["prefill"]["conv_state_written"]
                and results["prefill"]["ssm_state_written"]
                and results["determinism"]["out_identical"]
            )
    except Exception as exc:  # noqa: BLE001 - probe records failures verbatim
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        log("")
        log(f"FAILED: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        record["pass"] = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gdn-fixture-probe.log").write_text("\n".join(lines) + "\n")
    (args.out_dir / "gdn-fixture-probe.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(f"wrote {args.out_dir / 'gdn-fixture-probe.log'}")
    print(f"result: {'PASS' if record.get('pass') else 'FAIL'}")
    return 0 if record.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
