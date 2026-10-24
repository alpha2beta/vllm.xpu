#!/usr/bin/env python3
"""P0-01 platform pin collector for the EXL3 MoE (Qwen3.5-35B-A3B) bring-up.

Collects the immutable software/hardware baseline required by `tasks_moe.md`
and writes a human-readable manifest plus a machine-readable JSON sidecar:

    logs/moe-preflight/environment-moe.txt
    logs/moe-preflight/environment-moe.json

Read-only against the host. Does not import vLLM (that pulls in the XPU
context); versions are read from package metadata instead.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXL3_SRC = Path(os.environ.get("EXL3_SRC", "/home/yanchun/Projects/exl3xpu"))

PACKAGES = [
    "vllm",
    "torch",
    "transformers",
    "triton",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "numpy",
    "exl3xpu",
]


def run(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - probe must never crash
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def git_head(path: Path) -> dict[str, str]:
    if not (path / ".git").exists():
        return {"path": str(path), "commit": "<not a git repo>", "describe": ""}
    return {
        "path": str(path),
        "commit": run(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "branch": run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": run(["git", "-C", str(path), "describe", "--always", "--dirty"]),
    }


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<not installed>"


def read_sysfs(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def xpu_properties() -> dict:
    """Device properties independent of model loading."""
    try:
        import torch

        if not torch.xpu.is_available():
            return {"available": False}
        props = torch.xpu.get_device_properties(0)
        values = {}
        for key in dir(props):
            if key.startswith("_"):
                continue
            try:
                value = getattr(props, key)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, (str, int, float, bool, list, tuple)):
                values[key] = value
            elif hasattr(value, "tolist"):
                values[key] = value.tolist()
        free, total = torch.xpu.mem_get_info()
        return {
            "available": True,
            "device_count": torch.xpu.device_count(),
            "properties": values,
            "mem_info": {"free_bytes": free, "total_bytes": total},
            "torch_xpu": torch.xpu.__name__,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def level_zero_library() -> dict:
    candidates = [
        "/usr/lib/libze_loader.so.1",
        "/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
        "/usr/local/lib/libze_loader.so.1",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            real = path.resolve()
            return {
                "soname_path": candidate,
                "resolved": str(real),
                "version_suffix": real.name,
                "size_bytes": real.stat().st_size,
            }
    return {"error": "libze_loader.so.1 not found in standard locations"}


def gpu_driver() -> dict:
    info: dict[str, str | None] = {}
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        if not device.exists():
            continue
        info["card"] = card.name
        info["pci_id"] = read_sysfs(str(device / "device"))
        info["vendor_id"] = read_sysfs(str(device / "vendor"))
        driver = device / "driver"
        if driver.exists():
            info["driver"] = os.path.basename(os.readlink(driver))
        info["xe_max_mem"] = read_sysfs(
            "/sys/module/xe/parameters/max_vfs"
        )  # harmless presence marker
        break
    info["ttm_pages_limit"] = read_sysfs("/sys/module/ttm/parameters/pages_limit")
    info["ttm_page_pool_size"] = read_sysfs(
        "/sys/module/ttm/parameters/page_pool_size"
    )
    return info


def memory_state() -> dict:
    data: dict[str, object] = {}
    text = Path("/proc/meminfo").read_text()
    for line in text.splitlines():
        match = re.match(r"^(\w+):\s+(\d+) kB$", line)
        if match:
            key = match.group(1)
            data[key] = int(match.group(2)) * 1024
    data["swap_in_use_bytes"] = data.get("SwapTotal", 0) - data.get("SwapFree", 0)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "logs" / "moe-preflight",
        help="directory for environment-moe.{txt,json}",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "os": run(["bash", "-lc", "source /etc/os-release; echo $PRETTY_NAME"]),
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "<unknown>",
            ),
            "logical_cpus": os.cpu_count(),
            "page_size": os.sysconf("SC_PAGE_SIZE"),
        },
        "memory": memory_state(),
        "gpu": {
            "torch_xpu": xpu_properties(),
            "driver": gpu_driver(),
            "level_zero_loader": level_zero_library(),
            "sycl_ls": run(
                [
                    "/opt/intel/oneapi/compiler/latest/bin/sycl-ls",
                    "--verbose",
                ]
            ),
            "xpu_smi": run(["xpu-smi", "discovery"]),
        },
        "software": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {name: pkg_version(name) for name in PACKAGES},
            "oneapi_icpx": run(
                ["/opt/intel/oneapi/compiler/latest/bin/icpx", "--version"]
            ),
            "oneapi_env_script": Path("/opt/intel/oneapi/setvars.sh").exists(),
        },
        "repositories": {
            "vllm_xpu": git_head(ROOT),
            "exl3xpu": git_head(EXL3_SRC),
        },
        "runtime_env": {
            key: os.environ.get(key)
            for key in (
                "EXL3_VLLM_PATCHES",
                "EXL3_BACKEND",
                "EXL3_SMALL_M_MAX",
                "EXL3_INT8_PREFILL",
                "EXL3_RECON_SLICE_N",
                "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS",
                "ZE_AFFINITY_MASK",
                "VLLM_LOGGING_LEVEL",
                "OMP_NUM_THREADS",
                "HF_HOME",
            )
        },
        "disk": {
            "vllm_xpu_root": shutil.disk_usage(ROOT)._asdict(),
            "models_dir_exists": (ROOT / "models").is_dir(),
        },
    }

    json_path = args.out_dir / "environment-moe.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n")

    gpu_props = record["gpu"]["torch_xpu"].get("properties", {})  # type: ignore[union-attr]
    mem = record["memory"]
    lines = [
        "# EXL3 MoE preflight environment (P0-01)",
        f"generated_at_utc: {record['generated_at_utc']}",
        f"json_sidecar: {json_path}",
        "",
        "## Host",
        f"hostname: {record['host']['hostname']}",
        f"kernel: {record['host']['kernel']}",
        f"os: {record['host']['os']}",
        f"cpu: {record['host']['cpu']}",
        f"logical_cpus: {record['host']['logical_cpus']}",
        "",
        "## Memory (binary units, /proc/meminfo)",
        f"MemTotal: {mem['MemTotal'] / 2**30:.2f} GiB",
        f"MemAvailable: {mem['MemAvailable'] / 2**30:.2f} GiB",
        f"SwapTotal: {mem['SwapTotal'] / 2**30:.2f} GiB",
        f"SwapInUse: {mem['swap_in_use_bytes'] / 2**30:.2f} GiB",
        f"ttm.pages_limit: {record['gpu']['driver']['ttm_pages_limit']}",
        "",
        "## GPU (torch.xpu)",
        f"name: {gpu_props.get('name')}",
        f"platform: {gpu_props.get('platform_name')}",
        f"driver_version: {gpu_props.get('driver_version')}",
        f"total_memory: {gpu_props.get('total_memory', 0) / 2**30:.2f} GiB",
        f"max_compute_units: {gpu_props.get('max_compute_units')}",
        f"L2 cache: {gpu_props.get('last_level_cache_size', 0) / 2**10:.0f} KiB",
        f"memory_bus_width: {gpu_props.get('memory_bus_width')} bits",
        f"sub_group_sizes: {gpu_props.get('sub_group_sizes')}",
        f"is_integrated_gpu: {gpu_props.get('is_integrated_gpu')}",
        f"level_zero_loader: {record['gpu']['level_zero_loader']}",
        "",
        "## Software pins",
        *[
            f"{name}: {version}"
            for name, version in record["software"]["packages"].items()  # type: ignore[union-attr]
        ],
        f"python: {record['software']['python']}",
        f"icpx: {record['software']['oneapi_icpx'].splitlines()[0]}",
        "",
        "## Repositories",
        *[
            f"{name}: {head['commit']} ({head.get('branch', '')}, {head.get('describe', '')})"
            for name, head in record["repositories"].items()  # type: ignore[union-attr]
        ],
        "",
        "## Runtime environment",
        *[
            f"{key}={value}"
            for key, value in record["runtime_env"].items()  # type: ignore[union-attr]
        ],
    ]
    txt_path = args.out_dir / "environment-moe.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    print(txt_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
