# axonscope/utils/env.py

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil


def _safe_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _run_command(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return None


def _git_info() -> dict[str, Any]:
    status = _run_command(["git", "status", "--porcelain"])

    return {
        "commit": _run_command(["git", "rev-parse", "HEAD"]),
        "branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": _run_command(["git", "remote", "get-url", "origin"]),
        "is_dirty": bool(status),
    }


def _python_info() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "is_venv": sys.prefix != sys.base_prefix,
    }


def _os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }


def _cpu_info() -> dict[str, Any]:
    try:
        cpu_freq = psutil.cpu_freq()
    except Exception:
        cpu_freq = None

    return {
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "frequency_mhz": {
            "current": cpu_freq.current,
            "min": cpu_freq.min,
            "max": cpu_freq.max,
        }
        if cpu_freq
        else None,
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }


def _memory_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()

    return {
        "total_gb": round(vm.total / 1e9, 3),
        "available_gb": round(vm.available / 1e9, 3),
        "used_gb": round(vm.used / 1e9, 3),
        "percent": vm.percent,
    }


def _disk_info(path: str = "/") -> dict[str, Any]:
    du = psutil.disk_usage(path)

    return {
        "path": path,
        "total_gb": round(du.total / 1e9, 3),
        "used_gb": round(du.used / 1e9, 3),
        "free_gb": round(du.free / 1e9, 3),
        "percent": du.percent,
    }


def _package_versions() -> dict[str, str | None]:
    packages = [
        "axonscope",
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "mlx",
        "matplotlib",
        "pandas",
        "h5py",
        "psutil",
        "pytest",
        "pyinstrument",
        "memray",
        "nrv",
    ]

    return {pkg: _safe_version(pkg) for pkg in packages}


def _jax_info() -> dict[str, Any]:
    try:
        import jax

        devices = list(jax.devices())
        return {
            "available": True,
            "version": getattr(jax, "__version__", None),
            "default_backend": jax.default_backend(),
            "enable_x64": bool(jax.config.read("jax_enable_x64")),
            "devices": [str(device) for device in devices],
            "device_details": [_jax_device_info(device) for device in devices],
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
        }


def _jax_device_info(device: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "repr": str(device),
        "platform": getattr(device, "platform", None),
        "id": getattr(device, "id", None),
        "device_kind": getattr(device, "device_kind", None),
    }
    stats_fn = getattr(device, "memory_stats", None)
    if callable(stats_fn):
        try:
            stats = stats_fn() or {}
        except Exception as exc:
            info["memory_stats_error"] = repr(exc)
        else:
            info["memory_stats"] = {
                str(key): _json_safe_scalar(value)
                for key, value in dict(stats).items()
            }
    return info


def _gpu_info() -> dict[str, Any]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return {
            "available": False,
            "source": "nvidia-smi",
        }
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        name, driver, total_mib, used_mib, free_mib = parts
        rows.append(
            {
                "name": name,
                "driver_version": driver,
                "memory_total_mib": _parse_float(total_mib),
                "memory_used_mib": _parse_float(used_mib),
                "memory_free_mib": _parse_float(free_mib),
            }
        )
    return {
        "available": bool(rows),
        "source": "nvidia-smi",
        "devices": rows,
        "raw": output,
    }


def _mlx_info() -> dict[str, Any]:
    try:
        import mlx.core as mx

        return {
            "available": True,
            "version": _safe_version("mlx"),
            "default_device": str(mx.default_device()),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": repr(exc),
        }


def _environment_variables() -> dict[str, str | None]:
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "XLA_FLAGS",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "CUDA_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
    ]

    return {key: os.environ.get(key) for key in keys}


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        pass
    return str(value)


def collect_environment_info() -> dict[str, Any]:
    """
    Collect machine, backend, package, and runtime metadata.

    Useful for benchmark reproducibility, bug reports, and performance tracking.
    """
    return {
        "timestamp_unix": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": _os_info(),
        "python": _python_info(),
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "disk": _disk_info(),
        "packages": _package_versions(),
        "jax": _jax_info(),
        "gpu": _gpu_info(),
        "mlx": _mlx_info(),
        "environment_variables": _environment_variables(),
        "git": _git_info(),
    }


def save_environment_info(path: str | Path) -> dict[str, Any]:
    """
    Save environment information to a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    info = collect_environment_info()

    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    return info


def print_environment_info() -> None:
    """
    Print environment information as formatted JSON.
    """
    print(json.dumps(collect_environment_info(), indent=2, sort_keys=True))


if __name__ == "__main__":
    print_environment_info()
