"""JAX device-memory snapshots for benchmark instrumentation."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def benchmark_device_memory_snapshot() -> dict[str, Any]:
    """Return a best-effort JAX/NVIDIA memory snapshot."""

    jax_devices = _jax_device_snapshot()
    nvidia_smi = _nvidia_smi_snapshot()
    snapshot = {"jax_devices": jax_devices, "nvidia_smi": nvidia_smi}
    snapshot.update(_device_totals(jax_devices, nvidia_smi))
    return snapshot


def _jax_device_snapshot() -> list[dict[str, Any]]:
    try:
        import jax

        devices = jax.devices()
    except Exception as exc:
        return [{"available": False, "error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for device in devices:
        row = {
            "repr": str(device),
            "platform": getattr(device, "platform", None),
            "id": getattr(device, "id", None),
            "device_kind": getattr(device, "device_kind", None),
        }
        stats = getattr(device, "memory_stats", None)
        if callable(stats):
            try:
                row["memory_stats"] = _json_safe_dict(dict(stats() or {}))
            except Exception as exc:
                row["memory_stats_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _nvidia_smi_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return {"available": False, "source": "nvidia-smi"}
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total, used, free = parts
        devices.append(
            {
                "index": _number_or_none(index),
                "name": name,
                "memory_total_mib": _number_or_none(total),
                "memory_used_mib": _number_or_none(used),
                "memory_free_mib": _number_or_none(free),
            }
        )
    return {"available": bool(devices), "source": "nvidia-smi", "devices": devices}


def _device_totals(
    jax_devices: Sequence[Mapping[str, Any]],
    nvidia_smi: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "device_bytes_in_use": _sum_jax_stat(
            jax_devices,
            ("bytes_in_use", "bytes_used", "used_bytes"),
        ),
        "device_peak_bytes_in_use": _sum_jax_stat(
            jax_devices,
            ("peak_bytes_in_use", "peak_bytes"),
        ),
        "nvidia_smi_memory_used_mib": _sum_smi(nvidia_smi, "memory_used_mib"),
        "nvidia_smi_memory_total_mib": _sum_smi(nvidia_smi, "memory_total_mib"),
    }


def _sum_jax_stat(devices: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int | None:
    total = 0
    found = False
    for device in devices:
        stats = _mapping(device.get("memory_stats"))
        for key in keys:
            value = _number_or_none(stats.get(key))
            if value is not None:
                total += int(value)
                found = True
                break
    return total if found else None


def _sum_smi(snapshot: Mapping[str, Any], key: str) -> float | None:
    total = 0.0
    found = False
    for device in _sequence(snapshot.get("devices")):
        if not isinstance(device, Mapping):
            continue
        value = _number_or_none(device.get(key))
        if value is not None:
            total += float(value)
            found = True
    return total if found else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in values.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
