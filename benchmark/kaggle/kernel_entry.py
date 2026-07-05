"""Kaggle entry point for AxonScope benchmark campaigns.

This script is uploaded by `benchmark/kaggle/run_kernel.py`. It clones the
configured AxonScope branch, installs the benchmark dependencies, runs
`benchmark/run.py`, records Kaggle hardware metadata, and archives
`benchmark/results` as a downloadable output.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any


CONFIG_PATH = pathlib.Path(__file__).with_name("kaggle_config.json")
EMBEDDED_CONFIG: dict[str, Any] = {}
SENSITIVE_ENV_MARKERS = (
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)

WORK_DIR = pathlib.Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
CHECKOUT_DIR = pathlib.Path(os.environ.get("AXONSCOPE_CHECKOUT_DIR", "/tmp/AxonScope"))


def main() -> None:
    config = _load_config()
    run_id = str(config.get("run_id") or _default_run_id(config))
    output_dir = WORK_DIR / "benchmark" / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "kaggle_config_effective.json", config)
    _clone_repo(config)
    _install_repo(config)
    _verify_gpu_if_requested(config)
    _write_json(output_dir / "kaggle_hardware.json", _hardware_metadata(config))

    command = _benchmark_command(config, output_dir)
    _write_json(
        output_dir / "kaggle_command.json",
        {"command": command, "cwd": str(CHECKOUT_DIR)},
    )
    _run(command, cwd=CHECKOUT_DIR)

    archive_base = WORK_DIR / f"axonscope_benchmark_results_{run_id}"
    archive = shutil.make_archive(str(archive_base), "zip", output_dir)
    print(f"AxonScope benchmark results: {output_dir}")
    print(f"AxonScope benchmark archive: {archive}")


def _load_config() -> dict[str, Any]:
    config = dict(EMBEDDED_CONFIG)
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


def _default_run_id(config: dict[str, Any]) -> str:
    script = _safe_token(config.get("script", "benchmark"))
    preset = _safe_token(config.get("preset", "preset"))
    platform_name = _safe_token(config.get("platform", "platform"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"kaggle_{script}_{preset}_{platform_name}_{timestamp}"


def _clone_repo(config: dict[str, Any]) -> None:
    repo_url = str(config["repo_url"])
    branch = str(config["branch"])
    if CHECKOUT_DIR.exists():
        shutil.rmtree(CHECKOUT_DIR)
    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            repo_url,
            str(CHECKOUT_DIR),
        ]
    )
    _run(["git", "rev-parse", "--short", "HEAD"], cwd=CHECKOUT_DIR)


def _install_repo(config: dict[str, Any]) -> None:
    python = sys.executable
    install_target = str(config.get("install_target", ".[benchmark]"))
    _run([python, "-m", "pip", "install", "-U", "pip"])
    _run([python, "-m", "pip", "install", "-e", install_target], cwd=CHECKOUT_DIR)
    cuda_extra = str(config.get("jax_cuda_extra") or "").strip()
    if cuda_extra:
        _run([python, "-m", "pip", "install", "-U", f"jax[{cuda_extra}]"])


def _benchmark_command(config: dict[str, Any], output_dir: pathlib.Path) -> list[str]:
    command = [
        sys.executable,
        "benchmark/run.py",
        "--script",
        str(config["script"]),
        "--preset",
        str(config["preset"]),
        "--platform",
        str(config["platform"]),
        "--output",
        str(output_dir),
    ]
    command.extend(str(value) for value in config.get("benchmark_args", ()))
    return command


def _verify_gpu_if_requested(config: dict[str, Any]) -> None:
    if not bool(config.get("require_gpu", False)):
        return
    try:
        import jax
    except Exception as exc:
        raise RuntimeError(f"JAX import failed while checking GPU availability: {exc}") from exc
    platforms = {str(getattr(device, "platform", "")).lower() for device in jax.devices()}
    if not (platforms & {"gpu", "cuda", "rocm"}):
        raise RuntimeError(f"GPU was required, but visible JAX platforms are {sorted(platforms)}")


def _hardware_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "kaggle_env": _environment_snapshot(),
        "config": config,
    }
    metadata["git"] = _git_metadata()
    metadata["nvidia_smi"] = _command_snapshot(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        import jax

        metadata["jax"] = {
            "default_backend": jax.default_backend(),
            "devices": [
                {
                    "repr": str(device),
                    "platform": getattr(device, "platform", None),
                    "id": getattr(device, "id", None),
                    "device_kind": getattr(device, "device_kind", None),
                }
                for device in jax.devices()
            ],
        }
    except Exception as exc:
        metadata["jax_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _environment_snapshot() -> dict[str, str]:
    return {
        key: _redact_env_value(key, value)
        for key, value in os.environ.items()
        if key.startswith("KAGGLE") or key.startswith("CUDA") or key.startswith("NVIDIA")
    }


def _redact_env_value(key: str, value: str) -> str:
    name = key.upper()
    if any(marker in name for marker in SENSITIVE_ENV_MARKERS):
        return f"<redacted:{len(value)} chars>" if value else "<redacted>"
    return value


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _command_snapshot(["git", "rev-parse", "HEAD"], cwd=CHECKOUT_DIR),
        "short_commit": _command_snapshot(["git", "rev-parse", "--short", "HEAD"], cwd=CHECKOUT_DIR),
        "branch": _command_snapshot(["git", "branch", "--show-current"], cwd=CHECKOUT_DIR),
    }


def _command_snapshot(command: list[str], *, cwd: pathlib.Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _run(command: list[str], *, cwd: pathlib.Path | None = None) -> None:
    print("$", " ".join(shlex.quote(str(part)) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _safe_token(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"_", "-"} else "-" for char in text)


if __name__ == "__main__":
    main()
