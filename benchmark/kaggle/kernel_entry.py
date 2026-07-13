"""Kaggle entry point for AxonScope benchmark campaigns.

This script is uploaded by `benchmark/kaggle/run_kernel.py`. It clones the
configured AxonScope branch, installs the benchmark dependencies, runs
`benchmark/run.py` or a benchmark campaign runner, records Kaggle hardware
metadata, and archives `benchmark/results` as a downloadable output.
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
import time
import traceback
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
SCRATCH_DIR = pathlib.Path(os.environ.get("AXONSCOPE_KAGGLE_SCRATCH_DIR", "/tmp/axonscope_kaggle"))
CHECKOUT_DIR = pathlib.Path(os.environ.get("AXONSCOPE_CHECKOUT_DIR", "/tmp/AxonScope"))
PYTHON_EXECUTABLE = pathlib.Path(sys.executable)
os.environ.setdefault("MAMBA_ROOT_PREFIX", str(SCRATCH_DIR / "micromamba_root"))


def main() -> None:
    config = _load_config()
    run_id = str(config.get("run_id") or _default_run_id(config))
    output_dir = WORK_DIR / "benchmark" / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    try:
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
    except BaseException as exc:
        _write_json(
            output_dir / "kaggle_failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        traceback.print_exc()
        exit_code = 1
        raise
    finally:
        _write_process_snapshot(output_dir / "kaggle_processes_before_cleanup.txt")
        archive_base = WORK_DIR / f"axonscope_benchmark_results_{run_id}"
        archive = shutil.make_archive(str(archive_base), "zip", output_dir)
        print(f"AxonScope benchmark results: {output_dir}")
        print(f"AxonScope benchmark archive: {archive}")
        _terminate_native_leftovers()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def _load_config() -> dict[str, Any]:
    config = dict(EMBEDDED_CONFIG)
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


def _default_run_id(config: dict[str, Any]) -> str:
    script = _safe_token(config.get("campaign") or config.get("script") or "benchmark")
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
    global PYTHON_EXECUTABLE
    if bool(config.get("nrv_conda_env", False)):
        PYTHON_EXECUTABLE = _install_nrv_stack()
    python = str(PYTHON_EXECUTABLE)
    install_target = str(config.get("install_target", ".[benchmark]"))
    apt_packages = [
        str(package)
        for package in config.get("apt_packages", ())
        if str(package).strip()
    ]
    if apt_packages:
        _run(["apt-get", "update"])
        _run(["apt-get", "install", "-y", *apt_packages])
    _run([python, "-m", "pip", "install", "-U", "pip"])
    _run([python, "-m", "pip", "install", "-e", install_target], cwd=CHECKOUT_DIR)
    cuda_extra = str(config.get("jax_cuda_extra") or "").strip()
    if cuda_extra:
        _run([python, "-m", "pip", "install", "-U", f"jax[{cuda_extra}]"])
    pip_packages = [
        str(package)
        for package in config.get("pip_packages", ())
        if str(package).strip()
    ]
    if pip_packages:
        _run([python, "-m", "pip", "install", *pip_packages])


def _benchmark_command(config: dict[str, Any], output_dir: pathlib.Path) -> list[str]:
    python = str(PYTHON_EXECUTABLE)
    campaign = str(config.get("campaign") or "").strip()
    if campaign == "time_chunk_sweep":
        command = [
            python,
            "benchmark/campaigns/time_chunk_sweep.py",
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
    if campaign == "double_cable_solver_policy":
        command = [
            python,
            "benchmark/campaigns/double_cable_solver_policy.py",
            "--preset",
            str(config["preset"]),
            "--platform",
            str(config["platform"]),
            "--output",
            str(output_dir),
        ]
        command.extend(str(value) for value in config.get("benchmark_args", ()))
        return command
    if campaign == "single_cable_solver_policy":
        command = [
            python,
            "benchmark/campaigns/single_cable_solver_policy.py",
            "--preset",
            str(config["preset"]),
            "--platform",
            str(config["platform"]),
            "--output",
            str(output_dir),
        ]
        command.extend(str(value) for value in config.get("benchmark_args", ()))
        return command
    if campaign:
        raise RuntimeError(f"Unsupported benchmark campaign: {campaign!r}")
    command = [
        python,
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


def _install_nrv_stack() -> pathlib.Path:
    micromamba = _install_micromamba()
    env_dir = SCRATCH_DIR / "axonscope_nrv_env"
    env_yaml = SCRATCH_DIR / "nrv_linux.yaml"
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "curl",
            "-L",
            "-o",
            str(env_yaml),
            "https://raw.githubusercontent.com/nrv-framework/NRV/refs/heads/master/conda/nrv_linux.yaml",
        ],
        cwd=CHECKOUT_DIR,
    )
    _run(
        [
            str(micromamba),
            "create",
            "-y",
            "-p",
            str(env_dir),
            "-f",
            str(env_yaml),
        ],
        cwd=CHECKOUT_DIR,
    )
    env_yaml.unlink(missing_ok=True)
    _configure_conda_env_environment(env_dir)
    env_python = env_dir / "bin" / "python"
    _run([str(env_python), "-m", "pip", "install", "-U", "pip"])
    _run(
        [
            str(env_python),
            "-m",
            "pip",
            "install",
            "nrv-py",
            "opencv-python-headless",
        ],
        cwd=CHECKOUT_DIR,
    )
    return env_python


def _install_micromamba() -> pathlib.Path:
    micromamba_root = SCRATCH_DIR / "micromamba"
    micromamba = micromamba_root / "bin" / "micromamba"
    if micromamba.exists():
        return micromamba
    micromamba.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        f"| tar -xj -C {shlex.quote(str(micromamba_root))} bin/micromamba"
    )
    _run(["bash", "-lc", command], cwd=CHECKOUT_DIR)
    return micromamba


def _configure_conda_env_environment(env_dir: pathlib.Path) -> None:
    env_bin = env_dir / "bin"
    env_lib = env_dir / "lib"
    os.environ["CONDA_PREFIX"] = str(env_dir)
    os.environ["PATH"] = os.pathsep.join(
        [str(env_bin), os.environ.get("PATH", "")]
    )
    # Keep conda libraries visible for NRV/mpi4py, but do not expose Kaggle's
    # CUDA toolkit libraries to JAX CUDA pip wheels. The wheels provide CUDA
    # runtime libraries themselves; Kaggle only needs to provide libcuda.so.
    ld_paths = _nrv_safe_ld_library_paths(env_lib)
    if ld_paths:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    print(f"Using NRV conda env: {env_dir}")
    print(f"PATH starts with: {env_bin}")
    print(f"LD_LIBRARY_PATH for JAX CUDA: {os.environ.get('LD_LIBRARY_PATH', '')}")


def _nrv_safe_ld_library_paths(env_lib: pathlib.Path) -> list[str]:
    existing = [
        path
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    ]
    paths = [
        str(env_lib),
        str(env_lib.resolve()),
        str(env_lib / "openmpi"),
        str(env_lib.resolve() / "openmpi"),
    ]
    paths.extend(
        path
        for path in existing
        if "/cuda" not in path.lower() and path not in paths
    )
    for candidate in (
        pathlib.Path("/usr/local/nvidia/lib64"),
        pathlib.Path("/usr/local/nvidia/lib"),
    ):
        text = str(candidate)
        if candidate.exists() and text not in paths:
            paths.append(text)
    return paths


def _verify_gpu_if_requested(config: dict[str, Any]) -> None:
    if not bool(config.get("require_gpu", False)):
        return
    code = (
        "import json, jax; "
        "devices=jax.devices(); "
        "print(json.dumps({"
        "'default_backend': jax.default_backend(), "
        "'devices': [{'platform': getattr(d, 'platform', None), "
        "'device_kind': getattr(d, 'device_kind', None), "
        "'repr': str(d)} for d in devices]}))"
    )
    result = subprocess.run(
        [str(PYTHON_EXECUTABLE), "-c", code],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        env={**os.environ, "JAX_PLATFORMS": "cuda,cpu"},
    )
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(
            "JAX GPU check failed in benchmark Python "
            f"{PYTHON_EXECUTABLE}: {result.stderr.strip()}"
        )
    payload = json.loads(result.stdout)
    platforms = {
        str(device.get("platform", "")).lower()
        for device in payload.get("devices", ())
    }
    if not (platforms & {"gpu", "cuda", "rocm"}):
        raise RuntimeError(
            f"GPU was required in {PYTHON_EXECUTABLE}, but visible JAX "
            f"platforms are {sorted(platforms)}"
        )


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
    metadata["jax"] = _jax_metadata_for_python(PYTHON_EXECUTABLE)
    return metadata


def _jax_metadata_for_python(python: pathlib.Path) -> dict[str, Any]:
    code = (
        "import json, jax; "
        "print(json.dumps({"
        "'default_backend': jax.default_backend(), "
        "'devices': [{'repr': str(d), "
        "'platform': getattr(d, 'platform', None), "
        "'id': getattr(d, 'id', None), "
        "'device_kind': getattr(d, 'device_kind', None)} "
        "for d in jax.devices()]}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        return {
            "python": str(python),
            "available": False,
            "stderr": result.stderr.strip(),
            "stdout": result.stdout.strip(),
        }
    payload = json.loads(result.stdout)
    payload["python"] = str(python)
    payload["available"] = True
    return payload


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


def _write_process_snapshot(path: pathlib.Path) -> None:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,pgid,stat,comm,args"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        _write_json(
            path.with_suffix(".json"),
            {"type": type(exc).__name__, "message": str(exc)},
        )
        return
    path.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")


def _terminate_native_leftovers() -> None:
    descendants = _current_process_descendants()
    if descendants:
        subprocess.run(
            ["kill", "-TERM", *[str(pid) for pid in descendants]],
            check=False,
        )
        time.sleep(1.0)
        subprocess.run(
            ["kill", "-KILL", *[str(pid) for pid in descendants]],
            check=False,
        )
    pattern = r"(mpiexec|mpirun|orted|prte|pmix|hydra|nrniv|/special|gmsh)"
    subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
    time.sleep(1.0)
    subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)


def _current_process_descendants() -> list[int]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return []
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)

    current = os.getpid()
    descendants: list[int] = []
    stack = list(children.get(current, ()))
    while stack:
        pid = stack.pop()
        if pid == current:
            continue
        descendants.append(pid)
        stack.extend(children.get(pid, ()))
    return sorted(set(descendants), reverse=True)


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _safe_token(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"_", "-"} else "-" for char in text)


if __name__ == "__main__":
    main()
