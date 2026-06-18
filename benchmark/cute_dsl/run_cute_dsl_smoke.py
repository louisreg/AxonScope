"""Run a minimal CuTe DSL / JAX custom-call smoke.

The runner intentionally skips cleanly on unsupported environments. CuTe DSL is
interesting for future custom solver kernels, but it currently needs SM 8.0+
hardware, so Kaggle P100/T4 runs are expected to skip.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda")

import jax
import jax.numpy as jnp
import numpy as np


MIN_SM_MAJOR = 8
BLOCK = 256


@dataclass(frozen=True)
class EnvironmentReport:
    """Compatibility information for the current CuTe DSL runtime."""

    jax_version: str
    jax_backend: str
    jax_devices: list[str]
    cutlass_available: bool
    cutlass_version: str | None
    cuda_bindings_available: bool
    gpu_name: str | None
    compute_capability: str | None
    compatible: bool
    reason: str | None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096, help="Vector length.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Return non-zero on skip.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    if args.n < 1:
        raise ValueError("--n must be >= 1.")

    report = environment_report()
    payload: dict[str, object] = {"environment": asdict(report)}
    if not report.compatible:
        payload["status"] = "skipped"
        payload["reason"] = report.reason
        write_payload(args.json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2 if args.strict else 0

    result = run_vector_add_smoke(n=int(args.n), seed=int(args.seed))
    payload.update({"status": "passed", "result": result})
    write_payload(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def environment_report() -> EnvironmentReport:
    cutlass_available = _module_available("cutlass")
    cuda_available = _module_available("cuda")
    cutlass_version = None
    if cutlass_available:
        try:
            cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
        except importlib.metadata.PackageNotFoundError:
            cutlass_version = None

    gpu_name, compute_capability = query_first_gpu()
    reason = None
    compatible = True
    backend = jax.default_backend()
    if backend != "gpu":
        compatible = False
        reason = f"JAX backend is {backend!r}, expected 'gpu'."
    elif not cutlass_available:
        compatible = False
        reason = "Python package 'nvidia-cutlass-dsl' / module 'cutlass' is not installed."
    elif not cuda_available:
        compatible = False
        reason = "Python package 'cuda-bindings' / module 'cuda' is not installed."
    elif compute_capability is None:
        compatible = False
        reason = "Could not query GPU compute capability with nvidia-smi."
    else:
        major, _minor = parse_compute_capability(compute_capability)
        if major < MIN_SM_MAJOR:
            compatible = False
            reason = (
                f"CuTe DSL requires SM {MIN_SM_MAJOR}.0+; detected "
                f"SM {compute_capability} on {gpu_name or 'unknown GPU'}."
            )

    return EnvironmentReport(
        jax_version=jax.__version__,
        jax_backend=backend,
        jax_devices=[str(device) for device in jax.devices()],
        cutlass_available=cutlass_available,
        cutlass_version=cutlass_version,
        cuda_bindings_available=cuda_available,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        compatible=compatible,
        reason=reason,
    )


def run_vector_add_smoke(*, n: int, seed: int) -> dict[str, float | int]:
    import cutlass.jax as cjax

    from benchmark.cute_dsl.cute_dsl_jax_kernels import launch_vector_add

    @jax.jit
    def jax_vector_add(a, b):
        padded = ((a.shape[0] + BLOCK - 1) // BLOCK) * BLOCK
        a_pad = jnp.pad(a, (0, padded - a.shape[0]))
        b_pad = jnp.pad(b, (0, padded - b.shape[0]))
        a_3d = a_pad.reshape(1, BLOCK, padded // BLOCK)
        b_3d = b_pad.reshape(1, BLOCK, padded // BLOCK)
        call = cjax.cutlass_call(
            launch_vector_add,
            output_shape_dtype=jax.ShapeDtypeStruct.like(a_3d),
            use_static_tensors=True,
        )
        c_3d = call(a_3d, b_3d)
        return c_3d.reshape(-1)[: a.shape[0]]

    key_a, key_b = jax.random.split(jax.random.key(seed))
    a = jax.random.normal(key_a, (n,), dtype=jnp.float32)
    b = jax.random.normal(key_b, (n,), dtype=jnp.float32)

    compile_start = time.perf_counter()
    out = jax_vector_add(a, b).block_until_ready()
    first_run_ms = (time.perf_counter() - compile_start) * 1e3

    start = time.perf_counter()
    out = jax_vector_add(a, b).block_until_ready()
    steady_ms = (time.perf_counter() - start) * 1e3

    ref = a + b
    max_abs_error = float(jnp.max(jnp.abs(out - ref)))
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-6)
    return {
        "n": int(n),
        "first_run_ms": float(first_run_ms),
        "steady_ms": float(steady_ms),
        "max_abs_error": max_abs_error,
    }


def query_first_gpu() -> tuple[str | None, str | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    return parse_nvidia_smi_gpu_line(output.strip().splitlines()[0])


def parse_nvidia_smi_gpu_line(line: str) -> tuple[str | None, str | None]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def parse_compute_capability(value: str) -> tuple[int, int]:
    major_text, minor_text = value.strip().split(".", 1)
    return int(major_text), int(minor_text)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def write_payload(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
