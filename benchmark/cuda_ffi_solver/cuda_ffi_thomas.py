"""JAX FFI frontend for a benchmark-only CUDA block-Thomas solver."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


TARGET_NAME = "axonscope_double_cable_thomas_f32"
SYMBOL_NAME = "AxonScopeDoubleCableThomasF32"
SOURCE_PATH = Path(__file__).with_name("cuda_thomas_ffi.cu")

_registered = False
_loaded_library: ctypes.CDLL | None = None


def cuda_ffi_thomas_dependency_skip_reason() -> str | None:
    """Return why the benchmark-only CUDA FFI backend cannot run, if anything."""

    try:
        import jax
        import jax.ffi as ffi
    except ModuleNotFoundError as exc:
        return f"Python package {exc.name!r} is not installed."

    if not hasattr(ffi, "include_dir"):
        return "jax.ffi.include_dir() is unavailable."
    if not SOURCE_PATH.exists():
        return f"CUDA FFI source is missing: {SOURCE_PATH}."
    if _find_nvcc() is None:
        return "CUDA compiler 'nvcc' was not found on PATH or under /usr/local/cuda/bin."
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        return f"JAX GPU backend is unavailable: {exc}"
    if not devices:
        return "jax.devices('gpu') returned no devices."
    return None


def ensure_cuda_ffi_thomas_registered(*, force_rebuild: bool = False) -> Path:
    """Build/load the CUDA FFI shared library and register the JAX target."""

    global _registered, _loaded_library

    if _registered and _loaded_library is not None and not force_rebuild:
        return _library_path()

    skip_reason = cuda_ffi_thomas_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    import jax

    lib_path = build_cuda_ffi_thomas_library(force_rebuild=force_rebuild)
    _loaded_library = ctypes.CDLL(str(lib_path))
    handler = getattr(_loaded_library, SYMBOL_NAME)
    jax.ffi.register_ffi_target(
        TARGET_NAME,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )
    _registered = True
    return lib_path


def build_cuda_ffi_thomas_library(*, force_rebuild: bool = False) -> Path:
    """Compile the CUDA FFI solver into a shared library, cached under /tmp."""

    import jax
    import jax.ffi as ffi

    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError("CUDA compiler 'nvcc' was not found.")

    lib_path = _library_path()
    if lib_path.exists() and not force_rebuild:
        return lib_path

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-I",
        str(ffi.include_dir()),
        "-o",
        str(lib_path),
        str(SOURCE_PATH),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(
            "Failed to build CUDA FFI Thomas library with "
            f"JAX {jax.__version__}:\n$ {' '.join(command)}\n{output}"
        )
    return lib_path


def solve_block_tridiagonal_2x2_cuda_ffi_thomas(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    """Solve batched 2x2 block-tridiagonal systems through JAX FFI.

    Inputs are batch-first arrays. The experimental CUDA backend receives dense
    row-major ``[B, Nx]`` coefficient/RHS buffers and ``[B, Nx - 1]`` edge
    buffers. One-dimensional coefficient inputs are broadcast by JAX before the
    custom call, so the benchmark can test API shapes without making them public
    routing yet.
    """

    import jax
    import jax.numpy as jnp

    ensure_cuda_ffi_thomas_registered()

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    _check_rhs_pair(rhs0, rhs1)
    if rhs0.dtype != jnp.float32:
        raise TypeError(f"cuda_ffi_thomas supports only float32, got {rhs0.dtype}.")
    batch_size, nx = int(rhs0.shape[0]), int(rhs0.shape[1])

    arrays = (
        _space_tensor(a00, batch_size=batch_size, nx=nx, name="a00"),
        _space_tensor(a01, batch_size=batch_size, nx=nx, name="a01"),
        _space_tensor(a10, batch_size=batch_size, nx=nx, name="a10"),
        _space_tensor(a11, batch_size=batch_size, nx=nx, name="a11"),
        _edge_tensor(off0, batch_size=batch_size, nx=nx, name="off0"),
        _edge_tensor(off1, batch_size=batch_size, nx=nx, name="off1"),
        rhs0,
        rhs1,
    )

    result_shape_dtypes = (
        jax.ShapeDtypeStruct(rhs0.shape, rhs0.dtype),
        jax.ShapeDtypeStruct(rhs1.shape, rhs1.dtype),
    )
    return jax.ffi.ffi_call(
        TARGET_NAME,
        result_shape_dtypes,
        input_layouts=[(0, 1)] * len(arrays),
        output_layouts=[(0, 1), (0, 1)],
    )(*arrays)


def _check_rhs_pair(rhs0: Any, rhs1: Any) -> None:
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if tuple(rhs0.shape) != tuple(rhs1.shape):
        raise ValueError(f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}.")
    if int(rhs0.shape[1]) < 2:
        raise ValueError("Nx must be >= 2 for the CUDA FFI Thomas backend.")


def _space_tensor(values: Any, *, batch_size: int, nx: int, name: str):
    import jax.numpy as jnp

    tensor = jnp.asarray(values, dtype=jnp.float32)
    if tensor.ndim == 1 and int(tensor.shape[0]) == nx:
        return jnp.broadcast_to(tensor[None, :], (batch_size, nx))
    if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, nx):
        return tensor
    raise ValueError(f"{name} must have shape ({nx},) or ({batch_size}, {nx}), got {tuple(tensor.shape)}.")


def _edge_tensor(values: Any, *, batch_size: int, nx: int, name: str):
    import jax.numpy as jnp

    edge_n = nx - 1
    tensor = jnp.asarray(values, dtype=jnp.float32)
    if tensor.ndim == 1 and int(tensor.shape[0]) == edge_n:
        return jnp.broadcast_to(tensor[None, :], (batch_size, edge_n))
    if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, edge_n):
        return tensor
    raise ValueError(
        f"{name} must have shape ({edge_n},) or ({batch_size}, {edge_n}), got {tuple(tensor.shape)}."
    )


def _library_path() -> Path:
    build_dir = Path(
        os.environ.get(
            "AXONSCOPE_CUDA_FFI_BUILD_DIR",
            str(Path(tempfile.gettempdir()) / "axonscope_cuda_ffi"),
        )
    )
    digest = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:12]
    return build_dir / f"libaxonscope_cuda_ffi_thomas_{digest}.so"


def _find_nvcc() -> str | None:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    candidate = Path("/usr/local/cuda/bin/nvcc")
    if candidate.exists():
        return str(candidate)
    return None

