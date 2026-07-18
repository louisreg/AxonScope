"""Exact node-first single-cable Thomas solver implemented with jax-triton."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any


_KERNEL_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

try:  # pragma: no cover - exercised on CUDA benchmark workers.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - common local CPU/dev path.
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _tiled_scalar_thomas_loop_kernel(
        dl,
        d,
        du,
        rhs,
        cprime,
        solution,
        N: tl.constexpr,
        B: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        tile = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_B)
        batch = tile * BLOCK_B + lanes
        mask = batch < B

        offset = batch
        diagonal = tl.load(d + offset, mask=mask, other=1.0)
        cp = tl.load(du + offset, mask=mask, other=0.0) / diagonal
        dp = tl.load(rhs + offset, mask=mask, other=0.0) / diagonal
        tl.store(cprime + offset, cp, mask=mask)
        tl.store(solution + offset, dp, mask=mask)

        previous_cp = cp
        previous_dp = dp
        if N > 2:
            for x in tl.range(1, N - 1):
                offset = x * B + batch
                lower = tl.load(dl + offset, mask=mask, other=0.0)
                denominator = (
                    tl.load(d + offset, mask=mask, other=1.0)
                    - lower * previous_cp
                )
                cp = tl.load(du + offset, mask=mask, other=0.0) / denominator
                dp = (
                    tl.load(rhs + offset, mask=mask, other=0.0)
                    - lower * previous_dp
                ) / denominator
                tl.store(cprime + offset, cp, mask=mask)
                tl.store(solution + offset, dp, mask=mask)
                previous_cp = cp
                previous_dp = dp

        x = N - 1
        offset = x * B + batch
        lower = tl.load(dl + offset, mask=mask, other=0.0)
        denominator = (
            tl.load(d + offset, mask=mask, other=1.0)
            - lower * previous_cp
        )
        value = (
            tl.load(rhs + offset, mask=mask, other=0.0)
            - lower * previous_dp
        ) / denominator
        tl.store(cprime + offset, 0.0, mask=mask)
        tl.store(solution + offset, value, mask=mask)

        for reverse_index in tl.range(0, N - 1):
            x = N - 2 - reverse_index
            offset = x * B + batch
            value = (
                tl.load(solution + offset, mask=mask, other=0.0)
                - tl.load(cprime + offset, mask=mask, other=0.0) * value
            )
            tl.store(solution + offset, value, mask=mask)

else:
    _tiled_scalar_thomas_loop_kernel = None


def single_cable_triton_import_skip_reason() -> str | None:
    """Return why the optional Triton lowering cannot be imported."""

    if importlib.util.find_spec("triton") is None:
        return "Python package 'triton' is not installed."
    if importlib.util.find_spec("jax_triton") is None:
        return "Python package 'jax-triton' is not installed."
    try:
        import jax_triton  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        return f"Could not import jax_triton: {exc}"
    return None


def single_cable_triton_dependency_skip_reason() -> str | None:
    """Return why the retained CUDA single-cable route cannot run."""

    import_reason = single_cable_triton_import_skip_reason()
    if import_reason is not None:
        return import_reason
    import jax

    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        return f"JAX GPU backend is unavailable: {exc}"
    if not devices:
        return "jax.devices('gpu') returned no devices."
    return None


def solve_single_cable_tridiagonal_xb(
    dl: Any,
    d: Any,
    du: Any,
    rhs: Any,
    *,
    block_b: int = 128,
) -> Any:
    """Solve independent node-first ``[Nx, B]`` scalar systems exactly."""

    skip_reason = single_cable_triton_import_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if int(block_b) < 1:
        raise ValueError("block_b must be >= 1.")

    import jax
    import jax.numpy as jnp

    from axonscope.runtime.jax.kernels.triton_call_cache import cached_triton_call

    arrays = tuple(jnp.asarray(value) for value in (dl, d, du, rhs))
    shape = tuple(arrays[0].shape)
    if len(shape) != 2 or shape[0] < 2:
        raise ValueError("Tridiagonal arrays must have shape (Nx, batch_size), Nx >= 2.")
    for name, value in zip(("dl", "d", "du", "rhs"), arrays, strict=True):
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} must use float32, got {value.dtype}.")

    nx, batch_size = map(int, shape)
    out_shape = jax.ShapeDtypeStruct(shape, arrays[0].dtype)
    _, solution = cached_triton_call(
        *arrays,
        kernel=_tiled_scalar_thomas_loop_kernel,
        source_hash=_KERNEL_SOURCE_HASH,
        out_shape=(out_shape, out_shape),
        grid=((batch_size + int(block_b) - 1) // int(block_b),),
        name="axonscope_single_cable_tiled_thomas",
        N=nx,
        B=batch_size,
        BLOCK_B=int(block_b),
        num_warps=_num_warps_for_block_b(int(block_b)),
        num_stages=1,
    )
    return solution


def _num_warps_for_block_b(block_b: int) -> int:
    if block_b >= 128:
        return 4
    if block_b >= 64:
        return 2
    return 1


__all__ = [
    "single_cable_triton_dependency_skip_reason",
    "single_cable_triton_import_skip_reason",
    "solve_single_cable_tridiagonal_xb",
]
