from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from axonscope.solvers.common import Array


def solve_block_tridiagonal_2x2_pallas_thomas_batched(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    block_b: int = 128,
    interpret: bool | None = None,
    num_warps: int = 4,
) -> tuple[Array, Array]:
    """Benchmark-only Pallas block-Thomas solve for batch-first 2x2 systems.

    One Pallas program handles ``block_b`` fibers and the full ``Nx`` cable.
    This is a Phase 3 spike baseline, not a public solver backend.
    """

    a00 = jnp.asarray(a00)
    a01 = jnp.asarray(a01)
    a10 = jnp.asarray(a10)
    a11 = jnp.asarray(a11)
    off0 = jnp.asarray(off0)
    off1 = jnp.asarray(off1)
    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)

    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )
    batch_size, n = rhs0.shape
    if n < 2:
        raise ValueError("pallas_thomas requires Nx >= 2.")
    if batch_size % int(block_b) != 0:
        raise ValueError(
            f"batch_size must be divisible by block_b={int(block_b)}, got {batch_size}."
        )

    def as_batched(values: Array, *, length: int, name: str) -> Array:
        arr = jnp.asarray(values)
        if arr.shape[-1] != length:
            raise ValueError(f"{name} must have trailing length {length}, got {arr.shape}.")
        if arr.ndim == 1:
            return jnp.broadcast_to(arr[None, :], (batch_size, length))
        if arr.ndim == 2 and arr.shape[0] == batch_size:
            return arr
        raise ValueError(
            f"{name} must have shape ({length},) or ({batch_size}, {length}), got {arr.shape}."
        )

    a00_b = as_batched(a00, length=n, name="a00")
    a01_b = as_batched(a01, length=n, name="a01")
    a10_b = as_batched(a10, length=n, name="a10")
    a11_b = as_batched(a11, length=n, name="a11")
    off0_b = as_batched(off0, length=n - 1, name="off0")
    off1_b = as_batched(off1, length=n - 1, name="off1")

    if interpret is None:
        interpret = jax.default_backend() != "gpu"

    block_b = int(block_b)
    in_specs = (
        _block_spec_2d(block_b, n),
        _block_spec_2d(block_b, n),
        _block_spec_2d(block_b, n),
        _block_spec_2d(block_b, n),
        _block_spec_2d(block_b, n - 1),
        _block_spec_2d(block_b, n - 1),
        _block_spec_2d(block_b, n),
        _block_spec_2d(block_b, n),
    )
    out_spec = pl.BlockSpec((block_b, n, 2), lambda block_id: (block_id, 0, 0))
    scratch = _memory_ref((block_b, n, 6), rhs0.dtype)
    solve = pl.pallas_call(
        functools.partial(_pallas_thomas_2x2_kernel, n=n),
        out_shape=jax.ShapeDtypeStruct((batch_size, n, 2), rhs0.dtype),
        grid=(batch_size // block_b,),
        in_specs=in_specs,
        out_specs=out_spec,
        scratch_shapes=(scratch,),
        compiler_params=plgpu.TritonCompilerParams(num_warps=int(num_warps)),
        interpret=bool(interpret),
        name=f"double_cable_pallas_thomas_b{block_b}",
    )
    out = solve(a00_b, a01_b, a10_b, a11_b, off0_b, off1_b, rhs0, rhs1)
    return out[..., 0], out[..., 1]


def _block_spec_2d(block_b: int, n: int) -> pl.BlockSpec:
    return pl.BlockSpec((block_b, n), lambda block_id: (block_id, 0))


def _memory_ref(shape: tuple[int, ...], dtype: jnp.dtype):
    memory_ref = getattr(pl, "MemoryRef", None)
    memory_space = getattr(pl, "MemorySpace", None)
    if memory_ref is None or memory_space is None:
        from jax._src.pallas import core as pallas_core

        memory_ref = pallas_core.MemoryRef
        memory_space = pallas_core.MemorySpace
    return memory_ref(shape, dtype, memory_space.ANY)


def _pallas_thomas_2x2_kernel(
    a00_ref,
    a01_ref,
    a10_ref,
    a11_ref,
    off0_ref,
    off1_ref,
    rhs0_ref,
    rhs1_ref,
    out_ref,
    scratch_ref,
    *,
    n: int,
) -> None:
    zero = jnp.zeros((a00_ref.shape[0],), dtype=a00_ref.dtype)

    def inv_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    inv00, inv01, inv10, inv11 = inv_components(
        a00_ref[:, 0],
        a01_ref[:, 0],
        a10_ref[:, 0],
        a11_ref[:, 0],
    )
    c00 = inv00 * off0_ref[:, 0]
    c01 = inv01 * off1_ref[:, 0]
    c10 = inv10 * off0_ref[:, 0]
    c11 = inv11 * off1_ref[:, 0]
    d0 = inv00 * rhs0_ref[:, 0] + inv01 * rhs1_ref[:, 0]
    d1 = inv10 * rhs0_ref[:, 0] + inv11 * rhs1_ref[:, 0]
    _store_forward_row(scratch_ref, 0, c00, c01, c10, c11, d0, d1)

    def forward_body(i: int, carry: tuple[Array, Array, Array, Array, Array, Array]):
        c00_prev, c01_prev, c10_prev, c11_prev, d0_prev, d1_prev = carry
        lower0 = off0_ref[:, i - 1]
        lower1 = off1_ref[:, i - 1]
        upper0 = off0_ref[:, i]
        upper1 = off1_ref[:, i]

        m00 = a00_ref[:, i] - lower0 * c00_prev
        m01 = a01_ref[:, i] - lower0 * c01_prev
        m10 = a10_ref[:, i] - lower1 * c10_prev
        m11 = a11_ref[:, i] - lower1 * c11_prev
        inv00_i, inv01_i, inv10_i, inv11_i = inv_components(m00, m01, m10, m11)

        r0 = rhs0_ref[:, i] - lower0 * d0_prev
        r1 = rhs1_ref[:, i] - lower1 * d1_prev
        c00_i = inv00_i * upper0
        c01_i = inv01_i * upper1
        c10_i = inv10_i * upper0
        c11_i = inv11_i * upper1
        d0_i = inv00_i * r0 + inv01_i * r1
        d1_i = inv10_i * r0 + inv11_i * r1
        _store_forward_row(scratch_ref, i, c00_i, c01_i, c10_i, c11_i, d0_i, d1_i)
        return c00_i, c01_i, c10_i, c11_i, d0_i, d1_i

    c00, c01, c10, c11, d0, d1 = jax.lax.fori_loop(
        1,
        n - 1,
        forward_body,
        (c00, c01, c10, c11, d0, d1),
    )

    i = n - 1
    lower0 = off0_ref[:, i - 1]
    lower1 = off1_ref[:, i - 1]
    m00 = a00_ref[:, i] - lower0 * c00
    m01 = a01_ref[:, i] - lower0 * c01
    m10 = a10_ref[:, i] - lower1 * c10
    m11 = a11_ref[:, i] - lower1 * c11
    inv00, inv01, inv10, inv11 = inv_components(m00, m01, m10, m11)
    r0 = rhs0_ref[:, i] - lower0 * d0
    r1 = rhs1_ref[:, i] - lower1 * d1
    d0 = inv00 * r0 + inv01 * r1
    d1 = inv10 * r0 + inv11 * r1
    _store_forward_row(scratch_ref, i, zero, zero, zero, zero, d0, d1)
    out_ref[:, i, 0] = d0
    out_ref[:, i, 1] = d1

    def backward_body(k: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        next0, next1 = carry
        row = (n - 2) - k
        c00_i = scratch_ref[:, row, 0]
        c01_i = scratch_ref[:, row, 1]
        c10_i = scratch_ref[:, row, 2]
        c11_i = scratch_ref[:, row, 3]
        d0_i = scratch_ref[:, row, 4]
        d1_i = scratch_ref[:, row, 5]
        x0 = d0_i - c00_i * next0 - c01_i * next1
        x1 = d1_i - c10_i * next0 - c11_i * next1
        out_ref[:, row, 0] = x0
        out_ref[:, row, 1] = x1
        return x0, x1

    jax.lax.fori_loop(0, n - 1, backward_body, (d0, d1))


def _store_forward_row(
    scratch_ref,
    row: int,
    c00: Array,
    c01: Array,
    c10: Array,
    c11: Array,
    d0: Array,
    d1: Array,
) -> None:
    scratch_ref[:, row, 0] = c00
    scratch_ref[:, row, 1] = c01
    scratch_ref[:, row, 2] = c10
    scratch_ref[:, row, 3] = c11
    scratch_ref[:, row, 4] = d0
    scratch_ref[:, row, 5] = d1
