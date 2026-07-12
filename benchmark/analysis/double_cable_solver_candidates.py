from __future__ import annotations

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp

from axonscope.runtime.jax.common import solve_block_tridiagonal_2x2_pcr_soa_batched

Array: TypeAlias = Any


def solve_block_tridiagonal_2x2_pcr_soa_batched_ref(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Batch-native SoA PCR solve using internal JAX refs for work arrays.

    This benchmark-only candidate keeps the same exact algebra as
    ``solve_block_tridiagonal_2x2_pcr_soa_batched`` but stores the stage-local
    PCR work arrays in ``jax.new_ref`` buffers. The goal is to test whether
    JAX/XLA can shorten live ranges or reuse buffers more effectively on GPU.
    """

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )

    batch_size, n = rhs0.shape
    dtype = rhs0.dtype
    idx = jnp.arange(n)
    zero = jnp.zeros((), dtype=dtype)

    def broadcast_space(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length Nx={n}, got {arr.shape}.")
            return jnp.broadcast_to(arr[None, :], (batch_size, n))
        if arr.ndim == 2:
            if arr.shape != (batch_size, n):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {n}), got {arr.shape}."
                )
            return arr
        raise ValueError(
            f"{name} must have shape (Nx,) or (batch_size, Nx), got {arr.shape}."
        )

    def broadcast_edges(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        edge_shape = (batch_size, max(n - 1, 0))
        if arr.ndim == 1:
            if arr.shape[0] != edge_shape[1]:
                raise ValueError(
                    f"{name} must have length Nx - 1={edge_shape[1]}, got {arr.shape}."
                )
            return jnp.broadcast_to(arr[None, :], edge_shape)
        if arr.ndim == 2:
            if arr.shape != edge_shape:
                raise ValueError(f"{name} must have shape {edge_shape}, got {arr.shape}.")
            return arr
        raise ValueError(
            f"{name} must have shape (Nx - 1,) or (batch_size, Nx - 1), got {arr.shape}."
        )

    diag00_ref = jax.new_ref(broadcast_space("a00", a00))
    diag01_ref = jax.new_ref(broadcast_space("a01", a01))
    diag10_ref = jax.new_ref(broadcast_space("a10", a10))
    diag11_ref = jax.new_ref(broadcast_space("a11", a11))
    off0_batched = broadcast_edges("off0", off0)
    off1_batched = broadcast_edges("off1", off1)

    zero_col = jnp.zeros((batch_size, 1), dtype=dtype)
    zeros = jnp.zeros((batch_size, n), dtype=dtype)
    lower00_ref = jax.new_ref(jnp.concatenate([zero_col, off0_batched], axis=1))
    lower01_ref = jax.new_ref(zeros)
    lower10_ref = jax.new_ref(zeros)
    lower11_ref = jax.new_ref(jnp.concatenate([zero_col, off1_batched], axis=1))
    upper00_ref = jax.new_ref(jnp.concatenate([off0_batched, zero_col], axis=1))
    upper01_ref = jax.new_ref(zeros)
    upper10_ref = jax.new_ref(zeros)
    upper11_ref = jax.new_ref(jnp.concatenate([off1_batched, zero_col], axis=1))
    r0_ref = jax.new_ref(rhs0)
    r1_ref = jax.new_ref(rhs1)

    def inv2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    def matmul2_components(
        l00: Array,
        l01: Array,
        l10: Array,
        l11: Array,
        r00: Array,
        r01: Array,
        r10: Array,
        r11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        return (
            l00 * r00 + l01 * r10,
            l00 * r01 + l01 * r11,
            l10 * r00 + l11 * r10,
            l10 * r01 + l11 * r11,
        )

    def matvec2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
        v0: Array,
        v1: Array,
    ) -> tuple[Array, Array]:
        return m00 * v0 + m01 * v1, m10 * v0 + m11 * v1

    stride = 1
    while stride < n:
        left_idx = jnp.maximum(idx - stride, 0)
        right_idx = jnp.minimum(idx + stride, n - 1)
        has_left = (idx >= stride)[None, :]
        has_right = (idx + stride < n)[None, :]

        diag00 = diag00_ref[...]
        diag01 = diag01_ref[...]
        diag10 = diag10_ref[...]
        diag11 = diag11_ref[...]
        lower00 = lower00_ref[...]
        lower01 = lower01_ref[...]
        lower10 = lower10_ref[...]
        lower11 = lower11_ref[...]
        upper00 = upper00_ref[...]
        upper01 = upper01_ref[...]
        upper10 = upper10_ref[...]
        upper11 = upper11_ref[...]
        r0 = r0_ref[...]
        r1 = r1_ref[...]

        left_inv = inv2_components(
            diag00[:, left_idx],
            diag01[:, left_idx],
            diag10[:, left_idx],
            diag11[:, left_idx],
        )
        right_inv = inv2_components(
            diag00[:, right_idx],
            diag01[:, right_idx],
            diag10[:, right_idx],
            diag11[:, right_idx],
        )
        lf00, lf01, lf10, lf11 = matmul2_components(
            lower00,
            lower01,
            lower10,
            lower11,
            *left_inv,
        )
        rf00, rf01, rf10, rf11 = matmul2_components(
            upper00,
            upper01,
            upper10,
            upper11,
            *right_inv,
        )
        lf00 = jnp.where(has_left, lf00, zero)
        lf01 = jnp.where(has_left, lf01, zero)
        lf10 = jnp.where(has_left, lf10, zero)
        lf11 = jnp.where(has_left, lf11, zero)
        rf00 = jnp.where(has_right, rf00, zero)
        rf01 = jnp.where(has_right, rf01, zero)
        rf10 = jnp.where(has_right, rf10, zero)
        rf11 = jnp.where(has_right, rf11, zero)

        nl00, nl01, nl10, nl11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            lower00[:, left_idx],
            lower01[:, left_idx],
            lower10[:, left_idx],
            lower11[:, left_idx],
        )
        nu00, nu01, nu10, nu11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            upper00[:, right_idx],
            upper01[:, right_idx],
            upper10[:, right_idx],
            upper11[:, right_idx],
        )
        ldu00, ldu01, ldu10, ldu11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            upper00[:, left_idx],
            upper01[:, left_idx],
            upper10[:, left_idx],
            upper11[:, left_idx],
        )
        rdl00, rdl01, rdl10, rdl11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            lower00[:, right_idx],
            lower01[:, right_idx],
            lower10[:, right_idx],
            lower11[:, right_idx],
        )
        lr0, lr1 = matvec2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            r0[:, left_idx],
            r1[:, left_idx],
        )
        rr0, rr1 = matvec2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            r0[:, right_idx],
            r1[:, right_idx],
        )

        lower00_ref[...] = jnp.where(has_left, -nl00, zero)
        lower01_ref[...] = jnp.where(has_left, -nl01, zero)
        lower10_ref[...] = jnp.where(has_left, -nl10, zero)
        lower11_ref[...] = jnp.where(has_left, -nl11, zero)
        upper00_ref[...] = jnp.where(has_right, -nu00, zero)
        upper01_ref[...] = jnp.where(has_right, -nu01, zero)
        upper10_ref[...] = jnp.where(has_right, -nu10, zero)
        upper11_ref[...] = jnp.where(has_right, -nu11, zero)
        diag00_ref[...] = diag00 - ldu00 - rdl00
        diag01_ref[...] = diag01 - ldu01 - rdl01
        diag10_ref[...] = diag10 - ldu10 - rdl10
        diag11_ref[...] = diag11 - ldu11 - rdl11
        r0_ref[...] = r0 - lr0 - rr0
        r1_ref[...] = r1 - lr1 - rr1
        stride *= 2

    diag00 = jax.freeze(diag00_ref)
    diag01 = jax.freeze(diag01_ref)
    diag10 = jax.freeze(diag10_ref)
    diag11 = jax.freeze(diag11_ref)
    r0 = jax.freeze(r0_ref)
    r1 = jax.freeze(r1_ref)
    inv00, inv01, inv10, inv11 = inv2_components(diag00, diag01, diag10, diag11)
    return matvec2_components(inv00, inv01, inv10, inv11, r0, r1)


def solve_block_tridiagonal_2x2_pcr_soa_batched_nomask(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Batch-native SoA PCR solve without explicit boundary selects.

    This benchmark-only candidate relies on the PCR zero-coupling invariant:
    at the start of a stage with stride ``s``, lower couplings are already zero
    for columns ``< s`` and upper couplings are already zero for columns
    ``>= Nx - s``. With clamped neighbor indices, invalid left/right updates
    therefore multiply by a zero side factor and the explicit
    ``where(has_left/has_right, ..., 0)`` masks are algebraically redundant.
    """

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )

    batch_size, n = rhs0.shape
    dtype = rhs0.dtype
    idx = jnp.arange(n)

    def broadcast_space(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length Nx={n}, got {arr.shape}.")
            return jnp.broadcast_to(arr[None, :], (batch_size, n))
        if arr.ndim == 2:
            if arr.shape != (batch_size, n):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {n}), got {arr.shape}."
                )
            return arr
        raise ValueError(
            f"{name} must have shape (Nx,) or (batch_size, Nx), got {arr.shape}."
        )

    def broadcast_edges(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        edge_shape = (batch_size, max(n - 1, 0))
        if arr.ndim == 1:
            if arr.shape[0] != edge_shape[1]:
                raise ValueError(
                    f"{name} must have length Nx - 1={edge_shape[1]}, got {arr.shape}."
                )
            return jnp.broadcast_to(arr[None, :], edge_shape)
        if arr.ndim == 2:
            if arr.shape != edge_shape:
                raise ValueError(f"{name} must have shape {edge_shape}, got {arr.shape}.")
            return arr
        raise ValueError(
            f"{name} must have shape (Nx - 1,) or (batch_size, Nx - 1), got {arr.shape}."
        )

    diag00 = broadcast_space("a00", a00)
    diag01 = broadcast_space("a01", a01)
    diag10 = broadcast_space("a10", a10)
    diag11 = broadcast_space("a11", a11)
    off0_batched = broadcast_edges("off0", off0)
    off1_batched = broadcast_edges("off1", off1)

    zero_col = jnp.zeros((batch_size, 1), dtype=dtype)
    zeros = jnp.zeros((batch_size, n), dtype=dtype)
    lower00 = jnp.concatenate([zero_col, off0_batched], axis=1)
    lower01 = zeros
    lower10 = zeros
    lower11 = jnp.concatenate([zero_col, off1_batched], axis=1)
    upper00 = jnp.concatenate([off0_batched, zero_col], axis=1)
    upper01 = zeros
    upper10 = zeros
    upper11 = jnp.concatenate([off1_batched, zero_col], axis=1)
    r0 = rhs0
    r1 = rhs1

    def inv2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    def matmul2_components(
        l00: Array,
        l01: Array,
        l10: Array,
        l11: Array,
        r00: Array,
        r01: Array,
        r10: Array,
        r11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        return (
            l00 * r00 + l01 * r10,
            l00 * r01 + l01 * r11,
            l10 * r00 + l11 * r10,
            l10 * r01 + l11 * r11,
        )

    def matvec2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
        v0: Array,
        v1: Array,
    ) -> tuple[Array, Array]:
        return m00 * v0 + m01 * v1, m10 * v0 + m11 * v1

    stride = 1
    while stride < n:
        left_idx = jnp.maximum(idx - stride, 0)
        right_idx = jnp.minimum(idx + stride, n - 1)

        left_inv = inv2_components(
            diag00[:, left_idx],
            diag01[:, left_idx],
            diag10[:, left_idx],
            diag11[:, left_idx],
        )
        right_inv = inv2_components(
            diag00[:, right_idx],
            diag01[:, right_idx],
            diag10[:, right_idx],
            diag11[:, right_idx],
        )
        lf00, lf01, lf10, lf11 = matmul2_components(
            lower00,
            lower01,
            lower10,
            lower11,
            *left_inv,
        )
        rf00, rf01, rf10, rf11 = matmul2_components(
            upper00,
            upper01,
            upper10,
            upper11,
            *right_inv,
        )

        nl00, nl01, nl10, nl11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            lower00[:, left_idx],
            lower01[:, left_idx],
            lower10[:, left_idx],
            lower11[:, left_idx],
        )
        nu00, nu01, nu10, nu11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            upper00[:, right_idx],
            upper01[:, right_idx],
            upper10[:, right_idx],
            upper11[:, right_idx],
        )
        ldu00, ldu01, ldu10, ldu11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            upper00[:, left_idx],
            upper01[:, left_idx],
            upper10[:, left_idx],
            upper11[:, left_idx],
        )
        rdl00, rdl01, rdl10, rdl11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            lower00[:, right_idx],
            lower01[:, right_idx],
            lower10[:, right_idx],
            lower11[:, right_idx],
        )
        lr0, lr1 = matvec2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            r0[:, left_idx],
            r1[:, left_idx],
        )
        rr0, rr1 = matvec2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            r0[:, right_idx],
            r1[:, right_idx],
        )

        lower00 = -nl00
        lower01 = -nl01
        lower10 = -nl10
        lower11 = -nl11
        upper00 = -nu00
        upper01 = -nu01
        upper10 = -nu10
        upper11 = -nu11
        diag00 = diag00 - ldu00 - rdl00
        diag01 = diag01 - ldu01 - rdl01
        diag10 = diag10 - ldu10 - rdl10
        diag11 = diag11 - ldu11 - rdl11
        r0 = r0 - lr0 - rr0
        r1 = r1 - lr1 - rr1
        stride *= 2

    inv00, inv01, inv10, inv11 = inv2_components(diag00, diag01, diag10, diag11)
    return matvec2_components(inv00, inv01, inv10, inv11, r0, r1)


def solve_block_tridiagonal_2x2_pcr_soa_batched_shift(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Batch-native SoA PCR solve using static slice/concat neighbor shifts.

    This benchmark-only candidate targets the same stage body as
    ``pcr_soa_nomask`` but also removes the per-stage clamped gather indices.
    Neighbor arrays are shifted by the static PCR stride with identity diagonal
    fills and zero off-diagonal/RHS fills at invalid boundaries.
    """

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )

    batch_size, n = rhs0.shape
    dtype = rhs0.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)

    def broadcast_space(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length Nx={n}, got {arr.shape}.")
            return jnp.broadcast_to(arr[None, :], (batch_size, n))
        if arr.ndim == 2:
            if arr.shape != (batch_size, n):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {n}), got {arr.shape}."
                )
            return arr
        raise ValueError(
            f"{name} must have shape (Nx,) or (batch_size, Nx), got {arr.shape}."
        )

    def broadcast_edges(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        edge_shape = (batch_size, max(n - 1, 0))
        if arr.ndim == 1:
            if arr.shape[0] != edge_shape[1]:
                raise ValueError(
                    f"{name} must have length Nx - 1={edge_shape[1]}, got {arr.shape}."
                )
            return jnp.broadcast_to(arr[None, :], edge_shape)
        if arr.ndim == 2:
            if arr.shape != edge_shape:
                raise ValueError(f"{name} must have shape {edge_shape}, got {arr.shape}.")
            return arr
        raise ValueError(
            f"{name} must have shape (Nx - 1,) or (batch_size, Nx - 1), got {arr.shape}."
        )

    diag00 = broadcast_space("a00", a00)
    diag01 = broadcast_space("a01", a01)
    diag10 = broadcast_space("a10", a10)
    diag11 = broadcast_space("a11", a11)
    off0_batched = broadcast_edges("off0", off0)
    off1_batched = broadcast_edges("off1", off1)

    zero_col = jnp.zeros((batch_size, 1), dtype=dtype)
    zeros = jnp.zeros((batch_size, n), dtype=dtype)
    lower00 = jnp.concatenate([zero_col, off0_batched], axis=1)
    lower01 = zeros
    lower10 = zeros
    lower11 = jnp.concatenate([zero_col, off1_batched], axis=1)
    upper00 = jnp.concatenate([off0_batched, zero_col], axis=1)
    upper01 = zeros
    upper10 = zeros
    upper11 = jnp.concatenate([off1_batched, zero_col], axis=1)
    r0 = rhs0
    r1 = rhs1

    def inv2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    def matmul2_components(
        l00: Array,
        l01: Array,
        l10: Array,
        l11: Array,
        r00: Array,
        r01: Array,
        r10: Array,
        r11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        return (
            l00 * r00 + l01 * r10,
            l00 * r01 + l01 * r11,
            l10 * r00 + l11 * r10,
            l10 * r01 + l11 * r11,
        )

    def matvec2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
        v0: Array,
        v1: Array,
    ) -> tuple[Array, Array]:
        return m00 * v0 + m01 * v1, m10 * v0 + m11 * v1

    def fill_columns(value: Array, width: int) -> Array:
        return jnp.full((batch_size, width), value, dtype=dtype)

    def shift_right(values: Array, fill: Array, width: int) -> Array:
        return jnp.concatenate([fill_columns(fill, width), values[:, :-width]], axis=1)

    def shift_left(values: Array, fill: Array, width: int) -> Array:
        return jnp.concatenate([values[:, width:], fill_columns(fill, width)], axis=1)

    stride = 1
    while stride < n:
        left_diag00 = shift_right(diag00, one, stride)
        left_diag01 = shift_right(diag01, zero, stride)
        left_diag10 = shift_right(diag10, zero, stride)
        left_diag11 = shift_right(diag11, one, stride)
        right_diag00 = shift_left(diag00, one, stride)
        right_diag01 = shift_left(diag01, zero, stride)
        right_diag10 = shift_left(diag10, zero, stride)
        right_diag11 = shift_left(diag11, one, stride)

        left_inv = inv2_components(
            left_diag00,
            left_diag01,
            left_diag10,
            left_diag11,
        )
        right_inv = inv2_components(
            right_diag00,
            right_diag01,
            right_diag10,
            right_diag11,
        )
        lf00, lf01, lf10, lf11 = matmul2_components(
            lower00,
            lower01,
            lower10,
            lower11,
            *left_inv,
        )
        rf00, rf01, rf10, rf11 = matmul2_components(
            upper00,
            upper01,
            upper10,
            upper11,
            *right_inv,
        )

        left_lower00 = shift_right(lower00, zero, stride)
        left_lower01 = shift_right(lower01, zero, stride)
        left_lower10 = shift_right(lower10, zero, stride)
        left_lower11 = shift_right(lower11, zero, stride)
        left_upper00 = shift_right(upper00, zero, stride)
        left_upper01 = shift_right(upper01, zero, stride)
        left_upper10 = shift_right(upper10, zero, stride)
        left_upper11 = shift_right(upper11, zero, stride)
        left_r0 = shift_right(r0, zero, stride)
        left_r1 = shift_right(r1, zero, stride)

        right_lower00 = shift_left(lower00, zero, stride)
        right_lower01 = shift_left(lower01, zero, stride)
        right_lower10 = shift_left(lower10, zero, stride)
        right_lower11 = shift_left(lower11, zero, stride)
        right_upper00 = shift_left(upper00, zero, stride)
        right_upper01 = shift_left(upper01, zero, stride)
        right_upper10 = shift_left(upper10, zero, stride)
        right_upper11 = shift_left(upper11, zero, stride)
        right_r0 = shift_left(r0, zero, stride)
        right_r1 = shift_left(r1, zero, stride)

        nl00, nl01, nl10, nl11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            left_lower00,
            left_lower01,
            left_lower10,
            left_lower11,
        )
        nu00, nu01, nu10, nu11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            right_upper00,
            right_upper01,
            right_upper10,
            right_upper11,
        )
        ldu00, ldu01, ldu10, ldu11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            left_upper00,
            left_upper01,
            left_upper10,
            left_upper11,
        )
        rdl00, rdl01, rdl10, rdl11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            right_lower00,
            right_lower01,
            right_lower10,
            right_lower11,
        )
        lr0, lr1 = matvec2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            left_r0,
            left_r1,
        )
        rr0, rr1 = matvec2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            right_r0,
            right_r1,
        )

        lower00 = -nl00
        lower01 = -nl01
        lower10 = -nl10
        lower11 = -nl11
        upper00 = -nu00
        upper01 = -nu01
        upper10 = -nu10
        upper11 = -nu11
        diag00 = diag00 - ldu00 - rdl00
        diag01 = diag01 - ldu01 - rdl01
        diag10 = diag10 - ldu10 - rdl10
        diag11 = diag11 - ldu11 - rdl11
        r0 = r0 - lr0 - rr0
        r1 = r1 - lr1 - rr1
        stride *= 2

    inv00, inv01, inv10, inv11 = inv2_components(diag00, diag01, diag10, diag11)
    return matvec2_components(inv00, inv01, inv10, inv11, r0, r1)


def solve_block_tridiagonal_2x2_pcr_soa_batched_transposed(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Batch-native SoA PCR solve using an internal ``[Nx, B]`` layout."""

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )

    batch_size, n = rhs0.shape
    dtype = rhs0.dtype
    idx = jnp.arange(n)
    zero = jnp.zeros((), dtype=dtype)

    def transpose_space(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length Nx={n}, got {arr.shape}.")
            return jnp.broadcast_to(arr[:, None], (n, batch_size))
        if arr.ndim == 2:
            if arr.shape != (batch_size, n):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {n}), got {arr.shape}."
                )
            return jnp.swapaxes(arr, 0, 1)
        raise ValueError(
            f"{name} must have shape (Nx,) or (batch_size, Nx), got {arr.shape}."
        )

    def transpose_edges(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        edge_shape = (batch_size, max(n - 1, 0))
        if arr.ndim == 1:
            if arr.shape[0] != edge_shape[1]:
                raise ValueError(
                    f"{name} must have length Nx - 1={edge_shape[1]}, got {arr.shape}."
                )
            return jnp.broadcast_to(arr[:, None], (edge_shape[1], batch_size))
        if arr.ndim == 2:
            if arr.shape != edge_shape:
                raise ValueError(f"{name} must have shape {edge_shape}, got {arr.shape}.")
            return jnp.swapaxes(arr, 0, 1)
        raise ValueError(
            f"{name} must have shape (Nx - 1,) or (batch_size, Nx - 1), got {arr.shape}."
        )

    diag00 = transpose_space("a00", a00)
    diag01 = transpose_space("a01", a01)
    diag10 = transpose_space("a10", a10)
    diag11 = transpose_space("a11", a11)
    off0_transposed = transpose_edges("off0", off0)
    off1_transposed = transpose_edges("off1", off1)

    zero_row = jnp.zeros((1, batch_size), dtype=dtype)
    zeros = jnp.zeros((n, batch_size), dtype=dtype)
    lower00 = jnp.concatenate([zero_row, off0_transposed], axis=0)
    lower01 = zeros
    lower10 = zeros
    lower11 = jnp.concatenate([zero_row, off1_transposed], axis=0)
    upper00 = jnp.concatenate([off0_transposed, zero_row], axis=0)
    upper01 = zeros
    upper10 = zeros
    upper11 = jnp.concatenate([off1_transposed, zero_row], axis=0)
    r0 = jnp.swapaxes(rhs0, 0, 1)
    r1 = jnp.swapaxes(rhs1, 0, 1)

    def inv2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    def matmul2_components(
        l00: Array,
        l01: Array,
        l10: Array,
        l11: Array,
        r00: Array,
        r01: Array,
        r10: Array,
        r11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        return (
            l00 * r00 + l01 * r10,
            l00 * r01 + l01 * r11,
            l10 * r00 + l11 * r10,
            l10 * r01 + l11 * r11,
        )

    def matvec2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
        v0: Array,
        v1: Array,
    ) -> tuple[Array, Array]:
        return m00 * v0 + m01 * v1, m10 * v0 + m11 * v1

    stride = 1
    while stride < n:
        left_idx = jnp.maximum(idx - stride, 0)
        right_idx = jnp.minimum(idx + stride, n - 1)
        has_left = (idx >= stride)[:, None]
        has_right = (idx + stride < n)[:, None]

        left_inv = inv2_components(
            diag00[left_idx, :],
            diag01[left_idx, :],
            diag10[left_idx, :],
            diag11[left_idx, :],
        )
        right_inv = inv2_components(
            diag00[right_idx, :],
            diag01[right_idx, :],
            diag10[right_idx, :],
            diag11[right_idx, :],
        )
        lf00, lf01, lf10, lf11 = matmul2_components(
            lower00,
            lower01,
            lower10,
            lower11,
            *left_inv,
        )
        rf00, rf01, rf10, rf11 = matmul2_components(
            upper00,
            upper01,
            upper10,
            upper11,
            *right_inv,
        )
        lf00 = jnp.where(has_left, lf00, zero)
        lf01 = jnp.where(has_left, lf01, zero)
        lf10 = jnp.where(has_left, lf10, zero)
        lf11 = jnp.where(has_left, lf11, zero)
        rf00 = jnp.where(has_right, rf00, zero)
        rf01 = jnp.where(has_right, rf01, zero)
        rf10 = jnp.where(has_right, rf10, zero)
        rf11 = jnp.where(has_right, rf11, zero)

        nl00, nl01, nl10, nl11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            lower00[left_idx, :],
            lower01[left_idx, :],
            lower10[left_idx, :],
            lower11[left_idx, :],
        )
        nu00, nu01, nu10, nu11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            upper00[right_idx, :],
            upper01[right_idx, :],
            upper10[right_idx, :],
            upper11[right_idx, :],
        )
        ldu00, ldu01, ldu10, ldu11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            upper00[left_idx, :],
            upper01[left_idx, :],
            upper10[left_idx, :],
            upper11[left_idx, :],
        )
        rdl00, rdl01, rdl10, rdl11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            lower00[right_idx, :],
            lower01[right_idx, :],
            lower10[right_idx, :],
            lower11[right_idx, :],
        )
        lr0, lr1 = matvec2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            r0[left_idx, :],
            r1[left_idx, :],
        )
        rr0, rr1 = matvec2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            r0[right_idx, :],
            r1[right_idx, :],
        )

        lower00 = jnp.where(has_left, -nl00, zero)
        lower01 = jnp.where(has_left, -nl01, zero)
        lower10 = jnp.where(has_left, -nl10, zero)
        lower11 = jnp.where(has_left, -nl11, zero)
        upper00 = jnp.where(has_right, -nu00, zero)
        upper01 = jnp.where(has_right, -nu01, zero)
        upper10 = jnp.where(has_right, -nu10, zero)
        upper11 = jnp.where(has_right, -nu11, zero)
        diag00 = diag00 - ldu00 - rdl00
        diag01 = diag01 - ldu01 - rdl01
        diag10 = diag10 - ldu10 - rdl10
        diag11 = diag11 - ldu11 - rdl11
        r0 = r0 - lr0 - rr0
        r1 = r1 - lr1 - rr1
        stride *= 2

    inv00, inv01, inv10, inv11 = inv2_components(diag00, diag01, diag10, diag11)
    x0, x1 = matvec2_components(inv00, inv01, inv10, inv11, r0, r1)
    return jnp.swapaxes(x0, 0, 1), jnp.swapaxes(x1, 0, 1)


def solve_block_tridiagonal_2x2_pcr_soa_hybrid_batched(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    chain_stride: int = 8,
) -> tuple[Array, Array]:
    """Exact hybrid PCR/Thomas solve for batched double-cable systems.

    PCR stages first eliminate neighbors until remaining couplings jump by
    ``chain_stride`` compartments. The residual system then splits into
    independent chains by ``i % chain_stride``; each chain is solved exactly
    with a batch-native 2x2 block-Thomas pass.
    """

    target_stride = int(chain_stride)
    if target_stride < 1 or target_stride & (target_stride - 1):
        raise ValueError("chain_stride must be a positive power of two.")

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )

    batch_size, n = rhs0.shape
    dtype = rhs0.dtype
    idx = jnp.arange(n)
    zero = jnp.zeros((), dtype=dtype)

    def broadcast_space(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length Nx={n}, got {arr.shape}.")
            return jnp.broadcast_to(arr[None, :], (batch_size, n))
        if arr.ndim == 2:
            if arr.shape != (batch_size, n):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {n}), got {arr.shape}."
                )
            return arr
        raise ValueError(
            f"{name} must have shape (Nx,) or (batch_size, Nx), got {arr.shape}."
        )

    def broadcast_edges(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        edge_shape = (batch_size, max(n - 1, 0))
        if arr.ndim == 1:
            if arr.shape[0] != edge_shape[1]:
                raise ValueError(
                    f"{name} must have length Nx - 1={edge_shape[1]}, got {arr.shape}."
                )
            return jnp.broadcast_to(arr[None, :], edge_shape)
        if arr.ndim == 2:
            if arr.shape != edge_shape:
                raise ValueError(f"{name} must have shape {edge_shape}, got {arr.shape}.")
            return arr
        raise ValueError(
            f"{name} must have shape (Nx - 1,) or (batch_size, Nx - 1), got {arr.shape}."
        )

    diag00 = broadcast_space("a00", a00)
    diag01 = broadcast_space("a01", a01)
    diag10 = broadcast_space("a10", a10)
    diag11 = broadcast_space("a11", a11)
    off0_batched = broadcast_edges("off0", off0)
    off1_batched = broadcast_edges("off1", off1)

    zero_col = jnp.zeros((batch_size, 1), dtype=dtype)
    zeros = jnp.zeros((batch_size, n), dtype=dtype)
    lower00 = jnp.concatenate([zero_col, off0_batched], axis=1)
    lower01 = zeros
    lower10 = zeros
    lower11 = jnp.concatenate([zero_col, off1_batched], axis=1)
    upper00 = jnp.concatenate([off0_batched, zero_col], axis=1)
    upper01 = zeros
    upper10 = zeros
    upper11 = jnp.concatenate([off1_batched, zero_col], axis=1)
    r0 = rhs0
    r1 = rhs1

    def inv2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    def matmul2_components(
        l00: Array,
        l01: Array,
        l10: Array,
        l11: Array,
        r00: Array,
        r01: Array,
        r10: Array,
        r11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        return (
            l00 * r00 + l01 * r10,
            l00 * r01 + l01 * r11,
            l10 * r00 + l11 * r10,
            l10 * r01 + l11 * r11,
        )

    def matvec2_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
        v0: Array,
        v1: Array,
    ) -> tuple[Array, Array]:
        return m00 * v0 + m01 * v1, m10 * v0 + m11 * v1

    stride = 1
    while stride < n and stride < target_stride:
        left_idx = jnp.maximum(idx - stride, 0)
        right_idx = jnp.minimum(idx + stride, n - 1)
        has_left = (idx >= stride)[None, :]
        has_right = (idx + stride < n)[None, :]

        left_inv = inv2_components(
            diag00[:, left_idx],
            diag01[:, left_idx],
            diag10[:, left_idx],
            diag11[:, left_idx],
        )
        right_inv = inv2_components(
            diag00[:, right_idx],
            diag01[:, right_idx],
            diag10[:, right_idx],
            diag11[:, right_idx],
        )
        lf00, lf01, lf10, lf11 = matmul2_components(
            lower00,
            lower01,
            lower10,
            lower11,
            *left_inv,
        )
        rf00, rf01, rf10, rf11 = matmul2_components(
            upper00,
            upper01,
            upper10,
            upper11,
            *right_inv,
        )
        lf00 = jnp.where(has_left, lf00, zero)
        lf01 = jnp.where(has_left, lf01, zero)
        lf10 = jnp.where(has_left, lf10, zero)
        lf11 = jnp.where(has_left, lf11, zero)
        rf00 = jnp.where(has_right, rf00, zero)
        rf01 = jnp.where(has_right, rf01, zero)
        rf10 = jnp.where(has_right, rf10, zero)
        rf11 = jnp.where(has_right, rf11, zero)

        nl00, nl01, nl10, nl11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            lower00[:, left_idx],
            lower01[:, left_idx],
            lower10[:, left_idx],
            lower11[:, left_idx],
        )
        nu00, nu01, nu10, nu11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            upper00[:, right_idx],
            upper01[:, right_idx],
            upper10[:, right_idx],
            upper11[:, right_idx],
        )
        ldu00, ldu01, ldu10, ldu11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            upper00[:, left_idx],
            upper01[:, left_idx],
            upper10[:, left_idx],
            upper11[:, left_idx],
        )
        rdl00, rdl01, rdl10, rdl11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            lower00[:, right_idx],
            lower01[:, right_idx],
            lower10[:, right_idx],
            lower11[:, right_idx],
        )
        lr0, lr1 = matvec2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            r0[:, left_idx],
            r1[:, left_idx],
        )
        rr0, rr1 = matvec2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            r0[:, right_idx],
            r1[:, right_idx],
        )

        lower00 = jnp.where(has_left, -nl00, zero)
        lower01 = jnp.where(has_left, -nl01, zero)
        lower10 = jnp.where(has_left, -nl10, zero)
        lower11 = jnp.where(has_left, -nl11, zero)
        upper00 = jnp.where(has_right, -nu00, zero)
        upper01 = jnp.where(has_right, -nu01, zero)
        upper10 = jnp.where(has_right, -nu10, zero)
        upper11 = jnp.where(has_right, -nu11, zero)
        diag00 = diag00 - ldu00 - rdl00
        diag01 = diag01 - ldu01 - rdl01
        diag10 = diag10 - ldu10 - rdl10
        diag11 = diag11 - ldu11 - rdl11
        r0 = r0 - lr0 - rr0
        r1 = r1 - lr1 - rr1
        stride *= 2

    def solve_chain(
        c_diag00: Array,
        c_diag01: Array,
        c_diag10: Array,
        c_diag11: Array,
        c_lower00: Array,
        c_lower01: Array,
        c_lower10: Array,
        c_lower11: Array,
        c_upper00: Array,
        c_upper01: Array,
        c_upper10: Array,
        c_upper11: Array,
        c_rhs0: Array,
        c_rhs1: Array,
    ) -> tuple[Array, Array]:
        inv00, inv01, inv10, inv11 = inv2_components(
            c_diag00[:, 0],
            c_diag01[:, 0],
            c_diag10[:, 0],
            c_diag11[:, 0],
        )
        c00_0, c01_0, c10_0, c11_0 = matmul2_components(
            inv00,
            inv01,
            inv10,
            inv11,
            c_upper00[:, 0],
            c_upper01[:, 0],
            c_upper10[:, 0],
            c_upper11[:, 0],
        )
        d0_0, d1_0 = matvec2_components(
            inv00,
            inv01,
            inv10,
            inv11,
            c_rhs0[:, 0],
            c_rhs1[:, 0],
        )

        def fwd(carry, xs):
            c00_prev, c01_prev, c10_prev, c11_prev, d0_prev, d1_prev = carry
            (
                d00_i,
                d01_i,
                d10_i,
                d11_i,
                l00_i,
                l01_i,
                l10_i,
                l11_i,
                u00_i,
                u01_i,
                u10_i,
                u11_i,
                rhs0_i,
                rhs1_i,
            ) = xs
            lc00, lc01, lc10, lc11 = matmul2_components(
                l00_i,
                l01_i,
                l10_i,
                l11_i,
                c00_prev,
                c01_prev,
                c10_prev,
                c11_prev,
            )
            m00 = d00_i - lc00
            m01 = d01_i - lc01
            m10 = d10_i - lc10
            m11 = d11_i - lc11
            inv00_i, inv01_i, inv10_i, inv11_i = inv2_components(m00, m01, m10, m11)

            ld0, ld1 = matvec2_components(
                l00_i,
                l01_i,
                l10_i,
                l11_i,
                d0_prev,
                d1_prev,
            )
            r0_i = rhs0_i - ld0
            r1_i = rhs1_i - ld1
            c00_i, c01_i, c10_i, c11_i = matmul2_components(
                inv00_i,
                inv01_i,
                inv10_i,
                inv11_i,
                u00_i,
                u01_i,
                u10_i,
                u11_i,
            )
            d0_i, d1_i = matvec2_components(
                inv00_i,
                inv01_i,
                inv10_i,
                inv11_i,
                r0_i,
                r1_i,
            )
            out = (c00_i, c01_i, c10_i, c11_i, d0_i, d1_i)
            return out, out

        _, forward_tail = jax.lax.scan(
            fwd,
            (c00_0, c01_0, c10_0, c11_0, d0_0, d1_0),
            (
                c_diag00[:, 1:].T,
                c_diag01[:, 1:].T,
                c_diag10[:, 1:].T,
                c_diag11[:, 1:].T,
                c_lower00[:, 1:].T,
                c_lower01[:, 1:].T,
                c_lower10[:, 1:].T,
                c_lower11[:, 1:].T,
                c_upper00[:, 1:].T,
                c_upper01[:, 1:].T,
                c_upper10[:, 1:].T,
                c_upper11[:, 1:].T,
                c_rhs0[:, 1:].T,
                c_rhs1[:, 1:].T,
            ),
        )
        c00_tail, c01_tail, c10_tail, c11_tail, d0_tail, d1_tail = forward_tail
        c00 = jnp.concatenate([c00_0[None, :], c00_tail], axis=0)
        c01 = jnp.concatenate([c01_0[None, :], c01_tail], axis=0)
        c10 = jnp.concatenate([c10_0[None, :], c10_tail], axis=0)
        c11 = jnp.concatenate([c11_0[None, :], c11_tail], axis=0)
        d0 = jnp.concatenate([d0_0[None, :], d0_tail], axis=0)
        d1 = jnp.concatenate([d1_0[None, :], d1_tail], axis=0)

        def bwd(carry, xs):
            next0, next1 = carry
            c00_i, c01_i, c10_i, c11_i, d0_i, d1_i = xs
            cx0, cx1 = matvec2_components(
                c00_i,
                c01_i,
                c10_i,
                c11_i,
                next0,
                next1,
            )
            x0_i = d0_i - cx0
            x1_i = d1_i - cx1
            return (x0_i, x1_i), (x0_i, x1_i)

        x0_last = d0[-1]
        x1_last = d1[-1]
        _, reverse_tail = jax.lax.scan(
            bwd,
            (x0_last, x1_last),
            (
                c00[:-1][::-1],
                c01[:-1][::-1],
                c10[:-1][::-1],
                c11[:-1][::-1],
                d0[:-1][::-1],
                d1[:-1][::-1],
            ),
        )
        x0_rev, x1_rev = reverse_tail
        x0 = jnp.concatenate([x0_rev[::-1], x0_last[None, :]], axis=0)
        x1 = jnp.concatenate([x1_rev[::-1], x1_last[None, :]], axis=0)
        return x0.T, x1.T

    solution0 = jnp.zeros_like(rhs0)
    solution1 = jnp.zeros_like(rhs1)
    for residue in range(min(stride, n)):
        chain_idx = jnp.arange(residue, n, stride)
        chain0, chain1 = solve_chain(
            diag00[:, chain_idx],
            diag01[:, chain_idx],
            diag10[:, chain_idx],
            diag11[:, chain_idx],
            lower00[:, chain_idx],
            lower01[:, chain_idx],
            lower10[:, chain_idx],
            lower11[:, chain_idx],
            upper00[:, chain_idx],
            upper01[:, chain_idx],
            upper10[:, chain_idx],
            upper11[:, chain_idx],
            r0[:, chain_idx],
            r1[:, chain_idx],
        )
        solution0 = solution0.at[:, chain_idx].set(chain0)
        solution1 = solution1.at[:, chain_idx].set(chain1)
    return solution0, solution1


def apply_double_cable_block_system_soa(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    x0: Array,
    x1: Array,
) -> tuple[Array, Array]:
    """Apply the specialized double-cable block system to ``(x0, x1)``."""

    x0 = jnp.asarray(x0)
    x1 = jnp.asarray(x1)
    if x0.ndim != 2 or x1.ndim != 2:
        raise ValueError("x0 and x1 must have shape (batch_size, Nx).")
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}.")
    batch_size, n = x0.shape

    def as_space(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length Nx={n}, got {arr.shape}.")
            return jnp.broadcast_to(arr[None, :], (batch_size, n))
        if arr.ndim == 2 and arr.shape == (batch_size, n):
            return arr
        raise ValueError(
            f"{name} must have shape ({n},) or ({batch_size}, {n}), got {arr.shape}."
        )

    def as_edges(name: str, values: Array) -> Array:
        arr = jnp.asarray(values)
        edge_shape = (batch_size, max(n - 1, 0))
        if arr.ndim == 1:
            if arr.shape[0] != edge_shape[1]:
                raise ValueError(
                    f"{name} must have length Nx - 1={edge_shape[1]}, got {arr.shape}."
                )
            return jnp.broadcast_to(arr[None, :], edge_shape)
        if arr.ndim == 2 and arr.shape == edge_shape:
            return arr
        raise ValueError(
            f"{name} must have shape ({edge_shape[1]},) or {edge_shape}, got {arr.shape}."
        )

    a00_b = as_space("a00", a00)
    a01_b = as_space("a01", a01)
    a10_b = as_space("a10", a10)
    a11_b = as_space("a11", a11)
    off0_b = as_edges("off0", off0)
    off1_b = as_edges("off1", off1)
    zero_col = jnp.zeros((batch_size, 1), dtype=x0.dtype)

    y0 = a00_b * x0 + a01_b * x1
    y1 = a10_b * x0 + a11_b * x1
    y0 = y0 + jnp.concatenate([zero_col, off0_b * x0[:, :-1]], axis=1)
    y0 = y0 + jnp.concatenate([off0_b * x0[:, 1:], zero_col], axis=1)
    y1 = y1 + jnp.concatenate([zero_col, off1_b * x1[:, :-1]], axis=1)
    y1 = y1 + jnp.concatenate([off1_b * x1[:, 1:], zero_col], axis=1)
    return y0, y1


def double_cable_block_residual_norm(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    x0: Array,
    x1: Array,
    *,
    eps: float = 1e-12,
) -> Array:
    """Return per-row relative residual norms for a double-cable block solve."""

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    ax0, ax1 = apply_double_cable_block_system_soa(a00, a01, a10, a11, off0, off1, x0, x1)
    residual0 = ax0 - rhs0
    residual1 = ax1 - rhs1
    numerator = jnp.sqrt(jnp.sum(residual0 * residual0 + residual1 * residual1, axis=1))
    denominator = jnp.sqrt(jnp.sum(rhs0 * rhs0 + rhs1 * rhs1, axis=1))
    return numerator / (denominator + jnp.asarray(eps, dtype=rhs0.dtype))


def double_cable_power_bucket(
    nx: int,
    *,
    buckets: tuple[int, ...] = (32, 64, 128),
) -> int:
    """Return the smallest supported PCR padding bucket for ``nx``."""

    value = int(nx)
    if value < 1:
        raise ValueError("nx must be >= 1.")
    for bucket in buckets:
        if value <= int(bucket):
            return int(bucket)
    raise ValueError(f"nx={value} exceeds supported padding buckets {buckets}.")


def pad_double_cable_system_to_power_bucket(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    bucket: int | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Pad a batched 2x2 double-cable system to a power bucket.

    Padding rows are exact identity equations:

    ``I * x_pad = 0`` with zero coupling to the real system.

    The real solution is therefore unchanged after slicing back to the original
    ``Nx``. Coefficients may be shared ``[Nx]`` / ``[Nx - 1]`` or batched
    ``[B, Nx]`` / ``[B, Nx - 1]``; RHS arrays must be batch-first ``[B, Nx]``.
    """

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )

    _, nx = rhs0.shape
    target = double_cable_power_bucket(nx) if bucket is None else int(bucket)
    if target < nx:
        raise ValueError(f"bucket must be >= Nx={nx}, got {target}.")
    if target == nx:
        return a00, a01, a10, a11, off0, off1, rhs0, rhs1

    def pad_last_axis(values: Array, *, current: int, fill: float) -> Array:
        arr = jnp.asarray(values)
        if arr.shape[-1] != current:
            raise ValueError(
                f"expected trailing length {current}, got shape {arr.shape}."
            )
        pad = target - current
        pad_width = ((0, pad),) if arr.ndim == 1 else ((0, 0), (0, pad))
        return jnp.pad(arr, pad_width, constant_values=jnp.asarray(fill, dtype=arr.dtype))

    edge_count = max(nx - 1, 0)
    edge_target = max(target - 1, 0)

    return (
        pad_last_axis(a00, current=nx, fill=1.0),
        pad_last_axis(a01, current=nx, fill=0.0),
        pad_last_axis(a10, current=nx, fill=0.0),
        pad_last_axis(a11, current=nx, fill=1.0),
        pad_last_axis(off0, current=edge_count, fill=0.0)[..., :edge_target],
        pad_last_axis(off1, current=edge_count, fill=0.0)[..., :edge_target],
        pad_last_axis(rhs0, current=nx, fill=0.0),
        pad_last_axis(rhs1, current=nx, fill=0.0),
    )


def solve_block_tridiagonal_2x2_pcr_soa_batched_padded(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    bucket: int | None = None,
) -> tuple[Array, Array]:
    """Solve a batched SoA PCR system after exact identity-row padding."""

    nx = int(jnp.asarray(rhs0).shape[1])
    padded = pad_double_cable_system_to_power_bucket(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        bucket=bucket,
    )
    x0, x1 = solve_block_tridiagonal_2x2_pcr_soa_batched(*padded)
    return x0[:, :nx], x1[:, :nx]



def solve_block_tridiagonal_2x2_pcr_soa_batched_symmetric(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    """Benchmark-only symmetric PCR/SoA candidate for double-cable systems.

    This is not a production runtime route. It assumes the block tridiagonal
    matrix is symmetric, as in the current exact double-cable assembly:
    diagonal blocks satisfy ``a01 == a10`` and off-block couplings are diagonal
    with the same lower/upper values. Under that invariant, PCR upper couplings
    can be reconstructed from shifted/transposed lower couplings instead of
    carried as a second live state.

    The goal is to test whether reducing the live PCR state helps XLA lower the
    large tuple-producing GPU fusions observed in P11B lowering audits.
    """

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )

    batch_size, n = rhs0.shape
    dtype = rhs0.dtype
    idx = jnp.arange(n)
    zero = jnp.zeros((), dtype=dtype)

    def broadcast_space(name: str, values: Any) -> Any:
        arr = jnp.asarray(values)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length Nx={n}, got {arr.shape}.")
            return jnp.broadcast_to(arr[None, :], (batch_size, n))
        if arr.ndim == 2:
            if arr.shape != (batch_size, n):
                raise ValueError(
                    f"{name} must have shape ({batch_size}, {n}), got {arr.shape}."
                )
            return arr
        raise ValueError(
            f"{name} must have shape (Nx,) or (batch_size, Nx), got {arr.shape}."
        )

    def broadcast_edges(name: str, values: Any) -> Any:
        arr = jnp.asarray(values)
        edge_shape = (batch_size, max(n - 1, 0))
        if arr.ndim == 1:
            if arr.shape[0] != edge_shape[1]:
                raise ValueError(
                    f"{name} must have length Nx - 1={edge_shape[1]}, got {arr.shape}."
                )
            return jnp.broadcast_to(arr[None, :], edge_shape)
        if arr.ndim == 2:
            if arr.shape != edge_shape:
                raise ValueError(f"{name} must have shape {edge_shape}, got {arr.shape}.")
            return arr
        raise ValueError(
            f"{name} must have shape (Nx - 1,) or (batch_size, Nx - 1), got {arr.shape}."
        )

    diag00 = broadcast_space("a00", a00)
    diag01 = broadcast_space("a01", a01)
    diag10 = broadcast_space("a10", a10)
    diag11 = broadcast_space("a11", a11)
    off0_batched = broadcast_edges("off0", off0)
    off1_batched = broadcast_edges("off1", off1)

    zero_col = jnp.zeros((batch_size, 1), dtype=dtype)
    zeros = jnp.zeros((batch_size, n), dtype=dtype)
    lower00 = jnp.concatenate([zero_col, off0_batched], axis=1)
    lower01 = zeros
    lower10 = zeros
    lower11 = jnp.concatenate([zero_col, off1_batched], axis=1)
    r0 = rhs0
    r1 = rhs1

    def inv2_components(
        m00: Any,
        m01: Any,
        m10: Any,
        m11: Any,
    ) -> tuple[Any, Any, Any, Any]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    def matmul2_components(
        l00: Any,
        l01: Any,
        l10: Any,
        l11: Any,
        r00: Any,
        r01: Any,
        r10: Any,
        r11: Any,
    ) -> tuple[Any, Any, Any, Any]:
        return (
            l00 * r00 + l01 * r10,
            l00 * r01 + l01 * r11,
            l10 * r00 + l11 * r10,
            l10 * r01 + l11 * r11,
        )

    def matvec2_components(
        m00: Any,
        m01: Any,
        m10: Any,
        m11: Any,
        v0: Any,
        v1: Any,
    ) -> tuple[Any, Any]:
        return m00 * v0 + m01 * v1, m10 * v0 + m11 * v1

    stride = 1
    while stride < n:
        left_idx = jnp.maximum(idx - stride, 0)
        right_idx = jnp.minimum(idx + stride, n - 1)
        has_left = (idx >= stride)[None, :]
        has_right = (idx + stride < n)[None, :]

        left_inv = inv2_components(
            diag00[:, left_idx],
            diag01[:, left_idx],
            diag10[:, left_idx],
            diag11[:, left_idx],
        )
        right_inv = inv2_components(
            diag00[:, right_idx],
            diag01[:, right_idx],
            diag10[:, right_idx],
            diag11[:, right_idx],
        )

        upper00 = lower00[:, right_idx]
        upper01 = lower10[:, right_idx]
        upper10 = lower01[:, right_idx]
        upper11 = lower11[:, right_idx]

        lf00, lf01, lf10, lf11 = matmul2_components(
            lower00,
            lower01,
            lower10,
            lower11,
            *left_inv,
        )
        rf00, rf01, rf10, rf11 = matmul2_components(
            upper00,
            upper01,
            upper10,
            upper11,
            *right_inv,
        )
        lf00 = jnp.where(has_left, lf00, zero)
        lf01 = jnp.where(has_left, lf01, zero)
        lf10 = jnp.where(has_left, lf10, zero)
        lf11 = jnp.where(has_left, lf11, zero)
        rf00 = jnp.where(has_right, rf00, zero)
        rf01 = jnp.where(has_right, rf01, zero)
        rf10 = jnp.where(has_right, rf10, zero)
        rf11 = jnp.where(has_right, rf11, zero)

        left_lower00 = lower00[:, left_idx]
        left_lower01 = lower01[:, left_idx]
        left_lower10 = lower10[:, left_idx]
        left_lower11 = lower11[:, left_idx]
        right_lower00 = lower00[:, right_idx]
        right_lower01 = lower01[:, right_idx]
        right_lower10 = lower10[:, right_idx]
        right_lower11 = lower11[:, right_idx]

        nl00, nl01, nl10, nl11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            left_lower00,
            left_lower01,
            left_lower10,
            left_lower11,
        )
        ldu00, ldu01, ldu10, ldu11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            lower00,
            lower10,
            lower01,
            lower11,
        )
        rdl00, rdl01, rdl10, rdl11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            right_lower00,
            right_lower01,
            right_lower10,
            right_lower11,
        )
        lr0, lr1 = matvec2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            r0[:, left_idx],
            r1[:, left_idx],
        )
        rr0, rr1 = matvec2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            r0[:, right_idx],
            r1[:, right_idx],
        )

        lower00 = jnp.where(has_left, -nl00, zero)
        lower01 = jnp.where(has_left, -nl01, zero)
        lower10 = jnp.where(has_left, -nl10, zero)
        lower11 = jnp.where(has_left, -nl11, zero)
        diag00 = diag00 - ldu00 - rdl00
        diag01 = diag01 - ldu01 - rdl01
        diag10 = diag10 - ldu10 - rdl10
        diag11 = diag11 - ldu11 - rdl11
        r0 = r0 - lr0 - rr0
        r1 = r1 - lr1 - rr1
        stride *= 2

    inv00, inv01, inv10, inv11 = inv2_components(diag00, diag01, diag10, diag11)
    return matvec2_components(inv00, inv01, inv10, inv11, r0, r1)
