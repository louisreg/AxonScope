from __future__ import annotations

from typing import Any

import jax.numpy as jnp


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
        inv_det = jnp.reciprocal(det)
        return m11 * inv_det, -m01 * inv_det, -m10 * inv_det, m00 * inv_det

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
