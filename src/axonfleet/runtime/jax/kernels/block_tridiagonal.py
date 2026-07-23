from __future__ import annotations

import jax
import jax.numpy as jnp

from ..cable_geometry import Array


def solve_block_tridiagonal_2x2_scalar(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Solve a 2x2 block tridiagonal system with diagonal off-blocks."""

    zero = jnp.zeros((), dtype=a00.dtype)
    upper0 = jnp.concatenate([off0, zero[None]])
    upper1 = jnp.concatenate([off1, zero[None]])

    def inv_components(m00, m01, m10, m11):
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    inv00, inv01, inv10, inv11 = inv_components(a00[0], a01[0], a10[0], a11[0])
    c00_0 = inv00 * upper0[0]
    c01_0 = inv01 * upper1[0]
    c10_0 = inv10 * upper0[0]
    c11_0 = inv11 * upper1[0]
    d0_0 = inv00 * rhs0[0] + inv01 * rhs1[0]
    d1_0 = inv10 * rhs0[0] + inv11 * rhs1[0]

    def fwd(carry, xs):
        c00_prev, c01_prev, c10_prev, c11_prev, d0_prev, d1_prev = carry
        (
            a00_i,
            a01_i,
            a10_i,
            a11_i,
            lower0,
            lower1,
            upper0_i,
            upper1_i,
            rhs0_i,
            rhs1_i,
        ) = xs

        m00 = a00_i - lower0 * c00_prev
        m01 = a01_i - lower0 * c01_prev
        m10 = a10_i - lower1 * c10_prev
        m11 = a11_i - lower1 * c11_prev
        inv00_i, inv01_i, inv10_i, inv11_i = inv_components(m00, m01, m10, m11)

        r0 = rhs0_i - lower0 * d0_prev
        r1 = rhs1_i - lower1 * d1_prev
        c00_i = inv00_i * upper0_i
        c01_i = inv01_i * upper1_i
        c10_i = inv10_i * upper0_i
        c11_i = inv11_i * upper1_i
        d0_i = inv00_i * r0 + inv01_i * r1
        d1_i = inv10_i * r0 + inv11_i * r1
        out = (c00_i, c01_i, c10_i, c11_i, d0_i, d1_i)
        return out, out

    _, forward_tail = jax.lax.scan(
        fwd,
        (c00_0, c01_0, c10_0, c11_0, d0_0, d1_0),
        (
            a00[1:],
            a01[1:],
            a10[1:],
            a11[1:],
            off0,
            off1,
            upper0[1:],
            upper1[1:],
            rhs0[1:],
            rhs1[1:],
        ),
    )
    c00_tail, c01_tail, c10_tail, c11_tail, d0_tail, d1_tail = forward_tail
    c00 = jnp.concatenate([c00_0[None], c00_tail])
    c01 = jnp.concatenate([c01_0[None], c01_tail])
    c10 = jnp.concatenate([c10_0[None], c10_tail])
    c11 = jnp.concatenate([c11_0[None], c11_tail])
    d0 = jnp.concatenate([d0_0[None], d0_tail])
    d1 = jnp.concatenate([d1_0[None], d1_tail])

    def bwd(carry, xs):
        next0, next1 = carry
        c00_i, c01_i, c10_i, c11_i, d0_i, d1_i = xs
        x0_i = d0_i - c00_i * next0 - c01_i * next1
        x1_i = d1_i - c10_i * next0 - c11_i * next1
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
    x0 = jnp.concatenate([x0_rev[::-1], x0_last[None]])
    x1 = jnp.concatenate([x1_rev[::-1], x1_last[None]])
    return x0, x1
