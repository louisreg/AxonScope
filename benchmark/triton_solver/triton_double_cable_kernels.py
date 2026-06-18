"""Triton kernels for exact double-cable 2x2 block solves."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def block_thomas_forward_kernel(
    a00,
    a01,
    a10,
    a11,
    off0,
    off1,
    rhs0,
    rhs1,
    c00,
    c01,
    c10,
    c11,
    d0,
    d1,
    N: tl.constexpr,
):
    """Forward elimination for one 2x2 block-tridiagonal system per program."""

    batch = tl.program_id(0)
    base = batch * N
    edge_base = batch * (N - 1)

    m00 = tl.load(a00 + base)
    m01 = tl.load(a01 + base)
    m10 = tl.load(a10 + base)
    m11 = tl.load(a11 + base)
    r0 = tl.load(rhs0 + base)
    r1 = tl.load(rhs1 + base)
    det = m00 * m11 - m01 * m10
    inv00 = m11 / det
    inv01 = -m01 / det
    inv10 = -m10 / det
    inv11 = m00 / det

    u0 = tl.full((), 0.0, tl.float32)
    u1 = tl.full((), 0.0, tl.float32)
    if N > 1:
        u0 = tl.load(off0 + edge_base)
        u1 = tl.load(off1 + edge_base)
    cp00 = inv00 * u0
    cp01 = inv01 * u1
    cp10 = inv10 * u0
    cp11 = inv11 * u1
    dp0 = inv00 * r0 + inv01 * r1
    dp1 = inv10 * r0 + inv11 * r1
    tl.store(c00 + base, cp00)
    tl.store(c01 + base, cp01)
    tl.store(c10 + base, cp10)
    tl.store(c11 + base, cp11)
    tl.store(d0 + base, dp0)
    tl.store(d1 + base, dp1)

    prev_c00 = cp00
    prev_c01 = cp01
    prev_c10 = cp10
    prev_c11 = cp11
    prev_d0 = dp0
    prev_d1 = dp1

    for i in tl.static_range(1, N):
        offset = base + i
        l0 = tl.load(off0 + edge_base + i - 1)
        l1 = tl.load(off1 + edge_base + i - 1)
        m00 = tl.load(a00 + offset) - l0 * prev_c00
        m01 = tl.load(a01 + offset) - l0 * prev_c01
        m10 = tl.load(a10 + offset) - l1 * prev_c10
        m11 = tl.load(a11 + offset) - l1 * prev_c11
        r0 = tl.load(rhs0 + offset) - l0 * prev_d0
        r1 = tl.load(rhs1 + offset) - l1 * prev_d1

        det = m00 * m11 - m01 * m10
        inv00 = m11 / det
        inv01 = -m01 / det
        inv10 = -m10 / det
        inv11 = m00 / det

        u0 = tl.full((), 0.0, tl.float32)
        u1 = tl.full((), 0.0, tl.float32)
        if i < N - 1:
            u0 = tl.load(off0 + edge_base + i)
            u1 = tl.load(off1 + edge_base + i)
        cp00 = inv00 * u0
        cp01 = inv01 * u1
        cp10 = inv10 * u0
        cp11 = inv11 * u1
        dp0 = inv00 * r0 + inv01 * r1
        dp1 = inv10 * r0 + inv11 * r1

        tl.store(c00 + offset, cp00)
        tl.store(c01 + offset, cp01)
        tl.store(c10 + offset, cp10)
        tl.store(c11 + offset, cp11)
        tl.store(d0 + offset, dp0)
        tl.store(d1 + offset, dp1)
        prev_c00 = cp00
        prev_c01 = cp01
        prev_c10 = cp10
        prev_c11 = cp11
        prev_d0 = dp0
        prev_d1 = dp1


@triton.jit
def block_thomas_backward_kernel(
    c00,
    c01,
    c10,
    c11,
    d0,
    d1,
    out0,
    out1,
    N: tl.constexpr,
):
    """Backward substitution for one 2x2 block-tridiagonal system per program."""

    batch = tl.program_id(0)
    base = batch * N
    last = base + N - 1

    x0 = tl.load(d0 + last)
    x1 = tl.load(d1 + last)
    tl.store(out0 + last, x0)
    tl.store(out1 + last, x1)

    for rev in tl.static_range(0, N - 1):
        i = N - 2 - rev
        offset = base + i
        cp00 = tl.load(c00 + offset)
        cp01 = tl.load(c01 + offset)
        cp10 = tl.load(c10 + offset)
        cp11 = tl.load(c11 + offset)
        next_x0 = x0
        next_x1 = x1
        x0 = tl.load(d0 + offset) - cp00 * next_x0 - cp01 * next_x1
        x1 = tl.load(d1 + offset) - cp10 * next_x0 - cp11 * next_x1
        tl.store(out0 + offset, x0)
        tl.store(out1 + offset, x1)

