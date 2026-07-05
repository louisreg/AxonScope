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


@triton.jit
def pcr_soa_init_kernel(
    a00,
    a01,
    a10,
    a11,
    off0,
    off1,
    rhs0,
    rhs1,
    lower00,
    lower01,
    lower10,
    lower11,
    upper00,
    upper01,
    upper10,
    upper11,
    diag00,
    diag01,
    diag10,
    diag11,
    r0,
    r1,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Initialize full SoA PCR work arrays from compact tridiagonal inputs."""

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL
    i = offsets % N
    batch = offsets // N
    edge_base = batch * (N - 1)

    zero = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    has_left = i > 0
    has_right = i < N - 1
    left_edge = edge_base + i - 1
    right_edge = edge_base + i

    l0 = tl.load(off0 + left_edge, mask & has_left, other=0.0)
    l1 = tl.load(off1 + left_edge, mask & has_left, other=0.0)
    u0 = tl.load(off0 + right_edge, mask & has_right, other=0.0)
    u1 = tl.load(off1 + right_edge, mask & has_right, other=0.0)

    tl.store(lower00 + offsets, l0, mask)
    tl.store(lower01 + offsets, zero, mask)
    tl.store(lower10 + offsets, zero, mask)
    tl.store(lower11 + offsets, l1, mask)
    tl.store(upper00 + offsets, u0, mask)
    tl.store(upper01 + offsets, zero, mask)
    tl.store(upper10 + offsets, zero, mask)
    tl.store(upper11 + offsets, u1, mask)
    tl.store(diag00 + offsets, tl.load(a00 + offsets, mask, other=1.0), mask)
    tl.store(diag01 + offsets, tl.load(a01 + offsets, mask, other=0.0), mask)
    tl.store(diag10 + offsets, tl.load(a10 + offsets, mask, other=0.0), mask)
    tl.store(diag11 + offsets, tl.load(a11 + offsets, mask, other=1.0), mask)
    tl.store(r0 + offsets, tl.load(rhs0 + offsets, mask, other=0.0), mask)
    tl.store(r1 + offsets, tl.load(rhs1 + offsets, mask, other=0.0), mask)


@triton.jit
def pcr_soa_stage_kernel(
    lower00,
    lower01,
    lower10,
    lower11,
    upper00,
    upper01,
    upper10,
    upper11,
    diag00,
    diag01,
    diag10,
    diag11,
    r0,
    r1,
    out_lower00,
    out_lower01,
    out_lower10,
    out_lower11,
    out_upper00,
    out_upper01,
    out_upper10,
    out_upper11,
    out_diag00,
    out_diag01,
    out_diag10,
    out_diag11,
    out_r0,
    out_r1,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One global-memory SoA PCR stage over the flattened B x N grid."""

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL
    i = offsets % N
    batch_base = (offsets // N) * N
    left_i = tl.maximum(i - STRIDE, 0)
    right_i = tl.minimum(i + STRIDE, N - 1)
    left_offsets = batch_base + left_i
    right_offsets = batch_base + right_i
    has_left = i >= STRIDE
    has_right = i + STRIDE < N

    zero = tl.full((BLOCK_SIZE,), 0.0, tl.float32)

    d00 = tl.load(diag00 + offsets, mask, other=1.0)
    d01 = tl.load(diag01 + offsets, mask, other=0.0)
    d10 = tl.load(diag10 + offsets, mask, other=0.0)
    d11 = tl.load(diag11 + offsets, mask, other=1.0)
    cur_r0 = tl.load(r0 + offsets, mask, other=0.0)
    cur_r1 = tl.load(r1 + offsets, mask, other=0.0)

    l00 = tl.load(lower00 + offsets, mask, other=0.0)
    l01 = tl.load(lower01 + offsets, mask, other=0.0)
    l10 = tl.load(lower10 + offsets, mask, other=0.0)
    l11 = tl.load(lower11 + offsets, mask, other=0.0)
    u00 = tl.load(upper00 + offsets, mask, other=0.0)
    u01 = tl.load(upper01 + offsets, mask, other=0.0)
    u10 = tl.load(upper10 + offsets, mask, other=0.0)
    u11 = tl.load(upper11 + offsets, mask, other=0.0)

    ld00 = tl.load(diag00 + left_offsets, mask, other=1.0)
    ld01 = tl.load(diag01 + left_offsets, mask, other=0.0)
    ld10 = tl.load(diag10 + left_offsets, mask, other=0.0)
    ld11 = tl.load(diag11 + left_offsets, mask, other=1.0)
    ldet = ld00 * ld11 - ld01 * ld10
    linv00 = ld11 / ldet
    linv01 = -ld01 / ldet
    linv10 = -ld10 / ldet
    linv11 = ld00 / ldet

    rd00 = tl.load(diag00 + right_offsets, mask, other=1.0)
    rd01 = tl.load(diag01 + right_offsets, mask, other=0.0)
    rd10 = tl.load(diag10 + right_offsets, mask, other=0.0)
    rd11 = tl.load(diag11 + right_offsets, mask, other=1.0)
    rdet = rd00 * rd11 - rd01 * rd10
    rinv00 = rd11 / rdet
    rinv01 = -rd01 / rdet
    rinv10 = -rd10 / rdet
    rinv11 = rd00 / rdet

    lf00 = l00 * linv00 + l01 * linv10
    lf01 = l00 * linv01 + l01 * linv11
    lf10 = l10 * linv00 + l11 * linv10
    lf11 = l10 * linv01 + l11 * linv11
    rf00 = u00 * rinv00 + u01 * rinv10
    rf01 = u00 * rinv01 + u01 * rinv11
    rf10 = u10 * rinv00 + u11 * rinv10
    rf11 = u10 * rinv01 + u11 * rinv11

    lf00 = tl.where(has_left, lf00, zero)
    lf01 = tl.where(has_left, lf01, zero)
    lf10 = tl.where(has_left, lf10, zero)
    lf11 = tl.where(has_left, lf11, zero)
    rf00 = tl.where(has_right, rf00, zero)
    rf01 = tl.where(has_right, rf01, zero)
    rf10 = tl.where(has_right, rf10, zero)
    rf11 = tl.where(has_right, rf11, zero)

    ll00 = tl.load(lower00 + left_offsets, mask, other=0.0)
    ll01 = tl.load(lower01 + left_offsets, mask, other=0.0)
    ll10 = tl.load(lower10 + left_offsets, mask, other=0.0)
    ll11 = tl.load(lower11 + left_offsets, mask, other=0.0)
    lu00 = tl.load(upper00 + left_offsets, mask, other=0.0)
    lu01 = tl.load(upper01 + left_offsets, mask, other=0.0)
    lu10 = tl.load(upper10 + left_offsets, mask, other=0.0)
    lu11 = tl.load(upper11 + left_offsets, mask, other=0.0)
    lr0 = tl.load(r0 + left_offsets, mask, other=0.0)
    lr1 = tl.load(r1 + left_offsets, mask, other=0.0)

    ru00 = tl.load(upper00 + right_offsets, mask, other=0.0)
    ru01 = tl.load(upper01 + right_offsets, mask, other=0.0)
    ru10 = tl.load(upper10 + right_offsets, mask, other=0.0)
    ru11 = tl.load(upper11 + right_offsets, mask, other=0.0)
    rl00 = tl.load(lower00 + right_offsets, mask, other=0.0)
    rl01 = tl.load(lower01 + right_offsets, mask, other=0.0)
    rl10 = tl.load(lower10 + right_offsets, mask, other=0.0)
    rl11 = tl.load(lower11 + right_offsets, mask, other=0.0)
    rr0 = tl.load(r0 + right_offsets, mask, other=0.0)
    rr1 = tl.load(r1 + right_offsets, mask, other=0.0)

    nl00 = lf00 * ll00 + lf01 * ll10
    nl01 = lf00 * ll01 + lf01 * ll11
    nl10 = lf10 * ll00 + lf11 * ll10
    nl11 = lf10 * ll01 + lf11 * ll11
    nu00 = rf00 * ru00 + rf01 * ru10
    nu01 = rf00 * ru01 + rf01 * ru11
    nu10 = rf10 * ru00 + rf11 * ru10
    nu11 = rf10 * ru01 + rf11 * ru11

    ldu00 = lf00 * lu00 + lf01 * lu10
    ldu01 = lf00 * lu01 + lf01 * lu11
    ldu10 = lf10 * lu00 + lf11 * lu10
    ldu11 = lf10 * lu01 + lf11 * lu11
    rdl00 = rf00 * rl00 + rf01 * rl10
    rdl01 = rf00 * rl01 + rf01 * rl11
    rdl10 = rf10 * rl00 + rf11 * rl10
    rdl11 = rf10 * rl01 + rf11 * rl11
    lvr0 = lf00 * lr0 + lf01 * lr1
    lvr1 = lf10 * lr0 + lf11 * lr1
    rvr0 = rf00 * rr0 + rf01 * rr1
    rvr1 = rf10 * rr0 + rf11 * rr1

    tl.store(out_lower00 + offsets, tl.where(has_left, -nl00, zero), mask)
    tl.store(out_lower01 + offsets, tl.where(has_left, -nl01, zero), mask)
    tl.store(out_lower10 + offsets, tl.where(has_left, -nl10, zero), mask)
    tl.store(out_lower11 + offsets, tl.where(has_left, -nl11, zero), mask)
    tl.store(out_upper00 + offsets, tl.where(has_right, -nu00, zero), mask)
    tl.store(out_upper01 + offsets, tl.where(has_right, -nu01, zero), mask)
    tl.store(out_upper10 + offsets, tl.where(has_right, -nu10, zero), mask)
    tl.store(out_upper11 + offsets, tl.where(has_right, -nu11, zero), mask)
    tl.store(out_diag00 + offsets, d00 - ldu00 - rdl00, mask)
    tl.store(out_diag01 + offsets, d01 - ldu01 - rdl01, mask)
    tl.store(out_diag10 + offsets, d10 - ldu10 - rdl10, mask)
    tl.store(out_diag11 + offsets, d11 - ldu11 - rdl11, mask)
    tl.store(out_r0 + offsets, cur_r0 - lvr0 - rvr0, mask)
    tl.store(out_r1 + offsets, cur_r1 - lvr1 - rvr1, mask)


@triton.jit
def pcr_soa_final_kernel(
    diag00,
    diag01,
    diag10,
    diag11,
    r0,
    r1,
    out0,
    out1,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Invert the final independent 2x2 systems after PCR stages."""

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    d00 = tl.load(diag00 + offsets, mask, other=1.0)
    d01 = tl.load(diag01 + offsets, mask, other=0.0)
    d10 = tl.load(diag10 + offsets, mask, other=0.0)
    d11 = tl.load(diag11 + offsets, mask, other=1.0)
    v0 = tl.load(r0 + offsets, mask, other=0.0)
    v1 = tl.load(r1 + offsets, mask, other=0.0)
    det = d00 * d11 - d01 * d10
    tl.store(out0 + offsets, (d11 * v0 - d01 * v1) / det, mask)
    tl.store(out1 + offsets, (-d10 * v0 + d00 * v1) / det, mask)
