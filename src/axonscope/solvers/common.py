from __future__ import annotations

import math
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp

from axonscope.solvers.axon_runtime import SolverAxon
from axonscope.utils import units

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Array: TypeAlias = jnp.ndarray
Carry: TypeAlias = tuple[Array, Array]  # generic (V, gates) carry used by scan


def simulation_step_count(duration_ms: float, dt_ms: float) -> int:
    """Return the number of fixed time steps for an exact simulation grid.

    Current solver kernels use one fixed ``dt`` for every integration step. If
    ``duration_ms`` is not an integer multiple of ``dt_ms``, rounding up would
    silently run past the requested final time. Refuse that case until kernels
    grow an explicit partial-final-step policy.
    """

    duration = float(duration_ms)
    step = float(dt_ms)
    if duration <= 0.0:
        raise ValueError("duration_ms must be > 0.")
    if step <= 0.0:
        raise ValueError("dt_ms must be > 0.")

    ratio = duration / step
    steps = int(round(ratio))
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "duration_ms must be an integer multiple of dt_ms for the current "
            f"fixed-step solvers; got duration_ms={duration:g}, dt_ms={step:g}."
        )
    return steps


def resolve_time_args(
    *,
    tsim: Any | None = None,
    dt: Any | None = None,
) -> tuple[float, float]:
    """Resolve solver-level time values to ``(duration_ms, dt_ms)``.

    Solvers operate in milliseconds. Plain numeric values are interpreted as
    milliseconds; Pint-like quantities are converted at this boundary.
    """

    if tsim is None:
        raise ValueError("tsim is required.")
    if dt is None:
        raise ValueError("dt is required.")

    duration = units.to_ms(tsim)
    step = units.to_ms(dt)
    if duration <= 0.0:
        raise ValueError("tsim must be > 0.")
    if step <= 0.0:
        raise ValueError("dt must be > 0.")
    simulation_step_count(duration, step)
    return duration, step


def initial_voltage(axon: object, Nx: int, dtype_local: jnp.dtype) -> Array:
    """Initial voltage state."""
    return jnp.full((Nx,), axon.v_init, dtype=dtype_local)


def diffusion_operator_coeffs(
    axon: SolverAxon,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """
    Build the discrete diffusion operator coefficients on the current mesh.

    For interior node i on a non-uniform mesh with left/right edge lengths
    h_{i-1}, h_i, the second derivative is approximated by:

        d2Vdx2 |_i ≈ 2 * [ (V_{i+1} - V_i) / h_i - (V_i - V_{i-1}) / h_{i-1} ]
                     / (h_{i-1} + h_i)
    which reduces to the standard centered stencil on a uniform grid.

    At the two cable ends we impose sealed-end (zero-flux / Neumann) boundary
    conditions using mirrored ghost points, giving:

        d2Vdx2 |_0     ≈ 2 * (V_1 - V_0) / h_0^2
        d2Vdx2 |_{N-1} ≈ 2 * (V_{N-2} - V_{N-1}) / h_{N-2}^2
    """
    if bool(getattr(axon, "has_heterogeneous_cable_properties", False)):
        lengths_cm = compartment_length_cm(axon, dtype_local)
        diam_um = compartment_diam_um(axon, dtype_local)
        ra_ohm_cm = compartment_ra_ohm_cm(axon, dtype_local)
        cm_uF_cm2 = compartment_cm_uF_cm2(axon, dtype_local)

        area_cm2 = jnp.pi * (diam_um * dtype_local(1e-4)) * lengths_cm
        radius_cm = dtype_local(0.5) * diam_um * dtype_local(1e-4)
        cross_section_cm2 = jnp.pi * radius_cm**2
        left_half_cm = dtype_local(0.5) * lengths_cm[:-1]
        right_half_cm = dtype_local(0.5) * lengths_cm[1:]
        edge_resistance_ohm = (
            ra_ohm_cm[:-1] * left_half_cm / cross_section_cm2[:-1]
            + ra_ohm_cm[1:] * right_half_cm / cross_section_cm2[1:]
        )
        gax_i_mS = dtype_local(1e3) / jnp.maximum(edge_resistance_ohm, dtype_local(1e-18))
        cm_abs_uF = cm_uF_cm2 * area_cm2

        Nx = axon.n_compartments
        lower = jnp.zeros((Nx,), dtype=dtype_local)
        diag = jnp.zeros((Nx,), dtype=dtype_local)
        upper = jnp.zeros((Nx,), dtype=dtype_local)
        lower = lower.at[1:].set(gax_i_mS / cm_abs_uF[1:])
        upper = upper.at[:-1].set(gax_i_mS / cm_abs_uF[:-1])
        diag = -(lower + upper)
        return lower, diag, upper

    Nx = axon.n_compartments
    lower = jnp.zeros((Nx,), dtype=dtype_local)
    diag = jnp.zeros((Nx,), dtype=dtype_local)
    upper = jnp.zeros((Nx,), dtype=dtype_local)

    h = jnp.asarray(axon.h_cm, dtype=dtype_local)

    D = uniform_diffusion_coefficient(axon, dtype_local)

    if Nx >= 2:
        left_coef = 2.0 * D / (h[0] ** 2)
        right_coef = 2.0 * D / (h[-1] ** 2)

        diag = diag.at[0].set(-left_coef)
        upper = upper.at[0].set(left_coef)
        lower = lower.at[-1].set(right_coef)
        diag = diag.at[-1].set(-right_coef)

    if Nx > 2:
        h_left = h[:-1]
        h_right = h[1:]
        denom = h_left + h_right

        lower_inner = 2.0 * D / (h_left * denom)
        diag_inner = -2.0 * D / (h_left * h_right)
        upper_inner = 2.0 * D / (h_right * denom)

        lower = lower.at[1:-1].set(lower_inner)
        diag = diag.at[1:-1].set(diag_inner)
        upper = upper.at[1:-1].set(upper_inner)

    return lower, diag, upper


def compartment_diam_um(axon: SolverAxon, dtype_local: jnp.dtype) -> Array:
    return jnp.asarray(axon.diam_um, dtype=dtype_local)


def compartment_cm_uF_cm2(axon: SolverAxon, dtype_local: jnp.dtype) -> Array:
    return jnp.asarray(axon.Cm_uF_cm2, dtype=dtype_local)


def compartment_ra_ohm_cm(axon: SolverAxon, dtype_local: jnp.dtype) -> Array:
    return jnp.asarray(axon.Ra_ohm_cm, dtype=dtype_local)


def uniform_diffusion_coefficient(axon: SolverAxon, dtype_local: jnp.dtype) -> Array:
    """Return the scalar cable diffusion coefficient from compartment vectors."""

    diam_um = jnp.mean(compartment_diam_um(axon, dtype_local))
    ra_ohm_cm = jnp.mean(compartment_ra_ohm_cm(axon, dtype_local))
    cm_uF_cm2 = jnp.mean(compartment_cm_uF_cm2(axon, dtype_local))
    radius_cm = dtype_local(0.5) * diam_um * dtype_local(1e-4)
    cm = dtype_local(2.0) * jnp.pi * radius_cm * cm_uF_cm2 * dtype_local(1e-6)
    ra = ra_ohm_cm / (jnp.pi * radius_cm**2)
    return dtype_local(1.0) / (ra * cm) / dtype_local(1000.0)


def compartment_length_cm(axon: SolverAxon, dtype_local: jnp.dtype) -> Array:
    return jnp.asarray(axon.compartment_lengths_um, dtype=dtype_local) * dtype_local(1e-4)


def compartment_area_cm2(axon: SolverAxon, dtype_local: jnp.dtype) -> Array:
    diam = compartment_diam_um(axon, dtype_local)
    length_cm = compartment_length_cm(axon, dtype_local)
    return jnp.pi * (diam * dtype_local(1e-4)) * length_cm


def extracellular_absolute_arrays(
    axon: SolverAxon, dtype_local: jnp.dtype
) -> tuple[Array, Array, Array, Array]:
    """Return (Cm_abs, Cx_abs, Gx_abs, Gax_e) in (uF, uF, mS, mS-edge)."""
    area = compartment_area_cm2(axon, dtype_local)
    cm_uF_cm2 = compartment_cm_uF_cm2(axon, dtype_local)
    Cm_abs = cm_uF_cm2 * area

    xg = jnp.asarray(axon.xg_S_cm2, dtype=dtype_local)
    xc = jnp.asarray(axon.xc_uF_cm2, dtype=dtype_local)
    xraxial = jnp.asarray(axon.xraxial_MOhm_per_cm, dtype=dtype_local)
    dx_cm = jnp.asarray(axon.dx_cm, dtype=dtype_local)

    Cx_abs = xc * area
    Gx_abs = (xg * dtype_local(1e3)) * area

    if axon.n_compartments <= 1:
        Gax_e = jnp.zeros((0,), dtype=dtype_local)
    else:
        R_edge_MOhm = (
            xraxial[:-1] * (dtype_local(0.5) * dx_cm[:-1])
            + xraxial[1:] * (dtype_local(0.5) * dx_cm[1:])
        )
        Gax_e = dtype_local(1e-3) / jnp.maximum(R_edge_MOhm, dtype_local(1e-18))
    return Cm_abs, Cx_abs, Gx_abs, Gax_e


def solve_block_tridiagonal_2x2(
    A_lower: Array,
    A_diag: Array,
    A_upper: Array,
    rhs: Array,
) -> Array:
    """Solve a block tridiagonal system with 2x2 blocks."""
    N = A_diag.shape[0]

    def inv2(M):
        a = M[0, 0]
        b = M[0, 1]
        c = M[1, 0]
        d = M[1, 1]
        det = a * d - b * c
        return jnp.array([[d, -b], [-c, a]], dtype=M.dtype) / det

    invD0 = inv2(A_diag[0])
    C0 = invD0 @ A_upper[0]
    d0 = invD0 @ rhs[0]
    C = jnp.zeros_like(A_upper)
    d = jnp.zeros_like(rhs)
    C = C.at[0].set(C0)
    d = d.at[0].set(d0)

    def fwd(i, carry):
        C_local, d_local = carry
        Di = A_diag[i] - A_lower[i] @ C_local[i - 1]
        invDi = inv2(Di)
        Ci = jnp.where(
            i < N - 1,
            invDi @ A_upper[i],
            jnp.zeros((2, 2), dtype=A_diag.dtype),
        )
        di = invDi @ (rhs[i] - A_lower[i] @ d_local[i - 1])
        C_local = C_local.at[i].set(Ci)
        d_local = d_local.at[i].set(di)
        return C_local, d_local

    C, d = jax.lax.fori_loop(1, N, fwd, (C, d))
    x = jnp.zeros_like(rhs)
    x = x.at[N - 1].set(d[N - 1])

    def bwd(k, x_local):
        i = N - 2 - k
        xi = d[i] - C[i] @ x_local[i + 1]
        return x_local.at[i].set(xi)

    x = jax.lax.fori_loop(0, N - 1, bwd, x)
    return x


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
    """Solve a 2x2 block tridiagonal system with diagonal off-blocks.

    The diagonal block at compartment i is:

        [[a00[i], a01[i]],
         [a10[i], a11[i]]]

    The lower and upper off-diagonal blocks are diagonal and share the same
    edge coefficients:

        [[off0[e], 0],
         [0, off1[e]]]

    This matches the double-cable Vi/Ve system without materializing
    ``(Nx, 2, 2)`` block arrays inside the time loop.
    """
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


def solve_block_tridiagonal_2x2_pcr(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Solve the 2x2 block system with parallel cyclic reduction.

    This is a GPU-oriented alternative to
    :func:`solve_block_tridiagonal_2x2_scalar`. It keeps every compartment row
    active at each reduction stage, eliminates neighbors at strides
    ``1, 2, 4, ...``, and finishes with independent 2x2 solves. The algorithm
    does more local arithmetic and materializes full 2x2 off-diagonal blocks,
    but exposes spatial parallelism that the Thomas forward/backward scan does
    not.
    """

    n = int(a00.shape[0])
    dtype = a00.dtype
    idx = jnp.arange(n)
    zero = jnp.zeros((), dtype=dtype)

    lower = jnp.zeros((n, 2, 2), dtype=dtype)
    upper = jnp.zeros((n, 2, 2), dtype=dtype)
    diag = jnp.zeros((n, 2, 2), dtype=dtype)
    rhs = jnp.stack([rhs0, rhs1], axis=1)

    diag = diag.at[:, 0, 0].set(a00)
    diag = diag.at[:, 0, 1].set(a01)
    diag = diag.at[:, 1, 0].set(a10)
    diag = diag.at[:, 1, 1].set(a11)
    lower = lower.at[1:, 0, 0].set(off0)
    lower = lower.at[1:, 1, 1].set(off1)
    upper = upper.at[:-1, 0, 0].set(off0)
    upper = upper.at[:-1, 1, 1].set(off1)

    def inv2_blocks(blocks: Array) -> Array:
        m00 = blocks[:, 0, 0]
        m01 = blocks[:, 0, 1]
        m10 = blocks[:, 1, 0]
        m11 = blocks[:, 1, 1]
        det = m00 * m11 - m01 * m10
        out = jnp.zeros_like(blocks)
        out = out.at[:, 0, 0].set(m11 / det)
        out = out.at[:, 0, 1].set(-m01 / det)
        out = out.at[:, 1, 0].set(-m10 / det)
        out = out.at[:, 1, 1].set(m00 / det)
        return out

    def matmul2(left: Array, right: Array) -> Array:
        l00 = left[:, 0, 0]
        l01 = left[:, 0, 1]
        l10 = left[:, 1, 0]
        l11 = left[:, 1, 1]
        r00 = right[:, 0, 0]
        r01 = right[:, 0, 1]
        r10 = right[:, 1, 0]
        r11 = right[:, 1, 1]
        row0 = jnp.stack((l00 * r00 + l01 * r10, l00 * r01 + l01 * r11), axis=1)
        row1 = jnp.stack((l10 * r00 + l11 * r10, l10 * r01 + l11 * r11), axis=1)
        return jnp.stack((row0, row1), axis=1)

    def matvec2(matrix: Array, vector: Array) -> Array:
        m00 = matrix[:, 0, 0]
        m01 = matrix[:, 0, 1]
        m10 = matrix[:, 1, 0]
        m11 = matrix[:, 1, 1]
        v0 = vector[:, 0]
        v1 = vector[:, 1]
        return jnp.stack((m00 * v0 + m01 * v1, m10 * v0 + m11 * v1), axis=1)

    stride = 1
    while stride < n:
        left_idx = jnp.maximum(idx - stride, 0)
        right_idx = jnp.minimum(idx + stride, n - 1)
        has_left = idx >= stride
        has_right = idx + stride < n

        left_inv = inv2_blocks(diag[left_idx])
        right_inv = inv2_blocks(diag[right_idx])
        left_factor = matmul2(lower, left_inv)
        right_factor = matmul2(upper, right_inv)
        left_factor = jnp.where(has_left[:, None, None], left_factor, zero)
        right_factor = jnp.where(has_right[:, None, None], right_factor, zero)

        next_lower = -matmul2(left_factor, lower[left_idx])
        next_upper = -matmul2(right_factor, upper[right_idx])
        next_diag = (
            diag
            - matmul2(left_factor, upper[left_idx])
            - matmul2(right_factor, lower[right_idx])
        )
        next_rhs = (
            rhs
            - matvec2(left_factor, rhs[left_idx])
            - matvec2(right_factor, rhs[right_idx])
        )

        lower = jnp.where(has_left[:, None, None], next_lower, zero)
        upper = jnp.where(has_right[:, None, None], next_upper, zero)
        diag = next_diag
        rhs = next_rhs
        stride *= 2

    solution = matvec2(inv2_blocks(diag), rhs)
    return solution[:, 0], solution[:, 1]


def apply_diffusion_operator(V: Array, lower: Array, diag: Array, upper: Array) -> Array:
    """
    Apply the discrete diffusion operator represented by the tridiagonal rows.
    """
    Nx = V.shape[0]
    out = jnp.zeros_like(V)

    if Nx >= 2:
        out = out.at[0].set(upper[0] * (V[1] - V[0]))
        out = out.at[-1].set(lower[-1] * (V[-2] - V[-1]))

    if Nx > 2:
        out = out.at[1:-1].set(
            lower[1:-1] * (V[:-2] - V[1:-1])
            + upper[1:-1] * (V[2:] - V[1:-1])
        )

    return out


def build_cn_tridiagonal(
    lower: Array,
    diag: Array,
    upper: Array,
    dt: float,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """
    Build the tridiagonal matrix for the Crank-Nicolson solve.

    This returns the diagonals of:
        A = I - (dt / 2) * L
    where L is the discrete diffusion operator.
    """
    dt_local = dtype_local(dt)
    dl = -0.5 * dt_local * lower
    d = jnp.ones_like(diag, dtype=dtype_local) - 0.5 * dt_local * diag
    du = -0.5 * dt_local * upper

    return dl, d, du


def build_dense_from_tridiagonal(
    dl: Array,
    d: Array,
    du: Array,
    dtype_local: jnp.dtype,
) -> Array:
    """
    Materialize a dense matrix from tridiagonal coefficients.
    """
    Nx = d.shape[0]
    A = jnp.zeros((Nx, Nx), dtype=dtype_local)
    idx = jnp.arange(Nx)
    A = A.at[idx, idx].set(d)
    A = A.at[idx[1:], idx[:-1]].set(dl[1:])
    A = A.at[idx[:-1], idx[1:]].set(du[:-1])
    return A
