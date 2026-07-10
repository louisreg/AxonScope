from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp

from axonscope.solvers.axon_runtime import SolverAxon

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Array: TypeAlias = jnp.ndarray
Carry: TypeAlias = tuple[Array, Array]  # generic (V, gates) carry used by scan


class DoubleCableLinearSystem(NamedTuple):
    """Batch-first SoA coefficients/RHS for an exact double-cable solve."""

    a00: Array
    a01: Array
    a10: Array
    a11: Array
    off0: Array
    off1: Array
    rhs0: Array
    rhs1: Array


class DoubleCableLinearSystemXB(NamedTuple):
    """Node-first SoA coefficients/RHS for an exact double-cable solve."""

    a00: Array
    a01: Array
    a10: Array
    a11: Array
    off0: Array
    off1: Array
    rhs0: Array
    rhs1: Array


class DoubleCableLinearSystemStaticTerms(NamedTuple):
    """Static-ish batch terms used to assemble a double-cable linear system."""

    area: Array
    cm_over_dt: Array
    cx_over_dt: Array
    cx_plus_gx: Array
    a00_static: Array
    a11_static: Array
    off_i: Array
    off_e: Array
    background_abs: Array
    zero_abs: Array


class DoubleCableLinearSystemStaticTermsXB(NamedTuple):
    """Static-ish node-first terms used to assemble a double-cable system."""

    area: Array
    cm_over_dt: Array
    cx_over_dt: Array
    cx_plus_gx: Array
    a00_static: Array
    a11_static: Array
    off_i: Array
    off_e: Array
    background_abs: Array
    zero_abs: Array


def batch_double_cable_space(values: Array, *, batch_size: int, nx: int) -> Array:
    """Broadcast scalar or space-only values to batch-first double-cable space."""

    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (batch_size, nx))
    if arr.ndim == 1:
        return jnp.broadcast_to(arr[None, :], (batch_size, nx))
    return arr


def double_cable_space_to_xb(values: Array, *, batch_size: int, nx: int) -> Array:
    """Broadcast or transpose double-cable space values to node-first layout."""

    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (nx, batch_size))
    if arr.ndim == 1:
        if int(arr.shape[0]) != nx:
            raise ValueError(f"Expected space axis of length {nx}, got {arr.shape}.")
        return jnp.broadcast_to(arr[:, None], (nx, batch_size))
    if arr.ndim == 2:
        if tuple(arr.shape) == (nx, batch_size):
            return arr
        if tuple(arr.shape) == (batch_size, nx):
            return jnp.swapaxes(arr, 0, 1)
        if arr.shape[0] in (1, batch_size) and arr.shape[1] in (1, nx):
            return jnp.swapaxes(jnp.broadcast_to(arr, (batch_size, nx)), 0, 1)
    raise ValueError(
        f"Expected scalar, ({nx},), ({batch_size}, {nx}), "
        f"or ({nx}, {batch_size}); got {arr.shape}."
    )


def double_cable_edge_to_xb(values: Array, *, batch_size: int, nx: int) -> Array:
    """Broadcast or transpose double-cable edge values to node-first layout."""

    edge_count = nx - 1
    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (edge_count, batch_size))
    if arr.ndim == 1:
        if int(arr.shape[0]) != edge_count:
            raise ValueError(
                f"Expected edge axis of length {edge_count}, got {arr.shape}."
            )
        return jnp.broadcast_to(arr[:, None], (edge_count, batch_size))
    if arr.ndim == 2:
        if tuple(arr.shape) == (edge_count, batch_size):
            return arr
        if tuple(arr.shape) == (batch_size, edge_count):
            return jnp.swapaxes(arr, 0, 1)
        if arr.shape[0] in (1, batch_size) and arr.shape[1] in (1, edge_count):
            return jnp.swapaxes(
                jnp.broadcast_to(arr, (batch_size, edge_count)),
                0,
                1,
            )
    raise ValueError(
        f"Expected scalar, ({edge_count},), ({batch_size}, {edge_count}), "
        f"or ({edge_count}, {batch_size}); got {arr.shape}."
    )


def double_cable_space_from_xb(values: Array) -> Array:
    """Convert node-first double-cable space values back to batch-first layout."""

    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"Expected node-first 2D values, got {arr.shape}.")
    return jnp.swapaxes(arr, 0, 1)


def prepare_double_cable_linear_system_static_terms(
    *,
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    dt_ms: Array,
    batch_size: int,
    nx: int,
) -> DoubleCableLinearSystemStaticTerms:
    """Prepare reusable batch-first terms for double-cable system assembly."""

    area = batch_double_cable_space(area_cm2, batch_size=batch_size, nx=nx)
    cm_over_dt = batch_double_cable_space(Cm_abs, batch_size=batch_size, nx=nx) / dt_ms
    cx_over_dt = batch_double_cable_space(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    Gx_abs_batch = batch_double_cable_space(Gx_abs, batch_size=batch_size, nx=nx)
    left_i_batch = batch_double_cable_space(left_i, batch_size=batch_size, nx=nx)
    right_i_batch = batch_double_cable_space(right_i, batch_size=batch_size, nx=nx)
    left_e_batch = batch_double_cable_space(left_e, batch_size=batch_size, nx=nx)
    right_e_batch = batch_double_cable_space(right_e, batch_size=batch_size, nx=nx)
    background_batch = batch_double_cable_space(I_background, batch_size=batch_size, nx=nx)

    return DoubleCableLinearSystemStaticTerms(
        area=area,
        cm_over_dt=cm_over_dt,
        cx_over_dt=cx_over_dt,
        cx_plus_gx=cx_over_dt + Gx_abs_batch,
        a00_static=cm_over_dt + left_i_batch + right_i_batch,
        a11_static=cm_over_dt + cx_over_dt + Gx_abs_batch + left_e_batch + right_e_batch,
        off_i=-jnp.asarray(Gax_i),
        off_e=-jnp.asarray(Gax_e),
        background_abs=background_batch * area,
        zero_abs=jnp.zeros_like(area),
    )


def prepare_double_cable_linear_system_static_terms_xb(
    *,
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    dt_ms: Array,
    batch_size: int,
    nx: int,
) -> DoubleCableLinearSystemStaticTermsXB:
    """Prepare reusable node-first terms for double-cable system assembly."""

    area = double_cable_space_to_xb(area_cm2, batch_size=batch_size, nx=nx)
    cm_over_dt = (
        double_cable_space_to_xb(Cm_abs, batch_size=batch_size, nx=nx) / dt_ms
    )
    cx_over_dt = (
        double_cable_space_to_xb(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    )
    Gx_abs_batch = double_cable_space_to_xb(Gx_abs, batch_size=batch_size, nx=nx)
    left_i_batch = double_cable_space_to_xb(left_i, batch_size=batch_size, nx=nx)
    right_i_batch = double_cable_space_to_xb(right_i, batch_size=batch_size, nx=nx)
    left_e_batch = double_cable_space_to_xb(left_e, batch_size=batch_size, nx=nx)
    right_e_batch = double_cable_space_to_xb(right_e, batch_size=batch_size, nx=nx)
    background_batch = double_cable_space_to_xb(
        I_background,
        batch_size=batch_size,
        nx=nx,
    )

    return DoubleCableLinearSystemStaticTermsXB(
        area=area,
        cm_over_dt=cm_over_dt,
        cx_over_dt=cx_over_dt,
        cx_plus_gx=cx_over_dt + Gx_abs_batch,
        a00_static=cm_over_dt + left_i_batch + right_i_batch,
        a11_static=(
            cm_over_dt + cx_over_dt + Gx_abs_batch + left_e_batch + right_e_batch
        ),
        off_i=-double_cable_edge_to_xb(Gax_i, batch_size=batch_size, nx=nx),
        off_e=-double_cable_edge_to_xb(Gax_e, batch_size=batch_size, nx=nx),
        background_abs=background_batch * area,
        zero_abs=jnp.zeros_like(area),
    )


def assemble_double_cable_linear_system(
    *,
    Vi: Array,
    Ve: Array,
    Gm_abs: Array,
    GE_abs: Array,
    static: DoubleCableLinearSystemStaticTerms,
    Iinj_abs: Array,
    I_outward_abs: Array,
    I_corr_abs: Array,
    extracellular_drive_abs: Array,
) -> DoubleCableLinearSystem:
    """Assemble the exact double-cable SoA block-tridiagonal system."""

    Vm = Vi - Ve
    cm_plus_gm = static.cm_over_dt + Gm_abs
    membrane_charge = static.cm_over_dt * Vm
    a01 = -cm_plus_gm
    return DoubleCableLinearSystem(
        a00=static.a00_static + Gm_abs,
        a01=a01,
        a10=a01,
        a11=static.a11_static + Gm_abs,
        off0=static.off_i,
        off1=static.off_e,
        rhs0=membrane_charge + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs,
        rhs1=(
            -membrane_charge
            - GE_abs
            + static.cx_over_dt * Ve
            + extracellular_drive_abs
            + I_outward_abs
            + I_corr_abs
        ),
    )


def assemble_double_cable_linear_system_xb(
    *,
    Vi: Array,
    Ve: Array,
    Gm_abs: Array,
    GE_abs: Array,
    static: DoubleCableLinearSystemStaticTermsXB,
    Iinj_abs: Array,
    I_outward_abs: Array,
    I_corr_abs: Array,
    extracellular_drive_abs: Array,
) -> DoubleCableLinearSystemXB:
    """Assemble the exact double-cable SoA system in node-first layout."""

    Vm = Vi - Ve
    cm_plus_gm = static.cm_over_dt + Gm_abs
    membrane_charge = static.cm_over_dt * Vm
    a01 = -cm_plus_gm
    return DoubleCableLinearSystemXB(
        a00=static.a00_static + Gm_abs,
        a01=a01,
        a10=a01,
        a11=static.a11_static + Gm_abs,
        off0=static.off_i,
        off1=static.off_e,
        rhs0=membrane_charge + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs,
        rhs1=(
            -membrane_charge
            - GE_abs
            + static.cx_over_dt * Ve
            + extracellular_drive_abs
            + I_outward_abs
            + I_corr_abs
        ),
    )


def solve_double_cable_linear_system_pcr_soa_batched(
    system: DoubleCableLinearSystem,
) -> tuple[Array, Array]:
    """Solve a batch-first double-cable linear system with the PCR/SoA route."""

    return solve_block_tridiagonal_2x2_pcr_soa_batched(*system)


def solve_double_cable_linear_system_jax_triton_loop_xb(
    system: DoubleCableLinearSystemXB,
    *,
    block_b: int = 32,
) -> tuple[Array, Array]:
    """Solve a node-first double-cable system with the private jax-triton route."""

    from .jax_triton_double_cable import (
        solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb,
    )

    Vi_xb, Ve_xb = solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb(
        *system,
        block_b=block_b,
    )
    return double_cable_space_from_xb(Vi_xb), double_cable_space_from_xb(Ve_xb)


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


def solve_block_tridiagonal_2x2_scalar_batched(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Batch-native block-Thomas solve for double-cable 2x2 systems.

    RHS arrays are batch-first ``[B, Nx]``. Coefficients may be shared
    ``[Nx]`` / ``[Nx - 1]`` or batched ``[B, Nx]`` / ``[B, Nx - 1]``. The
    algorithm is the same exact scalarized block Thomas elimination as
    :func:`solve_block_tridiagonal_2x2_scalar`, but scans once over ``Nx`` with
    vector lanes over the batch instead of relying on an outer ``vmap``.
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
    off0_b = as_batched(off0, length=max(n - 1, 0), name="off0")
    off1_b = as_batched(off1, length=max(n - 1, 0), name="off1")

    a00_n = jnp.swapaxes(a00_b, 0, 1)
    a01_n = jnp.swapaxes(a01_b, 0, 1)
    a10_n = jnp.swapaxes(a10_b, 0, 1)
    a11_n = jnp.swapaxes(a11_b, 0, 1)
    off0_n = jnp.swapaxes(off0_b, 0, 1)
    off1_n = jnp.swapaxes(off1_b, 0, 1)
    rhs0_n = jnp.swapaxes(rhs0, 0, 1)
    rhs1_n = jnp.swapaxes(rhs1, 0, 1)

    zero = jnp.zeros((batch_size,), dtype=rhs0.dtype)
    upper0 = jnp.concatenate([off0_n, zero[None, :]], axis=0)
    upper1 = jnp.concatenate([off1_n, zero[None, :]], axis=0)

    def inv_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    inv00, inv01, inv10, inv11 = inv_components(
        a00_n[0],
        a01_n[0],
        a10_n[0],
        a11_n[0],
    )
    c00_0 = inv00 * upper0[0]
    c01_0 = inv01 * upper1[0]
    c10_0 = inv10 * upper0[0]
    c11_0 = inv11 * upper1[0]
    d0_0 = inv00 * rhs0_n[0] + inv01 * rhs1_n[0]
    d1_0 = inv10 * rhs0_n[0] + inv11 * rhs1_n[0]

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
            a00_n[1:],
            a01_n[1:],
            a10_n[1:],
            a11_n[1:],
            off0_n,
            off1_n,
            upper0[1:],
            upper1[1:],
            rhs0_n[1:],
            rhs1_n[1:],
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
    x0_n = jnp.concatenate([x0_rev[::-1], x0_last[None, :]], axis=0)
    x1_n = jnp.concatenate([x1_rev[::-1], x1_last[None, :]], axis=0)
    return jnp.swapaxes(x0_n, 0, 1), jnp.swapaxes(x1_n, 0, 1)


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
    """Solve the 2x2 block system with matrix-layout parallel cyclic reduction.

    This is a GPU-oriented alternative to
    :func:`solve_block_tridiagonal_2x2_scalar`. It keeps every compartment row
    active at each reduction stage, eliminates neighbors at strides
    ``1, 2, 4, ...``, and finishes with independent 2x2 solves. The tiny block
    products are scalarized manually so XLA does not lower them to GEMM/dot
    kernels, while the block arrays remain in ``(N, 2, 2)`` layout.
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


def solve_block_tridiagonal_2x2_pcr_soa(
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
    does more local arithmetic, but exposes spatial parallelism that the Thomas
    forward/backward scan does not. The implementation keeps the ``2x2`` block
    components in separate arrays so XLA does not lower tiny block products to
    many small GEMM/dot kernels.
    """

    n = int(a00.shape[0])
    dtype = a00.dtype
    idx = jnp.arange(n)
    zero = jnp.zeros((), dtype=dtype)

    lower00 = jnp.concatenate([zero[None], off0])
    lower01 = jnp.zeros((n,), dtype=dtype)
    lower10 = jnp.zeros((n,), dtype=dtype)
    lower11 = jnp.concatenate([zero[None], off1])
    upper00 = jnp.concatenate([off0, zero[None]])
    upper01 = jnp.zeros((n,), dtype=dtype)
    upper10 = jnp.zeros((n,), dtype=dtype)
    upper11 = jnp.concatenate([off1, zero[None]])
    diag00 = a00
    diag01 = a01
    diag10 = a10
    diag11 = a11
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
        has_left = idx >= stride
        has_right = idx + stride < n

        left_inv = inv2_components(
            diag00[left_idx],
            diag01[left_idx],
            diag10[left_idx],
            diag11[left_idx],
        )
        right_inv = inv2_components(
            diag00[right_idx],
            diag01[right_idx],
            diag10[right_idx],
            diag11[right_idx],
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
            lower00[left_idx],
            lower01[left_idx],
            lower10[left_idx],
            lower11[left_idx],
        )
        nu00, nu01, nu10, nu11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            upper00[right_idx],
            upper01[right_idx],
            upper10[right_idx],
            upper11[right_idx],
        )
        ldu00, ldu01, ldu10, ldu11 = matmul2_components(
            lf00,
            lf01,
            lf10,
            lf11,
            upper00[left_idx],
            upper01[left_idx],
            upper10[left_idx],
            upper11[left_idx],
        )
        rdl00, rdl01, rdl10, rdl11 = matmul2_components(
            rf00,
            rf01,
            rf10,
            rf11,
            lower00[right_idx],
            lower01[right_idx],
            lower10[right_idx],
            lower11[right_idx],
        )
        lr0, lr1 = matvec2_components(lf00, lf01, lf10, lf11, r0[left_idx], r1[left_idx])
        rr0, rr1 = matvec2_components(rf00, rf01, rf10, rf11, r0[right_idx], r1[right_idx])

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
    return matvec2_components(inv00, inv01, inv10, inv11, r0, r1)


def solve_block_tridiagonal_2x2_pcr_soa_batched(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Batch-native SoA PCR solve for exact double-cable 2x2 systems.

    Coefficients may be shared across the batch with shape ``[Nx]`` /
    ``[Nx - 1]`` or already materialized per row with shape ``[B, Nx]`` /
    ``[B, Nx - 1]``. Right-hand sides are batch-first ``[B, Nx]`` arrays.

    This is the Phase 1C GPU path from the double-cable solver roadmap: the
    cyclic-reduction stages operate directly over the ``B x Nx`` grid instead
    of relying on an outer ``vmap`` over one-fiber solves.
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

    inv00, inv01, inv10, inv11 = inv2_components(diag00, diag01, diag10, diag11)
    return matvec2_components(inv00, inv01, inv10, inv11, r0, r1)


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
