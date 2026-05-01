from axonscope.axons.base import AxonBase

import jax.numpy as jnp
import jax

from typing import Tuple

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Array = jnp.ndarray
Carry = Tuple[Array, Array]  # generic (V, gates) carry used by scan


def initial_voltage(axon: AxonBase, Nx: int, dtype_local: jnp.dtype) -> Array:
    """Initial voltage state."""
    return jnp.full((Nx,), axon.Vinit, dtype=dtype_local)


def diffusion_operator_coeffs(axon: AxonBase, dtype_local: jnp.dtype) -> Tuple[Array, Array, Array]:
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
        ra_ohm_cm = jnp.asarray(getattr(axon, "Ra_vec"), dtype=dtype_local)
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

        Nx = axon.Nx
        lower = jnp.zeros((Nx,), dtype=dtype_local)
        diag = jnp.zeros((Nx,), dtype=dtype_local)
        upper = jnp.zeros((Nx,), dtype=dtype_local)
        lower = lower.at[1:].set(gax_i_mS / cm_abs_uF[1:])
        upper = upper.at[:-1].set(gax_i_mS / cm_abs_uF[:-1])
        diag = -(lower + upper)
        return lower, diag, upper

    Nx = axon.Nx
    lower = jnp.zeros((Nx,), dtype=dtype_local)
    diag = jnp.zeros((Nx,), dtype=dtype_local)
    upper = jnp.zeros((Nx,), dtype=dtype_local)

    if hasattr(axon, "h_cm"):
        h = jnp.asarray(axon.h_cm, dtype=dtype_local)
    elif hasattr(axon, "x"):
        h = jnp.diff(jnp.asarray(axon.x, dtype=dtype_local)) * dtype_local(1e-4)
    else:
        dx_cm = jnp.asarray(axon.dx_cm, dtype=dtype_local)
        h = jnp.full((Nx - 1,), dx_cm[0], dtype=dtype_local)

    D = dtype_local(axon.D)

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


def compartment_diam_um(axon: AxonBase, dtype_local: jnp.dtype) -> Array:
    if hasattr(axon, "diam_vec"):
        return jnp.asarray(axon.diam_vec, dtype=dtype_local)
    return jnp.full((axon.Nx,), dtype_local(axon.d), dtype=dtype_local)


def compartment_cm_uF_cm2(axon: AxonBase, dtype_local: jnp.dtype) -> Array:
    if hasattr(axon, "Cm_vec"):
        return jnp.asarray(axon.Cm_vec, dtype=dtype_local)
    return jnp.full((axon.Nx,), dtype_local(axon.Cm), dtype=dtype_local)


def compartment_length_cm(axon: AxonBase, dtype_local: jnp.dtype) -> Array:
    if hasattr(axon, "compartment_lengths_um"):
        return jnp.asarray(axon.compartment_lengths_um, dtype=dtype_local) * dtype_local(1e-4)
    return jnp.asarray(axon.dx_cm, dtype=dtype_local)


def compartment_area_cm2(axon: AxonBase, dtype_local: jnp.dtype) -> Array:
    diam = compartment_diam_um(axon, dtype_local)
    length_cm = compartment_length_cm(axon, dtype_local)
    return jnp.pi * (diam * dtype_local(1e-4)) * length_cm


def extracellular_absolute_arrays(
    axon: AxonBase, dtype_local: jnp.dtype
) -> Tuple[Array, Array, Array, Array]:
    """Return (Cm_abs, Cx_abs, Gx_abs, Gax_e) in (uF, uF, mS, mS-edge)."""
    area = compartment_area_cm2(axon, dtype_local)
    cm_uF_cm2 = compartment_cm_uF_cm2(axon, dtype_local)
    Cm_abs = cm_uF_cm2 * area

    xg = jnp.asarray(getattr(axon, "xg_vec"), dtype=dtype_local)
    xc = jnp.asarray(getattr(axon, "xc_vec"), dtype=dtype_local)
    xraxial = jnp.asarray(getattr(axon, "xraxial_vec"), dtype=dtype_local)
    dx_cm = jnp.asarray(axon.dx_cm, dtype=dtype_local)

    Cx_abs = xc * area
    Gx_abs = (xg * dtype_local(1e3)) * area

    if axon.Nx <= 1:
        Gax_e = jnp.zeros((0,), dtype=dtype_local)
    else:
        R_edge_MOhm = xraxial[:-1] * (dtype_local(0.5) * dx_cm[:-1]) + xraxial[1:] * (dtype_local(0.5) * dx_cm[1:])
        Gax_e = dtype_local(1e-3) / jnp.maximum(R_edge_MOhm, dtype_local(1e-18))
    return Cm_abs, Cx_abs, Gx_abs, Gax_e


def solve_block_tridiagonal_2x2(A_lower: Array, A_diag: Array, A_upper: Array, rhs: Array) -> Array:
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
        Ci = jnp.where(i < N - 1, invDi @ A_upper[i], jnp.zeros((2, 2), dtype=A_diag.dtype))
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
) -> Tuple[Array, Array, Array]:
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


def build_dense_from_tridiagonal(dl: Array, d: Array, du: Array, dtype_local: jnp.dtype) -> Array:
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
