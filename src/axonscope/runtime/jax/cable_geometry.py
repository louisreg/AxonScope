from __future__ import annotations

from typing import TypeAlias

import jax.numpy as jnp

from axonscope.runtime.solver_axon import SolverAxon

Array: TypeAlias = jnp.ndarray


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
