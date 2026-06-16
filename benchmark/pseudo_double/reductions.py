"""Electrical double-to-single reduction helpers for pseudo-double validation.

These helpers implement the coefficient-level ideas from
``ideas/axonscope_double_to_single_electrical_reduction_plan.md``. They are
validation plumbing, not public solver API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


Array = Any


@dataclass(frozen=True)
class DoubleCableBlockCoefficients:
    """Scalar components of one double-cable block-tridiagonal system."""

    aii_lower: Array
    aii_diag: Array
    aii_upper: Array
    app_lower: Array
    app_diag: Array
    app_upper: Array
    aip_diag: Array
    api_diag: Array


@dataclass(frozen=True)
class SchurLocalReduction:
    """Scalar tridiagonal system produced by a local Schur approximation."""

    lower: Array
    diag: Array
    upper: Array
    rhs: Array


def series_equivalent(
    axolemma: Array,
    myelin: Array,
    *,
    eps: float = 1e-12,
) -> Array:
    """Return the local series equivalent of two positive admittances."""

    ax = jnp.asarray(axolemma)
    my = jnp.asarray(myelin, dtype=ax.dtype)
    return (ax * my) / jnp.maximum(ax + my, jnp.asarray(eps, dtype=ax.dtype))


def double_cable_coefficients_from_solver_terms(
    *,
    cm_over_dt: Array,
    cx_over_dt: Array,
    gm_abs: Array,
    gx_abs: Array,
    gax_i: Array,
    gax_e: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
) -> DoubleCableBlockCoefficients:
    """Build block coefficients using the exact solver's sign convention."""

    aii_diag = cm_over_dt + gm_abs + left_i + right_i
    coupling = -(cm_over_dt + gm_abs)
    app_diag = cm_over_dt + gm_abs + cx_over_dt + gx_abs + left_e + right_e
    return DoubleCableBlockCoefficients(
        aii_lower=-jnp.asarray(gax_i),
        aii_diag=aii_diag,
        aii_upper=-jnp.asarray(gax_i),
        app_lower=-jnp.asarray(gax_e),
        app_diag=app_diag,
        app_upper=-jnp.asarray(gax_e),
        aip_diag=coupling,
        api_diag=coupling,
    )


def schur_local_v1(
    coeffs: DoubleCableBlockCoefficients,
    rhs_i: Array,
    rhs_p: Array,
    *,
    eps: float = 1e-12,
    app_inverse_scale: float = 1.0,
) -> SchurLocalReduction:
    """Return the diagonal-App local Schur scalar reduction.

    This is exact when the eliminated periaxonal/myelin block has no spatial
    off-diagonals. With spatial periaxonal coupling it is the v1 approximation
    from the electrical-reduction plan.
    """

    app_diag = jnp.asarray(coeffs.app_diag)
    inv_app = jnp.asarray(app_inverse_scale, dtype=app_diag.dtype) / jnp.where(
        jnp.abs(app_diag) > eps,
        app_diag,
        jnp.sign(app_diag + eps) * eps,
    )
    diag = coeffs.aii_diag - coeffs.aip_diag * inv_app * coeffs.api_diag
    rhs = jnp.asarray(rhs_i) - coeffs.aip_diag * inv_app * jnp.asarray(rhs_p)
    return SchurLocalReduction(
        lower=jnp.asarray(coeffs.aii_lower),
        diag=diag,
        upper=jnp.asarray(coeffs.aii_upper),
        rhs=rhs,
    )


def tridiagonal_edges_to_jax(
    lower_edges: Array,
    diag: Array,
    upper_edges: Array,
) -> tuple[Array, Array, Array]:
    """Convert edge coefficients to JAX tridiagonal_solve arrays."""

    diag = jnp.asarray(diag)
    lower = jnp.asarray(lower_edges, dtype=diag.dtype)
    upper = jnp.asarray(upper_edges, dtype=diag.dtype)
    zero = jnp.zeros((1,), dtype=diag.dtype)
    return (
        jnp.concatenate([zero, lower], axis=0),
        diag,
        jnp.concatenate([upper, zero], axis=0),
    )


__all__ = [
    "DoubleCableBlockCoefficients",
    "SchurLocalReduction",
    "double_cable_coefficients_from_solver_terms",
    "schur_local_v1",
    "series_equivalent",
    "tridiagonal_edges_to_jax",
]
