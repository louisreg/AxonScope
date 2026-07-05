from __future__ import annotations

import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve
import numpy as np

from axonscope.backends.jax.common import solve_block_tridiagonal_2x2_scalar
from benchmark.pseudo_double.reductions import (
    DoubleCableBlockCoefficients,
    double_cable_coefficients_from_solver_terms,
    schur_local_v1,
    series_equivalent,
    tridiagonal_edges_to_jax,
)


def test_series_equivalent_is_bounded_by_inputs():
    ax = jnp.asarray([1.0, 2.0, 4.0])
    my = jnp.asarray([3.0, 2.0, 1.0])

    equivalent = np.asarray(series_equivalent(ax, my))

    assert np.all(np.isfinite(equivalent))
    assert np.all(equivalent > 0.0)
    assert np.all(equivalent <= np.minimum(np.asarray(ax), np.asarray(my)))


def test_local_schur_matches_exact_when_eliminated_block_is_diagonal():
    coeffs = DoubleCableBlockCoefficients(
        aii_lower=jnp.asarray([-0.2, -0.15, -0.1]),
        aii_diag=jnp.asarray([3.2, 3.5, 3.4, 3.1]),
        aii_upper=jnp.asarray([-0.2, -0.15, -0.1]),
        app_lower=jnp.zeros((3,)),
        app_diag=jnp.asarray([4.1, 4.3, 4.2, 4.4]),
        app_upper=jnp.zeros((3,)),
        aip_diag=jnp.asarray([-0.8, -0.9, -0.7, -0.6]),
        api_diag=jnp.asarray([-0.8, -0.9, -0.7, -0.6]),
    )
    rhs_i = jnp.asarray([1.0, -0.5, 0.25, 0.75])
    rhs_p = jnp.asarray([0.2, -0.1, 0.3, -0.4])

    exact_i, _ = solve_block_tridiagonal_2x2_scalar(
        coeffs.aii_diag,
        coeffs.aip_diag,
        coeffs.api_diag,
        coeffs.app_diag,
        coeffs.aii_upper,
        coeffs.app_upper,
        rhs_i,
        rhs_p,
    )
    reduced = schur_local_v1(coeffs, rhs_i, rhs_p)
    dl, d, du = tridiagonal_edges_to_jax(reduced.lower, reduced.diag, reduced.upper)
    schur_i = tridiagonal_solve(dl, d, du, reduced.rhs[:, None])[:, 0]

    np.testing.assert_allclose(np.asarray(schur_i), np.asarray(exact_i), rtol=1e-6, atol=1e-6)


def test_solver_term_coefficients_match_double_cable_diagonal_convention():
    cm_over_dt = jnp.asarray([10.0, 11.0, 12.0])
    cx_over_dt = jnp.asarray([2.0, 3.0, 4.0])
    gm_abs = jnp.asarray([0.5, 0.6, 0.7])
    gx_abs = jnp.asarray([0.1, 0.2, 0.3])
    gax_i = jnp.asarray([1.0, 1.5])
    gax_e = jnp.asarray([0.2, 0.3])
    left_i = jnp.asarray([0.0, 1.0, 1.5])
    right_i = jnp.asarray([1.0, 1.5, 0.0])
    left_e = jnp.asarray([0.0, 0.2, 0.3])
    right_e = jnp.asarray([0.2, 0.3, 0.0])

    coeffs = double_cable_coefficients_from_solver_terms(
        cm_over_dt=cm_over_dt,
        cx_over_dt=cx_over_dt,
        gm_abs=gm_abs,
        gx_abs=gx_abs,
        gax_i=gax_i,
        gax_e=gax_e,
        left_i=left_i,
        right_i=right_i,
        left_e=left_e,
        right_e=right_e,
    )

    np.testing.assert_allclose(
        np.asarray(coeffs.aii_diag),
        np.asarray(cm_over_dt + gm_abs + left_i + right_i),
    )
    np.testing.assert_allclose(np.asarray(coeffs.aip_diag), np.asarray(-(cm_over_dt + gm_abs)))
    np.testing.assert_allclose(
        np.asarray(coeffs.app_diag),
        np.asarray(cm_over_dt + gm_abs + cx_over_dt + gx_abs + left_e + right_e),
    )
    np.testing.assert_allclose(np.asarray(coeffs.aii_lower), np.asarray(-gax_i))
    np.testing.assert_allclose(np.asarray(coeffs.app_upper), np.asarray(-gax_e))


def test_local_schur_rejects_effectively_zero_app_diagonal_by_regularizing():
    coeffs = DoubleCableBlockCoefficients(
        aii_lower=jnp.asarray([-0.1]),
        aii_diag=jnp.asarray([1.0, 1.0]),
        aii_upper=jnp.asarray([-0.1]),
        app_lower=jnp.zeros((1,)),
        app_diag=jnp.asarray([0.0, 2.0]),
        app_upper=jnp.zeros((1,)),
        aip_diag=jnp.asarray([-0.2, -0.2]),
        api_diag=jnp.asarray([-0.2, -0.2]),
    )

    reduced = schur_local_v1(coeffs, jnp.ones((2,)), jnp.ones((2,)), eps=1e-6)

    assert np.all(np.isfinite(np.asarray(reduced.diag)))
    assert np.all(np.isfinite(np.asarray(reduced.rhs)))
