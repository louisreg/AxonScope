from types import SimpleNamespace

import jax
import numpy as np
import jax.numpy as jnp

from axonscope.solvers.common import (
    apply_diffusion_operator,
    diffusion_operator_coeffs,
    solve_block_tridiagonal_2x2,
    solve_block_tridiagonal_2x2_pcr,
    solve_block_tridiagonal_2x2_scalar,
)


def _uniform_cable_namespace(*, x_um, diffusion_coefficient: float) -> SimpleNamespace:
    diameter_um = 1.0
    Cm_uF_cm2 = 1.0
    radius_cm = 0.5 * diameter_um * 1e-4
    Ra_ohm_cm = radius_cm / (2.0 * diffusion_coefficient * Cm_uF_cm2 * 1e-6 * 1000.0)
    return SimpleNamespace(
        n_compartments=x_um.shape[0],
        h_cm=jnp.diff(x_um) * 1e-4,
        diam_um=jnp.full(x_um.shape, diameter_um, dtype=x_um.dtype),
        Ra_ohm_cm=jnp.full(x_um.shape, Ra_ohm_cm, dtype=x_um.dtype),
        Cm_uF_cm2=jnp.full(x_um.shape, Cm_uF_cm2, dtype=x_um.dtype),
        has_heterogeneous_cable_properties=False,
    )


def test_non_uniform_diffusion_operator_matches_quadratic_second_derivative():
    """
    On V(x) = x^2, the discrete non-uniform operator should recover d2V/dx2 = 2.
    """
    x_um = jnp.array([0.0, 20.0, 55.0, 90.0, 150.0], dtype=jnp.float32)
    x_cm = x_um * 1e-4
    D = 0.3

    axon = _uniform_cable_namespace(x_um=x_um, diffusion_coefficient=D)

    lower, diag, upper = diffusion_operator_coeffs(axon, jnp.float32)
    V = x_cm ** 2
    diffusion = apply_diffusion_operator(V, lower, diag, upper)
    np.testing.assert_allclose(np.asarray(diffusion)[1:-1], 2.0 * D, atol=2e-5, rtol=0.0)


def test_sealed_end_diffusion_operator_keeps_constant_profile_constant():
    """A constant voltage profile must remain diffusion-free everywhere."""
    x_um = jnp.array([0.0, 20.0, 55.0, 90.0, 150.0], dtype=jnp.float32)
    V = jnp.full_like(x_um, -67.5)

    axon = _uniform_cable_namespace(x_um=x_um, diffusion_coefficient=0.3)

    lower, diag, upper = diffusion_operator_coeffs(axon, jnp.float32)
    diffusion = apply_diffusion_operator(V, lower, diag, upper)
    np.testing.assert_allclose(np.asarray(diffusion), 0.0, atol=1e-7, rtol=0.0)


def test_scalar_block_tridiagonal_solver_matches_generic_2x2_solver():
    a00 = jnp.asarray([4.0, 4.2, 4.1, 4.3], dtype=jnp.float32)
    a01 = jnp.asarray([-1.1, -1.0, -1.2, -1.1], dtype=jnp.float32)
    a10 = jnp.asarray([-1.1, -1.0, -1.2, -1.1], dtype=jnp.float32)
    a11 = jnp.asarray([5.0, 5.1, 5.2, 5.3], dtype=jnp.float32)
    off0 = jnp.asarray([-0.15, -0.20, -0.18], dtype=jnp.float32)
    off1 = jnp.asarray([-0.05, -0.07, -0.06], dtype=jnp.float32)
    rhs0 = jnp.asarray([1.0, -0.5, 0.25, 0.75], dtype=jnp.float32)
    rhs1 = jnp.asarray([-0.2, 0.4, 0.8, -0.1], dtype=jnp.float32)

    N = a00.shape[0]
    A_diag = jnp.zeros((N, 2, 2), dtype=jnp.float32)
    A_lower = jnp.zeros((N, 2, 2), dtype=jnp.float32)
    A_upper = jnp.zeros((N, 2, 2), dtype=jnp.float32)
    A_diag = A_diag.at[:, 0, 0].set(a00)
    A_diag = A_diag.at[:, 0, 1].set(a01)
    A_diag = A_diag.at[:, 1, 0].set(a10)
    A_diag = A_diag.at[:, 1, 1].set(a11)
    A_lower = A_lower.at[1:, 0, 0].set(off0)
    A_lower = A_lower.at[1:, 1, 1].set(off1)
    A_upper = A_upper.at[:-1, 0, 0].set(off0)
    A_upper = A_upper.at[:-1, 1, 1].set(off1)
    rhs = jnp.stack([rhs0, rhs1], axis=1)

    generic = solve_block_tridiagonal_2x2(A_lower, A_diag, A_upper, rhs)
    scalar0, scalar1 = solve_block_tridiagonal_2x2_scalar(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )
    scalar = jnp.stack([scalar0, scalar1], axis=1)

    np.testing.assert_allclose(np.asarray(scalar), np.asarray(generic), rtol=1e-6, atol=1e-6)


def test_pcr_block_tridiagonal_solver_matches_thomas_for_non_power_of_two_size():
    n = 7
    x = jnp.arange(n, dtype=jnp.float32)
    a00 = 4.0 + 0.05 * x
    a01 = -0.9 - 0.01 * x
    a10 = -1.1 + 0.02 * x
    a11 = 5.0 + 0.07 * x
    off0 = -0.10 - 0.01 * jnp.arange(n - 1, dtype=jnp.float32)
    off1 = -0.07 - 0.005 * jnp.arange(n - 1, dtype=jnp.float32)
    rhs0 = jnp.sin(0.3 * x)
    rhs1 = jnp.cos(0.2 * x)

    thomas0, thomas1 = solve_block_tridiagonal_2x2_scalar(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )
    pcr0, pcr1 = solve_block_tridiagonal_2x2_pcr(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )

    np.testing.assert_allclose(np.asarray(pcr0), np.asarray(thomas0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pcr1), np.asarray(thomas1), rtol=1e-5, atol=1e-6)


def test_scalar_block_tridiagonal_solver_handles_single_row_under_jit():
    solve = jax.jit(solve_block_tridiagonal_2x2_scalar)

    scalar0, scalar1 = solve(
        jnp.asarray([4.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
        jnp.asarray([5.0], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([2.0], dtype=jnp.float32),
        jnp.asarray([3.0], dtype=jnp.float32),
    )

    expected = np.linalg.solve(
        np.asarray([[4.0, -1.0], [-1.0, 5.0]], dtype=np.float32),
        np.asarray([2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray([scalar0[0], scalar1[0]]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_pcr_block_tridiagonal_solver_handles_single_row_under_jit():
    solve = jax.jit(solve_block_tridiagonal_2x2_pcr)

    scalar0, scalar1 = solve(
        jnp.asarray([4.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
        jnp.asarray([-1.0], dtype=jnp.float32),
        jnp.asarray([5.0], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([], dtype=jnp.float32),
        jnp.asarray([2.0], dtype=jnp.float32),
        jnp.asarray([3.0], dtype=jnp.float32),
    )

    expected = np.linalg.solve(
        np.asarray([[4.0, -1.0], [-1.0, 5.0]], dtype=np.float32),
        np.asarray([2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray([scalar0[0], scalar1[0]]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
