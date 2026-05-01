from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from axonscope.solvers.common import (
    apply_diffusion_operator,
    diffusion_operator_coeffs,
    solve_block_tridiagonal_2x2,
    solve_block_tridiagonal_2x2_scalar,
)


def test_non_uniform_diffusion_operator_matches_quadratic_second_derivative():
    """
    On V(x) = x^2, the discrete non-uniform operator should recover d2V/dx2 = 2.
    """
    x_um = jnp.array([0.0, 20.0, 55.0, 90.0, 150.0], dtype=jnp.float32)
    x_cm = x_um * 1e-4
    D = 0.3

    axon = SimpleNamespace(
        Nx=x_um.shape[0],
        x=x_um,
        h_cm=jnp.diff(x_um) * 1e-4,
        D=D,
    )

    lower, diag, upper = diffusion_operator_coeffs(axon, jnp.float32)
    V = x_cm ** 2
    diffusion = apply_diffusion_operator(V, lower, diag, upper)
    np.testing.assert_allclose(np.asarray(diffusion)[1:-1], 2.0 * D, atol=2e-5, rtol=0.0)


def test_sealed_end_diffusion_operator_keeps_constant_profile_constant():
    """A constant voltage profile must remain diffusion-free everywhere."""
    x_um = jnp.array([0.0, 20.0, 55.0, 90.0, 150.0], dtype=jnp.float32)
    V = jnp.full_like(x_um, -67.5)

    axon = SimpleNamespace(
        Nx=x_um.shape[0],
        x=x_um,
        h_cm=jnp.diff(x_um) * 1e-4,
        D=0.3,
    )

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
