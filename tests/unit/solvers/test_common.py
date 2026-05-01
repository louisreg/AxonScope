from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from axonscope.solvers.common import apply_diffusion_operator, diffusion_operator_coeffs


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
