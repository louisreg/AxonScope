import jax
import jax.numpy as jnp
import numpy as np

from axonscope.backends.jax.common import solve_block_tridiagonal_2x2_pcr_soa_batched
from benchmark.analysis.double_cable_solver_candidates import (
    solve_block_tridiagonal_2x2_pcr_soa_batched_symmetric,
)


def test_symmetric_pcr_candidate_matches_masked_pcr_for_batched_coefficients():
    batch_size = 4
    n = 13
    batch = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    x = jnp.arange(n, dtype=jnp.float32)[None, :]
    edge = jnp.arange(n - 1, dtype=jnp.float32)[None, :]

    a00 = 4.0 + 0.05 * x + 0.01 * batch
    a01 = -0.9 - 0.01 * x + 0.002 * batch
    a10 = a01
    a11 = 5.0 + 0.07 * x + 0.008 * batch
    off0 = -0.10 - 0.01 * edge - 0.001 * batch
    off1 = -0.07 - 0.005 * edge - 0.0015 * batch
    rhs0 = jnp.sin(0.3 * x + 0.2 * batch)
    rhs1 = jnp.cos(0.2 * x - 0.1 * batch)

    masked0, masked1 = solve_block_tridiagonal_2x2_pcr_soa_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )
    candidate0, candidate1 = jax.jit(solve_block_tridiagonal_2x2_pcr_soa_batched_symmetric)(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )

    np.testing.assert_allclose(np.asarray(candidate0), np.asarray(masked0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(candidate1), np.asarray(masked1), rtol=1e-5, atol=1e-6)


def test_symmetric_pcr_candidate_matches_masked_pcr_for_shared_coefficients():
    batch_size = 3
    n = 17
    x = jnp.arange(n, dtype=jnp.float32)
    batch = jnp.arange(batch_size, dtype=jnp.float32)[:, None]

    a00 = 4.0 + 0.05 * x
    a01 = -0.9 - 0.01 * x
    a10 = a01
    a11 = 5.0 + 0.07 * x
    off0 = -0.10 - 0.01 * jnp.arange(n - 1, dtype=jnp.float32)
    off1 = -0.07 - 0.005 * jnp.arange(n - 1, dtype=jnp.float32)
    rhs0 = jnp.sin(0.3 * x[None, :] + 0.2 * batch)
    rhs1 = jnp.cos(0.2 * x[None, :] - 0.1 * batch)

    masked0, masked1 = solve_block_tridiagonal_2x2_pcr_soa_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )
    candidate0, candidate1 = jax.jit(solve_block_tridiagonal_2x2_pcr_soa_batched_symmetric)(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )

    np.testing.assert_allclose(np.asarray(candidate0), np.asarray(masked0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(candidate1), np.asarray(masked1), rtol=1e-5, atol=1e-6)
