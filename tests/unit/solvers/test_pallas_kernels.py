import jax
import jax.numpy as jnp
import numpy as np

from axonscope.solvers.common import solve_block_tridiagonal_2x2_scalar
from axonscope.solvers.pallas_kernels import (
    solve_block_tridiagonal_2x2_pallas_thomas_batched,
)


def test_pallas_thomas_matches_vmapped_thomas_in_interpret_mode():
    batch_size = 16
    n = 8
    batch = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    x = jnp.arange(n, dtype=jnp.float32)[None, :]
    edge = jnp.arange(n - 1, dtype=jnp.float32)[None, :]

    a00 = 4.0 + 0.05 * x + 0.01 * batch
    a01 = -0.9 - 0.01 * x + 0.002 * batch
    a10 = -1.1 + 0.02 * x - 0.003 * batch
    a11 = 5.0 + 0.07 * x + 0.008 * batch
    off0 = -0.10 - 0.01 * edge - 0.001 * batch
    off1 = -0.07 - 0.005 * edge - 0.0015 * batch
    rhs0 = jnp.sin(0.3 * x + 0.2 * batch)
    rhs1 = jnp.cos(0.2 * x - 0.1 * batch)

    thomas0, thomas1 = jax.vmap(solve_block_tridiagonal_2x2_scalar)(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )
    pallas0, pallas1 = solve_block_tridiagonal_2x2_pallas_thomas_batched(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        block_b=16,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(pallas0), np.asarray(thomas0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(pallas1), np.asarray(thomas1), rtol=1e-5, atol=1e-6)
