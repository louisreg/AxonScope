from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from axonscope.runtime.jax.membranes.kinetics import (
    dense_kinetic_matrix,
    solve_conserved_kinetic_step,
    solve_kinetic_transitions,
    solve_kinetic_step,
)


@pytest.mark.parametrize("width", (2, 6, 9))
def test_unrolled_kinetic_step_matches_dense_solve(width: int):
    rng = np.random.default_rng(42 + width)
    node_count = 37
    transitions = []
    for source in range(width):
        target = (source + 1) % width
        transitions.append(
            (source, target, jnp.asarray(rng.uniform(0.01, 20.0, node_count)))
        )
        transitions.append(
            (target, source, jnp.asarray(rng.uniform(0.01, 20.0, node_count)))
        )
    matrix = dense_kinetic_matrix(
        width=width,
        transitions=transitions,
        node_count=node_count,
        dtype=jnp.float32,
    )
    previous = rng.dirichlet(np.ones(width), size=node_count).astype(np.float32)
    dt = np.float32(0.005)

    actual = jax.jit(solve_kinetic_step)(matrix, jnp.asarray(previous), dt)
    system = np.eye(width, dtype=np.float32) - dt * np.asarray(matrix)
    expected = np.linalg.solve(system, previous[..., None])[..., 0]

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(np.asarray(actual).sum(axis=1), 1.0, atol=2e-6)


@pytest.mark.parametrize("width", (2, 6, 9))
def test_reduced_conserved_kinetic_step_matches_dense_solve(width: int):
    rng = np.random.default_rng(142 + width)
    node_count = 37
    transitions = []
    for source in range(width):
        target = (source + 1) % width
        transitions.append(
            (source, target, jnp.asarray(rng.uniform(0.01, 20.0, node_count)))
        )
        transitions.append(
            (target, source, jnp.asarray(rng.uniform(0.01, 20.0, node_count)))
        )
    matrix = dense_kinetic_matrix(
        width=width,
        transitions=transitions,
        node_count=node_count,
        dtype=jnp.float32,
    )
    previous = rng.dirichlet(np.ones(width), size=node_count).astype(np.float32)
    dt = np.float32(0.005)

    actual = jax.jit(solve_conserved_kinetic_step)(
        matrix,
        jnp.asarray(previous),
        dt,
    )
    direct = jax.jit(
        lambda state: solve_kinetic_transitions(
            width=width,
            transitions=transitions,
            previous=state,
            dt=dt,
            node_count=node_count,
            dtype=jnp.float32,
            conserve_probability=True,
        )
    )(jnp.asarray(previous))
    system = np.eye(width, dtype=np.float32) - dt * np.asarray(matrix)
    expected = np.linalg.solve(system, previous[..., None])[..., 0]

    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(direct, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(np.asarray(actual).sum(axis=1), 1.0, atol=2e-7)
