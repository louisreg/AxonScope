from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from axonfleet.runtime.jax.membranes.kinetics import (
    dense_kinetic_matrix,
    solve_conserved_kinetic_step,
    solve_kinetic_transitions,
    solve_kinetic_step,
)


def _random_valid_transitions(
    *,
    rng: np.random.Generator,
    width: int,
    node_count: int,
    dtype: np.dtype,
) -> list[tuple[int, int, jnp.ndarray]]:
    edges = {(source, (source + 1) % width) for source in range(width)}
    edges.update(((source + 1) % width, source) for source in range(width))
    for source in range(width):
        for target in range(width):
            if source != target and rng.random() < 0.35:
                edges.add((source, target))
    return [
        (
            source,
            target,
            jnp.asarray(rng.uniform(1e-4, 40.0, node_count).astype(dtype)),
        )
        for source, target in sorted(edges)
    ]


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


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("dt_ms", (0.001, 0.0125, 0.05, 0.1))
@pytest.mark.parametrize("width", (2, 3, 6, 9))
def test_matrix_free_conserved_step_matches_dense_on_random_valid_graphs(
    dtype: type[np.floating],
    dt_ms: float,
    width: int,
):
    rng = np.random.default_rng(10_000 + width)
    node_count = 11
    previous = rng.dirichlet(np.ones(width), size=node_count).astype(dtype)
    rtol = 8e-5 if dtype is np.float32 else 2e-12
    atol = 2e-6 if dtype is np.float32 else 2e-13

    with jax.enable_x64(dtype is np.float64):
        jax_dtype = jnp.float64 if dtype is np.float64 else jnp.float32
        transitions = _random_valid_transitions(
            rng=rng,
            width=width,
            node_count=node_count,
            dtype=np.dtype(dtype),
        )
        matrix = dense_kinetic_matrix(
            width=width,
            transitions=transitions,
            node_count=node_count,
            dtype=jax_dtype,
        )
        matrix_free = solve_kinetic_transitions(
            width=width,
            transitions=transitions,
            previous=jnp.asarray(previous),
            dt=jnp.asarray(dt_ms, dtype=jax_dtype),
            node_count=node_count,
            dtype=jax_dtype,
            conserve_probability=True,
        )
        actual = np.asarray(matrix_free)
        system = (
            np.eye(width, dtype=dtype)[None, :, :]
            - np.asarray(dt_ms, dtype=dtype) * np.asarray(matrix)
        )
        expected = np.linalg.solve(system, previous[..., None])[..., 0]
        assert actual.dtype == np.dtype(dtype)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0.0, atol=atol)
        assert np.min(actual) >= -atol
