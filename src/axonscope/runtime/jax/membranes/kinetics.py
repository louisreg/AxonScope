"""Shared JAX lowering helpers for finite-state membrane kinetics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp


def dense_kinetic_matrix(
    *,
    width: int,
    transitions: Iterable[tuple[int, int, Any]],
    node_count: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Materialize a batched transition generator from its static topology."""
    cells = _kinetic_generator_cells(
        width=width,
        transitions=transitions,
        node_count=node_count,
        dtype=dtype,
    )
    return jnp.stack(
        [jnp.stack(row, axis=1) for row in cells],
        axis=1,
    )


def solve_kinetic_transitions(
    *,
    width: int,
    transitions: Iterable[tuple[int, int, Any]],
    previous: jnp.ndarray,
    dt: jnp.ndarray,
    node_count: int,
    dtype: jnp.dtype,
    conserve_probability: bool,
) -> jnp.ndarray:
    """Solve one kinetic step directly from rates without a dense matrix."""
    cells = _kinetic_generator_cells(
        width=width,
        transitions=transitions,
        node_count=node_count,
        dtype=dtype,
    )
    rows = _implicit_system_rows_from_cells(cells, dt, dtype=dtype)
    if conserve_probability:
        return _solve_conserved_rows(rows, previous, dtype=dtype)
    rhs = [previous[:, row] for row in range(width)]
    return jnp.stack(_solve_unrolled(rows, rhs), axis=1)


def _kinetic_generator_cells(
    *,
    width: int,
    transitions: Iterable[tuple[int, int, Any]],
    node_count: int,
    dtype: jnp.dtype,
) -> list[list[jnp.ndarray]]:
    cells: list[list[list[jnp.ndarray]]] = [
        [[] for _ in range(width)] for _ in range(width)
    ]
    for source, target, rate in transitions:
        rate_vector = jnp.broadcast_to(
            jnp.asarray(rate, dtype=dtype),
            (node_count,),
        )
        cells[target][source].append(rate_vector)
        cells[source][source].append(-rate_vector)

    zero = jnp.zeros((node_count,), dtype=dtype)
    return [
        [
            sum(terms[1:], start=terms[0]) if terms else zero
            for terms in row
        ]
        for row in cells
    ]


def solve_kinetic_step(
    matrix: jnp.ndarray,
    previous: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Solve ``(I - dt Q) x = previous`` for a small kinetic generator Q.

    The update system is an M-matrix, so its statically unrolled LU elimination
    does not require pivoting. This avoids dispatching a generic batched dense
    solver for many tiny independent systems.
    """
    rows = _implicit_system_rows(matrix, dt)
    rhs = [previous[:, row] for row in range(matrix.shape[-1])]
    return jnp.stack(_solve_unrolled(rows, rhs), axis=1)


def solve_conserved_kinetic_step(
    matrix: jnp.ndarray,
    previous: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a probability-conserving kinetic step with one state eliminated."""
    width = matrix.shape[-1]
    if width == 1:
        return jnp.ones_like(previous)
    full_rows = _implicit_system_rows(matrix, dt)
    return _solve_conserved_rows(full_rows, previous, dtype=matrix.dtype)


def _solve_conserved_rows(
    full_rows: list[list[jnp.ndarray]],
    previous: jnp.ndarray,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    width = len(full_rows)
    if width == 1:
        return jnp.ones_like(previous)
    eliminated = width - 1
    rows = [
        [
            full_rows[row][column] - full_rows[row][eliminated]
            for column in range(eliminated)
        ]
        for row in range(eliminated)
    ]
    rhs = [
        previous[:, row] - full_rows[row][eliminated]
        for row in range(eliminated)
    ]
    solution = _solve_unrolled(rows, rhs)
    final_state = jnp.asarray(1.0, dtype=dtype)
    for value in solution:
        final_state = final_state - value
    reduced = jnp.stack(solution, axis=1)
    negative_residual = jnp.minimum(
        final_state,
        jnp.asarray(0.0, dtype=dtype),
    )
    dominant = jnp.argmax(reduced, axis=1)
    dominant_mask = jnp.arange(eliminated)[None, :] == dominant[:, None]
    reduced = reduced + negative_residual[:, None] * dominant_mask
    final_state = jnp.maximum(final_state, jnp.asarray(0.0, dtype=dtype))
    return jnp.concatenate((reduced, final_state[:, None]), axis=1)


def _implicit_system_rows(
    matrix: jnp.ndarray,
    dt: jnp.ndarray,
) -> list[list[jnp.ndarray]]:
    cells = [
        [matrix[:, row, column] for column in range(matrix.shape[-1])]
        for row in range(matrix.shape[-1])
    ]
    return _implicit_system_rows_from_cells(cells, dt, dtype=matrix.dtype)


def _implicit_system_rows_from_cells(
    cells: list[list[jnp.ndarray]],
    dt: jnp.ndarray,
    *,
    dtype: jnp.dtype,
) -> list[list[jnp.ndarray]]:
    width = len(cells)
    return [
        [
            (jnp.asarray(1.0, dtype=dtype) if row == column else 0.0)
            - dt * cells[row][column]
            for column in range(width)
        ]
        for row in range(width)
    ]


def _solve_unrolled(
    rows: list[list[jnp.ndarray]],
    rhs: list[jnp.ndarray],
) -> list[jnp.ndarray]:
    width = len(rows)
    for pivot_index in range(width - 1):
        pivot = rows[pivot_index][pivot_index]
        for row_index in range(pivot_index + 1, width):
            factor = rows[row_index][pivot_index] / pivot
            for column_index in range(pivot_index + 1, width):
                rows[row_index][column_index] = (
                    rows[row_index][column_index]
                    - factor * rows[pivot_index][column_index]
                )
            rhs[row_index] = rhs[row_index] - factor * rhs[pivot_index]

    solution: list[jnp.ndarray] = [rhs[0]] * width
    for row_index in range(width - 1, -1, -1):
        residual = rhs[row_index]
        for column_index in range(row_index + 1, width):
            residual = residual - rows[row_index][column_index] * solution[column_index]
        solution[row_index] = residual / rows[row_index][row_index]
    return solution
