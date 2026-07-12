from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

import jax

from axonscope.runtime.jax.common import (
    solve_block_tridiagonal_2x2_pcr_soa_batched,
    solve_block_tridiagonal_2x2_scalar,
)
from benchmark.analysis.large_population_solver import (
    block_b_candidates_for_nx_bucket,
    make_large_population_layout_plan,
    pad_large_population_double_cable_system,
    select_large_population_nx_bucket,
    solve_large_population_exact_double_cable_jax,
)


def test_large_population_bucket_selection_uses_dense_short_buckets():
    assert select_large_population_nx_bucket(1) == 32
    assert select_large_population_nx_bucket(32) == 32
    assert select_large_population_nx_bucket(33) == 48
    assert select_large_population_nx_bucket(49) == 64
    assert select_large_population_nx_bucket(65) == 80
    assert select_large_population_nx_bucket(81) == 96
    assert select_large_population_nx_bucket(97) == 128
    assert select_large_population_nx_bucket(129) == 160
    assert select_large_population_nx_bucket(161) == 192
    assert select_large_population_nx_bucket(193) == 256

    with pytest.raises(ValueError, match="exceeds supported"):
        select_large_population_nx_bucket(257)


def test_large_population_block_candidates_depend_on_nx_bucket():
    assert block_b_candidates_for_nx_bucket(32) == (64, 128, 256)
    assert block_b_candidates_for_nx_bucket(64) == (64, 128, 256)
    assert block_b_candidates_for_nx_bucket(80) == (32, 64, 128)
    assert block_b_candidates_for_nx_bucket(128) == (32, 64, 128)
    assert block_b_candidates_for_nx_bucket(160) == (16, 32, 64)


def test_large_population_layout_plan_pads_batch_and_nx_independently():
    plan = make_large_population_layout_plan(batch_size=5, nx_true=33, block_b=4)

    assert plan.batch_size == 5
    assert plan.batch_padded == 8
    assert plan.n_tiles == 2
    assert plan.nx_true == 33
    assert plan.nx_pad == 48
    assert plan.layout == "TILED"


def test_large_population_padding_uses_neutral_rows_for_shared_coefficients():
    plan = make_large_population_layout_plan(
        batch_size=3,
        nx_true=7,
        nx_pad=8,
        block_b=4,
    )
    inputs = _make_system(batch_size=3, n=7, shared=True)

    padded = pad_large_population_double_cable_system(*inputs, plan=plan)

    assert [array.shape for array in padded] == [
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 7),
        (4, 7),
        (4, 8),
        (4, 8),
    ]
    np.testing.assert_allclose(np.asarray(padded[0])[:, -1], np.ones(4))
    np.testing.assert_allclose(np.asarray(padded[3])[:, -1], np.ones(4))
    np.testing.assert_allclose(np.asarray(padded[1])[:, -1], np.zeros(4))
    np.testing.assert_allclose(np.asarray(padded[2])[:, -1], np.zeros(4))
    np.testing.assert_allclose(np.asarray(padded[6])[:, -1], np.zeros(4))
    np.testing.assert_allclose(np.asarray(padded[7])[:, -1], np.zeros(4))


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("layout", ["BX", "XB", "TILED"])
def test_large_population_exact_solver_matches_current_exact_solver(shared: bool, layout: str):
    inputs = _make_system(batch_size=5, n=7, shared=shared)

    expected0, expected1 = solve_block_tridiagonal_2x2_pcr_soa_batched(*inputs)
    actual0, actual1 = solve_large_population_exact_double_cable_jax(
        *inputs,
        nx_pad=8,
        block_b=4,
        layout=layout,  # type: ignore[arg-type]
    )

    np.testing.assert_allclose(np.asarray(actual0), np.asarray(expected0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual1), np.asarray(expected1), rtol=1e-5, atol=1e-6)


def test_large_population_exact_solver_matches_thomas_reference():
    inputs = _make_system(batch_size=4, n=9, shared=False)

    expected0, expected1 = jax.vmap(solve_block_tridiagonal_2x2_scalar)(*inputs)
    actual0, actual1 = solve_large_population_exact_double_cable_jax(
        *inputs,
        nx_pad=16,
        block_b=4,
        layout="TILED",
    )

    np.testing.assert_allclose(np.asarray(actual0), np.asarray(expected0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual1), np.asarray(expected1), rtol=1e-5, atol=1e-6)


def _make_system(
    *,
    batch_size: int,
    n: int,
    shared: bool,
) -> tuple[jnp.ndarray, ...]:
    batch = jnp.arange(batch_size, dtype=jnp.float32)[:, None]
    x = jnp.arange(n, dtype=jnp.float32)[None, :]
    edge = jnp.arange(n - 1, dtype=jnp.float32)[None, :]
    row_scale = 1.0 + 0.01 * batch

    a00 = row_scale * (4.0 + 0.05 * x)
    a01 = row_scale * (-0.35 - 0.01 * x)
    a10 = row_scale * (-0.25 + 0.02 * x)
    a11 = row_scale * (4.8 + 0.07 * x)
    off0 = row_scale * (-0.10 - 0.01 * edge)
    off1 = row_scale * (-0.07 - 0.005 * edge)
    rhs0 = row_scale * jnp.sin(0.3 * x + 0.2 * batch)
    rhs1 = row_scale * jnp.cos(0.2 * x - 0.1 * batch)

    if shared:
        return a00[0], a01[0], a10[0], a11[0], off0[0], off1[0], rhs0, rhs1
    return a00, a01, a10, a11, off0, off1, rhs0, rhs1
