from __future__ import annotations

import numpy as np

from benchmark.solvers.single_cable_triton import dependency_skip_reason
from benchmark.solvers.single_cable_triton_gate import (
    _jax_solve_xb,
    build_parser,
    dense_reference_subset,
    make_system_xb,
)


def test_benchmark_candidate_is_import_safe_without_gpu_stack():
    reason = dependency_skip_reason()

    assert reason is None or isinstance(reason, str)


def test_dense_reference_uses_jax_tridiagonal_padding_convention():
    lower, diagonal, upper, rhs = make_system_xb(7, 3, seed=5)
    values = dense_reference_subset((lower, diagonal, upper, rhs), count=3)

    for batch_index in range(3):
        reconstructed = (
            diagonal[:, batch_index] * values[:, batch_index]
        )
        reconstructed[1:] += lower[1:, batch_index] * values[:-1, batch_index]
        reconstructed[:-1] += upper[:-1, batch_index] * values[1:, batch_index]
        np.testing.assert_allclose(
            reconstructed,
            rhs[:, batch_index],
            rtol=2e-5,
            atol=2e-5,
        )


def test_batched_jax_reference_matches_dense_node_first_layout():
    import jax.numpy as jnp

    system = make_system_xb(11, 5, seed=9)
    dense = dense_reference_subset(system, count=5)
    actual = _jax_solve_xb(*(jnp.asarray(value) for value in system))

    np.testing.assert_allclose(np.asarray(actual), dense, rtol=2e-5, atol=2e-5)


def test_gate_parser_accepts_batch_tail_and_launch_sweep():
    args = build_parser().parse_args(
        ["--batch-sizes", "129,513", "--block-b", "64,128", "--dry-run"]
    )

    assert args.batch_sizes == "129,513"
    assert args.block_b == "64,128"
    assert args.dry_run is True
