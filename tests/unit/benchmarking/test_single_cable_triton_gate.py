from __future__ import annotations

import numpy as np

from benchmark.solvers.single_cable_triton import dependency_skip_reason
from benchmark.solvers import single_cable_triton
from benchmark.solvers.single_cable_triton_gate import (
    _jax_solve_xb,
    _print_summary,
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


def test_custom_vmap_collapses_rows_to_one_node_first_solve(monkeypatch):
    import jax
    import jax.numpy as jnp

    calls = []

    def fake_solve(dl, d, du, rhs, *, block_b=128):
        del dl, d, du, block_b
        calls.append(rhs.shape)
        return rhs + 3.0

    monkeypatch.setattr(single_cable_triton, "solve_tridiagonal_xb", fake_solve)
    rows = jnp.arange(20, dtype=jnp.float32).reshape((4, 5))
    actual = jax.vmap(single_cable_triton.solve_tridiagonal_row)(
        rows, rows + 10.0, rows, rows
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(rows + 3.0))
    # custom_vmap traces the scalar implementation to establish its output tree,
    # then emits the node-first batched rule used by the enclosing vmap.
    assert calls[-1] == (5, 4)


def test_gate_parser_accepts_batch_tail_and_launch_sweep():
    args = build_parser().parse_args(
        [
            "--nx",
            "17,63,200",
            "--batch-sizes",
            "129,513",
            "--block-b",
            "64,128",
            "--dry-run",
        ]
    )

    assert args.nx == "17,63,200"
    assert args.batch_sizes == "129,513"
    assert args.block_b == "64,128"
    assert args.dry_run is True


def test_print_summary_keys_reference_by_nx_and_batch(capsys):
    rows = [
        {
            "solver": solver,
            "nx": nx,
            "batch_size": 128,
            "block_b": "" if solver == "jax_tridiagonal_solve" else 128,
            "cold_s": 1.0,
            "warm_median_s": warm,
            "max_abs_error": 0.0,
        }
        for nx, solver, warm in (
            (17, "jax_tridiagonal_solve", 2.0),
            (17, "triton_tiled_thomas", 1.0),
            (200, "jax_tridiagonal_solve", 8.0),
            (200, "triton_tiled_thomas", 2.0),
        )
    ]

    _print_summary(rows)

    output = capsys.readouterr().out
    assert "triton_tiled_thomas,17,128,128,1000.000,1000.000,2.000" in output
    assert "triton_tiled_thomas,200,128,128,1000.000,2000.000,4.000" in output
