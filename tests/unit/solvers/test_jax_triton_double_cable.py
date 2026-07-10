from __future__ import annotations

from axonscope.runtime.jax.jax_triton_double_cable import (
    jax_triton_thomas_dependency_skip_reason,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb,
    solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_xb,
)


def test_jax_triton_dependency_probe_is_import_safe_without_gpu_stack():
    reason = jax_triton_thomas_dependency_skip_reason()

    assert reason is None or isinstance(reason, str)


def test_jax_triton_tiled_solver_names_document_layouts():
    assert (
        solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_xb.__name__
        == "solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_xb"
    )
    assert (
        solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched.__name__
        == "solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched"
    )
    assert (
        solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb.__name__
        == "solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_xb"
    )
    assert (
        solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched.__name__
        == "solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched"
    )
