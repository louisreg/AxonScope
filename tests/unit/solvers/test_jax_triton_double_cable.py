from __future__ import annotations

import axonscope.runtime.jax.kernels.triton_double_cable as triton_double_cable
from axonscope.runtime.jax.kernels.triton_double_cable import (
    jax_triton_thomas_dependency_skip_reason,
    solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb,
)


def test_jax_triton_dependency_probe_is_import_safe_without_gpu_stack():
    reason = jax_triton_thomas_dependency_skip_reason()

    assert reason is None or isinstance(reason, str)


def test_jax_triton_runtime_solver_name_documents_layout():
    assert (
        solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb.__name__
        == "solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb"
    )


def test_jax_triton_module_exports_only_retained_runtime_route():
    assert set(triton_double_cable.__all__) == {
        "jax_triton_thomas_dependency_skip_reason",
        "solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb",
    }
    assert not hasattr(
        triton_double_cable,
        "solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched",
    )
    assert not hasattr(
        triton_double_cable,
        "solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_loop_batched",
    )
    assert not hasattr(
        triton_double_cable,
        "solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_xb",
    )
