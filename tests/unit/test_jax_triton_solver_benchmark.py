from benchmark.jax_triton_solver.bench_double_cable_jax_triton import (
    JaxTritonCase,
    solver_function,
)
from benchmark.jax_triton_solver.jax_triton_thomas import (
    jax_triton_thomas_dependency_skip_reason,
)


def test_jax_triton_case_label_fields():
    case = JaxTritonCase(
        solver="jax_triton_block_thomas",
        batch_size=1024,
        nx=96,
        dtype="float32",
    )

    assert case.solver == "jax_triton_block_thomas"
    assert case.batch_size == 1024
    assert case.nx == 96
    assert case.dtype == "float32"


def test_jax_triton_dependency_probe_is_import_safe_without_gpu_stack():
    reason = jax_triton_thomas_dependency_skip_reason()

    assert reason is None or isinstance(reason, str)


def test_jax_triton_solver_function_selects_bridge_solver():
    solve = solver_function("jax_triton_block_thomas")

    assert solve.__name__ == "solve_block_tridiagonal_2x2_jax_triton_thomas"

