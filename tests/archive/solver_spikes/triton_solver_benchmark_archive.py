import pytest

from benchmark.triton_solver.bench_double_cable_triton import (
    TritonCase,
    generate_system_jax,
    percentile,
    solver_function,
)
from benchmark.triton_solver.triton_thomas import triton_thomas_dependency_skip_reason


def test_triton_case_label_fields():
    case = TritonCase(solver="triton_block_thomas", batch_size=1024, nx=51, dtype="float32")

    assert case.solver == "triton_block_thomas"
    assert case.batch_size == 1024
    assert case.nx == 51
    assert case.dtype == "float32"


def test_percentile_interpolates():
    assert percentile([1.0, 3.0, 5.0], 50.0) == 3.0
    assert percentile([1.0, 3.0], 95.0) == pytest.approx(2.9)


def test_triton_thomas_dependency_probe_is_import_safe_without_gpu_stack():
    reason = triton_thomas_dependency_skip_reason()

    assert reason is None or isinstance(reason, str)


def test_triton_solver_function_accepts_jax_bridge_solver():
    solve = solver_function("triton_block_thomas_jax_bridge")

    assert solve.__name__ == "solve_triton_block_thomas_jax_bridge"


def test_generate_system_jax_shapes():
    tensors = generate_system_jax(batch_size=3, nx=5, dtype="float32")

    assert [tuple(tensor.shape) for tensor in tensors] == [
        (3, 5),
        (3, 5),
        (3, 5),
        (3, 5),
        (3, 4),
        (3, 4),
        (3, 5),
        (3, 5),
    ]
