from pathlib import Path

from benchmark.cuda_ffi_solver.bench_double_cable_cuda_ffi import (
    CudaFfiCase,
    solver_function,
    write_summary_csv,
)
from benchmark.cuda_ffi_solver.cuda_ffi_thomas import (
    TARGET_NAME,
    cuda_ffi_thomas_dependency_skip_reason,
)


def test_cuda_ffi_case_label_fields():
    case = CudaFfiCase(solver="cuda_ffi_block_thomas", batch_size=1024, nx=96, dtype="float32")

    assert case.solver == "cuda_ffi_block_thomas"
    assert case.batch_size == 1024
    assert case.nx == 96
    assert case.dtype == "float32"


def test_cuda_ffi_dependency_probe_is_import_safe_without_gpu_stack():
    reason = cuda_ffi_thomas_dependency_skip_reason()

    assert reason is None or isinstance(reason, str)


def test_cuda_ffi_solver_function_selects_jax_ffi_solver():
    solve = solver_function("cuda_ffi_block_thomas")

    assert solve.__name__ == "solve_block_tridiagonal_2x2_cuda_ffi_thomas"
    assert TARGET_NAME == "axonscope_double_cable_thomas_f32"


def test_cuda_ffi_summary_includes_build_column(tmp_path: Path):
    path = tmp_path / "summary.csv"

    write_summary_csv(
        path,
        [
            {
                "solver": "cuda_ffi_block_thomas",
                "batch_size": 8,
                "nx": 16,
                "dtype": "float32",
                "ffi_build_ms": 12.5,
                "compile_first_ms": 1.0,
                "steady_min_ms": 0.5,
                "steady_median_ms": 0.6,
                "steady_p95_ms": 0.7,
                "node_solves_per_s": 1.0,
                "max_abs_error_vs_dense64_smoke": 0.0,
                "max_block_residual_norm": 0.0,
                "median_block_residual_norm": 0.0,
            }
        ],
    )

    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "ffi_build_ms" in header

