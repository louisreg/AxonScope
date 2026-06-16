import jax.numpy as jnp

from benchmark.solvers.bench_double_cable_linear_solvers import (
    generate_system,
    planned_cases,
    resolve_kernel_solver,
    main,
)


def test_generate_linear_solver_system_shapes():
    system = generate_system(batch_size=3, nx=5, dtype="float32")

    assert [array.shape for array in system] == [
        (3, 5),
        (3, 5),
        (3, 5),
        (3, 5),
        (3, 4),
        (3, 4),
        (3, 5),
        (3, 5),
    ]
    assert all(array.dtype == jnp.float32 for array in system)


def test_planned_cases_resolve_adaptive_kernel_solver():
    cases = planned_cases(
        batch_sizes=[4096, 4097],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["pcr_adaptive"],
        platform="gpu",
    )

    assert [case.kernel_solver for case in cases] == ["pcr_soa", "pcr"]
    assert resolve_kernel_solver("pcr_adaptive", batch_size=4096) == "pcr_soa"
    assert resolve_kernel_solver("pcr_adaptive", batch_size=4097) == "pcr"


def test_planned_cases_allow_benchmark_only_padded_pcr_soa():
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["pcr_soa_padded"],
        platform="gpu",
    )

    assert cases[0].requested_solver == "pcr_soa_padded"
    assert cases[0].resolved_solver == "pcr_soa"
    assert cases[0].kernel_solver == "pcr_soa_padded"
    assert resolve_kernel_solver("pcr_soa_padded", batch_size=512) == "pcr_soa_padded"


def test_planned_cases_allow_benchmark_only_transposed_pcr_soa():
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["pcr_soa_transposed"],
        platform="gpu",
    )

    assert cases[0].requested_solver == "pcr_soa_transposed"
    assert cases[0].resolved_solver == "pcr_soa"
    assert cases[0].kernel_solver == "pcr_soa_transposed"
    assert (
        resolve_kernel_solver("pcr_soa_transposed", batch_size=512)
        == "pcr_soa_transposed"
    )


def test_planned_cases_allow_benchmark_only_hybrid_pcr_soa_variants():
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["pcr_soa_hybrid_4", "pcr_soa_hybrid_8", "pcr_soa_hybrid_16"],
        platform="gpu",
    )

    assert [case.requested_solver for case in cases] == [
        "pcr_soa_hybrid_4",
        "pcr_soa_hybrid_8",
        "pcr_soa_hybrid_16",
    ]
    assert [case.resolved_solver for case in cases] == ["pcr_soa", "pcr_soa", "pcr_soa"]
    assert [case.kernel_solver for case in cases] == [
        "pcr_soa_hybrid_4",
        "pcr_soa_hybrid_8",
        "pcr_soa_hybrid_16",
    ]
    assert resolve_kernel_solver("pcr_soa_hybrid_4", batch_size=512) == "pcr_soa_hybrid_4"
    assert resolve_kernel_solver("pcr_soa_hybrid_8", batch_size=512) == "pcr_soa_hybrid_8"
    assert resolve_kernel_solver("pcr_soa_hybrid_16", batch_size=512) == "pcr_soa_hybrid_16"


def test_planned_cases_allow_benchmark_only_batched_thomas():
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["thomas_batched"],
        platform="gpu",
    )

    assert cases[0].requested_solver == "thomas_batched"
    assert cases[0].resolved_solver == "thomas"
    assert cases[0].kernel_solver == "thomas_batched"
    assert resolve_kernel_solver("thomas_batched", batch_size=512) == "thomas_batched"


def test_planned_cases_allow_benchmark_only_split_iterative_variants():
    solvers = [
        "split_jacobi_4",
        "split_jacobi_8",
        "split_gs_4",
        "split_gs_8",
        "split_richardson_4",
    ]
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=solvers,
        platform="gpu",
    )

    assert [case.requested_solver for case in cases] == solvers
    assert [case.resolved_solver for case in cases] == ["split_iterative"] * len(solvers)
    assert [case.kernel_solver for case in cases] == solvers
    for solver in solvers:
        assert resolve_kernel_solver(solver, batch_size=512) == solver


def test_linear_solver_benchmark_dry_run(capsys, tmp_path):
    main(
        [
            "--batch-sizes",
            "2",
            "--nx",
            "5",
            "--dtypes",
            "float32",
            "--solvers",
            "thomas",
            "split_jacobi_4",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "thomas -> thomas -> thomas B=2 Nx=5 dtype=float32",
        "split_jacobi_4 -> split_iterative -> split_jacobi_4 B=2 Nx=5 dtype=float32",
    ]
