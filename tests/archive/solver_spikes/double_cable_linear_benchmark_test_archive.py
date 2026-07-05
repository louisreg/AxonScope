import jax.numpy as jnp

from benchmark.solvers.bench_double_cable_linear_solvers import (
    generate_system,
    main,
    planned_cases,
    resolve_kernel_solver,
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


def test_planned_cases_cover_public_solver_routes():
    cases = planned_cases(
        batch_sizes=[4096, 4097],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["thomas", "pcr", "pcr_soa", "pcr_adaptive"],
        platform="gpu",
    )

    assert [(case.requested_solver, case.resolved_solver, case.kernel_solver) for case in cases] == [
        ("thomas", "thomas", "thomas"),
        ("pcr", "pcr", "pcr"),
        ("pcr_soa", "pcr_soa", "pcr_soa"),
        ("pcr_adaptive", "pcr_adaptive", "pcr_soa"),
        ("thomas", "thomas", "thomas"),
        ("pcr", "pcr", "pcr"),
        ("pcr_soa", "pcr_soa", "pcr_soa"),
        ("pcr_adaptive", "pcr_adaptive", "pcr"),
    ]
    assert resolve_kernel_solver("pcr_adaptive", batch_size=4096) == "pcr_soa"
    assert resolve_kernel_solver("pcr_adaptive", batch_size=4097) == "pcr"


def test_linear_solver_benchmark_dry_run_uses_public_solvers(capsys, tmp_path):
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
            "pcr_soa",
            "pcr_adaptive",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "thomas -> thomas -> thomas B=2 Nx=5 dtype=float32",
        "pcr_soa -> pcr_soa -> pcr_soa B=2 Nx=5 dtype=float32",
        "pcr_adaptive -> pcr_adaptive -> pcr_soa B=2 Nx=5 dtype=float32",
    ]
