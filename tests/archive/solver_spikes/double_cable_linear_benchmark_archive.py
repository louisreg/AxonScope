import jax.numpy as jnp
import pytest

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


def test_planned_cases_allow_benchmark_only_auto_layout_pcr_soa():
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["pcr_soa_layout_auto"],
        platform="gpu",
    )

    assert cases[0].requested_solver == "pcr_soa_layout_auto"
    assert cases[0].resolved_solver == "pcr_soa"
    assert cases[0].kernel_solver == "pcr_soa_layout_auto"
    assert (
        resolve_kernel_solver("pcr_soa_layout_auto", batch_size=512)
        == "pcr_soa_layout_auto"
    )


def test_planned_cases_allow_benchmark_only_ref_pcr_soa():
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["pcr_soa_ref"],
        platform="gpu",
    )

    assert cases[0].requested_solver == "pcr_soa_ref"
    assert cases[0].resolved_solver == "pcr_soa"
    assert cases[0].kernel_solver == "pcr_soa_ref"
    assert resolve_kernel_solver("pcr_soa_ref", batch_size=512) == "pcr_soa_ref"


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


def test_planned_cases_allow_benchmark_only_nomask_pcr_soa():
    cases = planned_cases(
        batch_sizes=[512],
        nx_values=[51],
        dtypes=["float32"],
        solvers=["pcr_soa_nomask", "pcr_soa_shift"],
        platform="gpu",
    )

    assert cases[0].requested_solver == "pcr_soa_nomask"
    assert cases[0].resolved_solver == "pcr_soa"
    assert cases[0].kernel_solver == "pcr_soa_nomask"
    assert cases[1].requested_solver == "pcr_soa_shift"
    assert cases[1].resolved_solver == "pcr_soa"
    assert cases[1].kernel_solver == "pcr_soa_shift"
    assert resolve_kernel_solver("pcr_soa_nomask", batch_size=512) == "pcr_soa_nomask"
    assert resolve_kernel_solver("pcr_soa_shift", batch_size=512) == "pcr_soa_shift"


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


def test_active_linear_benchmark_rejects_archived_solver_spikes():
    archived_solvers = [
        "assoc_backward",
        "assoc_transfer_dense",
        "pallas_pcr_128",
        "pallas_thomas_4",
        "split_jacobi_4",
        "split_gs_4",
    ]
    for solver in archived_solvers:
        with pytest.raises(ValueError, match="unknown solver choice"):
            planned_cases(
                batch_sizes=[512],
                nx_values=[51],
                dtypes=["float32"],
                solvers=[solver],
                platform="gpu",
            )
        with pytest.raises(ValueError, match="unsupported resolved solver"):
            resolve_kernel_solver(solver, batch_size=512)


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
            "pcr_adaptive",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "thomas -> thomas -> thomas B=2 Nx=5 dtype=float32",
        "pcr_adaptive -> pcr_adaptive -> pcr_soa B=2 Nx=5 dtype=float32",
    ]
