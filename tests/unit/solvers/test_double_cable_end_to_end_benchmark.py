from benchmark.solvers.bench_double_cable_end_to_end import (
    _build_iinj,
    planned_cases,
    resolve_e2e_solver,
    main,
)
import pytest


def test_end_to_end_planned_cases_expand_dimensions():
    cases = planned_cases(
        batch_sizes=[2],
        nx_values=[51],
        nt_values=[3],
        dt_ms=0.05,
        recordings=["none", "center"],
        iinj_modes=["none", "dense_zero"],
        solvers=["auto", "thomas"],
    )

    assert len(cases) == 8
    assert cases[0].duration_ms == 3 * 0.05
    assert {case.recording for case in cases} == {"none", "center"}
    assert {case.iinj_mode for case in cases} == {"none", "dense_zero"}


def test_end_to_end_iinj_modes_shape_and_none():
    assert _build_iinj("none", batch_size=2, nt=3, nx=5, dtype="float32") is None
    dense = _build_iinj("dense_zero", batch_size=2, nt=3, nx=5, dtype="float32")
    nonzero = _build_iinj("nonzero", batch_size=2, nt=10, nx=5, dtype="float32")

    assert dense.shape == (2, 3, 5)
    assert nonzero.shape == (2, 10, 5)
    assert float(nonzero.sum()) > 0.0


def test_end_to_end_allows_split_benchmark_solvers_for_array_recording():
    cases = planned_cases(
        batch_sizes=[2],
        nx_values=[51],
        nt_values=[3],
        dt_ms=0.05,
        recordings=["center"],
        iinj_modes=["none"],
        solvers=["split_gs_3", "split_gs_4"],
    )

    assert [case.requested_solver for case in cases] == ["split_gs_3", "split_gs_4"]
    assert resolve_e2e_solver("split_gs_3", platform="gpu") == "split_iterative"
    assert resolve_e2e_solver("split_gs_4", platform="gpu") == "split_iterative"


def test_end_to_end_allows_jax_triton_benchmark_solver_for_array_recording():
    cases = planned_cases(
        batch_sizes=[2],
        nx_values=[51],
        nt_values=[3],
        dt_ms=0.05,
        recordings=["center"],
        iinj_modes=["none"],
        solvers=["jax_triton_thomas"],
    )

    assert [case.requested_solver for case in cases] == ["jax_triton_thomas"]
    assert resolve_e2e_solver("jax_triton_thomas", platform="gpu") == "jax_triton_thomas"


def test_end_to_end_rejects_benchmark_solvers_for_observer_only_recording():
    with pytest.raises(ValueError, match="batch-native array kernel"):
        planned_cases(
            batch_sizes=[2],
            nx_values=[51],
            nt_values=[3],
            dt_ms=0.05,
            recordings=["none"],
            iinj_modes=["none"],
            solvers=["split_gs_3"],
        )


def test_end_to_end_benchmark_dry_run(capsys, tmp_path):
    main(
        [
            "--batch-sizes",
            "2",
            "--nx",
            "51",
            "--nt",
            "3",
            "--dt",
            "0.05",
            "--recordings",
            "center",
            "--iinj-modes",
            "none",
            "--solvers",
            "jax_triton_thomas",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "jax_triton_thomas->jax_triton_thomas B=2 targetNx=51 Nt=3 dt=0.05 recording=center iinj=none"
    ]
