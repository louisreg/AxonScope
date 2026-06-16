from benchmark.solvers.bench_double_cable_end_to_end import (
    _build_iinj,
    planned_cases,
    main,
)


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
            "thomas",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "thomas->thomas B=2 targetNx=51 Nt=3 dt=0.05 recording=center iinj=none"
    ]
