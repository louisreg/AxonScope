import jax.numpy as jnp
import pytest

from benchmark.solvers.validate_double_cable_solver_agreement import (
    compute_agreement_metrics,
    main,
    planned_agreement_cases,
)


def test_validation_planned_cases_expand_dimensions():
    cases = planned_agreement_cases(
        batch_sizes=[2],
        nx_values=[51],
        nt_values=[3],
        dt_ms=0.05,
        recordings=["center", "full"],
        iinj_modes=["none"],
        reference_solvers=["pcr_adaptive"],
        candidate_solvers=["split_gs_3", "split_gs_4"],
    )

    assert len(cases) == 4
    assert cases[0].duration_ms == 3 * 0.05
    assert {case.recording for case in cases} == {"center", "full"}
    assert {case.candidate_solver for case in cases} == {"split_gs_3", "split_gs_4"}


def test_validation_rejects_observer_only_recording():
    with pytest.raises(ValueError, match="requires recorded Vm"):
        planned_agreement_cases(
            batch_sizes=[2],
            nx_values=[51],
            nt_values=[3],
            dt_ms=0.05,
            recordings=["none"],
            iinj_modes=["none"],
            reference_solvers=["pcr_adaptive"],
            candidate_solvers=["split_gs_3"],
        )


def test_validation_metrics_capture_trace_and_activation_agreement():
    reference = jnp.asarray(
        [
            [[-80.0], [-10.0], [20.0]],
            [[-80.0], [-70.0], [-60.0]],
        ],
        dtype=jnp.float32,
    )
    candidate = reference.at[0, 1, 0].add(0.001)

    metrics = compute_agreement_metrics(
        reference,
        candidate,
        dt_ms=0.1,
        activation_threshold_mV=-20.0,
        activation_blanking_ms=0.0,
    )

    assert metrics["max_abs_mV"] == pytest.approx(0.001, abs=1e-6)
    assert metrics["activation_agreement"] == 1.0
    assert metrics["reference_activated_count"] == 1
    assert metrics["candidate_activated_count"] == 1
    assert metrics["first_crossing_time_abs_error_max_ms"] == 0.0


def test_validation_metrics_capture_extra_activation():
    reference = jnp.asarray(
        [
            [[-80.0], [-10.0], [20.0]],
            [[-80.0], [-70.0], [-60.0]],
        ],
        dtype=jnp.float32,
    )
    candidate = reference.at[1, 2, 0].set(-10.0)

    metrics = compute_agreement_metrics(
        reference,
        candidate,
        dt_ms=0.1,
        activation_threshold_mV=-20.0,
        activation_blanking_ms=0.0,
    )

    assert metrics["activation_agreement"] == 0.5
    assert metrics["extra_activation_count"] == 1
    assert metrics["missed_activation_count"] == 0


def test_validation_dry_run(capsys, tmp_path):
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
            "--reference-solvers",
            "pcr_adaptive",
            "--candidate-solvers",
            "split_gs_3",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "split_gs_3 vs pcr_adaptive B=2 targetNx=51 Nt=3 dt=0.05 recording=center iinj=none"
    ]
