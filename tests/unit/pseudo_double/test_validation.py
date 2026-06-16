from __future__ import annotations

import json

import pytest

from benchmark.pseudo_double.validation import (
    PseudoDoubleEffectiveConfig,
    PseudoDoubleSchurLocalConfig,
    PseudoDoubleSeriesConfig,
    PseudoDoubleSingleChainConfig,
    PseudoDoubleSplitConfig,
    build_validation_population,
    main,
    mode_metadata,
    normalize_validation_mode,
    score_validation_result,
    summarize_thresholds,
    write_outputs,
)


def test_validation_mode_normalization_and_metadata():
    assert normalize_validation_mode("double") == "exact_double"
    assert normalize_validation_mode("single-cable") == "mrg_single_cable_surrogate"

    exact = mode_metadata("exact_double")
    surrogate = mode_metadata("mrg_single_cable_surrogate")
    effective = mode_metadata("pseudo_double_effective")
    single_chain = mode_metadata("pseudo_double_single_myelinated_chain")
    series = mode_metadata("pseudo_double_series")
    split = mode_metadata("pseudo_double_split")
    schur = mode_metadata("pseudo_double_schur_local")

    assert exact["reference"] is True
    assert exact["experimental"] is False
    assert exact["implemented"] is True
    assert surrogate["reference"] is False
    assert surrogate["experimental"] is True
    assert surrogate["implemented"] is True
    assert effective["reference"] is False
    assert effective["experimental"] is True
    assert effective["implemented"] is True
    assert single_chain["reference"] is False
    assert single_chain["experimental"] is True
    assert single_chain["implemented"] is True
    assert series["reference"] is False
    assert series["experimental"] is True
    assert series["implemented"] is True
    assert split["reference"] is False
    assert split["experimental"] is True
    assert split["implemented"] is True
    assert schur["reference"] is False
    assert schur["experimental"] is True
    assert schur["implemented"] is True


def test_validation_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown pseudo-double validation mode"):
        normalize_validation_mode("magic-cable")


def test_validation_population_builds_exact_and_surrogate_formulations():
    exact = build_validation_population(
        "exact_double",
        size=1,
        nodes=3,
        diameter_um=5.7,
        amplitude_uA=20.0,
        pulse_start_ms=0.1,
        pulse_duration_ms=0.1,
        electrode_z_um=120.0,
        offset_span_um=40.0,
    )
    surrogate = build_validation_population(
        "mrg_single_cable_surrogate",
        size=1,
        nodes=3,
        diameter_um=5.7,
        amplitude_uA=20.0,
        pulse_start_ms=0.1,
        pulse_duration_ms=0.1,
        electrode_z_um=120.0,
        offset_span_um=40.0,
    )
    effective = build_validation_population(
        "pseudo_double_effective",
        size=1,
        nodes=3,
        diameter_um=5.7,
        duration_ms=0.2,
        dt_ms=0.05,
        amplitude_uA=20.0,
        pulse_start_ms=0.1,
        pulse_duration_ms=0.1,
        electrode_z_um=120.0,
        offset_span_um=40.0,
        effective_config=PseudoDoubleEffectiveConfig(vext_scale=1.25),
    )
    single_chain = build_validation_population(
        "pseudo_double_single_myelinated_chain",
        size=1,
        nodes=3,
        diameter_um=5.7,
        duration_ms=0.2,
        dt_ms=0.05,
        amplitude_uA=20.0,
        pulse_start_ms=0.1,
        pulse_duration_ms=0.1,
        electrode_z_um=120.0,
        offset_span_um=40.0,
        single_chain_config=PseudoDoubleSingleChainConfig(vext_scale=1.25),
    )
    series = build_validation_population(
        "pseudo_double_series",
        size=1,
        nodes=3,
        diameter_um=5.7,
        duration_ms=0.2,
        dt_ms=0.05,
        amplitude_uA=20.0,
        pulse_start_ms=0.1,
        pulse_duration_ms=0.1,
        electrode_z_um=120.0,
        offset_span_um=40.0,
    )
    split = build_validation_population(
        "pseudo_double_split",
        size=1,
        nodes=3,
        diameter_um=5.7,
        duration_ms=0.2,
        dt_ms=0.05,
        amplitude_uA=20.0,
        pulse_start_ms=0.1,
        pulse_duration_ms=0.1,
        electrode_z_um=120.0,
        offset_span_um=40.0,
        split_config=PseudoDoubleSplitConfig(vext_scale=1.25),
    )
    schur = build_validation_population(
        "pseudo_double_schur_local",
        size=1,
        nodes=3,
        diameter_um=5.7,
        duration_ms=0.2,
        dt_ms=0.05,
        amplitude_uA=20.0,
        pulse_start_ms=0.1,
        pulse_duration_ms=0.1,
        electrode_z_um=120.0,
        offset_span_um=40.0,
    )

    assert exact[0].axon.resolved_formulation == "double-cable"
    assert surrogate[0].axon.resolved_formulation == "single-cable"
    assert effective[0].axon.resolved_formulation == "single-cable"
    assert single_chain[0].axon.resolved_formulation == "single-cable"
    assert series[0].axon.resolved_formulation == "double-cable"
    assert split[0].axon.resolved_formulation == "single-cable"
    assert schur[0].axon.resolved_formulation == "double-cable"
    assert exact[0].extracellular_context is not None
    assert surrogate[0].extracellular_context is not None
    assert effective[0].extracellular_context is not None
    assert single_chain[0].extracellular_context is not None
    assert series[0].extracellular_context is not None
    assert split[0].extracellular_context is not None
    assert schur[0].extracellular_context is not None


def test_pseudo_double_configs_are_json_friendly():
    assert PseudoDoubleEffectiveConfig(vext_scale=1.25).as_dict() == {
        "vext_scale": 1.25
    }
    assert PseudoDoubleSingleChainConfig(
        vext_scale=1.25,
        cm_scale_node=1.1,
        cm_scale_mysa=1.2,
        cm_scale_flut=1.3,
        cm_scale_stin=1.4,
        gleak_scale_node=1.5,
        gleak_scale_mysa=1.6,
        gleak_scale_flut=1.7,
        gleak_scale_stin=1.8,
        axial_resistance_scale=1.9,
        vext_alpha_node=1.0,
        vext_alpha_mysa=0.9,
        vext_alpha_flut=0.8,
        vext_alpha_stin=0.7,
        use_series_capacitance=False,
        use_series_leak=False,
    ).as_dict() == {
        "vext_scale": 1.25,
        "cm_scale_node": 1.1,
        "cm_scale_mysa": 1.2,
        "cm_scale_flut": 1.3,
        "cm_scale_stin": 1.4,
        "gleak_scale_node": 1.5,
        "gleak_scale_mysa": 1.6,
        "gleak_scale_flut": 1.7,
        "gleak_scale_stin": 1.8,
        "axial_resistance_scale": 1.9,
        "vext_alpha_node": 1.0,
        "vext_alpha_mysa": 0.9,
        "vext_alpha_flut": 0.8,
        "vext_alpha_stin": 0.7,
        "use_series_capacitance": False,
        "use_series_leak": False,
        "active_nodes_only": True,
    }
    assert PseudoDoubleSeriesConfig(
        vext_scale=1.1,
        capacitance_floor_fraction=0.03,
        conductance_floor_fraction=0.04,
    ).as_dict() == {
        "vext_scale": 1.1,
        "capacitance_floor_fraction": 0.03,
        "conductance_floor_fraction": 0.04,
    }
    assert PseudoDoubleSplitConfig(
        vext_scale=1.25,
        direct_scale=0.5,
        aux_scale=1.5,
        aux_alpha=0.75,
        aux_tau_ms=0.08,
    ).as_dict() == {
        "vext_scale": 1.25,
        "direct_scale": 0.5,
        "aux_scale": 1.5,
        "aux_alpha": 0.75,
        "aux_tau_ms": 0.08,
    }
    assert PseudoDoubleSchurLocalConfig(
        vext_scale=0.9,
        app_inverse_scale=1.1,
    ).as_dict() == {
        "vext_scale": 0.9,
        "app_inverse_scale": 1.1,
    }


def test_unimplemented_pseudo_modes_are_explicit():
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        build_validation_population(
            "pseudo_double_modal",
            size=1,
            nodes=3,
            diameter_um=5.7,
            amplitude_uA=20.0,
            pulse_start_ms=0.1,
            pulse_duration_ms=0.1,
            electrode_z_um=120.0,
            offset_span_um=40.0,
        )


def test_threshold_summary_tracks_misses_and_relative_error():
    rows = [
        {"row": 0, "amplitude_uA": 10.0, "reference_activated": False, "candidate_activated": False},
        {"row": 0, "amplitude_uA": 20.0, "reference_activated": True, "candidate_activated": False},
        {"row": 0, "amplitude_uA": 30.0, "reference_activated": True, "candidate_activated": True},
        {"row": 1, "amplitude_uA": 10.0, "reference_activated": False, "candidate_activated": False},
        {"row": 1, "amplitude_uA": 20.0, "reference_activated": True, "candidate_activated": True},
    ]

    summary = summarize_thresholds(rows, size=2)

    assert summary["reference_thresholds_uA"] == [20.0, 20.0]
    assert summary["candidate_thresholds_uA"] == [30.0, 20.0]
    assert summary["missed_activation_count"] == 0
    assert summary["threshold_rel_error_mean"] == pytest.approx(0.25)


def test_score_validation_result_prioritizes_false_negatives():
    base = {
        "amplitude_summaries": [
            {
                "activation_agreement": 1.0,
                "false_negative_count": 0,
                "false_positive_count": 1,
                "peak_abs_error_p95_mV": 0.0,
            }
        ],
        "threshold_summary": {
            "missed_activation_count": 0,
            "extra_activation_count": 0,
            "threshold_rel_error_p95": None,
        },
    }
    missed = {
        "amplitude_summaries": [
            {
                "activation_agreement": 1.0,
                "false_negative_count": 1,
                "false_positive_count": 0,
                "peak_abs_error_p95_mV": 0.0,
            }
        ],
        "threshold_summary": {
            "missed_activation_count": 0,
            "extra_activation_count": 0,
            "threshold_rel_error_p95": None,
        },
    }

    assert score_validation_result(missed) > score_validation_result(base)


def test_dry_run_prints_validation_plan(capsys):
    main(
        [
            "--dry-run",
            "--size",
            "1",
            "--nodes",
            "3",
            "--amplitudes-uA",
            "10",
            "20",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "pseudo_double_validation reference=exact_double "
        "candidate=mrg_single_cable_surrogate size=1 nodes=3 "
        "amplitudes_uA=10.0,20.0"
    ]


def test_dry_run_prints_pseudo_effective_calibration_plan(capsys):
    main(
        [
            "--dry-run",
            "--candidate",
            "pseudo_double_effective",
            "--pseudo-vext-scale",
            "1.2",
            "--calibrate-vext-scales",
            "1.0",
            "1.2",
            "--size",
            "1",
            "--nodes",
            "3",
            "--amplitudes-uA",
            "10",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "pseudo_double_validation reference=exact_double "
        "candidate=pseudo_double_effective size=1 nodes=3 "
        "amplitudes_uA=10.0 pseudo_vext_scale=1.2 "
        "calibrate_vext_scales=1.0,1.2 include_baseline=true"
    ]


def test_dry_run_prints_pseudo_single_chain_calibration_plan(capsys):
    main(
        [
            "--dry-run",
            "--candidate",
            "pseudo_double_single_myelinated_chain",
            "--single-chain-vext-scale",
            "1.4",
            "--single-chain-alpha-mysa",
            "0.8",
            "--single-chain-alpha-flut",
            "0.6",
            "--single-chain-alpha-stin",
            "0.4",
            "--calibrate-vext-scales",
            "1.0",
            "1.4",
            "--size",
            "1",
            "--nodes",
            "3",
            "--amplitudes-uA",
            "10",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "pseudo_double_validation reference=exact_double "
        "candidate=pseudo_double_single_myelinated_chain size=1 nodes=3 "
        "amplitudes_uA=10.0 single_chain_vext_scale=1.4 "
        "single_chain_alpha=1.0,0.8,0.6,0.4 "
        "calibrate_vext_scales=1.0,1.4 include_baseline=true"
    ]


def test_dry_run_prints_pseudo_series_calibration_plan(capsys):
    main(
        [
            "--dry-run",
            "--candidate",
            "pseudo_double_series",
            "--series-vext-scale",
            "1.3",
            "--series-capacitance-floor-fraction",
            "0.04",
            "--calibrate-vext-scales",
            "1.0",
            "1.3",
            "--size",
            "1",
            "--nodes",
            "3",
            "--amplitudes-uA",
            "10",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "pseudo_double_validation reference=exact_double "
        "candidate=pseudo_double_series size=1 nodes=3 "
        "amplitudes_uA=10.0 series_vext_scale=1.3 "
        "series_capacitance_floor_fraction=0.04 "
        "calibrate_vext_scales=1.0,1.3 include_baseline=true"
    ]


def test_dry_run_prints_pseudo_split_calibration_plan(capsys):
    main(
        [
            "--dry-run",
            "--candidate",
            "pseudo_double_split",
            "--split-vext-scale",
            "1.3",
            "--split-aux-tau-ms",
            "0.08",
            "--calibrate-vext-scales",
            "1.0",
            "1.3",
            "--size",
            "1",
            "--nodes",
            "3",
            "--amplitudes-uA",
            "10",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "pseudo_double_validation reference=exact_double "
        "candidate=pseudo_double_split size=1 nodes=3 "
        "amplitudes_uA=10.0 split_vext_scale=1.3 "
        "split_aux_tau_ms=0.08 calibrate_vext_scales=1.0,1.3 "
        "include_baseline=true"
    ]


def test_dry_run_prints_pseudo_schur_calibration_plan(capsys):
    main(
        [
            "--dry-run",
            "--candidate",
            "pseudo_double_schur_local",
            "--schur-vext-scale",
            "0.9",
            "--schur-app-inverse-scale",
            "1.1",
            "--calibrate-vext-scales",
            "0.8",
            "1.0",
            "--size",
            "1",
            "--nodes",
            "3",
            "--amplitudes-uA",
            "10",
        ]
    )

    assert capsys.readouterr().out.splitlines() == [
        "pseudo_double_validation reference=exact_double "
        "candidate=pseudo_double_schur_local size=1 nodes=3 "
        "amplitudes_uA=10.0 schur_vext_scale=0.9 "
        "schur_app_inverse_scale=1.1 calibrate_vext_scales=0.8,1.0 "
        "include_baseline=true"
    ]


def test_write_outputs_creates_json_and_csv(tmp_path):
    result = {
        "rows": [
            {
                "amplitude_uA": 10.0,
                "row": 0,
                "reference_activated": False,
                "candidate_activated": False,
            }
        ]
    }

    json_path, csv_path = write_outputs(result, tmp_path)

    assert json.loads(json_path.read_text()) == result
    assert csv_path.read_text().splitlines()[0] == (
        "amplitude_uA,candidate_activated,reference_activated,row"
    )
