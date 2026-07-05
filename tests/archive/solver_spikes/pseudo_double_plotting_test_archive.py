from __future__ import annotations

from benchmark.pseudo_double.plotting import write_validation_plots


def test_write_validation_plots_creates_summary_and_trace_pngs(tmp_path):
    result = {
        "candidate_mode": "pseudo_double_series",
        "parameters": {"activation_threshold_mV": -20.0},
        "amplitude_summaries": [
            {
                "amplitude_uA": 20.0,
                "exact_active_count": 0,
                "candidate_active_count": 0,
                "activation_agreement": 1.0,
                "false_negative_count": 0,
                "false_positive_count": 0,
                "peak_abs_error_mean_mV": 2.0,
                "center_peak_abs_error_mean_mV": 1.0,
                "rms_vm_error_mean_mV": 3.0,
                "activation_time_abs_error_mean_ms": None,
            },
            {
                "amplitude_uA": 60.0,
                "exact_active_count": 1,
                "candidate_active_count": 1,
                "activation_agreement": 1.0,
                "false_negative_count": 0,
                "false_positive_count": 0,
                "peak_abs_error_mean_mV": 0.5,
                "center_peak_abs_error_mean_mV": 0.25,
                "rms_vm_error_mean_mV": 5.0,
                "activation_time_abs_error_mean_ms": 0.05,
            },
        ],
        "threshold_summary": {
            "reference_thresholds_uA": [60.0],
            "candidate_thresholds_uA": [60.0],
            "threshold_rel_error_mean": 0.0,
        },
        "timings": [
            {
                "amplitude_uA": 20.0,
                "reference_seconds": 0.02,
                "candidate_seconds": 0.01,
                "candidate_speedup_vs_reference": 2.0,
            },
            {
                "amplitude_uA": 60.0,
                "reference_seconds": 0.02,
                "candidate_seconds": 0.012,
                "candidate_speedup_vs_reference": 1.6667,
            },
        ],
        "trace_samples": [
            {
                "amplitude_uA": 60.0,
                "row": 0,
                "t_ms": [0.05, 0.10, 0.15],
                "positions_um": [0.0, 10.0, 20.0],
                "center_column": 1,
                "reference_peak_column": 2,
                "reference_vm_mV": [
                    [-80.0, -78.0, -75.0],
                    [-70.0, -30.0, 35.0],
                    [-79.0, -50.0, -10.0],
                ],
                "candidate_vm_mV": [
                    [-80.0, -79.0, -75.0],
                    [-72.0, -29.0, 33.0],
                    [-79.0, -51.0, -12.0],
                ],
            }
        ],
    }

    paths = write_validation_plots(result, tmp_path)

    names = {path.name for path in paths}
    assert {
        "activation_summary.png",
        "error_summary.png",
        "thresholds.png",
        "timings.png",
        "trace_amp_60_row_0.png",
    }.issubset(names)
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
