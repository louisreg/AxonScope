from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import axonscope as axs
from benchmark.workloads import curve_runtime
from benchmark.run import SCRIPTS, main as run_benchmark
from benchmark.workloads.curve_options import PRESETS, build_parser, resolved_options
from benchmark.workloads.curve_runtime import _build_pool, _update_pool_amplitudes


def test_benchmark_launcher_lists_two_curve_scripts_and_presets(capsys):
    assert set(SCRIPTS) == {"threshold_curves", "recruitment_curves"}

    assert run_benchmark(["--list"]) == 0

    out = capsys.readouterr().out
    assert "threshold_curves" in out
    assert "recruitment_curves" in out
    assert "quick" in out
    assert "gpu_realistic" in out


def test_curve_presets_have_explicit_scale_and_execution_defaults():
    assert PRESETS["quick"].n_axons == 1
    assert PRESETS["quick"].memory_trace == "rss"
    assert PRESETS["cpu_publication"].profile is True
    assert PRESETS["cpu_publication"].profile_backend == "jax"
    assert PRESETS["cpu_publication"].jax_device_memory_profile is True
    assert PRESETS["gpu_smoke"].platform == "gpu"
    assert PRESETS["gpu_smoke"].profile is False
    assert PRESETS["gpu_smoke"].jax_device_memory_profile is False
    assert PRESETS["gpu_trace_smoke"].platform == "gpu"
    assert PRESETS["gpu_trace_smoke"].n_axons <= 4
    assert PRESETS["gpu_trace_smoke"].profile is True
    assert PRESETS["gpu_trace_smoke"].jax_device_memory_profile is True
    assert PRESETS["nrv_full"].platform == "nrv"


def test_threshold_dry_run_writes_common_case_options(tmp_path: Path, capsys):
    assert (
        run_benchmark(
            [
                "--script",
                "threshold_curves",
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--dry-run",
                "--output",
                str(tmp_path),
                "--recording",
                "probe_vm",
                "--cable",
                "double_cable",
                "--diameters",
                "different_diameters",
            ]
        )
        == 0
    )

    assert "dry-run: threshold_curves" in capsys.readouterr().out
    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert rows[0]["script"] == "threshold_curves"
    assert rows[0]["platform"] == "cpu"
    assert rows[0]["recording"] == "probe_vm"
    assert rows[0]["cable"] == "double_cable"
    assert rows[0]["diameters"] == "different_diameters"


def test_recruitment_dry_run_supports_gpu_profile_and_case_filter(tmp_path: Path, capsys):
    assert (
        run_benchmark(
            [
                "--script",
                "recruitment_curves",
                "--preset",
                "gpu_smoke",
                "--dry-run",
                "--output",
                str(tmp_path),
                "--profile",
                "--profile-backend",
                "jax",
                "--profile-create-perfetto",
                "--jax-device-memory-profile",
                "--case-filter",
                "observer_only",
                "--amplitude-count",
                "4",
            ]
        )
        == 0
    )

    assert "dry-run: recruitment_curves" in capsys.readouterr().out
    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert rows[0]["script"] == "recruitment_curves"
    assert rows[0]["platform"] == "gpu"
    assert rows[0]["profile"] == "True"
    assert rows[0]["profile_backend"] == "jax"
    assert rows[0]["profile_create_perfetto"] == "True"
    assert rows[0]["jax_device_memory_profile"] == "True"


def test_launcher_can_disable_trace_flags_from_trace_preset(tmp_path: Path):
    assert (
        run_benchmark(
            [
                "--script",
                "threshold_curves",
                "--preset",
                "gpu_trace_smoke",
                "--dry-run",
                "--output",
                str(tmp_path),
                "--no-profile",
                "--no-profile-create-perfetto",
                "--no-jax-device-memory-profile",
                "--jax-device-memory-profile-stage",
                "runtime.prepare",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert rows[0]["profile"] == "False"
    assert rows[0]["profile_create_perfetto"] == "False"
    assert rows[0]["jax_device_memory_profile"] == "False"
    assert rows[0]["jax_device_memory_profile_stages"] == "('runtime.prepare',)"


def test_resolved_options_apply_preset_and_overrides():
    parser = build_parser("threshold_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "local_smoke",
            "--dt",
            "0.025",
            "--n-axons",
            "12",
            "--memory-trace",
            "tracemalloc",
            "--memory-top-n",
            "7",
        ]
    )

    options = resolved_options(args)

    assert options["tsim"] == PRESETS["local_smoke"].tsim
    assert options["dt"] == 0.025
    assert options["n_axons"] == 12
    assert options["memory_trace"] == "tracemalloc"
    assert options["memory_top_n"] == 7


def test_curve_workload_can_update_pool_amplitudes_without_rebuilding():
    parser = build_parser("recruitment_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "quick",
            "--n-axons",
            "2",
            "--nx",
            "5",
            "--stimulation",
            "monophasic",
        ]
    )
    options = resolved_options(args)
    pool, row_meta = _build_pool(
        options,
        np.asarray([0.1, 0.2], dtype=float),
        curve_context="threshold",
    )

    _update_pool_amplitudes(pool, np.asarray([0.3, 0.4], dtype=float), options)

    assert [meta["row"] for meta in row_meta] == [0, 1]
    currents = []
    for simulation in pool:
        stimulation = simulation.extracellular_stimulation
        assert stimulation is not None
        current = stimulation.drives[0].stimulus.evaluate(
            [0.21],
            unit=axs.uA,
        )
        currents.append(float(np.asarray(current)[0]))
    np.testing.assert_allclose(currents, [-0.3, -0.4])


def test_curve_workload_reuses_common_amplitude_stimulus_builds(monkeypatch):
    parser = build_parser("recruitment_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "quick",
            "--n-axons",
            "3",
            "--nx",
            "5",
            "--stimulation",
            "monophasic",
        ]
    )
    options = resolved_options(args)
    pool, _row_meta = _build_pool(
        options,
        np.asarray([0.1, 0.2, 0.3], dtype=float),
        curve_context="recruitment",
    )

    calls = []
    original_stimulus_for_amplitude = curve_runtime._stimulus_for_amplitude

    def stimulus_for_amplitude(options, amplitude_uA):
        calls.append(float(amplitude_uA))
        return original_stimulus_for_amplitude(options, amplitude_uA)

    monkeypatch.setattr(
        curve_runtime,
        "_stimulus_for_amplitude",
        stimulus_for_amplitude,
    )
    _update_pool_amplitudes(pool, np.asarray([0.5, 0.5, 0.5], dtype=float), options)

    stimulations = [simulation.extracellular_stimulation for simulation in pool]
    assert all(stimulation is not None for stimulation in stimulations)
    assert len({id(stimulation) for stimulation in stimulations}) == 3
    assert calls == [0.5]


def test_curve_workload_reuses_same_diameter_axon_templates():
    parser = build_parser("threshold_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "quick",
            "--n-axons",
            "4",
            "--nx",
            "5",
            "--cable",
            "single_cable",
            "--diameters",
            "same_diameter",
        ]
    )
    options = resolved_options(args)
    pool, row_meta = _build_pool(
        options,
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
        curve_context="threshold",
    )

    assert len({id(simulation.axon) for simulation in pool}) == 1
    assert len({id(simulation.extracellular_stimulation) for simulation in pool}) == 4
    assert [meta["diameter_um"] for meta in row_meta] == [0.8, 0.8, 0.8, 0.8]


def test_curve_workload_splits_templates_by_cable_model():
    parser = build_parser("threshold_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "quick",
            "--n-axons",
            "4",
            "--nx",
            "15",
            "--population",
            "mixed_models",
            "--diameters",
            "same_diameter",
        ]
    )
    options = resolved_options(args)
    pool, row_meta = _build_pool(
        options,
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
        curve_context="threshold",
    )

    assert len({id(simulation.axon) for simulation in pool}) == 2
    assert id(pool[0].axon) == id(pool[2].axon)
    assert id(pool[1].axon) == id(pool[3].axon)
    assert [meta["cable"] for meta in row_meta] == [
        "single_cable",
        "double_cable",
        "single_cable",
        "double_cable",
    ]


def test_curve_workload_keeps_distinct_templates_for_distinct_diameters():
    parser = build_parser("threshold_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "quick",
            "--n-axons",
            "4",
            "--nx",
            "5",
            "--cable",
            "single_cable",
            "--diameters",
            "different_diameters",
        ]
    )
    options = resolved_options(args)
    pool, row_meta = _build_pool(
        options,
        np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
        curve_context="threshold",
    )

    assert len({id(simulation.axon) for simulation in pool}) == 4
    np.testing.assert_allclose(
        [meta["diameter_um"] for meta in row_meta],
        [0.4, 0.67, 0.93, 1.2],
    )


def test_curve_workload_reuses_templates_after_axonscope_diameter_quantization():
    parser = build_parser("threshold_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "quick",
            "--n-axons",
            "100",
            "--nx",
            "5",
            "--cable",
            "single_cable",
            "--diameters",
            "different_diameters",
        ]
    )
    options = resolved_options(args)
    pool, row_meta = _build_pool(
        options,
        np.full(100, 0.1, dtype=float),
        curve_context="threshold",
    )

    diameters = [meta["diameter_um"] for meta in row_meta]
    assert len(set(diameters)) < len(diameters)
    assert len({id(simulation.axon) for simulation in pool}) == len(set(diameters))
    assert all(
        np.isclose(diameter * 100.0, round(diameter * 100.0))
        if diameter <= 1.0
        else np.isclose(diameter * 10.0, round(diameter * 10.0))
        for diameter in diameters
    )
