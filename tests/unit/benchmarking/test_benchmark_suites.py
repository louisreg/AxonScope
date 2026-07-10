from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import axonscope as axs
from benchmark.campaigns.double_cable_solver_policy import (
    main as run_solver_policy_campaign,
)
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
    assert PRESETS["cpu_publication"].memory_trace == "rss"
    assert PRESETS["cpu_publication"].profile is False
    assert PRESETS["cpu_publication"].jax_device_memory_profile is False
    assert PRESETS["gpu_smoke"].platform == "gpu"
    assert PRESETS["gpu_smoke"].memory_trace == "rss"
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
    assert rows[0]["time_chunk_policy"] == "default"
    assert rows[0]["time_chunk_steps"] == ""
    assert rows[0]["amplitude_min"] == "1.0"
    assert rows[0]["amplitude_max"] == "50.0"
    assert rows[0]["stimulation"] == "monophasic"


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


def test_curve_time_chunk_cli_distinguishes_default_unchunked_and_explicit():
    parser = build_parser("recruitment_curves", description="test parser")

    default_options = resolved_options(parser.parse_args(["--preset", "quick"]))
    unchunked_options = resolved_options(
        parser.parse_args(["--preset", "quick", "--time-chunk-steps", "unchunked"])
    )
    none_options = resolved_options(
        parser.parse_args(["--preset", "quick", "--time-chunk-steps", "none"])
    )
    explicit_options = resolved_options(
        parser.parse_args(["--preset", "quick", "--time-chunk-steps", "1000"])
    )

    assert default_options["time_chunk_policy"] == "default"
    assert default_options["time_chunk_steps"] is None
    assert curve_runtime._batch_options(default_options) is None

    assert unchunked_options["time_chunk_policy"] == "unchunked"
    assert unchunked_options["time_chunk_steps"] is None
    assert curve_runtime._batch_options(unchunked_options).time_chunk_steps is None

    assert none_options["time_chunk_policy"] == "unchunked"
    assert none_options["time_chunk_steps"] is None
    assert curve_runtime._batch_options(none_options).time_chunk_steps is None

    assert explicit_options["time_chunk_policy"] == "explicit"
    assert explicit_options["time_chunk_steps"] == 1000
    assert curve_runtime._batch_options(explicit_options).time_chunk_steps == 1000


def test_curve_time_chunk_cli_rejects_invalid_values():
    parser = build_parser("recruitment_curves", description="test parser")

    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", "quick", "--time-chunk-steps", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", "quick", "--time-chunk-steps", "nope"])


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
            "--benchmark-double-cable-block-solver",
            "jax_triton_loop_xb",
            "--benchmark-observer-state-scope",
            "full",
        ]
    )

    options = resolved_options(args)

    assert options["tsim"] == PRESETS["local_smoke"].tsim
    assert options["dt"] == 0.025
    assert options["n_axons"] == 12
    assert options["memory_trace"] == "tracemalloc"
    assert options["memory_top_n"] == 7
    assert options["benchmark_double_cable_block_solver"] == "jax_triton_loop_xb"
    assert options["benchmark_observer_state_scope"] == "full"


def test_solver_policy_campaign_expands_observer_scope_and_time_chunk_matrix(
    tmp_path: Path,
    capsys,
):
    assert (
        run_solver_policy_campaign(
            [
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--curve-script",
                "recruitment_curves",
                "--solver",
                "auto",
                "--recording",
                "observer_only",
                "--n-axons",
                "4",
                "--nx",
                "21",
                "--precision",
                "fp32",
                "--diameters",
                "same_diameter",
                "--time-chunk-steps",
                "default,2",
                "--benchmark-observer-state-scope",
                "default,full",
                "--output",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "planned: 4 double-cable solver policy runs" in out
    assert "__tc2" in out
    assert "__obs_full" in out
    assert "--benchmark-observer-state-scope full" in out

    manifest = json.loads(
        (tmp_path / "double_cable_solver_policy_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (run["time_chunk_steps"], run["observer_state_scope"])
        for run in manifest["runs"]
    ] == [
        ("default", "default"),
        ("default", "full"),
        ("2", "default"),
        ("2", "full"),
    ]
    assert "--benchmark-observer-state-scope" not in manifest["runs"][0]["command"]
    assert "--benchmark-observer-state-scope" in manifest["runs"][1]["command"]
    assert "--time-chunk-steps" in manifest["runs"][0]["command"]
    assert "--time-chunk-steps" in manifest["runs"][2]["command"]


def test_resolved_options_accept_public_tiled_thomas_solver_policy():
    parser = build_parser("recruitment_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "gpu_smoke",
            "--platform",
            "gpu",
            "--cable",
            "double_cable",
            "--double-cable-block-solver",
            "tiled_thomas",
            "--tiled-thomas-block-b",
            "64",
        ]
    )

    options = resolved_options(args)
    solver_policy = curve_runtime._solver_policy(options)

    assert options["double_cable_block_solver"] == "tiled_thomas"
    assert options["tiled_thomas_block_b"] == 64
    assert (
        solver_policy.double_cable.kind
        is axs.runtime.jax.DoubleCableSolverKind.TILED_THOMAS
    )
    assert solver_policy.double_cable.tiled_thomas_options.block_b == 64


def test_threshold_defaults_follow_example07_mrg_bounds():
    parser = build_parser("threshold_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "local_smoke",
            "--cable",
            "double_cable",
        ]
    )

    options = resolved_options(args)

    assert options["amplitude_min"] == 1.0
    assert options["amplitude_max"] == 50.0
    assert options["stimulation"] == "monophasic"


def test_threshold_explicit_amplitude_and_stimulation_overrides_win():
    parser = build_parser("threshold_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "local_smoke",
            "--cable",
            "double_cable",
            "--amplitude-min",
            "0",
            "--amplitude-max",
            "150",
            "--stimulation",
            "biphasic",
        ]
    )

    options = resolved_options(args)

    assert options["amplitude_min"] == 0.0
    assert options["amplitude_max"] == 150.0
    assert options["stimulation"] == "biphasic"


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


@pytest.mark.parametrize("curve_context", ["threshold", "recruitment"])
def test_curve_workload_fast_point_source_matches_public_helper(curve_context):
    parser = build_parser("recruitment_curves", description="test parser")
    args = parser.parse_args(
        [
            "--preset",
            "quick",
            "--n-axons",
            "1",
            "--nx",
            "5",
            "--stimulation",
            "monophasic",
        ]
    )
    options = resolved_options(args)
    pool, row_meta = _build_pool(
        options,
        np.asarray([0.1], dtype=float),
        curve_context=curve_context,
    )
    simulation = pool[0]
    stimulation = simulation.extracellular_stimulation
    assert stimulation is not None
    drive = stimulation.drives[0]
    metadata = drive.footprint.metadata
    electrode = axs.analytical.PointSourceElectrode(
        x=float(metadata["electrode_x_um"]) * axs.um,
        y=float(metadata["electrode_y_um"]) * axs.um,
        z=float(metadata["electrode_z_um"]) * axs.um,
        min_distance=(5.0 * axs.um if curve_context == "recruitment" else None),
    )
    expected = axs.analytical.point_source_stimulation(
        electrode,
        axs.units.Q_(
            simulation.axon.layout.position_values(unit="micrometer"),
            "micrometer",
        ),
        sigma=0.3 * axs.S_per_m,
        stimulus=drive.stimulus,
        axon_y=float(row_meta[0]["axon_y_um"]) * axs.um,
        axon_z=float(row_meta[0]["axon_z_um"]) * axs.um,
        axon_id=axs.AxonId("row_00000"),
    )

    np.testing.assert_allclose(
        drive.footprint.values_for_axon(axs.AxonId("row_00000")),
        expected.drives[0].footprint.values_for_axon(axs.AxonId("row_00000")),
        rtol=1e-12,
        atol=1e-12,
    )


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
