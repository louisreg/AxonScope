from __future__ import annotations

import csv
from pathlib import Path

from benchmark.run import SCRIPTS, main as run_benchmark
from benchmark.workloads.curve_options import PRESETS, build_parser, resolved_options


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
