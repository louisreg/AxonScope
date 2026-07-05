from __future__ import annotations

import csv
from pathlib import Path

from benchmark.curves import recruitment_curves, threshold_curves
from benchmark.run import main as run_benchmark


def test_threshold_curve_script_direct_dry_run(tmp_path: Path, capsys):
    assert (
        threshold_curves.main(
            [
                "--preset",
                "quick",
                "--dry-run",
                "--output",
                str(tmp_path),
                "--threshold-kind",
                "both",
            ]
        )
        == 0
    )

    assert "threshold_curves" in capsys.readouterr().out
    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert rows[0]["script"] == "threshold_curves"


def test_recruitment_curve_script_direct_dry_run(tmp_path: Path, capsys):
    assert (
        recruitment_curves.main(
            [
                "--preset",
                "local_smoke",
                "--dry-run",
                "--output",
                str(tmp_path),
                "--amplitude-count",
                "6",
            ]
        )
        == 0
    )

    assert "recruitment_curves" in capsys.readouterr().out
    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert rows[0]["script"] == "recruitment_curves"
    assert rows[0]["n_axons"] == "8"


def test_unified_launcher_dry_run_can_forward_advanced_options(tmp_path: Path):
    assert (
        run_benchmark(
            [
                "--script",
                "threshold_curves",
                "--preset",
                "quick",
                "--dry-run",
                "--output",
                str(tmp_path),
                "--spatial-recording",
                "indices",
                "--observer-criterion",
                "activation",
                "--cache-mode",
                "cold",
                "--retention",
                "debug_artifacts",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert rows[0]["spatial_recording"] == "indices"
    assert rows[0]["observer_criterion"] == "activation"
    assert rows[0]["cache_mode"] == "cold"
    assert rows[0]["retention"] == "debug_artifacts"
