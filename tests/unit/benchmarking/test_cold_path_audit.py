from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.analysis.cold_path_audit import (
    classify_stage,
    main,
    read_run,
    summarize_groups,
)


def test_classify_stage_groups() -> None:
    assert classify_stage("curve.build_pool") == "pool_build"
    assert classify_stage("dispatch.build_plan") == "dispatch"
    assert classify_stage("runtime.prepare") == "runtime_prepare"
    assert classify_stage("inputs.extracellular") == "input_lowering"
    assert classify_stage("observer.plan") == "input_lowering"
    assert classify_stage("kernel.dispatch_jax") == "kernel"
    assert classify_stage("results.materialize_vm") == "result_assembly"
    assert classify_stage("results.to_public") == "result_assembly"
    assert classify_stage("something.custom") == "other"


def test_read_run_and_group_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    _write_fake_run(run_dir)

    rows = read_run(run_dir)
    assert [row.stage for row in rows][:2] == [
        "simulation.pool.total",
        "runtime.prepare",
    ]
    assert rows[0].context.case_name == "fake_case"
    assert rows[0].context.git_commit == "abc123"
    assert rows[0].context.n_axons == "64"

    grouped = summarize_groups(rows)
    runtime = next(row for row in grouped if row["group"] == "runtime_prepare")
    assert runtime["stage_count"] == 1
    assert runtime["self_ms_sum"] == 80.0
    assert runtime["rss_delta_mib_max"] == 12.0


def test_main_writes_csv_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    output = tmp_path / "audit"
    _write_fake_run(run_dir)

    assert main([str(run_dir), "--output", str(output), "--no-plots"]) == 0

    stage_rows = _read_csv(output / "cold_path_stage_rows.csv")
    group_rows = _read_csv(output / "cold_path_group_summary.csv")
    assert {row["stage"] for row in stage_rows} >= {
        "runtime.prepare",
        "kernel.dispatch_jax",
    }
    assert {row["group"] for row in group_rows} >= {"runtime_prepare", "kernel"}


def _write_fake_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    _write_csv(
        run_dir / "summary.csv",
        ("name", "count", "total_ms", "self_ms", "mean_ms", "max_ms"),
        [
            {
                "name": "simulation.pool.total",
                "count": "1",
                "total_ms": "100.0",
                "self_ms": "5.0",
                "mean_ms": "100.0",
                "max_ms": "100.0",
            },
            {
                "name": "runtime.prepare",
                "count": "1",
                "total_ms": "80.0",
                "self_ms": "80.0",
                "mean_ms": "80.0",
                "max_ms": "80.0",
            },
            {
                "name": "kernel.dispatch_jax",
                "count": "4",
                "total_ms": "20.0",
                "self_ms": "20.0",
                "mean_ms": "5.0",
                "max_ms": "8.0",
            },
        ],
    )
    _write_csv(
        run_dir / "memory_summary.csv",
        (
            "name",
            "rss_delta_mib_max",
            "rss_end_mib_max",
            "tracemalloc_peak_delta_bytes_max",
            "device_bytes_in_use_end_max",
            "nvidia_smi_memory_used_end_mib_max",
        ),
        [
            {
                "name": "runtime.prepare",
                "rss_delta_mib_max": "12.0",
                "rss_end_mib_max": "300.0",
                "tracemalloc_peak_delta_bytes_max": str(2 * 1024 * 1024),
                "device_bytes_in_use_end_max": str(4 * 1024 * 1024),
                "nvidia_smi_memory_used_end_mib_max": "",
            }
        ],
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "script": "threshold_curves",
                "case_name": "fake_case",
                "options": {
                    "platform": "cpu",
                    "n_axons": 64,
                    "nx": 101,
                    "tsim": 20.0,
                    "dt": 0.005,
                    "recording": "observer_only",
                    "precision": "fp32",
                    "memory_trace": "rss",
                    "profile": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "git": {"short_commit": "abc123", "dirty": False},
                "compute_device_class": "cpu",
                "compute_device_models": ["cpu"],
                "host_os": "test-os",
                "host_ram_total_gb": 32.0,
            }
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
