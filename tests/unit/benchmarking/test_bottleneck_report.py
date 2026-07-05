from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.analysis.bottleneck_report import (
    main,
    read_event_rows,
    summarize_groups,
    summarize_stages,
)


def test_event_rows_compute_exclusive_self_time_and_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    _write_fake_event_run(run_dir)

    rows = read_event_rows(run_dir)
    by_stage = {row.stage: row for row in rows}

    assert by_stage["curve.simulate"].self_ms == 50.0
    assert by_stage["runtime.prepare"].self_ms == 80.0
    assert by_stage["runtime.prepare"].cache_misses == 1
    assert by_stage["runtime.prepare"].cache_hits == 1
    assert by_stage["runtime.prepare"].rss_delta_mib == 12.0


def test_summaries_rank_stage_and_group_bottlenecks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    _write_fake_event_run(run_dir)

    events = read_event_rows(run_dir)
    stages = summarize_stages(events)
    groups = summarize_groups(stages)

    assert stages[0]["stage"] == "curve.build_pool"
    runtime = next(row for row in stages if row["stage"] == "runtime.prepare")
    assert runtime["self_ms"] == 80.0
    assert runtime["workflow_share"] == 80.0 / 300.0
    runtime_group = next(row for row in groups if row["group"] == "runtime_prepare")
    assert runtime_group["event_count"] == 1
    assert runtime_group["cache_misses"] == 1


def test_main_writes_report_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    output = tmp_path / "report"
    _write_fake_event_run(run_dir)

    assert main([str(run_dir), "--output", str(output), "--top-n", "5"]) == 0

    event_rows = _read_csv(output / "bottleneck_event_rows.csv")
    stage_rows = _read_csv(output / "bottleneck_stage_rank.csv")
    group_rows = _read_csv(output / "bottleneck_group_rank.csv")
    report = (output / "bottleneck_report.md").read_text(encoding="utf-8")

    assert {row["stage"] for row in event_rows} >= {"runtime.prepare", "kernel.enqueue"}
    assert {row["stage"] for row in stage_rows} >= {"runtime.prepare", "kernel.enqueue"}
    assert {row["group"] for row in group_rows} >= {"runtime_prepare", "kernel"}
    assert "Cache Signals" in report
    assert "runtime.prepare" in report


def _write_fake_event_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    events = [
        {
            "event_id": 0,
            "parent_event_id": None,
            "depth": 0,
            "name": "curve.build_pool",
            "duration_ms": 100.0,
            "metadata": {
                "phase": "repeat",
                "repeat": 0,
                "iteration": -2,
                "curve": "activation_threshold",
                "memory": {"rss_delta_mib": 5.0, "rss_end_mib": 200.0},
            },
        },
        {
            "event_id": 1,
            "parent_event_id": None,
            "depth": 0,
            "name": "curve.simulate",
            "duration_ms": 200.0,
            "metadata": {"phase": "repeat", "repeat": 0, "iteration": 0},
        },
        {
            "event_id": 2,
            "parent_event_id": 1,
            "depth": 1,
            "name": "runtime.prepare",
            "duration_ms": 80.0,
            "metadata": {
                "batch_runtime_cache": "miss",
                "membrane_source_cache": ["hit"],
                "memory": {
                    "rss_delta_mib": 12.0,
                    "rss_end_mib": 300.0,
                    "device_bytes_in_use_end": 2 * 1024 * 1024,
                    "nvidia_smi_memory_used_end_mib": 512.0,
                },
            },
        },
        {
            "event_id": 3,
            "parent_event_id": 1,
            "depth": 1,
            "name": "kernel.enqueue",
            "duration_ms": 70.0,
            "metadata": {"memory": {"device_bytes_in_use_delta": 1024 * 1024}},
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "script": "threshold_curves",
                "case_name": "fake_case",
                "options": {
                    "platform": "gpu",
                    "n_axons": 64,
                    "nx": 31,
                    "tsim": 2.0,
                    "dt": 0.02,
                    "recording": "observer_only",
                    "precision": "fp32",
                    "memory_trace": "device",
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
                "compute_device_class": "gpu",
                "compute_device_models": ["test-gpu"],
                "host_os": "test-os",
                "host_ram_total_gb": 32.0,
            }
        ),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
