from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.analysis.run_pool_detail import write_run_pool_detail
from benchmark.protocols import recruitment_amplitude_batch


def test_p14_realistic_defaults_match_reference_workload() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        ["--workload", "p14_realistic", "--cable", "double"]
    )

    recruitment_amplitude_batch._resolve_workload_args(args)

    assert args.axon_count == 196
    assert args.duration_ms == 3.0
    assert args.dt_ms == 0.001
    assert args.amplitudes == recruitment_amplitude_batch.P14_REALISTIC_AMPLITUDES_UA
    assert len(args.amplitudes) == 21


def test_cable_counts_preserve_requested_population_size() -> None:
    assert recruitment_amplitude_batch._cable_counts("single", 196) == (196, 0)
    assert recruitment_amplitude_batch._cable_counts("double", 196) == (0, 196)
    assert recruitment_amplitude_batch._cable_counts("mixed", 197) == (98, 99)


def test_p14_dry_run_records_workload_shape(tmp_path: Path) -> None:
    assert (
        recruitment_amplitude_batch.main(
            [
                "--workload",
                "p14_realistic",
                "--cable",
                "single",
                "--policies",
                "1,2,full",
                "--dry-run",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["workload"] == "p14_realistic"
    assert manifest["cable"] == "single"
    assert manifest["n_axons"] == 196
    assert manifest["amplitudes_uA"] == list(
        recruitment_amplitude_batch.P14_REALISTIC_AMPLITUDES_UA
    )
    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert [row["policy"] for row in rows] == ["1", "2", "full"]
    assert {row["amplitude_count"] for row in rows} == {"21"}


def test_run_pool_detail_splits_single_and_double_group_timings(
    tmp_path: Path,
) -> None:
    events = [
        {"event_id": 1, "name": "simulation.run_pool", "duration_ms": 100.0},
        {
            "event_id": 2,
            "parent_event_id": 1,
            "name": "dispatch.group.total",
            "duration_ms": 80.0,
            "metadata": {"mode": "double"},
        },
        {
            "event_id": 3,
            "parent_event_id": 2,
            "name": "kernel.dispatch_jax",
            "duration_ms": 20.0,
        },
        {
            "event_id": 4,
            "parent_event_id": 2,
            "name": "kernel.wait",
            "duration_ms": 5.0,
        },
        {
            "event_id": 5,
            "parent_event_id": 1,
            "name": "dispatch.group.total",
            "duration_ms": 15.0,
            "metadata": {"mode": "single"},
        },
        {
            "event_id": 6,
            "parent_event_id": 5,
            "name": "kernel.dispatch_jax",
            "duration_ms": 2.0,
        },
        {
            "event_id": 7,
            "parent_event_id": 5,
            "name": "kernel.wait",
            "duration_ms": 1.0,
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    write_run_pool_detail(
        tmp_path,
        amplitudes=(300.0,),
        batch_amplitudes=False,
    )

    rows = {
        row["mode"]: row
        for row in csv.DictReader((tmp_path / "run_pool_detail.csv").open())
    }
    assert rows["all"]["amplitudes_uA"] == "300"
    assert float(rows["all"]["kernel_solver_ms"]) == 28.0
    assert float(rows["double"]["group_ms"]) == 80.0
    assert float(rows["double"]["kernel_solver_ms"]) == 25.0
    assert float(rows["double"]["kernel_wait_pct_solver"]) == 20.0
    assert float(rows["single"]["kernel_solver_ms"]) == 3.0

