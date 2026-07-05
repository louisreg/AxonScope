from __future__ import annotations

import json
from pathlib import Path

from benchmark.campaigns.time_chunk_sweep import main as run_time_chunk_sweep
from benchmark.campaigns.time_chunk_sweep import summarize_run


def test_time_chunk_sweep_dry_run_writes_manifest(tmp_path: Path, capsys):
    assert (
        run_time_chunk_sweep(
            [
                "--script",
                "recruitment_curves",
                "--preset",
                "quick",
                "--platform",
                "cpu",
                "--policies",
                "default,unchunked,100",
                "--output",
                str(tmp_path),
                "--dry-run",
                "--recording",
                "observer_only",
                "--amplitude-count",
                "1",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    manifest = json.loads(
        (tmp_path / "time_chunk_sweep_manifest.json").read_text(encoding="utf-8")
    )

    assert "benchmark/run.py" in out
    assert manifest["policies"] == ["default", "unchunked", "100"]
    assert [Path(run["run_dir"]).name for run in manifest["runs"]] == [
        "default",
        "unchunked",
        "chunk_100",
    ]
    commands = [" ".join(run["command"]) for run in manifest["runs"]]
    assert "--time-chunk-steps default" in commands[0]
    assert "--time-chunk-steps unchunked" in commands[1]
    assert "--time-chunk-steps 100" in commands[2]
    assert all("--recording observer_only" in command for command in commands)
    assert all("--amplitude-count 1" in command for command in commands)


def test_time_chunk_sweep_summarizes_kernel_events(tmp_path: Path):
    manifest = {
        "script": "recruitment_curves",
        "case_name": "case",
        "options": {
            "platform": "cpu",
            "recording": "observer_only",
            "n_axons": 1,
            "nx": 21,
            "tsim": 2.0,
            "dt": 0.02,
            "precision": "fp32",
            "time_chunk_policy": "explicit",
            "time_chunk_steps": 100,
        },
    }
    events = [
        {
            "event_id": 1,
            "parent_event_id": None,
            "name": "curve.simulate",
            "duration_ms": 100.0,
            "metadata": {"phase": "repeat"},
        },
        {
            "event_id": 2,
            "parent_event_id": 1,
            "name": "dispatch.group.total",
            "duration_ms": 90.0,
            "metadata": {},
        },
        {
            "event_id": 3,
            "parent_event_id": 2,
            "name": "kernel.enqueue",
            "duration_ms": 60.0,
            "metadata": {},
        },
        {
            "event_id": 4,
            "parent_event_id": 3,
            "name": "kernel.dispatch_jax",
            "duration_ms": 20.0,
            "metadata": {
                "chunk_count": 1,
                "chunk_steps": 100,
                "observer_state_scope": "chunk",
                "time_chunk_steps": 100,
                "memory": {"rss_end_mib": 321.0},
            },
        },
        {
            "event_id": 5,
            "parent_event_id": 3,
            "name": "kernel.combine_observer_chunks",
            "duration_ms": 7.0,
            "metadata": {},
        },
        {
            "event_id": 6,
            "parent_event_id": 2,
            "name": "kernel.wait",
            "duration_ms": 5.0,
            "metadata": {},
        },
        {
            "event_id": 7,
            "parent_event_id": 3,
            "name": "kernel.finalize_observer",
            "duration_ms": 2.0,
            "metadata": {},
        },
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    row = summarize_run(tmp_path, policy="100")

    assert row["policy"] == "100"
    assert row["time_chunk_policy"] == "explicit"
    assert row["time_chunk_steps"] == 100
    assert row["dispatch_time_chunk_steps"] == "100"
    assert row["observer_state_scopes"] == "chunk"
    assert row["dispatch_chunk_steps"] == "100"
    assert row["dispatch_chunk_count_max"] == "1"
    assert row["repeat_curve_simulate_ms"] == 100.0
    assert row["repeat_kernel_enqueue_ms"] == 60.0
    assert row["repeat_kernel_combine_observer_chunks_ms"] == 7.0
    assert row["repeat_kernel_dispatch_jax_ms"] == 20.0
    assert row["repeat_kernel_wait_ms"] == 5.0
    assert row["repeat_kernel_finalize_observer_ms"] == 2.0
    assert row["rss_end_mib_max"] == "321"
