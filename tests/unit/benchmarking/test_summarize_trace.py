from __future__ import annotations

import json
from pathlib import Path

from benchmark.analysis.trace_summary import _read_events, main


def test_trace_summary_reads_events_and_lists_artifacts(tmp_path: Path, capsys):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"name": "runtime.prepare", "duration_ns": 1_000_000}),
                json.dumps({"name": "kernel.wait", "duration_ns": 2_500_000}),
                json.dumps({"name": "kernel.wait", "duration_ns": 500_000}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_dir = tmp_path / "profiles" / "run"
    trace_dir.mkdir(parents=True)
    (trace_dir / "perfetto_trace.trace.json.gz").write_text("{}", encoding="utf-8")
    (tmp_path / "device.prof").write_text("profile", encoding="utf-8")

    assert [event["name"] for event in _read_events(events_path)] == [
        "runtime.prepare",
        "kernel.wait",
        "kernel.wait",
    ]
    assert main([str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "kernel.wait,2,3.000" in out
    assert "runtime.prepare,1,1.000" in out
    assert "perfetto_trace.trace.json.gz" in out
    assert "device.prof" in out


def test_trace_summary_returns_error_without_events(tmp_path: Path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "No events.jsonl" in capsys.readouterr().out
