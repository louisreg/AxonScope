from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from benchmark.runtime.summarize_trace import (
    collect_trace_events,
    discover_trace_files,
    summarize_events,
    write_summary_csv,
    write_summary_json,
)


def test_summarize_trace_complete_events(tmp_path: Path):
    trace_path = tmp_path / "run.trace.json.gz"
    _write_trace(
        trace_path,
        [
            {"name": "benchmark/hh/build_axon", "ph": "X", "ts": 0, "dur": 1000, "pid": 1, "tid": 1},
            {"name": "benchmark/hh/Solver/first_solve", "ph": "X", "ts": 1000, "dur": 2500, "pid": 1, "tid": 1},
            {"name": "benchmark/hh/Solver/measured_solve", "ph": "X", "ts": 3500, "dur": 500, "pid": 1, "tid": 1},
            {"name": "unrelated", "ph": "X", "ts": 4000, "dur": 999999, "pid": 1, "tid": 1},
        ],
    )

    assert discover_trace_files([tmp_path]) == [trace_path]

    events = collect_trace_events([tmp_path])
    rows = summarize_events(events)
    by_phase = {row.phase: row for row in rows}

    assert len(events) == 3
    assert by_phase["build_axon"].case_name == "hh"
    assert by_phase["build_axon"].solver_name == ""
    assert by_phase["build_axon"].total_ms == 1.0
    assert by_phase["first_solve"].solver_name == "Solver"
    assert by_phase["first_solve"].total_ms == 2.5
    assert by_phase["measured_solve"].total_ms == 0.5


def test_summarize_trace_begin_end_events_and_exports(tmp_path: Path):
    trace_path = tmp_path / "run.trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "benchmark/mrg/Solver/measured_materialize", "ph": "B", "ts": 100, "pid": 1, "tid": 7},
                    {"name": "ignored_end_name", "ph": "E", "ts": 350, "pid": 1, "tid": 7},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_events(collect_trace_events([trace_path]))
    assert len(rows) == 1
    assert rows[0].case_name == "mrg"
    assert rows[0].solver_name == "Solver"
    assert rows[0].phase == "measured_materialize"
    assert math.isclose(rows[0].total_ms, 0.25)

    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"
    write_summary_csv(rows, csv_path)
    write_summary_json(rows, json_path)

    assert "measured_materialize" in csv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert math.isclose(payload["rows"][0]["total_ms"], 0.25)


def test_summarize_trace_pool_annotations(tmp_path: Path):
    trace_path = tmp_path / "pool.trace.json.gz"
    _write_trace(
        trace_path,
        [
            {"name": "pool/double/build_vstim", "ph": "X", "ts": 0, "dur": 1200, "pid": 1, "tid": 1},
            {"name": "pool/double/batch_first", "ph": "X", "ts": 1200, "dur": 300, "pid": 1, "tid": 1},
            {"name": "benchmark/hh/build_axon", "ph": "X", "ts": 1500, "dur": 700, "pid": 1, "tid": 1},
        ],
    )

    rows = summarize_events(collect_trace_events([trace_path], pattern="pool/"))
    by_phase = {row.phase: row for row in rows}

    assert set(by_phase) == {"build_vstim", "batch_first"}
    assert by_phase["build_vstim"].case_name == "pool"
    assert by_phase["build_vstim"].solver_name == "double"
    assert math.isclose(by_phase["build_vstim"].total_ms, 1.2)


def _write_trace(path: Path, events):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)
