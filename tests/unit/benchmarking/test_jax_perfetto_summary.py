from __future__ import annotations

import gzip
import json
from pathlib import Path

from benchmark.analysis.jax_perfetto_summary import (
    complete_events,
    load_trace_events,
    summarize_names,
    track_labels,
)


def test_perfetto_summary_reads_tracks_and_groups_complete_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "perfetto_trace.json.gz"
    payload = {
        "traceEvents": [
            {
                "ph": "M",
                "name": "process_name",
                "pid": 1,
                "args": {"name": "/device:GPU:0"},
            },
            {
                "ph": "M",
                "name": "thread_name",
                "pid": 1,
                "tid": 2,
                "args": {"name": "Stream #1"},
            },
            {"ph": "X", "name": "solve", "pid": 1, "tid": 2, "dur": 4000},
            {"ph": "X", "name": "solve", "pid": 1, "tid": 2, "dur": 2000},
            {"ph": "X", "name": "other", "pid": 1, "tid": 2, "dur": 100},
        ]
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    events = load_trace_events(path)
    processes, threads = track_labels(events)
    selected = complete_events(events, pattern="solve", min_duration_ms=1.0)
    rows = summarize_names(selected)

    assert processes == {1: "/device:GPU:0"}
    assert threads == {(1, 2): "Stream #1"}
    assert len(rows) == 1
    assert rows[0]["count"] == 2
    assert rows[0]["total_ms"] == 6.0
    assert rows[0]["max_ms"] == 4.0
