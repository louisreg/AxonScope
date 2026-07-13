from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import axonscope as axs
from axonscope.benchmarking import (
    BenchmarkOptions,
    benchmark,
    benchmark_array_metadata,
    benchmark_report,
    benchmark_span,
    disable_benchmark,
    enable_benchmark,
    record_benchmark_metadata,
)


def test_benchmark_options_are_public_and_do_not_expose_legacy_solver_suite():
    import axonscope.benchmarking as benchmarking

    assert axs.BenchmarkOptions is BenchmarkOptions
    assert "BenchmarkOptions" in axs.__all__
    assert "SolverBenchmarkCase" not in benchmarking.__all__
    assert "run_solver_benchmark_case" not in benchmarking.__all__


def test_benchmark_options_configure_enable_disable_style(tmp_path: Path):
    options = BenchmarkOptions(
        print_summary=False,
        save=True,
        memory_trace="rss",
        profile=True,
        profile_runtime="none",
    )

    session = enable_benchmark(tmp_path, options=options)
    try:
        with benchmark_span("example.stage", case="smoke"):
            record_benchmark_metadata(note="ok")
    finally:
        report = disable_benchmark(print_summary=False)

    assert report is not None
    assert session.config.profile is True
    assert session.config.profile_runtime == "none"
    assert report.events[0].name == "example.stage"
    assert report.events[0].metadata["case"] == "smoke"
    assert report.events[0].metadata["note"] == "ok"
    assert report.metadata["profile"]["enabled"] is True
    assert report.metadata["profile"]["runtime"] == "none"
    assert report.metadata["profile"]["active"] is False
    assert (tmp_path / "events.jsonl").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "metadata.json").is_file()


def test_benchmark_context_manager_records_shape_metadata(tmp_path: Path):
    values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    with benchmark(tmp_path, print_summary=False, save=True):
        with benchmark_span("arrays"):
            record_benchmark_metadata(
                **benchmark_array_metadata("values", values, role="input")
            )
        assert benchmark_report(print_report=False) is not None

    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["name"] == "arrays"
    assert payload["metadata"]["values"]["shape"] == [2, 3]
    assert payload["metadata"]["values"]["role"] == "input"
    assert benchmark_report(print_report=False) is None
