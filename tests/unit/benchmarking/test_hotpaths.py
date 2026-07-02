from __future__ import annotations

import json

import numpy as np
import pytest

from axonscope.benchmarking import (
    benchmark,
    benchmark_array_metadata,
    benchmark_report,
    benchmark_span,
    benchmark_wait,
    disable_benchmark,
    enable_benchmark,
    record_benchmark_metadata,
    reset_benchmark,
)


def test_hotpath_session_records_nested_events_and_writes_files(tmp_path):
    enable_benchmark(tmp_path, print_summary=False)
    try:
        with benchmark_span("simulation.pool.total", pool_size=2):
            with benchmark_span("inputs.intracellular"):
                values = np.ones((2, 3), dtype=np.float32)
                record_benchmark_metadata(
                    **benchmark_array_metadata("iinj_mid", values, role="kernel_input")
                )
    finally:
        report = disable_benchmark(print_summary=False)

    assert report is not None
    assert [event.name for event in report.events] == [
        "inputs.intracellular",
        "simulation.pool.total",
    ]
    assert report.events[0].parent_event_id == report.events[1].event_id
    assert report.events[0].metadata["iinj_mid"]["shape"] == [2, 3]
    assert report.events[0].metadata["iinj_mid"]["dtype"] == "float32"
    assert report.events[0].metadata["iinj_mid"]["nbytes"] == 24
    assert (tmp_path / "events.jsonl").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "metadata.json").is_file()

    first_event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0])
    assert first_event["name"] == "inputs.intracellular"
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["compute_device_class"] in {"cpu", "gpu", "tpu", "unknown"}
    assert "compute_backend" in metadata
    assert "compute_device_models" in metadata
    assert "host_os" in metadata
    assert "host_ram_total_gb" in metadata
    assert "cpu_model" in metadata
    assert "gpu_models" in metadata
    assert "os" in metadata
    assert "cpu" in metadata
    assert "memory" in metadata
    assert "gpu" in metadata
    assert benchmark_report(print_report=False) is None


def test_benchmark_context_manager_disables_session(tmp_path):
    with benchmark(tmp_path, print_summary=False, save=False):
        with benchmark_span("dispatch.build_plan"):
            pass
        assert benchmark_report(print_report=False) is not None

    assert benchmark_report(print_report=False) is None


def test_nested_benchmark_sessions_are_rejected(tmp_path):
    enable_benchmark(tmp_path, print_summary=False, save=False)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            enable_benchmark(tmp_path / "nested", print_summary=False, save=False)
    finally:
        disable_benchmark(print_summary=False, save=False)


def test_enable_benchmark_accepts_jax_trace_metadata(tmp_path):
    session = enable_benchmark(
        tmp_path,
        print_summary=False,
        save=False,
        jax_trace=True,
        jax_trace_create_perfetto=True,
    )
    try:
        trace = session.metadata["jax_trace"]
    finally:
        disable_benchmark(print_summary=False, save=False)

    assert trace == {
        "enabled": True,
        "label": "benchmark",
        "trace_dir": str(tmp_path / "jax_traces" / "benchmark"),
        "create_perfetto_trace": True,
        "scope": "kernel",
    }


def test_enable_benchmark_rejects_unsupported_jax_trace_scope(tmp_path):
    with pytest.raises(ValueError, match="jax_trace_scope"):
        enable_benchmark(
            tmp_path,
            print_summary=False,
            save=False,
            jax_trace=True,
            jax_trace_scope="run",
        )


def test_reset_benchmark_clears_events(tmp_path):
    enable_benchmark(tmp_path, print_summary=False, save=False)
    try:
        with benchmark_span("dispatch.build_plan"):
            pass
        reset_benchmark()
        report = benchmark_report(print_report=False)
    finally:
        disable_benchmark(print_summary=False, save=False)

    assert report is not None
    assert report.events == ()


def test_benchmark_wait_uses_block_until_ready_only_when_enabled(tmp_path):
    class ReadyValue:
        def __init__(self) -> None:
            self.calls = 0

        def block_until_ready(self) -> None:
            self.calls += 1

    value = ReadyValue()
    benchmark_wait(value)
    assert value.calls == 0

    enable_benchmark(tmp_path, print_summary=False, save=False, sync_device=True)
    try:
        benchmark_wait(value)
    finally:
        disable_benchmark(print_summary=False, save=False)

    assert value.calls == 1
