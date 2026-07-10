from __future__ import annotations

import csv
import json

import numpy as np
import pytest

import axonscope.runtime.benchmarking as instrumentation
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


def test_session_records_nested_events_and_writes_files(tmp_path):
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
    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "memory_summary.csv").is_file()
    assert list(csv.DictReader((tmp_path / "memory_summary.csv").open())) == []

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


def test_memory_trace_records_rss_and_tracemalloc_summary(tmp_path):
    enable_benchmark(
        tmp_path,
        print_summary=False,
        memory_trace="all",
        memory_top_n=3,
    )
    try:
        with benchmark_span("allocate.python"):
            _ = [bytearray(2048) for _ in range(4)]
    finally:
        report = disable_benchmark(print_summary=False)

    assert report is not None
    event = report.events[0]
    memory = event.metadata["memory"]
    assert "rss_start_mib" in memory
    assert "rss_end_mib" in memory
    assert "tracemalloc_current_start_bytes" in memory
    assert "tracemalloc_current_end_bytes" in memory
    assert "tracemalloc_peak_delta_bytes" in memory
    assert "tracemalloc_top" in memory
    assert (tmp_path / "memory_summary.csv").is_file()

    rows = list(csv.DictReader((tmp_path / "memory_summary.csv").open()))
    assert rows[0]["name"] == "allocate.python"
    assert "rss_delta_mib_sum" in rows[0]
    assert "tracemalloc_peak_delta_bytes_max" in rows[0]


def test_device_memory_trace_uses_best_effort_snapshots(tmp_path, monkeypatch):
    snapshots = iter(
        [
            {
                "device_bytes_in_use": 100,
                "device_peak_bytes_in_use": 120,
                "nvidia_smi_memory_used_mib": 1.0,
                "jax_devices": [],
                "nvidia_smi": {"available": False},
            },
            {
                "device_bytes_in_use": 140,
                "device_peak_bytes_in_use": 160,
                "nvidia_smi_memory_used_mib": 1.5,
                "jax_devices": [],
                "nvidia_smi": {"available": False},
            },
        ]
    )
    monkeypatch.setattr(instrumentation, "_device_memory_snapshot", lambda: next(snapshots))

    enable_benchmark(tmp_path, print_summary=False, memory_trace="device")
    try:
        with benchmark_span("device.stage"):
            pass
    finally:
        report = disable_benchmark(print_summary=False)

    assert report is not None
    memory = report.events[0].metadata["memory"]
    assert memory["device_bytes_in_use_start"] == 100
    assert memory["device_bytes_in_use_end"] == 140
    assert memory["device_bytes_in_use_delta"] == 40
    assert memory["nvidia_smi_memory_used_delta_mib"] == 0.5


def test_rss_memory_trace_tolerates_missing_process_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(instrumentation, "_current_rss_mib", lambda: None)

    enable_benchmark(tmp_path, print_summary=False, memory_trace="rss")
    try:
        with benchmark_span("rss.missing"):
            pass
    finally:
        report = disable_benchmark(print_summary=False)

    assert report is not None
    memory = report.events[0].metadata["memory"]
    assert memory["rss_start_mib"] is None
    assert memory["rss_end_mib"] is None
    assert memory["rss_delta_mib"] is None


def test_jax_device_memory_profile_metadata_is_recorded(tmp_path, monkeypatch):
    def fake_profile(self, name, event_id):
        return {
            "enabled": True,
            "stage": name,
            "path": str(tmp_path / f"{event_id}_{name}.prof"),
            "format": "pprof",
        }

    monkeypatch.setattr(
        instrumentation.BenchmarkSession,
        "_device_memory_profile",
        fake_profile,
    )

    enable_benchmark(
        tmp_path,
        print_summary=False,
        jax_device_memory_profile=True,
        jax_device_memory_profile_stages=("profile.me",),
    )
    try:
        with benchmark_span("profile.me"):
            pass
    finally:
        report = disable_benchmark(print_summary=False)

    assert report is not None
    profile = report.events[0].metadata["memory"]["jax_device_memory_profile"]
    assert profile["enabled"] is True
    assert profile["stage"] == "profile.me"
    assert profile["path"].endswith("0_profile.me.prof")
    assert (tmp_path / "memory_summary.csv").is_file()


def test_jax_device_memory_profile_stage_string_is_one_stage(tmp_path):
    session = enable_benchmark(
        tmp_path,
        print_summary=False,
        save=False,
        jax_device_memory_profile=True,
        jax_device_memory_profile_stages="kernel.wait",
    )
    try:
        assert session.metadata["jax_device_memory_profile_stages"] == ["kernel.wait"]
    finally:
        disable_benchmark(print_summary=False, save=False)


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


def test_enable_benchmark_accepts_profile_metadata(tmp_path):
    session = enable_benchmark(
        tmp_path,
        print_summary=False,
        save=False,
        profile=True,
        profile_backend="none",
        profile_create_perfetto=True,
    )
    try:
        profile = session.metadata["profile"]
    finally:
        disable_benchmark(print_summary=False, save=False)

    assert profile == {
        "enabled": True,
        "backend": "none",
        "output": str(tmp_path / "profiles" / "run"),
        "create_perfetto_trace": True,
        "create_perfetto_link": False,
        "active": False,
    }


def test_enable_benchmark_rejects_unsupported_profile_backend(tmp_path):
    with pytest.raises(ValueError, match="profile_backend"):
        enable_benchmark(
            tmp_path,
            print_summary=False,
            save=False,
            profile=True,
            profile_backend="other",
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
