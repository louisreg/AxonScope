"""Opt-in hotpath instrumentation for developer performance diagnostics."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, is_dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np


_ACTIVE_BENCHMARK_SESSION: ContextVar["BenchmarkSession | None"] = ContextVar(
    "axonscope_active_benchmark_session",
    default=None,
)


@dataclass(frozen=True)
class BenchmarkOptions:
    """User-facing options for benchmark instrumentation sessions."""

    print_summary: bool = True
    save: bool = True
    sync_device: bool = True
    record_shapes: bool = True
    record_memory: bool = True
    level: str = "hotpaths"
    memory_trace: str = "off"
    memory_top_n: int = 0
    profile: bool = False
    profile_backend: str = "auto"
    profile_output: Path | None = None
    profile_create_perfetto: bool = False
    profile_create_perfetto_link: bool = False
    jax_device_memory_profile: bool = False
    jax_device_memory_profile_stages: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for one hotpath benchmark session."""

    output_dir: Path
    print_summary: bool = True
    save: bool = True
    sync_device: bool = True
    record_shapes: bool = True
    record_memory: bool = True
    level: str = "hotpaths"
    memory_trace: str = "off"
    memory_top_n: int = 0
    profile: bool = False
    profile_backend: str = "auto"
    profile_output: Path | None = None
    profile_create_perfetto: bool = False
    profile_create_perfetto_link: bool = False
    jax_device_memory_profile: bool = False
    jax_device_memory_profile_stages: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkEvent:
    """One completed timed span."""

    event_id: int
    name: str
    parent_event_id: int | None
    depth: int
    start_ns: int
    end_ns: int
    duration_ns: int
    metadata: Mapping[str, Any]

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""

        return self.duration_ns / 1e6

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event record."""

        return {
            "event_id": self.event_id,
            "name": self.name,
            "parent_event_id": self.parent_event_id,
            "depth": self.depth,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkSummaryRow:
    """Aggregated timing row for one event name."""

    name: str
    count: int
    total_ms: float
    self_ms: float
    mean_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary row."""

        return {
            "name": self.name,
            "count": self.count,
            "total_ms": self.total_ms,
            "self_ms": self.self_ms,
            "mean_ms": self.mean_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregated report for a benchmark session."""

    events: tuple[BenchmarkEvent, ...]
    summary: tuple[BenchmarkSummaryRow, ...]
    metadata: Mapping[str, Any]

    def format(self) -> str:
        """Format a compact text report."""

        if not self.events:
            return "AxonScope benchmark: no events recorded."

        lines = [
            f"AxonScope benchmark: {len(self.events)} events",
            "stage                         count    total ms     self ms     mean ms      max ms",
            "----------------------------  -----  ----------  ----------  ----------  ----------",
        ]
        for row in self.summary:
            lines.append(
                f"{row.name[:28]:28}  {row.count:5d}  "
                f"{row.total_ms:10.3f}  {row.self_ms:10.3f}  "
                f"{row.mean_ms:10.3f}  {row.max_ms:10.3f}"
            )
        return "\n".join(lines)

    def save(self, output_dir: str | Path) -> None:
        """Write raw events, aggregate summaries, and metadata to `output_dir`."""

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        with (path / "events.jsonl").open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True))
                handle.write("\n")

        with (path / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("name", "count", "total_ms", "self_ms", "mean_ms", "max_ms"),
            )
            writer.writeheader()
            for row in self.summary:
                writer.writerow(row.to_dict())

        memory_summary = _summarize_memory_events(self.events)
        if memory_summary:
            with (path / "memory_summary.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=_MEMORY_SUMMARY_FIELDNAMES)
                writer.writeheader()
                writer.writerows(memory_summary)

        with (path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(dict(self.metadata), handle, indent=2, sort_keys=True)
            handle.write("\n")


@dataclass
class _ActiveSpan:
    event_id: int
    name: str
    parent_event_id: int | None
    depth: int
    start_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_start: dict[str, Any] = field(default_factory=dict)
    tracemalloc_start_snapshot: Any | None = None


@dataclass
class BenchmarkSession:
    """Mutable state for one active benchmark session."""

    config: BenchmarkConfig
    events: list[BenchmarkEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    _stack: list[_ActiveSpan] = field(default_factory=list)
    _next_event_id: int = 0
    _token: Token[BenchmarkSession | None] | None = None
    _jax_trace_active: bool = False
    _profile_handle: Any | None = None
    _tracemalloc_started_by_session: bool = False

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        """Time one named stage and preserve original exceptions."""

        if not self.active:
            yield
            return

        event_id = self._next_event_id
        self._next_event_id += 1
        parent_event_id = self._stack[-1].event_id if self._stack else None
        memory_start, tracemalloc_start_snapshot = _memory_start_snapshot(self.config)
        active = _ActiveSpan(
            event_id=event_id,
            name=name,
            parent_event_id=parent_event_id,
            depth=len(self._stack),
            start_ns=time.perf_counter_ns(),
            metadata=_json_safe_dict(metadata),
            memory_start=memory_start,
            tracemalloc_start_snapshot=tracemalloc_start_snapshot,
        )
        self._stack.append(active)
        failed = False
        try:
            with self._span_jax_trace(name):
                yield
        except BaseException as exc:
            failed = True
            active.metadata.update(
                {
                    "failed": True,
                    "exception_type": type(exc).__name__,
                }
            )
            raise
        finally:
            end_ns = time.perf_counter_ns()
            popped = self._stack.pop()
            if popped is not active:
                raise RuntimeError("benchmark span stack became inconsistent.")
            if failed:
                active.metadata.setdefault("failed", True)
            memory = _memory_end_metadata(
                self.config,
                start=active.memory_start,
                tracemalloc_start_snapshot=active.tracemalloc_start_snapshot,
            )
            profile = self._save_jax_device_memory_profile(
                name=name,
                event_id=event_id,
            )
            if profile:
                memory["jax_device_memory_profile"] = profile
            if memory:
                active.metadata["memory"] = _json_safe_dict(memory)
            self.events.append(
                BenchmarkEvent(
                    event_id=event_id,
                    name=name,
                    parent_event_id=parent_event_id,
                    depth=active.depth,
                    start_ns=active.start_ns,
                    end_ns=end_ns,
                    duration_ns=end_ns - active.start_ns,
                    metadata=dict(active.metadata),
                )
            )

    @contextmanager
    def _span_jax_trace(self, name: str) -> Iterator[None]:
        trace = self.metadata.get("jax_trace")
        if (
            name != "kernel.enqueue"
            or self._jax_trace_active
            or not isinstance(trace, Mapping)
            or not trace.get("enabled", False)
            or trace.get("scope", "kernel") != "kernel"
        ):
            yield
            return

        trace_dir = Path(str(trace["trace_dir"]))
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._jax_trace_active = True
        try:
            from axonscope.backends.execution import (
                benchmark_profile_trace,
                benchmark_trace_annotation,
            )

            with benchmark_profile_trace(
                "jax",
                trace_dir,
                create_perfetto_trace=bool(trace.get("create_perfetto_trace", False)),
            ):
                with benchmark_trace_annotation("kernel.enqueue"):
                    yield
        finally:
            self._jax_trace_active = False

    def _save_jax_device_memory_profile(
        self,
        *,
        name: str,
        event_id: int,
    ) -> dict[str, Any]:
        if not _should_save_device_memory_profile(self.config, name):
            return {}
        profile_dir = self.config.output_dir / "device_memory_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{event_id:04d}_{_safe_stage_name(name)}.prof"
        try:
            from axonscope.backends.execution import (
                benchmark_save_device_memory_profile,
            )

            benchmark_save_device_memory_profile(profile_path, backend="jax")
        except Exception as exc:  # pragma: no cover - backend-dependent.
            return {
                "enabled": True,
                "stage": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "enabled": True,
            "stage": name,
            "path": str(profile_path),
            "format": "pprof",
            "view_hint": f"pprof --web {profile_path}",
        }

    def record_metadata(self, **metadata: Any) -> None:
        """Attach metadata to the currently active span."""

        if not self._stack:
            return
        self._stack[-1].metadata.update(_json_safe_dict(metadata))

    def report(self) -> BenchmarkReport:
        """Return an aggregate report without disabling the session."""

        events = tuple(self.events)
        return BenchmarkReport(
            events=events,
            summary=_summarize_events(events),
            metadata=dict(self.metadata),
        )

    def reset(self) -> None:
        """Clear recorded events while preserving configuration and metadata."""

        self.events.clear()
        self._stack.clear()
        self._next_event_id = 0

    def finish(
        self,
        *,
        print_summary: bool | None = None,
        save: bool | None = None,
    ) -> BenchmarkReport:
        """Finalize the session and optionally print/save the report."""

        self.active = False
        self._stop_profile()
        report = self.report()
        should_save = self.config.save if save is None else bool(save)
        should_print = self.config.print_summary if print_summary is None else bool(print_summary)
        if should_save:
            report.save(self.config.output_dir)
        if should_print:
            print(report.format())
        if self._tracemalloc_started_by_session:
            tracemalloc.stop()
            self._tracemalloc_started_by_session = False
        return report

    def start_profile(self) -> None:
        """Start the backend profiler for whole-session traces, if requested."""

        if not self.config.profile or self.config.profile_backend == "none":
            return
        profile_output = self.config.profile_output or self.config.output_dir / "profiles" / "run"
        self.metadata["profile"] = {
            "enabled": True,
            "backend": self.config.profile_backend,
            "output": str(profile_output),
            "create_perfetto_trace": bool(self.config.profile_create_perfetto),
            "create_perfetto_link": bool(self.config.profile_create_perfetto_link),
        }
        try:
            from axonscope.backends.execution import benchmark_profile_start

            self._profile_handle = benchmark_profile_start(
                self.config.profile_backend,
                profile_output,
                create_perfetto_trace=self.config.profile_create_perfetto,
                create_perfetto_link=self.config.profile_create_perfetto_link,
            )
        except Exception as exc:  # pragma: no cover - backend-dependent.
            self.metadata["profile"].update(
                {
                    "active": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            self.metadata["profile"]["active"] = self._profile_handle is not None

    def _stop_profile(self) -> None:
        if self._profile_handle is None:
            return
        try:
            from axonscope.backends.execution import benchmark_profile_stop

            metadata = benchmark_profile_stop(self._profile_handle)
        except Exception as exc:  # pragma: no cover - backend-dependent.
            metadata = {"stop_error": f"{type(exc).__name__}: {exc}"}
        profile = self.metadata.setdefault("profile", {})
        if isinstance(profile, dict):
            profile.update(_json_safe_dict(metadata))
            profile["active"] = False
        self._profile_handle = None


def enable_benchmark(
    output_dir: str | Path,
    *,
    options: BenchmarkOptions | None = None,
    print_summary: bool = True,
    save: bool = True,
    reset: bool = True,
    sync_device: bool = True,
    record_shapes: bool = True,
    record_memory: bool = True,
    level: str = "hotpaths",
    profile: bool | None = None,
    profile_backend: str | None = None,
    profile_output: str | Path | None = None,
    profile_create_perfetto: bool | None = None,
    profile_create_perfetto_link: bool | None = None,
    jax_trace: bool = False,
    jax_trace_dir: str | Path | None = None,
    jax_trace_create_perfetto: bool = False,
    jax_trace_scope: str = "kernel",
    memory_trace: str = "off",
    memory_top_n: int = 0,
    jax_device_memory_profile: bool = False,
    jax_device_memory_profile_stages: Sequence[str] | None = None,
) -> BenchmarkSession:
    """Enable hotpath instrumentation for subsequent AxonScope calls."""

    if options is not None:
        print_summary = options.print_summary
        save = options.save
        sync_device = options.sync_device
        record_shapes = options.record_shapes
        record_memory = options.record_memory
        level = options.level
        memory_trace = options.memory_trace
        memory_top_n = options.memory_top_n
        if profile is None:
            profile = options.profile
        if profile_backend is None:
            profile_backend = options.profile_backend
        if profile_output is None:
            profile_output = options.profile_output
        if profile_create_perfetto is None:
            profile_create_perfetto = options.profile_create_perfetto
        if profile_create_perfetto_link is None:
            profile_create_perfetto_link = options.profile_create_perfetto_link
        jax_device_memory_profile = options.jax_device_memory_profile
        if not jax_device_memory_profile_stages:
            jax_device_memory_profile_stages = options.jax_device_memory_profile_stages

    profile = bool(profile) if profile is not None else False
    profile_backend = str(profile_backend or "auto").lower()
    profile_create_perfetto = bool(profile_create_perfetto) if profile_create_perfetto is not None else False
    profile_create_perfetto_link = (
        bool(profile_create_perfetto_link)
        if profile_create_perfetto_link is not None
        else False
    )

    active = _ACTIVE_BENCHMARK_SESSION.get()
    if active is not None and active.active:
        raise RuntimeError("An AxonScope benchmark session is already active.")
    if level != "hotpaths":
        raise ValueError("Only level='hotpaths' is supported for now.")
    if profile_backend not in {"auto", "jax", "none"}:
        raise ValueError("profile_backend must be one of: auto, jax, none.")
    if jax_trace_scope not in {"kernel"}:
        raise ValueError("Only jax_trace_scope='kernel' is supported by enable_benchmark.")
    if memory_trace not in {"off", "rss", "tracemalloc", "device", "all"}:
        raise ValueError("memory_trace must be one of: off, rss, tracemalloc, device, all.")
    if memory_top_n < 0:
        raise ValueError("memory_top_n must be >= 0.")

    path = Path(output_dir)
    if save:
        path.mkdir(parents=True, exist_ok=True)
    if jax_device_memory_profile_stages is None:
        profile_stages = ("kernel.wait",)
    elif isinstance(jax_device_memory_profile_stages, str):
        profile_stages = (jax_device_memory_profile_stages,)
    else:
        profile_stages = tuple(str(stage) for stage in jax_device_memory_profile_stages)

    session = BenchmarkSession(
        config=BenchmarkConfig(
            output_dir=path,
            print_summary=print_summary,
            save=save,
            sync_device=sync_device,
            record_shapes=record_shapes,
            record_memory=record_memory,
            level=level,
            memory_trace=memory_trace,
            memory_top_n=int(memory_top_n),
            profile=profile,
            profile_backend=profile_backend,
            profile_output=Path(profile_output) if profile_output is not None else None,
            profile_create_perfetto=profile_create_perfetto,
            profile_create_perfetto_link=profile_create_perfetto_link,
            jax_device_memory_profile=bool(jax_device_memory_profile),
            jax_device_memory_profile_stages=profile_stages,
        ),
        metadata=_collect_benchmark_metadata(path),
    )
    if _trace_uses_tracemalloc(session.config) and not tracemalloc.is_tracing():
        tracemalloc.start(max(int(memory_top_n), 1))
        session._tracemalloc_started_by_session = True
    session.metadata.update(
        {
            "memory_trace": memory_trace,
            "memory_top_n": int(memory_top_n),
            "jax_device_memory_profile": bool(jax_device_memory_profile),
            "jax_device_memory_profile_stages": list(profile_stages),
            "memory_summary_file": "memory_summary.csv"
            if _trace_uses_measured_memory(session.config)
            else None,
        }
    )
    if profile:
        session.metadata["profile"] = {
            "enabled": True,
            "backend": profile_backend,
            "output": str(session.config.profile_output or path / "profiles" / "run"),
            "create_perfetto_trace": profile_create_perfetto,
            "create_perfetto_link": profile_create_perfetto_link,
            "active": False,
        }
    if jax_trace:
        trace_dir = Path(jax_trace_dir) if jax_trace_dir is not None else path / "jax_traces" / "benchmark"
        session.metadata["jax_trace"] = {
            "enabled": True,
            "label": "benchmark",
            "trace_dir": str(trace_dir),
            "create_perfetto_trace": bool(jax_trace_create_perfetto),
            "scope": "kernel",
        }
    if reset:
        session.reset()
    session._token = _ACTIVE_BENCHMARK_SESSION.set(session)
    session.start_profile()
    return session


def disable_benchmark(
    *,
    print_summary: bool | None = None,
    save: bool | None = None,
) -> BenchmarkReport | None:
    """Disable the active benchmark session and return its report."""

    session = _ACTIVE_BENCHMARK_SESSION.get()
    if session is None:
        return None
    try:
        report = session.finish(print_summary=print_summary, save=save)
    finally:
        if session._token is not None:
            _ACTIVE_BENCHMARK_SESSION.reset(session._token)
            session._token = None
        else:
            _ACTIVE_BENCHMARK_SESSION.set(None)
    return report


def benchmark_report(
    *,
    print_report: bool = True,
    save: bool = False,
) -> BenchmarkReport | None:
    """Return a report for the active benchmark session."""

    session = _ACTIVE_BENCHMARK_SESSION.get()
    if session is None:
        return None
    report = session.report()
    if save:
        report.save(session.config.output_dir)
    if print_report:
        print(report.format())
    return report


def reset_benchmark() -> None:
    """Clear events from the active benchmark session."""

    session = _ACTIVE_BENCHMARK_SESSION.get()
    if session is not None:
        session.reset()


@contextmanager
def benchmark(output_dir: str | Path, **options: Any) -> Iterator[BenchmarkSession]:
    """Context-manager form of `enable_benchmark`/`disable_benchmark`."""

    session = enable_benchmark(output_dir, **options)
    try:
        yield session
    finally:
        disable_benchmark()


def active_benchmark_session() -> BenchmarkSession | None:
    """Return the active benchmark session, if any."""

    session = _ACTIVE_BENCHMARK_SESSION.get()
    if session is None or not session.active:
        return None
    return session


def benchmark_span(name: str, **metadata: Any):
    """Return a no-op context manager when benchmarking is disabled."""

    session = active_benchmark_session()
    if session is None:
        return nullcontext()
    return session.span(name, **metadata)


def record_benchmark_metadata(**metadata: Any) -> None:
    """Attach metadata to the active span when benchmarking is enabled."""

    session = active_benchmark_session()
    if session is not None:
        session.record_metadata(**metadata)


def benchmark_array_metadata(name: str, array: Any, *, role: str | None = None) -> dict[str, Any]:
    """Return JSON-safe shape/dtype/device metadata for `array` if enabled."""

    session = active_benchmark_session()
    if session is None or not session.config.record_shapes:
        return {}
    return {name: _array_metadata(array, role=role, session=session)}


def benchmark_wait(value: Any) -> Any:
    """Synchronize JAX-like arrays only when the active session requests it."""

    session = active_benchmark_session()
    if session is None or not session.config.sync_device:
        return value
    _block_until_ready(value, seen=set())
    return value


def _summarize_events(events: Sequence[BenchmarkEvent]) -> tuple[BenchmarkSummaryRow, ...]:
    child_duration_by_parent: dict[int, int] = defaultdict(int)
    for event in events:
        if event.parent_event_id is not None:
            child_duration_by_parent[event.parent_event_id] += event.duration_ns

    aggregates: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    for index, event in enumerate(events):
        first_seen.setdefault(event.name, index)
        self_ns = max(event.duration_ns - child_duration_by_parent[event.event_id], 0)
        row = aggregates.setdefault(
            event.name,
            {"count": 0, "total_ns": 0, "self_ns": 0, "max_ns": 0},
        )
        row["count"] += 1
        row["total_ns"] += event.duration_ns
        row["self_ns"] += self_ns
        row["max_ns"] = max(row["max_ns"], event.duration_ns)

    rows = []
    for name, values in aggregates.items():
        count = int(values["count"])
        total_ms = float(values["total_ns"]) / 1e6
        rows.append(
            BenchmarkSummaryRow(
                name=name,
                count=count,
                total_ms=total_ms,
                self_ms=float(values["self_ns"]) / 1e6,
                mean_ms=total_ms / max(count, 1),
                max_ms=float(values["max_ns"]) / 1e6,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (-row.total_ms, first_seen[row.name]),
        )
    )


_MEMORY_SUMMARY_FIELDNAMES = (
    "name",
    "count",
    "total_ms",
    "mean_ms",
    "max_ms",
    "rss_delta_mib_sum",
    "rss_delta_mib_max",
    "rss_end_mib_max",
    "tracemalloc_current_delta_bytes_sum",
    "tracemalloc_current_delta_bytes_max",
    "tracemalloc_peak_delta_bytes_max",
    "device_bytes_in_use_delta_sum",
    "device_bytes_in_use_delta_max",
    "device_bytes_in_use_end_max",
    "nvidia_smi_memory_used_delta_mib_sum",
    "nvidia_smi_memory_used_delta_mib_max",
    "nvidia_smi_memory_used_end_mib_max",
    "estimated_tensor_nbytes_max",
    "retained_output_nbytes_max",
    "memory_estimate_gap_note",
)


def _summarize_memory_events(events: Sequence[BenchmarkEvent]) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for event in events:
        memory = _mapping(event.metadata.get("memory"))
        if not memory:
            continue
        row = aggregates.setdefault(
            event.name,
            {
                "name": event.name,
                "count": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "rss_delta_mib_sum": 0.0,
                "rss_delta_mib_max": None,
                "rss_end_mib_max": None,
                "tracemalloc_current_delta_bytes_sum": 0,
                "tracemalloc_current_delta_bytes_max": None,
                "tracemalloc_peak_delta_bytes_max": None,
                "device_bytes_in_use_delta_sum": 0,
                "device_bytes_in_use_delta_max": None,
                "device_bytes_in_use_end_max": None,
                "nvidia_smi_memory_used_delta_mib_sum": 0.0,
                "nvidia_smi_memory_used_delta_mib_max": None,
                "nvidia_smi_memory_used_end_mib_max": None,
                "estimated_tensor_nbytes_max": None,
                "retained_output_nbytes_max": None,
                "memory_estimate_gap_note": "",
            },
        )
        row["count"] += 1
        row["total_ms"] += event.duration_ms
        row["max_ms"] = max(float(row["max_ms"]), event.duration_ms)
        _add_sum_and_max(row, "rss_delta_mib", memory)
        _add_max(row, "rss_end_mib", memory)
        _add_sum_and_max(row, "tracemalloc_current_delta_bytes", memory)
        _add_max(row, "tracemalloc_peak_delta_bytes", memory)
        _add_sum_and_max(row, "device_bytes_in_use_delta", memory)
        _add_max(row, "device_bytes_in_use_end", memory)
        _add_sum_and_max(row, "nvidia_smi_memory_used_delta_mib", memory)
        _add_max(row, "nvidia_smi_memory_used_end_mib", memory)
        _update_max(row, "estimated_tensor_nbytes_max", _estimated_tensor_nbytes(event.metadata))
        _update_max(row, "retained_output_nbytes_max", _retained_output_nbytes(event.metadata))

    rows = []
    for row in aggregates.values():
        count = int(row["count"])
        row["mean_ms"] = row["total_ms"] / max(count, 1)
        row["memory_estimate_gap_note"] = _memory_estimate_gap_note(row)
        rows.append({key: row.get(key) for key in _MEMORY_SUMMARY_FIELDNAMES})
    return sorted(rows, key=lambda row: (-float(row["total_ms"]), str(row["name"])))


def _add_sum_and_max(row: dict[str, Any], metric: str, memory: Mapping[str, Any]) -> None:
    value = _numeric_or_none(memory.get(metric))
    if value is None:
        return
    row[f"{metric}_sum"] += value
    _update_max(row, f"{metric}_max", value)


def _add_max(row: dict[str, Any], metric: str, memory: Mapping[str, Any]) -> None:
    _update_max(row, f"{metric}_max", _numeric_or_none(memory.get(metric)))


def _update_max(row: dict[str, Any], key: str, value: float | int | None) -> None:
    if value is None:
        return
    current = row.get(key)
    row[key] = value if current is None else max(current, value)


def _memory_estimate_gap_note(row: Mapping[str, Any]) -> str:
    estimated = _numeric_or_none(row.get("estimated_tensor_nbytes_max"))
    if estimated is None or estimated <= 0:
        return ""
    notes: list[str] = []
    rss_delta_mib = _numeric_or_none(row.get("rss_delta_mib_max"))
    if rss_delta_mib is not None:
        rss_delta_bytes = max(rss_delta_mib, 0.0) * float(1024**2)
        if rss_delta_bytes > estimated * 4.0 and rss_delta_bytes - estimated > 10 * 1024**2:
            notes.append("rss_delta_exceeds_tensor_estimate")
    device_delta = _numeric_or_none(row.get("device_bytes_in_use_delta_max"))
    if device_delta is not None:
        device_delta_bytes = max(device_delta, 0.0)
        if device_delta_bytes > estimated * 4.0 and device_delta_bytes - estimated > 10 * 1024**2:
            notes.append("device_delta_exceeds_tensor_estimate")
    if rss_delta_mib is None and device_delta is None:
        notes.append("no_measured_rss_or_device_delta")
    return ";".join(notes)


def _collect_benchmark_metadata(output_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_unix": time.time(),
        "output_dir": str(output_dir),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_id": os.getpid(),
    }
    try:
        import jax

        metadata.update(
            {
                "jax_version": jax.__version__,
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    except Exception as exc:  # pragma: no cover - defensive metadata only.
        metadata["jax_metadata_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from axonscope.utils.env import collect_environment_info

        environment = collect_environment_info()
    except Exception as exc:  # pragma: no cover - defensive metadata only.
        metadata["environment_error"] = f"{type(exc).__name__}: {exc}"
    else:
        metadata.update(_environment_benchmark_metadata(environment))
    return _json_safe_dict(metadata)


def _environment_benchmark_metadata(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Return benchmark-oriented machine/backend metadata."""

    os_info = _mapping(environment.get("os"))
    cpu_info = _mapping(environment.get("cpu"))
    memory_info = _mapping(environment.get("memory"))
    gpu_info = _mapping(environment.get("gpu"))
    jax_info = _mapping(environment.get("jax"))
    backend = jax_info.get("default_backend")
    devices = tuple(
        _mapping(device) for device in _sequence(jax_info.get("device_details"))
    )
    device_platforms = tuple(
        str(device.get("platform"))
        for device in devices
        if device.get("platform") is not None
    )
    device_kinds = tuple(
        str(device.get("device_kind"))
        for device in devices
        if device.get("device_kind") is not None
    )
    gpu_devices = tuple(_mapping(device) for device in _sequence(gpu_info.get("devices")))
    gpu_models = tuple(
        str(device.get("name"))
        for device in gpu_devices
        if device.get("name") is not None
    )

    return {
        "environment": environment,
        "os": os_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "packages": _mapping(environment.get("packages")),
        "environment_variables": _mapping(environment.get("environment_variables")),
        "git": _mapping(environment.get("git")),
        "jax_details": jax_info,
        "compute_backend": backend,
        "compute_device_class": _device_class(
            str(backend) if backend is not None else None,
            device_platforms,
        ),
        "compute_device_platforms": list(device_platforms),
        "compute_device_models": list(device_kinds),
        "host_os": os_info.get("platform"),
        "host_ram_total_gb": memory_info.get("total_gb"),
        "host_ram_available_gb": memory_info.get("available_gb"),
        "cpu_model": cpu_info.get("model"),
        "gpu_models": list(gpu_models),
    }


def _device_class(backend: str | None, platforms: Sequence[str]) -> str:
    labels = {backend.lower()} if backend else set()
    labels.update(platform.lower() for platform in platforms)
    if labels & {"gpu", "cuda", "rocm"}:
        return "gpu"
    if "tpu" in labels:
        return "tpu"
    if "cpu" in labels:
        return "cpu"
    return "unknown"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _memory_start_snapshot(config: BenchmarkConfig) -> tuple[dict[str, Any], Any | None]:
    start: dict[str, Any] = {}
    tracemalloc_snapshot = None
    if _trace_uses_rss(config):
        start["rss_mib"] = _current_rss_mib()
    if _trace_uses_tracemalloc(config):
        current, peak = tracemalloc.get_traced_memory()
        start["tracemalloc_current_bytes"] = int(current)
        start["tracemalloc_peak_bytes"] = int(peak)
        if config.memory_top_n > 0:
            tracemalloc_snapshot = tracemalloc.take_snapshot()
    if _trace_uses_device(config):
        start["device"] = _device_memory_snapshot()
    return start, tracemalloc_snapshot


def _memory_end_metadata(
    config: BenchmarkConfig,
    *,
    start: Mapping[str, Any],
    tracemalloc_start_snapshot: Any | None,
) -> dict[str, Any]:
    memory: dict[str, Any] = {}
    if _trace_uses_rss(config):
        start_rss = _numeric_or_none(start.get("rss_mib"))
        end_rss = _current_rss_mib()
        memory["rss_start_mib"] = start_rss
        memory["rss_end_mib"] = end_rss
        memory["rss_delta_mib"] = (
            None if start_rss is None or end_rss is None else end_rss - start_rss
        )
    if _trace_uses_tracemalloc(config):
        current, peak = tracemalloc.get_traced_memory()
        start_current = int(start.get("tracemalloc_current_bytes", 0))
        start_peak = int(start.get("tracemalloc_peak_bytes", 0))
        memory.update(
            {
                "tracemalloc_current_start_bytes": start_current,
                "tracemalloc_current_end_bytes": int(current),
                "tracemalloc_current_delta_bytes": int(current) - start_current,
                "tracemalloc_peak_start_bytes": start_peak,
                "tracemalloc_peak_end_bytes": int(peak),
                "tracemalloc_peak_delta_bytes": max(int(peak) - start_peak, 0),
            }
        )
        if config.memory_top_n > 0 and tracemalloc_start_snapshot is not None:
            end_snapshot = tracemalloc.take_snapshot()
            memory["tracemalloc_top"] = _tracemalloc_top_deltas(
                tracemalloc_start_snapshot,
                end_snapshot,
                limit=config.memory_top_n,
            )
    if _trace_uses_device(config):
        start_device = _mapping(start.get("device"))
        end_device = _device_memory_snapshot()
        memory.update(_device_memory_delta(start_device, end_device))
    return memory


def _trace_uses_measured_memory(config: BenchmarkConfig) -> bool:
    return (
        _trace_uses_rss(config)
        or _trace_uses_tracemalloc(config)
        or _trace_uses_device(config)
        or config.jax_device_memory_profile
    )


def _trace_uses_rss(config: BenchmarkConfig) -> bool:
    return config.memory_trace in {"rss", "all"}


def _trace_uses_tracemalloc(config: BenchmarkConfig) -> bool:
    return config.memory_trace in {"tracemalloc", "all"}


def _trace_uses_device(config: BenchmarkConfig) -> bool:
    return config.memory_trace in {"device", "all"}


def _current_rss_mib() -> float | None:
    try:
        from axonscope.utils.progress_reporting import current_rss_mib

        return current_rss_mib()
    except Exception:
        return None


def _tracemalloc_top_deltas(start_snapshot: Any, end_snapshot: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for stat in end_snapshot.compare_to(start_snapshot, "lineno")[: int(limit)]:
        frame = stat.traceback[0] if stat.traceback else None
        rows.append(
            {
                "size_diff_bytes": int(stat.size_diff),
                "count_diff": int(stat.count_diff),
                "size_bytes": int(stat.size),
                "count": int(stat.count),
                "file": None if frame is None else str(frame.filename),
                "line": None if frame is None else int(frame.lineno),
            }
        )
    return rows


def _device_memory_snapshot() -> dict[str, Any]:
    jax_devices = _jax_device_memory_snapshot()
    nvidia_smi = _nvidia_smi_snapshot()
    snapshot = {
        "jax_devices": jax_devices,
        "nvidia_smi": nvidia_smi,
    }
    snapshot.update(_device_memory_totals(jax_devices, nvidia_smi))
    return snapshot


def _jax_device_memory_snapshot() -> list[dict[str, Any]]:
    try:
        import jax

        devices = jax.devices()
    except Exception as exc:
        return [{"available": False, "error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for device in devices:
        row: dict[str, Any] = {
            "repr": str(device),
            "platform": getattr(device, "platform", None),
            "id": getattr(device, "id", None),
            "device_kind": getattr(device, "device_kind", None),
        }
        stats_fn = getattr(device, "memory_stats", None)
        if callable(stats_fn):
            try:
                stats = stats_fn() or {}
            except Exception as exc:
                row["memory_stats_error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["memory_stats"] = {
                    str(key): _json_safe(value)
                    for key, value in dict(stats).items()
                }
        rows.append(row)
    return rows


def _nvidia_smi_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return {"available": False, "source": "nvidia-smi"}
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, total, used, free = parts
        devices.append(
            {
                "index": _numeric_or_none(index),
                "name": name,
                "memory_total_mib": _numeric_or_none(total),
                "memory_used_mib": _numeric_or_none(used),
                "memory_free_mib": _numeric_or_none(free),
            }
        )
    return {
        "available": bool(devices),
        "source": "nvidia-smi",
        "devices": devices,
        "raw": output,
    }


def _device_memory_totals(
    jax_devices: Sequence[Mapping[str, Any]],
    nvidia_smi: Mapping[str, Any],
) -> dict[str, Any]:
    bytes_in_use = _sum_jax_memory_stat(
        jax_devices,
        keys=("bytes_in_use", "bytes_used", "used_bytes", "current_bytes"),
    )
    peak_bytes = _sum_jax_memory_stat(
        jax_devices,
        keys=("peak_bytes_in_use", "bytes_peak", "peak_bytes", "max_bytes_in_use"),
    )
    nvidia_used = _sum_nvidia_smi_mib(nvidia_smi, key="memory_used_mib")
    nvidia_total = _sum_nvidia_smi_mib(nvidia_smi, key="memory_total_mib")
    return {
        "device_bytes_in_use": bytes_in_use,
        "device_peak_bytes_in_use": peak_bytes,
        "nvidia_smi_memory_used_mib": nvidia_used,
        "nvidia_smi_memory_total_mib": nvidia_total,
    }


def _device_memory_delta(start: Mapping[str, Any], end: Mapping[str, Any]) -> dict[str, Any]:
    start_bytes = _numeric_or_none(start.get("device_bytes_in_use"))
    end_bytes = _numeric_or_none(end.get("device_bytes_in_use"))
    start_peak = _numeric_or_none(start.get("device_peak_bytes_in_use"))
    end_peak = _numeric_or_none(end.get("device_peak_bytes_in_use"))
    start_smi = _numeric_or_none(start.get("nvidia_smi_memory_used_mib"))
    end_smi = _numeric_or_none(end.get("nvidia_smi_memory_used_mib"))
    return {
        "device_start": start,
        "device_end": end,
        "device_bytes_in_use_start": start_bytes,
        "device_bytes_in_use_end": end_bytes,
        "device_bytes_in_use_delta": _delta(end_bytes, start_bytes),
        "device_peak_bytes_in_use_start": start_peak,
        "device_peak_bytes_in_use_end": end_peak,
        "device_peak_bytes_in_use_delta": _delta(end_peak, start_peak),
        "nvidia_smi_memory_used_start_mib": start_smi,
        "nvidia_smi_memory_used_end_mib": end_smi,
        "nvidia_smi_memory_used_delta_mib": _delta(end_smi, start_smi),
    }


def _sum_jax_memory_stat(
    devices: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> int | None:
    values: list[int] = []
    for device in devices:
        stats = _mapping(device.get("memory_stats"))
        for key in keys:
            value = _numeric_or_none(stats.get(key))
            if value is not None:
                values.append(int(value))
                break
    return sum(values) if values else None


def _sum_nvidia_smi_mib(snapshot: Mapping[str, Any], *, key: str) -> float | None:
    values = [
        value
        for value in (
            _numeric_or_none(_mapping(device).get(key))
            for device in _sequence(snapshot.get("devices"))
        )
        if value is not None
    ]
    return float(sum(values)) if values else None


def _should_save_device_memory_profile(config: BenchmarkConfig, stage_name: str) -> bool:
    if not config.jax_device_memory_profile:
        return False
    stages = set(config.jax_device_memory_profile_stages)
    return not stages or "all" in stages or stage_name in stages


def _safe_stage_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return safe.strip("_") or "stage"


def _estimated_tensor_nbytes(metadata: Mapping[str, Any]) -> int | None:
    values = list(_array_nbytes(metadata, include_outputs=True))
    estimate = _numeric_or_none(metadata.get("memory_estimate_total_nbytes"))
    if estimate is not None:
        values.append(int(estimate))
    return sum(values) if values else None


def _retained_output_nbytes(metadata: Mapping[str, Any]) -> int | None:
    values = list(_array_nbytes(metadata, include_outputs=False))
    components = _mapping(metadata.get("memory_estimate_components_nbytes"))
    vm_output = _numeric_or_none(components.get("vm_output"))
    if vm_output is not None:
        values.append(int(vm_output))
    return sum(values) if values else None


def _array_nbytes(value: Any, *, include_outputs: bool) -> Iterator[int]:
    if isinstance(value, Mapping):
        if value.get("role") == "kernel_output" or include_outputs:
            nbytes = _numeric_or_none(value.get("nbytes"))
            if nbytes is not None and "shape" in value:
                yield int(nbytes)
        for key, nested in value.items():
            if key == "memory":
                continue
            yield from _array_nbytes(nested, include_outputs=include_outputs)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _array_nbytes(nested, include_outputs=include_outputs)


def _numeric_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(end: float | int | None, start: float | int | None) -> float | int | None:
    if end is None or start is None:
        return None
    return end - start


def _array_metadata(array: Any, *, role: str | None, session: BenchmarkSession) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if role is not None:
        metadata["role"] = role

    shape = getattr(array, "shape", None)
    if shape is not None:
        metadata["shape"] = [int(dim) for dim in tuple(shape)]
        metadata["size"] = int(np.prod(tuple(shape), dtype=np.int64)) if shape else 1

    dtype = getattr(array, "dtype", None)
    if dtype is not None:
        dtype_obj = np.dtype(dtype)
        metadata["dtype"] = str(dtype_obj)
        metadata["itemsize"] = int(dtype_obj.itemsize)
        if session.config.record_memory and "size" in metadata:
            metadata["nbytes"] = int(metadata["size"]) * int(dtype_obj.itemsize)
    elif session.config.record_memory:
        nbytes = getattr(array, "nbytes", None)
        if nbytes is not None:
            metadata["nbytes"] = int(nbytes)

    device = _array_device(array)
    if device is not None:
        metadata["device"] = device
    sharding = getattr(array, "sharding", None)
    if sharding is not None:
        metadata["sharding"] = repr(sharding)
    return _json_safe_dict(metadata)


def _array_device(array: Any) -> str | None:
    devices = getattr(array, "devices", None)
    if callable(devices):
        try:
            values = devices()
            return ", ".join(str(value) for value in values)
        except Exception:
            pass

    device = getattr(array, "device", None)
    if callable(device):
        try:
            device = device()
        except TypeError:
            pass
        except Exception:
            return None
    if device is None:
        return None
    return str(device)


def _block_until_ready(value: Any, *, seen: set[int]) -> None:
    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)

    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()
        return

    if isinstance(value, Mapping):
        for item in value.values():
            _block_until_ready(item, seen=seen)
        return

    if isinstance(value, (str, bytes)):
        return

    if is_dataclass(value) and not isinstance(value, type):
        for field_info in fields(value):
            _block_until_ready(getattr(value, field_info.name), seen=seen)
        return

    if isinstance(value, Sequence):
        for item in value:
            _block_until_ready(item, seen=seen)


def _json_safe_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in values.items()}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, Mapping):
        return _json_safe_dict(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkEvent",
    "BenchmarkReport",
    "BenchmarkSession",
    "BenchmarkSummaryRow",
    "active_benchmark_session",
    "benchmark",
    "benchmark_array_metadata",
    "benchmark_report",
    "benchmark_span",
    "benchmark_wait",
    "disable_benchmark",
    "enable_benchmark",
    "record_benchmark_metadata",
    "reset_benchmark",
]
