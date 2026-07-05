"""Private benchmark instrumentation runtime."""

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ACTIVE_SESSION: ContextVar["BenchmarkSession | None"] = ContextVar(
    "axonscope_benchmark_session",
    default=None,
)


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """User-facing options for one benchmark instrumentation session."""

    print_summary: bool = True
    save: bool = True
    sync_device: bool = True
    record_shapes: bool = True
    record_memory: bool = True
    memory_trace: str = "off"
    memory_top_n: int = 0
    profile: bool = False
    profile_backend: str = "auto"
    profile_output: Path | None = None
    profile_create_perfetto: bool = False
    profile_create_perfetto_link: bool = False
    jax_device_memory_profile: bool = False
    jax_device_memory_profile_stages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Resolved instrumentation configuration."""

    output_dir: Path
    print_summary: bool = True
    save: bool = True
    sync_device: bool = True
    record_shapes: bool = True
    record_memory: bool = True
    memory_trace: str = "off"
    memory_top_n: int = 0
    profile: bool = False
    profile_backend: str = "auto"
    profile_output: Path | None = None
    profile_create_perfetto: bool = False
    profile_create_perfetto_link: bool = False
    jax_device_memory_profile: bool = False
    jax_device_memory_profile_stages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkEvent:
    """One completed benchmark span."""

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
        return self.duration_ns / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
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


@dataclass(frozen=True, slots=True)
class BenchmarkSummaryRow:
    """Aggregate timing for one event name."""

    name: str
    count: int
    total_ms: float
    self_ms: float
    mean_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": self.total_ms,
            "self_ms": self.self_ms,
            "mean_ms": self.mean_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Completed or current benchmark session report."""

    events: tuple[BenchmarkEvent, ...]
    summary: tuple[BenchmarkSummaryRow, ...]
    metadata: Mapping[str, Any]

    def format(self) -> str:
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

        memory_rows = _memory_summary(self.events)
        with (path / "memory_summary.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=_MEMORY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(memory_rows)

        metadata = _json_safe(dict(self.metadata))
        for filename in ("metadata.json", "environment.json"):
            with (path / filename).open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
                handle.write("\n")


@dataclass(slots=True)
class _ActiveSpan:
    event_id: int
    name: str
    parent_event_id: int | None
    depth: int
    start_ns: int
    metadata: dict[str, Any]
    memory_start: dict[str, Any]
    tracemalloc_start: Any | None = None


@dataclass
class BenchmarkSession:
    """Mutable state for one active benchmark run."""

    config: BenchmarkConfig
    events: list[BenchmarkEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    _stack: list[_ActiveSpan] = field(default_factory=list)
    _next_event_id: int = 0
    _token: Token[BenchmarkSession | None] | None = None
    _profile_handle: Any | None = None
    _tracemalloc_started: bool = False

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.active:
            yield
            return

        event_id = self._next_event_id
        self._next_event_id += 1
        parent_id = self._stack[-1].event_id if self._stack else None
        memory_start, tracemalloc_start = _memory_start(self.config)
        span = _ActiveSpan(
            event_id=event_id,
            name=str(name),
            parent_event_id=parent_id,
            depth=len(self._stack),
            start_ns=time.perf_counter_ns(),
            metadata=_json_safe_dict(metadata),
            memory_start=memory_start,
            tracemalloc_start=tracemalloc_start,
        )
        self._stack.append(span)
        failed = False
        try:
            yield
        except BaseException as exc:
            failed = True
            span.metadata.update(
                {
                    "failed": True,
                    "exception_type": type(exc).__name__,
                }
            )
            raise
        finally:
            end_ns = time.perf_counter_ns()
            popped = self._stack.pop()
            if popped is not span:
                raise RuntimeError("benchmark span stack became inconsistent.")
            if failed:
                span.metadata.setdefault("failed", True)
            memory = _memory_end(
                self.config,
                start=span.memory_start,
                tracemalloc_start=span.tracemalloc_start,
            )
            profile = self._device_memory_profile(span.name, span.event_id)
            if profile:
                memory["jax_device_memory_profile"] = profile
            if memory:
                span.metadata["memory"] = _json_safe(memory)
            self.events.append(
                BenchmarkEvent(
                    event_id=span.event_id,
                    name=span.name,
                    parent_event_id=span.parent_event_id,
                    depth=span.depth,
                    start_ns=span.start_ns,
                    end_ns=end_ns,
                    duration_ns=end_ns - span.start_ns,
                    metadata=dict(span.metadata),
                )
            )

    def record_metadata(self, **metadata: Any) -> None:
        if self._stack:
            self._stack[-1].metadata.update(_json_safe_dict(metadata))

    def report(self) -> BenchmarkReport:
        events = tuple(self.events)
        return BenchmarkReport(
            events=events,
            summary=_summarize(events),
            metadata=dict(self.metadata),
        )

    def reset(self) -> None:
        self.events.clear()
        self._stack.clear()
        self._next_event_id = 0

    def start_profile(self) -> None:
        if not self.config.profile:
            return
        profile_output = self.config.profile_output or self.config.output_dir / "profiles" / "run"
        self.metadata["profile"] = {
            "enabled": True,
            "backend": self.config.profile_backend,
            "output": str(profile_output),
            "create_perfetto_trace": self.config.profile_create_perfetto,
            "create_perfetto_link": self.config.profile_create_perfetto_link,
            "active": False,
        }
        if self.config.profile_backend == "none":
            return
        try:
            from axonscope.backends.execution import benchmark_profile_start

            self._profile_handle = benchmark_profile_start(
                self.config.profile_backend,
                profile_output,
                create_perfetto_link=self.config.profile_create_perfetto_link,
                create_perfetto_trace=self.config.profile_create_perfetto,
            )
        except Exception as exc:  # pragma: no cover - backend-dependent.
            self.metadata["profile"]["error"] = f"{type(exc).__name__}: {exc}"
            return
        self.metadata["profile"]["active"] = self._profile_handle is not None

    def stop_profile(self) -> None:
        if self._profile_handle is None:
            profile = self.metadata.get("profile")
            if isinstance(profile, dict):
                profile["active"] = False
            return
        try:
            from axonscope.backends.execution import benchmark_profile_stop

            stopped = benchmark_profile_stop(self._profile_handle)
        except Exception as exc:  # pragma: no cover - backend-dependent.
            stopped = {"stop_error": f"{type(exc).__name__}: {exc}"}
        profile = self.metadata.setdefault("profile", {})
        if isinstance(profile, dict):
            profile.update(_json_safe_dict(stopped))
            profile["active"] = False
        self._profile_handle = None

    def finish(
        self,
        *,
        print_summary: bool | None = None,
        save: bool | None = None,
    ) -> BenchmarkReport:
        self.active = False
        self.stop_profile()
        report = self.report()
        should_save = self.config.save if save is None else bool(save)
        should_print = self.config.print_summary if print_summary is None else bool(print_summary)
        if should_save:
            report.save(self.config.output_dir)
        if should_print:
            print(report.format())
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
        return report

    def _device_memory_profile(self, name: str, event_id: int) -> dict[str, Any]:
        if not _should_profile_device_memory(self.config, name):
            return {}
        path = (
            self.config.output_dir
            / "device_memory_profiles"
            / f"{event_id:04d}_{_safe_filename(name)}.prof"
        )
        try:
            from axonscope.backends.execution import benchmark_save_device_memory_profile

            metadata = benchmark_save_device_memory_profile(path, backend="jax")
        except Exception as exc:  # pragma: no cover - backend-dependent.
            metadata = {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}
        metadata.setdefault("stage", name)
        return _json_safe_dict(metadata)


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
    memory_trace: str = "off",
    memory_top_n: int = 0,
    profile: bool | None = None,
    profile_backend: str | None = None,
    profile_output: str | Path | None = None,
    profile_create_perfetto: bool | None = None,
    profile_create_perfetto_link: bool | None = None,
    jax_device_memory_profile: bool = False,
    jax_device_memory_profile_stages: Sequence[str] | str | None = None,
) -> BenchmarkSession:
    """Enable benchmark instrumentation for subsequent AxonScope calls."""

    if options is not None:
        print_summary = options.print_summary
        save = options.save
        sync_device = options.sync_device
        record_shapes = options.record_shapes
        record_memory = options.record_memory
        memory_trace = options.memory_trace
        memory_top_n = options.memory_top_n
        profile = options.profile if profile is None else profile
        profile_backend = profile_backend or options.profile_backend
        profile_output = profile_output or options.profile_output
        profile_create_perfetto = (
            options.profile_create_perfetto
            if profile_create_perfetto is None
            else profile_create_perfetto
        )
        profile_create_perfetto_link = (
            options.profile_create_perfetto_link
            if profile_create_perfetto_link is None
            else profile_create_perfetto_link
        )
        jax_device_memory_profile = options.jax_device_memory_profile
        if jax_device_memory_profile_stages is None:
            jax_device_memory_profile_stages = options.jax_device_memory_profile_stages

    if active_benchmark_session() is not None:
        raise RuntimeError("An AxonScope benchmark session is already active.")

    memory_trace = str(memory_trace).lower()
    if memory_trace not in {"off", "rss", "tracemalloc", "device", "all"}:
        raise ValueError("memory_trace must be one of: off, rss, tracemalloc, device, all.")
    if int(memory_top_n) < 0:
        raise ValueError("memory_top_n must be >= 0.")

    resolved_profile = bool(profile) if profile is not None else False
    resolved_profile_backend = str(profile_backend or "auto").lower()
    if resolved_profile_backend not in {"auto", "jax", "none"}:
        raise ValueError("profile_backend must be one of: auto, jax, none.")
    profile_stages = _normalize_profile_stages(jax_device_memory_profile_stages)
    output = Path(output_dir)
    if save:
        output.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        output_dir=output,
        print_summary=bool(print_summary),
        save=bool(save),
        sync_device=bool(sync_device),
        record_shapes=bool(record_shapes),
        record_memory=bool(record_memory),
        memory_trace=memory_trace,
        memory_top_n=int(memory_top_n),
        profile=resolved_profile,
        profile_backend=resolved_profile_backend,
        profile_output=Path(profile_output) if profile_output is not None else None,
        profile_create_perfetto=bool(profile_create_perfetto)
        if profile_create_perfetto is not None
        else False,
        profile_create_perfetto_link=bool(profile_create_perfetto_link)
        if profile_create_perfetto_link is not None
        else False,
        jax_device_memory_profile=bool(jax_device_memory_profile),
        jax_device_memory_profile_stages=profile_stages,
    )
    session = BenchmarkSession(
        config=config,
        metadata=_collect_metadata(output, config),
    )
    if _uses_tracemalloc(config) and not tracemalloc.is_tracing():
        tracemalloc.start(max(config.memory_top_n, 1))
        session._tracemalloc_started = True
    if reset:
        session.reset()
    session._token = _ACTIVE_SESSION.set(session)
    session.start_profile()
    return session


def disable_benchmark(
    *,
    print_summary: bool | None = None,
    save: bool | None = None,
) -> BenchmarkReport | None:
    """Disable active benchmark instrumentation and return its report."""

    session = _ACTIVE_SESSION.get()
    if session is None:
        return None
    try:
        return session.finish(print_summary=print_summary, save=save)
    finally:
        if session._token is not None:
            _ACTIVE_SESSION.reset(session._token)
            session._token = None
        else:
            _ACTIVE_SESSION.set(None)


@contextmanager
def benchmark(output_dir: str | Path, **options: Any) -> Iterator[BenchmarkSession]:
    """Context-manager form of `enable_benchmark`/`disable_benchmark`."""

    session = enable_benchmark(output_dir, **options)
    try:
        yield session
    finally:
        disable_benchmark()


def active_benchmark_session() -> BenchmarkSession | None:
    session = _ACTIVE_SESSION.get()
    if session is None or not session.active:
        return None
    return session


def benchmark_span(name: str, **metadata: Any):
    session = active_benchmark_session()
    if session is None:
        return nullcontext()
    return session.span(name, **metadata)


def record_benchmark_metadata(**metadata: Any) -> None:
    session = active_benchmark_session()
    if session is not None:
        session.record_metadata(**metadata)


def benchmark_array_metadata(name: str, array: Any, *, role: str | None = None) -> dict[str, Any]:
    session = active_benchmark_session()
    if session is None or not session.config.record_shapes:
        return {}
    return {str(name): _array_metadata(array, role=role, config=session.config)}


def benchmark_wait(value: Any) -> Any:
    session = active_benchmark_session()
    if session is not None and session.config.sync_device:
        _block_until_ready(value, seen=set())
    return value


def benchmark_report(
    *,
    print_report: bool = True,
    save: bool = False,
) -> BenchmarkReport | None:
    session = active_benchmark_session()
    if session is None:
        return None
    report = session.report()
    if save:
        report.save(session.config.output_dir)
    if print_report:
        print(report.format())
    return report


def reset_benchmark() -> None:
    session = active_benchmark_session()
    if session is not None:
        session.reset()


def _summarize(events: Sequence[BenchmarkEvent]) -> tuple[BenchmarkSummaryRow, ...]:
    child_ns: dict[int, int] = defaultdict(int)
    for event in events:
        if event.parent_event_id is not None:
            child_ns[event.parent_event_id] += event.duration_ns

    first_seen: dict[str, int] = {}
    rows: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        first_seen.setdefault(event.name, index)
        row = rows.setdefault(
            event.name,
            {"count": 0, "total_ns": 0, "self_ns": 0, "max_ns": 0},
        )
        self_ns = max(event.duration_ns - child_ns[event.event_id], 0)
        row["count"] += 1
        row["total_ns"] += event.duration_ns
        row["self_ns"] += self_ns
        row["max_ns"] = max(row["max_ns"], event.duration_ns)

    summary = []
    for name, values in rows.items():
        count = int(values["count"])
        total_ms = float(values["total_ns"]) / 1_000_000.0
        summary.append(
            BenchmarkSummaryRow(
                name=name,
                count=count,
                total_ms=total_ms,
                self_ms=float(values["self_ns"]) / 1_000_000.0,
                mean_ms=total_ms / max(count, 1),
                max_ms=float(values["max_ns"]) / 1_000_000.0,
            )
        )
    return tuple(sorted(summary, key=lambda row: (-row.total_ms, first_seen[row.name])))


_MEMORY_FIELDNAMES = (
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
)


def _memory_summary(events: Sequence[BenchmarkEvent]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for event in events:
        memory = event.metadata.get("memory")
        if not isinstance(memory, Mapping):
            continue
        row = rows.setdefault(
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
            },
        )
        row["count"] += 1
        row["total_ms"] += event.duration_ms
        row["max_ms"] = max(float(row["max_ms"]), event.duration_ms)
        _sum_and_max(row, memory, "rss_delta_mib")
        _max(row, memory, "rss_end_mib")
        _sum_and_max(row, memory, "tracemalloc_current_delta_bytes")
        _max(row, memory, "tracemalloc_peak_delta_bytes")
        _sum_and_max(row, memory, "device_bytes_in_use_delta")
        _max(row, memory, "device_bytes_in_use_end")
        _sum_and_max(row, memory, "nvidia_smi_memory_used_delta_mib")
        _max(row, memory, "nvidia_smi_memory_used_end_mib")

    result = []
    for row in rows.values():
        count = int(row["count"])
        row["mean_ms"] = row["total_ms"] / max(count, 1)
        result.append({key: row.get(key) for key in _MEMORY_FIELDNAMES})
    return sorted(result, key=lambda row: (-float(row["total_ms"]), str(row["name"])))


def _sum_and_max(row: dict[str, Any], memory: Mapping[str, Any], key: str) -> None:
    value = _number_or_none(memory.get(key))
    if value is None:
        return
    row[f"{key}_sum"] += value
    _set_max(row, f"{key}_max", value)


def _max(row: dict[str, Any], memory: Mapping[str, Any], key: str) -> None:
    _set_max(row, f"{key}_max", _number_or_none(memory.get(key)))


def _set_max(row: dict[str, Any], key: str, value: float | int | None) -> None:
    if value is None:
        return
    current = row.get(key)
    row[key] = value if current is None else max(current, value)


def _collect_metadata(output_dir: Path, config: BenchmarkConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_unix": time.time(),
        "output_dir": str(output_dir),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "memory_trace": config.memory_trace,
        "memory_top_n": config.memory_top_n,
        "jax_device_memory_profile": config.jax_device_memory_profile,
        "jax_device_memory_profile_stages": list(config.jax_device_memory_profile_stages),
        "profile": {
            "enabled": config.profile,
            "backend": config.profile_backend,
            "output": str(config.profile_output) if config.profile_output else None,
            "active": False,
        },
        "git": _git_metadata(),
    }
    try:
        from axonscope.utils.env import collect_environment_info

        environment = collect_environment_info()
    except Exception as exc:  # pragma: no cover - metadata best effort.
        metadata["environment_error"] = f"{type(exc).__name__}: {exc}"
    else:
        metadata.update(_environment_metadata(environment))
    return _json_safe_dict(metadata)


def _environment_metadata(environment: Mapping[str, Any]) -> dict[str, Any]:
    os_info = _mapping(environment.get("os"))
    cpu_info = _mapping(environment.get("cpu"))
    memory_info = _mapping(environment.get("memory"))
    gpu_info = _mapping(environment.get("gpu"))
    jax_info = _mapping(environment.get("jax"))
    device_details = [_mapping(device) for device in _sequence(jax_info.get("device_details"))]
    platforms = [
        str(device["platform"])
        for device in device_details
        if device.get("platform") is not None
    ]
    device_models = [
        str(device["device_kind"])
        for device in device_details
        if device.get("device_kind") is not None
    ]
    gpu_models = [
        str(device["name"])
        for device in _sequence(gpu_info.get("devices"))
        if isinstance(device, Mapping) and device.get("name") is not None
    ]
    backend = jax_info.get("default_backend")
    return {
        "environment": environment,
        "os": os_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info,
        "packages": _mapping(environment.get("packages")),
        "environment_variables": _mapping(environment.get("environment_variables")),
        "jax_details": jax_info,
        "compute_backend": backend,
        "compute_device_class": _device_class(str(backend) if backend else None, platforms),
        "compute_device_platforms": platforms,
        "compute_device_models": device_models,
        "host_os": os_info.get("platform"),
        "host_ram_total_gb": memory_info.get("total_gb"),
        "host_ram_available_gb": memory_info.get("available_gb"),
        "cpu_model": cpu_info.get("model"),
        "gpu_models": gpu_models,
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


def _memory_start(config: BenchmarkConfig) -> tuple[dict[str, Any], Any | None]:
    start: dict[str, Any] = {}
    tracemalloc_snapshot = None
    if _uses_rss(config):
        start["rss_mib"] = _current_rss_mib()
    if _uses_tracemalloc(config):
        current, peak = tracemalloc.get_traced_memory()
        start["tracemalloc_current_bytes"] = int(current)
        start["tracemalloc_peak_bytes"] = int(peak)
        if config.memory_top_n > 0:
            tracemalloc_snapshot = tracemalloc.take_snapshot()
    if _uses_device(config):
        start["device"] = _device_memory_snapshot()
    return start, tracemalloc_snapshot


def _memory_end(
    config: BenchmarkConfig,
    *,
    start: Mapping[str, Any],
    tracemalloc_start: Any | None,
) -> dict[str, Any]:
    memory: dict[str, Any] = {}
    if _uses_rss(config):
        rss_start = _number_or_none(start.get("rss_mib"))
        rss_end = _current_rss_mib()
        memory["rss_start_mib"] = rss_start
        memory["rss_end_mib"] = rss_end
        memory["rss_delta_mib"] = None if rss_start is None or rss_end is None else rss_end - rss_start
    if _uses_tracemalloc(config):
        current, peak = tracemalloc.get_traced_memory()
        current_start = int(start.get("tracemalloc_current_bytes", 0))
        peak_start = int(start.get("tracemalloc_peak_bytes", 0))
        memory.update(
            {
                "tracemalloc_current_start_bytes": current_start,
                "tracemalloc_current_end_bytes": int(current),
                "tracemalloc_current_delta_bytes": int(current) - current_start,
                "tracemalloc_peak_start_bytes": peak_start,
                "tracemalloc_peak_end_bytes": int(peak),
                "tracemalloc_peak_delta_bytes": max(int(peak) - peak_start, 0),
            }
        )
        if config.memory_top_n > 0 and tracemalloc_start is not None:
            memory["tracemalloc_top"] = _tracemalloc_top(
                tracemalloc_start,
                tracemalloc.take_snapshot(),
                limit=config.memory_top_n,
            )
    if _uses_device(config):
        memory.update(_device_delta(_mapping(start.get("device")), _device_memory_snapshot()))
    return memory


def _uses_rss(config: BenchmarkConfig) -> bool:
    return config.memory_trace in {"rss", "all"}


def _uses_tracemalloc(config: BenchmarkConfig) -> bool:
    return config.memory_trace in {"tracemalloc", "all"}


def _uses_device(config: BenchmarkConfig) -> bool:
    return config.memory_trace in {"device", "all"}


def _current_rss_mib() -> float | None:
    try:
        from axonscope.utils.progress_reporting import current_rss_mib

        return current_rss_mib()
    except Exception:
        return None


def _tracemalloc_top(start: Any, end: Any, *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for stat in end.compare_to(start, "lineno")[: int(limit)]:
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
    jax_devices = _jax_device_snapshot()
    nvidia_smi = _nvidia_smi_snapshot()
    snapshot = {"jax_devices": jax_devices, "nvidia_smi": nvidia_smi}
    snapshot.update(_device_totals(jax_devices, nvidia_smi))
    return snapshot


def _jax_device_snapshot() -> list[dict[str, Any]]:
    try:
        import jax

        devices = jax.devices()
    except Exception as exc:
        return [{"available": False, "error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for device in devices:
        row = {
            "repr": str(device),
            "platform": getattr(device, "platform", None),
            "id": getattr(device, "id", None),
            "device_kind": getattr(device, "device_kind", None),
        }
        stats = getattr(device, "memory_stats", None)
        if callable(stats):
            try:
                row["memory_stats"] = _json_safe_dict(dict(stats() or {}))
            except Exception as exc:
                row["memory_stats_error"] = f"{type(exc).__name__}: {exc}"
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
                "index": _number_or_none(index),
                "name": name,
                "memory_total_mib": _number_or_none(total),
                "memory_used_mib": _number_or_none(used),
                "memory_free_mib": _number_or_none(free),
            }
        )
    return {"available": bool(devices), "source": "nvidia-smi", "devices": devices}


def _device_totals(
    jax_devices: Sequence[Mapping[str, Any]],
    nvidia_smi: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "device_bytes_in_use": _sum_jax_stat(jax_devices, ("bytes_in_use", "bytes_used", "used_bytes")),
        "device_peak_bytes_in_use": _sum_jax_stat(jax_devices, ("peak_bytes_in_use", "peak_bytes")),
        "nvidia_smi_memory_used_mib": _sum_smi(nvidia_smi, "memory_used_mib"),
        "nvidia_smi_memory_total_mib": _sum_smi(nvidia_smi, "memory_total_mib"),
    }


def _sum_jax_stat(devices: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int | None:
    total = 0
    found = False
    for device in devices:
        stats = _mapping(device.get("memory_stats"))
        for key in keys:
            value = _number_or_none(stats.get(key))
            if value is not None:
                total += int(value)
                found = True
                break
    return total if found else None


def _sum_smi(snapshot: Mapping[str, Any], key: str) -> float | None:
    total = 0.0
    found = False
    for device in _sequence(snapshot.get("devices")):
        if not isinstance(device, Mapping):
            continue
        value = _number_or_none(device.get(key))
        if value is not None:
            total += float(value)
            found = True
    return total if found else None


def _device_delta(start: Mapping[str, Any], end: Mapping[str, Any]) -> dict[str, Any]:
    start_bytes = _number_or_none(start.get("device_bytes_in_use"))
    end_bytes = _number_or_none(end.get("device_bytes_in_use"))
    start_smi = _number_or_none(start.get("nvidia_smi_memory_used_mib"))
    end_smi = _number_or_none(end.get("nvidia_smi_memory_used_mib"))
    return {
        "device_start": start,
        "device_end": end,
        "device_bytes_in_use_start": start_bytes,
        "device_bytes_in_use_end": end_bytes,
        "device_bytes_in_use_delta": _delta(end_bytes, start_bytes),
        "device_peak_bytes_in_use_start": _number_or_none(start.get("device_peak_bytes_in_use")),
        "device_peak_bytes_in_use_end": _number_or_none(end.get("device_peak_bytes_in_use")),
        "nvidia_smi_memory_used_start_mib": start_smi,
        "nvidia_smi_memory_used_end_mib": end_smi,
        "nvidia_smi_memory_used_delta_mib": _delta(end_smi, start_smi),
        "nvidia_smi_memory_total_mib": _number_or_none(end.get("nvidia_smi_memory_total_mib")),
    }


def _should_profile_device_memory(config: BenchmarkConfig, name: str) -> bool:
    if not config.jax_device_memory_profile:
        return False
    stages = config.jax_device_memory_profile_stages
    return not stages or name in stages


def _array_metadata(array: Any, *, role: str | None, config: BenchmarkConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if role is not None:
        metadata["role"] = role
    shape = getattr(array, "shape", None)
    if shape is not None:
        shape_tuple = tuple(shape)
        metadata["shape"] = [int(dim) for dim in shape_tuple]
        metadata["size"] = int(np.prod(shape_tuple, dtype=np.int64)) if shape_tuple else 1
    dtype = getattr(array, "dtype", None)
    if dtype is not None:
        dtype_obj = np.dtype(dtype)
        metadata["dtype"] = str(dtype_obj)
        metadata["itemsize"] = int(dtype_obj.itemsize)
        if config.record_memory and "size" in metadata:
            metadata["nbytes"] = int(metadata["size"]) * int(dtype_obj.itemsize)
    elif config.record_memory and getattr(array, "nbytes", None) is not None:
        metadata["nbytes"] = int(array.nbytes)
    device = _array_device(array)
    if device is not None:
        metadata["device"] = device
    return _json_safe_dict(metadata)


def _array_device(array: Any) -> str | None:
    devices = getattr(array, "devices", None)
    if callable(devices):
        try:
            values = devices()
        except Exception:
            values = None
        if values:
            return ",".join(sorted(str(device) for device in values))
    device = getattr(array, "device", None)
    if callable(device):
        try:
            device = device()
        except TypeError:
            pass
        except Exception:
            return None
    return None if device is None else str(device)


def _block_until_ready(value: Any, *, seen: set[int]) -> None:
    ident = id(value)
    if ident in seen:
        return
    seen.add(ident)
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
    if isinstance(value, Sequence):
        for item in value:
            _block_until_ready(item, seen=seen)
        return
    for attr in ("Vm", "t", "recordings", "observations", "diagnostics"):
        if hasattr(value, attr):
            _block_until_ready(getattr(value, attr), seen=seen)


def _normalize_profile_stages(stages: Sequence[str] | str | None) -> tuple[str, ...]:
    if stages is None:
        return ()
    if isinstance(stages, str):
        return (stages,)
    return tuple(str(stage) for stage in stages)


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "short_commit": _run_git("rev-parse", "--short", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
        "dirty": _run_git_dirty(),
    }


def _run_git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _run_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _number_or_none(value: Any) -> float | None:
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


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _json_safe_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in values.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkEvent",
    "BenchmarkOptions",
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
