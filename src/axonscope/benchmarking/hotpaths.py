"""Opt-in hotpath instrumentation for developer performance diagnostics."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
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
class BenchmarkConfig:
    """Configuration for one hotpath benchmark session."""

    output_dir: Path
    print_summary: bool = True
    save: bool = True
    sync_device: bool = True
    record_shapes: bool = True
    record_memory: bool = True
    level: str = "hotpaths"


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
        """Write raw events, aggregate summary, and metadata to `output_dir`."""

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

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        """Time one named stage and preserve original exceptions."""

        if not self.active:
            yield
            return

        event_id = self._next_event_id
        self._next_event_id += 1
        parent_event_id = self._stack[-1].event_id if self._stack else None
        active = _ActiveSpan(
            event_id=event_id,
            name=name,
            parent_event_id=parent_event_id,
            depth=len(self._stack),
            start_ns=time.perf_counter_ns(),
            metadata=_json_safe_dict(metadata),
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
            import jax

            with jax.profiler.trace(
                str(trace_dir),
                create_perfetto_trace=bool(trace.get("create_perfetto_trace", False)),
            ):
                with jax.profiler.StepTraceAnnotation("kernel.enqueue"):
                    yield
        finally:
            self._jax_trace_active = False

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
        report = self.report()
        should_save = self.config.save if save is None else bool(save)
        should_print = self.config.print_summary if print_summary is None else bool(print_summary)
        if should_save:
            report.save(self.config.output_dir)
        if should_print:
            print(report.format())
        return report


def enable_benchmark(
    output_dir: str | Path,
    *,
    print_summary: bool = True,
    save: bool = True,
    reset: bool = True,
    sync_device: bool = True,
    record_shapes: bool = True,
    record_memory: bool = True,
    level: str = "hotpaths",
    jax_trace: bool = False,
    jax_trace_dir: str | Path | None = None,
    jax_trace_create_perfetto: bool = False,
    jax_trace_scope: str = "kernel",
) -> BenchmarkSession:
    """Enable hotpath instrumentation for subsequent AxonScope calls."""

    active = _ACTIVE_BENCHMARK_SESSION.get()
    if active is not None and active.active:
        raise RuntimeError("An AxonScope benchmark session is already active.")
    if level != "hotpaths":
        raise ValueError("Only level='hotpaths' is supported for now.")
    if jax_trace_scope not in {"kernel"}:
        raise ValueError("Only jax_trace_scope='kernel' is supported by enable_benchmark.")

    path = Path(output_dir)
    if save:
        path.mkdir(parents=True, exist_ok=True)

    session = BenchmarkSession(
        config=BenchmarkConfig(
            output_dir=path,
            print_summary=print_summary,
            save=save,
            sync_device=sync_device,
            record_shapes=record_shapes,
            record_memory=record_memory,
            level=level,
        ),
        metadata=_collect_benchmark_metadata(path),
    )
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
    return _json_safe_dict(metadata)


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
