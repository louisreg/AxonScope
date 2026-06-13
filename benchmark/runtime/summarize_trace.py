from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PATTERN = "benchmark/"
TRACE_SUFFIXES = (
    ".trace.json.gz",
    ".trace.json",
    "perfetto_trace.json.gz",
    "perfetto_trace.json",
    ".perfetto_trace.json.gz",
    ".perfetto_trace.json",
)


@dataclass(frozen=True)
class TraceEvent:
    source_path: str
    name: str
    case_name: str
    solver_name: str
    phase: str
    start_ms: float | None
    duration_ms: float
    pid: str | int | None = None
    tid: str | int | None = None


@dataclass(frozen=True)
class TraceSummaryRow:
    source_path: str
    case_name: str
    solver_name: str
    phase: str
    count: int
    total_ms: float
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "case_name": self.case_name,
            "solver_name": self.solver_name,
            "phase": self.phase,
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark annotations from JAX profiler traces.")
    parser.add_argument("paths", nargs="+", type=Path, help="Trace files or directories to scan.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Trace event name substring to extract.")
    parser.add_argument("--min-duration-ms", type=float, default=0.0, help="Drop events shorter than this duration.")
    parser.add_argument("--top", type=int, default=None, help="Only print the N largest summary rows by total time.")
    parser.add_argument("--timeline", action="store_true", help="Also print individual matching events in timeline order.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional summary CSV output path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional summary JSON output path.")
    args = parser.parse_args()

    events = collect_trace_events(
        args.paths,
        pattern=args.pattern,
        min_duration_ms=args.min_duration_ms,
    )
    rows = summarize_events(events)

    print_summary(rows, top=args.top)
    if args.timeline:
        print_timeline(events)
    if args.csv_out is not None:
        write_summary_csv(rows, args.csv_out)
        print(f"csv : {args.csv_out}")
    if args.json_out is not None:
        write_summary_json(rows, args.json_out)
        print(f"json: {args.json_out}")


def collect_trace_events(
    paths: Sequence[Path],
    *,
    pattern: str = DEFAULT_PATTERN,
    min_duration_ms: float = 0.0,
) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for path in discover_trace_files(paths):
        events.extend(
            event
            for event in iter_trace_events(path, pattern=pattern)
            if event.duration_ms >= min_duration_ms
        )
    return sorted(events, key=lambda event: (event.source_path, event.start_ms or 0.0, event.name))


def discover_trace_files(paths: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_file():
            discovered.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for candidate in path.rglob("*"):
            if candidate.is_file() and _looks_like_trace(candidate):
                discovered.append(candidate)
    return sorted(dict.fromkeys(discovered))


def iter_trace_events(path: Path, *, pattern: str = DEFAULT_PATTERN) -> Iterable[TraceEvent]:
    payload = _read_json(path)
    raw_events = _raw_trace_events(payload)

    begin_stack: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        phase = raw_event.get("ph")
        if phase == "X":
            event = _complete_event(path, raw_event, pattern=pattern)
            if event is not None:
                yield event
            continue
        if phase == "B":
            begin_stack[(raw_event.get("pid"), raw_event.get("tid"))].append(raw_event)
            continue
        if phase == "E":
            stack = begin_stack.get((raw_event.get("pid"), raw_event.get("tid")))
            if not stack:
                continue
            begin = stack.pop()
            event = _begin_end_event(path, begin, raw_event, pattern=pattern)
            if event is not None:
                yield event


def summarize_events(events: Sequence[TraceEvent]) -> list[TraceSummaryRow]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for event in events:
        grouped[(event.source_path, event.case_name, event.solver_name, event.phase)].append(event.duration_ms)

    rows = []
    for (source_path, case_name, solver_name, phase), durations in grouped.items():
        rows.append(
            TraceSummaryRow(
                source_path=source_path,
                case_name=case_name,
                solver_name=solver_name,
                phase=phase,
                count=len(durations),
                total_ms=float(sum(durations)),
                mean_ms=float(statistics.fmean(durations)),
                median_ms=float(statistics.median(durations)),
                min_ms=float(min(durations)),
                max_ms=float(max(durations)),
            )
        )
    return sorted(rows, key=_summary_sort_key)


def print_summary(rows: Sequence[TraceSummaryRow], *, top: int | None = None) -> None:
    selected = sorted(rows, key=lambda row: row.total_ms, reverse=True)[:top] if top else list(rows)
    print("=== JAX benchmark trace summary ===")
    if not selected:
        print("No matching trace events found.")
        return

    print(
        f"{'case':32s} {'solver':18s} {'phase':24s} "
        f"{'n':>4s} {'total_ms':>11s} {'mean_ms':>10s} {'median_ms':>10s} {'min_ms':>10s} {'max_ms':>10s}"
    )
    print("-" * 137)
    for row in selected:
        print(
            f"{row.case_name[:32]:32s} {row.solver_name[:18]:18s} {row.phase[:24]:24s} "
            f"{row.count:4d} {row.total_ms:11.3f} {row.mean_ms:10.3f} "
            f"{row.median_ms:10.3f} {row.min_ms:10.3f} {row.max_ms:10.3f}"
        )


def print_timeline(events: Sequence[TraceEvent]) -> None:
    print("\n=== Matching trace timeline ===")
    if not events:
        print("No matching trace events found.")
        return
    print(f"{'start_ms':>12s} {'dur_ms':>10s} {'case':32s} {'solver':18s} {'phase':24s}")
    print("-" * 104)
    for event in events:
        start = "n/a" if event.start_ms is None else f"{event.start_ms:.3f}"
        print(
            f"{start:>12s} {event.duration_ms:10.3f} "
            f"{event.case_name[:32]:32s} {event.solver_name[:18]:18s} {event.phase[:24]:24s}"
        )


def write_summary_csv(rows: Sequence[TraceSummaryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TraceSummaryRow("", "", "", "", 0, 0.0, 0.0, 0.0, 0.0, 0.0).to_dict())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)


def write_summary_json(rows: Sequence[TraceSummaryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "rows": [row.to_dict() for row in rows]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _complete_event(path: Path, raw_event: Mapping[str, Any], *, pattern: str) -> TraceEvent | None:
    name = str(raw_event.get("name", ""))
    if pattern not in name or raw_event.get("dur") is None:
        return None
    duration_ms = _microseconds_to_ms(raw_event.get("dur"))
    if duration_ms is None:
        return None
    return _trace_event_from_parts(
        path,
        name,
        start_ms=_microseconds_to_ms(raw_event.get("ts")),
        duration_ms=duration_ms,
        pid=raw_event.get("pid"),
        tid=raw_event.get("tid"),
    )


def _begin_end_event(
    path: Path,
    begin: Mapping[str, Any],
    end: Mapping[str, Any],
    *,
    pattern: str,
) -> TraceEvent | None:
    name = str(begin.get("name", ""))
    if pattern not in name:
        return None
    start_ms = _microseconds_to_ms(begin.get("ts"))
    end_ms = _microseconds_to_ms(end.get("ts"))
    if start_ms is None or end_ms is None:
        return None
    return _trace_event_from_parts(
        path,
        name,
        start_ms=start_ms,
        duration_ms=max(0.0, end_ms - start_ms),
        pid=begin.get("pid"),
        tid=begin.get("tid"),
    )


def _trace_event_from_parts(
    path: Path,
    name: str,
    *,
    start_ms: float | None,
    duration_ms: float,
    pid: str | int | None,
    tid: str | int | None,
) -> TraceEvent:
    case_name, solver_name, phase = _parse_benchmark_name(name)
    return TraceEvent(
        source_path=str(path),
        name=name,
        case_name=case_name,
        solver_name=solver_name,
        phase=phase,
        start_ms=start_ms,
        duration_ms=float(duration_ms),
        pid=pid,
        tid=tid,
    )


def _parse_benchmark_name(name: str) -> tuple[str, str, str]:
    parts = name.split("/")
    if "benchmark" in parts:
        start = parts.index("benchmark")
        payload = parts[start + 1 :]
        if not payload:
            return "", "", name
        if len(payload) == 1:
            return payload[0], "", ""
        if len(payload) == 2:
            return payload[0], "", payload[1]
        return payload[0], payload[1], "/".join(payload[2:])

    try:
        start = parts.index("pool")
    except ValueError:
        return "", "", name

    payload = parts[start + 1 :]
    if not payload:
        return "pool", "", ""
    if len(payload) == 1:
        return "pool", payload[0], ""
    return "pool", payload[0], "/".join(payload[1:])


def _raw_trace_events(payload: Any) -> Iterable[Any]:
    if isinstance(payload, Mapping):
        events = payload.get("traceEvents")
        if isinstance(events, list):
            return events
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported trace JSON schema: expected a list or a traceEvents object.")


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _looks_like_trace(path: Path) -> bool:
    name = path.name
    return any(name.endswith(suffix) for suffix in TRACE_SUFFIXES)


def _microseconds_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def _summary_sort_key(row: TraceSummaryRow) -> tuple[str, str, int, str, str]:
    return (
        row.source_path,
        row.case_name,
        _phase_order(row.phase),
        row.solver_name,
        row.phase,
    )


def _phase_order(phase: str) -> int:
    known_order = {
        "build_axon": 10,
        "first_build_axon": 20,
        "first_solve": 30,
        "first_materialize": 40,
        "warmup_build_axon": 50,
        "warmup_solve": 60,
        "warmup_materialize": 70,
        "measured_build_axon": 80,
        "measured_solve": 90,
        "measured_materialize": 100,
    }
    return known_order.get(phase, 1000)


if __name__ == "__main__":
    main()
