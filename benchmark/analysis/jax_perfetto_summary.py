"""Summarize named events and tracks from a JAX Perfetto JSON trace."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_trace_events(path: Path) -> list[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("Perfetto trace must contain a traceEvents list.")
    return [event for event in events if isinstance(event, Mapping)]


def complete_events(
    events: Iterable[Mapping[str, Any]],
    *,
    pattern: str = "",
    min_duration_ms: float = 0.0,
) -> list[Mapping[str, Any]]:
    selected = []
    for event in events:
        name = str(event.get("name", ""))
        duration_us = event.get("dur")
        if event.get("ph") != "X" or duration_us is None or pattern not in name:
            continue
        if float(duration_us) / 1000.0 < min_duration_ms:
            continue
        selected.append(event)
    return selected


def track_labels(
    events: Iterable[Mapping[str, Any]],
) -> tuple[dict[Any, str], dict[tuple[Any, Any], str]]:
    processes: dict[Any, str] = {}
    threads: dict[tuple[Any, Any], str] = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        name = event.get("name")
        label = str(event.get("args", {}).get("name", ""))
        if name == "process_name":
            processes[event.get("pid")] = label
        elif name == "thread_name":
            threads[(event.get("pid"), event.get("tid"))] = label
    return processes, threads


def summarize_names(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, str], list[float]] = defaultdict(list)
    for event in events:
        grouped[(event.get("pid"), event.get("tid"), str(event.get("name", "")))].append(
            float(event["dur"]) / 1000.0
        )
    rows = [
        {
            "pid": pid,
            "tid": tid,
            "name": name,
            "count": len(durations),
            "total_ms": sum(durations),
            "mean_ms": sum(durations) / len(durations),
            "max_ms": max(durations),
        }
        for (pid, tid, name), durations in grouped.items()
    ]
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--pattern", default="")
    parser.add_argument(
        "--track-pattern",
        default="",
        help="Keep tracks whose process/thread label contains this text.",
    )
    parser.add_argument("--min-duration-ms", type=float, default=0.05)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--tracks", action="store_true")
    args = parser.parse_args(argv)

    events = load_trace_events(args.trace)
    processes, threads = track_labels(events)
    selected = complete_events(
        events,
        pattern=args.pattern,
        min_duration_ms=args.min_duration_ms,
    )
    if args.track_pattern:
        selected = [
            event
            for event in selected
            if args.track_pattern
            in (
                processes.get(event.get("pid"), "")
                + "/"
                + threads.get((event.get("pid"), event.get("tid")), "")
            )
        ]
    if args.tracks:
        grouped_tracks: dict[tuple[Any, Any], list[float]] = defaultdict(list)
        for event in selected:
            grouped_tracks[(event.get("pid"), event.get("tid"))].append(
                float(event["dur"]) / 1000.0
            )
        print("total_ms  count  process/thread")
        for (pid, tid), durations in sorted(
            grouped_tracks.items(),
            key=lambda item: sum(item[1]),
            reverse=True,
        ):
            process = processes.get(pid, str(pid))
            thread = threads.get((pid, tid), str(tid))
            print(f"{sum(durations):8.3f} {len(durations):6d}  {process}/{thread}")
        print()
    rows = summarize_names(selected)[: args.top]
    print("total_ms  count  mean_ms   max_ms  track | event")
    for row in rows:
        process = processes.get(row["pid"], str(row["pid"]))
        thread = threads.get((row["pid"], row["tid"]), str(row["tid"]))
        print(
            f"{row['total_ms']:8.3f} {row['count']:6d} "
            f"{row['mean_ms']:8.3f} {row['max_ms']:8.3f}  "
            f"{process}/{thread} | {row['name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
