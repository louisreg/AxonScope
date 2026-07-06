from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.analysis.cold_path_audit import RunContext, classify_stage, read_context


EVENT_FIELDS = (
    "run_label",
    "case_name",
    "script",
    "git_commit",
    "git_dirty",
    "platform",
    "device_class",
    "device_models",
    "host_os",
    "host_ram_total_gb",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "recording",
    "precision",
    "memory_trace",
    "profile",
    "event_id",
    "parent_event_id",
    "depth",
    "group",
    "stage",
    "duration_ms",
    "self_ms",
    "workflow_ms",
    "workflow_share",
    "phase",
    "repeat",
    "iteration",
    "curve",
    "rss_delta_mib",
    "rss_end_mib",
    "tracemalloc_peak_mib",
    "device_end_mib",
    "device_delta_mib",
    "nvidia_smi_end_mib",
    "cache_hits",
    "cache_misses",
    "cache_summary",
)

STAGE_FIELDS = (
    "run_label",
    "case_name",
    "script",
    "git_commit",
    "git_dirty",
    "platform",
    "device_class",
    "device_models",
    "host_os",
    "host_ram_total_gb",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "recording",
    "precision",
    "memory_trace",
    "profile",
    "group",
    "stage",
    "count",
    "total_ms",
    "self_ms",
    "mean_self_ms",
    "max_ms",
    "workflow_ms",
    "workflow_share",
    "rss_delta_mib_max",
    "rss_end_mib_max",
    "tracemalloc_peak_mib_max",
    "device_end_mib_max",
    "device_delta_mib_max",
    "nvidia_smi_end_mib_max",
    "cache_hits",
    "cache_misses",
    "cache_summary",
)

GROUP_FIELDS = (
    "run_label",
    "case_name",
    "script",
    "git_commit",
    "git_dirty",
    "platform",
    "device_class",
    "device_models",
    "host_os",
    "host_ram_total_gb",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "recording",
    "precision",
    "memory_trace",
    "profile",
    "group",
    "stage_count",
    "event_count",
    "total_ms",
    "self_ms",
    "workflow_ms",
    "workflow_share",
    "rss_delta_mib_max",
    "rss_end_mib_max",
    "tracemalloc_peak_mib_max",
    "device_end_mib_max",
    "device_delta_mib_max",
    "nvidia_smi_end_mib_max",
    "cache_hits",
    "cache_misses",
    "cache_summary",
)

_CACHE_DETAIL_SUFFIXES = (
    "_cache_keys",
    "_cache_paths",
    "_cache_hashes",
    "_cache_reasons",
)


@dataclass(frozen=True)
class BottleneckEvent:
    context: RunContext
    event_id: int
    parent_event_id: int | None
    depth: int
    group: str
    stage: str
    duration_ms: float
    self_ms: float
    workflow_ms: float
    phase: str
    repeat: str
    iteration: str
    curve: str
    rss_delta_mib: float | None
    rss_end_mib: float | None
    tracemalloc_peak_mib: float | None
    device_end_mib: float | None
    device_delta_mib: float | None
    nvidia_smi_end_mib: float | None
    cache_hits: int
    cache_misses: int
    cache_summary: str

    @property
    def workflow_share(self) -> float:
        if self.workflow_ms <= 0.0:
            return 0.0
        return self.self_ms / self.workflow_ms

    def to_dict(self) -> dict[str, Any]:
        result = _context_dict(self.context)
        result.update(
            {
                "event_id": self.event_id,
                "parent_event_id": "" if self.parent_event_id is None else self.parent_event_id,
                "depth": self.depth,
                "group": self.group,
                "stage": self.stage,
                "duration_ms": self.duration_ms,
                "self_ms": self.self_ms,
                "workflow_ms": self.workflow_ms,
                "workflow_share": self.workflow_share,
                "phase": self.phase,
                "repeat": self.repeat,
                "iteration": self.iteration,
                "curve": self.curve,
                "rss_delta_mib": self.rss_delta_mib,
                "rss_end_mib": self.rss_end_mib,
                "tracemalloc_peak_mib": self.tracemalloc_peak_mib,
                "device_end_mib": self.device_end_mib,
                "device_delta_mib": self.device_delta_mib,
                "nvidia_smi_end_mib": self.nvidia_smi_end_mib,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_summary": self.cache_summary,
            }
        )
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a P11B bottleneck report from benchmark event traces.",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_bottleneck_report"),
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=("all", "repeat", "warmup"),
        help="Filter events by inherited benchmark phase before ranking bottlenecks.",
    )
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args(argv)

    events: list[BottleneckEvent] = []
    for run_dir in args.run_dirs:
        events.extend(read_event_rows(run_dir, phase_filter=args.phase))
    if not events:
        print("No benchmark events found.")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    event_csv = args.output / "bottleneck_event_rows.csv"
    stage_csv = args.output / "bottleneck_stage_rank.csv"
    group_csv = args.output / "bottleneck_group_rank.csv"
    report_md = args.output / "bottleneck_report.md"

    stage_rows = summarize_stages(events)
    group_rows = summarize_groups(stage_rows)

    _write_csv(event_csv, EVENT_FIELDS, [row.to_dict() for row in events])
    _write_csv(stage_csv, STAGE_FIELDS, stage_rows)
    _write_csv(group_csv, GROUP_FIELDS, group_rows)
    write_report(
        report_md,
        events,
        stage_rows,
        group_rows,
        top_n=max(args.top_n, 1),
        phase_filter=args.phase,
    )

    print(f"wrote: {event_csv}")
    print(f"wrote: {stage_csv}")
    print(f"wrote: {group_csv}")
    print(f"wrote: {report_md}")
    _print_summary(stage_rows, top_n=max(args.top_n, 1))
    return 0


def read_event_rows(run_dir: Path, *, phase_filter: str = "all") -> list[BottleneckEvent]:
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"missing events.jsonl in {run_dir}")

    base_context = read_context(run_dir)
    context = replace(base_context, run_label=_run_label(run_dir, base_context))
    raw_events = [_mapping(json.loads(line)) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {
        int(_float(event.get("event_id")) or 0): event
        for event in raw_events
    }
    phases = {
        event_id: _event_phase(event, by_id)
        for event_id, event in by_id.items()
    }
    selected_events = [
        event
        for event in raw_events
        if phase_filter == "all"
        or phases.get(int(_float(event.get("event_id")) or 0), "") == phase_filter
    ]
    selected_ids = {
        int(_float(event.get("event_id")) or 0)
        for event in selected_events
    }
    child_ms: defaultdict[int, float] = defaultdict(float)
    workflow_ms = 0.0
    for event in selected_events:
        duration_ms = _float(event.get("duration_ms")) or 0.0
        parent_id = _int_or_none(event.get("parent_event_id"))
        if parent_id is None or parent_id not in selected_ids:
            workflow_ms += duration_ms
        else:
            child_ms[parent_id] += duration_ms

    rows = []
    for event in selected_events:
        metadata = _mapping(event.get("metadata"))
        memory = _mapping(metadata.get("memory"))
        event_id = int(_float(event.get("event_id")) or 0)
        duration_ms = _float(event.get("duration_ms")) or 0.0
        self_ms = max(duration_ms - child_ms.get(event_id, 0.0), 0.0)
        cache_hits, cache_misses, cache_summary = _cache_signals(metadata)
        stage = str(event.get("name") or "")
        rows.append(
            BottleneckEvent(
                context=context,
                event_id=event_id,
                parent_event_id=_int_or_none(event.get("parent_event_id")),
                depth=int(_float(event.get("depth")) or 0),
                group=classify_stage(stage),
                stage=stage,
                duration_ms=duration_ms,
                self_ms=self_ms,
                workflow_ms=workflow_ms,
                phase=phases.get(event_id, ""),
                repeat=str(metadata.get("repeat") if metadata.get("repeat") is not None else ""),
                iteration=str(
                    metadata.get("iteration") if metadata.get("iteration") is not None else ""
                ),
                curve=str(metadata.get("curve") or ""),
                rss_delta_mib=_float(memory.get("rss_delta_mib")),
                rss_end_mib=_float(memory.get("rss_end_mib")),
                tracemalloc_peak_mib=_bytes_to_mib(_float(memory.get("tracemalloc_peak_delta_bytes"))),
                device_end_mib=_bytes_to_mib(_float(memory.get("device_bytes_in_use_end"))),
                device_delta_mib=_bytes_to_mib(_float(memory.get("device_bytes_in_use_delta"))),
                nvidia_smi_end_mib=_float(memory.get("nvidia_smi_memory_used_end_mib")),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_summary=cache_summary,
            )
        )

    rows.sort(key=lambda row: (row.context.run_label, -row.self_ms, row.event_id))
    return rows


def summarize_stages(events: Sequence[BottleneckEvent]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    cache_buckets: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for event in events:
        key = (event.context.run_label, event.group, event.stage)
        bucket = buckets.setdefault(
            key,
            {
                **_context_dict(event.context),
                "group": event.group,
                "stage": event.stage,
                "count": 0,
                "total_ms": 0.0,
                "self_ms": 0.0,
                "mean_self_ms": 0.0,
                "max_ms": 0.0,
                "workflow_ms": event.workflow_ms,
                "workflow_share": 0.0,
                "rss_delta_mib_max": None,
                "rss_end_mib_max": None,
                "tracemalloc_peak_mib_max": None,
                "device_end_mib_max": None,
                "device_delta_mib_max": None,
                "nvidia_smi_end_mib_max": None,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_summary": "",
            },
        )
        bucket["count"] += 1
        bucket["total_ms"] += event.duration_ms
        bucket["self_ms"] += event.self_ms
        bucket["max_ms"] = max(float(bucket["max_ms"]), event.duration_ms)
        bucket["cache_hits"] += event.cache_hits
        bucket["cache_misses"] += event.cache_misses
        _set_max(bucket, "rss_delta_mib_max", event.rss_delta_mib)
        _set_max(bucket, "rss_end_mib_max", event.rss_end_mib)
        _set_max(bucket, "tracemalloc_peak_mib_max", event.tracemalloc_peak_mib)
        _set_max(bucket, "device_end_mib_max", event.device_end_mib)
        _set_max(bucket, "device_delta_mib_max", event.device_delta_mib)
        _set_max(bucket, "nvidia_smi_end_mib_max", event.nvidia_smi_end_mib)
        if event.cache_summary:
            cache_buckets[key].update(event.cache_summary.split("; "))

    rows = []
    for key, row in buckets.items():
        count = int(row["count"])
        workflow_ms = float(row["workflow_ms"])
        row["mean_self_ms"] = float(row["self_ms"]) / float(count) if count else 0.0
        row["workflow_share"] = float(row["self_ms"]) / workflow_ms if workflow_ms else 0.0
        row["cache_summary"] = _format_counter(cache_buckets.get(key, Counter()))
        rows.append(row)
    rows.sort(key=lambda row: (str(row["run_label"]), -float(row["self_ms"])))
    return rows


def summarize_groups(stage_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    cache_buckets: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    stages_by_group: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for stage in stage_rows:
        key = (str(stage["run_label"]), str(stage["group"]))
        bucket = buckets.setdefault(
            key,
            {
                **{field: stage.get(field) for field in _CONTEXT_FIELDS},
                "group": stage["group"],
                "stage_count": 0,
                "event_count": 0,
                "total_ms": 0.0,
                "self_ms": 0.0,
                "workflow_ms": stage["workflow_ms"],
                "workflow_share": 0.0,
                "rss_delta_mib_max": None,
                "rss_end_mib_max": None,
                "tracemalloc_peak_mib_max": None,
                "device_end_mib_max": None,
                "device_delta_mib_max": None,
                "nvidia_smi_end_mib_max": None,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_summary": "",
            },
        )
        stages_by_group[key].add(str(stage["stage"]))
        bucket["event_count"] += int(stage["count"])
        bucket["total_ms"] += float(stage["total_ms"])
        bucket["self_ms"] += float(stage["self_ms"])
        bucket["cache_hits"] += int(stage["cache_hits"])
        bucket["cache_misses"] += int(stage["cache_misses"])
        _set_max(bucket, "rss_delta_mib_max", _float(stage.get("rss_delta_mib_max")))
        _set_max(bucket, "rss_end_mib_max", _float(stage.get("rss_end_mib_max")))
        _set_max(bucket, "tracemalloc_peak_mib_max", _float(stage.get("tracemalloc_peak_mib_max")))
        _set_max(bucket, "device_end_mib_max", _float(stage.get("device_end_mib_max")))
        _set_max(bucket, "device_delta_mib_max", _float(stage.get("device_delta_mib_max")))
        _set_max(bucket, "nvidia_smi_end_mib_max", _float(stage.get("nvidia_smi_end_mib_max")))
        if stage.get("cache_summary"):
            cache_buckets[key].update(str(stage["cache_summary"]).split("; "))

    rows = []
    for key, row in buckets.items():
        workflow_ms = float(row["workflow_ms"])
        row["stage_count"] = len(stages_by_group[key])
        row["workflow_share"] = float(row["self_ms"]) / workflow_ms if workflow_ms else 0.0
        row["cache_summary"] = _format_counter(cache_buckets.get(key, Counter()))
        rows.append(row)
    rows.sort(key=lambda row: (str(row["run_label"]), -float(row["self_ms"])))
    return rows


def write_report(
    path: Path,
    events: Sequence[BottleneckEvent],
    stage_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
    phase_filter: str,
) -> None:
    by_run_stage = _group_by_run(stage_rows)
    by_run_group = _group_by_run(group_rows)
    contexts = _contexts_by_run(events)
    lines: list[str] = [
        "# P11B Bottleneck Report",
        "",
        "This report ranks benchmark spans by exclusive self time. Use it to select",
        "the next optimization target; it is not a speed claim by itself.",
        "",
        f"Phase filter: `{phase_filter}`.",
        "",
        "## Runs",
        "",
    ]
    lines.extend(
        _markdown_table(
            (
                "run",
                "script",
                "platform",
                "device",
                "git",
                "dirty",
                "Naxons",
                "Nx",
                "dt",
                "memory",
            ),
            [
                (
                    label,
                    context.script,
                    context.platform,
                    _short(context.device_models or context.device_class, 26),
                    context.git_commit,
                    context.git_dirty,
                    context.n_axons,
                    context.nx,
                    context.dt,
                    context.memory_trace,
                )
                for label, context in contexts.items()
            ],
        )
    )
    for run_label in contexts:
        lines.extend(
            [
                "",
                f"## {run_label}",
                "",
                "Top stages by self time:",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                ("stage", "group", "count", "self ms", "share", "total ms", "cache"),
                [
                    (
                        _short(str(row["stage"]), 42),
                        row["group"],
                        row["count"],
                        _fmt(float(row["self_ms"])),
                        _pct(float(row["workflow_share"])),
                        _fmt(float(row["total_ms"])),
                        _short(str(row["cache_summary"]), 36),
                    )
                    for row in by_run_stage.get(run_label, [])[:top_n]
                ],
            )
        )
        lines.extend(["", "Stage groups:", ""])
        lines.extend(
            _markdown_table(
                ("group", "self ms", "share", "events", "cache miss", "rss max", "device max", "nvidia max"),
                [
                    (
                        row["group"],
                        _fmt(float(row["self_ms"])),
                        _pct(float(row["workflow_share"])),
                        row["event_count"],
                        row["cache_misses"],
                        _fmt_optional(row.get("rss_delta_mib_max")),
                        _fmt_optional(row.get("device_end_mib_max")),
                        _fmt_optional(row.get("nvidia_smi_end_mib_max")),
                    )
                    for row in by_run_group.get(run_label, [])
                ],
            )
        )

    cache_rows = [
        row
        for row in stage_rows
        if int(row.get("cache_hits") or 0) or int(row.get("cache_misses") or 0)
    ]
    if cache_rows:
        lines.extend(["", "## Cache Signals", ""])
        lines.extend(
            _markdown_table(
                ("run", "stage", "hits", "misses", "summary"),
                [
                    (
                        row["run_label"],
                        _short(str(row["stage"]), 42),
                        row["cache_hits"],
                        row["cache_misses"],
                        _short(str(row["cache_summary"]), 54),
                    )
                    for row in sorted(
                        cache_rows,
                        key=lambda row: (
                            -int(row.get("cache_misses") or 0),
                            -float(row.get("self_ms") or 0.0),
                        ),
                    )[:top_n]
                ],
            )
        )

    if any(_float(row.get("nvidia_smi_end_mib_max")) for row in stage_rows):
        lines.extend(
            [
                "",
                "## Memory Note",
                "",
                "GPU runs can show high `nvidia-smi` memory because JAX reserves a device",
                "pool. Prefer `device_end_mib_max` for live JAX array pressure, and keep",
                "`nvidia_smi_end_mib_max` as allocator/process context.",
            ]
        )
    if any(context.memory_trace in {"device", "all"} for context in contexts.values()):
        lines.extend(
            [
                "",
                "## Timing Hygiene",
                "",
                "`memory_trace=device` and `memory_trace=all` sample JAX device",
                "stats and `nvidia-smi` around benchmark spans. They are useful for",
                "memory cartography, but they can add visible per-span overhead on GPU",
                "and should not be used alone for fine solver timing. For solver",
                "optimization runs, pair a timing-focused run (`memory_trace=off` or",
                "`rss`) with a tiny memory/profiling run.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cache_signals(metadata: Mapping[str, Any]) -> tuple[int, int, str]:
    items: list[tuple[str, Any]] = []
    for key, value in metadata.items():
        if "cache" not in str(key):
            continue
        if any(str(key).endswith(suffix) for suffix in _CACHE_DETAIL_SUFFIXES):
            continue
        items.append((str(key), value))

    hits = 0
    misses = 0
    summary_parts = []
    for key, value in items:
        values = list(_flatten(value))
        hits += sum(1 for item in values if str(item).lower() == "hit")
        misses += sum(1 for item in values if str(item).lower() == "miss")
        summary_parts.append(f"{key}={_compact_value(value)}")
    return hits, misses, "; ".join(summary_parts)


def _flatten(value: Any) -> Sequence[Any]:
    if isinstance(value, Mapping):
        result: list[Any] = []
        for nested in value.values():
            result.extend(_flatten(nested))
        return result
    if isinstance(value, list | tuple | set):
        result = []
        for nested in value:
            result.extend(_flatten(nested))
        return result
    return (value,)


def _compact_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        if all(isinstance(item, str) and item in {"hit", "miss"} for item in value):
            return "/".join(value)
        return f"{len(value)} items"
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_summary(stage_rows: Sequence[Mapping[str, Any]], *, top_n: int) -> None:
    print("\nTop bottleneck stages:")
    print("run,group,stage,self_ms,share,total_ms,cache_misses")
    for row in sorted(stage_rows, key=lambda item: -float(item.get("self_ms") or 0.0))[:top_n]:
        print(
            f"{row['run_label']},{row['group']},{row['stage']},"
            f"{float(row['self_ms']):.3f},{float(row['workflow_share']):.3f},"
            f"{float(row['total_ms']):.3f},{int(row['cache_misses'])}"
        )


def _contexts_by_run(events: Sequence[BottleneckEvent]) -> dict[str, RunContext]:
    contexts: dict[str, RunContext] = {}
    for event in events:
        contexts.setdefault(event.context.run_label, event.context)
    return contexts


def _run_label(run_dir: Path, context: RunContext) -> str:
    parts = [
        _script_short(context.script),
        context.platform,
        context.recording,
        run_dir.name,
        context.git_commit,
    ]
    label = "_".join(part for part in parts if part)
    return label or run_dir.name


def _script_short(script: str) -> str:
    if script == "threshold_curves":
        return "thr"
    if script == "recruitment_curves":
        return "rec"
    return script


def _group_by_run(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["run_label"])].append(row)
    return dict(result)


def _markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = ["| " + " | ".join(str(header) for header in headers) + " |"]
    result.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        result.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return result


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _context_dict(context: RunContext) -> dict[str, Any]:
    return {
        "run_label": context.run_label,
        "case_name": context.case_name,
        "script": context.script,
        "git_commit": context.git_commit,
        "git_dirty": context.git_dirty,
        "platform": context.platform,
        "device_class": context.device_class,
        "device_models": context.device_models,
        "host_os": context.host_os,
        "host_ram_total_gb": context.host_ram_total_gb,
        "n_axons": context.n_axons,
        "nx": context.nx,
        "tsim": context.tsim,
        "dt": context.dt,
        "recording": context.recording,
        "precision": context.precision,
        "memory_trace": context.memory_trace,
        "profile": context.profile,
    }


_CONTEXT_FIELDS = (
    "run_label",
    "case_name",
    "script",
    "git_commit",
    "git_dirty",
    "platform",
    "device_class",
    "device_models",
    "host_os",
    "host_ram_total_gb",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "recording",
    "precision",
    "memory_trace",
    "profile",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _event_phase(
    event: Mapping[str, Any],
    by_id: Mapping[int, Mapping[str, Any]],
) -> str:
    current: Mapping[str, Any] | None = event
    seen: set[int] = set()
    while current is not None:
        metadata = _mapping(current.get("metadata"))
        phase = metadata.get("phase")
        if phase:
            return str(phase)
        parent = current.get("parent_event_id")
        if parent is None:
            return ""
        parent_id = _int_or_none(parent)
        if parent_id is None or parent_id in seen:
            return ""
        seen.add(parent_id)
        current = by_id.get(parent_id)
    return ""


def _bytes_to_mib(value: float | None) -> float | None:
    if value is None:
        return None
    return value / float(1024**2)


def _set_max(row: dict[str, Any], key: str, value: float | None) -> None:
    if value is None:
        return
    current = row.get(key)
    row[key] = value if current is None else max(float(current), value)


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return "; ".join(f"{key} x{count}" if count > 1 else key for key, count in counter.most_common())


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def _fmt_optional(value: Any) -> str:
    parsed = _float(value)
    return "" if parsed is None else _fmt(parsed)


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _short(value: str, limit: int = 70) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
