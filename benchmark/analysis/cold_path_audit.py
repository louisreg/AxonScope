from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGE_GROUPS = (
    "curve",
    "pool_build",
    "dispatch",
    "runtime_prepare",
    "input_lowering",
    "kernel",
    "result_assembly",
    "other",
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
    "mean_ms",
    "max_ms",
    "rss_delta_mib_max",
    "rss_end_mib_max",
    "tracemalloc_peak_mib_max",
    "device_end_mib_max",
    "nvidia_smi_end_mib_max",
)

GROUP_FIELDS = (
    "run_label",
    "case_name",
    "script",
    "git_commit",
    "platform",
    "device_class",
    "n_axons",
    "nx",
    "tsim",
    "dt",
    "recording",
    "precision",
    "memory_trace",
    "group",
    "stage_count",
    "total_ms_sum",
    "self_ms_sum",
    "rss_delta_mib_max",
    "rss_end_mib_max",
    "tracemalloc_peak_mib_max",
    "device_end_mib_max",
)


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    run_label: str
    case_name: str
    script: str
    git_commit: str
    git_dirty: str
    platform: str
    device_class: str
    device_models: str
    host_os: str
    host_ram_total_gb: str
    n_axons: str
    nx: str
    tsim: str
    dt: str
    recording: str
    precision: str
    memory_trace: str
    profile: str


@dataclass(frozen=True)
class StageRow:
    context: RunContext
    group: str
    stage: str
    count: int
    total_ms: float
    self_ms: float
    mean_ms: float
    max_ms: float
    rss_delta_mib_max: float | None
    rss_end_mib_max: float | None
    tracemalloc_peak_mib_max: float | None
    device_end_mib_max: float | None
    nvidia_smi_end_mib_max: float | None

    def to_dict(self) -> dict[str, Any]:
        base = _context_dict(self.context)
        base.update(
            {
                "group": self.group,
                "stage": self.stage,
                "count": self.count,
                "total_ms": self.total_ms,
                "self_ms": self.self_ms,
                "mean_ms": self.mean_ms,
                "max_ms": self.max_ms,
                "rss_delta_mib_max": self.rss_delta_mib_max,
                "rss_end_mib_max": self.rss_end_mib_max,
                "tracemalloc_peak_mib_max": self.tracemalloc_peak_mib_max,
                "device_end_mib_max": self.device_end_mib_max,
                "nvidia_smi_end_mib_max": self.nvidia_smi_end_mib_max,
            }
        )
        return base


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a P11B cold-path timing and memory audit from benchmark runs.",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p11b_cold_path_audit"),
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    rows: list[StageRow] = []
    for run_dir in args.run_dirs:
        rows.extend(read_run(run_dir))
    if not rows:
        print("No benchmark stage rows found.")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    stage_csv = args.output / "cold_path_stage_rows.csv"
    group_csv = args.output / "cold_path_group_summary.csv"
    write_stage_rows(stage_csv, rows)
    write_group_summary(group_csv, rows)
    if not args.no_plots:
        write_plots(args.output / "plots", rows, top_n=max(int(args.top_n), 1))

    print(f"wrote: {stage_csv}")
    print(f"wrote: {group_csv}")
    print_summary(rows)
    return 0


def read_run(run_dir: Path) -> list[StageRow]:
    summary_path = run_dir / "summary.csv"
    memory_path = run_dir / "memory_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary.csv in {run_dir}")
    context = read_context(run_dir)
    summary = _read_csv_by_name(summary_path)
    memory = _read_csv_by_name(memory_path) if memory_path.is_file() else {}
    rows = []
    for stage, timing in summary.items():
        memory_row = memory.get(stage, {})
        rows.append(
            StageRow(
                context=context,
                group=classify_stage(stage),
                stage=stage,
                count=int(_float(timing.get("count")) or 0),
                total_ms=_float(timing.get("total_ms")) or 0.0,
                self_ms=_float(timing.get("self_ms")) or 0.0,
                mean_ms=_float(timing.get("mean_ms")) or 0.0,
                max_ms=_float(timing.get("max_ms")) or 0.0,
                rss_delta_mib_max=_float(memory_row.get("rss_delta_mib_max")),
                rss_end_mib_max=_float(memory_row.get("rss_end_mib_max")),
                tracemalloc_peak_mib_max=_bytes_to_mib(
                    _float(memory_row.get("tracemalloc_peak_delta_bytes_max"))
                ),
                device_end_mib_max=_bytes_to_mib(
                    _float(memory_row.get("device_bytes_in_use_end_max"))
                ),
                nvidia_smi_end_mib_max=_float(
                    memory_row.get("nvidia_smi_memory_used_end_mib_max")
                ),
            )
        )
    rows.sort(key=lambda row: row.total_ms, reverse=True)
    return rows


def read_context(run_dir: Path) -> RunContext:
    manifest = _read_json(run_dir / "manifest.json")
    environment = _read_json(run_dir / "environment.json")
    options = _mapping(manifest.get("options"))
    git = _mapping(environment.get("git"))
    profile = _mapping(environment.get("profile"))

    return RunContext(
        run_dir=run_dir,
        run_label=run_dir.name,
        case_name=str(manifest.get("case_name") or environment.get("benchmark_case_name") or run_dir.name),
        script=str(manifest.get("script") or environment.get("benchmark_script") or ""),
        git_commit=str(git.get("short_commit") or git.get("commit") or ""),
        git_dirty=str(git.get("dirty") if git.get("dirty") is not None else ""),
        platform=str(options.get("platform") or environment.get("compute_backend") or ""),
        device_class=str(environment.get("compute_device_class") or ""),
        device_models=";".join(str(item) for item in _sequence(environment.get("compute_device_models"))),
        host_os=str(environment.get("host_os") or ""),
        host_ram_total_gb=str(environment.get("host_ram_total_gb") or ""),
        n_axons=str(options.get("n_axons") or ""),
        nx=str(options.get("nx") or ""),
        tsim=str(options.get("tsim") or ""),
        dt=str(options.get("dt") or ""),
        recording=str(options.get("recording") or ""),
        precision=str(options.get("precision") or ""),
        memory_trace=str(options.get("memory_trace") or environment.get("memory_trace") or ""),
        profile=str(options.get("profile") if options.get("profile") is not None else profile.get("enabled") or ""),
    )


def classify_stage(stage: str) -> str:
    if stage.startswith("curve.build_pool"):
        return "pool_build"
    if stage.startswith("curve."):
        return "curve"
    if stage.startswith("dispatch."):
        return "dispatch"
    if stage.startswith("runtime.prepare"):
        return "runtime_prepare"
    if stage.startswith("inputs.") or stage == "observer.plan":
        return "input_lowering"
    if stage.startswith("kernel."):
        return "kernel"
    if stage.startswith("results.") or stage.startswith("curve.analyze"):
        return "result_assembly"
    if stage.startswith("simulation."):
        return "curve"
    return "other"


def write_stage_rows(path: Path, rows: Sequence[StageRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def write_group_summary(path: Path, rows: Sequence[StageRow]) -> None:
    grouped = summarize_groups(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        for row in grouped:
            writer.writerow({field: row.get(field) for field in GROUP_FIELDS})


def summarize_groups(rows: Sequence[StageRow]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.context.run_label, row.group)
        bucket = buckets.setdefault(
            key,
            {
                **_context_dict(row.context),
                "group": row.group,
                "stage_count": 0,
                "total_ms_sum": 0.0,
                "self_ms_sum": 0.0,
                "rss_delta_mib_max": None,
                "rss_end_mib_max": None,
                "tracemalloc_peak_mib_max": None,
                "device_end_mib_max": None,
            },
        )
        bucket["stage_count"] += 1
        bucket["total_ms_sum"] += row.total_ms
        bucket["self_ms_sum"] += row.self_ms
        _set_max(bucket, "rss_delta_mib_max", row.rss_delta_mib_max)
        _set_max(bucket, "rss_end_mib_max", row.rss_end_mib_max)
        _set_max(bucket, "tracemalloc_peak_mib_max", row.tracemalloc_peak_mib_max)
        _set_max(bucket, "device_end_mib_max", row.device_end_mib_max)

    result = list(buckets.values())
    order = {name: index for index, name in enumerate(STAGE_GROUPS)}
    result.sort(key=lambda row: (row["run_label"], order.get(str(row["group"]), 999)))
    return result


def write_plots(output: Path, rows: Sequence[StageRow], *, top_n: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    try:
        mpl_config = Path("benchmark/results/.matplotlib")
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        print(f"plots skipped: {type(exc).__name__}: {exc}")
        return

    group_rows = summarize_groups(rows)
    run_labels = tuple(dict.fromkeys(str(row["run_label"]) for row in group_rows))
    x = np.arange(len(run_labels))
    bottoms = np.zeros(len(run_labels), dtype=float)
    colors = {
        "curve": "#4C78A8",
        "pool_build": "#F28E2B",
        "dispatch": "#59A14F",
        "runtime_prepare": "#B07AA1",
        "input_lowering": "#76B7B2",
        "kernel": "#E15759",
        "result_assembly": "#EDC948",
        "other": "#BAB0AC",
    }
    by_run_group = {
        (str(row["run_label"]), str(row["group"])): float(row["self_ms_sum"])
        for row in group_rows
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    for group in STAGE_GROUPS:
        values = np.asarray([by_run_group.get((label, group), 0.0) for label in run_labels])
        if not np.any(values):
            continue
        ax.bar(x, values, bottom=bottoms, label=group, color=colors[group])
        bottoms += values
    ax.set_xticks(x, [_short(label, 34) for label in run_labels], rotation=20, ha="right")
    ax.set_ylabel("self time sum [ms]")
    ax.set_title("P11B cold-path time by stage group")
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(output / "cold_path_group_time.png", dpi=160)
    plt.close(fig)

    top_rows = sorted(rows, key=lambda row: row.total_ms, reverse=True)[:top_n]
    stage_labels = [f"{row.context.run_label}: {row.stage}" for row in top_rows]
    stage_values = [row.total_ms for row in top_rows]
    fig, ax = plt.subplots(figsize=(10.0, max(5.0, top_n * 0.33)), constrained_layout=True)
    y = np.arange(len(top_rows))
    ax.barh(y, stage_values, color="#59A14F")
    ax.set_yticks(y, [_short(label) for label in stage_labels])
    ax.invert_yaxis()
    ax.set_xlabel("total time [ms]")
    ax.set_title("Top cold-path benchmark spans")
    fig.savefig(output / "cold_path_top_stages.png", dpi=160)
    plt.close(fig)

    memory_rows = [
        row
        for row in rows
        if row.rss_delta_mib_max is not None or row.device_end_mib_max is not None
    ][:top_n]
    if memory_rows:
        rss = [row.rss_delta_mib_max or 0.0 for row in memory_rows]
        device = [row.device_end_mib_max or 0.0 for row in memory_rows]
        memory_labels = [f"{row.context.run_label}: {row.stage}" for row in memory_rows]
        y = np.arange(len(memory_rows))
        fig, ax = plt.subplots(figsize=(10.0, max(5.0, len(memory_rows) * 0.33)), constrained_layout=True)
        ax.barh(y - 0.18, rss, height=0.36, label="RSS delta", color="#F28E2B")
        ax.barh(y + 0.18, device, height=0.36, label="device end", color="#E15759")
        ax.set_yticks(y, [_short(label) for label in memory_labels])
        ax.invert_yaxis()
        ax.set_xlabel("memory [MiB]")
        ax.set_title("Cold-path memory pressure by span")
        ax.legend()
        fig.savefig(output / "cold_path_memory.png", dpi=160)
        plt.close(fig)


def print_summary(rows: Sequence[StageRow]) -> None:
    print("\nTop stages:")
    print("run,group,stage,total_ms,self_ms,rss_delta_mib_max")
    for row in sorted(rows, key=lambda item: item.total_ms, reverse=True)[:10]:
        print(
            f"{row.context.run_label},{row.group},{row.stage},"
            f"{row.total_ms:.3f},{row.self_ms:.3f},{_format_float(row.rss_delta_mib_max)}"
        )


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


def _read_csv_by_name(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["name"]): row for row in rows if row.get("name")}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bytes_to_mib(value: float | None) -> float | None:
    if value is None:
        return None
    return value / float(1024**2)


def _set_max(row: dict[str, Any], key: str, value: float | None) -> None:
    if value is None:
        return
    current = row.get(key)
    row[key] = value if current is None else max(float(current), value)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> Iterable[Any]:
    return value if isinstance(value, list | tuple) else ()


def _short(value: str, limit: int = 70) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
