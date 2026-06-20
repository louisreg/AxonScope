"""Compare Kaggle realistic-example recording modes.

The script consumes downloaded Kaggle artifact directories for:

- full Vm recording
- center/single-probe Vm recording
- VmRaster observer-only recording

It writes long-form CSVs and plots for complete workflow timings, selected
profile spans, and memory metrics. It intentionally works from downloaded
artifacts so the expensive Kaggle runs can be reused.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_EVENTS = (
    "simulation.pool.total",
    "dispatch.group.total",
    "runtime.prepare",
    "dispatch.build_plan",
    "kernel.enqueue",
    "kernel.wait",
    "results.split_batch",
    "inputs.extracellular",
    "inputs.intracellular",
    "inputs.positions",
    "results.to_public",
)


@dataclass(frozen=True)
class RunInput:
    mode: str
    artifact: Path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = collect_run_inputs(args)
    if not runs:
        raise SystemExit("No runs provided.")

    prefix = args.prefix or datetime.now().strftime("recording_mode_compare_%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str]] = []
    profile_rows: list[dict[str, str]] = []
    for run in runs:
        summary_path = find_summary_csv(run.artifact)
        profile_path = find_profile_csv(run.artifact)
        for row in read_csv_rows(summary_path):
            summary_rows.append(add_mode_columns(row, run, summary_path, profile_path))
        for row in read_csv_rows(profile_path):
            profile_rows.append(add_mode_columns(row, run, summary_path, profile_path))

    event_rows = aggregate_profile_events(profile_rows, events=args.events)
    memory_rows = build_memory_rows(summary_rows, event_rows)

    summary_csv = out_dir / f"{prefix}_summary.csv"
    profile_csv = out_dir / f"{prefix}_profile_events.csv"
    memory_csv = out_dir / f"{prefix}_memory.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(profile_csv, event_rows)
    write_csv(memory_csv, memory_rows)
    print(f"summary_csv: {summary_csv}")
    print(f"profile_events_csv: {profile_csv}")
    print(f"memory_csv: {memory_csv}")

    if args.plots:
        for plot_path in write_plots(
            summary_rows=summary_rows,
            event_rows=event_rows,
            memory_rows=memory_rows,
            out_dir=out_dir,
            prefix=prefix,
            events=args.events,
        ):
            print(f"plot: {plot_path}")

    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, help="Kaggle artifact dir for full Vm.")
    parser.add_argument("--full-cpu", type=Path, help="Kaggle artifact dir for CPU full Vm.")
    parser.add_argument("--center", type=Path, help="Kaggle artifact dir for center Vm.")
    parser.add_argument("--center-cpu", type=Path, help="Kaggle artifact dir for CPU center Vm.")
    parser.add_argument(
        "--observer",
        type=Path,
        help="Kaggle artifact dir for VmRaster observer-only.",
    )
    parser.add_argument(
        "--observer-cpu",
        type=Path,
        help="Kaggle artifact dir for CPU VmRaster observer-only.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help="Additional or custom run input, e.g. observer=/path/to/run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/realistic_examples/recording_mode_compare"),
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--events",
        nargs="+",
        default=list(DEFAULT_EVENTS),
        help="Profile event names to aggregate and plot.",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def collect_run_inputs(args: argparse.Namespace) -> list[RunInput]:
    runs: list[RunInput] = []
    for mode, path in (
        ("full", args.full),
        ("full_cpu", args.full_cpu),
        ("center", args.center),
        ("center_cpu", args.center_cpu),
        ("observer", args.observer),
        ("observer_cpu", args.observer_cpu),
    ):
        if path is not None:
            runs.append(RunInput(mode=mode, artifact=path))

    for value in args.run:
        if "=" not in value:
            raise ValueError(f"--run must be MODE=PATH, got {value!r}")
        mode, raw_path = value.split("=", 1)
        runs.append(RunInput(mode=sanitize_mode(mode), artifact=Path(raw_path)))

    seen: set[str] = set()
    unique_runs = []
    for run in runs:
        if run.mode in seen:
            raise ValueError(f"Duplicate run mode {run.mode!r}.")
        seen.add(run.mode)
        unique_runs.append(run)
    return unique_runs


def sanitize_mode(value: str) -> str:
    clean = value.strip().replace(" ", "_")
    if not clean:
        raise ValueError("empty mode label")
    return clean


def find_summary_csv(path: Path) -> Path:
    return find_single_csv(
        path,
        suffixes=(
            "realistic_examples_gpu.csv",
            "realistic_examples_cpu.csv",
        ),
    )


def find_profile_csv(path: Path) -> Path:
    return find_single_csv(
        path,
        suffixes=(
            "realistic_examples_gpu_profile.csv",
            "realistic_examples_cpu_profile.csv",
        ),
    )


def find_single_csv(path: Path, *, suffixes: Sequence[str]) -> Path:
    if path.is_file():
        if path.name in suffixes:
            return path
        if path.name.endswith("_profile.csv") and any(
            suffix.endswith("_profile.csv") for suffix in suffixes
        ):
            return path
        raise FileNotFoundError(f"{path} is not one of {', '.join(suffixes)}.")

    for suffix in suffixes:
        candidates = sorted(path.glob(f"**/{suffix}"))
        if not candidates:
            continue
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    raise FileNotFoundError(f"Could not find any of {', '.join(suffixes)} under {path}.")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add_mode_columns(
    row: dict[str, str],
    run: RunInput,
    summary_path: Path,
    profile_path: Path,
) -> dict[str, str]:
    output = dict(row)
    output["mode"] = run.mode
    output["platform_suffix"] = platform_suffix(summary_path)
    output["artifact_dir"] = str(run.artifact)
    output["summary_csv"] = str(summary_path)
    output["profile_csv"] = str(profile_path)
    return output


def platform_suffix(path: Path) -> str:
    name = path.name
    if "_cpu" in name:
        return "cpu"
    if "_gpu" in name:
        return "gpu"
    return "unknown"


def aggregate_profile_events(
    rows: Sequence[dict[str, str]],
    *,
    events: Sequence[str],
) -> list[dict[str, str]]:
    wanted = set(events)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "warm_repeat":
            continue
        if row.get("event_name") not in wanted:
            continue
        groups[event_key(row)].append(row)

    out_rows = []
    for key in sorted(groups):
        group_rows = groups[key]
        first = group_rows[0]
        out_rows.append(
            {
                "mode": first["mode"],
                "workflow": first["workflow"],
                "fiber_type": first["fiber_type"],
                "run_count": first["run_count"],
                "duration_ms": first["duration_ms"],
                "dt_ms": first["dt_ms"],
                "recording": first["recording"],
                "protocol_steps": first["protocol_steps"],
                "event_name": first["event_name"],
                "warm_repeats": str(len(group_rows)),
                "event_count_mean": fmt(mean(row.get("event_count") for row in group_rows)),
                "total_ms_mean": fmt(mean(row.get("total_ms") for row in group_rows)),
                "self_ms_mean": fmt(mean(row.get("self_ms") for row in group_rows)),
                "max_ms_mean": fmt(mean(row.get("max_ms") for row in group_rows)),
                "run_elapsed_s_mean": fmt(mean(row.get("run_elapsed_s") for row in group_rows)),
                "memory_estimate_total_mib_max": fmt(
                    max_number(row.get("memory_estimate_total_mib_max") for row in group_rows)
                ),
                "memory_estimate_device_fraction_max": fmt(
                    max_number(
                        row.get("memory_estimate_device_fraction_max") for row in group_rows
                    )
                ),
                "vstim_footprint_cache_hits_sum": fmt(
                    sum_number(row.get("vstim_footprint_cache_hits") for row in group_rows)
                ),
                "vstim_footprint_cache_misses_sum": fmt(
                    sum_number(row.get("vstim_footprint_cache_misses") for row in group_rows)
                ),
            }
        )
    return out_rows


def event_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["mode"],
        row["workflow"],
        row["fiber_type"],
        row["run_count"],
        row["duration_ms"],
        row["dt_ms"],
        row["recording"],
        row["protocol_steps"],
        row["event_name"],
    )


def build_memory_rows(
    summary_rows: Sequence[dict[str, str]],
    event_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    estimates: dict[tuple[str, ...], float] = {}
    for row in event_rows:
        key = case_key(row)
        value = parse_number(row.get("memory_estimate_total_mib_max"))
        if math.isfinite(value):
            estimates[key] = max(value, estimates.get(key, float("-inf")))

    out_rows = []
    for row in summary_rows:
        key = case_key(row)
        out_rows.append(
            {
                "mode": row["mode"],
                "workflow": row["workflow"],
                "fiber_type": row["fiber_type"],
                "run_count": row["run_count"],
                "duration_ms": row["duration_ms"],
                "dt_ms": row["dt_ms"],
                "recording": row["recording"],
                "protocol_steps": row["protocol_steps"],
                "build_peak_rss_mib": row.get("build_peak_rss_mib", ""),
                "build_rss_delta_mib": row.get("build_rss_delta_mib", ""),
                "first_run_peak_rss_mib": row.get("first_run_peak_rss_mib", ""),
                "first_run_rss_delta_mib": row.get("first_run_rss_delta_mib", ""),
                "warm_peak_rss_mib": row.get("warm_peak_rss_mib", ""),
                "warm_mean_peak_rss_mib": row.get("warm_mean_peak_rss_mib", ""),
                "warm_max_rss_delta_mib": row.get("warm_max_rss_delta_mib", ""),
                "process_peak_rss_mib": row.get("process_peak_rss_mib", ""),
                "memory_estimate_total_mib_max": fmt(estimates.get(key, float("nan"))),
            }
        )
    return out_rows


def case_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["mode"],
        row["workflow"],
        row["fiber_type"],
        row["run_count"],
        row["duration_ms"],
        row["dt_ms"],
        row["recording"],
        row["protocol_steps"],
    )


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    *,
    summary_rows: Sequence[dict[str, str]],
    event_rows: Sequence[dict[str, str]],
    memory_rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
    events: Sequence[str],
) -> list[Path]:
    pyplot = import_pyplot()
    if pyplot is None:
        return []
    outputs: list[Path] = []
    outputs.extend(plot_warm_timings(pyplot, summary_rows, out_dir, prefix))
    outputs.extend(plot_event_timings(pyplot, event_rows, out_dir, prefix, events))
    outputs.extend(plot_memory(pyplot, memory_rows, out_dir, prefix))
    outputs.extend(plot_example08_breakdown(pyplot, event_rows, out_dir, prefix, events))
    return outputs


def plot_warm_timings(pyplot: object, rows: Sequence[dict[str, str]], out_dir: Path, prefix: str) -> list[Path]:
    if not rows:
        return []
    modes = sorted({row["mode"] for row in rows})
    cases = sorted({case_plot_key(row) for row in rows})
    values = {(row["mode"], case_plot_key(row)): parse_number(row.get("warm.mean_s")) for row in rows}

    fig, ax = pyplot.subplots(figsize=(plot_width(len(cases)), 5.0), constrained_layout=True)
    width = min(0.24, 0.75 / max(1, len(modes)))
    xs = list(range(len(cases)))
    for offset, mode in enumerate(modes):
        shift = (offset - (len(modes) - 1) / 2.0) * width
        ax.bar(
            [x + shift for x in xs],
            [values.get((mode, case), float("nan")) for case in cases],
            width,
            label=mode,
        )
    ax.set_yscale("log")
    ax.set_ylabel("warm mean (s, log)")
    ax.set_title("Realistic workflow warm timings by recording mode")
    ax.set_xticks(xs, [short_case_label(case) for case in cases], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_warm_timings")


def plot_event_timings(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
    events: Sequence[str],
) -> list[Path]:
    if not rows:
        return []
    modes = sorted({row["mode"] for row in rows})
    cases = sorted({case_plot_key(row) for row in rows})
    event_names = [event for event in events if any(row["event_name"] == event for row in rows)]
    if not event_names:
        return []

    cols = 2
    rows_n = math.ceil(len(event_names) / cols)
    fig, axes = pyplot.subplots(
        rows_n,
        cols,
        figsize=(plot_width(len(cases)), max(4.0, 2.8 * rows_n)),
        constrained_layout=True,
        squeeze=False,
    )
    xs = list(range(len(cases)))
    values = {
        (row["event_name"], row["mode"], case_plot_key(row)): parse_number(row["total_ms_mean"]) / 1000.0
        for row in rows
    }
    width = min(0.22, 0.75 / max(1, len(modes)))
    for index, event_name in enumerate(event_names):
        ax = axes[index // cols][index % cols]
        for offset, mode in enumerate(modes):
            shift = (offset - (len(modes) - 1) / 2.0) * width
            ax.bar(
                [x + shift for x in xs],
                [values.get((event_name, mode, case), float("nan")) for case in cases],
                width,
                label=mode,
            )
        ax.set_yscale("log")
        ax.set_title(event_name)
        ax.set_ylabel("total per warm run (s, log)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(xs, [compact_case_label(case) for case in cases], rotation=35, ha="right")
    for index in range(len(event_names), rows_n * cols):
        axes[index // cols][index % cols].axis("off")
    axes[0][0].legend(frameon=False)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_event_timings")


def plot_memory(pyplot: object, rows: Sequence[dict[str, str]], out_dir: Path, prefix: str) -> list[Path]:
    if not rows:
        return []
    modes = sorted({row["mode"] for row in rows})
    cases = sorted({case_plot_key(row) for row in rows})
    metrics = [
        ("warm_mean_peak_rss_mib", "warm RSS peak mean (MiB)"),
        ("warm_max_rss_delta_mib", "warm max RSS delta (MiB)"),
        ("process_peak_rss_mib", "process high-water RSS (MiB)"),
        ("memory_estimate_total_mib_max", "device estimate max (MiB)"),
    ]
    fig, axes = pyplot.subplots(
        len(metrics),
        1,
        figsize=(plot_width(len(cases)), 3.1 * len(metrics)),
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    xs = list(range(len(cases)))
    width = min(0.22, 0.75 / max(1, len(modes)))
    for ax, (metric, title) in zip(axes, metrics):
        values = {
            (row["mode"], case_plot_key(row)): parse_number(row.get(metric))
            for row in rows
        }
        for offset, mode in enumerate(modes):
            shift = (offset - (len(modes) - 1) / 2.0) * width
            ax.bar(
                [x + shift for x in xs],
                [values.get((mode, case), float("nan")) for case in cases],
                width,
                label=mode,
            )
        ax.set_title(title)
        ax.set_ylabel("MiB")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(xs, [compact_case_label(case) for case in cases], rotation=35, ha="right")
        if metric == "memory_estimate_total_mib_max":
            ax.set_yscale("log")
    axes[0].legend(frameon=False)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_memory")


def plot_example08_breakdown(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
    events: Sequence[str],
) -> list[Path]:
    selected = [
        row
        for row in rows
        if row.get("workflow") == "example08_recruitment"
        and row.get("event_name") in events
        and row.get("event_name") != "simulation.pool.total"
    ]
    if not selected:
        return []
    bars = sorted({(row["mode"], row["run_count"]) for row in selected})
    event_names = [
        event
        for event in events
        if event != "simulation.pool.total" and any(row["event_name"] == event for row in selected)
    ]
    values = {
        (row["mode"], row["run_count"], row["event_name"]): parse_number(row["total_ms_mean"]) / 1000.0
        for row in selected
    }

    fig, ax = pyplot.subplots(figsize=(max(8.0, 0.9 * len(bars) + 3.0), 5.0), constrained_layout=True)
    xs = list(range(len(bars)))
    bottoms = [0.0 for _ in bars]
    for event_name in event_names:
        heights = [values.get((mode, run_count, event_name), 0.0) for mode, run_count in bars]
        ax.bar(xs, heights, bottom=bottoms, label=event_name)
        bottoms = [base + height for base, height in zip(bottoms, heights)]
    ax.set_ylabel("mean warm event total (s)")
    ax.set_title("Example08 warm profile breakdown by recording mode")
    ax.set_xticks(xs, [f"{mode}\nB={run_count}" for mode, run_count in bars])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize="small", ncols=2)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_example08_breakdown")


def case_plot_key(row: dict[str, str]) -> tuple[str, str, int, int]:
    return (
        row["workflow"],
        row["fiber_type"],
        int(float(row["run_count"])),
        int(float(row["protocol_steps"])),
    )


def short_case_label(case: tuple[str, str, int, int]) -> str:
    workflow, fiber_type, run_count, protocol_steps = case
    return (
        f"{workflow.removeprefix('example').replace('_', ' ')}\n"
        f"{fiber_type} B={run_count} P={protocol_steps}"
    )


def compact_case_label(case: tuple[str, str, int, int]) -> str:
    workflow, fiber_type, run_count, protocol_steps = case
    workflow_short = {
        "example06_velocity": "06 vel",
        "example07_threshold": "07 thr",
        "example08_recruitment": "08 rec",
    }.get(workflow, workflow)
    return f"{workflow_short}\n{fiber_type} B={run_count} P={protocol_steps}"


def import_pyplot() -> object | None:
    try:
        configure_matplotlib_cache()
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
    except Exception as exc:
        print(f"plot skipped: {exc}", flush=True)
        return None
    return pyplot


def configure_matplotlib_cache() -> None:
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "axonscope_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def save_plot(fig: object, pyplot: object, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    for output in outputs:
        fig.savefig(output, dpi=160)
    pyplot.close(fig)
    return outputs


def plot_width(row_count: int) -> float:
    return min(28.0, max(9.0, 0.78 * row_count + 3.5))


def mean(values: Iterable[str | None]) -> float:
    parsed = [parse_number(value) for value in values]
    parsed = [value for value in parsed if math.isfinite(value)]
    if not parsed:
        return float("nan")
    return sum(parsed) / len(parsed)


def max_number(values: Iterable[str | None]) -> float:
    parsed = [parse_number(value) for value in values]
    parsed = [value for value in parsed if math.isfinite(value)]
    if not parsed:
        return float("nan")
    return max(parsed)


def sum_number(values: Iterable[str | None]) -> float:
    parsed = [parse_number(value) for value in values]
    return sum(value for value in parsed if math.isfinite(value))


def parse_number(value: str | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.9g}"


if __name__ == "__main__":
    raise SystemExit(main())
