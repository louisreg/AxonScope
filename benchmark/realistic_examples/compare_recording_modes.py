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
    ratio_rows = build_case_ratio_rows(summary_rows)
    fraction_rows = build_event_fraction_rows(event_rows)
    bottleneck_rows = build_bottleneck_rows(event_rows)

    summary_csv = out_dir / f"{prefix}_summary.csv"
    profile_csv = out_dir / f"{prefix}_profile_events.csv"
    memory_csv = out_dir / f"{prefix}_memory.csv"
    ratio_csv = out_dir / f"{prefix}_case_ratios.csv"
    fraction_csv = out_dir / f"{prefix}_event_fractions.csv"
    bottleneck_csv = out_dir / f"{prefix}_bottlenecks.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(profile_csv, event_rows)
    write_csv(memory_csv, memory_rows)
    write_csv(ratio_csv, ratio_rows)
    write_csv(fraction_csv, fraction_rows)
    write_csv(bottleneck_csv, bottleneck_rows)
    print(f"summary_csv: {summary_csv}")
    print(f"profile_events_csv: {profile_csv}")
    print(f"memory_csv: {memory_csv}")
    print(f"case_ratios_csv: {ratio_csv}")
    print(f"event_fractions_csv: {fraction_csv}")
    print(f"bottlenecks_csv: {bottleneck_csv}")

    if args.plots:
        for plot_path in write_plots(
            summary_rows=summary_rows,
            event_rows=event_rows,
            memory_rows=memory_rows,
            ratio_rows=ratio_rows,
            fraction_rows=fraction_rows,
            bottleneck_rows=bottleneck_rows,
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


def build_case_ratio_rows(summary_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in summary_rows:
        groups[case_key_without_mode(row)][row["mode"]] = row

    out_rows = []
    for key in sorted(groups):
        rows_by_mode = groups[key]
        first = next(iter(rows_by_mode.values()))
        metrics = {
            mode: parse_number(row.get("warm.mean_s"))
            for mode, row in rows_by_mode.items()
        }
        out_rows.append(
            {
                "workflow": first["workflow"],
                "fiber_type": first["fiber_type"],
                "run_count": first["run_count"],
                "duration_ms": first["duration_ms"],
                "dt_ms": first["dt_ms"],
                "protocol_steps": first["protocol_steps"],
                "gpu_full_s": fmt(metrics.get("full", float("nan"))),
                "gpu_center_s": fmt(metrics.get("center", float("nan"))),
                "gpu_observer_s": fmt(metrics.get("observer", float("nan"))),
                "cpu_full_s": fmt(metrics.get("full_cpu", float("nan"))),
                "cpu_center_s": fmt(metrics.get("center_cpu", float("nan"))),
                "cpu_observer_s": fmt(metrics.get("observer_cpu", float("nan"))),
                "cpu_full_over_gpu_full": fmt_ratio(metrics.get("full_cpu"), metrics.get("full")),
                "cpu_center_over_gpu_center": fmt_ratio(
                    metrics.get("center_cpu"), metrics.get("center")
                ),
                "cpu_observer_over_gpu_observer": fmt_ratio(
                    metrics.get("observer_cpu"), metrics.get("observer")
                ),
                "gpu_center_over_full": fmt_ratio(metrics.get("center"), metrics.get("full")),
                "gpu_observer_over_full": fmt_ratio(metrics.get("observer"), metrics.get("full")),
                "gpu_observer_over_center": fmt_ratio(
                    metrics.get("observer"), metrics.get("center")
                ),
                "cpu_center_over_full": fmt_ratio(
                    metrics.get("center_cpu"), metrics.get("full_cpu")
                ),
                "cpu_observer_over_full": fmt_ratio(
                    metrics.get("observer_cpu"), metrics.get("full_cpu")
                ),
                "cpu_observer_over_center": fmt_ratio(
                    metrics.get("observer_cpu"), metrics.get("center_cpu")
                ),
            }
        )
    return out_rows


def build_event_fraction_rows(event_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    totals: dict[tuple[str, ...], float] = {}
    for row in event_rows:
        if row["event_name"] == "simulation.pool.total":
            totals[event_case_key(row)] = parse_number(row.get("total_ms_mean"))

    out_rows = []
    for row in event_rows:
        total_ms = parse_number(row.get("total_ms_mean"))
        simulation_ms = totals.get(event_case_key(row), float("nan"))
        fraction = total_ms / simulation_ms if valid_denominator(simulation_ms) else float("nan")
        output = dict(row)
        output["total_s_mean"] = fmt(total_ms / 1000.0)
        output["fraction_of_simulation_total"] = fmt(fraction)
        output["percent_of_simulation_total"] = fmt(fraction * 100.0)
        out_rows.append(output)
    return out_rows


def build_bottleneck_rows(event_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    totals: dict[tuple[str, ...], float] = {}
    for row in event_rows:
        key = event_case_key(row)
        if row["event_name"] == "simulation.pool.total":
            totals[key] = parse_number(row.get("total_ms_mean"))
        else:
            groups[key].append(row)

    out_rows = []
    for key in sorted(groups):
        total_ms = totals.get(key, float("nan"))
        ranked = sorted(
            groups[key],
            key=lambda row: parse_number(row.get("total_ms_mean")),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            event_ms = parse_number(row.get("total_ms_mean"))
            fraction = event_ms / total_ms if valid_denominator(total_ms) else float("nan")
            out_rows.append(
                {
                    "rank": str(rank),
                    "mode": row["mode"],
                    "workflow": row["workflow"],
                    "fiber_type": row["fiber_type"],
                    "run_count": row["run_count"],
                    "duration_ms": row["duration_ms"],
                    "dt_ms": row["dt_ms"],
                    "recording": row["recording"],
                    "protocol_steps": row["protocol_steps"],
                    "event_name": row["event_name"],
                    "event_total_s_mean": fmt(event_ms / 1000.0),
                    "event_self_s_mean": fmt(parse_number(row.get("self_ms_mean")) / 1000.0),
                    "event_count_mean": row.get("event_count_mean", ""),
                    "percent_of_simulation_total": fmt(fraction * 100.0),
                    "investigation_hint": investigation_hint(row["event_name"]),
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


def case_key_without_mode(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["workflow"],
        row["fiber_type"],
        row["run_count"],
        row["duration_ms"],
        row["dt_ms"],
        row["protocol_steps"],
    )


def event_case_key(row: dict[str, str]) -> tuple[str, ...]:
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


def investigation_hint(event_name: str) -> str:
    if event_name == "runtime.prepare":
        return "cache/reuse static solver runtimes and materialized inputs"
    if event_name == "dispatch.build_plan":
        return "cache dispatch groups/probe plans for iterative protocols"
    if event_name == "kernel.wait":
        return "device execution or CPU backend solve time"
    if event_name == "kernel.enqueue":
        return "JIT launch/enqueue overhead and executable reuse"
    if event_name == "results.split_batch":
        return "result slicing/concatenation and host packaging"
    if event_name == "inputs.extracellular":
        return "Vext/stimulation materialization and footprint cache"
    if event_name == "inputs.intracellular":
        return "Iinj materialization; avoid dense zeros"
    if event_name == "results.to_public":
        return "public result conversion and host transfer"
    return "inspect span owner"


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
    ratio_rows: Sequence[dict[str, str]],
    fraction_rows: Sequence[dict[str, str]],
    bottleneck_rows: Sequence[dict[str, str]],
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
    outputs.extend(plot_workflow_totals(pyplot, summary_rows, out_dir, prefix))
    outputs.extend(plot_cpu_gpu_speedups(pyplot, ratio_rows, out_dir, prefix))
    outputs.extend(plot_recording_mode_ratios(pyplot, ratio_rows, out_dir, prefix))
    outputs.extend(plot_event_heatmaps(pyplot, event_rows, out_dir, prefix, events))
    outputs.extend(plot_event_fraction_heatmaps(pyplot, fraction_rows, out_dir, prefix, events))
    outputs.extend(plot_bottleneck_pareto(pyplot, bottleneck_rows, out_dir, prefix))
    outputs.extend(plot_memory_scatter(pyplot, summary_rows, memory_rows, out_dir, prefix))
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


def plot_workflow_totals(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    if not rows:
        return []
    workflows = sorted({row["workflow"] for row in rows})
    modes = sorted_modes(row["mode"] for row in rows)
    values: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        values[(row["mode"], row["workflow"])] += parse_number(row.get("warm.mean_s"))

    fig, ax = pyplot.subplots(figsize=(max(8.0, 1.2 * len(workflows) + 4.0), 4.8), constrained_layout=True)
    xs = list(range(len(workflows)))
    width = min(0.16, 0.75 / max(1, len(modes)))
    for offset, mode in enumerate(modes):
        shift = (offset - (len(modes) - 1) / 2.0) * width
        ax.bar(
            [x + shift for x in xs],
            [values.get((mode, workflow), float("nan")) for workflow in workflows],
            width,
            label=mode,
        )
    ax.set_yscale("log")
    ax.set_ylabel("warm total by workflow (s, log)")
    ax.set_title("Workflow totals by platform/recording mode")
    ax.set_xticks(xs, [workflow.replace("_", "\n") for workflow in workflows])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_workflow_totals")


def plot_cpu_gpu_speedups(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    if not rows:
        return []
    cases = [case_plot_key(row) for row in rows]
    metrics = [
        ("cpu_full_over_gpu_full", "full"),
        ("cpu_center_over_gpu_center", "center"),
        ("cpu_observer_over_gpu_observer", "observer"),
    ]
    fig, ax = pyplot.subplots(figsize=(plot_width(len(cases)), 5.0), constrained_layout=True)
    xs = list(range(len(cases)))
    width = 0.22
    for offset, (metric, label) in enumerate(metrics):
        shift = (offset - (len(metrics) - 1) / 2.0) * width
        ax.bar(
            [x + shift for x in xs],
            [parse_number(row.get(metric)) for row in rows],
            width,
            label=label,
        )
    ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0, label="parity")
    ax.set_yscale("log")
    ax.set_ylabel("CPU / GPU warm ratio (log)")
    ax.set_title("CPU vs GPU warm ratio by recording mode")
    ax.set_xticks(xs, [compact_case_label(case) for case in cases], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_cpu_gpu_ratios")


def plot_recording_mode_ratios(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    if not rows:
        return []
    cases = [case_plot_key(row) for row in rows]
    panels = [
        (
            "GPU recording ratios",
            [
                ("gpu_center_over_full", "center/full"),
                ("gpu_observer_over_full", "observer/full"),
                ("gpu_observer_over_center", "observer/center"),
            ],
        ),
        (
            "CPU recording ratios",
            [
                ("cpu_center_over_full", "center/full"),
                ("cpu_observer_over_full", "observer/full"),
                ("cpu_observer_over_center", "observer/center"),
            ],
        ),
    ]
    fig, axes = pyplot.subplots(2, 1, figsize=(plot_width(len(cases)), 8.0), constrained_layout=True)
    xs = list(range(len(cases)))
    width = 0.22
    for ax, (title, metrics) in zip(axes, panels):
        for offset, (metric, label) in enumerate(metrics):
            shift = (offset - (len(metrics) - 1) / 2.0) * width
            ax.bar(
                [x + shift for x in xs],
                [parse_number(row.get(metric)) for row in rows],
                width,
                label=label,
            )
        ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("warm ratio (log)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(xs, [compact_case_label(case) for case in cases], rotation=35, ha="right")
        ax.legend(frameon=False)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_recording_mode_ratios")


def plot_event_heatmaps(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
    events: Sequence[str],
) -> list[Path]:
    outputs: list[Path] = []
    for workflow in sorted({row["workflow"] for row in rows}):
        selected = [row for row in rows if row["workflow"] == workflow]
        if not selected:
            continue
        outputs.extend(
            plot_event_heatmap(
                pyplot,
                selected,
                out_dir,
                f"{prefix}_event_heatmap_{sanitize_label(workflow)}",
                events,
                title=f"{workflow}: profile span totals",
                value_key="total_ms_mean",
                transform=lambda value: math.log10(max(value / 1000.0, 1e-6)),
                colorbar_label="log10(seconds)",
            )
        )
    return outputs


def plot_event_fraction_heatmaps(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
    events: Sequence[str],
) -> list[Path]:
    outputs: list[Path] = []
    for workflow in sorted({row["workflow"] for row in rows}):
        selected = [row for row in rows if row["workflow"] == workflow]
        if not selected:
            continue
        outputs.extend(
            plot_event_heatmap(
                pyplot,
                selected,
                out_dir,
                f"{prefix}_event_fraction_heatmap_{sanitize_label(workflow)}",
                events,
                title=f"{workflow}: span share of simulation.pool.total",
                value_key="percent_of_simulation_total",
                transform=lambda value: value,
                colorbar_label="% of simulation.pool.total",
                vmin=0.0,
                vmax=100.0,
            )
        )
    return outputs


def plot_event_heatmap(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    stem: str,
    events: Sequence[str],
    *,
    title: str,
    value_key: str,
    transform: object,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[Path]:
    event_names = [event for event in events if any(row["event_name"] == event for row in rows)]
    row_keys = sorted(
        {event_case_label_key(row) for row in rows},
        key=lambda item: (mode_sort_key(item[0]), item[1], item[2], item[3]),
    )
    if not event_names or not row_keys:
        return []
    values = {
        (event_case_label_key(row), row["event_name"]): parse_number(row.get(value_key))
        for row in rows
    }
    matrix = []
    for row_key in row_keys:
        matrix_row = []
        for event_name in event_names:
            value = values.get((row_key, event_name), float("nan"))
            matrix_row.append(transform(value) if math.isfinite(value) else float("nan"))
        matrix.append(matrix_row)

    fig_height = max(4.8, 0.32 * len(row_keys) + 2.2)
    fig, ax = pyplot.subplots(figsize=(max(9.5, 0.72 * len(event_names) + 3.0), fig_height), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(event_names)), event_names, rotation=35, ha="right")
    ax.set_yticks(range(len(row_keys)), [event_case_label(row_key) for row_key in row_keys])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    return save_plot(fig, pyplot, out_dir / stem)


def plot_bottleneck_pareto(
    pyplot: object,
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    if not rows:
        return []
    modes = sorted_modes(row["mode"] for row in rows)
    event_totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        event_totals[(row["mode"], row["event_name"])] += parse_number(
            row.get("event_total_s_mean")
        )

    fig, axes = pyplot.subplots(
        len(modes),
        1,
        figsize=(9.5, max(4.0, 2.0 * len(modes))),
        constrained_layout=True,
    )
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        pairs = [
            (event_name, value)
            for (pair_mode, event_name), value in event_totals.items()
            if pair_mode == mode and value > 0.0
        ]
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:8]
        labels = [event_name for event_name, _value in reversed(pairs)]
        values = [value for _event_name, value in reversed(pairs)]
        ax.barh(range(len(labels)), values)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("summed warm event total across cases (s)")
        ax.set_title(f"Top bottlenecks: {mode}")
        ax.grid(axis="x", alpha=0.25)
    return save_plot(fig, pyplot, out_dir / f"{prefix}_bottleneck_pareto")


def plot_memory_scatter(
    pyplot: object,
    summary_rows: Sequence[dict[str, str]],
    memory_rows: Sequence[dict[str, str]],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    if not summary_rows or not memory_rows:
        return []
    warm_by_case = {
        case_key(row): parse_number(row.get("warm.mean_s"))
        for row in summary_rows
    }
    rows = []
    for row in memory_rows:
        warm_s = warm_by_case.get(case_key(row), float("nan"))
        if math.isfinite(warm_s):
            rows.append((row, warm_s))
    if not rows:
        return []

    metrics = [
        ("warm_mean_peak_rss_mib", "warm RSS peak mean (MiB)"),
        ("memory_estimate_total_mib_max", "device estimate max (MiB)"),
        ("process_peak_rss_mib", "process high-water RSS (MiB)"),
    ]
    fig, axes = pyplot.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.8), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    modes = sorted_modes(row["mode"] for row, _warm_s in rows)
    for ax, (metric, title) in zip(axes, metrics):
        for mode in modes:
            xs = [
                parse_number(row.get(metric))
                for row, _warm_s in rows
                if row["mode"] == mode
            ]
            ys = [
                warm_s
                for row, warm_s in rows
                if row["mode"] == mode
            ]
            ax.scatter(xs, ys, label=mode, s=30, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(title)
        ax.set_ylabel("warm mean (s)")
        ax.set_title(f"Warm time vs {title}")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize="small")
    return save_plot(fig, pyplot, out_dir / f"{prefix}_memory_vs_time")


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


def event_case_label_key(row: dict[str, str]) -> tuple[str, str, int, int, str]:
    return (
        row["mode"],
        row["fiber_type"],
        int(float(row["run_count"])),
        int(float(row["protocol_steps"])),
        row["recording"],
    )


def event_case_label(key: tuple[str, str, int, int, str]) -> str:
    mode, fiber_type, run_count, protocol_steps, recording = key
    return f"{mode} | {fiber_type} B={run_count} P={protocol_steps} | {recording}"


def sorted_modes(modes: Iterable[str]) -> list[str]:
    return sorted(set(modes), key=mode_sort_key)


def mode_sort_key(mode: str) -> tuple[int, str]:
    order = {
        "full_cpu": 0,
        "center_cpu": 1,
        "observer_cpu": 2,
        "full": 3,
        "center": 4,
        "observer": 5,
    }
    return (order.get(mode, 100), mode)


def sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


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


def valid_denominator(value: float) -> bool:
    return math.isfinite(value) and abs(value) > 1e-12


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.9g}"


def fmt_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None:
        return ""
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if not math.isfinite(numerator_value) or not valid_denominator(denominator_value):
        return ""
    return fmt(numerator_value / denominator_value)


if __name__ == "__main__":
    raise SystemExit(main())
