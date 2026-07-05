"""Benchmark one AxonScope simulation and plot what happened.

Run:
    MPLBACKEND=Agg python examples/advanced/runtime/04_benchmarking_options.py

This is a teaching example, not a benchmark campaign. It shows the smallest
useful workflow:

1. build a normal AxonScope simulation;
2. wrap `AxonSimulation(...).run()` with `axs.benchmark(...)`;
3. read the generated `summary.csv`, `events.jsonl`, and `memory_summary.csv`;
4. plot the time and memory spent in each recorded stage.

The default memory trace is `rss` because it is cheap enough for ordinary local
debugging. Use `--memory-trace all` only on tiny diagnostic runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Keep Matplotlib cache files with generated benchmark outputs. This avoids
# warnings on machines where the user-level Matplotlib config dir is read-only.
_MPLCONFIGDIR = Path("benchmark/results/.matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR.resolve()))

import matplotlib.pyplot as plt

import axonscope as axs


MEMORY_TRACE_HELP = {
    "off": "timing only; fastest; no measured memory fields",
    "rss": "process resident memory; cheap and useful for local CPU runs",
    "tracemalloc": "Python/NumPy-visible allocations; useful but slower",
    "device": "JAX/device memory snapshots when the backend exposes them",
    "all": "rss + tracemalloc + device; reserve for tiny trace cases",
}


@dataclass(frozen=True)
class StageRow:
    name: str
    count: int
    total_ms: float
    self_ms: float
    rss_delta_mib: float | None
    rss_end_mib: float | None
    tracemalloc_peak_mib: float | None
    device_end_mib: float | None


@dataclass(frozen=True)
class EventRow:
    name: str
    depth: int
    start_ms: float
    duration_ms: float


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    benchmark_dir = output_dir / "benchmark_run"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print_memory_trace_options(args.memory_trace)

    # Step 1: the simulation is ordinary AxonScope code. Nothing benchmark-
    # specific is needed in the model, stimulation, or recording definition.
    print("\nBuilding one small Hodgkin-Huxley simulation...")

    # Step 2: `axs.benchmark(...)` turns on instrumentation while the simulation
    # runs. AxonScope internals emit spans such as dispatch, preparation, input
    # lowering, kernel execution, and result assembly.
    print(f"Running benchmark -> {benchmark_dir}")
    with axs.benchmark(
        benchmark_dir,
        print_summary=False,
        sync_device=True,
        record_shapes=True,
        memory_trace=args.memory_trace,
        memory_top_n=args.memory_top_n,
        profile=args.profile,
        profile_backend="jax" if args.profile else "auto",
        jax_device_memory_profile=args.jax_device_memory_profile,
        jax_device_memory_profile_stages=("kernel.wait",),
    ):
        result = run_axonscope_simulation(args)

    # Step 3: use the result normally. The benchmark session is just a wrapper.
    peak_mV = float(result.single.peak_voltage_values(unit=axs.mV)[0])
    print(f"Peak center Vm: {peak_mV:.2f} mV")

    # Step 4: read benchmark artifacts and make simple diagnostic plots.
    stages = read_stage_rows(benchmark_dir)
    events = read_event_rows(benchmark_dir / "events.jsonl")
    plot_stage_timing(stages, plots_dir / "stage_timing.png")
    plot_stage_memory(stages, plots_dir / "stage_memory.png")
    plot_event_timeline(events, plots_dir / "stage_timeline.png")

    print_stage_table(stages)
    print("\nBenchmark files:")
    for filename in (
        "summary.csv",
        "memory_summary.csv",
        "events.jsonl",
        "environment.json",
    ):
        print(f"  {benchmark_dir / filename}")
    print("Plots:")
    for filename in ("stage_timing.png", "stage_memory.png", "stage_timeline.png"):
        print(f"  {plots_dir / filename}")

    if args.compare_memory_traces:
        compare_dir = output_dir / "memory_trace_comparison"
        compare_memory_trace_modes(args, compare_dir, plots_dir)

    if args.show:
        plt.show()
    else:
        plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pedagogical AxonScope benchmark instrumentation example.",
    )
    parser.add_argument(
        "--output",
        default="benchmark/results/examples/runtime_benchmarking_single_sim",
        help="Directory where benchmark CSV files and plots are written.",
    )
    parser.add_argument("--duration-ms", type=float, default=1.0)
    parser.add_argument("--dt-ms", type=float, default=0.01)
    parser.add_argument("--compartments", type=int, default=9)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument(
        "--memory-trace",
        choices=tuple(MEMORY_TRACE_HELP),
        default="rss",
        help="Memory tracing mode for the main benchmark run.",
    )
    parser.add_argument(
        "--memory-top-n",
        type=int,
        default=5,
        help="Number of top tracemalloc frames to retain when tracemalloc is enabled.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Also start a JAX profiler trace for this tiny run.",
    )
    parser.add_argument(
        "--jax-device-memory-profile",
        action="store_true",
        help="Save a JAX device-memory pprof on kernel.wait.",
    )
    parser.add_argument(
        "--compare-memory-traces",
        action="store_true",
        help="Run the same tiny simulation with off/rss/tracemalloc/all.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots.")
    return parser.parse_args()


def print_memory_trace_options(selected: str) -> None:
    print("Available memory_trace options:")
    for mode, note in MEMORY_TRACE_HELP.items():
        marker = "*" if mode == selected else " "
        print(f" {marker} {mode:12s} {note}")
    print(
        "\nProfiling is separate: --profile writes a JAX profiler trace, and "
        "--jax-device-memory-profile writes a pprof device-memory snapshot for "
        "selected stages. Keep those enabled only on tiny runs."
    )


def run_axonscope_simulation(args: argparse.Namespace):
    length = 120.0 * axs.um
    axon = axs.axons.HodgkinHuxley(
        length=length,
        diameter=0.6 * axs.um,
        compartments=args.compartments,
        celsius=6.3 * axs.degC,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_current_clamp(
        position=length / 2.0,
        current=axs.Stimulus.pulse(
            start=0.10 * axs.ms,
            duration=0.20 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )

    device = axs.Device.gpu(0) if args.platform == "gpu" else axs.Device.cpu()
    precision = (
        axs.PrecisionPolicy.float64()
        if args.precision == "fp64"
        else axs.PrecisionPolicy.float32()
    )
    policy = axs.ExecutionPolicy(
        runtime=axs.Runtime.JAX,
        device=device,
        precision=precision,
    )

    return axs.AxonSimulation(
        simulation,
        duration=args.duration_ms * axs.ms,
        dt=args.dt_ms * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        execution_policy=policy,
    ).run()


def read_stage_rows(benchmark_dir: Path) -> list[StageRow]:
    timing_rows = read_csv_by_name(benchmark_dir / "summary.csv")
    memory_rows = read_csv_by_name(benchmark_dir / "memory_summary.csv")
    stages: list[StageRow] = []
    for name, timing in timing_rows.items():
        memory = memory_rows.get(name, {})
        stages.append(
            StageRow(
                name=name,
                count=int(float(timing.get("count") or 0)),
                total_ms=float(timing.get("total_ms") or 0.0),
                self_ms=float(timing.get("self_ms") or 0.0),
                rss_delta_mib=optional_float(memory.get("rss_delta_mib_max")),
                rss_end_mib=optional_float(memory.get("rss_end_mib_max")),
                tracemalloc_peak_mib=bytes_to_mib(
                    optional_float(memory.get("tracemalloc_peak_delta_bytes_max"))
                ),
                device_end_mib=bytes_to_mib(
                    optional_float(memory.get("device_bytes_in_use_end_max"))
                ),
            )
        )
    stages.sort(key=lambda row: row.total_ms, reverse=True)
    return stages


def read_csv_by_name(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["name"]): row for row in rows}


def read_event_rows(path: Path) -> list[EventRow]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            events.append(json.loads(line))
    if not events:
        return []
    origin = min(float(event["start_ns"]) for event in events)
    rows = [
        EventRow(
            name=str(event["name"]),
            depth=int(event["depth"]),
            start_ms=(float(event["start_ns"]) - origin) / 1_000_000.0,
            duration_ms=float(event["duration_ms"]),
        )
        for event in events
    ]
    rows.sort(key=lambda row: row.start_ms)
    return rows


def plot_stage_timing(stages: list[StageRow], path: Path) -> None:
    top = stages[:10]
    labels = [short_label(row.name) for row in top]
    total_ms = [row.total_ms for row in top]
    self_ms = [row.self_ms for row in top]
    child_ms = [max(total - self_, 0.0) for total, self_ in zip(total_ms, self_ms)]

    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    y = np.arange(len(top))
    ax.barh(y, child_ms, color="#A0CBE8", label="nested spans")
    ax.barh(y, self_ms, left=child_ms, color="#4C78A8", label="self time")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("time [ms]")
    ax.set_title("Where the simulation spent time")
    ax.legend()
    fig.savefig(path, dpi=160)


def plot_stage_memory(stages: list[StageRow], path: Path) -> None:
    top = [row for row in stages if has_any_memory(row)][:10]
    if not top:
        top = stages[:10]
    labels = [short_label(row.name) for row in top]
    y = np.arange(len(top))
    width = 0.24

    rss = [row.rss_delta_mib or 0.0 for row in top]
    traced = [row.tracemalloc_peak_mib or 0.0 for row in top]
    device = [row.device_end_mib or 0.0 for row in top]

    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.barh(y - width, rss, height=width, color="#59A14F", label="RSS delta")
    ax.barh(y, traced, height=width, color="#F28E2B", label="tracemalloc peak")
    ax.barh(y + width, device, height=width, color="#E15759", label="device end")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("memory [MiB]")
    ax.set_title("Measured memory by benchmark span")
    ax.legend()
    fig.savefig(path, dpi=160)


def plot_event_timeline(events: list[EventRow], path: Path) -> None:
    top = events[:40]
    fig, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    for index, event in enumerate(top):
        ax.barh(
            index,
            event.duration_ms,
            left=event.start_ms,
            height=0.72,
            color=timeline_color(event.depth),
        )
    labels = [f"{'  ' * min(event.depth, 4)}{short_label(event.name, 34)}" for event in top]
    ax.set_yticks(np.arange(len(top)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("time since first span [ms]")
    ax.set_title("Benchmark event timeline")
    fig.savefig(path, dpi=160)


def print_stage_table(stages: list[StageRow]) -> None:
    print("\nTop benchmark stages:")
    print("  stage                         total ms    rss delta    traced peak")
    print("  ----------------------------  --------  -----------  -------------")
    for row in stages[:8]:
        print(
            f"  {row.name[:28]:28s}  "
            f"{row.total_ms:8.2f}  "
            f"{format_mib(row.rss_delta_mib):>11s}  "
            f"{format_mib(row.tracemalloc_peak_mib):>13s}"
        )


def compare_memory_trace_modes(
    args: argparse.Namespace,
    output_dir: Path,
    plots_dir: Path,
) -> None:
    print("\nComparing memory_trace overhead on the same tiny simulation...")
    rows = []
    for mode in ("off", "rss", "tracemalloc", "all"):
        case_dir = output_dir / mode
        with axs.benchmark(
            case_dir,
            print_summary=False,
            sync_device=True,
            record_shapes=True,
            memory_trace=mode,
            memory_top_n=args.memory_top_n,
        ):
            run_axonscope_simulation(args)
        stage = read_stage_rows(case_dir)[0]
        rows.append((mode, stage.total_ms, stage.rss_delta_mib, stage.tracemalloc_peak_mib))

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    labels = [row[0] for row in rows]
    total_ms = [row[1] for row in rows]
    ax.bar(labels, total_ms, color="#4C78A8")
    ax.set_ylabel("top-stage total [ms]")
    ax.set_title("Cost of memory_trace modes")
    path = plots_dir / "memory_trace_mode_cost.png"
    fig.savefig(path, dpi=160)
    print(f"Wrote: {path}")


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bytes_to_mib(value: float | None) -> float | None:
    if value is None:
        return None
    return value / float(1024**2)


def has_any_memory(row: StageRow) -> bool:
    return any(
        value is not None and value != 0.0
        for value in (
            row.rss_delta_mib,
            row.rss_end_mib,
            row.tracemalloc_peak_mib,
            row.device_end_mib,
        )
    )


def format_mib(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} MiB"


def short_label(value: str, limit: int = 36) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def timeline_color(depth: int) -> str:
    palette = ("#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#E15759")
    return palette[min(max(depth, 0), len(palette) - 1)]


if __name__ == "__main__":
    main()
