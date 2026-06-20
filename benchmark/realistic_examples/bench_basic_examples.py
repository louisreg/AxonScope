"""Benchmark realistic workflows based on basic examples 06/07/08.

The script intentionally uses the public AxonScope examples and protocols
instead of isolated solver kernels. It is meant to answer questions such as:

- how long do velocity, threshold, and recruitment workflows take?
- how does runtime scale with run count and fiber family?
- what does CPU vs GPU look like for complete public workflows?

For CPU/GPU comparison, use ``--platforms cpu gpu``. The parent process will
spawn one child process per platform with ``JAX_PLATFORM_NAME`` set before JAX
or AxonScope are imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import psutil

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))


WORKFLOWS = ("example06_velocity", "example07_threshold", "example08_recruitment")
PLATFORMS = ("current", "cpu", "gpu")
CPU_OBSERVER_LOW_MEMORY_XLA_FLAGS = (
    "--xla_cpu_parallel_codegen_split_count=1",
    "--xla_cpu_multi_thread_eigen=false",
    "intra_op_parallelism_threads=1",
)
CPU_OBSERVER_LOW_MEMORY_ENV = {
    "OMP_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "AXONSCOPE_CPU_OBSERVER_LOW_MEMORY_XLA": "1",
}


@dataclass(frozen=True)
class TimingStats:
    repeats: int
    mean_s: float
    median_s: float
    min_s: float
    max_s: float
    std_s: float

    @classmethod
    def from_samples(cls, samples_s: Sequence[float]) -> "TimingStats":
        samples = [float(value) for value in samples_s]
        return cls(
            repeats=len(samples),
            mean_s=float(statistics.fmean(samples)),
            median_s=float(statistics.median(samples)),
            min_s=float(min(samples)),
            max_s=float(max(samples)),
            std_s=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        )


@dataclass(frozen=True)
class MemoryStats:
    rss_start_mib: float
    rss_end_mib: float
    rss_peak_mib: float
    rss_delta_mib: float
    ru_maxrss_mib: float


@dataclass(frozen=True)
class WorkflowCase:
    workflow: str
    fiber_type: str
    run_count: int
    duration_ms: float
    dt_ms: float
    recording: str
    protocol_steps: int


@dataclass(frozen=True)
class WorkflowBenchmarkRow:
    workflow: str
    fiber_type: str
    run_count: int
    platform_label: str
    jax_backend: str
    jax_devices: str
    duration_ms: float
    dt_ms: float
    recording: str
    protocol_steps: int
    build_s: float
    build_peak_rss_mib: float
    build_rss_delta_mib: float
    first_run_s: float
    first_run_peak_rss_mib: float
    first_run_rss_delta_mib: float
    total_first_s: float
    warm_peak_rss_mib: float
    warm_mean_peak_rss_mib: float
    warm_max_rss_delta_mib: float
    process_peak_rss_mib: float
    warm: TimingStats
    summary: str


@dataclass(frozen=True)
class ProfileRunSpec:
    root: Path
    platform_label: str
    jax_backend: str
    double_cable_solver_requested: str
    double_cable_solver_resolved: str


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.platforms != ["current"]:
        return spawn_platform_runs(args)
    return run_current_platform(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "standard", "stress"),
        default="smoke",
        help="Default case sizes. Explicit size flags override this.",
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        choices=WORKFLOWS,
        default=list(WORKFLOWS),
        help="Workflow families to benchmark.",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=PLATFORMS,
        default=["current"],
        help="Run in the current process, or spawn isolated CPU/GPU child processes.",
    )
    parser.add_argument(
        "--platform-label",
        default="current",
        help="Label written by a child run; normally set by --platforms.",
    )
    parser.add_argument(
        "--run-counts",
        type=int,
        nargs="+",
        default=None,
        help="Pool sizes used by examples 06 and 07.",
    )
    parser.add_argument(
        "--family-counts",
        type=int,
        nargs="+",
        default=None,
        help="Per-family population sizes used by example 08.",
    )
    parser.add_argument(
        "--fiber-types",
        nargs="+",
        choices=("hh", "rattay", "mrg", "mixed"),
        default=("hh", "rattay", "mrg", "mixed"),
        help="Fiber families to include when a workflow supports them.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Warm measured repeats.")
    parser.add_argument("--warmups", type=int, default=0, help="Warmup runs before repeats.")
    parser.add_argument(
        "--example07-max-iterations",
        type=int,
        default=None,
        help="Bisection iterations for threshold curves.",
    )
    parser.add_argument(
        "--example08-amplitude-count",
        type=int,
        default=None,
        help="Number of amplitude samples for recruitment sweeps.",
    )
    parser.add_argument(
        "--example08-recording",
        choices=("full", "center", "observer_only"),
        default="full",
        help=(
            "Recording policy for example 08 recruitment. Use center to retain a "
            "single Vm column, or observer_only to benchmark compact solver-side "
            "activation decisions instead of stored Vm."
        ),
    )
    parser.add_argument(
        "--example08-observer-cpu-chunk-size",
        type=int,
        default=0,
        help=(
            "If >0, split example 08 CPU observer-only recruitment into chunks of "
            "this many fibers to reduce CPU XLA/LLVM compile memory."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/realistic_examples"),
        help="Output directory.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write SVG/PNG timing plots next to the CSV outputs.",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record AxonScope hotpath/solver timing spans for first and measured "
            "warm runs."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Print plain progress for sweep amplitudes and dispatch groups. "
            "Useful for long Kaggle stress cases."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned cases only.")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if args.example08_observer_cpu_chunk_size < 0:
        raise ValueError("--example08-observer-cpu-chunk-size must be >= 0.")
    if any(count < 1 for count in preset_run_counts(args)):
        raise ValueError("run counts must be >= 1.")
    if any(count < 1 for count in preset_family_counts(args)):
        raise ValueError("family counts must be >= 1.")
    return args


def preset_run_counts(args: argparse.Namespace) -> list[int]:
    if args.run_counts is not None:
        return [int(value) for value in args.run_counts]
    if args.preset == "smoke":
        return [2]
    if args.preset == "stress":
        return [5, 10, 20]
    return [2, 5, 10]


def preset_family_counts(args: argparse.Namespace) -> list[int]:
    if args.family_counts is not None:
        return [int(value) for value in args.family_counts]
    if args.preset == "smoke":
        return [2]
    if args.preset == "stress":
        return [25, 50]
    return [5, 25, 50]


def example07_max_iterations(args: argparse.Namespace) -> int:
    if args.example07_max_iterations is not None:
        return int(args.example07_max_iterations)
    return 2 if args.preset == "smoke" else 20


def example08_amplitude_count(args: argparse.Namespace) -> int:
    if args.example08_amplitude_count is not None:
        return int(args.example08_amplitude_count)
    return 2 if args.preset == "smoke" else 8


def example08_observer_cpu_chunk_size(args: argparse.Namespace) -> int:
    if args.platform_label != "cpu":
        return 0
    if args.example08_recording != "observer_only":
        return 0
    return int(args.example08_observer_cpu_chunk_size)


def example08_recording_label(args: argparse.Namespace) -> str:
    chunk_size = example08_observer_cpu_chunk_size(args)
    if chunk_size > 0:
        return f"observer_only_cpu_chunk{chunk_size}"
    return str(args.example08_recording)


def spawn_platform_runs(args: argparse.Namespace) -> int:
    if "current" in args.platforms and len(args.platforms) > 1:
        raise ValueError("--platforms current cannot be combined with cpu/gpu.")
    if args.prefix is None:
        args.prefix = datetime.now().strftime("basic_examples_%Y%m%d_%H%M%S")

    for platform in args.platforms:
        env = dict(os.environ)
        if platform != "current":
            env["JAX_PLATFORM_NAME"] = platform
        if _should_use_low_memory_cpu_observer_env(args, platform):
            _apply_low_memory_cpu_observer_env(env)
        command = child_command(args, platform_label=platform)
        print("\n$", " ".join(command), flush=True)
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    if not args.dry_run and len(args.platforms) > 1:
        comparison_path = write_platform_comparison(
            out_dir=args.out_dir,
            prefix=str(args.prefix),
            platforms=args.platforms,
        )
        if comparison_path is not None:
            print(f"comparison_csv: {comparison_path}")
            if args.plots:
                for plot_path in write_comparison_plots(comparison_path):
                    print(f"plot: {plot_path}")
        if args.profile:
            profile_comparison_path = write_profile_comparison(
                out_dir=args.out_dir,
                prefix=str(args.prefix),
            )
            if profile_comparison_path is not None:
                print(f"profile_comparison_csv: {profile_comparison_path}")
    return 0


def _should_use_low_memory_cpu_observer_env(
    args: argparse.Namespace,
    platform: str,
) -> bool:
    return platform == "cpu" and args.example08_recording == "observer_only"


def _apply_low_memory_cpu_observer_env(env: dict[str, str]) -> None:
    existing = env.get("XLA_FLAGS", "").strip()
    extra = " ".join(CPU_OBSERVER_LOW_MEMORY_XLA_FLAGS)
    env["XLA_FLAGS"] = f"{existing} {extra}".strip()
    for key, value in CPU_OBSERVER_LOW_MEMORY_ENV.items():
        env.setdefault(key, value)


def child_command(args: argparse.Namespace, *, platform_label: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preset",
        args.preset,
        "--workflows",
        *args.workflows,
        "--platforms",
        "current",
        "--platform-label",
        platform_label,
        "--fiber-types",
        *args.fiber_types,
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
        "--out-dir",
        str(args.out_dir),
    ]
    if args.prefix is not None:
        command.extend(["--prefix", args.prefix])
    if args.run_counts is not None:
        command.extend(["--run-counts", *(str(value) for value in args.run_counts)])
    if args.family_counts is not None:
        command.extend(["--family-counts", *(str(value) for value in args.family_counts)])
    if args.example07_max_iterations is not None:
        command.extend(["--example07-max-iterations", str(args.example07_max_iterations)])
    if args.example08_amplitude_count is not None:
        command.extend(["--example08-amplitude-count", str(args.example08_amplitude_count)])
    command.extend(["--example08-recording", str(args.example08_recording)])
    if args.example08_observer_cpu_chunk_size:
        command.extend(
            [
                "--example08-observer-cpu-chunk-size",
                str(args.example08_observer_cpu_chunk_size),
            ]
        )
    if not args.plots:
        command.append("--no-plots")
    if args.profile:
        command.append("--profile")
    if args.progress:
        command.append("--progress")
    if args.dry_run:
        command.append("--dry-run")
    return command


def run_current_platform(args: argparse.Namespace) -> int:
    cases = planned_cases(args)
    if args.dry_run:
        for case in cases:
            print(format_case(case))
        return 0

    import jax

    configure_matplotlib_cache()
    preload_workflow_modules(args.workflows)

    prefix = args.prefix or datetime.now().strftime("basic_examples_%Y%m%d_%H%M%S")
    platform_suffix = sanitize_label(args.platform_label)
    profile_spec = profile_run_spec(
        args=args,
        prefix=prefix,
        platform_suffix=platform_suffix,
        jax_module=jax,
    )

    rows: list[WorkflowBenchmarkRow] = []
    profile_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {format_case(case)}", flush=True)
        row, case_profile_rows = benchmark_case(
            case,
            args=args,
            jax_module=jax,
            profile_spec=profile_spec,
        )
        rows.append(row)
        profile_rows.extend(case_profile_rows)
        print(
            "completed "
            f"[{index}/{len(cases)}] {case.workflow} fiber={case.fiber_type} "
            f"runs={case.run_count} first_run={row.first_run_s:.3f}s "
            f"warm_mean={row.warm.mean_s:.3f}s "
            f"peak_rss={row.process_peak_rss_mib:.1f}MiB",
            flush=True,
        )

    json_path, csv_path = write_outputs(
        rows,
        args.out_dir,
        prefix=f"{prefix}_{platform_suffix}",
        metadata={
            "benchmark": "realistic_basic_examples",
            "preset": args.preset,
            "platform_label": args.platform_label,
            "workflows": list(args.workflows),
            "run_counts": preset_run_counts(args),
            "family_counts": preset_family_counts(args),
            "example07_max_iterations": example07_max_iterations(args),
            "example08_amplitude_count": example08_amplitude_count(args),
            "example08_recording": args.example08_recording,
            "example08_observer_cpu_chunk_size": int(
                args.example08_observer_cpu_chunk_size
            ),
            "cpu_observer_low_memory_xla": (
                os.environ.get("AXONSCOPE_CPU_OBSERVER_LOW_MEMORY_XLA") == "1"
            ),
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
            "profile": bool(args.profile),
            "progress": bool(args.progress),
        },
    )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    if profile_rows:
        profile_csv = write_profile_summary(
            profile_rows,
            args.out_dir,
            prefix=f"{prefix}_{platform_suffix}_profile",
        )
        print(f"profile_csv: {profile_csv}")
    if args.plots:
        for plot_path in write_platform_timing_plots(csv_path):
            print(f"plot: {plot_path}")
    return 0


def profile_run_spec(
    *,
    args: argparse.Namespace,
    prefix: str,
    platform_suffix: str,
    jax_module: Any,
) -> ProfileRunSpec | None:
    if not args.profile:
        return None
    from axonscope.solvers import resolve_double_cable_block_solver

    backend = str(jax_module.default_backend())
    return ProfileRunSpec(
        root=args.out_dir / f"{prefix}_{platform_suffix}_profiles",
        platform_label=args.platform_label,
        jax_backend=backend,
        double_cable_solver_requested="auto",
        double_cable_solver_resolved=resolve_double_cable_block_solver(
            "auto",
            platform=backend,
        ),
    )


def preload_workflow_modules(workflows: Sequence[str]) -> None:
    """Import public workflow modules outside the measured case timings."""

    import axonscope  # noqa: F401

    if "example06_velocity" in workflows:
        from examples.basic import example_06_velocity_vs_diameter  # noqa: F401
    if "example07_threshold" in workflows:
        from examples.basic import example_07_threshold_vs_diameter  # noqa: F401
    if "example08_recruitment" in workflows:
        from examples.basic import example_08_recruitment_curve_population  # noqa: F401


def planned_cases(args: argparse.Namespace) -> list[WorkflowCase]:
    cases: list[WorkflowCase] = []
    run_counts = preset_run_counts(args)
    family_counts = preset_family_counts(args)
    fiber_types = set(args.fiber_types)

    if "example06_velocity" in args.workflows:
        for count in run_counts:
            if "hh" in fiber_types:
                cases.append(
                    WorkflowCase(
                        workflow="example06_velocity",
                        fiber_type="hh",
                        run_count=count,
                        duration_ms=10.0,
                        dt_ms=0.001,
                        recording="full",
                        protocol_steps=1,
                    )
                )
            if "mrg" in fiber_types:
                cases.append(
                    WorkflowCase(
                        workflow="example06_velocity",
                        fiber_type="mrg",
                        run_count=count,
                        duration_ms=5.0,
                        dt_ms=0.001,
                        recording="full",
                        protocol_steps=1,
                    )
                )

    if "example07_threshold" in args.workflows:
        iterations = example07_max_iterations(args)
        for count in run_counts:
            if "rattay" in fiber_types:
                cases.append(
                    WorkflowCase(
                        workflow="example07_threshold",
                        fiber_type="rattay",
                        run_count=count,
                        duration_ms=6.0,
                        dt_ms=0.01,
                        recording="probes9",
                        protocol_steps=iterations + 2,
                    )
                )
            if "mrg" in fiber_types:
                cases.append(
                    WorkflowCase(
                        workflow="example07_threshold",
                        fiber_type="mrg",
                        run_count=count,
                        duration_ms=5.0,
                        dt_ms=0.01,
                        recording="probes9",
                        protocol_steps=iterations + 2,
                    )
                )

    if "example08_recruitment" in args.workflows and "mixed" in fiber_types:
        amplitudes = example08_amplitude_count(args)
        for family_count in family_counts:
            cases.append(
                WorkflowCase(
                    workflow="example08_recruitment",
                    fiber_type="mixed",
                    run_count=2 * family_count,
                    duration_ms=4.0,
                    dt_ms=0.025,
                    recording=example08_recording_label(args),
                    protocol_steps=amplitudes,
                )
            )
    return cases


def benchmark_case(
    case: WorkflowCase,
    *,
    args: argparse.Namespace,
    jax_module: Any,
    profile_spec: ProfileRunSpec | None = None,
) -> tuple[WorkflowBenchmarkRow, list[dict[str, Any]]]:
    build_s, built, build_memory = measure_call(lambda: build_case(case, args=args))
    profile_rows: list[dict[str, Any]] = []
    first_run_s, first_summary, first_profile, first_memory = time_profiled_run(
        lambda: run_built_case(case, built, args=args),
        case=case,
        phase="first",
        repeat_index=0,
        profile_spec=profile_spec,
    )
    profile_rows.extend(first_profile)

    for _ in range(args.warmups):
        _, warm_built = time_call(lambda: build_case(case, args=args))
        time_call(lambda: run_built_case(case, warm_built, args=args))

    samples = []
    warm_memory_samples: list[MemoryStats] = []
    last_summary = first_summary
    for repeat_index in range(args.repeats):
        _, repeat_built = time_call(lambda: build_case(case, args=args))
        elapsed_s, last_summary, repeat_profile, repeat_memory = time_profiled_run(
            lambda: run_built_case(case, repeat_built, args=args),
            case=case,
            phase="warm_repeat",
            repeat_index=repeat_index + 1,
            profile_spec=profile_spec,
        )
        samples.append(elapsed_s)
        warm_memory_samples.append(repeat_memory)
        profile_rows.extend(repeat_profile)

    devices = ",".join(str(device) for device in jax_module.devices())
    warm_peak_rss_mib = max(
        (memory.rss_peak_mib for memory in warm_memory_samples),
        default=first_memory.rss_peak_mib,
    )
    warm_mean_peak_rss_mib = (
        float(statistics.fmean(memory.rss_peak_mib for memory in warm_memory_samples))
        if warm_memory_samples
        else first_memory.rss_peak_mib
    )
    warm_max_rss_delta_mib = max(
        (memory.rss_delta_mib for memory in warm_memory_samples),
        default=0.0,
    )
    return (
        WorkflowBenchmarkRow(
            workflow=case.workflow,
            fiber_type=case.fiber_type,
            run_count=case.run_count,
            platform_label=args.platform_label,
            jax_backend=str(jax_module.default_backend()),
            jax_devices=devices,
            duration_ms=case.duration_ms,
            dt_ms=case.dt_ms,
            recording=case.recording,
            protocol_steps=case.protocol_steps,
            build_s=build_s,
            build_peak_rss_mib=build_memory.rss_peak_mib,
            build_rss_delta_mib=build_memory.rss_delta_mib,
            first_run_s=first_run_s,
            first_run_peak_rss_mib=first_memory.rss_peak_mib,
            first_run_rss_delta_mib=first_memory.rss_delta_mib,
            total_first_s=build_s + first_run_s,
            warm_peak_rss_mib=warm_peak_rss_mib,
            warm_mean_peak_rss_mib=warm_mean_peak_rss_mib,
            warm_max_rss_delta_mib=warm_max_rss_delta_mib,
            process_peak_rss_mib=process_peak_rss_mib(),
            warm=TimingStats.from_samples(samples),
            summary=json.dumps(last_summary, sort_keys=True),
        ),
        profile_rows,
    )


def build_case(case: WorkflowCase, *, args: argparse.Namespace) -> Any:
    if case.workflow == "example06_velocity":
        return build_example06(case)
    if case.workflow == "example07_threshold":
        return build_example07(case, args=args)
    if case.workflow == "example08_recruitment":
        return build_example08(case, args=args)
    raise ValueError(f"unknown workflow: {case.workflow!r}")


def run_built_case(case: WorkflowCase, built: Any, *, args: argparse.Namespace) -> dict[str, Any]:
    if case.workflow == "example06_velocity":
        return run_example06(case, built)
    if case.workflow == "example07_threshold":
        return run_example07(case, built, args=args)
    if case.workflow == "example08_recruitment":
        return run_example08(case, built, args=args)
    raise ValueError(f"unknown workflow: {case.workflow!r}")


def build_example06(case: WorkflowCase) -> tuple[Any, ...]:
    import axonscope as axs
    import numpy as np
    from examples.basic import example_06_velocity_vs_diameter as ex06

    if case.fiber_type == "hh":
        simulations = []
        diameters = (
            ex06.HH_DIAMETERS[: case.run_count]
            if case.run_count <= len(ex06.HH_DIAMETERS)
            else np.linspace(0.1, 2.0, case.run_count) * axs.um
        )
        for diameter in diameters:
            axon = axs.axons.HodgkinHuxley(
                length=5000.0 * axs.um,
                diameter=diameter,
                compartments=501,
                celsius=32.0 * axs.degC,
                v_init=-67.5 * axs.mV,
                include_passive_leak=True,
                g_pas=0.001,
                e_pas=-70.0,
            )
            sim = axs.AxonInstance(axon)
            sim.add_current_clamp(
                position=0.0 * axs.um,
                current=axs.Stimulus.pulse(
                    start=ex06.CLAMP_START,
                    duration=ex06.CLAMP_DURATION,
                    amplitude=ex06.CLAMP_CURRENT,
                ),
            )
            simulations.append(sim)
        return tuple(simulations)

    if case.fiber_type == "mrg":
        simulations = []
        diameters = (
            ex06.MRG_DIAMETERS[: case.run_count]
            if case.run_count <= len(ex06.MRG_DIAMETERS)
            else np.linspace(2.0, 20.0, case.run_count) * axs.um
        )
        for diameter in diameters:
            axon = axs.axons.MRG(
                diameter=diameter,
                nodes=21,
                compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
            )
            sim = axs.AxonInstance(axon)
            sim.add_current_clamp(
                position=ex06.first_compartment_position(axon),
                current=axs.Stimulus.pulse(
                    start=ex06.CLAMP_START,
                    duration=ex06.CLAMP_DURATION,
                    amplitude=ex06.CLAMP_CURRENT,
                ),
            )
            simulations.append(sim)
        return tuple(simulations)

    raise ValueError(f"example06 does not support {case.fiber_type!r}")


def run_example06(case: WorkflowCase, pool: tuple[Any, ...]) -> dict[str, Any]:
    import axonscope as axs

    results = axs.simulate_pool(
        pool,
        duration=case.duration_ms * axs.ms,
        dt=case.dt_ms * axs.ms,
        progress=False,
    )
    speeds = [float(axs.analysis.conduction_velocity(result)) for result in results]
    return {
        "speed_mean_m_s": float(statistics.fmean(speeds)) if speeds else 0.0,
        "speed_min_m_s": float(min(speeds)) if speeds else 0.0,
        "speed_max_m_s": float(max(speeds)) if speeds else 0.0,
    }


def build_example07(case: WorkflowCase, *, args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import axonscope as axs
    import numpy as np
    from examples.basic import example_07_threshold_vs_diameter as ex07

    pulse_width = ex07.PULSE_WIDTHS[0]
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=ex07.STIM_START,
        target=axs.positions.DISTAL,
    )
    if case.fiber_type == "rattay":
        diameters = (
            ex07.RATTAY_DIAMETERS[: case.run_count]
            if case.run_count <= len(ex07.RATTAY_DIAMETERS)
            else np.linspace(0.5, 2.0, case.run_count) * axs.um
        )
        pool = tuple(
            ex07.make_rattay_simulation(diameter, pulse_width=pulse_width)
            for diameter in diameters
        )
        bounds = (20.0 * axs.uA, 250.0 * axs.uA)
    elif case.fiber_type == "mrg":
        diameters = (
            ex07.MRG_DIAMETERS[: case.run_count]
            if case.run_count <= len(ex07.MRG_DIAMETERS)
            else np.linspace(5.7, 15.0, case.run_count) * axs.um
        )
        pool = tuple(
            ex07.make_mrg_simulation(diameter, pulse_width=pulse_width)
            for diameter in diameters
        )
        bounds = (5.0 * axs.uA, 100.0 * axs.uA)
    else:
        raise ValueError(f"example07 does not support {case.fiber_type!r}")
    return pool, diameters, bounds, pulse_width, criterion


def run_example07(case: WorkflowCase, built: Any, *, args: argparse.Namespace) -> dict[str, Any]:
    import axonscope as axs
    import numpy as np
    from examples.basic import example_07_threshold_vs_diameter as ex07

    pool, diameters, bounds, pulse_width, criterion = built
    curve = axs.protocols.find_activation_threshold_curve(
        pool,
        rows=diameters,
        update=lambda sim, current, pw=pulse_width: ex07.update_point_source_current(
            sim,
            current,
            pulse_width=pw,
        ),
        bounds=bounds,
        duration=case.duration_ms * axs.ms,
        dt=case.dt_ms * axs.ms,
        criterion=criterion,
        tolerance=0.01 * axs.uA,
        relative_tolerance=0.01,
        max_iterations=example07_max_iterations(args),
        recording=axs.Recording.probes(axs.signals.Vm, count=9),
        progress=False,
    )
    finite = curve.threshold_uA[np.isfinite(curve.threshold_uA)]
    return {
        "threshold_count": int(finite.size),
        "threshold_mean_uA": float(np.mean(finite)) if finite.size else None,
        "status": ",".join(str(value) for value in curve.status),
    }


def build_example08(case: WorkflowCase, *, args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import axonscope as axs
    import numpy as np
    from examples.basic import example_08_recruitment_curve_population as ex08

    family_count = case.run_count // 2
    rng = np.random.default_rng(ex08.RNG_SEED)
    context = ex08.make_shared_context()
    unmyelinated_diameters = rng.uniform(
        ex08.UNMYELINATED_DIAMETER_RANGE_UM[0],
        ex08.UNMYELINATED_DIAMETER_RANGE_UM[1],
        family_count,
    ) * axs.um
    myelinated_diameters = rng.choice(
        ex08.MRG_DIAMETER_CHOICES_UM,
        size=family_count,
        replace=True,
    ) * axs.um
    y_unmyelinated, z_unmyelinated = ex08.random_positions_in_disk(
        family_count,
        radius=ex08.CIRCLE_RADIUS,
        rng=rng,
    )
    y_myelinated, z_myelinated = ex08.random_positions_in_disk(
        family_count,
        radius=ex08.CIRCLE_RADIUS,
        rng=rng,
    )

    simulations = []
    for diameter, y, z in zip(
        unmyelinated_diameters,
        y_unmyelinated,
        z_unmyelinated,
        strict=True,
    ):
        simulations.append(
            ex08.make_unmyelinated_simulation(
                diameter=diameter,
                y=y,
                z=z,
                context=context,
            )
        )
    for diameter, y, z in zip(
        myelinated_diameters,
        y_myelinated,
        z_myelinated,
        strict=True,
    ):
        simulations.append(
            ex08.make_myelinated_simulation(
                diameter=diameter,
                y=y,
                z=z,
                context=context,
            )
        )

    families = np.asarray(["unmyelinated"] * family_count + ["myelinated"] * family_count)
    amplitudes = ex08.CURRENT_STEPS[: example08_amplitude_count(args)]
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=ex08.STIM_START,
        target=axs.positions.ALL,
    )
    return tuple(simulations), families, amplitudes, criterion


def run_example08(
    case: WorkflowCase,
    built: Any,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import axonscope as axs
    import jax
    import numpy as np
    from examples.basic import example_08_recruitment_curve_population as ex08

    pool, families, amplitudes, criterion = built
    recording = (
        axs.Recording.none()
        if args.example08_recording == "observer_only"
        else (
            axs.Recording.center(axs.signals.Vm)
            if args.example08_recording == "center"
            else axs.Recording.voltage()
        )
    )
    _print_example08_solver_route(
        platform=str(jax.default_backend()),
        batch_size=len(pool),
        recording=str(args.example08_recording),
    )
    chunk_size = example08_observer_cpu_chunk_size(args)
    if chunk_size > 0:
        activated_chunks = []
        for start in range(0, len(pool), chunk_size):
            stop = min(start + chunk_size, len(pool))
            chunk_curve = axs.protocols.recruitment_sweep(
                pool[start:stop],
                update=ex08.update_point_source_current,
                amplitudes=amplitudes,
                duration=case.duration_ms * axs.ms,
                dt=case.dt_ms * axs.ms,
                criterion=criterion,
                recording=recording,
                progress="plain" if args.progress else False,
                solver_progress="plain" if args.progress else False,
            )
            activated_chunks.append(np.asarray(chunk_curve.activated, dtype=bool))
        activated = (
            np.concatenate(activated_chunks, axis=1)
            if activated_chunks
            else np.zeros((len(amplitudes), 0), dtype=bool)
        )
    else:
        curve = axs.protocols.recruitment_sweep(
            pool,
            update=ex08.update_point_source_current,
            amplitudes=amplitudes,
            duration=case.duration_ms * axs.ms,
            dt=case.dt_ms * axs.ms,
            criterion=criterion,
            recording=recording,
            progress="plain" if args.progress else False,
            solver_progress="plain" if args.progress else False,
        )
        activated = np.asarray(curve.activated, dtype=bool)
    fraction = np.mean(activated, axis=1) if activated.shape[1] else np.zeros(len(amplitudes))
    return {
        "final_fraction": float(fraction[-1]) if len(fraction) else 0.0,
        "amplitude_count": int(len(amplitudes)),
        "observer_cpu_chunk_size": int(chunk_size),
        "unmyelinated_final": float(np.mean(activated[-1, families == "unmyelinated"])),
        "myelinated_final": float(np.mean(activated[-1, families == "myelinated"])),
    }


def _print_example08_solver_route(
    *,
    platform: str,
    batch_size: int,
    recording: str,
) -> None:
    from axonscope.solvers import resolve_double_cable_block_solver
    from axonscope.solvers.batch_kernels import (
        _resolve_double_cable_kernel_block_solver,
        _use_batch_native_double_cable_pcr_soa_solver,
    )

    run_solver = resolve_double_cable_block_solver("auto", platform=platform)
    kernel_solver = _resolve_double_cable_kernel_block_solver(
        run_solver,
        batch_size=batch_size,
    )
    batch_native = _use_batch_native_double_cable_pcr_soa_solver(
        kernel_solver,
        batch_size=batch_size,
    )
    print(
        "example08 solver_route: "
        f"platform={platform} recording={recording} B={batch_size} "
        f"auto={run_solver} kernel={kernel_solver} "
        f"batch_native_pcr_soa={batch_native}",
        flush=True,
    )


class RssMonitor:
    """Sample process RSS while a benchmark phase runs."""

    def __init__(self, *, interval_s: float = 0.02) -> None:
        self.interval_s = float(interval_s)
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_bytes = current_rss_bytes(self._process)
        self._end_bytes = self._start_bytes
        self._peak_bytes = self._start_bytes

    def __enter__(self) -> "RssMonitor":
        self._start_bytes = current_rss_bytes(self._process)
        self._end_bytes = self._start_bytes
        self._peak_bytes = self._start_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._end_bytes = current_rss_bytes(self._process)
        self._peak_bytes = max(self._peak_bytes, self._end_bytes)

    @property
    def stats(self) -> MemoryStats:
        return MemoryStats(
            rss_start_mib=bytes_to_mib(self._start_bytes),
            rss_end_mib=bytes_to_mib(self._end_bytes),
            rss_peak_mib=bytes_to_mib(self._peak_bytes),
            rss_delta_mib=bytes_to_mib(self._end_bytes - self._start_bytes),
            ru_maxrss_mib=process_peak_rss_mib(),
        )

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._peak_bytes = max(self._peak_bytes, current_rss_bytes(self._process))


def current_rss_bytes(process: psutil.Process | None = None) -> int:
    active_process = psutil.Process(os.getpid()) if process is None else process
    return int(active_process.memory_info().rss)


def bytes_to_mib(value: int | float) -> float:
    return float(value) / float(1024**2)


def process_peak_rss_mib() -> float:
    maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return bytes_to_mib(maxrss)
    return maxrss / 1024.0


def time_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    elapsed_s, value, _ = measure_call(fn)
    return elapsed_s, value


def measure_call(fn: Callable[[], Any]) -> tuple[float, Any, MemoryStats]:
    monitor = RssMonitor()
    start = time.perf_counter()
    with monitor:
        value = fn()
        block_until_ready(value)
    return time.perf_counter() - start, value, monitor.stats


def time_profiled_run(
    fn: Callable[[], Any],
    *,
    case: WorkflowCase,
    phase: str,
    repeat_index: int,
    profile_spec: ProfileRunSpec | None,
) -> tuple[float, Any, list[dict[str, Any]], MemoryStats]:
    if profile_spec is None:
        elapsed_s, value, memory = measure_call(fn)
        return elapsed_s, value, [], memory

    import axonscope as axs

    run_dir = profile_spec.root / profile_run_label(case, phase=phase, repeat_index=repeat_index)
    metadata = profile_run_metadata(
        case,
        phase=phase,
        repeat_index=repeat_index,
        profile_spec=profile_spec,
    )
    start = time.perf_counter()
    report = None
    session_started = False
    monitor = RssMonitor()
    try:
        with monitor:
            session = axs.enable_benchmark(
                run_dir,
                print_summary=False,
                save=True,
                sync_device=True,
                record_shapes=True,
                record_memory=True,
            )
            session_started = True
            session.metadata.update(metadata)
            value = fn()
            block_until_ready(value)
    finally:
        elapsed_s = time.perf_counter() - start
        if session_started:
            report = axs.disable_benchmark(print_summary=False, save=True)
    return (
        elapsed_s,
        value,
        profile_summary_rows(
            report,
            metadata=metadata,
            elapsed_s=elapsed_s,
            profile_dir=run_dir,
        ),
        monitor.stats,
    )


def profile_run_label(case: WorkflowCase, *, phase: str, repeat_index: int) -> str:
    return sanitize_label(
        f"{case.workflow}_{case.fiber_type}_B{case.run_count}_P{case.protocol_steps}"
        f"_{phase}_{repeat_index}"
    )


def profile_run_metadata(
    case: WorkflowCase,
    *,
    phase: str,
    repeat_index: int,
    profile_spec: ProfileRunSpec,
) -> dict[str, Any]:
    return {
        "workflow": case.workflow,
        "fiber_type": case.fiber_type,
        "run_count": int(case.run_count),
        "duration_ms": float(case.duration_ms),
        "dt_ms": float(case.dt_ms),
        "recording": case.recording,
        "protocol_steps": int(case.protocol_steps),
        "phase": phase,
        "repeat_index": int(repeat_index),
        "platform_label": profile_spec.platform_label,
        "jax_backend": profile_spec.jax_backend,
        "double_cable_solver_requested": profile_spec.double_cable_solver_requested,
        "double_cable_solver_resolved": profile_spec.double_cable_solver_resolved,
    }


def profile_summary_rows(
    report: Any,
    *,
    metadata: dict[str, Any],
    elapsed_s: float,
    profile_dir: Path,
) -> list[dict[str, Any]]:
    if report is None:
        return []
    rows = []
    event_metadata = profile_event_metadata_by_name(report)
    for summary in report.summary:
        extra = event_metadata.get(summary.name, {})
        rows.append(
            {
                **metadata,
                "event_name": summary.name,
                "event_count": int(summary.count),
                "total_ms": float(summary.total_ms),
                "self_ms": float(summary.self_ms),
                "mean_ms": float(summary.mean_ms),
                "max_ms": float(summary.max_ms),
                "run_elapsed_s": float(elapsed_s),
                "profile_dir": str(profile_dir),
                "memory_estimate_total_nbytes_max": extra.get(
                    "memory_estimate_total_nbytes_max",
                    "",
                ),
                "memory_estimate_total_mib_max": extra.get(
                    "memory_estimate_total_mib_max",
                    "",
                ),
                "device_memory_capacity_bytes_max": extra.get(
                    "device_memory_capacity_bytes_max",
                    "",
                ),
                "memory_estimate_device_fraction_max": extra.get(
                    "memory_estimate_device_fraction_max",
                    "",
                ),
                "vstim_footprint_cache_hits": extra.get(
                    "vstim_footprint_cache_hits",
                    "",
                ),
                "vstim_footprint_cache_misses": extra.get(
                    "vstim_footprint_cache_misses",
                    "",
                ),
            }
        )
    return rows


def profile_event_metadata_by_name(report: Any) -> dict[str, dict[str, Any]]:
    """Aggregate selected raw event metadata into profile summary CSV columns."""

    values: dict[str, dict[str, Any]] = {}
    for event in getattr(report, "events", ()):
        name = str(getattr(event, "name", ""))
        metadata = dict(getattr(event, "metadata", {}) or {})
        row = values.setdefault(name, {})
        _max_metadata(row, "memory_estimate_total_nbytes", metadata)
        _max_metadata(row, "memory_estimate_total_mib", metadata)
        _max_metadata(row, "device_memory_capacity_bytes", metadata)
        _max_metadata(row, "memory_estimate_device_fraction", metadata)
        cache_status = metadata.get("vstim_footprint_cache")
        if cache_status == "hit":
            row["vstim_footprint_cache_hits"] = (
                int(row.get("vstim_footprint_cache_hits", 0)) + 1
            )
        elif cache_status == "miss":
            row["vstim_footprint_cache_misses"] = (
                int(row.get("vstim_footprint_cache_misses", 0)) + 1
            )
    return values


def _max_metadata(target: dict[str, Any], key: str, metadata: dict[str, Any]) -> None:
    value = metadata.get(key)
    if value in (None, ""):
        return
    out_key = f"{key}_max"
    previous = target.get(out_key)
    if previous in (None, "") or float(value) > float(previous):
        target[out_key] = value


def block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            block_until_ready(item)


def write_outputs(
    rows: Sequence[WorkflowBenchmarkRow],
    out_dir: Path,
    *,
    prefix: str,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    row_dicts = [row_to_dict(row) for row in rows]
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "results": row_dicts,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    flat_rows = [flatten_row(row) for row in row_dicts]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    return json_path, csv_path


def write_platform_comparison(
    *,
    out_dir: Path,
    prefix: str,
    platforms: Sequence[str],
) -> Path | None:
    rows_by_platform: dict[str, dict[tuple[str, str, str, str, str, str, str], dict[str, str]]] = {}
    for platform in platforms:
        csv_path = out_dir / f"{prefix}_{sanitize_label(platform)}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows_by_platform[platform] = {comparison_key(row): row for row in rows}

    if "cpu" not in rows_by_platform or "gpu" not in rows_by_platform:
        return None

    comparison_rows = []
    common_keys = sorted(set(rows_by_platform["cpu"]) & set(rows_by_platform["gpu"]))
    for key in common_keys:
        cpu = rows_by_platform["cpu"][key]
        gpu = rows_by_platform["gpu"][key]
        comparison_rows.append(
            {
                "workflow": cpu["workflow"],
                "fiber_type": cpu["fiber_type"],
                "run_count": cpu["run_count"],
                "duration_ms": cpu["duration_ms"],
                "dt_ms": cpu["dt_ms"],
                "recording": cpu["recording"],
                "protocol_steps": cpu["protocol_steps"],
                "cpu_first_run_s": cpu["first_run_s"],
                "gpu_first_run_s": gpu["first_run_s"],
                "first_run_speedup_cpu_over_gpu": speedup(cpu["first_run_s"], gpu["first_run_s"]),
                "cpu_total_first_s": cpu["total_first_s"],
                "gpu_total_first_s": gpu["total_first_s"],
                "total_first_speedup_cpu_over_gpu": speedup(
                    cpu["total_first_s"],
                    gpu["total_first_s"],
                ),
                "cpu_warm_mean_s": cpu["warm.mean_s"],
                "gpu_warm_mean_s": gpu["warm.mean_s"],
                "warm_speedup_cpu_over_gpu": speedup(cpu["warm.mean_s"], gpu["warm.mean_s"]),
                "cpu_first_run_peak_rss_mib": cpu.get("first_run_peak_rss_mib", ""),
                "gpu_first_run_peak_rss_mib": gpu.get("first_run_peak_rss_mib", ""),
                "cpu_warm_peak_rss_mib": cpu.get("warm_peak_rss_mib", ""),
                "gpu_warm_peak_rss_mib": gpu.get("warm_peak_rss_mib", ""),
                "cpu_process_peak_rss_mib": cpu.get("process_peak_rss_mib", ""),
                "gpu_process_peak_rss_mib": gpu.get("process_peak_rss_mib", ""),
                "cpu_backend": cpu["jax_backend"],
                "gpu_backend": gpu["jax_backend"],
            }
        )

    if not comparison_rows:
        return None
    comparison_path = out_dir / f"{prefix}_cpu_vs_gpu.csv"
    fieldnames = list(comparison_rows[0])
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    return comparison_path


def write_profile_summary(
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    prefix: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_profile_comparison(*, out_dir: Path, prefix: str) -> Path | None:
    cpu_path = out_dir / f"{prefix}_cpu_profile.csv"
    gpu_path = out_dir / f"{prefix}_gpu_profile.csv"
    if not cpu_path.exists() or not gpu_path.exists():
        return None
    cpu_rows = {profile_comparison_key(row): row for row in read_csv_rows(cpu_path)}
    gpu_rows = {profile_comparison_key(row): row for row in read_csv_rows(gpu_path)}

    comparison_rows = []
    for key in sorted(set(cpu_rows) & set(gpu_rows)):
        cpu = cpu_rows[key]
        gpu = gpu_rows[key]
        comparison_rows.append(
            {
                "workflow": cpu["workflow"],
                "fiber_type": cpu["fiber_type"],
                "run_count": cpu["run_count"],
                "duration_ms": cpu["duration_ms"],
                "dt_ms": cpu["dt_ms"],
                "recording": cpu["recording"],
                "protocol_steps": cpu["protocol_steps"],
                "phase": cpu["phase"],
                "repeat_index": cpu["repeat_index"],
                "event_name": cpu["event_name"],
                "cpu_total_ms": cpu["total_ms"],
                "gpu_total_ms": gpu["total_ms"],
                "total_speedup_cpu_over_gpu": speedup(cpu["total_ms"], gpu["total_ms"]),
                "cpu_self_ms": cpu["self_ms"],
                "gpu_self_ms": gpu["self_ms"],
                "self_speedup_cpu_over_gpu": speedup(cpu["self_ms"], gpu["self_ms"]),
                "cpu_event_count": cpu["event_count"],
                "gpu_event_count": gpu["event_count"],
                "cpu_profile_dir": cpu["profile_dir"],
                "gpu_profile_dir": gpu["profile_dir"],
            }
        )

    if not comparison_rows:
        return None
    comparison_path = out_dir / f"{prefix}_profile_cpu_vs_gpu.csv"
    fieldnames = list(comparison_rows[0])
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    return comparison_path


def write_platform_timing_plots(csv_path: Path) -> list[Path]:
    rows = read_csv_rows(csv_path)
    if not rows:
        return []
    pyplot = import_pyplot()
    if pyplot is None:
        return []

    labels = [plot_label(row) for row in rows]
    first_run_s = sanitize_plot_values(row["first_run_s"] for row in rows)
    warm_mean_s = sanitize_plot_values(row["warm.mean_s"] for row in rows)
    x_values = list(range(len(rows)))
    width = 0.38

    fig, ax = pyplot.subplots(
        figsize=(plot_width(len(rows)), 4.8),
        constrained_layout=True,
    )
    ax.bar([x - width / 2 for x in x_values], first_run_s, width, label="first run")
    ax.bar([x + width / 2 for x in x_values], warm_mean_s, width, label="warm mean")
    ax.set_yscale("log")
    ax.set_ylabel("seconds (log)")
    ax.set_title(f"Workflow timings: {csv_path.stem}")
    ax.set_xticks(x_values, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return save_plot(fig, pyplot, csv_path.with_name(f"{csv_path.stem}_timings"))


def write_comparison_plots(comparison_path: Path) -> list[Path]:
    rows = read_csv_rows(comparison_path)
    if not rows:
        return []
    pyplot = import_pyplot()
    if pyplot is None:
        return []

    labels = [plot_label(row) for row in rows]
    first_run_speedup = sanitize_plot_values(
        row["first_run_speedup_cpu_over_gpu"] for row in rows
    )
    warm_speedup = sanitize_plot_values(row["warm_speedup_cpu_over_gpu"] for row in rows)
    x_values = list(range(len(rows)))
    width = 0.38

    fig, ax = pyplot.subplots(
        figsize=(plot_width(len(rows)), 4.8),
        constrained_layout=True,
    )
    ax.axhline(1.0, color="0.25", linewidth=1.0, linestyle="--", label="parity")
    ax.bar(
        [x - width / 2 for x in x_values],
        first_run_speedup,
        width,
        label="first run",
    )
    ax.bar(
        [x + width / 2 for x in x_values],
        warm_speedup,
        width,
        label="warm mean",
    )
    ax.set_yscale("log")
    ax.set_ylabel("CPU/GPU speedup (log)")
    ax.set_title(f"CPU vs GPU speedup: {comparison_path.stem}")
    ax.set_xticks(x_values, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return save_plot(
        fig,
        pyplot,
        comparison_path.with_name(f"{comparison_path.stem}_speedup"),
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def import_pyplot() -> Any | None:
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


def sanitize_plot_values(values: Iterable[str]) -> list[float]:
    clean_values = []
    for value in values:
        parsed = float(value)
        if parsed > 0.0 and math.isfinite(parsed):
            clean_values.append(parsed)
        else:
            clean_values.append(float("nan"))
    return clean_values


def plot_width(row_count: int) -> float:
    return min(22.0, max(8.0, 0.75 * row_count + 2.5))


def save_plot(fig: Any, pyplot: Any, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(".svg"), stem.with_suffix(".png")]
    for output in outputs:
        fig.savefig(output, dpi=160)
    pyplot.close(fig)
    return outputs


def plot_label(row: dict[str, str]) -> str:
    workflow = row["workflow"].removeprefix("example").replace("_", " ")
    return (
        f"{workflow}\n"
        f"{row['fiber_type']} B={row['run_count']} P={row['protocol_steps']}"
    )


def comparison_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    return (
        row["workflow"],
        row["fiber_type"],
        row["run_count"],
        row["duration_ms"],
        row["dt_ms"],
        row["recording"],
        row["protocol_steps"],
    )


def profile_comparison_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    return (
        row["workflow"],
        row["fiber_type"],
        row["run_count"],
        row["duration_ms"],
        row["dt_ms"],
        row["recording"],
        row["protocol_steps"],
        row["phase"],
        row["repeat_index"],
        row["event_name"],
    )


def speedup(numerator: str, denominator: str) -> str:
    den = float(denominator)
    if den == 0.0:
        return "inf"
    return f"{float(numerator) / den:.6g}"


def row_to_dict(row: WorkflowBenchmarkRow) -> dict[str, Any]:
    return asdict(row)


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    warm = flat.pop("warm")
    for key, value in warm.items():
        flat[f"warm.{key}"] = value
    return flat


def sanitize_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def format_case(case: WorkflowCase) -> str:
    return (
        f"{case.workflow} fiber={case.fiber_type} runs={case.run_count} "
        f"duration={case.duration_ms:g}ms dt={case.dt_ms:g}ms "
        f"recording={case.recording} protocol_steps={case.protocol_steps}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
