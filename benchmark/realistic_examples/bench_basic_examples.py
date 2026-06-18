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
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))


WORKFLOWS = ("example06_velocity", "example07_threshold", "example08_recruitment")
PLATFORMS = ("current", "cpu", "gpu")


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
    first_run_s: float
    total_first_s: float
    warm: TimingStats
    summary: str


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.platforms != ["current"]:
        return spawn_platform_runs(args)
    return run_current_platform(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "standard"),
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
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/realistic_examples"),
        help="Output directory.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned cases only.")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if any(count < 1 for count in preset_run_counts(args)):
        raise ValueError("run counts must be >= 1.")
    if any(count < 1 for count in preset_family_counts(args)):
        raise ValueError("family counts must be >= 1.")
    return args


def preset_run_counts(args: argparse.Namespace) -> list[int]:
    if args.run_counts is not None:
        return [int(value) for value in args.run_counts]
    return [2] if args.preset == "smoke" else [2, 5, 10]


def preset_family_counts(args: argparse.Namespace) -> list[int]:
    if args.family_counts is not None:
        return [int(value) for value in args.family_counts]
    return [2] if args.preset == "smoke" else [5, 25, 50]


def example07_max_iterations(args: argparse.Namespace) -> int:
    if args.example07_max_iterations is not None:
        return int(args.example07_max_iterations)
    return 2 if args.preset == "smoke" else 20


def example08_amplitude_count(args: argparse.Namespace) -> int:
    if args.example08_amplitude_count is not None:
        return int(args.example08_amplitude_count)
    return 2 if args.preset == "smoke" else 8


def spawn_platform_runs(args: argparse.Namespace) -> int:
    if "current" in args.platforms and len(args.platforms) > 1:
        raise ValueError("--platforms current cannot be combined with cpu/gpu.")

    for platform in args.platforms:
        env = dict(os.environ)
        if platform != "current":
            env["JAX_PLATFORM_NAME"] = platform
        command = child_command(args, platform_label=platform)
        print("\n$", " ".join(command), flush=True)
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


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

    rows: list[WorkflowBenchmarkRow] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {format_case(case)}", flush=True)
        rows.append(benchmark_case(case, args=args, jax_module=jax))

    prefix = args.prefix or datetime.now().strftime("basic_examples_%Y%m%d_%H%M%S")
    platform_suffix = sanitize_label(args.platform_label)
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
            "repeats": int(args.repeats),
            "warmups": int(args.warmups),
        },
    )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    return 0


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
                    recording="full",
                    protocol_steps=amplitudes,
                )
            )
    return cases


def benchmark_case(
    case: WorkflowCase,
    *,
    args: argparse.Namespace,
    jax_module: Any,
) -> WorkflowBenchmarkRow:
    build_s, built = time_call(lambda: build_case(case, args=args))
    first_run_s, first_summary = time_call(lambda: run_built_case(case, built, args=args))

    for _ in range(args.warmups):
        _, warm_built = time_call(lambda: build_case(case, args=args))
        time_call(lambda: run_built_case(case, warm_built, args=args))

    samples = []
    last_summary = first_summary
    for _ in range(args.repeats):
        _, repeat_built = time_call(lambda: build_case(case, args=args))
        elapsed_s, last_summary = time_call(
            lambda: run_built_case(case, repeat_built, args=args)
        )
        samples.append(elapsed_s)

    devices = ",".join(str(device) for device in jax_module.devices())
    return WorkflowBenchmarkRow(
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
        first_run_s=first_run_s,
        total_first_s=build_s + first_run_s,
        warm=TimingStats.from_samples(samples),
        summary=json.dumps(last_summary, sort_keys=True),
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
        return run_example08(case, built)
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


def run_example08(case: WorkflowCase, built: Any) -> dict[str, Any]:
    import axonscope as axs
    import numpy as np
    from examples.basic import example_08_recruitment_curve_population as ex08

    pool, families, amplitudes, criterion = built
    curve = axs.protocols.recruitment_sweep(
        pool,
        update=ex08.update_point_source_current,
        amplitudes=amplitudes,
        duration=case.duration_ms * axs.ms,
        dt=case.dt_ms * axs.ms,
        criterion=criterion,
        recording=axs.Recording.voltage(),
        progress=False,
    )
    return {
        "final_fraction": float(curve.fraction[-1]) if len(curve.fraction) else 0.0,
        "amplitude_count": int(len(amplitudes)),
        "unmyelinated_final": float(np.mean(curve.activated[-1, families == "unmyelinated"])),
        "myelinated_final": float(np.mean(curve.activated[-1, families == "myelinated"])),
    }


def time_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    value = fn()
    block_until_ready(value)
    return time.perf_counter() - start, value


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
