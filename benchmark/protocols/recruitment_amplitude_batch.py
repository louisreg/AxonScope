"""Benchmark native recruitment amplitude batching policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import axonscope as axs
from axonscope.benchmarking import benchmark_span


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "recruitment_amplitude_batch"
DEFAULT_AMPLITUDES_UA = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0)

INTERESTING_STAGES = (
    "benchmark.build_population",
    "benchmark.recruitment_sweep",
    "protocol.sweep.value",
    "protocol.sweep.batched_values",
    "protocol.sweep.build_amplitude_pool",
    "simulation.run_pool",
    "dispatch.build_plan",
    "runtime.prepare",
    "inputs.positions",
    "inputs.extracellular",
    "kernel.enqueue",
    "kernel.dispatch_jax",
    "kernel.wait",
    "kernel.finalize_observer",
    "results.to_public",
)


@dataclass(frozen=True, slots=True)
class BatchPolicy:
    label: str
    batch_amplitudes: bool
    amplitude_batch_size: int | None

    @property
    def expanded_rows(self) -> str:
        if not self.batch_amplitudes:
            return "sequential"
        if self.amplitude_batch_size is None:
            return "full"
        return str(self.amplitude_batch_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"), default="cpu")
    parser.add_argument("--policies", default="sequential,1,10,20,full")
    parser.add_argument("--fibers-per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration-ms", type=float, default=4.0)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--memory-trace",
        choices=("off", "rss", "tracemalloc", "device", "all"),
        default="rss",
    )
    parser.add_argument("--memory-top-n", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")
    if args.warmups < 0:
        raise SystemExit("--warmups must be >= 0.")
    if args.fibers_per_family < 1:
        raise SystemExit("--fibers-per-family must be >= 1.")

    policies = _parse_policies(args.policies)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    _write_manifest(output, args, policies)
    if args.dry_run:
        _write_cases(output, args, policies)
        print(f"dry-run: recruitment_amplitude_batch -> {output}")
        return 0

    rows: list[dict[str, Any]] = []
    reference_counts: np.ndarray | None = None
    phase_plan = [("cold", 0)]
    phase_plan.extend(("warmup", index) for index in range(args.warmups))
    phase_plan.extend(("warm", index) for index in range(args.repeats))

    for policy in policies:
        for phase, repeat in phase_plan:
            row, counts = _run_one(args, output, policy, phase=phase, repeat=repeat)
            if reference_counts is None:
                reference_counts = counts
                row["matches_reference"] = True
            else:
                row["matches_reference"] = bool(np.array_equal(counts, reference_counts))
            row["activation_counts"] = " ".join(str(int(value)) for value in counts)
            rows.append(row)
            _write_runs(output, rows)
            print(_format_progress(row))

    _write_report(output, rows)
    return 0


def _run_one(
    args: argparse.Namespace,
    output: Path,
    policy: BatchPolicy,
    *,
    phase: str,
    repeat: int,
) -> tuple[dict[str, Any], np.ndarray]:
    run_dir = output / policy.label / f"{phase}_{repeat:02d}"
    start = time.perf_counter_ns()
    failed = False
    error = ""
    counts = np.asarray([], dtype=int)
    try:
        with axs.benchmark(
            run_dir,
            print_summary=False,
            save=True,
            sync_device=True,
            record_shapes=True,
            memory_trace=args.memory_trace,
            memory_top_n=args.memory_top_n,
            profile=False,
            profile_runtime="auto",
            profile_create_perfetto=False,
            jax_device_memory_profile=False,
        ):
            with benchmark_span(
                "benchmark.build_population",
                policy=policy.label,
                phase=phase,
                repeat=repeat,
                fibers_per_family=args.fibers_per_family,
            ):
                pool, update, current_steps, criterion = _build_workload(args)
            execution_policy = _execution_policy(args.platform)
            with benchmark_span(
                "benchmark.recruitment_sweep",
                policy=policy.label,
                phase=phase,
                repeat=repeat,
                batch_amplitudes=policy.batch_amplitudes,
                amplitude_batch_size=policy.amplitude_batch_size,
            ):
                curve = axs.protocols.recruitment_sweep(
                    pool,
                    update=update,
                    values=current_steps,
                    duration=float(args.duration_ms) * axs.ms,
                    dt=float(args.dt_ms) * axs.ms,
                    criterion=criterion,
                    recording=axs.Recording.none(),
                    batch_amplitudes=policy.batch_amplitudes,
                    amplitude_batch_size=policy.amplitude_batch_size,
                    execution_policy=execution_policy,
                    progress=False,
                    solver_progress=False,
                )
            counts = np.asarray(curve.activated, dtype=bool).sum(axis=1)
    except BaseException as exc:
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end = time.perf_counter_ns()
        row = _row_from_run_dir(run_dir)
        row.update(
            {
                "policy": policy.label,
                "batch_amplitudes": policy.batch_amplitudes,
                "amplitude_batch_size": (
                    "full"
                    if policy.batch_amplitudes and policy.amplitude_batch_size is None
                    else policy.amplitude_batch_size
                ),
                "phase": phase,
                "repeat": repeat,
                "platform": args.platform,
                "fibers_per_family": args.fibers_per_family,
                "n_axons": args.fibers_per_family * 2,
                "amplitude_count": len(DEFAULT_AMPLITUDES_UA),
                "wall_ms": (end - start) / 1_000_000.0,
                "failed": failed,
                "error": error,
            }
        )
    return row, counts


def _build_workload(args: argparse.Namespace) -> tuple[
    tuple[axs.AxonInstance, ...],
    Any,
    Any,
    axs.analysis.ActivationCriterion,
]:
    rng = np.random.default_rng(int(args.seed))
    fibers_per_family = int(args.fibers_per_family)
    circle_radius = 125.0 * axs.um
    fiber_length = 1500.0 * axs.um
    stim_start = 0.20 * axs.ms
    pulse_width = 0.10 * axs.ms
    sigma = 0.3 * axs.S_per_m
    current_steps = np.asarray(DEFAULT_AMPLITUDES_UA, dtype=float) * axs.uA

    electrode = axs.analytical.PointSourceElectrode(
        x=fiber_length / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
        min_distance=5.0 * axs.um,
    )
    zero_current = axs.Stimulus.pulse(
        start=stim_start,
        duration=pulse_width,
        amplitude=0.0 * axs.uA,
    )

    radius_um = circle_radius.to(axs.um).magnitude
    unmyelinated_angles = rng.uniform(0.0, 2.0 * np.pi, fibers_per_family)
    unmyelinated_radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, fibers_per_family))
    unmyelinated_y = unmyelinated_radii * np.cos(unmyelinated_angles) * axs.um
    unmyelinated_z = unmyelinated_radii * np.sin(unmyelinated_angles) * axs.um

    myelinated_angles = rng.uniform(0.0, 2.0 * np.pi, fibers_per_family)
    myelinated_radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, fibers_per_family))
    myelinated_y = myelinated_radii * np.cos(myelinated_angles) * axs.um
    myelinated_z = myelinated_radii * np.sin(myelinated_angles) * axs.um

    unmyelinated_diameters = rng.uniform(0.4, 1.2, fibers_per_family) * axs.um
    myelinated_diameters = (
        rng.choice(np.asarray([7.3, 10.0, 12.8]), size=fibers_per_family)
        * axs.um
    )

    pool: list[axs.AxonInstance] = []
    for diameter, y, z in zip(
        unmyelinated_diameters,
        unmyelinated_y,
        unmyelinated_z,
        strict=True,
    ):
        axon = axs.axons.RattayAberham(
            length=fiber_length,
            diameter=diameter,
            compartments=61,
            celsius=37.0 * axs.degC,
        )
        extracellular = axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit=axs.um) * axs.um,
            sigma=sigma,
            stimulus=zero_current,
            axon_y=y,
            axon_z=z,
        )
        row = axs.AxonInstance(axon)
        row.add_extracellular_stimulation(stimulation=extracellular)
        pool.append(row)

    for diameter, y, z in zip(
        myelinated_diameters,
        myelinated_y,
        myelinated_z,
        strict=True,
    ):
        axon = axs.axons.MRG(
            diameter=diameter,
            nodes=4,
            length=fiber_length,
            compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
        )
        extracellular = axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit=axs.um) * axs.um,
            sigma=sigma,
            stimulus=zero_current,
            axon_y=y,
            axon_z=z,
        )
        row = axs.AxonInstance(axon)
        row.add_extracellular_stimulation(stimulation=extracellular)
        pool.append(row)

    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=stim_start,
        target=axs.positions.ALL,
    )

    def update_point_source_current(row: axs.AxonInstance, current_magnitude: Any) -> None:
        stimulation = row.extracellular_stimulation
        if stimulation is None:
            raise ValueError("simulation has no extracellular stimulation to update.")
        drive = stimulation.drives[0]
        row.add_extracellular_stimulation(
            stimulation=stimulation.replace_drive(
                drive.id,
                stimulus=axs.Stimulus.pulse(
                    start=stim_start,
                    duration=pulse_width,
                    amplitude=-current_magnitude,
                ),
            ),
            replace=True,
        )

    return tuple(pool), update_point_source_current, current_steps, criterion


def _execution_policy(platform: str) -> axs.ExecutionPolicy:
    device = axs.Device.gpu(0) if platform == "gpu" else axs.Device.cpu()
    precision = axs.PrecisionPolicy.float32() if platform == "gpu" else None
    return axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=device,
        precision=precision,
    )


def _parse_policies(value: str) -> tuple[BatchPolicy, ...]:
    policies: list[BatchPolicy] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"sequential", "seq"}:
            policies.append(BatchPolicy("sequential", False, None))
        elif token in {"full", "none"}:
            policies.append(BatchPolicy("full", True, None))
        else:
            size = int(token)
            if size < 1:
                raise SystemExit("amplitude batch sizes must be positive.")
            policies.append(BatchPolicy(str(size), True, size))
    if not policies:
        raise SystemExit("--policies selected no policies.")
    return tuple(policies)


def _row_from_run_dir(run_dir: Path) -> dict[str, Any]:
    totals = {f"{stage}_ms": "" for stage in INTERESTING_STAGES}
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return totals
    with summary_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            if name in INTERESTING_STAGES:
                totals[f"{name}_ms"] = row.get("total_ms", "")
    return totals


def _write_runs(output: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "policy",
        "batch_amplitudes",
        "amplitude_batch_size",
        "phase",
        "repeat",
        "platform",
        "fibers_per_family",
        "n_axons",
        "amplitude_count",
        "wall_ms",
        *[f"{stage}_ms" for stage in INTERESTING_STAGES],
        "matches_reference",
        "activation_counts",
        "failed",
        "error",
    ]
    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cases(
    output: Path,
    args: argparse.Namespace,
    policies: tuple[BatchPolicy, ...],
) -> None:
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("script", "preset", "platform", "policy", "n_axons"),
        )
        writer.writeheader()
        for policy in policies:
            writer.writerow(
                {
                    "script": "recruitment_amplitude_batch",
                    "preset": args.preset,
                    "platform": args.platform,
                    "policy": policy.label,
                    "n_axons": args.fibers_per_family * 2,
                }
            )


def _write_manifest(
    output: Path,
    args: argparse.Namespace,
    policies: tuple[BatchPolicy, ...],
) -> None:
    payload = {
        "script": "recruitment_amplitude_batch",
        "preset": args.preset,
        "platform": args.platform,
        "policies": [policy.label for policy in policies],
        "fibers_per_family": args.fibers_per_family,
        "n_axons": args.fibers_per_family * 2,
        "amplitudes_uA": list(DEFAULT_AMPLITUDES_UA),
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "memory_trace": args.memory_trace,
        "output": str(output),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_report(output: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Recruitment Amplitude Batch Benchmark",
        "",
        "| policy | phase | wall ms | build plan ms | build pool ms | run pool ms | dispatch_jax ms | wait ms | counts match |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {phase} | {wall} | {build_plan} | {build_pool} | {run_pool} | {dispatch} | {wait} | {match} |".format(
                policy=row["policy"],
                phase=row["phase"],
                wall=_fmt(row.get("wall_ms", "")),
                build_plan=_fmt(row.get("dispatch.build_plan_ms", "")),
                build_pool=_fmt(row.get("protocol.sweep.build_amplitude_pool_ms", "")),
                run_pool=_fmt(row.get("simulation.run_pool_ms", "")),
                dispatch=_fmt(row.get("kernel.dispatch_jax_ms", "")),
                wait=_fmt(row.get("kernel.wait_ms", "")),
                match=row.get("matches_reference", ""),
            )
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"{row['policy']} {row['phase']}#{row['repeat']}: "
        f"wall={float(row['wall_ms']):.1f} ms "
        f"build_plan={_fmt(row.get('dispatch.build_plan_ms', ''))} ms "
        f"build_pool={_fmt(row.get('protocol.sweep.build_amplitude_pool_ms', ''))} ms "
        f"wait={_fmt(row.get('kernel.wait_ms', ''))} ms"
    )


def _fmt(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
