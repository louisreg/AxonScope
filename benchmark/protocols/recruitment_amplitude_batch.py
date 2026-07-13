"""Benchmark native recruitment amplitude batching policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
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
    "protocol.sweep.refresh_amplitude_pool",
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
    parser.add_argument(
        "--capture-double-cable-jit-phases",
        action="store_true",
        help="Capture trace/lower/compile/first-execution for the first production GPU JIT.",
    )
    parser.add_argument(
        "--validate-double-cable-kernel",
        action="store_true",
        help="Validate the active Triton solver against dense NumPy solves after timing.",
    )
    parser.add_argument(
        "--triton-cache-replay",
        action="store_true",
        help=(
            "Run two fresh GPU processes against one persistent Triton kernel cache "
            "and compare cache-miss/cache-hit lowering."
        ),
    )
    parser.add_argument(
        "--disable-batch-membrane-capability",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cold-only", action="store_true", help=argparse.SUPPRESS)
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
    if args.triton_cache_replay:
        return _run_triton_cache_replay(args, output)

    if args.capture_double_cable_jit_phases:
        from benchmark.analysis.jax_phase_capture import (
            install_production_double_cable_capture,
        )

        install_production_double_cable_capture(output / "double_cable_jit_phases.json")
    if args.disable_batch_membrane_capability:
        from axonscope.runtime.jax.membranes.backend import (
            GatedLeakStackMembraneBackend,
        )

        GatedLeakStackMembraneBackend.batch_cn_gate_update = None
        GatedLeakStackMembraneBackend.batch_membrane_conductance_terms = None

    rows: list[dict[str, Any]] = []
    reference_counts: np.ndarray | None = None
    phase_plan = [("cold", 0)]
    if not args.cold_only:
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
    if args.validate_double_cable_kernel:
        from benchmark.analysis.jax_triton_validation import (
            validate_double_cable_tiled_thomas,
        )

        validate_double_cable_tiled_thomas(
            output / "double_cable_kernel_validation.json"
        )
    return 0


def _run_triton_cache_replay(args: argparse.Namespace, output: Path) -> int:
    if args.platform != "gpu":
        raise SystemExit("--triton-cache-replay requires --platform gpu.")

    cache_root = output / "triton_kernel_cache"
    if cache_root.exists() and any(cache_root.iterdir()):
        raise SystemExit(
            "--triton-cache-replay requires a fresh output directory; "
            f"cache artifacts already exist under {cache_root}."
        )

    records: list[dict[str, Any]] = []
    for index, (label, expected_status) in enumerate(
        (("cache_miss", "miss"), ("cache_hit", "hit"))
    ):
        child_output = output / label
        command = _triton_cache_child_command(
            args,
            child_output,
            validate=args.validate_double_cable_kernel and index == 1,
        )
        environment = os.environ.copy()
        environment["AXONSCOPE_TRITON_KERNEL_CACHE"] = str(cache_root)
        environment["MPLCONFIGDIR"] = str(child_output / ".matplotlib")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Triton cache replay child {label!r} failed with exit code "
                f"{completed.returncode}."
            )
        record = _read_triton_cache_child(child_output, label=label)
        actual_status = record["triton_kernel_cache"]["status"]
        if actual_status != expected_status:
            raise RuntimeError(
                f"Triton cache replay expected {expected_status!r} for {label!r}, "
                f"got {actual_status!r}."
            )
        records.append(record)

    counts_match = records[0]["activation_counts"] == records[1]["activation_counts"]
    if not counts_match:
        raise RuntimeError("Triton cache replay changed recruitment activation counts.")

    miss = records[0]
    hit = records[1]
    payload = {
        "cache_root": str(cache_root),
        "activation_counts_match": counts_match,
        "lower_saved_s": miss["lower_s"] - hit["lower_s"],
        "lower_speedup": _ratio(miss["lower_s"], hit["lower_s"]),
        "total_cold_saved_s": miss["total_cold_s"] - hit["total_cold_s"],
        "total_cold_speedup": _ratio(
            miss["total_cold_s"],
            hit["total_cold_s"],
        ),
        "processes": records,
    }
    (output / "triton_cache_replay.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_triton_cache_replay_report(output, payload)
    print(
        "triton cache replay: "
        f"lower {miss['lower_s']:.3f}s -> {hit['lower_s']:.3f}s; "
        f"cold {miss['total_cold_s']:.3f}s -> {hit['total_cold_s']:.3f}s"
    )
    return 0


def _triton_cache_child_command(
    args: argparse.Namespace,
    output: Path,
    *,
    validate: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preset",
        str(args.preset),
        "--platform",
        "gpu",
        "--policies",
        "full",
        "--fibers-per-family",
        str(args.fibers_per_family),
        "--seed",
        str(args.seed),
        "--duration-ms",
        str(args.duration_ms),
        "--dt-ms",
        str(args.dt_ms),
        "--repeats",
        "1",
        "--warmups",
        "0",
        "--output",
        str(output),
        "--memory-trace",
        "off",
        "--capture-double-cable-jit-phases",
        "--cold-only",
    ]
    if args.disable_batch_membrane_capability:
        command.append("--disable-batch-membrane-capability")
    if validate:
        command.append("--validate-double-cable-kernel")
    return command


def _read_triton_cache_child(output: Path, *, label: str) -> dict[str, Any]:
    phase_path = output / "double_cable_jit_phases.json"
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    cache_event = phase.get("triton_kernel_cache")
    if not isinstance(cache_event, dict):
        raise RuntimeError(f"Triton cache replay child {label!r} recorded no event.")

    with (output / "runs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(
            f"Triton cache replay child {label!r} produced {len(rows)} runs, "
            "expected one cold run."
        )
    row = rows[0]
    return {
        "label": label,
        "activation_counts": row["activation_counts"],
        "wall_ms": float(row["wall_ms"]),
        "trace_s": float(phase["trace_s"]),
        "lower_s": float(phase["lower_s"]),
        "compile_s": float(phase["compile_s"]),
        "first_execution_s": float(phase["first_execution_s"]),
        "total_cold_s": float(phase["total_cold_s"]),
        "stablehlo_bytes": int(phase["stablehlo_bytes"]),
        "stablehlo_lines": int(phase["stablehlo_lines"]),
        "stablehlo_custom_calls": int(phase["stablehlo_custom_calls"]),
        "triton_kernel_cache": cache_event,
    }


def _write_triton_cache_replay_report(
    output: Path,
    payload: dict[str, Any],
) -> None:
    lines = [
        "# Triton Persistent Cache Replay",
        "",
        "| process | cache | trace s | lower s | compile s | first exec s | cold s | wall ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in payload["processes"]:
        lines.append(
            "| {label} | {status} | {trace_s:.4f} | {lower_s:.4f} | "
            "{compile_s:.4f} | {first_execution_s:.4f} | {total_cold_s:.4f} | "
            "{wall_ms:.1f} |".format(
                status=record["triton_kernel_cache"]["status"],
                **record,
            )
        )
    lines.extend(
        [
            "",
            f"Activation counts match: {payload['activation_counts_match']}",
            f"Lowering speedup: {payload['lower_speedup']:.3f}x",
            f"Cold-phase speedup: {payload['total_cold_speedup']:.3f}x",
        ]
    )
    (output / "triton_cache_replay.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("inf")


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
        "cold_only": args.cold_only,
        "capture_double_cable_jit_phases": args.capture_double_cable_jit_phases,
        "validate_double_cable_kernel": args.validate_double_cable_kernel,
        "triton_cache_replay": args.triton_cache_replay,
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
        "| policy | phase | wall ms | build plan ms | build pool ms | "
        "refresh pool ms | run pool ms | dispatch_jax ms | wait ms | counts match |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            (
                "| {policy} | {phase} | {wall} | {build_plan} | {build_pool} | "
                "{refresh_pool} | {run_pool} | {dispatch} | {wait} | {match} |"
            ).format(
                policy=row["policy"],
                phase=row["phase"],
                wall=_fmt(row.get("wall_ms", "")),
                build_plan=_fmt(row.get("dispatch.build_plan_ms", "")),
                build_pool=_fmt(row.get("protocol.sweep.build_amplitude_pool_ms", "")),
                refresh_pool=_fmt(
                    row.get("protocol.sweep.refresh_amplitude_pool_ms", "")
                ),
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
        "refresh_pool="
        f"{_fmt(row.get('protocol.sweep.refresh_amplitude_pool_ms', ''))} ms "
        f"wait={_fmt(row.get('kernel.wait_ms', ''))} ms"
    )


def _fmt(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
