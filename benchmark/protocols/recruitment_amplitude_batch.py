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
from benchmark.analysis.run_pool_detail import write_run_pool_detail
from axonscope.benchmarking import benchmark_span, record_benchmark_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "recruitment_amplitude_batch"
DEFAULT_AMPLITUDES_UA = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0)
P14_REALISTIC_AMPLITUDES_UA = tuple(float(value) for value in np.linspace(0.0, 300.0, 21))
P14_MRG_DIAMETERS_UM = (7.3, 10.0, 12.8)

INTERESTING_STAGES = (
    "benchmark.build_population",
    "benchmark.recruitment_sweep",
    "protocol.sweep.amplitude_chunk",
    "simulation.run_pool",
    "dispatch.build_plan",
    "runtime.prepare",
    "runtime.prepare.materialize_axons",
    "runtime.prepare.base_runtime",
    "runtime.prepare.stack_cable",
    "runtime.prepare.stack_membrane",
    "runtime.prepare.membrane_vm0_rows",
    "runtime.prepare.membrane_encode_rows",
    "runtime.prepare.membrane_encode_unique_rows",
    "runtime.prepare.membrane_gather_rows",
    "runtime.prepare.membrane_initial_gates",
    "runtime.prepare.membrane_device_arrays",
    "runtime.prepare.stack_extracellular",
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


RecruitmentWorkload = tuple[
    tuple[axs.AxonInstance, ...],
    Any,
    Any,
    axs.analysis.ActivationCriterion,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"), default="cpu")
    parser.add_argument(
        "--workload",
        choices=("legacy", "p14_realistic"),
        default="legacy",
    )
    parser.add_argument(
        "--cable",
        choices=("single", "double", "mixed"),
        default="mixed",
    )
    parser.add_argument(
        "--drive-count",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Use one variable extracellular drive or one variable plus one "
            "independent static drive."
        ),
    )
    parser.add_argument("--policies", default="sequential,1,10,20,full")
    parser.add_argument("--fibers-per-family", type=int, default=100)
    parser.add_argument(
        "--axon-count",
        type=int,
        help="Total axon count; defaults to 196 for p14_realistic.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mrg-template-count",
        type=int,
        default=3,
        help=(
            "Requested number of distinct MRG diameter/node-shift templates. "
            "Counts above 3 retain the same three diameters and add intrinsic "
            "node shifts."
        ),
    )
    parser.add_argument(
        "--axon-template-policy",
        choices=("shared", "distinct"),
        default="shared",
        help=(
            "Population-construction A/B control. Production-like runs share exact "
            "axon templates; 'distinct' preserves the former per-row construction."
        ),
    )
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--dt-ms", type=float)
    parser.add_argument(
        "--amplitudes-uA",
        help="Comma-separated recruitment amplitudes in microamperes.",
    )
    parser.add_argument("--time-chunk-steps", type=int, default=128)
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
        "--capture-jit-phases",
        action="store_true",
        help=(
            "Capture trace/lower/compile/first-execution and compiler IR for "
            "the first production JIT of each selected cable."
        ),
    )
    parser.add_argument(
        "--validate-double-cable-kernel",
        action="store_true",
        help="Validate the active Triton solver against dense NumPy solves after timing.",
    )
    parser.add_argument(
        "--compilation-cache-replay",
        action="store_true",
        help=(
            "Run two fresh processes against shared persistent JAX/XLA and "
            "Triton caches and compare cold compilation replay."
        ),
    )
    parser.add_argument(
        "--disable-batch-membrane-capability",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cold-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-scope",
        choices=("run", "sweep", "run_pool"),
        default="run",
        help="Profile the complete case, recruitment_sweep, or simulation.run_pool.",
    )
    parser.add_argument("--profile-create-perfetto", action="store_true")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_workload_args(args: argparse.Namespace) -> None:
    realistic = args.workload == "p14_realistic"
    if args.duration_ms is None:
        args.duration_ms = 3.0 if realistic else 4.0
    if args.dt_ms is None:
        args.dt_ms = 0.001 if realistic else 0.025
    if args.axon_count is None:
        if realistic:
            args.axon_count = 196
        elif args.cable == "mixed":
            args.axon_count = 2 * int(args.fibers_per_family)
        else:
            args.axon_count = int(args.fibers_per_family)
    args.amplitudes = _parse_amplitudes(
        args.amplitudes_uA,
        default=(P14_REALISTIC_AMPLITUDES_UA if realistic else DEFAULT_AMPLITUDES_UA),
    )


def _parse_amplitudes(
    value: str | None,
    *,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if value is None:
        return default
    amplitudes = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not amplitudes:
        raise SystemExit("--amplitudes-uA selected no amplitudes.")
    return amplitudes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_workload_args(args)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")
    if args.warmups < 0:
        raise SystemExit("--warmups must be >= 0.")
    if args.fibers_per_family < 1:
        raise SystemExit("--fibers-per-family must be >= 1.")
    if args.axon_count < 1:
        raise SystemExit("--axon-count must be >= 1.")
    if args.time_chunk_steps < 1:
        raise SystemExit("--time-chunk-steps must be >= 1.")
    if args.mrg_template_count < 1:
        raise SystemExit("--mrg-template-count must be >= 1.")
    if args.duration_ms <= 0.0:
        raise SystemExit("--duration-ms must be > 0.")
    if args.dt_ms <= 0.0:
        raise SystemExit("--dt-ms must be > 0.")

    policies = _parse_policies(args.policies)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    _write_manifest(output, args, policies)
    if args.dry_run:
        _write_cases(output, args, policies)
        print(f"dry-run: recruitment_amplitude_batch -> {output}")
        return 0
    if args.compilation_cache_replay:
        return _run_compilation_cache_replay(args, output)

    if args.capture_jit_phases:
        from benchmark.analysis.jax_phase_capture import (
            install_production_jax_captures,
        )

        cables = ("single", "double") if args.cable == "mixed" else (args.cable,)
        install_production_jax_captures(
            output / "jax_phase_capture",
            cables=cables,
            platform=args.platform,
        )
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
        workload, source_population_build_ms = _build_source_workload(
            args,
            output,
            policy,
        )
        for phase, repeat in phase_plan:
            row, counts = _run_one(
                args,
                output,
                policy,
                workload=workload,
                source_population_build_ms=source_population_build_ms,
                phase=phase,
                repeat=repeat,
            )
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


def _run_compilation_cache_replay(args: argparse.Namespace, output: Path) -> int:
    if args.cable == "mixed":
        raise SystemExit("--compilation-cache-replay requires one cable route.")

    xla_cache_root = output / "jax_xla_cache"
    triton_cache_root = output / "triton_kernel_cache"
    cache_roots = (xla_cache_root, triton_cache_root)
    if any(root.exists() and any(root.iterdir()) for root in cache_roots):
        raise SystemExit(
            "--compilation-cache-replay requires a fresh output directory."
        )

    records: list[dict[str, Any]] = []
    for label in ("cache_miss", "cache_replay", "dynamic_values_replay"):
        child_output = output / label
        command = _compilation_cache_child_command(
            args,
            child_output,
            dynamic_values=label == "dynamic_values_replay",
        )
        before_xla = _cache_tree_snapshot(xla_cache_root)
        before_triton = _cache_tree_snapshot(triton_cache_root)
        environment = os.environ.copy()
        environment["AXONSCOPE_JAX_COMPILATION_CACHE"] = str(xla_cache_root)
        environment["AXONSCOPE_JAX_CACHE_MIN_COMPILE_TIME_S"] = "0"
        environment["AXONSCOPE_JAX_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
        environment["AXONSCOPE_JAX_PERSISTENT_XLA_CACHES"] = "all"
        environment["AXONSCOPE_TRITON_KERNEL_CACHE"] = str(triton_cache_root)
        environment["JAX_EXPLAIN_CACHE_MISSES"] = "true"
        environment["MPLCONFIGDIR"] = str(child_output / ".matplotlib")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Compilation cache replay child {label!r} failed with exit code "
                f"{completed.returncode}."
            )
        record = _read_compilation_cache_child(
            child_output,
            label=label,
            cable=args.cable,
        )
        record["jax_xla_cache"] = _cache_tree_delta(
            before_xla,
            _cache_tree_snapshot(xla_cache_root),
        )
        record["triton_cache"] = _cache_tree_delta(
            before_triton,
            _cache_tree_snapshot(triton_cache_root),
        )
        records.append(record)

    if args.validate_double_cable_kernel:
        from benchmark.analysis.jax_triton_validation import (
            validate_double_cable_tiled_thomas,
        )

        validate_double_cable_tiled_thomas(
            output / "double_cable_kernel_validation.json"
        )

    counts_match = records[0]["activation_counts"] == records[1]["activation_counts"]
    if not counts_match:
        raise RuntimeError("Compilation cache replay changed activation counts.")

    miss = records[0]
    hit = records[1]
    dynamic = records[2]
    stablehlo_match = miss["stablehlo_sha256"] == hit["stablehlo_sha256"]
    if not stablehlo_match:
        raise RuntimeError("Compilation replay produced different StableHLO programs.")
    dynamic_stablehlo_match = (
        miss["stablehlo_sha256"] == dynamic["stablehlo_sha256"]
    )
    if not dynamic_stablehlo_match:
        raise RuntimeError(
            "Same-shape dynamic values produced a different StableHLO program."
        )
    payload = {
        "jax_xla_cache_root": str(xla_cache_root),
        "triton_cache_root": str(triton_cache_root),
        "activation_counts_match": counts_match,
        "stablehlo_match": stablehlo_match,
        "dynamic_stablehlo_match": dynamic_stablehlo_match,
        "dynamic_activation_counts": dynamic["activation_counts"],
        "lower_saved_s": miss["lower_s"] - hit["lower_s"],
        "lower_speedup": _ratio(miss["lower_s"], hit["lower_s"]),
        "total_cold_saved_s": miss["total_cold_s"] - hit["total_cold_s"],
        "total_cold_speedup": _ratio(
            miss["total_cold_s"],
            hit["total_cold_s"],
        ),
        "processes": records,
    }
    (output / "compilation_cache_replay.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_compilation_cache_replay_report(output, payload)
    print(
        "compilation cache replay: "
        f"lower {miss['lower_s']:.3f}s -> {hit['lower_s']:.3f}s; "
        f"cold {miss['total_cold_s']:.3f}s -> {hit['total_cold_s']:.3f}s"
    )
    return 0


def _compilation_cache_child_command(
    args: argparse.Namespace,
    output: Path,
    *,
    dynamic_values: bool,
) -> list[str]:
    seed = int(args.seed) + (1 if dynamic_values else 0)
    amplitudes = (
        tuple(float(value) * 0.8 for value in args.amplitudes)
        if dynamic_values
        else args.amplitudes
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preset",
        str(args.preset),
        "--platform",
        str(args.platform),
        "--workload",
        str(args.workload),
        "--cable",
        str(args.cable),
        "--drive-count",
        str(args.drive_count),
        "--policies",
        "full",
        "--fibers-per-family",
        str(args.fibers_per_family),
        "--axon-count",
        str(args.axon_count),
        "--seed",
        str(seed),
        "--mrg-template-count",
        str(args.mrg_template_count),
        "--axon-template-policy",
        str(args.axon_template_policy),
        "--duration-ms",
        str(args.duration_ms),
        "--dt-ms",
        str(args.dt_ms),
        "--amplitudes-uA",
        ",".join(str(value) for value in amplitudes),
        "--time-chunk-steps",
        str(args.time_chunk_steps),
        "--repeats",
        "1",
        "--warmups",
        "0",
        "--output",
        str(output),
        "--memory-trace",
        "off",
        "--capture-jit-phases",
        "--cold-only",
    ]
    if args.disable_batch_membrane_capability:
        command.append("--disable-batch-membrane-capability")
    return command


def _read_compilation_cache_child(
    output: Path,
    *,
    label: str,
    cable: str,
) -> dict[str, Any]:
    phase_path = output / "jax_phase_capture" / f"{cable}.jit_phases.json"
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    cache_event = phase.get("triton_kernel_cache")

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
        "stablehlo_bytes": int(phase["stablehlo"]["bytes"]),
        "stablehlo_lines": int(phase["stablehlo"]["lines"]),
        "stablehlo_custom_calls": int(phase["stablehlo"]["custom_calls"]),
        "stablehlo_sha256": str(phase["stablehlo"]["sha256"]),
        "triton_kernel_cache": cache_event,
    }


def _write_compilation_cache_replay_report(
    output: Path,
    payload: dict[str, Any],
) -> None:
    lines = [
        "# JAX/XLA And Triton Compilation Cache Replay",
        "",
        "| process | Triton | trace s | lower s | compile s | first exec s | cold s | wall ms | new XLA files |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in payload["processes"]:
        lines.append(
            "| {label} | {status} | {trace_s:.4f} | {lower_s:.4f} | "
            "{compile_s:.4f} | {first_execution_s:.4f} | {total_cold_s:.4f} | "
            "{wall_ms:.1f} | {new_xla_files} |".format(
                status=(record["triton_kernel_cache"] or {}).get("status", "n/a"),
                new_xla_files=record["jax_xla_cache"]["new_file_count"],
                **record,
            )
        )
    lines.extend(
        [
            "",
            f"Activation counts match: {payload['activation_counts_match']}",
            f"StableHLO matches: {payload['stablehlo_match']}",
            f"Dynamic-value StableHLO matches: {payload['dynamic_stablehlo_match']}",
            f"Dynamic-value activations: {payload['dynamic_activation_counts']}",
            f"Lowering speedup: {payload['lower_speedup']:.3f}x",
            f"Cold-phase speedup: {payload['total_cold_speedup']:.3f}x",
        ]
    )
    (output / "compilation_cache_replay.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _cache_tree_snapshot(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


def _cache_tree_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, Any]:
    new_files = sorted(set(after) - set(before))
    changed_files = sorted(
        path for path in set(after) & set(before) if after[path] != before[path]
    )
    return {
        "file_count": len(after),
        "bytes": sum(after.values()),
        "new_file_count": len(new_files),
        "new_bytes": sum(after[path] for path in new_files),
        "changed_file_count": len(changed_files),
        "new_files": new_files,
        "changed_files": changed_files,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("inf")


def _run_one(
    args: argparse.Namespace,
    output: Path,
    policy: BatchPolicy,
    *,
    workload: RecruitmentWorkload,
    source_population_build_ms: float,
    phase: str,
    repeat: int,
) -> tuple[dict[str, Any], np.ndarray]:
    run_dir = output / policy.label / f"{phase}_{repeat:02d}"
    start = time.perf_counter_ns()
    failed = False
    error = ""
    counts = np.asarray([], dtype=int)
    pool, update, current_steps, criterion = workload
    n_axons = len(pool)
    try:
        with axs.benchmark(
            run_dir,
            print_summary=False,
            save=True,
            sync_device=True,
            record_shapes=True,
            memory_trace=args.memory_trace,
            memory_top_n=args.memory_top_n,
            profile=bool(args.profile),
            profile_runtime="jax" if args.profile else "auto",
            profile_span={
                "sweep": "benchmark.recruitment_sweep",
                "run_pool": "simulation.run_pool",
            }.get(args.profile_scope),
            profile_output=(
                run_dir / "profiles" / args.profile_scope if args.profile else None
            ),
            profile_create_perfetto=bool(args.profile_create_perfetto),
            jax_device_memory_profile=False,
        ):
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
                    batch_options=axs.BatchOptions.none(
                        time_chunk_steps=int(args.time_chunk_steps)
                    ),
                    batch_amplitudes=policy.batch_amplitudes,
                    amplitude_batch_size=policy.amplitude_batch_size,
                    execution_policy=execution_policy,
                    progress=False,
                    solver_progress=False,
                )
            counts = np.asarray(curve.activated, dtype=bool).sum(axis=1)
        if args.drive_count == 2:
            _validate_multi_drive_routes(
                run_dir,
                cable=args.cable,
                platform=args.platform,
            )
        _validate_compact_activation_route(run_dir, cable=args.cable)
        write_run_pool_detail(
            run_dir,
            amplitudes=args.amplitudes,
        )
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
                "workload": args.workload,
                "cable": args.cable,
                "drive_count": args.drive_count,
                "n_axons": n_axons or args.axon_count,
                "amplitude_count": len(args.amplitudes),
                "wall_ms": (end - start) / 1_000_000.0,
                "source_population_build_ms": source_population_build_ms,
                "one_shot_wall_ms": (
                    (end - start) / 1_000_000.0 + source_population_build_ms
                    if phase == "cold"
                    else ""
                ),
                "failed": failed,
                "error": error,
            }
        )
    return row, counts


def _build_source_workload(
    args: argparse.Namespace,
    output: Path,
    policy: BatchPolicy,
) -> tuple[RecruitmentWorkload, float]:
    """Build and profile the immutable source workload once per batch policy."""

    run_dir = output / policy.label / "source_population"
    start = time.perf_counter_ns()
    with axs.benchmark(
        run_dir,
        print_summary=False,
        save=True,
        sync_device=False,
        record_shapes=True,
        memory_trace=args.memory_trace,
        memory_top_n=args.memory_top_n,
        profile=False,
        jax_device_memory_profile=False,
    ):
        with benchmark_span(
            "benchmark.build_population",
            policy=policy.label,
            fibers_per_family=args.fibers_per_family,
        ):
            workload = _build_workload(args)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    source_row = _row_from_run_dir(run_dir)
    measured_ms = source_row.get("benchmark.build_population_ms", "")
    if measured_ms not in {"", None}:
        elapsed_ms = float(measured_ms)
    return workload, elapsed_ms


def _validate_multi_drive_routes(
    run_dir: Path,
    *,
    cable: str,
    platform: str,
) -> None:
    """Reject dense or non-production routes in the multi-drive benchmark."""

    events_path = run_dir / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_modes = (
        {"single", "double"}
        if cable == "mixed"
        else {str(cable)}
    )
    extracellular_by_mode: dict[str, dict[str, Any]] = {}
    numeric_axis_by_mode: dict[str, dict[str, Any]] = {}
    groups_by_mode: dict[str, dict[str, Any]] = {}
    for event in events:
        metadata = event.get("metadata", {})
        if event.get("name") == "inputs.extracellular":
            capability = str(metadata.get("extracellular_capability_cable", ""))
            mode = capability.removesuffix("-cable")
            if mode in expected_modes:
                extracellular_by_mode[mode] = metadata
        elif event.get("name") == "inputs.numeric_axis":
            mode = str(metadata.get("mode", ""))
            if mode in expected_modes:
                numeric_axis_by_mode[mode] = metadata
        elif event.get("name") == "dispatch.group.total":
            mode = str(metadata.get("mode", ""))
            if mode in expected_modes:
                groups_by_mode[mode] = metadata

    if set(extracellular_by_mode) != expected_modes:
        raise RuntimeError(
            "multi-drive validation did not observe every expected extracellular "
            f"group: got {sorted(extracellular_by_mode)}, "
            f"expected {sorted(expected_modes)}."
        )
    if set(groups_by_mode) != expected_modes:
        raise RuntimeError(
            "multi-drive validation did not observe every expected dispatch group: "
            f"got {sorted(groups_by_mode)}, expected {sorted(expected_modes)}."
        )
    if set(numeric_axis_by_mode) != expected_modes:
        raise RuntimeError(
            "multi-drive validation did not observe every numeric-axis lowering: "
            f"got {sorted(numeric_axis_by_mode)}, expected {sorted(expected_modes)}."
        )

    for mode, metadata in extracellular_by_mode.items():
        if metadata.get("extracellular_format") != "factorized_footprint":
            raise RuntimeError(f"{mode} multi-drive execution used a dense input route.")
        if metadata.get("dense_vstim_avoided") is not True:
            raise RuntimeError(f"{mode} multi-drive execution materialized dense Vext.")
        if int(metadata.get("nstim", 0)) != 2:
            raise RuntimeError(f"{mode} multi-drive execution did not retain both drives.")

    for mode, metadata in numeric_axis_by_mode.items():
        if int(metadata.get("extracellular_waveform_drive_count", 0)) != 2:
            raise RuntimeError(f"{mode} numeric axis did not lower two drives.")
    for mode, metadata in groups_by_mode.items():
        if (
            metadata.get("prepared_input_contract_extracellular_format")
            != "factorized_footprint"
        ):
            raise RuntimeError(f"{mode} prepared a non-factorized input contract.")

    if platform == "gpu" and "double" in expected_modes:
        block_solver = groups_by_mode["double"].get(
            "execution_policy_double_cable_block_solver"
        )
        if block_solver != "jax_triton_loop_xb":
            raise RuntimeError(
                "double-cable GPU multi-drive execution did not use the production "
                f"Triton route: got {block_solver!r}."
            )

    payload = {
        "expected_modes": sorted(expected_modes),
        "factorized_modes": sorted(extracellular_by_mode),
        "dense_vstim_avoided": True,
        "drive_count": 2,
        "double_cable_block_solver": groups_by_mode.get("double", {}).get(
            "execution_policy_double_cable_block_solver"
        ),
    }
    (run_dir / "multi_drive_route_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_compact_activation_route(run_dir: Path, *, cable: str) -> None:
    """Reject a recruitment benchmark that falls back to temporal VmRaster."""

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_modes = {"single", "double"} if cable == "mixed" else {str(cable)}
    sinks: dict[str, str] = {}
    dispatch_observers: set[str] = set()
    for event in events:
        metadata = event.get("metadata", {})
        if event.get("name") == "dispatch.group.total":
            mode = str(metadata.get("mode", ""))
            if mode in expected_modes:
                sinks[mode] = str(
                    metadata.get("prepared_input_contract_output_sink", "")
                )
                if metadata.get("runtime_input_contract_supports_threshold_observer") is not True:
                    raise RuntimeError(
                        f"{mode} runtime contract does not support threshold observers."
                    )
        elif event.get("name") == "kernel.dispatch_jax":
            mode = str(metadata.get("mode", ""))
            if mode in expected_modes:
                dispatch_observers.add(str(metadata.get("observer", "")))

    if set(sinks) != expected_modes:
        raise RuntimeError(
            "compact activation validation did not observe every dispatch mode: "
            f"got {sorted(sinks)}, expected {sorted(expected_modes)}."
        )
    fallback_modes = sorted(mode for mode, sink in sinks.items() if sink != "activation")
    if fallback_modes:
        raise RuntimeError(
            "recruitment benchmark used a non-compact observer sink for "
            f"{fallback_modes}: {sinks}."
        )
    if dispatch_observers != {"activation"}:
        raise RuntimeError(
            "recruitment benchmark dispatched an unexpected observer route: "
            f"{sorted(dispatch_observers)}."
        )

    payload = {
        "expected_modes": sorted(expected_modes),
        "output_sinks": sinks,
        "dispatch_observer": "activation",
        "vm_raster_fallback": False,
    }
    (run_dir / "compact_activation_route_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_workload(args: argparse.Namespace) -> RecruitmentWorkload:
    rng = np.random.default_rng(int(args.seed))
    single_count, double_count = _cable_counts(args.cable, int(args.axon_count))
    realistic = args.workload == "p14_realistic"
    circle_radius = (250.0 if realistic else 125.0) * axs.um
    fiber_length = (5_000.0 if realistic else 1_500.0) * axs.um
    stim_start = 0.20 * axs.ms
    pulse_width = 0.10 * axs.ms
    sigma = 0.3 * axs.S_per_m
    current_steps = np.asarray(args.amplitudes, dtype=float) * axs.uA

    electrode = axs.analytical.PointSourceElectrode(
        x=fiber_length / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
        min_distance=5.0 * axs.um,
    )
    secondary_electrode = axs.analytical.PointSourceElectrode(
        x=0.7 * fiber_length,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
        min_distance=5.0 * axs.um,
    )
    variable_drive_id = axs.DriveId("variable")
    static_drive_id = axs.DriveId("static")
    zero_current = axs.Stimulus.pulse(
        start=stim_start,
        duration=pulse_width,
        amplitude=0.0 * axs.uA,
    )
    static_current = axs.Stimulus.pulse(
        start=stim_start + 0.25 * axs.ms,
        duration=0.08 * axs.ms,
        amplitude=-5.0 * axs.uA,
    )

    radius_um = circle_radius.to(axs.um).magnitude
    unmyelinated_angles = rng.uniform(0.0, 2.0 * np.pi, single_count)
    unmyelinated_radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, single_count))
    unmyelinated_y = unmyelinated_radii * np.cos(unmyelinated_angles) * axs.um
    unmyelinated_z = unmyelinated_radii * np.sin(unmyelinated_angles) * axs.um

    myelinated_angles = rng.uniform(0.0, 2.0 * np.pi, double_count)
    myelinated_radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, double_count))
    myelinated_y = myelinated_radii * np.cos(myelinated_angles) * axs.um
    myelinated_z = myelinated_radii * np.sin(myelinated_angles) * axs.um

    unmyelinated_diameters_um = axs.axons.round_axon_diameter_values_um(
        rng.uniform(0.4, 1.2, single_count)
    )
    myelinated_diameters, myelinated_x_shifts = _mrg_population_templates(
        rng,
        row_count=double_count,
        template_count=int(args.mrg_template_count),
    )
    myelinated_diameters = myelinated_diameters * axs.um
    myelinated_x_shifts = myelinated_x_shifts * axs.um

    pool: list[axs.AxonInstance] = []
    unmyelinated_templates: dict[float, tuple[Any, Any]] = {}
    mrg_templates: dict[tuple[Any, ...], Any] = {}
    for diameter_um, y, z in zip(
        unmyelinated_diameters_um,
        unmyelinated_y,
        unmyelinated_z,
        strict=True,
    ):
        diameter_key = float(diameter_um)
        template = (
            unmyelinated_templates.get(diameter_key)
            if args.axon_template_policy == "shared"
            else None
        )
        if template is None:
            axon = axs.axons.RattayAberham(
                length=fiber_length,
                diameter=diameter_key * axs.um,
                compartments=(200 if realistic else 61),
                celsius=37.0 * axs.degC,
            )
            positions = axon.layout.position_values(unit=axs.um) * axs.um
            template = (axon, positions)
            if args.axon_template_policy == "shared":
                unmyelinated_templates[diameter_key] = template
        axon, positions = template
        extracellular = axs.analytical.point_source_stimulation(
            electrode,
            positions,
            sigma=sigma,
            stimulus=zero_current,
            drive_id=variable_drive_id,
            axon_y=y,
            axon_z=z,
        )
        if args.drive_count == 2:
            static_drive = axs.analytical.point_source_drive(
                secondary_electrode,
                positions,
                sigma=sigma,
                stimulus=static_current,
                drive_id=static_drive_id,
                axon_y=y,
                axon_z=z,
            )
            extracellular = extracellular.add(static_drive)
        row = axs.AxonInstance(axon)
        row.add_extracellular_stimulation(stimulation=extracellular)
        pool.append(row)

    for diameter, x_shift, y, z in zip(
        myelinated_diameters,
        myelinated_x_shifts,
        myelinated_y,
        myelinated_z,
        strict=True,
    ):
        if realistic:
            nodes = max(
                2,
                axs.axons.mrg_like_nodes_from_length(
                    diameter,
                    fiber_length,
                    x_shift=x_shift,
                ),
            )
            compartments: Any = 1
        else:
            nodes = 4
            compartments = {"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1}
        template_key = (
            float(diameter.to(axs.um).magnitude),
            int(nodes),
            float(fiber_length.to(axs.um).magnitude),
            float(x_shift.to(axs.um).magnitude),
            repr(compartments),
        )
        axon = (
            mrg_templates.get(template_key)
            if args.axon_template_policy == "shared"
            else None
        )
        if axon is None:
            axon = axs.axons.MRG(
                diameter=diameter,
                nodes=nodes,
                length=fiber_length,
                x_shift=x_shift,
                compartments=compartments,
            )
            if args.axon_template_policy == "shared":
                mrg_templates[template_key] = axon
        extracellular = axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit=axs.um) * axs.um,
            sigma=sigma,
            stimulus=zero_current,
            drive_id=variable_drive_id,
            axon_y=y,
            axon_z=z,
        )
        if args.drive_count == 2:
            static_drive = axs.analytical.point_source_drive(
                secondary_electrode,
                axon.layout.position_values(unit=axs.um) * axs.um,
                sigma=sigma,
                stimulus=static_current,
                drive_id=static_drive_id,
                axon_y=y,
                axon_z=z,
            )
            extracellular = extracellular.add(static_drive)
        row = axs.AxonInstance(axon)
        row.add_extracellular_stimulation(stimulation=extracellular)
        pool.append(row)

    unique_axons = {id(row.axon): row.axon for row in pool}
    unique_sections = {
        id(section): section
        for axon in unique_axons.values()
        for section in axon.layout.sections
    }
    unique_membranes = {
        id(section.membrane): section.membrane for section in unique_sections.values()
    }
    record_benchmark_metadata(
        axon_template_policy=args.axon_template_policy,
        population_rows=len(pool),
        population_unique_axon_templates=len(unique_axons),
        population_unique_sections=len(unique_sections),
        population_unique_membrane_models=len(unique_membranes),
        population_requested_mrg_templates=int(args.mrg_template_count),
        population_realized_mrg_templates=len(mrg_templates),
    )

    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=stim_start,
        target=axs.positions.ALL,
    )

    update_point_source_current = axs.protocols.ExtracellularWaveformUpdate(
        lambda current_magnitude: axs.Stimulus.pulse(
            start=stim_start,
            duration=pulse_width,
            amplitude=-current_magnitude,
        ),
        drive_id=variable_drive_id,
    )

    return tuple(pool), update_point_source_current, current_steps, criterion


def _cable_counts(cable: str, axon_count: int) -> tuple[int, int]:
    if cable == "single":
        return axon_count, 0
    if cable == "double":
        return 0, axon_count
    single_count = axon_count // 2
    return single_count, axon_count - single_count


def _mrg_population_templates(
    rng: np.random.Generator,
    *,
    row_count: int,
    template_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return balanced MRG diameters and intrinsic node shifts for a benchmark."""

    if row_count <= 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    realized_count = min(int(template_count), int(row_count))
    if realized_count == 3:
        diameters = rng.choice(np.asarray([7.3, 10.0, 12.8]), size=row_count)
        return diameters.astype(float), np.zeros((row_count,), dtype=float)

    base_diameters = np.asarray(P14_MRG_DIAMETERS_UM, dtype=float)
    shift_level_count = int(np.ceil(realized_count / len(base_diameters)))
    template_diameters = np.empty((realized_count,), dtype=float)
    template_shifts = np.empty((realized_count,), dtype=float)
    for index in range(realized_count):
        diameter_um = float(base_diameters[index % len(base_diameters)])
        shift_level = index // len(base_diameters)
        shift_fraction = float(shift_level) / float(shift_level_count)
        template_diameters[index] = diameter_um
        template_shifts[index] = shift_fraction * float(
            axs.axons.mrg_like_node_spacing(diameter_um * axs.um)
        )

    template_indices = np.arange(row_count, dtype=np.int64) % realized_count
    rng.shuffle(template_indices)
    return template_diameters[template_indices], template_shifts[template_indices]


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
        "workload",
        "cable",
        "drive_count",
        "fibers_per_family",
        "n_axons",
        "amplitude_count",
        "wall_ms",
        "source_population_build_ms",
        "one_shot_wall_ms",
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
            fieldnames=(
                "script",
                "preset",
                "platform",
                "workload",
                "cable",
                "drive_count",
                "policy",
                "n_axons",
                "amplitude_count",
            ),
        )
        writer.writeheader()
        for policy in policies:
            writer.writerow(
                {
                    "script": "recruitment_amplitude_batch",
                    "preset": args.preset,
                    "platform": args.platform,
                    "workload": args.workload,
                    "cable": args.cable,
                    "drive_count": args.drive_count,
                    "policy": policy.label,
                    "n_axons": args.axon_count,
                    "amplitude_count": len(args.amplitudes),
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
        "workload": args.workload,
        "cable": args.cable,
        "drive_count": args.drive_count,
        "axon_template_policy": args.axon_template_policy,
        "mrg_template_count": args.mrg_template_count,
        "policies": [policy.label for policy in policies],
        "fibers_per_family": args.fibers_per_family,
        "n_axons": args.axon_count,
        "amplitudes_uA": list(args.amplitudes),
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "time_chunk_steps": args.time_chunk_steps,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "cold_only": args.cold_only,
        "capture_jit_phases": args.capture_jit_phases,
        "validate_double_cable_kernel": args.validate_double_cable_kernel,
        "compilation_cache_replay": args.compilation_cache_replay,
        "memory_trace": args.memory_trace,
        "profile": bool(args.profile),
        "profile_scope": args.profile_scope,
        "profile_create_perfetto": bool(args.profile_create_perfetto),
        "output": str(output),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_report(output: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Recruitment Amplitude Batch Benchmark",
        "",
        "| cable | policy | phase | wall ms | source build ms | one-shot ms | "
        "build plan ms | chunk ms | "
        "run pool ms | dispatch_jax ms | wait ms | counts match |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- |",
    ]
    for row in rows:
        lines.append(
            (
                "| {cable} | {policy} | {phase} | {wall} | {source_build} | "
                "{one_shot} | {build_plan} | {chunk} | {run_pool} | {dispatch} | "
                "{wait} | {match} |"
            ).format(
                cable=row["cable"],
                policy=row["policy"],
                phase=row["phase"],
                wall=_fmt(row.get("wall_ms", "")),
                source_build=_fmt(row.get("source_population_build_ms", "")),
                one_shot=_fmt(row.get("one_shot_wall_ms", "")),
                build_plan=_fmt(row.get("dispatch.build_plan_ms", "")),
                chunk=_fmt(row.get("protocol.sweep.amplitude_chunk_ms", "")),
                run_pool=_fmt(row.get("simulation.run_pool_ms", "")),
                dispatch=_fmt(row.get("kernel.dispatch_jax_ms", "")),
                wait=_fmt(row.get("kernel.wait_ms", "")),
                match=row.get("matches_reference", ""),
            )
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"{row['workload']} {row['cable']} {row['policy']} "
        f"{row['phase']}#{row['repeat']}: "
        f"wall={float(row['wall_ms']):.1f} ms "
        f"build_plan={_fmt(row.get('dispatch.build_plan_ms', ''))} ms "
        f"chunk={_fmt(row.get('protocol.sweep.amplitude_chunk_ms', ''))} ms "
        f"wait={_fmt(row.get('kernel.wait_ms', ''))} ms"
    )


def _fmt(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
