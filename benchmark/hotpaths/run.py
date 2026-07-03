"""Run Phase 2.5 hotpath diagnostic workloads.

Examples:
    python benchmark/hotpaths/run.py --list
    python benchmark/hotpaths/run.py --workload all --preset smoke
    python benchmark/hotpaths/run.py --workload all --preset scale
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import jax
import numpy as np
import jax.numpy as jnp

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import axonscope as axs
from axonscope.backends.jax.input_batches import (
    build_footprint_vstim_midpoint_batch,
    build_intracellular_current_density_batch,
    build_vstim_midpoint_batch,
)
from axonscope.benchmarking.hotpaths import (
    benchmark_array_metadata,
    benchmark_span,
    benchmark_wait,
    record_benchmark_metadata,
)
from axonscope.backends.jax.batch_kernels import SingleCableVStimBatchKernel
from axonscope.solvers import (
    BatchOptions,
    resolve_double_cable_block_solver,
)
from axonscope.backends.jax.runtime import prepare_solver_runtime
from benchmark.hotpaths.catalog import HOTPATH_PRESETS, HOTPATH_WORKLOADS


DEFAULT_OUT_DIR = Path("benchmark/results/hotpaths")
DIRECT_WORKLOADS = frozenset(
    {
        "solver_only_precomputed",
        "typed_footprint_drive_matrix",
    }
)


@dataclass(frozen=True)
class PlannedRun:
    """One workload/size combination."""

    workload: str
    size: int


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        default="all",
        choices=("all", *HOTPATH_WORKLOADS),
        help="Workload to run.",
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=tuple(HOTPATH_PRESETS),
        help="Named size preset.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit population sizes. Overrides --preset.",
    )
    parser.add_argument("--compartments", type=int, default=11)
    parser.add_argument("--length-um", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=0.30, help="Duration in ms.")
    parser.add_argument("--dt", type=float, default=0.05, help="Step size in ms.")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--time-chunk-steps",
        type=int,
        default=None,
        help="Optional batch-kernel time chunk size for long-run probes.",
    )
    parser.add_argument(
        "--double-cable-block-solver",
        choices=("auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive"),
        default="auto",
        help=(
            "Double-cable batch block solver: auto chooses adaptive PCR on GPU "
            "and Thomas elsewhere."
        ),
    )
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=3,
        help="Number of repeated simulations for the footprint_reuse_sweep workload.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None, help="Optional run directory prefix.")
    parser.add_argument("--list", action="store_true", help="List workloads and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit.")
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each AxonScope hotpath report.",
    )
    parser.add_argument(
        "--sync-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize JAX arrays at kernel.wait.",
    )
    parser.add_argument(
        "--jax-log-compiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable JAX compile logging for cold-start diagnostic runs.",
    )
    parser.add_argument(
        "--jax-trace",
        action="store_true",
        help="Capture JAX profiler traces for measured hotpath runs.",
    )
    parser.add_argument(
        "--jax-trace-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for JAX profiler traces. Passing this also enables "
            "trace capture. Defaults to <run-root>/jax_traces when --jax-trace "
            "is set."
        ),
    )
    parser.add_argument(
        "--jax-trace-create-perfetto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask JAX to also write a Perfetto trace artifact when supported.",
    )
    parser.add_argument(
        "--jax-trace-scope",
        choices=("kernel", "run"),
        default="kernel",
        help=(
            "Trace only kernel.enqueue by default. Use 'run' to include "
            "dispatch, input preparation, kernel execution, and result packaging."
        ),
    )
    parser.add_argument(
        "--memory-trace",
        choices=("off", "rss", "tracemalloc", "device", "all"),
        default="off",
        help=(
            "Record measured per-span memory metadata. rss samples process RSS, "
            "tracemalloc samples Python/NumPy-visible allocations, device "
            "samples JAX device memory_stats and nvidia-smi when available."
        ),
    )
    parser.add_argument(
        "--memory-top-n",
        type=int,
        default=0,
        help="Include the top N tracemalloc allocation deltas per span.",
    )
    parser.add_argument(
        "--jax-device-memory-profile",
        action="store_true",
        help=(
            "Save JAX device memory .prof artifacts for selected spans. The "
            "default selected stage is kernel.wait, after block_until_ready()."
        ),
    )
    parser.add_argument(
        "--jax-device-memory-profile-stage",
        action="append",
        default=None,
        help=(
            "Stage name for JAX device memory profile capture. Repeat to select "
            "several stages, or pass 'all'. Defaults to kernel.wait."
        ),
    )
    args = parser.parse_args(argv)

    if args.list:
        print_workloads()
        return
    if args.compartments < 3:
        raise ValueError("--compartments must be >= 3.")
    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if args.time_chunk_steps is not None and args.time_chunk_steps < 1:
        raise ValueError("--time-chunk-steps must be >= 1.")
    if args.sweep_repeats < 1:
        raise ValueError("--sweep-repeats must be >= 1.")
    if args.memory_top_n < 0:
        raise ValueError("--memory-top-n must be >= 0.")

    runs = planned_runs(args.workload, resolve_sizes(args.preset, args.sizes))
    if args.dry_run:
        for run in runs:
            print(f"{run.workload} size={run.size}")
        return

    jax_compile_logging = configure_jax_compile_logging(bool(args.jax_log_compiles))
    timing_mode = _timing_mode(args.warmups)
    timing_signature = _timing_signature(args.warmups)
    jax_backend = jax.default_backend()
    resolved_double_cable_block_solver = resolve_double_cable_block_solver(
        args.double_cable_block_solver,
        platform=jax_backend,
    )
    run_root = make_run_root(args.out_dir, prefix=args.prefix)
    jax_trace_enabled = bool(args.jax_trace or args.jax_trace_dir is not None)
    jax_trace_root = _resolve_jax_trace_root(
        run_root=run_root,
        requested_dir=args.jax_trace_dir,
        enabled=jax_trace_enabled,
    )
    benchmark_options = _benchmark_options(args)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "sizes": list(dict.fromkeys(run.size for run in runs)),
        "workloads": list(dict.fromkeys(run.workload for run in runs)),
        "parameters": {
            "compartments": int(args.compartments),
            "length_um": float(args.length_um),
            "duration_ms": float(args.duration),
            "dt_ms": float(args.dt),
            "warmups": int(args.warmups),
            "timing_mode": timing_mode,
            "timing_signature": timing_signature,
            "time_chunk_steps": args.time_chunk_steps,
            "double_cable_block_solver": args.double_cable_block_solver,
            "double_cable_block_solver_resolved": resolved_double_cable_block_solver,
            "jax_default_backend": jax_backend,
            "sweep_repeats": int(args.sweep_repeats),
            "sync_device": bool(args.sync_device),
            "jax_log_compiles": bool(args.jax_log_compiles),
            "jax_trace": jax_trace_enabled,
            "jax_trace_dir": None if jax_trace_root is None else str(jax_trace_root),
            "jax_trace_create_perfetto": bool(args.jax_trace_create_perfetto),
            "jax_trace_scope": args.jax_trace_scope,
            "memory_trace": args.memory_trace,
            "memory_top_n": int(args.memory_top_n),
            "jax_device_memory_profile": bool(args.jax_device_memory_profile),
            "jax_device_memory_profile_stages": _profile_stages_for_metadata(args),
        },
        "jax_compile_logging": jax_compile_logging,
        "runs": [],
    }

    for run in runs:
        if run.workload in DIRECT_WORKLOADS:
            run_record = run_direct_workload(
                run.workload,
                size=run.size,
                compartments=args.compartments,
                length_um=args.length_um,
                duration_ms=args.duration,
                dt_ms=args.dt,
                warmups=args.warmups,
                timing_mode=timing_mode,
                sync_device=bool(args.sync_device),
                print_summary=bool(args.print_summary),
                output_dir=run_root / f"{run.workload}_n{run.size}",
                jax_log_compiles=bool(args.jax_log_compiles),
                time_chunk_steps=args.time_chunk_steps,
                jax_trace=_jax_trace_record(
                    jax_trace_root,
                    workload=run.workload,
                    size=run.size,
                    create_perfetto_trace=bool(args.jax_trace_create_perfetto),
                    scope=args.jax_trace_scope,
                ),
                benchmark_options=benchmark_options,
            )
            manifest["runs"].append(run_record)
            print(
                f"{run.workload} size={run.size}: "
                f"{run_record['event_count']} events -> {run_record['output_dir']}"
            )
            jax_trace = run_record.get("jax_trace", {})
            if isinstance(jax_trace, dict) and jax_trace.get("enabled"):
                print(f"  jax trace: {jax_trace['trace_dir']}")
            continue

        simulations = build_simulations(
            run.workload,
            size=run.size,
            compartments=args.compartments,
            length_um=args.length_um,
            duration_ms=args.duration,
            dt_ms=args.dt,
            sweep_repeats=args.sweep_repeats,
        )
        simulations = _with_batch_options(
            simulations,
            time_chunk_steps=args.time_chunk_steps,
            double_cable_block_solver=args.double_cable_block_solver,
        )
        estimates = [simulation.estimate().to_dict() for simulation in simulations]
        simulation_labels = _simulation_labels(run.workload, len(simulations))
        for _ in range(args.warmups):
            for simulation in simulations:
                simulation.run()

        output_dir = run_root / f"{run.workload}_n{run.size}"
        session = axs.enable_benchmark(
            output_dir,
            print_summary=False,
            sync_device=bool(args.sync_device),
            **benchmark_options,
        )
        session.metadata.update(
            {
                "workload": run.workload,
                "size": int(run.size),
                "simulation_count": len(simulations),
                "simulation_labels": list(simulation_labels),
                "warmup_count": int(args.warmups),
                "timing_mode": timing_mode,
                "timing_signature": timing_signature,
                "time_chunk_steps": args.time_chunk_steps,
                "double_cable_block_solver": args.double_cable_block_solver,
                "double_cable_block_solver_resolved": resolved_double_cable_block_solver,
                "jax_default_backend": jax_backend,
                "jax_log_compiles": bool(args.jax_log_compiles),
            }
        )
        jax_trace = _jax_trace_record(
            jax_trace_root,
            workload=run.workload,
            size=run.size,
            create_perfetto_trace=bool(args.jax_trace_create_perfetto),
            scope=args.jax_trace_scope,
        )
        session.metadata["jax_trace"] = jax_trace
        try:
            with _jax_profiler_trace(
                jax_trace,
                workload=run.workload,
                size=run.size,
            ):
                result_batches = _run_labeled_simulations(
                    simulations,
                    labels=simulation_labels,
                    workload=run.workload,
                    size=run.size,
                )
        finally:
            report = axs.disable_benchmark(print_summary=bool(args.print_summary))

        run_record = {
            "workload": run.workload,
            "size": run.size,
            "simulation_count": len(simulations),
            "warmup_count": int(args.warmups),
            "timing_mode": timing_mode,
            "timing_signature": timing_signature,
            "output_dir": str(output_dir),
            "simulation_labels": list(simulation_labels),
            "result_count": sum(_result_count(results) for results in result_batches),
            "vm_shapes": _sample_vm_shapes(result_batches),
            "observation_names": _sample_observation_names(result_batches),
            "memory_estimate": estimates[0],
            "memory_estimates": estimates,
            "workload_metadata": _describe_simulations(simulation_labels, simulations),
            "jax_trace": jax_trace,
            "event_count": 0 if report is None else len(report.events),
            "summary": [] if report is None else [row.to_dict() for row in report.summary],
        }
        manifest["runs"].append(run_record)
        print(
            f"{run.workload} size={run.size}: "
            f"{run_record['event_count']} events -> {output_dir}"
        )
        if jax_trace["enabled"]:
            print(f"  jax trace: {jax_trace['trace_dir']}")

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest: {manifest_path}")


def run_direct_workload(
    workload: str,
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
    warmups: int,
    timing_mode: str,
    sync_device: bool,
    print_summary: bool,
    output_dir: Path,
    jax_log_compiles: bool,
    time_chunk_steps: int | None,
    jax_trace: dict[str, object],
    benchmark_options: dict[str, object],
) -> dict[str, object]:
    """Run a backend-level hotpath workload that bypasses public dispatch."""

    if workload == "solver_only_precomputed":
        return _run_solver_only_precomputed(
            workload=workload,
            size=size,
            compartments=compartments,
            length_um=length_um,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            warmups=warmups,
            timing_mode=timing_mode,
            sync_device=sync_device,
            print_summary=print_summary,
            output_dir=output_dir,
            jax_log_compiles=jax_log_compiles,
            time_chunk_steps=time_chunk_steps,
            jax_trace=jax_trace,
            benchmark_options=benchmark_options,
        )
    if workload == "typed_footprint_drive_matrix":
        return _run_typed_footprint_drive_matrix(
            workload=workload,
            size=size,
            compartments=compartments,
            length_um=length_um,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            warmups=warmups,
            timing_mode=timing_mode,
            sync_device=sync_device,
            print_summary=print_summary,
            output_dir=output_dir,
            jax_log_compiles=jax_log_compiles,
            time_chunk_steps=time_chunk_steps,
            jax_trace=jax_trace,
            benchmark_options=benchmark_options,
        )
    raise ValueError(f"Unknown direct hotpath workload: {workload!r}.")


def _run_labeled_simulations(
    simulations: Sequence[axs.AxonSimulation],
    *,
    labels: Sequence[str],
    workload: str,
    size: int,
) -> tuple[axs.AxonSimulationResult, ...]:
    """Run public simulations under case-level benchmark labels."""

    results = []
    for label, simulation in zip(labels, simulations, strict=True):
        with benchmark_span(
            "simulation.case",
            workload=workload,
            simulation_label=label,
            size=int(size),
        ):
            results.append(simulation.run())
    return tuple(results)


def _benchmark_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "memory_trace": args.memory_trace,
        "memory_top_n": int(args.memory_top_n),
        "jax_device_memory_profile": bool(args.jax_device_memory_profile),
        "jax_device_memory_profile_stages": _profile_stages_for_session(args),
    }


def _profile_stages_for_session(args: argparse.Namespace) -> tuple[str, ...] | None:
    stages = args.jax_device_memory_profile_stage
    if stages is None:
        return None
    return tuple(str(stage) for stage in stages)


def _profile_stages_for_metadata(args: argparse.Namespace) -> list[str]:
    stages = _profile_stages_for_session(args)
    return ["kernel.wait"] if stages is None else list(stages)


def _run_solver_only_precomputed(
    *,
    workload: str,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
    warmups: int,
    timing_mode: str,
    sync_device: bool,
    print_summary: bool,
    output_dir: Path,
    jax_log_compiles: bool,
    time_chunk_steps: int | None,
    jax_trace: dict[str, object],
    benchmark_options: dict[str, object],
) -> dict[str, object]:
    """Run kernels with runtime and inputs prepared before benchmarking."""

    intra_axons = build_intracellular_pool(
        size=size,
        compartments=compartments,
        length_um=length_um,
    )
    extra_axons = build_point_source_pool(
        size=size,
        compartments=compartments,
        length_um=length_um,
    )
    intra_runtime = _single_cable_runtime(intra_axons[0], duration_ms=duration_ms, dt_ms=dt_ms)
    extra_runtime = _single_cable_runtime(extra_axons[0], duration_ms=duration_ms, dt_ms=dt_ms)
    intra_iinj = build_intracellular_current_density_batch(intra_axons, intra_runtime)
    intra_vstim = jnp.zeros(
        (size, intra_runtime.grid.Nt, intra_runtime.membrane.Nx),
        dtype=intra_runtime.membrane.dtype,
    )
    extra_vstim = _stimulation_vstim_for_instances(
        extra_axons,
        extra_runtime,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
    )
    cases = (
        {
            "label": "solver_only_single_intracellular_precomputed",
            "runtime": intra_runtime,
            "iinj": intra_iinj,
            "vstim": intra_vstim,
            "has_driven_extracellular": False,
            "time_chunk_steps": time_chunk_steps,
            "estimate": _simulation_estimate(
                intra_axons,
                duration_ms=duration_ms,
                dt_ms=dt_ms,
            ),
        },
        {
            "label": "solver_only_single_point_source_precomputed",
            "runtime": extra_runtime,
            "iinj": None,
            "vstim": extra_vstim,
            "has_driven_extracellular": True,
            "time_chunk_steps": time_chunk_steps,
            "estimate": _simulation_estimate(
                extra_axons,
                duration_ms=duration_ms,
                dt_ms=dt_ms,
            ),
        },
    )

    for _ in range(int(warmups)):
        for case in cases:
            _run_precomputed_single_cable_case(case)

    session = axs.enable_benchmark(
        output_dir,
        print_summary=False,
        sync_device=bool(sync_device),
        **benchmark_options,
    )
    session.metadata.update(
        {
            "workload": workload,
            "size": int(size),
            "simulation_count": len(cases),
            "simulation_labels": [str(case["label"]) for case in cases],
            "warmup_count": int(warmups),
            "timing_mode": timing_mode,
            "timing_signature": _timing_signature(warmups),
            "jax_log_compiles": bool(jax_log_compiles),
            "direct_backend_workload": True,
            "precomputed_inputs": True,
            "time_chunk_steps": time_chunk_steps,
            "jax_trace": jax_trace,
        }
    )
    try:
        with _jax_profiler_trace(jax_trace, workload=workload, size=size):
            outputs = tuple(_run_precomputed_single_cable_case(case) for case in cases)
    finally:
        report = axs.disable_benchmark(print_summary=bool(print_summary))

    return _direct_run_record(
        workload=workload,
        size=size,
        output_dir=output_dir,
        warmups=warmups,
        timing_mode=timing_mode,
        labels=[str(case["label"]) for case in cases],
        estimates=[case["estimate"] for case in cases],
        outputs=outputs,
        report=report,
        metadata=[
            _direct_case_metadata(
                str(case["label"]),
                case,
                bypasses_input_materialization=True,
            )
            for case in cases
        ],
        jax_trace=jax_trace,
    )


def _run_typed_footprint_drive_matrix(
    *,
    workload: str,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
    warmups: int,
    timing_mode: str,
    sync_device: bool,
    print_summary: bool,
    output_dir: Path,
    jax_log_compiles: bool,
    time_chunk_steps: int | None,
    jax_trace: dict[str, object],
    benchmark_options: dict[str, object],
) -> dict[str, object]:
    """Compare generic stimulation lowering against typed drive lowering."""

    axons = build_point_source_pool(
        size=size,
        compartments=compartments,
        length_um=length_um,
    )
    runtime = _single_cable_runtime(axons[0], duration_ms=duration_ms, dt_ms=dt_ms)
    estimate = _simulation_estimate(axons, duration_ms=duration_ms, dt_ms=dt_ms)
    label = "typed_footprint_drive_single_point_source"

    for _ in range(int(warmups)):
        stimulation_vstim = _stimulation_vstim_for_instances(
            axons,
            runtime,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )
        typed_vstim = _typed_drive_vstim_for_instances(
            axons,
            runtime,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )
        _run_precomputed_single_cable_case(
            {
                "label": label,
                "runtime": runtime,
                "iinj": None,
                "vstim": typed_vstim,
                "has_driven_extracellular": True,
                "time_chunk_steps": time_chunk_steps,
            }
        )
        benchmark_wait(stimulation_vstim)

    session = axs.enable_benchmark(
        output_dir,
        print_summary=False,
        sync_device=bool(sync_device),
        **benchmark_options,
    )
    session.metadata.update(
        {
            "workload": workload,
            "size": int(size),
            "simulation_count": 1,
            "simulation_labels": [label],
            "warmup_count": int(warmups),
            "timing_mode": timing_mode,
            "timing_signature": _timing_signature(warmups),
            "jax_log_compiles": bool(jax_log_compiles),
            "direct_backend_workload": True,
            "typed_footprint_drive": True,
            "time_chunk_steps": time_chunk_steps,
            "jax_trace": jax_trace,
        }
    )
    try:
        with _jax_profiler_trace(jax_trace, workload=workload, size=size):
            with benchmark_span(
                "inputs.extracellular.stimulation",
                workload=workload,
                simulation_label=label,
                input_format="typed_stimulation_dense_vstim",
            ):
                stimulation_vstim = _stimulation_vstim_for_instances(
                    axons,
                    runtime,
                    duration_ms=duration_ms,
                    dt_ms=dt_ms,
                )
                record_benchmark_metadata(
                    **benchmark_array_metadata(
                        "vstim_mid_stimulation",
                        stimulation_vstim,
                        role="kernel_input",
                    )
                )
            with benchmark_span(
                "inputs.extracellular.typed_drive",
                workload=workload,
                simulation_label=label,
                input_format="typed_footprint_drive_dense_vstim",
            ):
                typed_vstim = _typed_drive_vstim_for_instances(
                    axons,
                    runtime,
                    duration_ms=duration_ms,
                    dt_ms=dt_ms,
                )
                record_benchmark_metadata(
                    **benchmark_array_metadata(
                        "vstim_mid_typed_drive",
                        typed_vstim,
                        role="kernel_input",
                    )
                )
            with benchmark_span("inputs.extracellular.compare", workload=workload):
                delta = np.asarray(stimulation_vstim) - np.asarray(typed_vstim)
                record_benchmark_metadata(max_abs_delta_mV=float(np.max(np.abs(delta))))
            output = _run_precomputed_single_cable_case(
                {
                    "label": label,
                    "runtime": runtime,
                    "iinj": None,
                    "vstim": typed_vstim,
                    "has_driven_extracellular": True,
                    "time_chunk_steps": time_chunk_steps,
                }
            )
    finally:
        report = axs.disable_benchmark(print_summary=bool(print_summary))

    return _direct_run_record(
        workload=workload,
        size=size,
        output_dir=output_dir,
        warmups=warmups,
        timing_mode=timing_mode,
        labels=[label],
        estimates=[estimate],
        outputs=(output,),
        report=report,
        metadata=[
            {
                "label": label,
                "direct_backend_workload": True,
                "path_family": "single_point_source_extracellular",
                "stimulation": "typed_footprint_drive",
                "recording_policy": {"spatial": "center", "signals": ["membrane_voltage"]},
                "comparison_axes": {
                    "path_family": "single_point_source_extracellular",
                    "stimulation": "typed_footprint_drive",
                    "recording_spatial": "center",
                    "recording_voltage": True,
                    "observer_mode": "none",
                },
            }
        ],
        jax_trace=jax_trace,
    )


def print_workloads() -> None:
    print("Hotpath workloads:")
    for name, workload in HOTPATH_WORKLOADS.items():
        print(f"  {name:28s} {workload.description}")
    print("Presets:")
    for name, sizes in HOTPATH_PRESETS.items():
        joined = ", ".join(str(size) for size in sizes)
        print(f"  {name:28s} sizes={joined}")


def _single_cable_runtime(
    representative: axs.AxonInstance,
    *,
    duration_ms: float,
    dt_ms: float,
):
    return prepare_solver_runtime(
        representative,
        tsim_ms=duration_ms,
        dt_ms=dt_ms,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
        compile_stimulation=False,
    )


def _with_batch_options(
    simulations: Sequence[axs.AxonSimulation],
    *,
    time_chunk_steps: int | None,
    double_cable_block_solver: str,
) -> tuple[axs.AxonSimulation, ...]:
    """Return simulations with benchmark batch-kernel options applied."""

    updated = []
    for simulation in simulations:
        batch_options = simulation.batch_options or BatchOptions()
        updated_options = replace(
            batch_options,
            double_cable_block_solver=double_cable_block_solver,
        )
        if time_chunk_steps is not None:
            updated_options = replace(
                updated_options,
                time_chunk_steps=int(time_chunk_steps),
            )
        updated.append(
            axs.AxonSimulation(
                simulation.population,
                duration=simulation.duration,
                dt=simulation.dt,
                recording=simulation.recording,
                solver=simulation.solver,
                solver_options=simulation.solver_options,
                batch_options=updated_options,
                observers=simulation.observers,
                progress=simulation.progress,
            )
        )
    return tuple(updated)


def _simulation_estimate(
    instances: Sequence[axs.AxonInstance],
    *,
    duration_ms: float,
    dt_ms: float,
) -> axs.SimulationEstimate:
    simulation = axs.AxonSimulation(
        axs.AxonPopulation(instances),
        duration=duration_ms * axs.ms,
        dt=dt_ms * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )
    return simulation.estimate()


def _x_positions_m_for_instances(instances: Sequence[axs.AxonInstance]) -> np.ndarray:
    rows = []
    for instance in instances:
        x_um = np.asarray(instance.axon.layout.position_values(unit="micrometer"), dtype=float)
        rows.append(x_um * 1e-6)
    return np.stack(rows, axis=0)


def _axon_y_um_for_instances(instances: Sequence[axs.AxonInstance]) -> np.ndarray:
    return np.zeros((len(instances),), dtype=float)


def _axon_z_um_for_instances(instances: Sequence[axs.AxonInstance]) -> np.ndarray:
    return np.zeros((len(instances),), dtype=float)


def _stimulation_vstim_for_instances(
    instances: Sequence[axs.AxonInstance],
    runtime: object,
    *,
    duration_ms: float,
    dt_ms: float,
):
    return build_vstim_midpoint_batch(
        instances[0],
        [instance.extracellular_stimulation for instance in instances],
        tsim_ms=duration_ms,
        dt_ms=dt_ms,
        x_positions_m=_x_positions_m_for_instances(instances),
        axon_y_um=_axon_y_um_for_instances(instances),
        axon_z_um=_axon_z_um_for_instances(instances),
        dtype_local=runtime.membrane.dtype,
    )


def _typed_drive_vstim_for_instances(
    instances: Sequence[axs.AxonInstance],
    runtime: object,
    *,
    duration_ms: float,
    dt_ms: float,
):
    stimulations = [instance.extracellular_stimulation for instance in instances]
    if any(stimulation is None for stimulation in stimulations):
        raise ValueError("typed drive workload requires extracellular stimulations.")
    first_stimulation = stimulations[0]
    first_drive = first_stimulation.drives[0]
    stimulus = first_drive.stimulus

    positions_um = np.asarray(
        instances[0].axon.layout.position_values(unit="micrometer"),
        dtype=float,
    )
    values = np.stack(
        [
            stimulation.drives[0].footprint.values_for_axon()
            for stimulation in stimulations
            if stimulation is not None
        ],
        axis=0,
    )
    axon_ids = tuple(axs.AxonId(f"row_{index}") for index in range(len(instances)))
    footprint = axs.ExtracellularFootprint(
        values=values,
        positions=positions_um * axs.um,
        axon_ids=axon_ids,
        source_id="hotpath_point_source",
        metadata={"builder": "benchmark.hotpaths.typed_footprint_drive_matrix"},
    )
    drive = axs.ExtracellularDrive(
        id=axs.DriveId("point_source"),
        footprint=footprint,
        stimulus=stimulus,
    )
    stimulation = axs.ExtracellularStimulation([drive])
    vstim = build_footprint_vstim_midpoint_batch(
        stimulus=drive.stimulus,
        footprint_V_per_A=drive.footprint.values_V_per_A,
        tsim_ms=duration_ms,
        dt_ms=dt_ms,
        dtype_local=runtime.membrane.dtype,
    )
    record_benchmark_metadata(
        drive_ids=[str(value) for value in stimulation.names],
        footprint_rows=len(axon_ids),
        footprint_positions=footprint.n_positions,
    )
    return vstim


def _run_precomputed_single_cable_case(case: dict[str, object]):
    runtime = case["runtime"]
    time_chunk_steps = case.get("time_chunk_steps")
    chunk_steps = None if time_chunk_steps is None else int(time_chunk_steps)
    with benchmark_span(
        "kernel.enqueue",
        workload="direct_backend",
        simulation_label=str(case["label"]),
        recording_mode="center",
    ):
        out = SingleCableVStimBatchKernel(
            runtime=runtime,
            Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
            has_driven_extracellular=bool(case["has_driven_extracellular"]),
        ).run(
            intracellular_current_density_mid=case.get("iinj"),
            extracellular_potential_mid_mV=case["vstim"],
            options=BatchOptions.center(time_chunk_steps=chunk_steps),
        )
        if out.Vm is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
            )
    with benchmark_span(
        "kernel.wait",
        workload="direct_backend",
        simulation_label=str(case["label"]),
    ):
        benchmark_wait(out.Vm)
    return out


def _direct_case_metadata(
    label: str,
    case: dict[str, object],
    *,
    bypasses_input_materialization: bool,
) -> dict[str, object]:
    stimulation = (
        "analytical_point_source_extracellular"
        if bool(case["has_driven_extracellular"])
        else "intracellular_current_clamp"
    )
    path_family = (
        "single_point_source_extracellular"
        if bool(case["has_driven_extracellular"])
        else "single_intracellular"
    )
    return {
        "label": label,
        "direct_backend_workload": True,
        "precomputed_inputs": bool(bypasses_input_materialization),
        "path_family": path_family,
        "stimulation": stimulation,
        "recording_policy": {"spatial": "center", "signals": ["membrane_voltage"]},
        "comparison_axes": {
            "path_family": path_family,
            "stimulation": stimulation,
            "recording_spatial": "center",
            "recording_voltage": True,
            "observer_mode": "none",
        },
    }


def _direct_run_record(
    *,
    workload: str,
    size: int,
    output_dir: Path,
    warmups: int,
    timing_mode: str,
    labels: Sequence[str],
    estimates: Sequence[axs.SimulationEstimate],
    outputs: Sequence[object],
    report: axs.BenchmarkReport | None,
    metadata: Sequence[dict[str, object]],
    jax_trace: dict[str, object],
) -> dict[str, object]:
    return {
        "workload": workload,
        "size": int(size),
        "simulation_count": len(labels),
        "warmup_count": int(warmups),
        "timing_mode": timing_mode,
        "timing_signature": _timing_signature(warmups),
        "output_dir": str(output_dir),
        "simulation_labels": list(labels),
        "result_count": int(size) * len(labels),
        "vm_shapes": [
            list(np.asarray(output.Vm).shape)
            for output in outputs
            if getattr(output, "Vm", None) is not None
        ][:3],
        "observation_names": [],
        "memory_estimate": estimates[0].to_dict(),
        "memory_estimates": [estimate.to_dict() for estimate in estimates],
        "workload_metadata": list(metadata),
        "jax_trace": jax_trace,
        "event_count": 0 if report is None else len(report.events),
        "summary": [] if report is None else [row.to_dict() for row in report.summary],
    }


def resolve_sizes(preset: str, sizes: Sequence[int] | None) -> tuple[int, ...]:
    if sizes is not None:
        resolved = tuple(int(size) for size in sizes)
    else:
        resolved = HOTPATH_PRESETS[preset]
    if not resolved:
        raise ValueError("at least one size is required.")
    if any(size < 1 for size in resolved):
        raise ValueError("all sizes must be >= 1.")
    return resolved


def planned_runs(workload: str, sizes: Sequence[int]) -> tuple[PlannedRun, ...]:
    workload_names = tuple(HOTPATH_WORKLOADS) if workload == "all" else (workload,)
    return tuple(
        PlannedRun(workload=name, size=int(size))
        for name in workload_names
        for size in sizes
    )


def make_run_root(out_dir: Path, *, prefix: str | None) -> Path:
    stem = prefix or f"hotpaths_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = out_dir / stem
    root.mkdir(parents=True, exist_ok=True)
    return root


def _timing_mode(warmups: int) -> str:
    """Return a stable label for whether measured events include cold setup."""

    return "cold" if int(warmups) == 0 else "warm"


def _timing_signature(warmups: int) -> dict[str, object]:
    """Return explicit first-call labels for hotpath manifests."""

    warmup_count = int(warmups)
    cold_start = warmup_count == 0
    return {
        "label": "cold_first_call" if cold_start else "warm_post_warmup",
        "mode": _timing_mode(warmup_count),
        "warmup_count": warmup_count,
        "first_call_included": cold_start,
        "setup_may_be_included": cold_start,
        "jax_compile_may_be_included": cold_start,
    }


def configure_jax_compile_logging(enabled: bool) -> dict[str, object]:
    """Enable JAX compile logging for diagnostic runs when requested."""

    if not enabled:
        return {"enabled": False}

    import jax

    jax.config.update("jax_log_compiles", True)
    return {
        "enabled": True,
        "mechanism": "jax.config.update('jax_log_compiles', True)",
    }


def _resolve_jax_trace_root(
    *,
    run_root: Path,
    requested_dir: Path | None,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    return requested_dir if requested_dir is not None else run_root / "jax_traces"


def _jax_trace_record(
    root: Path | None,
    *,
    workload: str,
    size: int,
    create_perfetto_trace: bool,
    scope: str,
) -> dict[str, object]:
    if root is None:
        return {"enabled": False}
    label = _safe_trace_label(f"{workload}_n{int(size)}")
    trace_dir = root / label
    return {
        "enabled": True,
        "label": label,
        "trace_dir": str(trace_dir),
        "create_perfetto_trace": bool(create_perfetto_trace),
        "scope": str(scope),
    }


@contextmanager
def _jax_profiler_trace(
    trace: dict[str, object],
    *,
    workload: str,
    size: int,
):
    if not trace.get("enabled", False):
        yield
        return
    if trace.get("scope", "kernel") != "run":
        yield
        return

    trace_dir = Path(str(trace["trace_dir"]))
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(
        str(trace_dir),
        create_perfetto_trace=bool(trace.get("create_perfetto_trace", False)),
    ):
        with jax.profiler.StepTraceAnnotation(
            "hotpath_run",
            workload=str(workload),
            size=int(size),
        ):
            yield


def _safe_trace_label(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(value)
    ).strip("_")
    return cleaned or "run"


def build_simulation(
    workload: str,
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
) -> axs.AxonSimulation:
    return build_simulations(
        workload,
        size=size,
        compartments=compartments,
        length_um=length_um,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        sweep_repeats=1,
    )[0]


def build_simulations(
    workload: str,
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
    sweep_repeats: int = 1,
) -> tuple[axs.AxonSimulation, ...]:
    if workload == "intracellular_only":
        instances = build_intracellular_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    elif workload == "cold_run_micro":
        return build_cold_run_micro_matrix(
            size=size,
            compartments=compartments,
            length_um=length_um,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )
    elif workload == "point_source_extracellular":
        instances = build_point_source_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    elif workload == "double_cable_extracellular":
        instances = build_double_cable_extracellular_pool(
            size=size,
            compartments=compartments,
        )
    elif workload == "double_cable_observer":
        instances = build_double_cable_extracellular_pool(
            size=size,
            compartments=compartments,
        )
        activation = axs.analysis.Activation(
            threshold=-80.0 * axs.mV,
            target=axs.positions.CENTER,
        )
        latency = axs.analysis.Latency(
            threshold=-80.0 * axs.mV,
            target=axs.positions.CENTER,
            name="latency_center",
        )
        return (
            axs.AxonSimulation(
                axs.AxonPopulation(instances),
                duration=duration_ms * axs.ms,
                dt=dt_ms * axs.ms,
                recording=axs.Recording.none(),
                observers=[activation, latency],
            ),
        )
    elif workload == "footprint_reuse_sweep":
        return tuple(
            axs.AxonSimulation(
                axs.AxonPopulation(
                    build_point_source_pool(
                        size=size,
                        compartments=compartments,
                        length_um=length_um,
                        amplitude_uA=20.0 + 2.5 * repeat,
                    )
                ),
                duration=duration_ms * axs.ms,
                dt=dt_ms * axs.ms,
                recording=axs.Recording.center(axs.signals.Vm),
            )
            for repeat in range(int(sweep_repeats))
        )
    elif workload == "observer_only":
        instances = build_intracellular_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
        activation = axs.analysis.Activation(
            threshold=-80.0 * axs.mV,
            target=axs.positions.CENTER,
        )
        latency = axs.analysis.Latency(
            threshold=-80.0 * axs.mV,
            target=axs.positions.CENTER,
            name="latency_center",
        )
        return (
            axs.AxonSimulation(
                axs.AxonPopulation(instances),
                duration=duration_ms * axs.ms,
                dt=dt_ms * axs.ms,
                recording=axs.Recording.none(),
                observers=[activation, latency],
            ),
        )
    elif workload == "realistic_mixed_population":
        instances = build_realistic_mixed_population(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    elif workload == "hotpath_matrix":
        return build_hotpath_matrix(
            size=size,
            compartments=compartments,
            length_um=length_um,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )
    elif workload == "path_comparison_matrix":
        return build_path_comparison_matrix(
            size=size,
            compartments=compartments,
            length_um=length_um,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )
    else:
        raise ValueError(f"Unknown hotpath workload: {workload!r}.")

    return (
        axs.AxonSimulation(
            axs.AxonPopulation(instances),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
    )


def build_cold_run_micro_matrix(
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
) -> tuple[axs.AxonSimulation, ...]:
    """Return the short P9 cold-run baseline matrix."""

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    latency = axs.analysis.Latency(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
        name="latency_center",
    )
    return (
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.none(),
            observers=[activation, latency],
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_point_source_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
    )


def build_realistic_mixed_population(
    *,
    size: int,
    compartments: int,
    length_um: float,
) -> list[axs.AxonInstance]:
    """Build a deliberately mixed population for Phase 7.6 hotpath probes."""

    stimulus = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.10 * axs.ms,
        amplitude=18.0 * axs.uA,
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=0.5 * length_um * axs.um,
        z=140.0 * axs.um,
        stimulus=stimulus,
    )

    diameter_cycle_um = (0.6, 0.8, 1.0, 1.2)
    offsets = np.linspace(-60.0, 60.0, size) if size > 1 else np.asarray([0.0])
    axon_templates: dict[tuple[str, float, int], axs.axons.Axon] = {}
    instances: list[axs.AxonInstance] = []
    for index, offset_um in enumerate(offsets):
        diameter_um = diameter_cycle_um[index % len(diameter_cycle_um)]
        row_compartments = max(3, int(compartments) + 2 * (index % 3))
        model_kind = "hh" if index % 2 == 0 else "rattay_aberham"
        template_key = (model_kind, float(diameter_um), int(row_compartments))
        axon = axon_templates.get(template_key)
        if axon is None and model_kind == "hh":
            axon = axs.axons.HodgkinHuxley(
                length=length_um * axs.um,
                diameter=diameter_um * axs.um,
                compartments=row_compartments,
                celsius=6.3 * axs.degC,
            )
            axon_templates[template_key] = axon
        elif axon is None:
            axon = axs.axons.RattayAberham(
                length=length_um * axs.um,
                diameter=diameter_um * axs.um,
                compartments=row_compartments,
                celsius=37.0 * axs.degC,
            )
            axon_templates[template_key] = axon

        instance = axs.AxonInstance(axon)
        instance.add_current_clamp(
            position=0.5 * length_um * axs.um,
            current=axs.Stimulus.pulse(
                start=0.08 * axs.ms,
                duration=0.12 * axs.ms,
                amplitude=(0.45 + 0.02 * (index % 5)) * axs.nA,
            ),
        )
        if index % 3 == 0:
            instance.add_extracellular_stimulation(
                stimulation=axs.analytical.point_source_stimulation(
                    electrode,
                    axon.layout.position_values(unit=axs.um) * axs.um,
                    sigma=0.3 * axs.S_per_m,
                    axon_y=float(offset_um) * axs.um,
                )
            )
        instances.append(instance)
    return instances


def build_hotpath_matrix(
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
) -> tuple[axs.AxonSimulation, ...]:
    """Return a compact matrix of representative Phase 7.6 hotpath scenarios."""

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    latency = axs.analysis.Latency(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
        name="latency_center",
    )
    return (
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.probes(axs.signals.Vm, count=5),
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.none(),
            observers=[activation, latency],
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_point_source_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_realistic_mixed_population(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
    )


def build_path_comparison_matrix(
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
) -> tuple[axs.AxonSimulation, ...]:
    """Return controlled path comparisons for Phase 7.6.1 decisions."""

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    latency = axs.analysis.Latency(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
        name="latency_center",
    )

    def simulation(
        instances: list[axs.AxonInstance],
        *,
        recording: axs.Recording,
        observers: list[object] | None = None,
    ) -> axs.AxonSimulation:
        return axs.AxonSimulation(
            axs.AxonPopulation(instances),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=recording,
            observers=observers,
        )

    return (
        simulation(
            build_intracellular_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        simulation(
            build_intracellular_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.probes(axs.signals.Vm, count=5),
        ),
        simulation(
            build_intracellular_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.voltage(),
        ),
        simulation(
            build_intracellular_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.none(),
            observers=[activation, latency],
        ),
        simulation(
            build_point_source_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        simulation(
            build_point_source_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.probes(axs.signals.Vm, count=5),
        ),
        simulation(
            build_point_source_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.voltage(),
        ),
        simulation(
            build_point_source_pool(
                size=size,
                compartments=compartments,
                length_um=length_um,
            ),
            recording=axs.Recording.none(),
            observers=[activation, latency],
        ),
        simulation(
            build_double_cable_extracellular_pool(
                size=size,
                compartments=compartments,
            ),
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        simulation(
            build_double_cable_extracellular_pool(
                size=size,
                compartments=compartments,
            ),
            recording=axs.Recording.voltage(),
        ),
    )


def build_intracellular_pool(
    *,
    size: int,
    compartments: int,
    length_um: float,
) -> list[axs.AxonInstance]:
    axon = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.9 * axs.um,
        compartments=compartments,
        celsius=6.3 * axs.degC,
    )
    instances = []
    for index in range(size):
        instance = axs.AxonInstance(axon)
        instance.add_current_clamp(
            position=0.5 * length_um * axs.um,
            current=axs.Stimulus.pulse(
                start=0.10 * axs.ms,
                duration=0.10 * axs.ms,
                amplitude=(0.5 + 0.01 * index) * axs.nA,
            ),
        )
        instances.append(instance)
    return instances


def build_point_source_pool(
    *,
    size: int,
    compartments: int,
    length_um: float,
    amplitude_uA: float = 25.0,
) -> list[axs.AxonInstance]:
    axon = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.9 * axs.um,
        compartments=compartments,
        celsius=6.3 * axs.degC,
    )
    stimulus = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.10 * axs.ms,
        amplitude=float(amplitude_uA) * axs.uA,
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=0.5 * length_um * axs.um,
        z=120.0 * axs.um,
        stimulus=stimulus,
    )

    offsets = np.linspace(-40.0, 40.0, size) if size > 1 else np.asarray([0.0])
    instances = []
    for offset_um in offsets:
        instance = axs.AxonInstance(axon)
        instance.add_extracellular_stimulation(
            stimulation=axs.analytical.point_source_stimulation(
                electrode,
                axon.layout.position_values(unit=axs.um) * axs.um,
                sigma=0.3 * axs.S_per_m,
                axon_y=float(offset_um) * axs.um,
            )
        )
        instances.append(instance)
    return instances


def build_double_cable_extracellular_pool(
    *,
    size: int,
    compartments: int,
) -> list[axs.AxonInstance]:
    """Build a homogeneous myelinated double-cable extracellular workload."""

    nodes = _mrg_nodes_for_target_compartments(compartments)
    axon = axs.axons.MRG(
        diameter=5.7 * axs.um,
        nodes=nodes,
    )
    center_x_um = 0.5 * float(axon.length)
    stimulus = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.10 * axs.ms,
        amplitude=60.0 * axs.uA,
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=center_x_um * axs.um,
        z=120.0 * axs.um,
        stimulus=stimulus,
    )

    offsets = np.linspace(-80.0, 80.0, size) if size > 1 else np.asarray([0.0])
    instances = []
    for offset_um in offsets:
        instance = axs.AxonInstance(axon)
        instance.add_extracellular_stimulation(
            stimulation=axs.analytical.point_source_stimulation(
                electrode,
                axon.layout.position_values(unit=axs.um) * axs.um,
                sigma=0.3 * axs.S_per_m,
                axon_y=float(offset_um) * axs.um,
            )
        )
        instances.append(instance)
    return instances


def _mrg_nodes_for_target_compartments(compartments: int) -> int:
    """Return a small MRG node count close to a target compartment count."""

    return max(3, int((int(compartments) + 10) // 11))


def _simulation_labels(workload: str, count: int) -> tuple[str, ...]:
    if workload == "cold_run_micro":
        labels = (
            "single_intracellular_center",
            "single_intracellular_observer_none",
            "single_point_source_center",
        )
        return labels[:count]
    if workload == "hotpath_matrix":
        labels = (
            "homogeneous_center",
            "homogeneous_probes",
            "observer_only_none",
            "point_source_center",
            "realistic_mixed_center",
        )
        return labels[:count]
    if workload == "path_comparison_matrix":
        labels = (
            "single_intracellular_center",
            "single_intracellular_probes",
            "single_intracellular_full_vm",
            "single_intracellular_observer_none",
            "single_point_source_center",
            "single_point_source_probes",
            "single_point_source_full_vm",
            "single_point_source_observer_none",
            "double_mrg_point_source_center",
            "double_mrg_point_source_full_vm",
        )
        return labels[:count]
    if workload == "footprint_reuse_sweep":
        return tuple(f"repeat_{index}" for index in range(count))
    return tuple(workload for _ in range(count))


def _describe_simulations(
    labels: Sequence[str],
    simulations: Sequence[axs.AxonSimulation],
) -> list[dict[str, object]]:
    return [
        _describe_simulation(label, simulation)
        for label, simulation in zip(labels, simulations, strict=True)
    ]


def _describe_simulation(
    label: str,
    simulation: axs.AxonSimulation,
) -> dict[str, object]:
    instances = tuple(simulation.axons)
    diameters_um = [float(instance.axon.diameter) for instance in instances]
    compartments = [int(instance.axon.n_compartments) for instance in instances]
    return {
        "label": label,
        "axon_count": len(instances),
        "model_counts": _count_values(type(instance.axon).__name__ for instance in instances),
        "formulation_counts": _count_values(
            instance.axon.resolved_formulation for instance in instances
        ),
        "diameter_um": _numeric_distribution(diameters_um),
        "compartments": _numeric_distribution(compartments),
        "intracellular_rows": sum(bool(instance.intracellular_contexts) for instance in instances),
        "extracellular_rows": sum(
            instance.extracellular_stimulation is not None for instance in instances
        ),
        "recording_policy": _recording_policy(simulation.recording),
        "observer_names": [
            str(getattr(observer, "name", type(observer).__name__))
            for observer in (simulation.observers or ())
        ],
        "comparison_axes": _comparison_axes(label, simulation),
    }


def _comparison_axes(
    label: str,
    simulation: axs.AxonSimulation,
) -> dict[str, object]:
    """Return stable matrix axes for controlled comparison labels."""

    if label.startswith("single_intracellular_"):
        path_family = "single_intracellular"
        stimulation = "intracellular_current_clamp"
    elif label.startswith("single_point_source_"):
        path_family = "single_point_source_extracellular"
        stimulation = "analytical_point_source_extracellular"
    elif label.startswith("double_mrg_point_source_"):
        path_family = "double_mrg_point_source_extracellular"
        stimulation = "analytical_point_source_extracellular"
    else:
        return {}

    recording = simulation.recording
    recording_voltage = True if recording is None else bool(recording.voltage)
    if recording is None:
        recording_spatial = "default"
    elif not recording_voltage:
        recording_spatial = "none"
    else:
        recording_spatial = recording.spatial.value
    return {
        "path_family": path_family,
        "stimulation": stimulation,
        "recording_spatial": recording_spatial,
        "recording_voltage": recording_voltage,
        "observer_mode": "solver_side" if simulation.observers else "none",
    }


def _count_values(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _numeric_distribution(values: Sequence[float | int]) -> dict[str, object]:
    if not values:
        return {"unique": [], "min": None, "max": None}
    unique = sorted({float(value) for value in values})
    formatted_unique: list[float | int] = [
        int(value) if value.is_integer() else value for value in unique[:12]
    ]
    distribution: dict[str, object] = {
        "unique": formatted_unique,
        "min": min(values),
        "max": max(values),
    }
    if len(unique) > 12:
        distribution["unique_truncated"] = True
        distribution["unique_count"] = len(unique)
    return distribution


def _recording_policy(recording: axs.Recording | None) -> dict[str, object]:
    if recording is None:
        return {"spatial": "default", "signals": ["membrane_voltage"]}
    if not recording.voltage:
        return {"spatial": "none", "signals": []}
    return {
        "spatial": recording.spatial.value,
        "signals": [str(signal.id) for signal in recording.signals],
        "probe_count": recording.probe_count,
        "indices": None if recording.record_indices is None else list(recording.record_indices),
    }


def _result_count(results: object) -> int:
    if isinstance(results, axs.AxonSimulationResult):
        return len(results)
    return 1


def _sample_vm_shapes(result_batches: Sequence[object]) -> list[list[int]]:
    shapes: list[list[int]] = []
    for results in result_batches:
        if isinstance(results, axs.AxonSimulationResult):
            for result in results[: max(0, 3 - len(shapes))]:
                try:
                    shapes.append(list(np.asarray(result.Vm).shape))
                except (AttributeError, ValueError):
                    continue
        else:
            try:
                shapes.append(list(np.asarray(results.Vm).shape))  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                continue
        if len(shapes) >= 3:
            break
    return shapes


def _sample_observation_names(result_batches: Sequence[object]) -> list[str]:
    for results in result_batches:
        observations = getattr(results, "observations", None)
        if observations:
            return sorted(str(name) for name in observations)
    return []


if __name__ == "__main__":
    main()
