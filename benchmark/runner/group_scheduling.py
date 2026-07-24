"""Benchmark sync vs async dispatch-group scheduling."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

import axonfleet as axs
from axonfleet.benchmarking import benchmark_span
from axonfleet.runner import _RunnerSchedulingOptions
from axonfleet.dispatcher.plan import build_dispatch_plan
from axonfleet.runtime.execution import (
    execution_context,
)
from axonfleet.solvers import BatchOptions


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "runner_group_scheduling"

INTERESTING_STAGES = (
    "benchmark.build_population",
    "benchmark.run_pool",
    "dispatch.build_plan",
    "dispatch.group.total",
    "dispatch.async_flush",
    "runtime.prepare",
    "inputs.positions",
    "inputs.intracellular",
    "inputs.extracellular",
    "observer.plan",
    "kernel.enqueue",
    "kernel.dispatch_jax",
    "kernel.wait",
    "kernel.finalize_observer",
    "results.split_batch",
)


@dataclass(frozen=True, slots=True)
class SchedulePolicy:
    label: str
    async_groups: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"), default="cpu")
    parser.add_argument("--policies", default="sync,async")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--rows-per-group", type=int, default=32)
    parser.add_argument("--max-pending-groups", type=int, default=4)
    parser.add_argument("--recording", choices=("observer", "center"), default="observer")
    parser.add_argument("--duration-ms", type=float, default=1.0)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--memory-trace",
        choices=("off", "rss", "tracemalloc", "device", "all"),
        default="rss",
    )
    parser.add_argument("--memory-top-n", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    policies = _parse_policies(args.policies)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    _write_manifest(output, args, policies)
    if args.dry_run:
        _write_cases(output, args, policies)
        print(f"dry-run: runner_group_scheduling -> {output}")
        return 0

    rows: list[dict[str, Any]] = []
    reference_signature: tuple[Any, ...] | None = None
    phase_plan = [("cold", 0)]
    phase_plan.extend(("warmup", index) for index in range(args.warmups))
    phase_plan.extend(("warm", index) for index in range(args.repeats))
    for policy in policies:
        for phase, repeat in phase_plan:
            row, signature = _run_one(args, output, policy, phase=phase, repeat=repeat)
            if reference_signature is None:
                reference_signature = signature
                row["matches_reference"] = True
            else:
                row["matches_reference"] = signature == reference_signature
            row["result_signature_hash"] = _signature_hash(signature)
            rows.append(row)
            _write_runs(output, rows)
            print(_format_progress(row))
    _write_report(output, rows)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.platform == "nrv":
        raise SystemExit("runner_group_scheduling supports cpu/gpu only.")
    if args.groups < 1:
        raise SystemExit("--groups must be >= 1.")
    if args.rows_per_group < 1:
        raise SystemExit("--rows-per-group must be >= 1.")
    if args.max_pending_groups < 1:
        raise SystemExit("--max-pending-groups must be >= 1.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")
    if args.warmups < 0:
        raise SystemExit("--warmups must be >= 0.")


def _run_one(
    args: argparse.Namespace,
    output: Path,
    policy: SchedulePolicy,
    *,
    phase: str,
    repeat: int,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    run_dir = output / policy.label / f"{phase}_{repeat:02d}"
    start = time.perf_counter_ns()
    failed = False
    error = ""
    signature: tuple[Any, ...] = ()
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
                groups=args.groups,
                rows_per_group=args.rows_per_group,
            ):
                pool = _build_single_cable_groups(args)
                plan = build_dispatch_plan(pool)
            execution_policy = _execution_policy(args.platform)
            with execution_context(execution_policy, instances=pool) as context:
                batch_options, observers = _batch_options_and_observers(args)
                with benchmark_span(
                    "benchmark.run_pool",
                    policy=policy.label,
                    phase=phase,
                    repeat=repeat,
                    async_groups=policy.async_groups,
                    max_pending_groups=args.max_pending_groups,
                    dispatch_group_count=len(plan.groups),
                ):
                    runner = axs.Runner(
                        _scheduling_options=_RunnerSchedulingOptions(
                            async_groups=policy.async_groups,
                            max_pending_groups=int(args.max_pending_groups),
                        )
                    )
                    result = runner._execute_dispatch_plan(
                        plan,
                        tsim_ms=float(args.duration_ms),
                        dt_ms=float(args.dt_ms),
                        batch_options=batch_options,
                        observers=observers,
                        runtime_context=context,
                    )
            signature = _result_signature(result)
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
                "async_groups": policy.async_groups,
                "phase": phase,
                "repeat": repeat,
                "platform": args.platform,
                "groups": args.groups,
                "rows_per_group": args.rows_per_group,
                "n_axons": args.groups * args.rows_per_group,
                "max_pending_groups": args.max_pending_groups,
                "recording": args.recording,
                "wall_ms": (end - start) / 1_000_000.0,
                "failed": failed,
                "error": error,
            }
        )
    return row, signature


def _build_single_cable_groups(args: argparse.Namespace) -> tuple[axs.AxonInstance, ...]:
    rng = np.random.default_rng(int(args.seed))
    rows: list[axs.AxonInstance] = []
    length = 1000.0 * axs.um
    for group_index in range(int(args.groups)):
        nx = 31 + 2 * group_index
        for row_index in range(int(args.rows_per_group)):
            diameter = float(rng.uniform(0.6, 1.0)) * axs.um
            axon_model = axs.axons.HodgkinHuxley(
                length=length,
                diameter=diameter,
                compartments=nx,
                celsius=6.3 * axs.degC,
            )
            row = axs.AxonInstance(axon_model)
            row.add_current_clamp(
                position=500.0 * axs.um,
                current=axs.Stimulus.pulse(
                    start=0.10 * axs.ms,
                    duration=0.20 * axs.ms,
                    amplitude=float(0.25 + 0.01 * row_index) * axs.nA,
                ),
            )
            rows.append(row)
    return tuple(rows)


def _execution_policy(platform: str) -> axs.ExecutionPolicy:
    device = axs.Device.gpu(0) if platform == "gpu" else axs.Device.cpu()
    precision = axs.PrecisionPolicy.float32() if platform == "gpu" else None
    return axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=device,
        precision=precision,
    )


def _batch_options_and_observers(
    args: argparse.Namespace,
) -> tuple[BatchOptions, tuple[Any, ...] | None]:
    if args.recording == "center":
        return BatchOptions.center(), None
    activation = axs.analysis.Activation(
        threshold=-20.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    return BatchOptions.none(), (activation,)


def _result_signature(result: Any) -> tuple[Any, ...]:
    rows: list[Any] = []
    for record in result:
        raw_indices = getattr(record, "indices", None)
        if raw_indices is None:
            raw_indices = (record.index,)
        indices = tuple(int(value) for value in raw_indices)
        if record.observations is not None:
            raster = record.observations.get(axs.VM_RASTER_OBSERVATION_KEY)
            words = np.asarray(raster.words, dtype=np.uint32)
            rows.append((indices, tuple(int(v) for v in words.shape), int(words.sum())))
            continue
        vm = np.asarray(record.Vm)
        rows.append((indices, tuple(int(v) for v in vm.shape), float(np.sum(vm))))
    return tuple(rows)


def _parse_policies(value: str) -> tuple[SchedulePolicy, ...]:
    policies: list[SchedulePolicy] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"sync", "sequential"}:
            policies.append(SchedulePolicy("sync", False))
        elif token in {"async", "async_groups"}:
            policies.append(SchedulePolicy("async", True))
        else:
            raise SystemExit(f"unknown scheduling policy: {raw!r}")
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
        "async_groups",
        "phase",
        "repeat",
        "platform",
        "groups",
        "rows_per_group",
        "n_axons",
        "max_pending_groups",
        "recording",
        "wall_ms",
        *[f"{stage}_ms" for stage in INTERESTING_STAGES],
        "matches_reference",
        "result_signature_hash",
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
    policies: tuple[SchedulePolicy, ...],
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
                    "script": "runner_group_scheduling",
                    "preset": args.preset,
                    "platform": args.platform,
                    "policy": policy.label,
                    "n_axons": args.groups * args.rows_per_group,
                }
            )


def _write_manifest(
    output: Path,
    args: argparse.Namespace,
    policies: tuple[SchedulePolicy, ...],
) -> None:
    payload = {
        "script": "runner_group_scheduling",
        "preset": args.preset,
        "platform": args.platform,
        "policies": [policy.label for policy in policies],
        "groups": args.groups,
        "rows_per_group": args.rows_per_group,
        "n_axons": args.groups * args.rows_per_group,
        "max_pending_groups": args.max_pending_groups,
        "recording": args.recording,
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
        "# Dispatcher Group Scheduling Benchmark",
        "",
        "| policy | phase | wall ms | run pool ms | async flush ms | enqueue ms | dispatch_jax ms | wait ms | split ms | match |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {phase} | {wall} | {run_pool} | {flush} | {enqueue} | {dispatch} | {wait} | {split} | {match} |".format(
                policy=row["policy"],
                phase=row["phase"],
                wall=_fmt(row.get("wall_ms", "")),
                run_pool=_fmt(row.get("benchmark.run_pool_ms", "")),
                flush=_fmt(row.get("dispatch.async_flush_ms", "")),
                enqueue=_fmt(row.get("kernel.enqueue_ms", "")),
                dispatch=_fmt(row.get("kernel.dispatch_jax_ms", "")),
                wait=_fmt(row.get("kernel.wait_ms", "")),
                split=_fmt(row.get("results.split_batch_ms", "")),
                match=row.get("matches_reference", ""),
            )
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"{row['policy']} {row['phase']}#{row['repeat']}: "
        f"wall={float(row['wall_ms']):.1f} ms "
        f"run_pool={_fmt(row.get('benchmark.run_pool_ms', ''))} ms "
        f"enqueue={_fmt(row.get('kernel.enqueue_ms', ''))} ms "
        f"wait={_fmt(row.get('kernel.wait_ms', ''))} ms "
        f"match={row.get('matches_reference', '')}"
    )


def _fmt(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.3f}"


def _signature_hash(signature: tuple[Any, ...]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
