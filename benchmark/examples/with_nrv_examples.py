"""Benchmark runnable NRV integration examples as public workflow gates."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import axonscope as axs
from axonscope.benchmarking import benchmark_span


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "with_nrv_examples"

INTERESTING_STAGES = (
    "example.run",
    "nrv_bridge.population_from_nrv",
    "nrv_bridge.footprints_from_nrv",
    "nrv_bridge.stimulated_population",
    "protocol.recruitment_sweep",
    "protocol.sweep.value",
    "simulation.setup",
    "simulation.run_pool",
    "dispatch.build_plan",
    "runtime.prepare",
    "kernel.enqueue",
    "kernel.dispatch_jax",
    "kernel.wait",
    "kernel.finalize_observer",
    "results.to_public",
)


@dataclass(frozen=True, slots=True)
class ExampleSpec:
    key: str
    path: Path
    label: str


@dataclass(frozen=True, slots=True)
class AmplitudeBatchPolicy:
    label: str
    batch_amplitudes: bool
    amplitude_batch_size: int | None


EXAMPLES = {
    "01": ExampleSpec(
        key="01",
        path=REPO_ROOT / "examples" / "with_nrv" / "01_synthetic_fascicle_geometry.py",
        label="synthetic_fascicle_geometry",
    ),
    "02": ExampleSpec(
        key="02",
        path=REPO_ROOT / "examples" / "with_nrv" / "02_realistic_fascicle_geometry.py",
        label="realistic_fascicle_geometry",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu", "nrv"), default="cpu")
    parser.add_argument("--examples", default="01,02")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--cold-only", action="store_true")
    parser.add_argument("--axons-per-fascicle", type=int, default=6)
    parser.add_argument("--amplitudes-uA", default="0,150,300")
    parser.add_argument(
        "--amplitude-batch-policy",
        default="sequential",
        help="Recruitment amplitude policy: sequential, a positive chunk size, or full.",
    )
    parser.add_argument("--duration-ms", type=float, default=0.5)
    parser.add_argument("--dt-ms", type=float, default=0.01)
    parser.add_argument(
        "--observer-time-chunk-steps",
        type=_parse_observer_time_chunk_steps,
        default=axs.DEFAULT_OBSERVER_TIME_CHUNK_STEPS,
        help="Observer scan chunk size, or 'none' for one scan over the full duration.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-trace", choices=("off", "rss", "tracemalloc", "device", "all"))
    parser.add_argument("--memory-top-n", type=int)
    parser.add_argument("--profile", action="store_true", dest="profile", default=None)
    parser.add_argument("--no-profile", action="store_false", dest="profile")
    parser.add_argument("--profile-runtime", choices=("auto", "jax", "none"))
    parser.add_argument("--profile-output")
    parser.add_argument("--profile-create-perfetto", action="store_true", default=None)
    parser.add_argument("--no-profile-create-perfetto", action="store_false", dest="profile_create_perfetto")
    parser.add_argument("--jax-device-memory-profile", action="store_true", default=None)
    parser.add_argument("--no-jax-device-memory-profile", action="store_false", dest="jax_device_memory_profile")
    parser.add_argument("--jax-device-memory-profile-stage", action="append", default=[])
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record array shape/device metadata in benchmark events.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = _parse_examples(args.examples)
    amplitudes = _parse_amplitudes(args.amplitudes_uA)
    amplitude_batch_policy = _parse_amplitude_batch_policy(args.amplitude_batch_policy)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")
    if args.warmups < 0:
        raise SystemExit("--warmups must be >= 0.")
    if args.axons_per_fascicle < 1:
        raise SystemExit("--axons-per-fascicle must be >= 1.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    _write_manifest(output, args, specs, amplitudes, amplitude_batch_policy)

    if args.dry_run:
        _write_cases(output, args, specs, amplitudes, amplitude_batch_policy)
        print(f"dry-run: with_nrv_examples -> {output}")
        return 0

    rows: list[dict[str, Any]] = []
    for spec in specs:
        module = _load_example(spec)
        phase_plan = [("cold", 0)]
        if not args.cold_only:
            phase_plan.extend(("warmup", index) for index in range(args.warmups))
            phase_plan.extend(("warm", index) for index in range(args.repeats))
        for phase, repeat in phase_plan:
            row = _run_one(
                module,
                spec,
                phase=phase,
                repeat=repeat,
                args=args,
                amplitudes=amplitudes,
                amplitude_batch_policy=amplitude_batch_policy,
                output=output,
            )
            rows.append(row)
            _write_runs(output, rows)
            print(_format_progress(row))

    _write_report(output, rows)
    return 0


def _parse_examples(value: str) -> list[ExampleSpec]:
    selected: list[ExampleSpec] = []
    for raw in value.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in EXAMPLES:
            allowed = ", ".join(sorted(EXAMPLES))
            raise SystemExit(f"Unknown with_nrv example {key!r}; expected one of: {allowed}.")
        selected.append(EXAMPLES[key])
    if not selected:
        raise SystemExit("--examples selected no examples.")
    return selected


def _parse_amplitudes(value: str) -> tuple[float, ...]:
    amplitudes = tuple(float(raw.strip()) for raw in value.split(",") if raw.strip())
    if not amplitudes:
        raise SystemExit("--amplitudes-uA selected no amplitudes.")
    return amplitudes


def _parse_amplitude_batch_policy(value: str) -> AmplitudeBatchPolicy:
    normalized = str(value).strip().lower()
    if normalized == "sequential":
        return AmplitudeBatchPolicy("sequential", False, None)
    if normalized == "full":
        return AmplitudeBatchPolicy("full", True, None)
    try:
        chunk_size = int(normalized)
    except ValueError as exc:
        raise SystemExit(
            "--amplitude-batch-policy must be sequential, full, or a positive integer."
        ) from exc
    if chunk_size < 1:
        raise SystemExit("--amplitude-batch-policy integer must be >= 1.")
    return AmplitudeBatchPolicy(normalized, True, chunk_size)


def _parse_observer_time_chunk_steps(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "full", "unchunked"}:
        return None
    try:
        chunk_steps = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "observer time chunk steps must be a positive integer or 'none'."
        ) from exc
    if chunk_steps < 1:
        raise argparse.ArgumentTypeError(
            "observer time chunk steps must be >= 1."
        )
    return chunk_steps


def _load_example(spec: ExampleSpec):
    module_name = f"_axonscope_with_nrv_example_{spec.key}_{spec.label}"
    spec_obj = importlib.util.spec_from_file_location(module_name, spec.path)
    if spec_obj is None or spec_obj.loader is None:
        raise RuntimeError(f"Could not load {spec.path}.")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{spec.path} does not define main().")
    return module


def _run_one(
    module: Any,
    spec: ExampleSpec,
    *,
    phase: str,
    repeat: int,
    args: argparse.Namespace,
    amplitudes: tuple[float, ...],
    amplitude_batch_policy: AmplitudeBatchPolicy,
    output: Path,
) -> dict[str, Any]:
    run_dir = output / spec.key / amplitude_batch_policy.label / f"{phase}_{repeat:02d}"
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start = time.perf_counter_ns()
    failed = False
    error = ""
    try:
        with _plot_safe_context():
            with axs.benchmark(
                run_dir,
                print_summary=False,
                save=True,
                sync_device=True,
                record_shapes=args.record_shapes,
                memory_trace=args.memory_trace or "rss",
                memory_top_n=0 if args.memory_top_n is None else args.memory_top_n,
                profile=bool(args.profile) if args.profile is not None else False,
                profile_runtime=args.profile_runtime or "auto",
                profile_output=args.profile_output,
                profile_create_perfetto=bool(args.profile_create_perfetto)
                if args.profile_create_perfetto is not None
                else False,
                jax_device_memory_profile=bool(args.jax_device_memory_profile)
                if args.jax_device_memory_profile is not None
                else False,
                jax_device_memory_profile_stages=tuple(args.jax_device_memory_profile_stage),
            ):
                with benchmark_span("example.run", example=spec.key, phase=phase, repeat=repeat):
                    _ensure_nrv_imported()
                    with _maybe_quiet(args.quiet, stdout_buffer, stderr_buffer):
                        result = module.main(
                            _example_config(
                                module,
                                args,
                                amplitudes,
                                amplitude_batch_policy,
                            )
                        )
                    _write_recruitment_result(run_dir / "recruitment_result.json", result)
    except BaseException as exc:
        failed = True
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        end = time.perf_counter_ns()
        row = _row_from_run_dir(run_dir)
        _write_run_pool_detail(
            run_dir,
            amplitudes=amplitudes,
            amplitude_batch_policy=amplitude_batch_policy,
        )
        row.update(
            {
                "example": spec.key,
                "label": spec.label,
                "phase": phase,
                "repeat": repeat,
                "amplitude_batch_policy": amplitude_batch_policy.label,
                "wall_ms": (end - start) / 1_000_000.0,
                "failed": failed,
                "error": error,
                "stdout_chars": len(stdout_buffer.getvalue()),
                "stderr_chars": len(stderr_buffer.getvalue()),
            }
        )
        _write_text(run_dir / "stdout.txt", stdout_buffer.getvalue())
        _write_text(run_dir / "stderr.txt", stderr_buffer.getvalue())
    return row


def _example_config(
    module: Any,
    args: argparse.Namespace,
    amplitudes: tuple[float, ...],
    amplitude_batch_policy: AmplitudeBatchPolicy,
) -> Any:
    return module.ExampleConfig(
        axons_per_fascicle=args.axons_per_fascicle,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        observer_time_chunk_steps=args.observer_time_chunk_steps,
        recruitment_amplitudes_uA=amplitudes,
        random_seed=int(args.seed),
        solver_progress=False,
        fem_n_proc=1,
        gmsh_n_core=1,
        execution_policy=_execution_policy_for_platform(args.platform),
        batch_amplitudes=amplitude_batch_policy.batch_amplitudes,
        amplitude_batch_size=amplitude_batch_policy.amplitude_batch_size,
    )


def _execution_policy_for_platform(platform: str) -> axs.ExecutionPolicy | None:
    if platform == "gpu":
        return axs.ExecutionPolicy(runtime=axs.runtime.jax, device=axs.Device.gpu())
    if platform == "cpu":
        return axs.ExecutionPolicy(runtime=axs.runtime.jax, device=axs.Device.cpu())
    return None


def _ensure_nrv_imported() -> None:
    # NRV enables faulthandler during import and expects stderr to expose a real
    # file descriptor, so import it before redirecting stdout/stderr to buffers.
    import nrv  # noqa: F401


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
        "example",
        "label",
        "phase",
        "repeat",
        "amplitude_batch_policy",
        "wall_ms",
        *[f"{stage}_ms" for stage in INTERESTING_STAGES],
        "failed",
        "error",
        "stdout_chars",
        "stderr_chars",
    ]
    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_recruitment_result(path: Path, result: Any) -> None:
    if result is None:
        return
    activated = getattr(result, "activated", None)
    amplitudes_uA = getattr(result, "amplitudes_uA", None)
    if activated is None or amplitudes_uA is None:
        return
    import numpy as np

    activated_array = np.asarray(activated, dtype=bool)
    amplitudes_array = np.asarray(amplitudes_uA, dtype=float)
    payload = {
        "amplitudes_uA": amplitudes_array.tolist(),
        "activated": activated_array.astype(int).tolist(),
        "count": np.sum(activated_array, axis=1).astype(int).tolist(),
        "fraction": (
            np.mean(activated_array, axis=1).astype(float).tolist()
            if activated_array.shape[1]
            else [0.0 for _ in range(activated_array.shape[0])]
        ),
        "first_activation_uA": np.asarray(
            getattr(result, "first_activation_uA"), dtype=float
        ).tolist(),
        "row_labels": [str(value) for value in getattr(result, "row_labels", ())],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_cases(
    output: Path,
    args: argparse.Namespace,
    specs: list[ExampleSpec],
    amplitudes: tuple[float, ...],
    amplitude_batch_policy: AmplitudeBatchPolicy,
) -> None:
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "script",
                "preset",
                "platform",
                "example",
                "label",
                "axons_per_fascicle",
                "amplitudes_uA",
                "duration_ms",
                "dt_ms",
                "observer_time_chunk_steps",
                "seed",
                "amplitude_batch_policy",
            ),
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "script": "with_nrv_examples",
                    "preset": args.preset,
                    "platform": args.platform,
                    "example": spec.key,
                    "label": spec.label,
                    "axons_per_fascicle": args.axons_per_fascicle,
                    "amplitudes_uA": ",".join(str(value) for value in amplitudes),
                    "duration_ms": args.duration_ms,
                    "dt_ms": args.dt_ms,
                    "observer_time_chunk_steps": args.observer_time_chunk_steps,
                    "seed": int(args.seed),
                    "amplitude_batch_policy": amplitude_batch_policy.label,
                }
            )


def _write_manifest(
    output: Path,
    args: argparse.Namespace,
    specs: list[ExampleSpec],
    amplitudes: tuple[float, ...],
    amplitude_batch_policy: AmplitudeBatchPolicy,
) -> None:
    payload = {
        "script": "with_nrv_examples",
        "preset": args.preset,
        "platform": args.platform,
        "examples": [spec.key for spec in specs],
        "repeats": args.repeats,
        "warmups": args.warmups,
        "axons_per_fascicle": args.axons_per_fascicle,
        "amplitudes_uA": amplitudes,
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "observer_time_chunk_steps": args.observer_time_chunk_steps,
        "seed": int(args.seed),
        "amplitude_batch_policy": amplitude_batch_policy.label,
        "memory_trace": args.memory_trace or "rss",
        "output": str(output),
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_report(output: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# With NRV Example Benchmark",
        "",
        "| example | policy | phase | repeat | wall ms | kernel.wait ms | results.to_public ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {example} {label} | {policy} | {phase} | {repeat} | {wall_ms:.3f} | {wait} | {public} |".format(
                example=row["example"],
                label=row["label"],
                policy=row["amplitude_batch_policy"],
                phase=row["phase"],
                repeat=row["repeat"],
                wall_ms=float(row["wall_ms"]),
                wait=_fmt_cell(row.get("kernel.wait_ms", "")),
                public=_fmt_cell(row.get("results.to_public_ms", "")),
            )
        )
    _write_text(output / "report.md", "\n".join(lines) + "\n")


def _fmt_cell(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.3f}"


def _format_progress(row: dict[str, Any]) -> str:
    return (
        f"{row['example']} {row['phase']}#{row['repeat']}: "
        f"policy={row['amplitude_batch_policy']} "
        f"wall={float(row['wall_ms']):.1f} ms "
        f"kernel.wait={_fmt_cell(row.get('kernel.wait_ms', '')) or 'n/a'} ms"
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_RUN_POOL_DETAIL_STAGES = (
    "dispatch.build_plan",
    "runtime.prepare",
    "inputs.positions",
    "inputs.extracellular",
    "kernel.enqueue",
    "kernel.dispatch_jax",
    "kernel.wait",
    "kernel.finalize_observer",
    "results.split_batch",
    "results.to_public",
)


def _write_run_pool_detail(
    run_dir: Path,
    *,
    amplitudes: tuple[float, ...],
    amplitude_batch_policy: AmplitudeBatchPolicy,
) -> None:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {int(event["event_id"]): event for event in events}
    run_pool_events = [event for event in events if event.get("name") == "simulation.run_pool"]
    rows: list[dict[str, Any]] = []
    completed_value_count = 0
    for unit_index, run_pool_event in enumerate(run_pool_events):
        descendants = [
            event
            for event in events
            if _is_descendant(event, int(run_pool_event["event_id"]), by_id)
        ]
        if amplitude_batch_policy.batch_amplitudes:
            batch_span = _nearest_ancestor_named(
                run_pool_event,
                "protocol.sweep.batched_values",
                by_id,
            )
            value_count = int((batch_span or {}).get("metadata", {}).get("value_count", 0))
            unit_amplitudes = amplitudes[
                completed_value_count : completed_value_count + value_count
            ]
        else:
            value_count = 1
            unit_amplitudes = amplitudes[unit_index : unit_index + 1]

        run_pool_ms = float(run_pool_event.get("duration_ms", 0.0))
        all_wait_ms = _sum_stage(descendants, "kernel.wait")
        base = {
            "unit_index": unit_index,
            "amplitude_count": value_count,
            "amplitudes_uA": " ".join(f"{value:g}" for value in unit_amplitudes),
            "run_pool_ms": run_pool_ms,
            "kernel_wait_ms": all_wait_ms,
            "kernel_wait_pct_run_pool": _percent(all_wait_ms, run_pool_ms),
        }
        for mode in ("all", "double", "single"):
            mode_events = descendants if mode == "all" else [
                event for event in descendants if _event_mode(event, by_id) == mode
            ]
            row = dict(base)
            row["mode"] = mode
            group_ms = (
                run_pool_ms
                if mode == "all"
                else _sum_stage(mode_events, "dispatch.group.total")
            )
            row["group_ms"] = group_ms
            for stage in _RUN_POOL_DETAIL_STAGES:
                row[f"{stage}_ms"] = _sum_stage(mode_events, stage)
            wait_ms = float(row["kernel.wait_ms"])
            row["kernel_wait_pct_group"] = _percent(wait_ms, group_ms)
            rows.append(row)
        completed_value_count += value_count

    if not rows:
        return
    fieldnames = list(rows[0])
    with (run_dir / "run_pool_detail.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_descendant(
    event: dict[str, Any],
    ancestor_id: int,
    by_id: dict[int, dict[str, Any]],
) -> bool:
    parent_id = event.get("parent_event_id")
    while parent_id is not None:
        if int(parent_id) == ancestor_id:
            return True
        parent = by_id.get(int(parent_id))
        if parent is None:
            return False
        parent_id = parent.get("parent_event_id")
    return False


def _nearest_ancestor_named(
    event: dict[str, Any],
    name: str,
    by_id: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    parent_id = event.get("parent_event_id")
    while parent_id is not None:
        parent = by_id.get(int(parent_id))
        if parent is None:
            return None
        if parent.get("name") == name:
            return parent
        parent_id = parent.get("parent_event_id")
    return None


def _event_mode(
    event: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
) -> str | None:
    current: dict[str, Any] | None = event
    while current is not None:
        mode = current.get("metadata", {}).get("mode")
        if mode in {"single", "double"}:
            return str(mode)
        parent_id = current.get("parent_event_id")
        current = None if parent_id is None else by_id.get(int(parent_id))
    return None


def _sum_stage(events: list[dict[str, Any]], name: str) -> float:
    return sum(
        float(event.get("duration_ms", 0.0))
        for event in events
        if event.get("name") == name
    )


def _percent(value: float, total: float) -> float:
    return 0.0 if total <= 0.0 else 100.0 * value / total


@contextlib.contextmanager
def _maybe_quiet(quiet: bool, stdout: io.StringIO, stderr: io.StringIO):
    if not quiet:
        yield
        return
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield


@contextlib.contextmanager
def _plot_safe_context():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = old_show
        plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
