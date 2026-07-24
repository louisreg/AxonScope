"""Validate lazy runnable plans and the local Runner on CPU or one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import platform as platform_module
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import axonfleet as axs
from axonfleet.protocols.observer_path import _activation_observations_from_pool_result


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "runner_plan_validation"
INTERESTING_STAGES = (
    "runner.study.task",
    "protocol.sweep.value_batch",
    "dispatch.build_plan",
    "dispatch.group.total",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scales",
        help="Comma-separated population sizes. Defaults to 8,32 for quick and 1024,4096 otherwise.",
    )
    parser.add_argument("--duration-ms", type=float, default=0.10)
    parser.add_argument("--dt-ms", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--memory-trace",
        choices=("off", "rss", "tracemalloc", "device", "all"),
        default="rss",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scales = _parse_scales(args.scales, preset=args.preset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "manifest.json", _manifest(args, scales))
    _write_cases(output, args, scales)
    if args.dry_run:
        print(f"dry-run: runner_plan_validation -> {output}")
        return 0

    started = time.perf_counter()
    validation: dict[str, Any] = {
        "schema": "axonfleet.runner_plan_validation.v1",
        "status": "running",
        "platform": args.platform,
        "preset": args.preset,
        "scales": list(scales),
        "runs": [],
    }
    try:
        policy = _execution_policy(args.platform)
        validation["environment"] = _environment(args.platform)
        built_at = time.perf_counter()
        study, activation = _build_study(args, policy)
        validation["plan_build_ms"] = (time.perf_counter() - built_at) * 1000.0
        validation["plan"] = _plan_summary(study)

        runner = axs.Runner()
        cold = _run_profiled(
            output,
            "study_cold",
            runner,
            study,
            activation,
            memory_trace=args.memory_trace,
        )
        validation["runs"].append(cold)
        warm = _run_profiled(
            output,
            "study_warm",
            runner,
            study,
            activation,
            memory_trace=args.memory_trace,
        )
        validation["runs"].append(warm)
        _require_equal(cold["signature"], warm["signature"], "cold/warm study")

        cache_before_clear = _cache_counts(runner)
        runner.clear()
        cache_after_clear = _cache_counts(runner)
        if any(cache_after_clear.values()):
            raise AssertionError(f"Runner.clear() left cache entries: {cache_after_clear}")
        rebuilt = _run_profiled(
            output,
            "study_after_clear",
            runner,
            study,
            activation,
            memory_trace=args.memory_trace,
        )
        validation["runs"].append(rebuilt)
        _require_equal(cold["signature"], rebuilt["signature"], "study after clear")
        validation["cache_clear"] = {
            "passed": True,
            "before": cache_before_clear,
            "after": cache_after_clear,
            "rebuilt": _cache_counts(runner),
        }

        estimate = runner.estimate(study)
        inspection = runner.inspect(study)
        validation["estimate"] = {
            "expected_rows": int(estimate.expected_rows),
            "simulation_executions_min": int(estimate.simulation_executions_min),
            "simulation_executions_max": int(estimate.simulation_executions_max),
            "peak_bytes": int(estimate.peak_bytes),
        }
        validation["inspection"] = {
            "expected_rows": int(inspection.expected_rows),
            "simulation_executions_min": int(inspection.simulation_executions_min),
            "simulation_executions_max": int(inspection.simulation_executions_max),
            "component_kinds": [item.plan_kind for item in inspection.components],
        }
        validation["cancellation"] = _validate_cancellation(study)
        validation["cache_invalidation"] = _validate_cache_invalidation(args, policy)

        scale_rows = []
        for size in scales:
            scale_rows.append(
                _validate_scale(
                    output,
                    args,
                    policy,
                    activation,
                    size=size,
                    memory_trace=args.memory_trace,
                )
            )
        validation["scale_validation"] = scale_rows
        validation["status"] = "passed"
    except BaseException as exc:
        validation["status"] = "failed"
        validation["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        validation["wall_s"] = time.perf_counter() - started
        _write_json(output / "validation.json", validation)
        _write_run_table(output, validation.get("runs", ()), validation.get("scale_validation", ()))
        _write_report(output, validation)

    print(f"runner_plan_validation passed: {output}")
    return 0


def _execution_policy(platform: str) -> axs.ExecutionPolicy:
    return axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.gpu(0) if platform == "gpu" else axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
    )


def _build_study(
    args: argparse.Namespace,
    policy: axs.ExecutionPolicy,
) -> tuple[axs.StudyPlan, axs.analysis.Activation]:
    duration = float(args.duration_ms) * axs.ms
    dt = float(args.dt_ms) * axs.ms
    activation = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=0.02 * axs.ms,
        target=axs.positions.CENTER,
    )

    hh = axs.axons.HodgkinHuxley(
        length=200.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=11,
    )
    driven_hh = axs.AxonInstance(hh)
    driven_hh.add_current_clamp(
        position=100.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.03 * axs.ms,
            amplitude=1.0 * axs.nA,
        ),
    )
    simple = axs.AxonSimulation(
        driven_hh,
        duration=duration,
        dt=dt,
        recording=axs.Recording.center(axs.signals.Vm),
        execution_policy=policy,
    ).plan()

    mrg = axs.axons.MRG(diameter=10.0 * axs.um, nodes=3)
    mixed = axs.AxonSimulation(
        (driven_hh, axs.AxonInstance(mrg)),
        duration=duration,
        dt=dt,
        recording=axs.Recording.center(axs.signals.Vm),
        execution_policy=policy,
    ).plan()

    sweep_pool = _extracellular_population(4)
    amplitudes = np.asarray([0.0, 25.0, 50.0], dtype=float) * axs.uA
    update = axs.protocols.ExtracellularWaveformUpdate(
        lambda value: axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.03 * axs.ms,
            amplitude=-value,
        )
    )
    sweep = axs.protocols.recruitment_sweep_plan(
        sweep_pool,
        update=update,
        values=amplitudes,
        duration=duration,
        dt=dt,
        criterion=activation,
        recording=axs.Recording.none(),
        execution_policy=policy,
        batch_amplitudes=True,
        amplitude_batch_size=len(amplitudes),
    )
    axis_builder = update.prepare_numeric_axis(sweep_pool)
    numeric_axis = sweep.source.with_numeric_axis(
        axis_builder.numeric_axis_input(tuple(amplitudes))
    )

    threshold = axs.ThresholdPlan(
        source=simple,
        update=_identity_update,
        decode=_never_satisfied,
        bounds=(0.0 * axs.uA, 1.0 * axs.uA),
        row_labels=("hh",),
        max_iterations=2,
    )
    return (
        axs.StudyPlan(
            name="p20_local_runner_validation",
            tasks=(
                axs.StudyTask("simple", simple),
                axs.StudyTask("mixed", mixed, depends_on=("simple",)),
                axs.StudyTask("numeric_axis", numeric_axis, depends_on=("mixed",)),
                axs.StudyTask("sweep", sweep, depends_on=("numeric_axis",)),
                axs.StudyTask("threshold", threshold, depends_on=("sweep",)),
            ),
        ),
        activation,
    )


def _identity_update(row: Any, _value: Any) -> Any:
    return row


def _never_satisfied(result: Any) -> tuple[bool, ...]:
    return tuple(False for _ in result)


def _extracellular_population(count: int) -> tuple[axs.AxonInstance, ...]:
    axon = axs.axons.HodgkinHuxley(
        length=200.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=11,
    )
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    electrode = axs.analytical.PointSourceElectrode(
        x=100.0 * axs.um,
        y=20.0 * axs.um,
        z=0.0 * axs.um,
    )
    stimulation = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.03 * axs.ms,
            amplitude=0.0 * axs.uA,
        ),
    )
    rows = []
    for _ in range(int(count)):
        row = axs.AxonInstance(axon)
        row.add_extracellular_stimulation(stimulation=stimulation)
        rows.append(row)
    return tuple(rows)


def _observer_plan(
    args: argparse.Namespace,
    policy: axs.ExecutionPolicy,
    activation: axs.analysis.Activation,
    size: int,
) -> axs.SimulationPlan:
    axon = axs.axons.HodgkinHuxley(
        length=200.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=11,
    )
    rows = tuple(axon for _ in range(int(size)))
    return axs.AxonSimulation(
        rows,
        duration=float(args.duration_ms) * axs.ms,
        dt=float(args.dt_ms) * axs.ms,
        recording=axs.Recording.none(),
        observers=(activation,),
        execution_policy=policy,
    ).plan()


def _run_profiled(
    output: Path,
    label: str,
    runner: axs.Runner,
    plan: Any,
    activation: axs.analysis.Activation,
    *,
    memory_trace: str,
) -> dict[str, Any]:
    run_dir = output / "profiles" / label
    start = time.perf_counter()
    with axs.benchmark(
        run_dir,
        print_summary=False,
        save=True,
        sync_device=True,
        record_shapes=True,
        memory_trace=memory_trace,
        profile=False,
        profile_runtime="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
    ):
        result = runner.run(plan)
    return {
        "label": label,
        "wall_ms": (time.perf_counter() - start) * 1000.0,
        "stages_ms": _stage_totals(run_dir),
        "signature": _study_signature(result, activation),
        "cache_counts": _cache_counts(runner),
    }


def _validate_scale(
    output: Path,
    args: argparse.Namespace,
    policy: axs.ExecutionPolicy,
    activation: axs.analysis.Activation,
    *,
    size: int,
    memory_trace: str,
) -> dict[str, Any]:
    plan = _observer_plan(args, policy, activation, size)
    runner = axs.Runner()
    cold = _run_scale_profiled(
        output, f"scale_{size}_cold", runner, plan, activation, memory_trace=memory_trace
    )
    warm = _run_scale_profiled(
        output, f"scale_{size}_warm", runner, plan, activation, memory_trace=memory_trace
    )
    _require_equal(cold["signature"], warm["signature"], f"scale {size} cold/warm")
    return {"n_axons": size, "cold": cold, "warm": warm, "passed": True}


def _run_scale_profiled(
    output: Path,
    label: str,
    runner: axs.Runner,
    plan: axs.SimulationPlan,
    activation: axs.analysis.Activation,
    *,
    memory_trace: str,
) -> dict[str, Any]:
    run_dir = output / "profiles" / label
    start = time.perf_counter()
    with axs.benchmark(
        run_dir,
        print_summary=False,
        save=True,
        sync_device=True,
        record_shapes=True,
        memory_trace=memory_trace,
        profile=False,
        profile_runtime="auto",
        profile_create_perfetto=False,
        jax_device_memory_profile=False,
    ):
        result = runner.run(plan)
    values = tuple(
        bool(value)
        for value in _activation_observations_from_pool_result(result, activation)
    )
    methods = Counter(str(view.diagnostics["dispatch_method"]) for view in result)
    return {
        "label": label,
        "wall_ms": (time.perf_counter() - start) * 1000.0,
        "stages_ms": _stage_totals(run_dir),
        "signature": {
            "rows": len(values),
            "activated_count": int(sum(values)),
            "first": list(values[:8]),
            "last": list(values[-8:]),
            "dispatch_methods": dict(sorted(methods.items())),
        },
        "cache_counts": _cache_counts(runner),
    }


def _study_signature(result: axs.StudyResult, activation: axs.analysis.Activation) -> dict[str, Any]:
    simple = result["simple"]
    mixed = result["mixed"]
    numeric_axis = result["numeric_axis"]
    sweep = result["sweep"]
    threshold = result["threshold"]
    return {
        "keys": list(result.keys),
        "simple_voltage": _voltage_signature(simple),
        "mixed_voltage": _voltage_signature(mixed),
        "numeric_axis_activation": np.asarray(
            _activation_observations_from_pool_result(numeric_axis, activation),
            dtype=bool,
        ).tolist(),
        "sweep_activation": np.asarray(sweep.observations, dtype=bool).tolist(),
        "threshold": {
            "status": [str(item) for item in threshold.status],
            "threshold_uA": _float_list(threshold.threshold_uA),
            "tested_uA": [_float_list(values) for values in threshold.tested_uA],
        },
    }


def _voltage_signature(result: Any) -> list[dict[str, Any]]:
    rows = []
    for view in result:
        values = np.asarray(view.voltage_values(unit=axs.mV), dtype=float)
        rows.append(
            {
                "shape": list(values.shape),
                "minimum_mV": round(float(np.min(values)), 7),
                "maximum_mV": round(float(np.max(values)), 7),
                "final_mV": round(float(values.reshape(-1)[-1]), 7),
                "sum_mV": round(float(np.sum(values)), 7),
                "dispatch_method": str(view.diagnostics["dispatch_method"]),
            }
        )
    return rows


def _validate_cancellation(study: axs.StudyPlan) -> dict[str, Any]:
    token = axs.CancellationToken()
    token.cancel()
    try:
        axs.Runner().run(study, cancellation=token)
    except axs.PlanCancelledError as exc:
        pending = list(exc.pending_keys)
        expected = [task.key for task in study.ordered_tasks()]
        if pending != expected:
            raise AssertionError(f"unexpected cancellation pending keys: {pending}")
        return {"passed": True, "pending_keys": pending}
    raise AssertionError("pre-cancelled study executed instead of raising PlanCancelledError")


def _validate_cache_invalidation(
    args: argparse.Namespace,
    policy: axs.ExecutionPolicy,
) -> dict[str, Any]:
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=7,
    )
    row = axs.AxonInstance(axon)
    plan = axs.AxonSimulation(
        row,
        duration=float(args.duration_ms) * axs.ms,
        dt=float(args.dt_ms) * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        execution_policy=policy,
    ).plan()
    runner = axs.Runner()
    first = runner._dispatch_plan(plan)
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    row.add_extracellular_stimulation(
        stimulation=axs.analytical.point_source_stimulation(
            axs.analytical.PointSourceElectrode(
                x=50.0 * axs.um,
                y=20.0 * axs.um,
                z=0.0 * axs.um,
            ),
            positions,
            sigma=0.3 * axs.S_per_m,
            stimulus=axs.Stimulus.constant(0.0 * axs.uA),
        )
    )
    second = runner._dispatch_plan(plan)
    if second is first:
        raise AssertionError("dispatch plan cache ignored a structural row mutation")
    return {"passed": True, "cache_counts": _cache_counts(runner)}


def _cache_counts(runner: axs.Runner) -> dict[str, int]:
    return {
        "populations": len(runner._populations),
        "dispatch_plans": len(runner._dispatch_plans),
        "prepared_cohorts": len(runner._prepared_cohorts),
    }


def _plan_summary(study: axs.StudyPlan) -> dict[str, Any]:
    return {
        "name": study.name,
        "expected_rows": study.expected_rows,
        "tasks": [
            {
                "key": task.key,
                "kind": task.plan.plan_kind,
                "expected_rows": task.plan.expected_rows,
                "depends_on": list(task.depends_on),
            }
            for task in study.ordered_tasks()
        ],
    }


def _stage_totals(run_dir: Path) -> dict[str, float]:
    totals = {name: 0.0 for name in INTERESTING_STAGES}
    path = run_dir / "summary.csv"
    if not path.exists():
        return totals
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("name", ""))
            if name in totals:
                totals[name] = float(row.get("total_ms") or 0.0)
    return totals


def _parse_scales(value: str | None, *, preset: str) -> tuple[int, ...]:
    raw = value or ("8,32" if preset == "quick" else "1024,4096")
    scales = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not scales or any(item < 1 for item in scales):
        raise SystemExit("--scales must contain positive integers.")
    return scales


def _manifest(args: argparse.Namespace, scales: tuple[int, ...]) -> dict[str, Any]:
    return {
        "script": "runner_plan_validation",
        "preset": args.preset,
        "platform": args.platform,
        "scales": list(scales),
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "memory_trace": args.memory_trace,
    }


def _environment(requested_platform: str) -> dict[str, Any]:
    import jax

    devices = [
        {"platform": device.platform, "kind": device.device_kind, "id": int(device.id)}
        for device in jax.devices()
    ]
    if requested_platform == "gpu" and not any(item["platform"] == "gpu" for item in devices):
        raise RuntimeError(f"GPU validation requested but JAX devices are {devices}")
    return {
        "python": sys.version.split()[0],
        "system": platform_module.platform(),
        "jax_version": jax.__version__,
        "jax_devices": devices,
    }


def _write_cases(output: Path, args: argparse.Namespace, scales: tuple[int, ...]) -> None:
    rows = [
        {"case": "study_cold_warm_clear", "n_axons": 7, "platform": args.platform},
        {"case": "cancellation", "n_axons": 0, "platform": args.platform},
        {"case": "cache_invalidation", "n_axons": 1, "platform": args.platform},
    ]
    rows.extend(
        {"case": "population_scale", "n_axons": size, "platform": args.platform}
        for size in scales
    )
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "n_axons", "platform"))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_table(
    output: Path,
    study_runs: Any,
    scale_rows: Any,
) -> None:
    rows = []
    for run in study_runs:
        rows.append(_flat_run_row("study", run))
    for scale in scale_rows:
        rows.append(_flat_run_row(f"scale_{scale['n_axons']}", scale["cold"]))
        rows.append(_flat_run_row(f"scale_{scale['n_axons']}", scale["warm"]))
    fields = ("case", "label", "wall_ms", *[f"{name}_ms" for name in INTERESTING_STAGES])
    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flat_run_row(case: str, run: dict[str, Any]) -> dict[str, Any]:
    row = {"case": case, "label": run["label"], "wall_ms": run["wall_ms"]}
    row.update({f"{name}_ms": value for name, value in run["stages_ms"].items()})
    return row


def _write_report(output: Path, validation: dict[str, Any]) -> None:
    lines = [
        "# Runner Plan Validation",
        "",
        f"- status: `{validation.get('status', 'unknown')}`",
        f"- platform: `{validation.get('platform', 'unknown')}`",
        f"- preset: `{validation.get('preset', 'unknown')}`",
        f"- wall: `{float(validation.get('wall_s', 0.0)):.3f} s`",
        "",
        "| case | wall ms | dispatch ms | prepare ms | enqueue ms | wait ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    runs = list(validation.get("runs", ()))
    for scale in validation.get("scale_validation", ()):
        runs.extend((scale["cold"], scale["warm"]))
    for run in runs:
        stages = run["stages_ms"]
        lines.append(
            f"| {run['label']} | {run['wall_ms']:.3f} | "
            f"{stages['dispatch.build_plan']:.3f} | {stages['runtime.prepare']:.3f} | "
            f"{stages['kernel.enqueue']:.3f} | {stages['kernel.wait']:.3f} |"
        )
    if validation.get("error"):
        lines.extend(("", f"Error: `{validation['error']}`"))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float_list(values: Any) -> list[float | None]:
    result = []
    for value in np.asarray(values, dtype=float).reshape(-1):
        result.append(None if not np.isfinite(value) else round(float(value), 7))
    return result


def _require_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise AssertionError(f"{label} signatures differ")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
