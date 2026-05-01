from __future__ import annotations

import csv
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from axonscope.simresult import SimResult


AxonFactory = Callable[[], Any]
SolverFactory = Callable[[], Any]


@dataclass(frozen=True)
class BenchmarkComparisonMetric:
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    relative_delta: float | None
    threshold: float | None
    status: str


@dataclass(frozen=True)
class BenchmarkComparisonRow:
    case_name: str
    solver_name: str
    metrics: tuple[BenchmarkComparisonMetric, ...]
    status: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SolverBenchmarkCase:
    """One reproducible solver workload."""

    name: str
    build_axon: AxonFactory
    tsim_ms: float
    dt_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimingStats:
    repeats: int
    mean_s: float
    median_s: float
    min_s: float
    max_s: float
    std_s: float

    @classmethod
    def from_samples(cls, samples_s: Iterable[float]) -> "TimingStats":
        samples = [float(x) for x in samples_s]
        if not samples:
            raise ValueError("TimingStats requires at least one sample.")
        return cls(
            repeats=len(samples),
            mean_s=float(statistics.fmean(samples)),
            median_s=float(statistics.median(samples)),
            min_s=float(min(samples)),
            max_s=float(max(samples)),
            std_s=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        )


@dataclass(frozen=True)
class SolverBenchmarkResult:
    case_name: str
    solver_name: str
    tsim_ms: float
    dt_ms: float
    construction: TimingStats
    first_solve_s: float
    warm_solve: TimingStats
    output: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rss_before_mb: float | None = None
    rss_after_first_solve_mb: float | None = None

    @property
    def rss_first_solve_delta_mb(self) -> float | None:
        if self.rss_before_mb is None or self.rss_after_first_solve_mb is None:
            return None
        return float(self.rss_after_first_solve_mb - self.rss_before_mb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "solver_name": self.solver_name,
            "tsim_ms": self.tsim_ms,
            "dt_ms": self.dt_ms,
            "construction": self.construction.__dict__,
            "first_solve_s": self.first_solve_s,
            "warm_solve": self.warm_solve.__dict__,
            "rss_before_mb": self.rss_before_mb,
            "rss_after_first_solve_mb": self.rss_after_first_solve_mb,
            "rss_first_solve_delta_mb": self.rss_first_solve_delta_mb,
            "output": dict(self.output),
            "metadata": dict(self.metadata),
        }


def default_solver_factories() -> dict[str, SolverFactory]:
    from axonscope.solvers import CrankNicholson

    return {"crank_nicholson": CrankNicholson}


def default_solver_benchmark_cases() -> dict[str, SolverBenchmarkCase]:
    from axonscope.axons import HodgkinHuxley, MRG, RattayAberham, Schild97
    from axonscope.electrodes import PointSourceElectrode
    from axonscope.stimulus import Stimulus

    def hh_intracellular_small():
        length_um = 500.0
        axon = HodgkinHuxley(L=length_um, d=0.5, Nx=41, celsius=6.3)
        axon.insert_I_Clamp(
            position=length_um / 2.0,
            stimulus=Stimulus.pulse(start=1.0, duration=0.5, amplitude=2.0),
        )
        return axon

    def rattay_intracellular_small():
        length_um = 1000.0
        axon = RattayAberham(L=length_um, d=0.6, Nx=81, celsius=37.0)
        axon.insert_I_Clamp(
            position=length_um / 2.0,
            stimulus=Stimulus.pulse(start=1.0, duration=0.5, amplitude=1.0),
        )
        return axon

    def schild97_intracellular_small():
        length_um = 1200.0
        axon = Schild97(L=length_um, d=0.8, Nx=31)
        axon.insert_I_Clamp(
            position=length_um / 2.0,
            stimulus=Stimulus.pulse(start=1.0, duration=0.5, amplitude=0.8),
        )
        return axon

    def mrg_extracellular_small():
        axon = MRG(d=10.0, nodes=5)
        x0_m = float(np.asarray(axon.x)[axon.Nx // 2]) * 1e-6
        electrode = PointSourceElectrode(
            x0_m=x0_m,
            y0_m=100e-6,
            z0_m=0.0,
            sigma_S_m=0.2,
        )
        stimulus = Stimulus.biphasic(
            start=0.5,
            cathodic_amplitude=80e-6,
            cathodic_duration=0.05,
            anodic_amplitude=20e-6,
            interphase=0.02,
        )
        axon.add_extracellular_ctx(electrode, stimulus, replace=True)
        return axon

    return {
        "hh_intracellular_small": SolverBenchmarkCase(
            name="hh_intracellular_small",
            build_axon=hh_intracellular_small,
            tsim_ms=3.0,
            dt_ms=0.02,
            metadata={"model": "HodgkinHuxley", "stimulation": "intracellular", "Nx": 41},
        ),
        "rattay_intracellular_small": SolverBenchmarkCase(
            name="rattay_intracellular_small",
            build_axon=rattay_intracellular_small,
            tsim_ms=4.0,
            dt_ms=0.02,
            metadata={"model": "RattayAberham", "stimulation": "intracellular", "Nx": 81},
        ),
        "schild97_intracellular_small": SolverBenchmarkCase(
            name="schild97_intracellular_small",
            build_axon=schild97_intracellular_small,
            tsim_ms=4.0,
            dt_ms=0.02,
            metadata={"model": "Schild97", "stimulation": "intracellular", "Nx": 31},
        ),
        "mrg_extracellular_small": SolverBenchmarkCase(
            name="mrg_extracellular_small",
            build_axon=mrg_extracellular_small,
            tsim_ms=2.0,
            dt_ms=0.01,
            metadata={"model": "MRG", "stimulation": "extracellular", "nodes": 5},
        ),
    }


def run_solver_benchmark_case(
    case: SolverBenchmarkCase,
    solver_factory: SolverFactory,
    *,
    repeats: int = 3,
    warmups: int = 1,
    solve_kwargs: Mapping[str, Any] | None = None,
) -> SolverBenchmarkResult:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")
    if warmups < 0:
        raise ValueError(f"warmups must be >= 0, got {warmups}.")

    kwargs = dict(solve_kwargs or {})
    construction_times = [_time_call(case.build_axon)[0] for _ in range(repeats)]

    axon = case.build_axon()
    solver = solver_factory()
    solver_name = solver.__class__.__name__
    rss_before = _rss_mb()
    first_solve_s, first_result = _time_solve(
        solver,
        axon,
        tsim_ms=case.tsim_ms,
        dt_ms=case.dt_ms,
        solve_kwargs=kwargs,
    )
    rss_after = _rss_mb()
    output = summarize_sim_result(first_result)

    for _ in range(warmups):
        warmup_axon = case.build_axon()
        warmup_solver = solver_factory()
        _time_solve(
            warmup_solver,
            warmup_axon,
            tsim_ms=case.tsim_ms,
            dt_ms=case.dt_ms,
            solve_kwargs=kwargs,
        )

    warm_solve_times = []
    for _ in range(repeats):
        measured_axon = case.build_axon()
        measured_solver = solver_factory()
        elapsed_s, _ = _time_solve(
            measured_solver,
            measured_axon,
            tsim_ms=case.tsim_ms,
            dt_ms=case.dt_ms,
            solve_kwargs=kwargs,
        )
        warm_solve_times.append(elapsed_s)

    return SolverBenchmarkResult(
        case_name=case.name,
        solver_name=solver_name,
        tsim_ms=float(case.tsim_ms),
        dt_ms=float(case.dt_ms),
        construction=TimingStats.from_samples(construction_times),
        first_solve_s=float(first_solve_s),
        warm_solve=TimingStats.from_samples(warm_solve_times),
        output=output,
        metadata=case.metadata,
        rss_before_mb=rss_before,
        rss_after_first_solve_mb=rss_after,
    )


def run_solver_benchmark_suite(
    cases: Iterable[SolverBenchmarkCase],
    solver_factories: Mapping[str, SolverFactory],
    *,
    repeats: int = 3,
    warmups: int = 1,
    solve_kwargs: Mapping[str, Any] | None = None,
) -> list[SolverBenchmarkResult]:
    results: list[SolverBenchmarkResult] = []
    for case in cases:
        for solver_factory in solver_factories.values():
            results.append(
                run_solver_benchmark_case(
                    case,
                    solver_factory,
                    repeats=repeats,
                    warmups=warmups,
                    solve_kwargs=solve_kwargs,
                )
            )
    return results


def write_benchmark_results(
    results: Iterable[SolverBenchmarkResult],
    out_dir: Path,
    *,
    prefix: str = "solver_benchmark",
    run_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    result_list = list(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"

    json_payload = benchmark_results_document(result_list, run_metadata=run_metadata)
    json_path.write_text(json.dumps(_jsonable(json_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [_flatten_result(result) for result in result_list]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def benchmark_results_document(
    results: Iterable[SolverBenchmarkResult],
    *,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = collect_benchmark_metadata()
    if run_metadata:
        metadata.update(dict(run_metadata))
    return {
        "schema_version": 1,
        "metadata": metadata,
        "results": [result.to_dict() for result in results],
    }


def collect_benchmark_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "git": _git_metadata(),
    }
    try:
        import jax

        metadata["jax"] = jax.__version__
        metadata["jax_devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:
        metadata["jax_error"] = str(exc)
    return metadata


def load_benchmark_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return payload["results"], metadata
    raise ValueError(f"Unsupported benchmark result JSON schema in {path}.")


def compare_benchmark_results(
    baseline_results: Iterable[Mapping[str, Any]],
    current_results: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> list[BenchmarkComparisonRow]:
    thresholds_map = {
        "construction.mean_s": 0.15,
        "first_solve_s": 0.20,
        "warm_solve.mean_s": 0.10,
        "rss_first_solve_delta_mb": 0.15,
    }
    if thresholds is not None:
        thresholds_map.update(thresholds)

    baseline_by_key = {_result_key(result): dict(result) for result in baseline_results}
    current_by_key = {_result_key(result): dict(result) for result in current_results}
    rows: list[BenchmarkComparisonRow] = []

    for key in sorted(set(baseline_by_key) | set(current_by_key)):
        case_name, solver_name = key
        baseline = baseline_by_key.get(key)
        current = current_by_key.get(key)
        if baseline is None:
            rows.append(
                BenchmarkComparisonRow(
                    case_name=case_name,
                    solver_name=solver_name,
                    metrics=(),
                    status="missing_baseline",
                    notes=("Result exists only in current run.",),
                )
            )
            continue
        if current is None:
            rows.append(
                BenchmarkComparisonRow(
                    case_name=case_name,
                    solver_name=solver_name,
                    metrics=(),
                    status="missing_current",
                    notes=("Result exists only in baseline run.",),
                )
            )
            continue

        metrics = tuple(
            _compare_metric(
                metric_name,
                baseline,
                current,
                threshold=threshold,
            )
            for metric_name, threshold in thresholds_map.items()
        )
        notes = tuple(_output_guard_notes(baseline, current))
        status = "regression" if any(metric.status == "regression" for metric in metrics) else "ok"
        if notes and status == "ok":
            status = "changed_output"

        rows.append(
            BenchmarkComparisonRow(
                case_name=case_name,
                solver_name=solver_name,
                metrics=metrics,
                status=status,
                notes=notes,
            )
        )
    return rows


def summarize_sim_result(result: SimResult | Any) -> dict[str, Any]:
    vm = np.asarray(result.Vm, dtype=float)
    t = np.asarray(result.t, dtype=float)
    summary = {
        "vm_shape": tuple(int(x) for x in vm.shape),
        "t_size": int(t.size),
        "vm_min_mV": float(np.min(vm)),
        "vm_max_mV": float(np.max(vm)),
        "vm_mean_mV": float(np.mean(vm)),
        "t_start_ms": float(t[0]) if t.size else None,
        "t_stop_ms": float(t[-1]) if t.size else None,
    }
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        output_mode = metadata.get("output_mode")
        if output_mode is not None:
            summary["output_mode"] = str(output_mode)
        probe_indices = metadata.get("probe_indices")
        if probe_indices is not None:
            summary["probe_indices"] = tuple(int(i) for i in probe_indices)
    return summary


def _time_call(func: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    value = func()
    elapsed = time.perf_counter() - start
    return float(elapsed), value


def _time_solve(
    solver: Any,
    axon: Any,
    *,
    tsim_ms: float,
    dt_ms: float,
    solve_kwargs: Mapping[str, Any],
) -> tuple[float, Any]:
    accepted_kwargs = _accepted_solve_kwargs(solver, solve_kwargs)
    start = time.perf_counter()
    result = solver.solve(axon, tsim=tsim_ms, dt=dt_ms, **accepted_kwargs)
    _block_until_ready(result)
    elapsed = time.perf_counter() - start
    return float(elapsed), result


def _accepted_solve_kwargs(solver: Any, solve_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if not solve_kwargs:
        return {}
    signature = inspect.signature(solver.solve)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(solve_kwargs)
    return {key: value for key, value in solve_kwargs.items() if key in parameters}


def _block_until_ready(value: Any) -> None:
    for attr in ("Vm", "t"):
        arr = getattr(value, attr, None)
        if hasattr(arr, "block_until_ready"):
            arr.block_until_ready()
    recordings = getattr(value, "recordings", None)
    if isinstance(recordings, Mapping):
        for group in recordings.values():
            if isinstance(group, Mapping):
                for arr in group.values():
                    if hasattr(arr, "block_until_ready"):
                        arr.block_until_ready()
    diagnostics = getattr(value, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        for arr in diagnostics.values():
            if hasattr(arr, "block_until_ready"):
                arr.block_until_ready()


def _rss_mb() -> float | None:
    try:
        import os

        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return None


def _git_metadata() -> dict[str, Any]:
    return {
        "sha": _run_git(["rev-parse", "--short", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": _run_git_dirty(),
    }


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _run_git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def _flatten_result(result: SolverBenchmarkResult) -> dict[str, Any]:
    payload = result.to_dict()
    row: dict[str, Any] = {
        "case_name": payload["case_name"],
        "solver_name": payload["solver_name"],
        "tsim_ms": payload["tsim_ms"],
        "dt_ms": payload["dt_ms"],
        "first_solve_s": payload["first_solve_s"],
        "rss_before_mb": payload["rss_before_mb"],
        "rss_after_first_solve_mb": payload["rss_after_first_solve_mb"],
        "rss_first_solve_delta_mb": payload["rss_first_solve_delta_mb"],
    }
    for prefix in ("construction", "warm_solve", "output", "metadata"):
        values = payload[prefix]
        if isinstance(values, Mapping):
            for key, value in values.items():
                row[f"{prefix}.{key}"] = _jsonable(value)
    return row


def _result_key(result: Mapping[str, Any]) -> tuple[str, str]:
    return str(result.get("case_name", "")), str(result.get("solver_name", ""))


def _compare_metric(
    metric: str,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    threshold: float | None,
) -> BenchmarkComparisonMetric:
    baseline_value = _nested_float(baseline, metric)
    current_value = _nested_float(current, metric)
    if baseline_value is None or current_value is None:
        return BenchmarkComparisonMetric(
            metric=metric,
            baseline=baseline_value,
            current=current_value,
            delta=None,
            relative_delta=None,
            threshold=threshold,
            status="missing",
        )

    delta = current_value - baseline_value
    if baseline_value == 0.0:
        relative_delta = 0.0 if current_value == 0.0 else float("inf")
    else:
        relative_delta = delta / abs(baseline_value)
    status = "regression" if threshold is not None and relative_delta > threshold else "ok"
    return BenchmarkComparisonMetric(
        metric=metric,
        baseline=baseline_value,
        current=current_value,
        delta=delta,
        relative_delta=relative_delta,
        threshold=threshold,
        status=status,
    )


def _nested_float(data: Mapping[str, Any], dotted_key: str) -> float | None:
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _output_guard_notes(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    baseline_output = baseline.get("output", {})
    current_output = current.get("output", {})
    if not isinstance(baseline_output, Mapping) or not isinstance(current_output, Mapping):
        return []

    notes: list[str] = []
    if baseline_output.get("vm_shape") != current_output.get("vm_shape"):
        notes.append(f"Vm shape changed: {baseline_output.get('vm_shape')} -> {current_output.get('vm_shape')}")
    for name in ("vm_min_mV", "vm_max_mV", "vm_mean_mV"):
        baseline_value = _nested_float(baseline_output, name)
        current_value = _nested_float(current_output, name)
        if baseline_value is None or current_value is None:
            continue
        if abs(current_value - baseline_value) > 1e-6:
            notes.append(f"{name} changed: {baseline_value:.6g} -> {current_value:.6g}")
    return notes


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
