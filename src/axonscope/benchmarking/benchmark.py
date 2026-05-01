from __future__ import annotations

import csv
import inspect
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from axonscope.simresult import SimResult


AxonFactory = Callable[[], Any]
SolverFactory = Callable[[], Any]


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
) -> tuple[Path, Path]:
    result_list = list(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"

    json_payload = [_jsonable(result.to_dict()) for result in result_list]
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [_flatten_result(result) for result in result_list]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def summarize_sim_result(result: SimResult | Any) -> dict[str, Any]:
    vm = np.asarray(result.Vm, dtype=float)
    t = np.asarray(result.t, dtype=float)
    return {
        "vm_shape": tuple(int(x) for x in vm.shape),
        "t_size": int(t.size),
        "vm_min_mV": float(np.min(vm)),
        "vm_max_mV": float(np.max(vm)),
        "vm_mean_mV": float(np.mean(vm)),
        "t_start_ms": float(t[0]) if t.size else None,
        "t_stop_ms": float(t[-1]) if t.size else None,
    }


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


def _jsonable(value: Any) -> Any:
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
