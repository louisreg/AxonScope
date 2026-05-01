from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from axonscope.benchmarking import (
    SolverBenchmarkCase,
    TimingStats,
    run_solver_benchmark_case,
    write_benchmark_results,
)


class DummySolver:
    def solve(self, axon, tsim: float, dt: float, **_kwargs):
        nt = int(np.ceil(tsim / dt))
        vm = np.full((nt, axon.Nx), -70.0, dtype=float)
        t = (np.arange(nt, dtype=float) + 1.0) * dt
        return SimpleNamespace(Vm=vm, t=t)


class StrictDummySolver:
    def solve(self, axon, tsim: float, dt: float):
        nt = int(np.ceil(tsim / dt))
        vm = np.full((nt, axon.Nx), -60.0, dtype=float)
        t = (np.arange(nt, dtype=float) + 1.0) * dt
        return SimpleNamespace(Vm=vm, t=t)


def test_timing_stats_from_samples():
    stats = TimingStats.from_samples([1.0, 2.0, 3.0])

    assert stats.repeats == 3
    assert stats.mean_s == 2.0
    assert stats.median_s == 2.0
    assert stats.min_s == 1.0
    assert stats.max_s == 3.0


def test_run_solver_benchmark_case_with_dummy_solver():
    case = SolverBenchmarkCase(
        name="dummy",
        build_axon=lambda: SimpleNamespace(Nx=3),
        tsim_ms=0.2,
        dt_ms=0.1,
        metadata={"model": "dummy"},
    )

    result = run_solver_benchmark_case(case, DummySolver, repeats=2, warmups=1)

    assert result.case_name == "dummy"
    assert result.solver_name == "DummySolver"
    assert result.construction.repeats == 2
    assert result.warm_solve.repeats == 2
    assert result.output["vm_shape"] == (2, 3)
    assert result.output["vm_min_mV"] == -70.0


def test_run_solver_benchmark_filters_unsupported_kwargs():
    case = SolverBenchmarkCase(
        name="strict",
        build_axon=lambda: SimpleNamespace(Nx=2),
        tsim_ms=0.2,
        dt_ms=0.1,
    )

    result = run_solver_benchmark_case(
        case,
        StrictDummySolver,
        repeats=1,
        warmups=0,
        solve_kwargs={"record_observables": True},
    )

    assert result.solver_name == "StrictDummySolver"
    assert result.output["vm_min_mV"] == -60.0


def test_write_benchmark_results(tmp_path: Path):
    case = SolverBenchmarkCase(
        name="dummy",
        build_axon=lambda: SimpleNamespace(Nx=2),
        tsim_ms=0.2,
        dt_ms=0.1,
    )
    result = run_solver_benchmark_case(case, DummySolver, repeats=1, warmups=0)

    json_path, csv_path = write_benchmark_results([result], tmp_path, prefix="dummy_bench")

    assert json_path.exists()
    assert csv_path.exists()
    assert "dummy" in json_path.read_text(encoding="utf-8")
    assert "case_name" in csv_path.read_text(encoding="utf-8")
