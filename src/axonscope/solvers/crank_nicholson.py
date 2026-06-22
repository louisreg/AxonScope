"""Optimized Crank-Nicholson solver entry point."""

from __future__ import annotations

from typing import Any

from axonscope.backends.jax.scalar_runner import run_jax_crank_nicholson
from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.axons.axon import Axon
from axonscope.results import SimResult
from axonscope.timebase import resolve_time_args

from .base import Solver
from .options import SolverOptions


class CrankNicholson(Solver):
    """Crank-Nicholson/Hines solver for single- and double-cable axons."""

    def __init__(self, *, solver_options: SolverOptions | None = None) -> None:
        self.solver_options = (
            SolverOptions() if solver_options is None else solver_options
        )

    def solve(
        self,
        axon: Axon | AxonInstance,
        tsim: Any | None = None,
        dt: Any | None = None,
        record_diagnostics: bool = False,
        record_observables: bool = False,
        record_voltage: bool = True,
        observers: tuple[Any, ...] | None = None,
    ) -> SimResult:
        """Execute a Crank-Nicholson simulation."""

        simulation = as_axon_instance(axon)
        duration, step = resolve_time_args(tsim=tsim, dt=dt)

        out = run_jax_crank_nicholson(
            simulation,
            duration_ms=duration,
            dt_ms=step,
            solver_options=self.solver_options,
            record_diagnostics=record_diagnostics,
            record_observables=record_observables,
            record_voltage=record_voltage,
            observers=observers,
        )
        return SimResult(
            simulation.axon,
            out.Vm,
            out.t,
            diagnostics=out.diagnostics,
            recordings=out.recordings,
            observations=out.observations,
            simulation=simulation,
        )
