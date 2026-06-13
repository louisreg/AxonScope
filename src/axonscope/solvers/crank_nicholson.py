"""Optimized Crank-Nicholson solver entry point."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from axonscope.axon_simulation import AxonSimulation, as_axon_simulation
from axonscope.axons.axon import Axon
from axonscope.results import SimResult

from .base import Solver
from .common import resolve_time_args
from .axon_runtime import build_solver_axon
from .kernels import DoubleCableKernel, SingleCableKernel
from .options import SolverOptions
from .runtime import prepare_solver_runtime


class CrankNicholson(Solver):
    """Crank-Nicholson/Hines solver for single- and double-cable axons."""

    def __init__(self, *, solver_options: SolverOptions | None = None) -> None:
        self.solver_options = (
            SolverOptions() if solver_options is None else solver_options
        )

    def solve(
        self,
        axon: Axon | AxonSimulation,
        tsim: Any | None = None,
        dt: Any | None = None,
        record_diagnostics: bool = False,
        record_observables: bool = False,
        *,
        duration_ms: Any | None = None,
        dt_ms: Any | None = None,
    ) -> SimResult:
        """Execute a Crank-Nicholson simulation."""

        simulation = as_axon_simulation(axon)
        duration, step = resolve_time_args(
            tsim=tsim,
            dt=dt,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )

        use_extracellular = bool(getattr(simulation, "use_extracellular", False))
        solver_axon = build_solver_axon(simulation)
        is_double_cable = solver_axon.formulation == "double-cable"

        runtime = prepare_solver_runtime(
            simulation,
            duration,
            step,
            solver_axon=solver_axon,
            include_extracellular=use_extracellular and is_double_cable,
            include_area=use_extracellular and is_double_cable,
            precompute_intracellular=True,
            precompute_extracellular=use_extracellular,
            solver_options=self.solver_options,
        )
        if use_extracellular and is_double_cable:
            kernel = DoubleCableKernel(
                runtime=runtime,
                Veinit_mV=float(getattr(simulation, "Veinit", 0.0)),
            )
        else:
            kernel = SingleCableKernel(
                runtime=runtime,
                Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
            )

        out = kernel.run(
            record_diagnostics=record_diagnostics,
            record_observables=record_observables,
        )
        return SimResult(
            simulation.axon,
            out.Vm,
            out.t,
            diagnostics=out.diagnostics,
            recordings=out.recordings,
            simulation=simulation,
        )
