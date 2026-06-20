"""JAX scalar execution for Crank-Nicholson solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from axonscope.axon_instance import AxonInstance
from axonscope.solvers.axon_runtime import build_solver_axon
from axonscope.solvers.kernels import DoubleCableKernel, SingleCableKernel
from axonscope.solvers.observer_runtime import build_vm_raster_plan
from axonscope.solvers.options import SolverOptions
from axonscope.solvers.runtime import prepare_solver_runtime


@dataclass(frozen=True)
class JaxScalarSolveResult:
    """Backend output for one scalar axon solve."""

    Vm: Any | None
    t: Any
    diagnostics: dict[str, Any] | None
    recordings: dict[str, Any] | None
    observations: dict[str, Any] | None


def run_jax_crank_nicholson(
    simulation: AxonInstance,
    *,
    duration_ms: float,
    dt_ms: float,
    solver_options: SolverOptions,
    record_diagnostics: bool = False,
    record_observables: bool = False,
    record_voltage: bool = True,
    observers: tuple[Any, ...] | None = None,
) -> JaxScalarSolveResult:
    """Run one axon instance through the JAX Crank-Nicholson backend."""

    use_extracellular = bool(getattr(simulation, "use_extracellular", False))
    solver_axon = build_solver_axon(simulation)
    is_double_cable = solver_axon.formulation == "double-cable"

    runtime = prepare_solver_runtime(
        simulation,
        duration_ms,
        dt_ms,
        solver_axon=solver_axon,
        include_extracellular=use_extracellular and is_double_cable,
        include_area=use_extracellular and is_double_cable,
        precompute_intracellular=True,
        precompute_extracellular=use_extracellular,
        solver_options=solver_options,
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

    observer_plan = build_vm_raster_plan(
        observers,
        positions_um=solver_axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    out = kernel.run(
        record_diagnostics=record_diagnostics,
        record_observables=record_observables,
        record_voltage=record_voltage,
        observers=observer_plan,
    )
    return JaxScalarSolveResult(
        Vm=out.Vm,
        t=out.t,
        diagnostics=out.diagnostics,
        recordings=out.recordings,
        observations=out.observations,
    )


__all__ = ["JaxScalarSolveResult", "run_jax_crank_nicholson"]
