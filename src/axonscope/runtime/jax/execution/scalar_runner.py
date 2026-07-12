"""JAX scalar execution for Crank-Nicholson solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.benchmarking import (
    benchmark_array_metadata,
    benchmark_span,
    benchmark_wait,
    record_benchmark_metadata,
)
from axonscope.runtime.jax.kernels import DoubleCableKernel, SingleCableKernel
from axonscope.runtime.jax.observer_runtime import build_vm_raster_plan
from axonscope.runtime.jax.runtime import prepare_solver_runtime
from axonscope.solvers.axon_runtime import build_solver_axon
from axonscope.solvers.options import SolverOptions


@dataclass(frozen=True)
class JaxScalarSolveResult:
    """Backend output for one scalar axon solve."""

    Vm: Any | None
    t: Any
    diagnostics: dict[str, Any] | None
    recordings: dict[str, Any] | None
    observations: dict[str, Any] | None


def _scalar_wait_target(out: Any) -> Any:
    """Return a JAX/NumPy object that synchronizes a scalar kernel result."""

    if out.Vm is not None:
        return out.Vm
    if out.observations:
        first = next(iter(out.observations.values()))
        if hasattr(first, "words"):
            return first.words
        if hasattr(first, "values"):
            return first.values
        return first
    return out.t


def _recording_mode(record_voltage: bool) -> str:
    """Return the scalar kernel recording label used in benchmark metadata."""

    return "full" if record_voltage else "none"


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
    mode = "double" if is_double_cable else "single"
    nx = int(np.asarray(solver_axon.x_um).shape[0])

    with benchmark_span(
        "runtime.prepare",
        group_size=1,
        mode=mode,
        nx=nx,
        tsim_ms=duration_ms,
        dt_ms=dt_ms,
        route="scalar",
    ):
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
        record_benchmark_metadata(
            nt=runtime.grid.Nt,
            nx=runtime.membrane.Nx,
            dtype=str(runtime.membrane.dtype),
        )

    with benchmark_span(
        "inputs.positions",
        group_size=1,
        mode=mode,
        nx=runtime.membrane.Nx,
        route="scalar",
    ):
        x_positions_m = np.asarray(solver_axon.x_um, dtype=float) * 1e-6
        record_benchmark_metadata(
            **benchmark_array_metadata(
                "x_positions_m",
                x_positions_m,
                role="positions",
            ),
            extracellular_stimulation_count=len(
                tuple(getattr(simulation, "extracellular_stimulations", ()))
            ),
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

    with benchmark_span(
        "observer.plan",
        group_size=1,
        mode=mode,
        recording_mode=_recording_mode(record_voltage),
        route="scalar",
    ):
        observer_plan = build_vm_raster_plan(
            observers,
            positions_um=solver_axon.x_um,
            dtype=runtime.membrane.dtype,
        )
        record_benchmark_metadata(
            observer_count=0 if observers is None else len(observers),
            vm_raster_count=0 if observer_plan is None else observer_plan.raster_count,
        )

    with benchmark_span(
        "inputs.intracellular",
        group_size=1,
        mode=mode,
        nt=runtime.grid.Nt,
        nx=runtime.membrane.Nx,
        route="scalar",
    ):
        iinj_mid = runtime.stimulation.intracellular_current_density_mid
        metadata: dict[str, Any] = {
            "input_format": "dense_precomputed",
            "intracellular_context_count": len(
                tuple(getattr(simulation, "intracellular_contexts", ()))
            ),
        }
        if iinj_mid is not None:
            metadata.update(
                benchmark_array_metadata(
                    "iinj_mid",
                    iinj_mid,
                    role="kernel_input",
                )
            )
        record_benchmark_metadata(**metadata)

    with benchmark_span(
        "inputs.extracellular",
        group_size=1,
        mode=mode,
        nt=runtime.grid.Nt,
        nx=runtime.membrane.Nx,
        route="scalar",
    ):
        vstim_mid = runtime.stimulation.extracellular_potential_mid_mV
        metadata = {
            "input_format": (
                "dense_precomputed"
                if use_extracellular
                else "zero_no_extracellular_stimulation"
            ),
            "extracellular_stimulation_count": len(
                tuple(getattr(simulation, "extracellular_stimulations", ()))
            ),
            "has_driven_extracellular": runtime.stimulation.has_driven_extracellular,
        }
        if vstim_mid is not None:
            metadata.update(
                benchmark_array_metadata(
                    "vstim_mid",
                    vstim_mid,
                    role="kernel_input",
                )
            )
        if runtime.stimulation.extracellular_potential_initial_previous_mV is not None:
            metadata.update(
                benchmark_array_metadata(
                    "vstim_previous",
                    runtime.stimulation.extracellular_potential_initial_previous_mV,
                    role="kernel_input",
                )
            )
        record_benchmark_metadata(**metadata)

    with benchmark_span(
        "kernel.enqueue",
        group_size=1,
        mode=mode,
        recording_mode=_recording_mode(record_voltage),
        route="scalar",
    ):
        out = kernel.run(
            record_diagnostics=record_diagnostics,
            record_observables=record_observables,
            record_voltage=record_voltage,
            observers=observer_plan,
        )
        if out.Vm is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm", out.Vm, role="kernel_output")
            )

    with benchmark_span(
        "kernel.wait",
        group_size=1,
        mode=mode,
        route="scalar",
    ):
        benchmark_wait(_scalar_wait_target(out))

    return JaxScalarSolveResult(
        Vm=out.Vm,
        t=out.t,
        diagnostics=out.diagnostics,
        recordings=out.recordings,
        observations=out.observations,
    )


__all__ = ["JaxScalarSolveResult", "run_jax_crank_nicholson"]
