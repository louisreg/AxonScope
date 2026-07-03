"""Backend execution adapters used by public simulation orchestration.

This module is the backend boundary for public simulation entry points,
estimates, and inspection. It keeps concrete JAX modules out of public
orchestration modules while still centralizing the currently supported backend
route.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

from axonscope.recording import Recording, RecordingPlan
from axonscope.solvers import BatchOptions, resolve_double_cable_block_solver

if TYPE_CHECKING:
    from axonscope.performance import ExecutionPolicy


def execution_context(
    policy: ExecutionPolicy | None,
    *,
    instances: Sequence[Any],
):
    """Return the execution context for the currently supported backend."""

    from axonscope.backends.jax.execution_policy import jax_execution_context

    return jax_execution_context(policy, instances=instances)


def batch_options_from_recording(
    recording: Recording | RecordingPlan | None,
    *,
    batch_options: BatchOptions | None,
) -> BatchOptions | None:
    """Lower a public recording request to backend batch options."""

    from axonscope.backends.jax.recording import (
        batch_options_from_recording as jax_batch_options_from_recording,
    )

    return jax_batch_options_from_recording(recording, batch_options=batch_options)


def benchmark_lower_recording_options(
    group: Any,
    batch_options: BatchOptions,
    *,
    observers: tuple[Any, ...] | None,
) -> BatchOptions:
    """Return backend recording options for host-side estimates and inspection."""

    from axonscope.backends.jax.benchmark import (
        benchmark_lower_recording_options as jax_benchmark_lower_recording_options,
    )

    return jax_benchmark_lower_recording_options(
        group,
        batch_options,
        observers=observers,
    )


def benchmark_observer_output_label(
    observers: tuple[Any, ...] | None,
    *,
    recording_mode: str,
) -> str:
    """Return the backend observer output route for host-side reports."""

    from axonscope.backends.jax.benchmark import (
        benchmark_observer_output_label as jax_benchmark_observer_output_label,
    )

    return jax_benchmark_observer_output_label(
        observers,
        recording_mode=recording_mode,
    )


def benchmark_observers_are_vm_raster_compatible(
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether observers can use the compact backend VmRaster route."""

    from axonscope.backends.jax.benchmark import (
        benchmark_observers_are_vm_raster_compatible as jax_vm_raster_compatible,
    )

    return jax_vm_raster_compatible(observers)


def benchmark_vm_raster_definitions(
    observers: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    """Return observer definitions supported by backend VmRaster lowering."""

    from axonscope.backends.jax.benchmark import (
        benchmark_vm_raster_definitions as jax_benchmark_vm_raster_definitions,
    )

    return jax_benchmark_vm_raster_definitions(observers)


def benchmark_plan_input_lowering(
    *,
    group_mode: str,
    axons: Sequence[Any],
    stimulation_rows: Sequence[tuple[Any, ...]],
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    observer_plan: bool,
):
    """Return backend input-lowering formats without materializing arrays."""

    from axonscope.backends.jax.benchmark import (
        benchmark_plan_input_lowering as jax_benchmark_plan_input_lowering,
    )

    return jax_benchmark_plan_input_lowering(
        group_mode=group_mode,
        axons=axons,
        stimulation_rows=stimulation_rows,
        kernel_options=kernel_options,
        observers=observers,
        observer_plan=observer_plan,
    )


def benchmark_membrane_output_names(
    model: Any,
    method_name: str,
) -> tuple[str, ...]:
    """Return backend membrane output names for estimate-only reporting."""

    from axonscope.backends.jax.benchmark import (
        benchmark_membrane_output_names as jax_benchmark_membrane_output_names,
    )

    return jax_benchmark_membrane_output_names(model, method_name)


def batch_options_for_execution_context(
    batch_options: BatchOptions | None,
    context: Any,
) -> BatchOptions | None:
    """Apply effective backend/device routing to batch-only solver options."""

    platform = getattr(context, "platform", None)
    if platform is None:
        return batch_options
    options = BatchOptions.full() if batch_options is None else batch_options
    if options.double_cable_block_solver != "auto":
        return options
    return replace(
        options,
        double_cable_block_solver=resolve_double_cable_block_solver(
            "auto",
            platform=platform,
        ),
    )


def run_batch_group(
    group: Any,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: Any | None,
    observers: Sequence[Any] | None,
    progress_callback: Any = None,
    backend_context: Any | None = None,
):
    """Execute a prepared dispatch group through the active concrete backend."""

    from axonscope.backends.jax.group_runner import run_jax_batch_group

    return run_jax_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        observers=observers,
        progress_callback=progress_callback,
        backend_context=backend_context,
    )


__all__ = [
    "batch_options_for_execution_context",
    "batch_options_from_recording",
    "benchmark_lower_recording_options",
    "benchmark_membrane_output_names",
    "benchmark_observer_output_label",
    "benchmark_observers_are_vm_raster_compatible",
    "benchmark_plan_input_lowering",
    "benchmark_vm_raster_definitions",
    "execution_context",
    "run_batch_group",
]
