"""Runtime execution adapters used by public simulation orchestration.

This module is the runtime boundary for public simulation entry points,
estimates, and inspection. It keeps concrete JAX modules out of public
orchestration modules while still centralizing the currently supported runtime
route.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from axonscope.recording import Recording, RecordingPlan
from axonscope.solvers import BatchOptions

if TYPE_CHECKING:
    from axonscope.runtime import ExecutionPolicy


def execution_context(
    policy: ExecutionPolicy | None,
    *,
    instances: Sequence[Any],
):
    """Return the execution context for the currently supported runtime."""

    from axonscope.runtime.jax.execution_policy import jax_execution_context

    return jax_execution_context(policy, instances=instances)


def batch_options_from_recording(
    recording: Recording | RecordingPlan | None,
    *,
    batch_options: BatchOptions | None,
) -> BatchOptions | None:
    """Lower a public recording request to runtime batch options."""

    from axonscope.runtime.jax.recording import (
        batch_options_from_recording as jax_batch_options_from_recording,
    )

    return jax_batch_options_from_recording(recording, batch_options=batch_options)


def benchmark_lower_recording_options(
    group: Any,
    batch_options: BatchOptions,
    *,
    observers: tuple[Any, ...] | None,
) -> BatchOptions:
    """Return runtime recording options for host-side estimates and inspection."""

    from axonscope.runtime.jax.benchmark import (
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
    """Return the runtime observer output route for host-side reports."""

    from axonscope.runtime.jax.benchmark import (
        benchmark_observer_output_label as jax_benchmark_observer_output_label,
    )

    return jax_benchmark_observer_output_label(
        observers,
        recording_mode=recording_mode,
    )


def benchmark_observers_are_vm_raster_compatible(
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether observers can use the compact runtime VmRaster route."""

    from axonscope.runtime.jax.benchmark import (
        benchmark_observers_are_vm_raster_compatible as jax_vm_raster_compatible,
    )

    return jax_vm_raster_compatible(observers)


def benchmark_vm_raster_definitions(
    observers: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    """Return observer definitions supported by runtime VmRaster lowering."""

    from axonscope.runtime.jax.benchmark import (
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
    """Return runtime input-lowering formats without materializing arrays."""

    from axonscope.runtime.jax.benchmark import (
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
    """Return runtime membrane output names for estimate-only reporting."""

    from axonscope.runtime.jax.benchmark import (
        benchmark_membrane_output_names as jax_benchmark_membrane_output_names,
    )

    return jax_benchmark_membrane_output_names(model, method_name)


def benchmark_profile_start(
    runtime: str,
    output_dir: str | Path,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
) -> Any:
    """Start a runtime-owned profiler trace for benchmark instrumentation."""

    resolved = _resolve_benchmark_profile_runtime(runtime)
    if resolved is None:
        return None
    if resolved == "jax":
        from axonscope.runtime.jax.benchmark import benchmark_profile_start as start

        return start(
            output_dir,
            create_perfetto_link=create_perfetto_link,
            create_perfetto_trace=create_perfetto_trace,
        )
    raise ValueError(f"Unsupported benchmark profile runtime: {runtime!r}.")


def benchmark_profile_stop(handle: Any) -> dict[str, Any]:
    """Stop a runtime-owned profiler trace and return JSON-safe metadata."""

    if handle is None:
        return {"enabled": False}
    stop = getattr(handle, "stop", None)
    if callable(stop):
        metadata = stop()
        return dict(metadata or {})
    return {"enabled": True, "stopped": False, "error": "profile handle has no stop method"}


@contextmanager
def benchmark_profile_trace(
    runtime: str,
    output_dir: str | Path,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
):
    """Context manager for a runtime-owned profiler trace."""

    handle = benchmark_profile_start(
        runtime,
        output_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
    )
    try:
        yield Path(output_dir)
    finally:
        benchmark_profile_stop(handle)


def benchmark_trace_annotation(name: str):
    """Return a runtime-owned trace annotation context when available."""

    try:
        from axonscope.runtime.jax.benchmark import (
            benchmark_trace_annotation as jax_trace_annotation,
        )

        return jax_trace_annotation(name)
    except Exception:
        return nullcontext()


def benchmark_save_device_memory_profile(
    output_path: str | Path,
    *,
    runtime: str = "auto",
) -> dict[str, Any]:
    """Save a runtime device-memory profile and return metadata."""

    resolved = _resolve_benchmark_profile_runtime(runtime)
    if resolved is None:
        return {"enabled": False}
    if resolved == "jax":
        from axonscope.runtime.jax.benchmark import (
            benchmark_save_device_memory_profile as save,
        )

        return save(output_path)
    raise ValueError(f"Unsupported benchmark profile runtime: {runtime!r}.")


def _resolve_benchmark_profile_runtime(runtime: str) -> str | None:
    normalized = str(runtime).lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        return "jax"
    if normalized == "jax":
        return "jax"
    raise ValueError("benchmark profile runtime must be one of: auto, jax, none.")


def double_cable_block_solver_from_execution_policy(
    policy: ExecutionPolicy | None,
) -> str | None:
    """Return the effective runtime double-cable block solver for reporting."""

    if policy is None:
        return None
    from axonscope.runtime.jax.execution_policy import (
        jax_double_cable_block_solver_for_policy,
    )

    return jax_double_cable_block_solver_for_policy(policy)


def single_cable_solver_from_execution_policy(
    policy: ExecutionPolicy | None,
) -> str | None:
    """Return the effective runtime single-cable solver for reporting."""

    if policy is None:
        return None
    from axonscope.runtime.jax.execution_policy import (
        jax_single_cable_solver_for_policy,
    )

    return jax_single_cable_solver_for_policy(policy)


def batch_options_for_execution_context(
    batch_options: BatchOptions | None,
    context: Any,
) -> BatchOptions | None:
    """Return output/chunking options unchanged for the active context."""

    return batch_options


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
    """Execute a prepared dispatch group through the active concrete runtime."""

    from axonscope.runtime.jax.group_runner import run_jax_batch_group

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
    "benchmark_profile_start",
    "benchmark_profile_stop",
    "benchmark_profile_trace",
    "benchmark_save_device_memory_profile",
    "benchmark_trace_annotation",
    "benchmark_vm_raster_definitions",
    "double_cable_block_solver_from_execution_policy",
    "execution_context",
    "run_batch_group",
    "single_cable_solver_from_execution_policy",
]
