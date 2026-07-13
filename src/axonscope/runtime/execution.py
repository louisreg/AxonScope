"""Runtime execution adapters used by public simulation orchestration.

This module is the runtime boundary for public simulation entry points,
estimates, and inspection. It keeps concrete JAX modules out of public
orchestration modules while still centralizing the currently supported runtime
route.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from axonscope.recording import Recording, RecordingPlan
from axonscope.runtime.output_contract import (
    observer_output_label,
    observers_are_vm_raster_compatible,
    vm_raster_definitions,
)
from axonscope.solvers import BatchOptions

if TYPE_CHECKING:
    from axonscope.runtime import ExecutionPolicy


@dataclass(frozen=True)
class CableSolverRoute:
    """Resolved solver route for one cable family."""

    cable: str
    requested: str | None
    runtime_route: str | None
    internal: bool = False
    options: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class RuntimeSolverRoute:
    """Runtime-owned solver route summary for reporting and inspection."""

    runtime: str | None
    platform: str | None
    engine_name: str | None
    single_cable: CableSolverRoute | None
    double_cable: CableSolverRoute | None

    def for_cable(self, cable_mode: str) -> CableSolverRoute | None:
        """Return the route for a dispatch cable mode."""

        normalized = str(cable_mode).replace("-", "_")
        if normalized in {"single", "single_cable"}:
            return self.single_cable
        if normalized in {"double", "double_cable"}:
            return self.double_cable
        raise ValueError(f"Unsupported cable mode: {cable_mode!r}.")


def execution_context(
    policy: ExecutionPolicy | None,
    *,
    instances: Sequence[Any],
):
    """Return the execution context for the currently supported runtime."""

    from axonscope.runtime.jax.policy.execution import jax_execution_context

    return jax_execution_context(policy, instances=instances)


def batch_options_from_recording(
    recording: Recording | RecordingPlan | None,
    *,
    batch_options: BatchOptions | None,
) -> BatchOptions | None:
    """Lower a public recording request to runtime batch options."""

    from axonscope.runtime.recording import (
        batch_options_from_recording as runtime_batch_options_from_recording,
    )

    return runtime_batch_options_from_recording(
        recording,
        batch_options=batch_options,
    )


def benchmark_lower_recording_options(
    group: Any,
    batch_options: BatchOptions,
    *,
    observers: tuple[Any, ...] | None,
) -> BatchOptions:
    """Return runtime recording options for host-side estimates and inspection."""

    from axonscope.runtime.recording import (
        lower_batch_recording_options,
    )

    return lower_batch_recording_options(
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

    return observer_output_label(observers, recording_mode=recording_mode)


def benchmark_observers_are_vm_raster_compatible(
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether observers can use the compact runtime VmRaster route."""

    return observers_are_vm_raster_compatible(observers)


def benchmark_vm_raster_definitions(
    observers: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    """Return observer definitions supported by runtime VmRaster lowering."""

    return vm_raster_definitions(observers)


def benchmark_plan_input_lowering(
    *,
    group_mode: str,
    axons: Sequence[Any],
    stimulation_rows: Sequence[tuple[Any, ...]],
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
):
    """Return runtime input-lowering formats without materializing arrays."""

    from axonscope.runtime.jax.benchmarking.profile import (
        benchmark_plan_input_lowering as jax_benchmark_plan_input_lowering,
    )

    return jax_benchmark_plan_input_lowering(
        group_mode=group_mode,
        axons=axons,
        stimulation_rows=stimulation_rows,
        kernel_options=kernel_options,
        observers=observers,
    )


def benchmark_membrane_output_names(
    model: Any,
    method_name: str,
) -> tuple[str, ...]:
    """Return runtime membrane output names for estimate-only reporting."""

    from axonscope.runtime.jax.benchmarking.profile import (
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
        from axonscope.runtime.jax.benchmarking.profile import benchmark_profile_start as start

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
        from axonscope.runtime.jax.benchmarking.profile import (
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
        from axonscope.runtime.jax.benchmarking.profile import (
            benchmark_save_device_memory_profile as save,
        )

        return save(output_path)
    raise ValueError(f"Unsupported benchmark profile runtime: {runtime!r}.")


def benchmark_device_memory_snapshot(
    *,
    runtime: str = "auto",
) -> dict[str, Any]:
    """Return a concrete-runtime device-memory snapshot."""

    resolved = _resolve_benchmark_profile_runtime(runtime)
    if resolved is None:
        return {"available": False}
    if resolved == "jax":
        from axonscope.runtime.jax.benchmarking.memory import (
            benchmark_device_memory_snapshot as snapshot,
        )

        return snapshot()
    raise ValueError(f"Unsupported benchmark memory runtime: {runtime!r}.")


def _resolve_benchmark_profile_runtime(runtime: str) -> str | None:
    normalized = str(runtime).lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        return "jax"
    if normalized == "jax":
        return "jax"
    raise ValueError("benchmark profile runtime must be one of: auto, jax, none.")


def solver_route_from_execution_policy(
    policy: ExecutionPolicy | None,
) -> RuntimeSolverRoute | None:
    """Return the effective runtime solver route for reporting."""

    if policy is None:
        return None
    from axonscope.runtime.jax.policy.execution import jax_solver_engine_for_policy

    solver_engine = jax_solver_engine_for_policy(policy)
    if solver_engine is None:
        return None
    solver_policy = policy.solver_policy
    return RuntimeSolverRoute(
        runtime="jax",
        platform=solver_engine.platform,
        engine_name=solver_engine.name,
        single_cable=CableSolverRoute(
            cable="single_cable",
            requested=_solver_request_label(solver_policy.single_cable),
            runtime_route=solver_engine.single_cable_solver,
        ),
        double_cable=CableSolverRoute(
            cable="double_cable",
            requested=_solver_request_label(solver_policy.double_cable),
            runtime_route=solver_engine.double_cable_block_solver,
            internal=solver_engine.allow_internal_double_cable_block_solver,
            options=_double_cable_solver_options(
                solver_policy.double_cable,
                tiled_thomas_block_b=solver_engine.tiled_thomas_block_b,
            ),
        ),
    )


def _solver_request_label(request: Any | None) -> str:
    if request is None:
        return "auto"
    kind = getattr(request, "kind", None)
    value = getattr(kind, "value", None)
    if value is not None:
        return str(value)
    return str(kind if kind is not None else request)


def _double_cable_solver_options(
    request: Any | None,
    *,
    tiled_thomas_block_b: int | None,
) -> tuple[tuple[str, Any], ...]:
    options: list[tuple[str, Any]] = []
    request_label = _solver_request_label(request)
    if request_label == "tiled_thomas" and tiled_thomas_block_b is not None:
        options.append(("block_b", int(tiled_thomas_block_b)))
    return tuple(options)


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
    recording_plan: RecordingPlan | None = None,
    progress_callback: Any = None,
    runtime_context: Any | None = None,
):
    """Execute a prepared dispatch group through the active concrete runtime."""

    pending = enqueue_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        observers=observers,
        recording_plan=recording_plan,
        progress_callback=progress_callback,
        runtime_context=runtime_context,
    )
    return finalize_batch_group(pending)


def enqueue_batch_group(
    group: Any,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: Any | None,
    observers: Sequence[Any] | None,
    recording_plan: RecordingPlan | None = None,
    progress_callback: Any = None,
    runtime_context: Any | None = None,
):
    """Prepare and enqueue a batch group through the active concrete runtime."""

    from axonscope.runtime.jax.group_runner import enqueue_jax_batch_group

    return enqueue_jax_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        observers=observers,
        recording_plan=recording_plan,
        progress_callback=progress_callback,
        runtime_context=runtime_context,
    )


def finalize_batch_group(pending: Any):
    """Synchronize and assemble one pending concrete-runtime batch group."""

    from axonscope.runtime.jax.group_runner import finalize_jax_batch_group

    return finalize_jax_batch_group(pending)


__all__ = [
    "CableSolverRoute",
    "batch_options_for_execution_context",
    "batch_options_from_recording",
    "benchmark_lower_recording_options",
    "benchmark_membrane_output_names",
    "benchmark_observer_output_label",
    "benchmark_observers_are_vm_raster_compatible",
    "benchmark_device_memory_snapshot",
    "benchmark_plan_input_lowering",
    "benchmark_profile_start",
    "benchmark_profile_stop",
    "benchmark_profile_trace",
    "benchmark_save_device_memory_profile",
    "benchmark_trace_annotation",
    "benchmark_vm_raster_definitions",
    "enqueue_batch_group",
    "execution_context",
    "finalize_batch_group",
    "run_batch_group",
    "RuntimeSolverRoute",
    "solver_route_from_execution_policy",
]
