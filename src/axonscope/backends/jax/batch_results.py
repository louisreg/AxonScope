"""JAX batch result assembly helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonscope.benchmarking import (
    benchmark_array_metadata,
    benchmark_span,
    record_benchmark_metadata,
)
from axonscope.backends.jax.batch_kernels import BatchKernelResult
from axonscope.dispatcher.plan import DispatchGroup, DispatchItem
from axonscope.dispatcher._records import (
    DispatchCohortRecord,
    DispatchRecord,
    DispatchRowRecord,
)
from axonscope.solvers._outputs import SolverOutput
from axonscope.solvers.options import BatchOptions


def trim_batch_kernel_result(
    out: BatchKernelResult,
    *,
    batch_size: int,
) -> BatchKernelResult:
    """Drop backend-only padded batch rows before public result assembly."""

    size = int(batch_size)
    Vm = None if out.Vm is None else out.Vm[:size]
    observations = trim_observations_batch(out.observations, batch_size=size)
    return BatchKernelResult(Vm=Vm, t=out.t, observations=observations)


def dispatch_results_from_batch(
    group: DispatchGroup,
    *,
    Vm: Any | None,
    t: Any,
    observations: dict[str, Any] | None,
    observer_definitions: tuple[Any, ...] | None,
    method: str,
    batch_options: BatchOptions,
    kernel_batch_options: BatchOptions,
) -> tuple[DispatchRecord, ...]:
    """Convert a batched solver output to compact dispatch records."""

    kernel_indices = kernel_batch_options.recording.indices_for(group.nx)
    kernel_record_indices = (
        None if kernel_indices is None else tuple(int(value) for value in kernel_indices)
    )
    with benchmark_span(
        "results.materialize_vm",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=batch_options.recording.mode,
        kernel_recording_mode=kernel_batch_options.recording.mode,
        output="none" if Vm is None else "Vm",
        method=method,
    ):
        vm_values = None if Vm is None else np.asarray(Vm)
        if vm_values is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata("Vm_host", vm_values, role="result_output"),
                retained_width=int(vm_values.shape[-1]) if vm_values.ndim >= 2 else "",
            )

    if vm_values is None and observations is not None:
        with benchmark_span(
            "results.assemble_cohort_record",
            group_id=group.group_id,
            group_size=group.size,
            recording_mode=batch_options.recording.mode,
            output="observations",
            method=method,
        ):
            return (
                DispatchCohortRecord(
                    indices=tuple(item.index for item in group.items),
                    axons=tuple(item.simulation.axon for item in group.items),
                    simulations=tuple(item.simulation for item in group.items),
                    Vm=None,
                    t=t,
                    group_id=group.group_id,
                    method=method,
                    record_indices=tuple(None for _ in group.items),
                    observations=observations,
                    group_size=group.size,
                    batch_kind=group.batch_kind,
                    geometry_shared=group.geometry_shared,
                    has_padding=group.has_padding,
                ),
            )

    observer_definition_count = 0 if observer_definitions is None else len(observer_definitions)
    with benchmark_span(
        "results.assemble_rows",
        group_id=group.group_id,
        group_size=group.size,
        recording_mode=batch_options.recording.mode,
        kernel_recording_mode=kernel_batch_options.recording.mode,
        output="Vm" if vm_values is not None else "none",
        has_observations=observations is not None,
        observer_definition_count=observer_definition_count,
        method=method,
    ):
        results = []
        posthoc_row_count = 0
        row_trim_count = 0
        for row_index, item in enumerate(group.items):
            original_nx = int(item.solver_axon.n_compartments)
            row_vm = None if vm_values is None else vm_values[row_index]
            record_indices = kernel_record_indices

            if row_vm is not None and kernel_indices is None:
                row_vm = row_vm[:, :original_nx]
                requested_indices = batch_options.recording.indices_for(original_nx)
                if requested_indices is not None:
                    row_vm = np.take(row_vm, np.asarray(requested_indices), axis=1)
                    record_indices = tuple(int(value) for value in requested_indices)
                    row_trim_count += 1
                else:
                    record_indices = None
            if row_vm is None:
                record_indices = None

            row_observations = observations
            observations_are_batched = row_observations is not None
            if row_observations is None and observer_definitions and row_vm is not None:
                row_observations = _posthoc_observations_for_row(
                    item,
                    row_vm=row_vm,
                    t=t,
                    record_indices=record_indices,
                    observer_definitions=observer_definitions,
                )
                observations_are_batched = False
                posthoc_row_count += 1

            results.append(
                DispatchRowRecord(
                    index=item.index,
                    axon=item.simulation.axon,
                    simulation=item.simulation,
                    Vm=row_vm,
                    t=t,
                    group_id=group.group_id,
                    method=method,
                    record_indices=record_indices,
                    observations=row_observations,
                    observations_are_batched=observations_are_batched,
                    group_size=group.size,
                    batch_kind=group.batch_kind,
                    geometry_shared=group.geometry_shared,
                    has_padding=group.has_padding,
                )
            )
        record_benchmark_metadata(
            row_count=len(results),
            row_trim_count=row_trim_count,
            posthoc_row_count=posthoc_row_count,
        )
        return tuple(results)


def trim_observations_batch(
    observations: dict[str, Any] | None,
    *,
    batch_size: int,
) -> dict[str, Any] | None:
    if observations is None:
        return None
    return {
        name: trim_observation_batch(value, batch_size=batch_size)
        for name, value in observations.items()
    }


def trim_observation_batch(value: Any, *, batch_size: int) -> Any:
    current_size = getattr(value, "batch_size", None)
    if current_size is None or int(current_size) <= int(batch_size):
        return value
    slice_batch = getattr(value, "slice_batch", None)
    concat_batch = getattr(type(value), "concat_batch", None)
    if callable(slice_batch) and callable(concat_batch):
        return concat_batch([slice_batch(index) for index in range(int(batch_size))])
    raise TypeError(
        f"cannot trim padded observation result type {type(value).__name__}."
    )


def _posthoc_observations_for_row(
    item: DispatchItem,
    *,
    row_vm: np.ndarray,
    t: Any,
    record_indices: tuple[int, ...] | None,
    observer_definitions: tuple[Any, ...],
) -> dict[str, Any]:
    """Evaluate observers post-hoc when Vm was intentionally recorded."""

    row_result = SolverOutput(
        item.simulation.axon,
        row_vm,
        np.asarray(t),
        record_indices=record_indices,
        simulation=item.simulation,
    )
    observations = {}
    for definition in observer_definitions:
        analysis = definition.evaluate(row_result)
        observations[analysis.name] = analysis
    return observations


__all__ = [
    "dispatch_results_from_batch",
    "trim_batch_kernel_result",
    "trim_observation_batch",
    "trim_observations_batch",
]
