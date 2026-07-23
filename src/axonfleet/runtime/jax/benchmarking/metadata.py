"""Benchmark metadata helpers for JAX batch execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonfleet.runtime.inputs.contracts import (
    dense_nbytes_for_shape,
    dense_shape_for_group,
)
from axonfleet.runtime.inputs.planning import factorized_drive_count_from_rows
from axonfleet.runtime.memory_estimates import estimate_runtime_group_memory
from axonfleet.runtime.jax.inputs.lowering import (
    JAX_DOUBLE_CABLE_INPUT_CONTRACT,
    JAX_SINGLE_CABLE_INPUT_CONTRACT,
    LoweredExtracellularInput,
    LoweredIntracellularInput,
)
from axonfleet.runtime.jax.types import SolverRuntime
from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.runtime.benchmarking import benchmark_array_metadata
from axonfleet.dispatcher.plan import DispatchGroup
from axonfleet.preparation.cohort import PreparedCohort
from axonfleet.solvers.options import BatchOptions


def record_intracellular_lowering_metadata(
    lowered: LoweredIntracellularInput,
    *,
    group: DispatchGroup,
    runtime: SolverRuntime,
) -> None:
    """Record benchmark metadata for the selected intracellular input format."""

    if lowered.format == "sparse_current_clamp":
        iinj_mid = lowered.midpoint
        record_benchmark_metadata(
            input_role="intracellular",
            intracellular_format="sparse_current_clamp",
            target_nx=iinj_mid.target_nx,
            max_sparse_entries=iinj_mid.max_sparse_entries,
            **benchmark_array_metadata(
                "iinj_density_mid",
                iinj_mid.density_mid,
                role="kernel_input",
            ),
            **benchmark_array_metadata(
                "iinj_indices",
                iinj_mid.indices,
                role="kernel_input",
            ),
            **benchmark_array_metadata("iinj_mask", iinj_mid.mask, role="kernel_input"),
        )
    elif lowered.format == "zero_no_intracellular_context":
        _record_zero_intracellular_metadata(group=group, runtime=runtime)
    else:
        record_benchmark_metadata(
            input_role="intracellular",
            intracellular_format="dense",
            **benchmark_array_metadata("iinj_mid", lowered.midpoint, role="kernel_input"),
        )


def record_extracellular_lowering_metadata(
    lowered: LoweredExtracellularInput,
    *,
    group: DispatchGroup,
    runtime: SolverRuntime,
) -> None:
    """Record benchmark metadata for the selected extracellular input format."""

    if lowered.format == "zero_no_extracellular_stimulation":
        dtype = np.dtype(runtime.membrane.dtype)
        skipped_shape = dense_shape_for_group(group=group, runtime=runtime)
        record_benchmark_metadata(
            input_role="extracellular",
            extracellular_format="zero_no_extracellular_stimulation",
            **_extracellular_contract_metadata(lowered, group_mode=group.mode),
            skipped_dense_vstim_shape=list(skipped_shape),
            skipped_dense_vstim_nbytes=dense_nbytes_for_shape(skipped_shape, dtype=dtype),
        )
    elif lowered.format == "factorized_footprint":
        factorized = lowered.factorized
        if factorized is None:
            raise TypeError("factorized_footprint lowering requires a factorized payload.")
        record_benchmark_metadata(
            input_role="extracellular",
            extracellular_format="factorized_footprint",
            **_extracellular_contract_metadata(lowered, group_mode=group.mode),
            target_nx=factorized.target_nx,
            factorized_rank=factorized.drive_count,
            nstim=factorized.drive_count,
            shared_current=factorized.shared_current,
            scaled_shared_waveform=factorized.scaled_shared_waveform,
            dense_vstim_avoided=True,
            **_extracellular_array_metadata(
                "vstim_current_mid_A",
                factorized.current_mid_A,
                role="kernel_input",
            ),
            **_extracellular_array_metadata(
                "vstim_footprint_mV_per_A",
                factorized.footprint_mV_per_A,
                role="kernel_input",
            ),
        )
        if factorized.current_initial_previous_A is not None:
            record_benchmark_metadata(
                **_extracellular_array_metadata(
                    "vstim_current_initial_previous_A",
                    factorized.current_initial_previous_A,
                    role="kernel_input",
                )
            )
        if factorized.current_row_scales is not None:
            record_benchmark_metadata(
                **_extracellular_array_metadata(
                    "vstim_current_row_scales",
                    factorized.current_row_scales,
                    role="kernel_input",
                )
            )
    else:
        metadata = {
            "input_role": "extracellular",
            "extracellular_format": "dense",
            **_extracellular_contract_metadata(lowered, group_mode=group.mode),
            **benchmark_array_metadata(
                "vstim_mid",
                lowered.midpoint,
                role="kernel_input",
            ),
        }
        if lowered.initial_previous is not None:
            metadata.update(
                benchmark_array_metadata(
                    "vstim_previous",
                    lowered.initial_previous,
                    role="kernel_input",
                )
            )
        if lowered.dense_fallback_reason is not None:
            metadata["dense_fallback_reason"] = lowered.dense_fallback_reason
        record_benchmark_metadata(**metadata)


def _extracellular_array_metadata(
    name: str,
    array: Any,
    *,
    role: str,
) -> dict[str, Any]:
    with benchmark_span("inputs.extracellular.metadata.array", array_name=name):
        return benchmark_array_metadata(name, array, role=role)


def _extracellular_contract_metadata(
    lowered: LoweredExtracellularInput,
    *,
    group_mode: str,
) -> dict[str, Any]:
    """Return primitive metadata for the shared extracellular input contract."""

    metadata: dict[str, Any] = {}
    if lowered.mode is not None:
        metadata["extracellular_mode"] = lowered.mode.value
    if group_mode == "single":
        contract = JAX_SINGLE_CABLE_INPUT_CONTRACT
    elif group_mode == "double":
        contract = JAX_DOUBLE_CABLE_INPUT_CONTRACT
    else:
        raise ValueError(f"Unsupported dispatch group mode: {group_mode!r}.")
    metadata.update(
        contract.extracellular.as_metadata(prefix="extracellular_capability_")
    )
    return metadata


def record_group_memory_estimate(
    *,
    group: DispatchGroup,
    runtime: SolverRuntime,
    cohort: PreparedCohort,
    kernel_options: BatchOptions,
    intracellular_format: str,
    extracellular_format: str,
    include_vstim_previous: bool,
) -> None:
    """Attach a conservative per-group memory estimate to the group span."""

    dtype = np.dtype(runtime.membrane.dtype)
    positions_nbytes = int(np.asarray(cohort.x_positions_m).nbytes)
    factorized_rank = (
        factorized_drive_count_from_rows(cohort.stimulations)
        if extracellular_format == "factorized_footprint"
        else None
    )
    estimate = estimate_runtime_group_memory(
        batch_size=group.size,
        nt=runtime.grid.Nt,
        nx=group.nx,
        dtype=dtype,
        positions_nbytes=positions_nbytes,
        recording_width=kernel_options.recording.width_for(group.nx),
        intracellular_format=intracellular_format,
        extracellular_format=extracellular_format,
        include_vstim_previous=include_vstim_previous,
        factorized_rank=factorized_rank,
    )
    capacity_bytes = _default_device_memory_capacity_bytes()
    metadata: dict[str, Any] = {
        **estimate.as_metadata(),
        "memory_estimate_intracellular_format": intracellular_format,
        "memory_estimate_extracellular_format": extracellular_format,
    }
    if capacity_bytes is not None and capacity_bytes > 0:
        metadata["device_memory_capacity_bytes"] = int(capacity_bytes)
        metadata["memory_estimate_device_fraction"] = (
            estimate.total_nbytes / float(capacity_bytes)
        )
    record_benchmark_metadata(**metadata)


def _record_zero_intracellular_metadata(
    *,
    group: DispatchGroup,
    runtime: SolverRuntime,
) -> None:
    """Record skipped dense-Iinj metadata for zero-input cohorts."""

    dtype = np.dtype(runtime.membrane.dtype)
    skipped_shape = (group.size, runtime.grid.Nt, group.nx)
    record_benchmark_metadata(
        input_role="intracellular",
        intracellular_format="zero_no_intracellular_context",
        skipped_dense_iinj_shape=list(skipped_shape),
        skipped_dense_iinj_nbytes=int(np.prod(skipped_shape)) * int(dtype.itemsize),
    )


def _default_device_memory_capacity_bytes() -> int | None:
    """Best-effort capacity for the first JAX device, when the backend exposes it."""

    try:
        import jax

        devices = jax.devices()
        if not devices:
            return None
        stats_fn = getattr(devices[0], "memory_stats", None)
        if callable(stats_fn):
            stats = stats_fn() or {}
            for key in (
                "bytes_limit",
                "device_memory_capacity",
                "memory_limit",
                "bytes_reserved",
            ):
                value = stats.get(key)
                if value is not None:
                    return int(value)
    except Exception:
        return None
    return None
