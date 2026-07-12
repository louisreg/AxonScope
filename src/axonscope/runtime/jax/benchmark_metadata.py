"""Benchmark metadata helpers for JAX batch execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonscope.runtime.input_contract import (
    dense_nbytes_for_shape,
    dense_shape_for_group,
)
from axonscope.runtime.input_planning import factorized_drive_count_from_rows
from axonscope.runtime.jax.input_lowering import (
    LoweredExtracellularInput,
    LoweredIntracellularInput,
)
from axonscope.runtime.jax.runtime import SolverRuntime
from axonscope.benchmarking import (
    benchmark_array_metadata,
    record_benchmark_metadata,
)
from axonscope.dispatcher.plan import DispatchGroup
from axonscope.preparation.cohort import PreparedCohort
from axonscope.solvers.options import BatchOptions


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
            **_extracellular_contract_metadata(lowered),
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
            **_extracellular_contract_metadata(lowered),
            target_nx=factorized.target_nx,
            factorized_rank=factorized.drive_count,
            nstim=factorized.drive_count,
            shared_current=factorized.shared_current,
            scaled_shared_waveform=factorized.scaled_shared_waveform,
            dense_vstim_avoided=True,
            **benchmark_array_metadata(
                "vstim_current_mid_A",
                factorized.current_mid_A,
                role="kernel_input",
            ),
            **benchmark_array_metadata(
                "vstim_footprint_mV_per_A",
                factorized.footprint_mV_per_A,
                role="kernel_input",
            ),
        )
        if factorized.current_initial_previous_A is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata(
                    "vstim_current_initial_previous_A",
                    factorized.current_initial_previous_A,
                    role="kernel_input",
                )
            )
        if factorized.current_row_scales is not None:
            record_benchmark_metadata(
                **benchmark_array_metadata(
                    "vstim_current_row_scales",
                    factorized.current_row_scales,
                    role="kernel_input",
                )
            )
    else:
        metadata = {
            "input_role": "extracellular",
            "extracellular_format": "dense",
            **_extracellular_contract_metadata(lowered),
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


def _extracellular_contract_metadata(
    lowered: LoweredExtracellularInput,
) -> dict[str, Any]:
    """Return primitive metadata for the shared extracellular input contract."""

    metadata: dict[str, Any] = {}
    if lowered.mode is not None:
        metadata["extracellular_mode"] = lowered.mode.value
    if lowered.capabilities is not None:
        metadata.update(
            lowered.capabilities.as_metadata(prefix="extracellular_capability_")
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
    batch_size = int(group.size)
    nt = int(runtime.grid.Nt)
    nx = int(group.nx)
    itemsize = int(dtype.itemsize)
    positions_nbytes = int(np.asarray(cohort.x_positions_m).nbytes)
    dense_shape = dense_shape_for_group(group=group, runtime=runtime)
    dense_nbytes = dense_nbytes_for_shape(dense_shape, dtype=dtype)
    if extracellular_format == "zero_no_extracellular_stimulation":
        vstim_mid_nbytes = 0
    elif extracellular_format == "factorized_footprint":
        factorized_rank = factorized_drive_count_from_rows(cohort.stimulations)
        vstim_mid_nbytes = (
            batch_size * factorized_rank * nt
            + batch_size * factorized_rank * nx
        ) * itemsize
    else:
        vstim_mid_nbytes = dense_nbytes
    if not include_vstim_previous:
        vstim_previous_nbytes = 0
    elif extracellular_format == "factorized_footprint":
        factorized_rank = factorized_drive_count_from_rows(cohort.stimulations)
        vstim_previous_nbytes = batch_size * factorized_rank * itemsize
    else:
        vstim_previous_nbytes = batch_size * nx * itemsize
    iinj_dense_nbytes = dense_nbytes if intracellular_format == "dense" else 0
    output_width = int(kernel_options.recording.width_for(nx))
    vm_output_nbytes = batch_size * nt * output_width * itemsize
    components = {
        "positions": positions_nbytes,
        "vstim_mid": vstim_mid_nbytes,
        "vstim_previous": vstim_previous_nbytes,
        "iinj_dense": iinj_dense_nbytes,
        "vm_output": vm_output_nbytes,
    }
    total_nbytes = int(sum(components.values()))
    capacity_bytes = _default_device_memory_capacity_bytes()
    metadata: dict[str, Any] = {
        "memory_estimate_components_nbytes": components,
        "memory_estimate_total_nbytes": total_nbytes,
        "memory_estimate_total_mib": total_nbytes / (1024**2),
        "memory_estimate_dtype": str(dtype),
        "memory_estimate_shape": {
            "batch_size": batch_size,
            "nt": nt,
            "nx": nx,
            "recording_width": output_width,
        },
        "memory_estimate_intracellular_format": intracellular_format,
        "memory_estimate_extracellular_format": extracellular_format,
    }
    if extracellular_format == "factorized_footprint":
        metadata["memory_estimate_vstim_dense_equivalent_nbytes"] = dense_nbytes
        metadata["memory_estimate_factorized_rank"] = factorized_drive_count_from_rows(
            cohort.stimulations
        )
    if capacity_bytes is not None and capacity_bytes > 0:
        metadata["device_memory_capacity_bytes"] = int(capacity_bytes)
        metadata["memory_estimate_device_fraction"] = total_nbytes / float(capacity_bytes)
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


__all__ = [
    "record_extracellular_lowering_metadata",
    "record_group_memory_estimate",
    "record_intracellular_lowering_metadata",
]
