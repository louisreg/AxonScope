"""Input lowering contracts for JAX batch execution.

This module owns the decision from prepared AxonScope rows to concrete kernel
input representations. Callers should not decide dense/factorized/zero input
formats directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from axonscope.runtime.input_contract import (
    ExtracellularLoweringCapabilities,
    ExtracellularLoweringMode,
    IntracellularLoweringMode,
    RuntimeInputContract,
)


if TYPE_CHECKING:
    from axonscope.axon_instance import AxonInstance
    from axonscope.runtime.jax.batch_inputs import FactorizedExtracellularPotentialBatch
    from axonscope.runtime.jax.runtime import SolverRuntime
    from axonscope.dispatcher.plan import DispatchGroup
    from axonscope.preparation.cohort import PreparedCohort
    from axonscope.solvers.options import BatchOptions


IntracellularInputFormat = Literal[
    "dense",
    "sparse_current_clamp",
    "zero_no_intracellular_context",
]
ExtracellularInputFormat = Literal[
    "dense",
    "factorized_footprint",
    "zero_no_extracellular_stimulation",
]

JAX_SINGLE_CABLE_EXTRACELLULAR_CAPABILITIES = ExtracellularLoweringCapabilities(
    cable="single-cable",
    supports_zero=True,
    supports_shared_current=True,
    supports_scaled_shared_waveform=True,
    supports_current_table=True,
    supports_dense_fallback=True,
)
JAX_DOUBLE_CABLE_EXTRACELLULAR_CAPABILITIES = ExtracellularLoweringCapabilities(
    cable="double-cable",
    supports_zero=True,
    supports_shared_current=True,
    supports_scaled_shared_waveform=True,
    supports_current_table=False,
    supports_dense_fallback=True,
    requires_initial_previous=True,
)
JAX_SINGLE_CABLE_INPUT_CONTRACT = RuntimeInputContract(
    cable="single-cable",
    intracellular_modes=frozenset(
        {
            IntracellularLoweringMode.ZERO,
            IntracellularLoweringMode.DENSE,
            IntracellularLoweringMode.SPARSE_CURRENT_CLAMP,
        }
    ),
    extracellular=JAX_SINGLE_CABLE_EXTRACELLULAR_CAPABILITIES,
    supports_padding=True,
    supports_row_specific_parameters=True,
    supports_observer_only_vm_raster=True,
)
JAX_DOUBLE_CABLE_INPUT_CONTRACT = RuntimeInputContract(
    cable="double-cable",
    intracellular_modes=frozenset(
        {
            IntracellularLoweringMode.ZERO,
            IntracellularLoweringMode.DENSE,
        }
    ),
    extracellular=JAX_DOUBLE_CABLE_EXTRACELLULAR_CAPABILITIES,
    supports_padding=True,
    supports_row_specific_parameters=True,
    supports_observer_only_vm_raster=True,
)


@dataclass(frozen=True)
class LoweredIntracellularInput:
    """Concrete intracellular input selected for one dispatch group."""

    format: IntracellularInputFormat
    midpoint: Any | None


@dataclass(frozen=True)
class LoweredExtracellularInput:
    """Concrete extracellular input selected for one dispatch group."""

    format: ExtracellularInputFormat
    midpoint: Any | None
    initial_previous: Any | None = None
    dense_fallback_reason: str | None = None
    mode: ExtracellularLoweringMode | None = None
    capabilities: ExtracellularLoweringCapabilities | None = None

    @property
    def factorized(self) -> FactorizedExtracellularPotentialBatch | None:
        from axonscope.runtime.jax.batch_inputs import (
            FactorizedExtracellularPotentialBatch,
        )

        if isinstance(self.midpoint, FactorizedExtracellularPotentialBatch):
            return self.midpoint
        return None

    @property
    def factorized_rank(self) -> int | None:
        factorized = self.factorized
        return None if factorized is None else factorized.drive_count


@dataclass(frozen=True)
class PlannedInputLowering:
    """Format-only lowering description for inspection and estimates."""

    intracellular_format: IntracellularInputFormat
    extracellular_format: ExtracellularInputFormat
    factorized_rank: int | None = None
    extracellular_mode: ExtracellularLoweringMode | None = None


def lower_single_cable_intracellular_input(
    *,
    group: DispatchGroup,
    cohort: PreparedCohort,
    runtime: SolverRuntime,
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> LoweredIntracellularInput:
    """Lower single-cable intracellular inputs to sparse, dense, or zero."""

    from axonscope.runtime.jax.input_batches import (
        build_intracellular_current_density_batch,
        build_sparse_intracellular_current_density_batch,
    )

    if should_use_sparse_intracellular_batch(
        group_mode=group.mode,
        axons=cohort.axons,
        kernel_options=kernel_options,
        observers=observers,
    ):
        return LoweredIntracellularInput(
            format="sparse_current_clamp",
            midpoint=build_sparse_intracellular_current_density_batch(
                cohort.axons,
                runtime,
                solver_axons=cohort.solver_axons,
                target_nx=cohort.nx,
            ),
        )
    if not has_intracellular_contexts(cohort.axons):
        return LoweredIntracellularInput(
            format="zero_no_intracellular_context",
            midpoint=None,
        )
    return LoweredIntracellularInput(
        format="dense",
        midpoint=build_intracellular_current_density_batch(
            cohort.axons,
            runtime,
            solver_axons=cohort.solver_axons,
            target_nx=cohort.nx,
        ),
    )


def lower_double_cable_intracellular_input(
    *,
    cohort: PreparedCohort,
    runtime: SolverRuntime,
) -> LoweredIntracellularInput:
    """Lower double-cable intracellular inputs to dense or zero."""

    from axonscope.runtime.jax.input_batches import (
        build_intracellular_current_density_batch,
    )

    if not has_intracellular_contexts(cohort.axons):
        return LoweredIntracellularInput(
            format="zero_no_intracellular_context",
            midpoint=None,
        )
    return LoweredIntracellularInput(
        format="dense",
        midpoint=build_intracellular_current_density_batch(
            cohort.axons,
            runtime,
            solver_axons=cohort.solver_axons,
            target_nx=cohort.nx,
        ),
    )


def lower_single_cable_extracellular_input(
    *,
    group: DispatchGroup,
    cohort: PreparedCohort,
    runtime: SolverRuntime,
    tsim_ms: float,
    dt_ms: float,
    intracellular: LoweredIntracellularInput,
    observer_plan: Any | None,
) -> LoweredExtracellularInput:
    """Lower single-cable extracellular inputs to zero, factorized, or dense."""

    from axonscope.runtime.jax.input_batches import (
        build_factorized_vstim_midpoint_batch,
        build_vstim_midpoint_batch,
    )

    if (
        intracellular.format == "sparse_current_clamp"
        and cohort.extracellular_stimulation_count == 0
    ):
        return LoweredExtracellularInput(
            format="zero_no_extracellular_stimulation",
            midpoint=None,
            mode=ExtracellularLoweringMode.ZERO,
            capabilities=JAX_SINGLE_CABLE_EXTRACELLULAR_CAPABILITIES,
        )

    factorized = build_factorized_vstim_midpoint_batch(
        cohort.representative,
        cohort.stimulations,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        x_positions_m=cohort.x_positions_m,
        axon_y_um=cohort.axon_y_um,
        axon_z_um=cohort.axon_z_um,
        dtype_local=runtime.membrane.dtype,
    )
    if factorized is not None:
        return LoweredExtracellularInput(
            format="factorized_footprint",
            midpoint=factorized,
            mode=_factorized_extracellular_mode(factorized),
            capabilities=JAX_SINGLE_CABLE_EXTRACELLULAR_CAPABILITIES,
        )

    if intracellular.format == "sparse_current_clamp" and observer_plan is not None:
        raise NotImplementedError(
            "single-cable observer-only batch execution requires factorized "
            "footprint/drive extracellular stimulation. Use sampled "
            "footprint/drive inputs, record Vm explicitly, or split the "
            "unsupported stimulation into a separate workflow."
        )

    return LoweredExtracellularInput(
        format="dense",
        midpoint=build_vstim_midpoint_batch(
            cohort.representative,
            cohort.stimulations,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            x_positions_m=cohort.x_positions_m,
            axon_y_um=cohort.axon_y_um,
            axon_z_um=cohort.axon_z_um,
            dtype_local=runtime.membrane.dtype,
        ),
        dense_fallback_reason="unsupported_factorized_footprint_rows",
        mode=ExtracellularLoweringMode.DENSE,
        capabilities=JAX_SINGLE_CABLE_EXTRACELLULAR_CAPABILITIES,
    )


def lower_double_cable_extracellular_input(
    *,
    cohort: PreparedCohort,
    runtime: SolverRuntime,
    tsim_ms: float,
    dt_ms: float,
    observer_plan: Any | None,
    kernel_options: BatchOptions,
) -> LoweredExtracellularInput:
    """Lower double-cable extracellular inputs.

    Double-cable prefers the compact factorized representation for rank-1
    shared currents and scaled shared waveforms. Rank-K/current-table inputs
    are explicitly materialized as dense until that solver path has equivalence
    and benchmark coverage.
    """

    from axonscope.runtime.jax.input_batches import (
        build_factorized_vstim_midpoint_batch,
        build_vstim_midpoint_and_initial_previous_batch,
    )

    factorized = build_factorized_vstim_midpoint_batch(
        cohort.representative,
        cohort.stimulations,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        x_positions_m=cohort.x_positions_m,
        axon_y_um=cohort.axon_y_um,
        axon_z_um=cohort.axon_z_um,
        dtype_local=runtime.membrane.dtype,
        include_initial_previous=True,
    )
    if factorized is not None and supports_compact_double_cable_factorized(
        factorized
    ):
        return LoweredExtracellularInput(
            format="factorized_footprint",
            midpoint=factorized,
            mode=_factorized_extracellular_mode(factorized),
            capabilities=JAX_DOUBLE_CABLE_EXTRACELLULAR_CAPABILITIES,
        )

    midpoint, initial_previous = build_vstim_midpoint_and_initial_previous_batch(
        cohort.representative,
        cohort.stimulations,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        x_positions_m=cohort.x_positions_m,
        axon_y_um=cohort.axon_y_um,
        axon_z_um=cohort.axon_z_um,
        dtype_local=runtime.membrane.dtype,
    )
    return LoweredExtracellularInput(
        format="dense",
        midpoint=midpoint,
        initial_previous=initial_previous,
        dense_fallback_reason="unsupported_double_cable_factorized_mode",
        mode=ExtracellularLoweringMode.DENSE,
        capabilities=JAX_DOUBLE_CABLE_EXTRACELLULAR_CAPABILITIES,
    )


def plan_input_lowering(
    *,
    group_mode: str,
    axons: Sequence[AxonInstance],
    stimulation_rows: Sequence[tuple[Any, ...]],
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
    observer_plan: bool,
) -> PlannedInputLowering:
    """Return the input formats that runtime lowering will select."""

    sparse_intracellular = should_use_sparse_intracellular_batch(
        group_mode=group_mode,
        axons=axons,
        kernel_options=kernel_options,
        observers=observers,
    )
    if sparse_intracellular:
        intracellular_format: IntracellularInputFormat = "sparse_current_clamp"
    elif has_intracellular_contexts(axons):
        intracellular_format = "dense"
    else:
        intracellular_format = "zero_no_intracellular_context"

    stimulation_count = extracellular_stimulation_count(stimulation_rows)
    factorized_rank = factorized_drive_count_from_rows(stimulation_rows)
    if (
        group_mode == "single"
        and sparse_intracellular
        and stimulation_count == 0
    ):
        extracellular_format: ExtracellularInputFormat = "zero_no_extracellular_stimulation"
        extracellular_mode = ExtracellularLoweringMode.ZERO
    elif group_mode == "single" and can_factorize_footprint_rows(stimulation_rows):
        extracellular_format = "factorized_footprint"
        extracellular_mode = None
    elif observer_plan and can_plan_compact_double_cable_factorized_rows(
        stimulation_rows
    ):
        extracellular_format = "factorized_footprint"
        extracellular_mode = ExtracellularLoweringMode.SHARED_CURRENT
    else:
        extracellular_format = "dense"
        extracellular_mode = ExtracellularLoweringMode.DENSE

    return PlannedInputLowering(
        intracellular_format=intracellular_format,
        extracellular_format=extracellular_format,
        factorized_rank=(
            factorized_rank if extracellular_format == "factorized_footprint" else None
        ),
        extracellular_mode=extracellular_mode,
    )


def should_use_sparse_intracellular_batch(
    *,
    group_mode: str,
    axons: Sequence[AxonInstance],
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> bool:
    """Return whether sparse point-clamp lowering can feed this group."""

    from axonscope.runtime.jax.input_batches import (
        can_build_sparse_intracellular_current_density_batch,
    )

    return (
        group_mode == "single"
        and observers is not None
        and kernel_options.recording.mode == "none"
        and can_build_sparse_intracellular_current_density_batch(axons)
    )


def has_intracellular_contexts(axons: Sequence[AxonInstance]) -> bool:
    """Return whether any row has an attached intracellular input."""

    return any(getattr(axon, "intracellular_contexts", ()) for axon in axons)


def can_factorize_footprint_rows(
    rows: Sequence[tuple[Any, ...]],
) -> bool:
    """Return whether rows use sampled footprint/drive stimulation objects."""

    if not rows or not any(rows):
        return False
    for row in rows:
        for stimulation in row:
            drives = tuple(getattr(stimulation, "drives", ()))
            if not drives:
                return False
            for drive in drives:
                if getattr(drive, "stimulus", None) is None:
                    return False
                if getattr(drive, "footprint", None) is None:
                    return False
    return True


def factorized_drive_count_from_rows(rows: Sequence[tuple[Any, ...]]) -> int:
    """Return the maximum factorized drive count per row."""

    max_count = 1
    for row in rows:
        row_count = 0
        for stimulation in row:
            row_count += len(tuple(getattr(stimulation, "drives", ())))
        max_count = max(max_count, row_count)
    return int(max_count)


def can_plan_compact_double_cable_factorized_rows(
    rows: Sequence[tuple[Any, ...]],
) -> bool:
    """Conservatively predict the current double-cable compact factorized path."""

    if not can_factorize_footprint_rows(rows):
        return False
    if factorized_drive_count_from_rows(rows) != 1:
        return False

    shared_stimulus_id: int | None = None
    for row in rows:
        row_stimuli = [
            getattr(drive, "stimulus", None)
            for stimulation in row
            for drive in tuple(getattr(stimulation, "drives", ()))
        ]
        if len(row_stimuli) != 1 or row_stimuli[0] is None:
            return False
        stimulus_id = id(row_stimuli[0])
        if shared_stimulus_id is None:
            shared_stimulus_id = stimulus_id
        elif stimulus_id != shared_stimulus_id:
            return False
    return shared_stimulus_id is not None


def extracellular_stimulation_count(rows: Sequence[tuple[Any, ...]]) -> int:
    """Return the number of attached extracellular stimulation objects."""

    return sum(len(tuple(row)) for row in rows)


def supports_compact_double_cable_factorized(
    factorized: FactorizedExtracellularPotentialBatch,
) -> bool:
    """Return whether the double-cable kernel can consume this factorized input compactly."""

    import jax.numpy as jnp

    previous = factorized.current_initial_previous_A
    if previous is None:
        return False
    mode = _factorized_extracellular_mode(factorized)
    if not JAX_DOUBLE_CABLE_EXTRACELLULAR_CAPABILITIES.supports(mode):
        return False
    if factorized.drive_count != 1:
        return False
    previous_is_scalar = jnp.asarray(previous).ndim == 0
    if factorized.shared_current:
        return bool(previous_is_scalar)
    if factorized.current_row_scales is not None:
        return bool(previous_is_scalar)
    return False


def _factorized_extracellular_mode(
    factorized: FactorizedExtracellularPotentialBatch,
) -> ExtracellularLoweringMode:
    """Return the semantic lowering mode for a factorized payload."""

    current_shape = tuple(
        int(dim) for dim in getattr(factorized.current_mid_A, "shape", ())
    )
    if factorized.current_row_scales is not None:
        return ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM
    if factorized.shared_current:
        return ExtracellularLoweringMode.SHARED_CURRENT
    if factorized.current_row_indices is not None or len(current_shape) in {2, 3}:
        return ExtracellularLoweringMode.CURRENT_TABLE
    return ExtracellularLoweringMode.CURRENT_TABLE


def dense_shape_for_group(
    *,
    group: DispatchGroup,
    runtime: SolverRuntime,
) -> tuple[int, int, int]:
    """Return the dense ``(B, Nt, Nx)`` equivalent shape for group inputs."""

    return (int(group.size), int(runtime.grid.Nt), int(group.nx))


def dense_nbytes_for_shape(
    shape: tuple[int, ...],
    *,
    dtype: np.dtype[Any],
) -> int:
    """Return byte count for a dense array shape."""

    return int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)


__all__ = [
    "ExtracellularInputFormat",
    "IntracellularInputFormat",
    "JAX_DOUBLE_CABLE_EXTRACELLULAR_CAPABILITIES",
    "JAX_DOUBLE_CABLE_INPUT_CONTRACT",
    "JAX_SINGLE_CABLE_EXTRACELLULAR_CAPABILITIES",
    "JAX_SINGLE_CABLE_INPUT_CONTRACT",
    "LoweredExtracellularInput",
    "LoweredIntracellularInput",
    "PlannedInputLowering",
    "can_plan_compact_double_cable_factorized_rows",
    "can_factorize_footprint_rows",
    "dense_nbytes_for_shape",
    "dense_shape_for_group",
    "extracellular_stimulation_count",
    "factorized_drive_count_from_rows",
    "has_intracellular_contexts",
    "lower_double_cable_extracellular_input",
    "lower_double_cable_intracellular_input",
    "lower_single_cable_extracellular_input",
    "lower_single_cable_intracellular_input",
    "plan_input_lowering",
    "should_use_sparse_intracellular_batch",
    "supports_compact_double_cable_factorized",
]
