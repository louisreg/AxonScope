"""Input lowering contracts for JAX batch execution.

This module owns the decision from prepared AxonScope rows to concrete kernel
input representations. Callers should not decide dense/factorized/zero input
formats directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

from axonscope.dispatcher.numeric_axis import (
    ExtracellularWaveformAxisInput,
    NumericAxisInput,
)
from axonscope.runtime.input_contract import (
    ExtracellularInputFormat,
    ExtracellularLoweringCapabilities,
    ExtracellularLoweringMode,
    IntracellularInputFormat,
    IntracellularLoweringMode,
    RuntimeInputContract,
)
import axonscope.runtime.input_planning as input_planning


if TYPE_CHECKING:
    from axonscope.axon_instance import AxonInstance
    from axonscope.runtime.input_payloads import FactorizedExtracellularPotentialBatch
    from axonscope.runtime.jax.types import SolverRuntime
    from axonscope.dispatcher.plan import DispatchGroup
    from axonscope.preparation.cohort import PreparedCohort
    from axonscope.solvers.options import BatchOptions


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
    supports_current_table=True,
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
        from axonscope.runtime.input_payloads import (
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


def lower_numeric_axis_input(
    extracellular: LoweredExtracellularInput,
    axis_input: NumericAxisInput,
    *,
    source_size: int,
    tsim_ms: float,
    dt_ms: float,
    dtype_local: Any,
    include_initial_previous: bool,
) -> LoweredExtracellularInput:
    """Lower one typed numeric-axis input through the backend input contract."""

    if not isinstance(axis_input, ExtracellularWaveformAxisInput):
        raise RuntimeError(
            "JAX does not support numeric-axis input "
            f"{type(axis_input).__name__!r}."
        )

    factorized = extracellular.factorized
    if factorized is None:
        raise RuntimeError(
            "compact waveform-axis execution requires factorized extracellular input; "
            f"lowering selected {extracellular.format!r}."
        )
    from axonscope.runtime.jax.inputs.extracellular import (
        with_extracellular_waveform_axis,
    )

    axis_payload = with_extracellular_waveform_axis(
        factorized,
        axis_input,
        source_size=source_size,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        dtype_local=dtype_local,
        include_initial_previous=include_initial_previous,
    )
    return replace(
        extracellular,
        midpoint=axis_payload,
        initial_previous=None,
        mode=ExtracellularLoweringMode.CURRENT_TABLE,
    )


def lower_single_cable_intracellular_input(
    *,
    group: DispatchGroup,
    cohort: PreparedCohort,
    runtime: SolverRuntime,
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> LoweredIntracellularInput:
    """Lower single-cable intracellular inputs to sparse, dense, or zero."""

    from axonscope.runtime.jax.inputs.intracellular import (
        build_intracellular_current_density_batch,
        build_sparse_intracellular_current_density_batch,
        build_zero_sparse_intracellular_current_density_batch,
    )

    has_contexts = has_intracellular_contexts(cohort.axons)
    if should_use_sparse_intracellular_batch(
        group_mode=group.mode,
        axons=cohort.axons,
        kernel_options=kernel_options,
        observers=observers,
    ):
        if not has_contexts:
            return LoweredIntracellularInput(
                format="sparse_current_clamp",
                midpoint=build_zero_sparse_intracellular_current_density_batch(
                    batch_size=group.size,
                    step_count=runtime.grid.Nt,
                    target_nx=cohort.nx,
                    dtype_local=runtime.membrane.dtype,
                ),
            )
        return LoweredIntracellularInput(
            format="sparse_current_clamp",
            midpoint=build_sparse_intracellular_current_density_batch(
                cohort.axons,
                runtime,
                solver_axons=cohort.solver_axons,
                target_nx=cohort.nx,
            ),
        )
    if not has_contexts:
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

    from axonscope.runtime.jax.inputs.intracellular import (
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
    cohort: PreparedCohort,
    runtime: SolverRuntime,
    tsim_ms: float,
    dt_ms: float,
    intracellular: LoweredIntracellularInput,
    observer_plan: Any | None,
    require_factorized: bool = False,
    numeric_axis_shape: tuple[int, int] | None = None,
) -> LoweredExtracellularInput:
    """Lower single-cable extracellular inputs to zero, factorized, or dense."""

    from axonscope.runtime.jax.inputs.extracellular import (
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
        single_cable_lower=runtime.cable.lower,
        single_cable_upper=runtime.cable.upper,
        numeric_axis_shape=numeric_axis_shape,
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
    if require_factorized:
        raise RuntimeError(
            "JAX GPU observer-only execution resolved to unsupported single-cable "
            "factorized extracellular rows; dense materialization is disabled."
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
    require_factorized: bool = False,
    numeric_axis_shape: tuple[int, int] | None = None,
) -> LoweredExtracellularInput:
    """Lower double-cable extracellular inputs.

    Double-cable prefers the compact factorized representation for rank-1
    shared currents and scaled shared waveforms. Rank-K/current-table inputs
    are explicitly materialized as dense until that solver path has equivalence
    and benchmark coverage.
    """

    from axonscope.runtime.jax.inputs.extracellular import (
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
        numeric_axis_shape=numeric_axis_shape,
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

    if require_factorized:
        raise RuntimeError(
            "JAX GPU observer-only execution resolved to unsupported double-cable "
            "factorized extracellular rows; dense materialization is disabled."
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

    stimulation_count = input_planning.extracellular_stimulation_count(
        stimulation_rows
    )
    factorized_rank = input_planning.factorized_drive_count_from_rows(
        stimulation_rows
    )
    factorized_mode = input_planning.planned_factorized_extracellular_mode_from_rows(
        stimulation_rows
    )
    if (
        group_mode == "single"
        and sparse_intracellular
        and stimulation_count == 0
    ):
        extracellular_format: ExtracellularInputFormat = "zero_no_extracellular_stimulation"
        extracellular_mode = ExtracellularLoweringMode.ZERO
    elif group_mode == "single" and input_planning.can_factorize_footprint_rows(
        stimulation_rows
    ):
        extracellular_format = "factorized_footprint"
        extracellular_mode = factorized_mode
    elif group_mode == "double" and factorized_mode in {
        ExtracellularLoweringMode.SHARED_CURRENT,
        ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM,
        ExtracellularLoweringMode.CURRENT_TABLE,
    }:
        extracellular_format = "factorized_footprint"
        extracellular_mode = factorized_mode
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

    from axonscope.runtime.jax.inputs.intracellular import (
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
    if factorized.drive_count > 1:
        current_shape = tuple(
            int(dim) for dim in getattr(factorized.current_mid_A, "shape", ())
        )
        previous_shape = tuple(int(dim) for dim in getattr(previous, "shape", ()))
        footprint_shape = tuple(
            int(dim) for dim in getattr(factorized.footprint_mV_per_A, "shape", ())
        )
        return bool(
            len(footprint_shape) == 3
            and len(current_shape) in {2, 3}
            and len(previous_shape) in {1, 2}
        )
    previous_is_scalar = jnp.asarray(previous).ndim == 0
    if factorized.shared_current:
        return bool(previous_is_scalar)
    if factorized.current_row_scales is not None:
        return bool(previous_is_scalar)
    if factorized.current_row_indices is not None:
        previous_shape = tuple(
            int(dim) for dim in getattr(previous, "shape", ())
        )
        return previous_is_scalar or len(previous_shape) == 1
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
    "has_intracellular_contexts",
    "lower_double_cable_extracellular_input",
    "lower_double_cable_intracellular_input",
    "lower_numeric_axis_input",
    "lower_single_cable_extracellular_input",
    "lower_single_cable_intracellular_input",
    "plan_input_lowering",
    "should_use_sparse_intracellular_batch",
    "supports_compact_double_cable_factorized",
]
