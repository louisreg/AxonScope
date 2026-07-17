"""Internal runtime input-lowering contracts.

These types describe semantic input-lowering modes shared by concrete runtime
implementations. They are intentionally not part of the public ``axs.runtime``
facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np


CableFormulation = Literal["single-cable", "double-cable"]
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


class IntracellularLoweringMode(Enum):
    """Semantic intracellular payload selected before cable kernels run."""

    ZERO = "zero"
    DENSE = "dense"
    SPARSE_CURRENT_CLAMP = "sparse_current_clamp"


class ExtracellularLoweringMode(Enum):
    """Semantic extracellular payload selected before cable kernels run.

    The shared/scaled modes are ``Nstim``-aware: a single point source is the
    ``Nstim=1`` case, while multi-contact stimulation keeps the stimulation
    axis instead of introducing another semantic mode. ``CURRENT_TABLE`` is the
    fallback for arbitrary temporal currents that cannot be represented as
    shared waveforms plus row scales.
    """

    ZERO = "zero"
    SHARED_CURRENT = "shared_current"
    SCALED_SHARED_WAVEFORM = "scaled_shared_waveform"
    CURRENT_TABLE = "current_table"
    DENSE = "dense"


@dataclass(frozen=True)
class ExtracellularLoweringCapabilities:
    """Supported extracellular lowering modes for one runtime/cable path."""

    cable: CableFormulation
    supports_zero: bool
    supports_shared_current: bool
    supports_scaled_shared_waveform: bool
    supports_current_table: bool
    supports_dense_fallback: bool
    requires_initial_previous: bool = False

    def supports(self, mode: ExtracellularLoweringMode) -> bool:
        """Return whether this cable path can consume ``mode`` compactly."""

        if not isinstance(mode, ExtracellularLoweringMode):
            raise TypeError("mode must be an ExtracellularLoweringMode value.")
        return {
            ExtracellularLoweringMode.ZERO: self.supports_zero,
            ExtracellularLoweringMode.SHARED_CURRENT: self.supports_shared_current,
            ExtracellularLoweringMode.SCALED_SHARED_WAVEFORM: (
                self.supports_scaled_shared_waveform
            ),
            ExtracellularLoweringMode.CURRENT_TABLE: self.supports_current_table,
            ExtracellularLoweringMode.DENSE: self.supports_dense_fallback,
        }[mode]

    def as_metadata(self, *, prefix: str = "lowering_capability_") -> dict[str, Any]:
        """Return primitive benchmark/inspection metadata."""

        return {
            f"{prefix}cable": self.cable,
            f"{prefix}supports_zero": self.supports_zero,
            f"{prefix}supports_shared_current": self.supports_shared_current,
            f"{prefix}supports_scaled_shared_waveform": (
                self.supports_scaled_shared_waveform
            ),
            f"{prefix}supports_current_table": self.supports_current_table,
            f"{prefix}supports_dense_fallback": self.supports_dense_fallback,
            f"{prefix}requires_initial_previous": self.requires_initial_previous,
        }


@dataclass(frozen=True)
class RuntimeInputContract:
    """Runtime-neutral input contract for one prepared cable batch.

    This is the semantic contract shared by concrete runtimes. It deliberately
    does not prescribe array libraries, kernel names, or solver algorithms:
    JAX, NumPy/SciPy, or a future runtime may materialize these modes
    differently as long as they accept the same grouped batch semantics.
    """

    cable: CableFormulation
    intracellular_modes: frozenset[IntracellularLoweringMode]
    extracellular: ExtracellularLoweringCapabilities
    supports_padding: bool
    supports_row_specific_parameters: bool
    supports_threshold_observer: bool

    def supports_intracellular(self, mode: IntracellularLoweringMode) -> bool:
        """Return whether this runtime/cable path can consume ``mode``."""

        if not isinstance(mode, IntracellularLoweringMode):
            raise TypeError("mode must be an IntracellularLoweringMode value.")
        return mode in self.intracellular_modes

    def supports_extracellular(self, mode: ExtracellularLoweringMode) -> bool:
        """Return whether this runtime/cable path can consume ``mode``."""

        return self.extracellular.supports(mode)

    def as_metadata(self, *, prefix: str = "runtime_input_contract_") -> dict[str, Any]:
        """Return primitive benchmark/inspection metadata."""

        metadata = {
            f"{prefix}cable": self.cable,
            f"{prefix}intracellular_modes": tuple(
                mode.value
                for mode in sorted(
                    self.intracellular_modes,
                    key=lambda item: item.value,
                )
            ),
            f"{prefix}supports_padding": self.supports_padding,
            f"{prefix}supports_row_specific_parameters": (
                self.supports_row_specific_parameters
            ),
            f"{prefix}supports_threshold_observer": self.supports_threshold_observer,
        }
        metadata.update(
            self.extracellular.as_metadata(
                prefix=f"{prefix}extracellular_",
            )
        )
        return metadata


@dataclass(frozen=True)
class PreparedRuntimeInputSummary:
    """Runtime-neutral summary for one concrete prepared batch.

    The summary is intentionally primitive and metadata-friendly. It captures the
    semantic contract a concrete runtime is about to execute without prescribing
    array classes, JIT behavior, or solver implementation details.
    """

    cable: CableFormulation
    batch_size: int
    nx: int
    nt: int
    dtype: str
    has_padding: bool
    row_specific_parameters: bool
    recording_mode: str
    output_sink: str
    observer_count: int
    time_chunk_steps: int | None
    solver_policy: str
    intracellular_format: IntracellularInputFormat
    intracellular_mode: IntracellularLoweringMode
    extracellular_format: ExtracellularInputFormat
    extracellular_mode: ExtracellularLoweringMode
    extracellular_requires_initial_previous: bool
    extracellular_has_initial_previous: bool

    def validate_against(self, contract: RuntimeInputContract) -> None:
        """Raise when this prepared batch violates ``contract``."""

        errors = validate_prepared_runtime_input(self, contract)
        if errors:
            raise ValueError(
                "prepared runtime input violates contract: " + "; ".join(errors)
            )

    def as_metadata(
        self,
        *,
        prefix: str = "prepared_input_contract_",
    ) -> dict[str, Any]:
        """Return primitive benchmark/inspection metadata."""

        return {
            f"{prefix}cable": self.cable,
            f"{prefix}batch_size": int(self.batch_size),
            f"{prefix}nx": int(self.nx),
            f"{prefix}nt": int(self.nt),
            f"{prefix}dtype": self.dtype,
            f"{prefix}has_padding": bool(self.has_padding),
            f"{prefix}row_specific_parameters": bool(self.row_specific_parameters),
            f"{prefix}recording_mode": self.recording_mode,
            f"{prefix}output_sink": self.output_sink,
            f"{prefix}observer_count": int(self.observer_count),
            f"{prefix}time_chunk_steps": self.time_chunk_steps,
            f"{prefix}solver_policy": self.solver_policy,
            f"{prefix}intracellular_format": self.intracellular_format,
            f"{prefix}intracellular_mode": self.intracellular_mode.value,
            f"{prefix}extracellular_format": self.extracellular_format,
            f"{prefix}extracellular_mode": self.extracellular_mode.value,
            f"{prefix}extracellular_requires_initial_previous": (
                self.extracellular_requires_initial_previous
            ),
            f"{prefix}extracellular_has_initial_previous": (
                self.extracellular_has_initial_previous
            ),
        }


def intracellular_mode_from_format(
    value: IntracellularInputFormat,
) -> IntracellularLoweringMode:
    """Return the semantic mode represented by an intracellular format label."""

    if value == "zero_no_intracellular_context":
        return IntracellularLoweringMode.ZERO
    if value == "dense":
        return IntracellularLoweringMode.DENSE
    if value == "sparse_current_clamp":
        return IntracellularLoweringMode.SPARSE_CURRENT_CLAMP
    raise ValueError(f"Unsupported intracellular input format: {value!r}.")


def extracellular_mode_from_format(
    value: ExtracellularInputFormat,
    *,
    explicit_mode: ExtracellularLoweringMode | None = None,
) -> ExtracellularLoweringMode:
    """Return the semantic mode represented by an extracellular format label."""

    if explicit_mode is not None:
        if not isinstance(explicit_mode, ExtracellularLoweringMode):
            raise TypeError("explicit_mode must be an ExtracellularLoweringMode value.")
        return explicit_mode
    if value == "zero_no_extracellular_stimulation":
        return ExtracellularLoweringMode.ZERO
    if value == "dense":
        return ExtracellularLoweringMode.DENSE
    raise ValueError(
        f"Extracellular format {value!r} needs an explicit semantic lowering mode."
    )


def validate_prepared_runtime_input(
    summary: PreparedRuntimeInputSummary,
    contract: RuntimeInputContract,
) -> tuple[str, ...]:
    """Return contract violations for one prepared runtime input summary."""

    errors: list[str] = []
    if normalize_cable_formulation(summary.cable) != contract.cable:
        errors.append(
            f"cable {summary.cable!r} does not match contract {contract.cable!r}"
        )
    for field_name in ("batch_size", "nx", "nt"):
        if int(getattr(summary, field_name)) <= 0:
            errors.append(f"{field_name} must be positive")
    if not str(summary.dtype).strip():
        errors.append("dtype must be non-empty")
    if summary.has_padding and not contract.supports_padding:
        errors.append("runtime does not support padded batches")
    if summary.row_specific_parameters and not contract.supports_row_specific_parameters:
        errors.append("runtime does not support row-specific parameters")
    if (
        summary.output_sink in {"activation", "first_crossing", "vm_raster"}
        and not contract.supports_threshold_observer
    ):
        errors.append("runtime does not support observer-only threshold output")
    if not contract.supports_intracellular(summary.intracellular_mode):
        errors.append(
            f"intracellular mode {summary.intracellular_mode.value!r} is unsupported"
        )
    if not contract.supports_extracellular(summary.extracellular_mode):
        errors.append(
            f"extracellular mode {summary.extracellular_mode.value!r} is unsupported"
        )
    if (
        summary.extracellular_requires_initial_previous
        and summary.extracellular_mode is not ExtracellularLoweringMode.ZERO
        and not summary.extracellular_has_initial_previous
    ):
        errors.append("extracellular mode requires an initial-previous sample")
    return tuple(errors)


def normalize_cable_formulation(value: str) -> CableFormulation:
    """Return the canonical runtime-neutral cable formulation label."""

    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"single", "single-cable"}:
        return "single-cable"
    if normalized in {"double", "double-cable"}:
        return "double-cable"
    raise ValueError(f"Unsupported cable formulation: {value!r}.")


def dense_shape_for_group(
    *,
    group: Any,
    runtime: Any,
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
    "CableFormulation",
    "ExtracellularInputFormat",
    "ExtracellularLoweringCapabilities",
    "ExtracellularLoweringMode",
    "IntracellularInputFormat",
    "IntracellularLoweringMode",
    "PreparedRuntimeInputSummary",
    "RuntimeInputContract",
    "dense_nbytes_for_shape",
    "dense_shape_for_group",
    "extracellular_mode_from_format",
    "intracellular_mode_from_format",
    "normalize_cable_formulation",
    "validate_prepared_runtime_input",
]
