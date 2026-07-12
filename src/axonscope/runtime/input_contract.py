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
    supports_observer_only_vm_raster: bool

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
            f"{prefix}supports_observer_only_vm_raster": (
                self.supports_observer_only_vm_raster
            ),
        }
        metadata.update(
            self.extracellular.as_metadata(
                prefix=f"{prefix}extracellular_",
            )
        )
        return metadata


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
    "RuntimeInputContract",
    "dense_nbytes_for_shape",
    "dense_shape_for_group",
    "normalize_cable_formulation",
]
