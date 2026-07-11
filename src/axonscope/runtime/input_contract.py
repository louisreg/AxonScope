"""Internal runtime input-lowering contracts.

These types describe semantic input-lowering modes shared by concrete runtime
implementations. They are intentionally not part of the public ``axs.runtime``
facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


CableFormulation = Literal["single-cable", "double-cable"]


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


__all__ = [
    "CableFormulation",
    "ExtracellularLoweringCapabilities",
    "ExtracellularLoweringMode",
]
