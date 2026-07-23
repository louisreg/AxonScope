"""Recorded result axes shared by scalar and population result views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonfleet.utils import units


@dataclass(frozen=True)
class RecordedAxis:
    """Spatial axis represented by one membrane-voltage recording.

    ``positions_um`` are intrinsic axon coordinates, not world/anatomical
    coordinates. ``original_indices`` map recorded columns back to the source
    axon layout.
    """

    positions_um: tuple[float, ...]
    original_indices: tuple[int, ...]

    @classmethod
    def from_result(cls, result: Any) -> "RecordedAxis":
        """Build the recorded spatial axis for a one-axon result-like object."""

        try:
            vm = np.asarray(result.Vm)
        except AttributeError as exc:
            raise ValueError("result does not contain a Vm recording.") from exc
        if vm.ndim != 2:
            raise ValueError(f"result.Vm must be 2D (time, position), got shape {vm.shape}.")
        if not hasattr(result.axon, "layout"):
            raise ValueError("result.axon must expose a layout for spatial analysis.")

        positions = np.asarray(
            result.axon.layout.position_values(unit="micrometer"),
            dtype=float,
        )
        if positions.ndim != 1:
            raise ValueError(f"result axon positions must be 1D, got shape {positions.shape}.")

        record_indices = getattr(result, "record_indices", None)
        if record_indices is None:
            if vm.shape[1] != positions.shape[0]:
                raise ValueError(
                    "result.Vm is spatially filtered but result.record_indices is missing; "
                    "cannot infer intrinsic positions for analysis."
                )
            indices = np.arange(vm.shape[1], dtype=int)
            recorded_positions = positions
        else:
            indices = np.asarray(record_indices, dtype=int)
            if indices.shape != (vm.shape[1],):
                raise ValueError(
                    "record_indices must contain one entry per Vm column; "
                    f"got {indices.shape[0]} indices for {vm.shape[1]} columns."
                )
            if np.any(indices < 0) or np.any(indices >= positions.shape[0]):
                raise ValueError("record_indices contains values outside axon positions.")
            recorded_positions = positions[indices]

        return cls(
            positions_um=tuple(float(value) for value in recorded_positions),
            original_indices=tuple(int(value) for value in indices),
        )

    @property
    def size(self) -> int:
        """Number of recorded spatial columns."""

        return len(self.positions_um)

    def position_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return recorded intrinsic positions as plain values in ``unit``."""

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(
            units.Q_(np.asarray(self.positions_um, dtype=float), "micrometer"),
            unit_label,
            dtype=float,
        )

    def index_values(self) -> np.ndarray:
        """Return original axon layout indices represented by recorded columns."""

        return np.asarray(self.original_indices, dtype=int)


__all__ = ["RecordedAxis"]
