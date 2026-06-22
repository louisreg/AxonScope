"""Typed public position selectors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from axonscope.utils import units


class PositionMode(Enum):
    """Closed set of named one-dimensional position selectors."""

    ALL = "all"
    PROXIMAL = "proximal"
    CENTER = "center"
    DISTAL = "distal"


@dataclass(frozen=True)
class PositionSelector:
    """Typed selector for recorded axon positions."""

    mode: PositionMode | None = None
    positions_um: tuple[float, ...] | None = None
    index_values: tuple[int, ...] | None = None

    @classmethod
    def all(cls) -> "PositionSelector":
        """Select all recorded positions."""

        return cls(mode=PositionMode.ALL)

    @classmethod
    def center(cls) -> "PositionSelector":
        """Select the recorded position nearest the axon center."""

        return cls(mode=PositionMode.CENTER)

    @classmethod
    def proximal(cls) -> "PositionSelector":
        """Select the most proximal recorded position."""

        return cls(mode=PositionMode.PROXIMAL)

    @classmethod
    def distal(cls) -> "PositionSelector":
        """Select the most distal recorded position."""

        return cls(mode=PositionMode.DISTAL)

    @classmethod
    def at(cls, values: Sequence[Any]) -> "PositionSelector":
        """Select recorded positions nearest the requested physical positions."""

        positions_um = tuple(float(value) for value in units.to_um_array(values))
        if not positions_um:
            raise ValueError("PositionSelector.at requires at least one position.")
        return cls(positions_um=positions_um)

    @classmethod
    def indices(cls, values: Sequence[int]) -> "PositionSelector":
        """Select explicit original compartment indices."""

        indices = tuple(int(value) for value in values)
        if not indices:
            raise ValueError("PositionSelector.indices requires at least one index.")
        if any(value < 0 for value in indices):
            raise ValueError("position indices must be non-negative.")
        return cls(index_values=indices)

    def columns(
        self,
        *,
        positions_um: np.ndarray,
        original_indices: np.ndarray,
    ) -> np.ndarray:
        """Resolve this selector to recorded result columns."""

        if self.index_values is not None:
            columns = [
                int(np.flatnonzero(original_indices == index)[0])
                for index in self.index_values
                if np.any(original_indices == index)
            ]
            if len(columns) != len(self.index_values):
                missing = sorted(set(self.index_values).difference(set(original_indices)))
                raise ValueError(f"indices are not present in this result: {missing}.")
            return np.asarray(columns, dtype=int)

        if self.positions_um is not None:
            selected = np.asarray(
                [
                    int(np.argmin(np.abs(positions_um - target)))
                    for target in self.positions_um
                ],
                dtype=int,
            )
            return np.unique(selected)

        mode = self.mode or PositionMode.ALL
        if mode is PositionMode.ALL:
            return np.arange(positions_um.shape[0], dtype=int)
        if mode is PositionMode.PROXIMAL:
            return np.asarray([int(np.argmin(positions_um))], dtype=int)
        if mode is PositionMode.CENTER:
            center_um = 0.5 * (float(np.min(positions_um)) + float(np.max(positions_um)))
            return np.asarray([int(np.argmin(np.abs(positions_um - center_um)))], dtype=int)
        if mode is PositionMode.DISTAL:
            return np.asarray([int(np.argmax(positions_um))], dtype=int)
        raise ValueError(f"Unsupported position selector mode: {mode!r}.")


ALL = PositionSelector.all()
PROXIMAL = PositionSelector.proximal()
CENTER = PositionSelector.center()
DISTAL = PositionSelector.distal()


def At(values: Any | Sequence[Any]) -> PositionSelector:
    """Select recorded positions nearest one or more physical positions."""

    if isinstance(values, (list, tuple)):
        return PositionSelector.at(values)
    return PositionSelector.at((values,))


def Indices(values: Sequence[int]) -> PositionSelector:
    """Select explicit original compartment indices."""

    return PositionSelector.indices(values)


__all__ = [
    "PositionMode",
    "PositionSelector",
    "ALL",
    "PROXIMAL",
    "CENTER",
    "DISTAL",
    "At",
    "Indices",
]
