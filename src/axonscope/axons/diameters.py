"""Axon diameter canonicalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def round_axon_diameter_um(value: Any) -> float:
    """Round one nominal axon/fiber diameter in micrometers."""

    diameter = float(value)
    step = 0.1 if diameter > 1.0 else 0.01
    rounded = np.floor(diameter / step + 0.5) * step
    return float(np.round(rounded, 8))


def round_axon_diameter_values_um(values: Any) -> np.ndarray:
    """Round an array of nominal axon/fiber diameters in micrometers."""

    diameters = np.asarray(values, dtype=float)
    steps = np.where(diameters > 1.0, 0.1, 0.01)
    rounded = np.floor(diameters / steps + 0.5) * steps
    return np.round(rounded, 8)


__all__ = [
    "round_axon_diameter_um",
    "round_axon_diameter_values_um",
]
