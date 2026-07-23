"""Private helpers shared by Schild-family membrane source models."""

from __future__ import annotations

import math


def nernst_mV(temp_c: float, z: float, outside_mM: float, inside_mM: float) -> float:
    return 8314.0 * (temp_c + 273.15) / (z * 96500.0) * math.log(outside_mM / inside_mM)
