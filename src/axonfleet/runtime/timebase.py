"""Backend-neutral time-grid validation helpers."""

from __future__ import annotations

import math
from typing import Any

from axonfleet.utils import units


def simulation_step_count(duration_ms: float, dt_ms: float) -> int:
    """Return the fixed-step count for an exact simulation grid.

    Current solver kernels use one fixed ``dt`` for every integration step. If
    ``duration_ms`` is not an integer multiple of ``dt_ms``, rounding up would
    silently run past the requested final time. Refuse that case until kernels
    grow an explicit partial-final-step policy.
    """

    duration = float(duration_ms)
    step = float(dt_ms)
    if duration <= 0.0:
        raise ValueError("duration_ms must be > 0.")
    if step <= 0.0:
        raise ValueError("dt_ms must be > 0.")

    ratio = duration / step
    steps = int(round(ratio))
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "duration_ms must be an integer multiple of dt_ms for the current "
            f"fixed-step solvers; got duration_ms={duration:g}, dt_ms={step:g}."
        )
    return steps


def resolve_time(
    *,
    duration: Any | None = None,
    dt: Any | None = None,
) -> tuple[float, float]:
    """Resolve time values to an exact ``(duration_ms, dt_ms)`` grid.

    Solvers operate in milliseconds. Plain numeric values are interpreted as
    milliseconds; Pint-like quantities are converted at this boundary.
    """

    if duration is None:
        raise ValueError("duration is required.")
    if dt is None:
        raise ValueError("dt is required.")

    duration_ms = units.to_ms(duration)
    step = units.to_ms(dt)
    if duration_ms <= 0.0:
        raise ValueError("duration must be > 0.")
    if step <= 0.0:
        raise ValueError("dt must be > 0.")
    simulation_step_count(duration_ms, step)
    return duration_ms, step


__all__ = ["resolve_time", "simulation_step_count"]
