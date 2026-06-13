from __future__ import annotations

import numpy as np

from axonscope.stimulation import ExtracellularContext
from axonscope.stimulus import ArrayLike, Stimulus


def evaluate_stimulus_numpy(stimulus: Stimulus, t_query: ArrayLike) -> np.ndarray:
    """Evaluate a descriptive stimulus on a NumPy time grid."""
    tq = np.asarray(t_query, dtype=float)

    if stimulus.mode == "linear":
        return np.interp(tq, stimulus.t, stimulus.y, left=stimulus.y[0], right=stimulus.y[-1])

    idx = np.searchsorted(stimulus.t, tq, side="right") - 1
    idx = np.clip(idx, 0, len(stimulus.y) - 1)
    return stimulus.y[idx]


def evaluate_extracellular_context_numpy(
    ctx: ExtracellularContext,
    x_positions_m: ArrayLike,
    t_ms: ArrayLike,
) -> np.ndarray:
    """Evaluate one extracellular context on a time-position grid."""
    fp = ctx.electrode.footprint(x_positions_m)
    current_A = evaluate_stimulus_numpy(ctx.stimulus, t_ms)
    return current_A[:, None] * fp[None, :]
