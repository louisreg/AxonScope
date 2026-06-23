"""Host-side row helpers for dispatcher batch preparation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.stimulation import ExtracellularContext

AxonLike = Axon | AxonInstance
ContextBatchRow = ExtracellularContext | Sequence[ExtracellularContext] | None


def extracellular_context_rows(
    axons: Sequence[AxonLike],
) -> tuple[tuple[ExtracellularContext, ...], ...]:
    """Return one enabled extracellular-context row per axon."""

    return tuple(
        tuple(axon.extracellular_contexts)
        if bool(getattr(axon, "use_extracellular", False))
        else ()
        for axon in axons
    )


def x_positions_batch_m(
    axons: Sequence[AxonLike],
    *,
    target_nx: int | None = None,
) -> np.ndarray:
    """Return batched intrinsic axial positions in meters."""

    rows = []
    for axon in axons:
        x_row = (
            np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
        )
        if target_nx is not None:
            x_row = _pad_numpy_space_array(x_row, target_nx=target_nx)
        rows.append(x_row)
    return np.stack(rows, axis=0)


def axon_transverse_positions_um(axons: Sequence[AxonLike]) -> tuple[np.ndarray, np.ndarray]:
    """Return zero transverse offsets for local-intrinsic AxonScope rows."""

    y = np.zeros((len(axons),), dtype=float)
    z = np.zeros((len(axons),), dtype=float)
    return y, z


def scale_extracellular_contexts(
    contexts: Sequence[ExtracellularContext],
    scale: float,
) -> tuple[ExtracellularContext, ...]:
    """Return contexts with their current amplitudes scaled by ``scale``."""

    return tuple(
        ctx.with_electrodes(
            tuple(electrode.with_scaled_stimulus(scale) for electrode in ctx.electrodes)
        )
        for ctx in contexts
    )


def _pad_numpy_space_array(values: np.ndarray, *, target_nx: int) -> np.ndarray:
    """Pad one spatial vector by repeating the final position."""

    arr = np.asarray(values)
    pad_count = int(target_nx) - int(arr.shape[-1])
    if pad_count < 0:
        raise ValueError(
            f"target_nx must be >= array width, got target_nx={target_nx}, "
            f"width={arr.shape[-1]}."
        )
    if pad_count == 0:
        return arr
    if arr.shape[-1] == 0:
        raise ValueError("cannot pad an empty spatial row.")
    return np.pad(arr, (0, pad_count), mode="edge")


__all__ = [
    "AxonLike",
    "ContextBatchRow",
    "axon_transverse_positions_um",
    "extracellular_context_rows",
    "scale_extracellular_contexts",
    "x_positions_batch_m",
]
