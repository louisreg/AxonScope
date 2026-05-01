from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import jax.numpy as jnp

from axonscope.axons.base import AxonBase


OutputMode = Literal["full_trace", "probes", "final_state"]


@dataclass(frozen=True)
class SolverOutputSpec:
    """Solver output shape requested by the caller."""

    mode: OutputMode
    probe_indices: jnp.ndarray | None = None

    @property
    def stores_full_trace(self) -> bool:
        return self.mode == "full_trace"


def resolve_solver_output_spec(
    axon: AxonBase,
    *,
    output_mode: str = "full_trace",
    probe_indices: Sequence[int] | None = None,
    probe_positions_um: Sequence[float] | None = None,
) -> SolverOutputSpec:
    """Normalize output mode aliases and resolve probe positions to indices."""
    mode = _normalize_output_mode(output_mode)
    if mode != "probes":
        if probe_indices is not None or probe_positions_um is not None:
            raise ValueError("Probe selectors are only valid with output_mode='probes'.")
        return SolverOutputSpec(mode=mode)

    if probe_indices is not None and probe_positions_um is not None:
        raise ValueError("Use either probe_indices or probe_positions_um, not both.")
    if probe_indices is None and probe_positions_um is None:
        raise ValueError("output_mode='probes' requires probe_indices or probe_positions_um.")

    if probe_indices is not None:
        idx = [int(i) for i in probe_indices]
    else:
        x = jnp.asarray(axon.x)
        idx = [
            int(jnp.argmin(jnp.abs(x - float(position_um))))
            for position_um in probe_positions_um or ()
        ]

    if not idx:
        raise ValueError("At least one probe index is required.")
    for i in idx:
        if i < 0 or i >= int(axon.Nx):
            raise ValueError(f"Probe index {i} is outside [0, {int(axon.Nx) - 1}].")
    return SolverOutputSpec(
        mode="probes",
        probe_indices=jnp.asarray(idx, dtype=jnp.int32),
    )


def step_voltage_output(Vm: jnp.ndarray, spec: SolverOutputSpec) -> jnp.ndarray:
    if spec.mode == "probes":
        assert spec.probe_indices is not None
        return Vm[spec.probe_indices]
    return Vm


def output_metadata(axon: AxonBase, spec: SolverOutputSpec) -> dict[str, Any]:
    metadata: dict[str, Any] = {"output_mode": spec.mode}
    if spec.mode == "probes":
        assert spec.probe_indices is not None
        idx = tuple(int(i) for i in list(spec.probe_indices))
        x = jnp.asarray(axon.x)
        metadata["probe_indices"] = idx
        metadata["x_um"] = tuple(float(x[i]) for i in idx)
    return metadata


def _normalize_output_mode(output_mode: str) -> OutputMode:
    aliases = {
        "full": "full_trace",
        "full_trace": "full_trace",
        "trace": "full_trace",
        "probes": "probes",
        "probe": "probes",
        "final": "final_state",
        "final_state": "final_state",
    }
    try:
        return aliases[output_mode]
    except KeyError as exc:
        raise ValueError(
            "output_mode must be one of 'full_trace', 'probes', or 'final_state'."
        ) from exc
