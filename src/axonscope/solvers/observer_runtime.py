"""Solver-side online analysis observers.

This module lowers public analysis definitions to compact JAX arrays that can
be carried inside solver scans. The public API remains the analysis definition
objects; this file is only the numerical bridge used by backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from axonscope.analysis.activation import ActivationEvent
from axonscope.analysis.core import AnalysisResult, AnalysisStatus
from axonscope.analysis.definitions import Activation, PeakVoltage
from axonscope.positions import PositionSelector
from axonscope.signals import MEMBRANE_VOLTAGE, Signal
from axonscope.utils import units


ObserverKind = Literal["peak_voltage", "activation"]
ObserverState = tuple[Any, Any, Any, Any, Any, Any, Any, Any]

PEAK_VOLTAGE_KIND = 0
ACTIVATION_KIND = 1


@dataclass(frozen=True)
class SolverObserverPlan:
    """Lowered analysis definitions consumed by solver kernels."""

    definitions: tuple[Any, ...]
    names: tuple[str, ...]
    kinds: tuple[ObserverKind, ...]
    kind_codes: Any
    indices: Any
    mask: Any
    original_indices: Any
    positions_um: Any
    thresholds_mV: Any
    blanking_ms: Any

    @property
    def count(self) -> int:
        """Number of observer definitions in this plan."""

        return len(self.names)


def _require_vm_signal(signal: Any) -> None:
    if not isinstance(signal, Signal) or signal.id != MEMBRANE_VOLTAGE.id:
        raise NotImplementedError(
            "solver-side observers currently support membrane voltage only."
        )


def _observer_kind(definition: Any) -> ObserverKind:
    if isinstance(definition, PeakVoltage):
        return "peak_voltage"
    if isinstance(definition, Activation):
        return "activation"
    raise NotImplementedError(
        "solver-side observers currently support axs.analysis.PeakVoltage and "
        "axs.analysis.Activation."
    )


def build_solver_observer_plan(
    definitions: Any,
    *,
    positions_um: Any,
    original_indices: Any | None = None,
    dtype: Any = jnp.float32,
) -> SolverObserverPlan | None:
    """Lower public observer definitions to compact solver arrays."""

    if definitions is None:
        return None
    observer_defs = tuple(definitions)
    if not observer_defs:
        return None

    positions = np.asarray(positions_um, dtype=float)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("observer positions must be a non-empty 1D array.")

    if original_indices is None:
        originals = np.arange(positions.shape[0], dtype=np.int32)
    else:
        originals = np.asarray(original_indices, dtype=np.int32)
        if originals.shape != positions.shape:
            raise ValueError(
                "observer original_indices must contain one entry per position."
            )

    names: list[str] = []
    kinds: list[ObserverKind] = []
    columns: list[np.ndarray] = []
    thresholds: list[float] = []
    blanking_values: list[float] = []

    for definition in observer_defs:
        kind = _observer_kind(definition)
        _require_vm_signal(getattr(definition, "signal", None))
        target = getattr(definition, "target", None)
        if not isinstance(target, PositionSelector):
            raise TypeError("observer target must be an axonscope PositionSelector.")

        selected = target.columns(
            positions_um=positions,
            original_indices=originals,
        ).astype(np.int32, copy=False)
        if selected.size == 0:
            raise ValueError(f"observer {definition.name!r} selects no positions.")

        names.append(str(definition.name))
        kinds.append(kind)
        columns.append(selected)
        thresholds.append(
            units.to_mV(definition.threshold)
            if kind == "activation"
            else -np.inf
        )
        blanking = units.to_ms(definition.blanking) if kind == "activation" else 0.0
        if blanking < 0.0:
            raise ValueError("observer blanking must be non-negative.")
        blanking_values.append(blanking)

    if len(set(names)) != len(names):
        raise ValueError("solver-side observer names must be unique.")

    width = max(int(values.size) for values in columns)
    index_table = np.zeros((len(columns), width), dtype=np.int32)
    mask_table = np.zeros((len(columns), width), dtype=bool)
    original_table = np.full((len(columns), width), -1, dtype=np.int32)
    position_table = np.full((len(columns), width), np.nan, dtype=float)

    for row, selected in enumerate(columns):
        count = int(selected.size)
        index_table[row, :count] = selected
        mask_table[row, :count] = True
        original_table[row, :count] = originals[selected]
        position_table[row, :count] = positions[selected]

    kind_code_values = np.asarray(
        [PEAK_VOLTAGE_KIND if kind == "peak_voltage" else ACTIVATION_KIND for kind in kinds],
        dtype=np.int32,
    )

    return SolverObserverPlan(
        definitions=observer_defs,
        names=tuple(names),
        kinds=tuple(kinds),
        kind_codes=jnp.asarray(kind_code_values, dtype=jnp.int32),
        indices=jnp.asarray(index_table, dtype=jnp.int32),
        mask=jnp.asarray(mask_table, dtype=bool),
        original_indices=jnp.asarray(original_table, dtype=jnp.int32),
        positions_um=jnp.asarray(position_table, dtype=dtype),
        thresholds_mV=jnp.asarray(thresholds, dtype=dtype),
        blanking_ms=jnp.asarray(blanking_values, dtype=dtype),
    )


def init_observer_state(
    plan: SolverObserverPlan,
    *,
    batch_size: int | None = None,
) -> ObserverState:
    """Return an initial observer carry for scalar or batched scans."""

    shape = (plan.count,) if batch_size is None else (int(batch_size), plan.count)
    dtype = jnp.asarray(plan.thresholds_mV).dtype
    peak_mV = jnp.full(shape, -jnp.inf, dtype=dtype)
    peak_time_ms = jnp.full(shape, jnp.nan, dtype=dtype)
    peak_index = jnp.full(shape, -1, dtype=jnp.int32)
    activated = jnp.zeros(shape, dtype=bool)
    first_time_ms = jnp.full(shape, jnp.nan, dtype=dtype)
    first_position_um = jnp.full(shape, jnp.nan, dtype=dtype)
    first_index = jnp.full(shape, -1, dtype=jnp.int32)
    saw_sample = jnp.zeros(shape, dtype=bool)
    return (
        peak_mV,
        peak_time_ms,
        peak_index,
        activated,
        first_time_ms,
        first_position_um,
        first_index,
        saw_sample,
    )


def update_observer_state_scalar(
    state: ObserverState,
    *,
    vm_mV: Any,
    time_ms: Any,
    kind_codes: Any,
    indices: Any,
    mask: Any,
    original_indices: Any,
    positions_um: Any,
    thresholds_mV: Any,
    blanking_ms: Any,
) -> ObserverState:
    """Update scalar observer state from one Vm time step."""

    selected = jnp.take(vm_mV, indices, axis=0)
    selected = jnp.where(mask, selected, -jnp.inf)
    current_peak = jnp.max(selected, axis=1)
    current_col = jnp.argmax(selected, axis=1)
    gathered_col = current_col[:, None]
    current_index = jnp.take_along_axis(original_indices, gathered_col, axis=1)[:, 0]
    current_position = jnp.take_along_axis(positions_um, gathered_col, axis=1)[:, 0]

    is_peak = kind_codes == PEAK_VOLTAGE_KIND
    is_activation = kind_codes == ACTIVATION_KIND
    eligible = time_ms >= blanking_ms
    sample_seen = is_peak | (is_activation & eligible)

    (
        peak_mV,
        peak_time_ms,
        peak_index,
        activated,
        first_time_ms,
        first_position_um,
        first_index,
        saw_sample,
    ) = state

    better_peak = sample_seen & (current_peak > peak_mV)
    peak_mV = jnp.where(better_peak, current_peak, peak_mV)
    peak_time_ms = jnp.where(better_peak, time_ms, peak_time_ms)
    peak_index = jnp.where(better_peak, current_index, peak_index)

    crossing = (
        is_activation[:, None]
        & eligible[:, None]
        & mask
        & (selected >= thresholds_mV[:, None])
    )
    has_crossing = jnp.any(crossing, axis=1)
    first_col = jnp.argmax(crossing, axis=1)
    first_col_gather = first_col[:, None]
    crossing_index = jnp.take_along_axis(original_indices, first_col_gather, axis=1)[:, 0]
    crossing_position = jnp.take_along_axis(positions_um, first_col_gather, axis=1)[:, 0]
    activate_now = is_activation & (~activated) & has_crossing

    activated = activated | activate_now
    first_time_ms = jnp.where(activate_now, time_ms, first_time_ms)
    first_position_um = jnp.where(activate_now, crossing_position, first_position_um)
    first_index = jnp.where(activate_now, crossing_index, first_index)
    saw_sample = saw_sample | sample_seen

    return (
        peak_mV,
        peak_time_ms,
        peak_index,
        activated,
        first_time_ms,
        first_position_um,
        first_index,
        saw_sample,
    )


def update_observer_state_batch(
    state: ObserverState,
    *,
    vm_mV: Any,
    time_ms: Any,
    kind_codes: Any,
    indices: Any,
    mask: Any,
    original_indices: Any,
    positions_um: Any,
    thresholds_mV: Any,
    blanking_ms: Any,
) -> ObserverState:
    """Update batched observer state from one Vm time step."""

    selected = jnp.take(vm_mV, indices, axis=1)
    selected = jnp.where(mask[None, :, :], selected, -jnp.inf)
    current_peak = jnp.max(selected, axis=2)
    current_col = jnp.argmax(selected, axis=2)
    current_index = jnp.take_along_axis(
        jnp.broadcast_to(original_indices[None, :, :], selected.shape),
        current_col[:, :, None],
        axis=2,
    )[:, :, 0]
    current_position = jnp.take_along_axis(
        jnp.broadcast_to(positions_um[None, :, :], selected.shape),
        current_col[:, :, None],
        axis=2,
    )[:, :, 0]

    is_peak = kind_codes == PEAK_VOLTAGE_KIND
    is_activation = kind_codes == ACTIVATION_KIND
    eligible = time_ms >= blanking_ms
    sample_seen = is_peak[None, :] | (is_activation[None, :] & eligible[None, :])

    (
        peak_mV,
        peak_time_ms,
        peak_index,
        activated,
        first_time_ms,
        first_position_um,
        first_index,
        saw_sample,
    ) = state

    better_peak = sample_seen & (current_peak > peak_mV)
    peak_mV = jnp.where(better_peak, current_peak, peak_mV)
    peak_time_ms = jnp.where(better_peak, time_ms, peak_time_ms)
    peak_index = jnp.where(better_peak, current_index, peak_index)

    crossing = (
        is_activation[None, :, None]
        & eligible[None, :, None]
        & mask[None, :, :]
        & (selected >= thresholds_mV[None, :, None])
    )
    has_crossing = jnp.any(crossing, axis=2)
    first_col = jnp.argmax(crossing, axis=2)
    crossing_index = jnp.take_along_axis(
        jnp.broadcast_to(original_indices[None, :, :], selected.shape),
        first_col[:, :, None],
        axis=2,
    )[:, :, 0]
    crossing_position = jnp.take_along_axis(
        jnp.broadcast_to(positions_um[None, :, :], selected.shape),
        first_col[:, :, None],
        axis=2,
    )[:, :, 0]
    activate_now = is_activation[None, :] & (~activated) & has_crossing

    activated = activated | activate_now
    first_time_ms = jnp.where(activate_now, time_ms, first_time_ms)
    first_position_um = jnp.where(activate_now, crossing_position, first_position_um)
    first_index = jnp.where(activate_now, crossing_index, first_index)
    saw_sample = saw_sample | sample_seen

    return (
        peak_mV,
        peak_time_ms,
        peak_index,
        activated,
        first_time_ms,
        first_position_um,
        first_index,
        saw_sample,
    )


def _maybe_float(value: Any) -> float | None:
    value = float(value)
    return None if not np.isfinite(value) else value


def _maybe_int(value: Any) -> int | None:
    value = int(value)
    return None if value < 0 else value


def finalize_observer_state(
    plan: SolverObserverPlan,
    state: ObserverState,
) -> dict[str, AnalysisResult]:
    """Convert compact solver observer state to public analysis results."""

    (
        peak_mV,
        peak_time_ms,
        peak_index,
        activated,
        first_time_ms,
        first_position_um,
        first_index,
        saw_sample,
    ) = (np.asarray(values) for values in state)

    if peak_mV.ndim == 1:
        peak_mV = peak_mV[None, :]
        peak_time_ms = peak_time_ms[None, :]
        peak_index = peak_index[None, :]
        activated = activated[None, :]
        first_time_ms = first_time_ms[None, :]
        first_position_um = first_position_um[None, :]
        first_index = first_index[None, :]
        saw_sample = saw_sample[None, :]

    observations: dict[str, AnalysisResult] = {}
    for obs_index, (definition, name, kind) in enumerate(
        zip(plan.definitions, plan.names, plan.kinds, strict=True)
    ):
        statuses = tuple(
            AnalysisStatus.VALID if bool(seen) else AnalysisStatus.UNDETERMINED
            for seen in saw_sample[:, obs_index]
        )
        messages = tuple(
            "" if status is AnalysisStatus.VALID else "observer received no eligible samples."
            for status in statuses
        )

        if kind == "peak_voltage":
            values = np.where(
                saw_sample[:, obs_index],
                peak_mV[:, obs_index],
                np.nan,
            )
            observations[name] = AnalysisResult(
                name=name,
                values=values,
                statuses=statuses,
                messages=messages,
                unit="millivolt",
                definition=definition,
            )
            continue

        events = []
        for row in range(peak_mV.shape[0]):
            events.append(
                ActivationEvent(
                    activated=bool(activated[row, obs_index]),
                    first_time_ms=_maybe_float(first_time_ms[row, obs_index]),
                    first_position_um=_maybe_float(first_position_um[row, obs_index]),
                    first_index=_maybe_int(first_index[row, obs_index]),
                    peak_mV=_maybe_float(peak_mV[row, obs_index]),
                    peak_time_ms=_maybe_float(peak_time_ms[row, obs_index]),
                    peak_index=_maybe_int(peak_index[row, obs_index]),
                )
            )

        observations[name] = AnalysisResult(
            name=name,
            values=np.asarray(activated[:, obs_index], dtype=bool),
            statuses=statuses,
            messages=messages,
            definition=definition,
            events=tuple(events),
        )

    return observations


__all__ = [
    "ACTIVATION_KIND",
    "PEAK_VOLTAGE_KIND",
    "ObserverKind",
    "ObserverState",
    "SolverObserverPlan",
    "build_solver_observer_plan",
    "finalize_observer_state",
    "init_observer_state",
    "update_observer_state_batch",
    "update_observer_state_scalar",
]
