"""Public recording policy objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, cast

from axonscope.signals import (
    CONDUCTANCES,
    CURRENTS,
    GATES,
    MEMBRANE_VOLTAGE,
    STATE_VARIABLES,
    Signal,
    SignalSelection,
)
from axonscope.solvers import BatchOptions, BatchRecording
from axonscope.utils import units


class RecordingSpatial(Enum):
    """Closed set of spatial retention policies for public recordings."""

    FULL = "full"
    CENTER = "center"
    PROBES = "probes"
    INDICES = "indices"


def _normalize_signals(
    values: SignalSelection | None,
) -> tuple[Signal, ...] | None:
    if values is None:
        return None
    if isinstance(values, Signal):
        return (values,)
    if isinstance(values, str):
        raise TypeError("signals must use axonscope.signals values, not strings.")
    try:
        candidates = tuple(values)
    except TypeError as exc:
        raise TypeError("signals must be a Signal or a sequence of Signal values.") from exc
    if not candidates:
        raise ValueError("signals must contain at least one entry.")
    invalid = [
        f"{index}: {type(value).__name__}"
        for index, value in enumerate(candidates)
        if not isinstance(value, Signal)
    ]
    if invalid:
        detail = ", ".join(invalid)
        raise TypeError(f"signals contains invalid entries: {detail}.")

    deduplicated: list[Signal] = []
    for signal in candidates:
        if signal not in deduplicated:
            deduplicated.append(signal)
    return tuple(deduplicated)


def _signals_from_flags(
    *,
    voltage: bool,
    gates: bool,
    currents: bool,
    conductances: bool,
    state_variables: bool,
) -> tuple[Signal, ...]:
    signals: list[Signal] = []
    if voltage:
        signals.append(MEMBRANE_VOLTAGE)
    if gates:
        signals.append(GATES)
    if currents:
        signals.append(CURRENTS)
    if conductances:
        signals.append(CONDUCTANCES)
    if state_variables:
        signals.append(STATE_VARIABLES)
    return tuple(signals)


def _normalize_indices(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    normalized = tuple(int(value) for value in values)
    if not normalized:
        raise ValueError("indices must contain at least one compartment index.")
    if any(value < 0 for value in normalized):
        raise ValueError("indices must be non-negative compartment indices.")
    return normalized


@dataclass(frozen=True)
class Recording:
    """Public recording policy for simulation outputs.

    `signals` must use typed values from `axonscope.signals`; raw strings are
    not accepted as recording selectors. `positions` must carry length units
    and is stored internally as `positions_um`. `sample_dt` must carry time
    units and is stored internally as `sample_dt_ms`. Pint-like quantities are
    converted at construction time. Pool runs can record all compartments, the
    center compartment, evenly spaced probes, or explicit compartment indices.
    Spatial single-axon filters and temporal filters are descriptive for now;
    unsupported combinations raise at simulation dispatch until the matching
    solver plumbing lands.
    """

    voltage: bool = True
    gates: bool = False
    currents: bool = False
    conductances: bool = False
    state_variables: bool = False
    signals: tuple[Signal, ...] = (MEMBRANE_VOLTAGE,)
    positions_um: tuple[float, ...] | None = None
    record_indices: tuple[int, ...] | None = None
    sample_dt_ms: float | None = None
    every_n_steps: int | None = None
    spatial: RecordingSpatial = RecordingSpatial.FULL
    probe_count: int = 8

    def __init__(
        self,
        *,
        voltage: bool = True,
        gates: bool = False,
        currents: bool = False,
        conductances: bool = False,
        state_variables: bool = False,
        signals: SignalSelection | None = None,
        positions: Sequence[Any] | None = None,
        indices: Sequence[int] | None = None,
        sample_dt: Any | None = None,
        every_n_steps: int | None = None,
        spatial: RecordingSpatial = RecordingSpatial.FULL,
        probe_count: int = 8,
    ) -> None:
        normalized_signals = _normalize_signals(signals)
        normalized_positions_um = None
        if positions is not None:
            try:
                normalized_positions_um = tuple(
                    float(value)
                    for value in units.require_length_array_um(
                        cast(Any, positions),
                        name="positions",
                        dtype=float,
                    )
                )
            except TypeError:
                normalized_positions_um = tuple(
                    units.require_length_um(value, name="positions")
                    for value in positions
                )
        normalized_indices = _normalize_indices(indices)
        normalized_sample_dt_ms = (
            None if sample_dt is None else units.require_time_ms(sample_dt, name="sample_dt")
        )

        if normalized_signals is not None:
            selected = set(normalized_signals)
            voltage = MEMBRANE_VOLTAGE in selected
            gates = GATES in selected
            currents = CURRENTS in selected
            conductances = CONDUCTANCES in selected
            state_variables = STATE_VARIABLES in selected
        else:
            normalized_signals = _signals_from_flags(
                voltage=bool(voltage),
                gates=bool(gates),
                currents=bool(currents),
                conductances=bool(conductances),
                state_variables=bool(state_variables),
            )

        if normalized_sample_dt_ms is not None and every_n_steps is not None:
            raise ValueError("sample_dt and every_n_steps are mutually exclusive.")
        if every_n_steps is not None and int(every_n_steps) < 1:
            raise ValueError("every_n_steps must be >= 1.")
        if normalized_sample_dt_ms is not None and normalized_sample_dt_ms <= 0.0:
            raise ValueError("sample_dt must be > 0.")
        if not isinstance(spatial, RecordingSpatial):
            raise TypeError("spatial must be a RecordingSpatial value.")
        if int(probe_count) < 1:
            raise ValueError("probe_count must be >= 1.")
        if normalized_positions_um is not None and normalized_indices is not None:
            raise ValueError("positions and indices are mutually exclusive.")
        if normalized_indices is not None:
            if spatial not in {RecordingSpatial.FULL, RecordingSpatial.INDICES}:
                raise ValueError("indices cannot be combined with center/probes modes.")
            spatial = RecordingSpatial.INDICES
        elif spatial is RecordingSpatial.INDICES:
            raise ValueError("RecordingSpatial.INDICES requires indices.")

        object.__setattr__(self, "voltage", bool(voltage))
        object.__setattr__(self, "gates", bool(gates))
        object.__setattr__(self, "currents", bool(currents))
        object.__setattr__(self, "conductances", bool(conductances))
        object.__setattr__(self, "state_variables", bool(state_variables))
        object.__setattr__(self, "signals", normalized_signals)
        object.__setattr__(self, "positions_um", normalized_positions_um)
        object.__setattr__(self, "record_indices", normalized_indices)
        object.__setattr__(self, "sample_dt_ms", normalized_sample_dt_ms)
        object.__setattr__(
            self,
            "every_n_steps",
            None if every_n_steps is None else int(every_n_steps),
        )
        object.__setattr__(self, "spatial", spatial)
        object.__setattr__(self, "probe_count", int(probe_count))

    @classmethod  # type: ignore[no-redef]
    def voltage(cls) -> "Recording":
        """Record only membrane voltage."""

        return cls(voltage=True)

    @classmethod
    def full(cls) -> "Recording":
        """Record voltage and all currently exposed observable groups."""

        return cls(
            voltage=True,
            gates=True,
            currents=True,
            conductances=True,
            state_variables=True,
        )

    @classmethod
    def none(cls) -> "Recording":
        """Request no stored outputs.

        This is reserved for future observer-only runs; current public solvers
        still require voltage storage.
        """

        return cls(voltage=False)

    @classmethod
    def only(cls, *signals: Signal) -> "Recording":
        """Record only the requested signal groups."""

        return cls(signals=signals)

    @classmethod
    def center(
        cls,
        signals: SignalSelection = MEMBRANE_VOLTAGE,
    ) -> "Recording":
        """Record the central compartment for pool Vm outputs."""

        return cls(signals=signals, spatial=RecordingSpatial.CENTER)

    @classmethod
    def probes(
        cls,
        signals: SignalSelection = MEMBRANE_VOLTAGE,
        *,
        count: int = 8,
    ) -> "Recording":
        """Record up to ``count`` spatial probes for pool Vm outputs."""

        return cls(signals=signals, spatial=RecordingSpatial.PROBES, probe_count=count)

    @classmethod
    def indices(
        cls,
        values: Sequence[int],
        signals: SignalSelection = MEMBRANE_VOLTAGE,
    ) -> "Recording":
        """Record explicit compartment indices for pool Vm outputs."""

        return cls(signals=signals, indices=values, spatial=RecordingSpatial.INDICES)

    @property
    def wants_observables(self) -> bool:
        """Return whether non-voltage observables were requested."""

        return bool(
            self.gates
            or self.currents
            or self.conductances
            or self.state_variables
        )

    def to_batch_options(self) -> BatchOptions:
        """Translate this public policy to the current pool batch options."""

        if self.wants_observables:
            raise NotImplementedError("pool recording currently supports Vm only.")
        if not self.voltage:
            raise NotImplementedError("pool recording currently requires Vm.")
        if self.positions_um is not None:
            raise NotImplementedError(
                "position-based batch recording is not wired yet; "
                "use center/probes/indices/full."
            )
        if self.sample_dt_ms is not None or self.every_n_steps is not None:
            raise NotImplementedError("temporal recording subsampling is not wired yet.")
        if self.spatial is RecordingSpatial.CENTER:
            recording = BatchRecording.center()
        elif self.spatial is RecordingSpatial.PROBES:
            recording = BatchRecording.probes(self.probe_count)
        elif self.spatial is RecordingSpatial.INDICES:
            if self.record_indices is None:
                raise ValueError("indices recording requires record_indices.")
            recording = BatchRecording.indices(self.record_indices)
        else:
            recording = BatchRecording.full()
        return BatchOptions(recording=recording)


__all__ = ["Recording", "RecordingSpatial"]
