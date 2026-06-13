"""Public recording policy objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

from axonscope.solvers import BatchOptions, BatchRecording
from axonscope.utils import units


RecordingSpatialMode: TypeAlias = Literal["full", "center", "probes", "indices"]
RecordingVariable: TypeAlias = str


def _normalize_variables(
    values: RecordingVariable | Sequence[RecordingVariable] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        normalized = (values.strip(),)
    else:
        normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError("variables must contain at least one entry.")
    return normalized


def _selects(values: tuple[str, ...], *names: str) -> bool:
    wanted = tuple(name.lower() for name in names)
    lowered = tuple(value.lower() for value in values)
    return any(
        value == name or value.startswith(f"{name}.")
        for value in lowered
        for name in wanted
    )


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

    Plain numeric `positions_um` and `sample_dt_ms` values are interpreted in
    micrometers and milliseconds. Pint-like quantities are converted at
    construction time. Pool runs can record all compartments, the center
    compartment, evenly spaced probes, or explicit compartment indices. Spatial
    single-axon filters and temporal filters are descriptive for now; unsupported
    combinations raise at simulation dispatch until the matching solver plumbing
    lands.
    """

    voltage: bool = True
    gates: bool = False
    currents: bool = False
    conductances: bool = False
    state_variables: bool = False
    variables: tuple[str, ...] | None = None
    positions_um: tuple[float, ...] | None = None
    record_indices: tuple[int, ...] | None = None
    sample_dt_ms: float | None = None
    every_n_steps: int | None = None
    spatial_mode: RecordingSpatialMode = "full"
    probe_count: int = 8

    def __init__(
        self,
        *,
        voltage: bool = True,
        gates: bool = False,
        currents: bool = False,
        conductances: bool = False,
        state_variables: bool = False,
        variables: RecordingVariable | Sequence[RecordingVariable] | None = None,
        positions_um: Sequence[Any] | None = None,
        indices: Sequence[int] | None = None,
        sample_dt_ms: Any | None = None,
        every_n_steps: int | None = None,
        spatial_mode: RecordingSpatialMode = "full",
        probe_count: int = 8,
    ) -> None:
        normalized_variables = _normalize_variables(variables)
        normalized_positions_um = (
            None
            if positions_um is None
            else tuple(float(value) for value in units.to_um_array(positions_um))
        )
        normalized_indices = _normalize_indices(indices)
        normalized_sample_dt_ms = (
            None if sample_dt_ms is None else units.to_ms(sample_dt_ms)
        )

        if normalized_variables is not None:
            voltage = _selects(normalized_variables, "vm", "voltage")
            gates = _selects(normalized_variables, "gates")
            currents = _selects(normalized_variables, "currents")
            conductances = _selects(normalized_variables, "conductances")
            state_variables = _selects(normalized_variables, "state_variables", "states")

        if normalized_sample_dt_ms is not None and every_n_steps is not None:
            raise ValueError("sample_dt_ms and every_n_steps are mutually exclusive.")
        if every_n_steps is not None and int(every_n_steps) < 1:
            raise ValueError("every_n_steps must be >= 1.")
        if normalized_sample_dt_ms is not None and normalized_sample_dt_ms <= 0.0:
            raise ValueError("sample_dt_ms must be > 0.")
        if spatial_mode not in {"full", "center", "probes", "indices"}:
            raise ValueError(f"unknown spatial_mode: {spatial_mode!r}.")
        if int(probe_count) < 1:
            raise ValueError("probe_count must be >= 1.")
        if normalized_positions_um is not None and normalized_indices is not None:
            raise ValueError("positions_um and indices are mutually exclusive.")
        if normalized_indices is not None:
            if spatial_mode not in {"full", "indices"}:
                raise ValueError("indices cannot be combined with center/probes modes.")
            spatial_mode = "indices"
        elif spatial_mode == "indices":
            raise ValueError("spatial_mode='indices' requires indices.")

        object.__setattr__(self, "voltage", bool(voltage))
        object.__setattr__(self, "gates", bool(gates))
        object.__setattr__(self, "currents", bool(currents))
        object.__setattr__(self, "conductances", bool(conductances))
        object.__setattr__(self, "state_variables", bool(state_variables))
        object.__setattr__(self, "variables", normalized_variables)
        object.__setattr__(self, "positions_um", normalized_positions_um)
        object.__setattr__(self, "record_indices", normalized_indices)
        object.__setattr__(self, "sample_dt_ms", normalized_sample_dt_ms)
        object.__setattr__(
            self,
            "every_n_steps",
            None if every_n_steps is None else int(every_n_steps),
        )
        object.__setattr__(self, "spatial_mode", spatial_mode)
        object.__setattr__(self, "probe_count", int(probe_count))

    @classmethod
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
    def only(cls, *variables: str) -> "Recording":
        """Record only the named variable groups."""

        return cls(variables=variables)

    @classmethod
    def center(
        cls,
        variables: RecordingVariable | Sequence[RecordingVariable] | None = None,
    ) -> "Recording":
        """Record the central compartment for pool Vm outputs."""

        return cls(variables=variables, spatial_mode="center")

    @classmethod
    def probes(
        cls,
        variables: RecordingVariable | Sequence[RecordingVariable] | None = None,
        *,
        count: int = 8,
    ) -> "Recording":
        """Record up to ``count`` spatial probes for pool Vm outputs."""

        return cls(variables=variables, spatial_mode="probes", probe_count=count)

    @classmethod
    def indices(
        cls,
        values: Sequence[int],
        variables: RecordingVariable | Sequence[RecordingVariable] | None = None,
    ) -> "Recording":
        """Record explicit compartment indices for pool Vm outputs."""

        return cls(variables=variables, indices=values, spatial_mode="indices")

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

        if not self.voltage:
            raise NotImplementedError("pool recording currently requires Vm.")
        if self.wants_observables:
            raise NotImplementedError("pool recording currently supports Vm only.")
        if self.positions_um is not None:
            raise NotImplementedError(
                "position-based batch recording is not wired yet; "
                "use center/probes/indices/full."
            )
        if self.sample_dt_ms is not None or self.every_n_steps is not None:
            raise NotImplementedError("temporal recording subsampling is not wired yet.")
        if self.spatial_mode == "center":
            recording = BatchRecording.center()
        elif self.spatial_mode == "probes":
            recording = BatchRecording.probes(self.probe_count)
        elif self.spatial_mode == "indices":
            if self.record_indices is None:
                raise ValueError("indices recording requires record_indices.")
            recording = BatchRecording.indices(self.record_indices)
        else:
            recording = BatchRecording.full()
        return BatchOptions(recording=recording)


__all__ = ["Recording", "RecordingSpatialMode", "RecordingVariable"]
