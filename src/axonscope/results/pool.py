"""Cohort-backed pool simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Sequence, overload

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.results.single import ObservationDict, RecordingDict, ResultArray, SimResult
from axonscope.signals import MEMBRANE_VOLTAGE, Signal

if TYPE_CHECKING:
    from axonscope.dispatcher.results import DispatchResult
    from axonscope.recording import Recording


def _dispatch_diagnostics(result: DispatchResult) -> dict[str, Any]:
    """Return public diagnostics carried by one dispatched pool row."""

    return {
        "pool_index": result.index,
        "dispatch_group_id": result.group_id,
        "dispatch_method": result.method,
        "dispatch_group_size": result.group_size,
        "dispatch_batch_kind": result.batch_kind,
        "dispatch_geometry_shared": result.geometry_shared,
        "dispatch_has_padding": result.has_padding,
    }


def _require_signal(signal: Any) -> Signal[Any]:
    if isinstance(signal, str):
        raise TypeError("signal must use axonscope.signals values, not strings.")
    if not isinstance(signal, Signal):
        raise TypeError("signal must be an axonscope.Signal descriptor.")
    return signal


@dataclass(frozen=True)
class RecordedSignal:
    """Manifest row for one signal available in one or more dense cohorts."""

    signal: Signal[Any]
    result_key: str
    unit: Any | None
    cohort_indices: tuple[int, ...]
    cohort_shapes: tuple[tuple[int, ...], ...]
    cohort_dtypes: tuple[str, ...]

    @property
    def cohort_count(self) -> int:
        """Number of cohorts carrying this signal."""

        return len(self.cohort_indices)


@dataclass(frozen=True)
class CohortResult:
    """Dense result block for compatible pool rows.

    ``Vm`` is indexed as ``(axon, time, recorded_position)``. ``input_indices``
    maps each dense row back to the original user-provided pool order.
    """

    input_indices: tuple[int, ...]
    axons: tuple[Axon, ...]
    simulations: tuple[AxonInstance, ...]
    Vm: ResultArray
    t: ResultArray
    diagnostics: tuple[dict[str, Any], ...]
    record_indices: tuple[tuple[int, ...] | None, ...]
    recording: Recording | None = None

    @property
    def size(self) -> int:
        """Number of axons stored in this dense cohort."""

        return len(self.input_indices)


@dataclass(frozen=True)
class RecordingManifest:
    """Structured description of requested and available result signals."""

    requested_signals: tuple[Signal[Any], ...]
    available: tuple[RecordedSignal, ...]
    policy: Recording | None = None

    @classmethod
    def from_cohorts(
        cls,
        cohorts: Sequence[CohortResult],
        *,
        policy: Recording | None = None,
    ) -> RecordingManifest:
        """Build the current Vm-only manifest from dense result cohorts."""

        requested = policy.signals if policy is not None else (MEMBRANE_VOLTAGE,)
        available = (
            RecordedSignal(
                signal=MEMBRANE_VOLTAGE,
                result_key=MEMBRANE_VOLTAGE.result_key,
                unit=MEMBRANE_VOLTAGE.unit,
                cohort_indices=tuple(range(len(cohorts))),
                cohort_shapes=tuple(tuple(np.asarray(cohort.Vm).shape) for cohort in cohorts),
                cohort_dtypes=tuple(str(np.asarray(cohort.Vm).dtype) for cohort in cohorts),
            ),
        )
        return cls(
            requested_signals=tuple(requested),
            available=available,
            policy=policy,
        )

    @property
    def available_signals(self) -> tuple[Signal[Any], ...]:
        """Signal descriptors present in the result."""

        return tuple(entry.signal for entry in self.available)

    def signal(self, signal: Any) -> RecordedSignal:
        """Return manifest metadata for one available signal."""

        descriptor = _require_signal(signal)
        for entry in self.available:
            if entry.signal.id == descriptor.id:
                return entry
        raise KeyError(f"signal {descriptor.id!s} is not available in this result.")

    def has(self, signal: Any) -> bool:
        """Return whether a signal is available in this result."""

        try:
            self.signal(signal)
        except KeyError:
            return False
        return True


@dataclass(frozen=True)
class AxonResultView:
    """One-axon view into an ``AxonSimulationResult``.

    The view exposes the same common result surface as ``SimResult`` while
    keeping pool data stored in dense cohorts. Use ``to_sim_result()`` when an
    API requires a standalone mutable ``SimResult`` object.
    """

    parent: AxonSimulationResult
    index: int

    @property
    def _cohort_row(self) -> tuple[CohortResult, int]:
        return self.parent._cohort_row(self.index)

    @property
    def axon(self) -> Axon:
        """Descriptive axon associated with this result row."""

        cohort, row = self._cohort_row
        return cohort.axons[row]

    @property
    def simulation(self) -> AxonInstance:
        """Executable axon instance associated with this result row."""

        cohort, row = self._cohort_row
        return cohort.simulations[row]

    @property
    def Vm(self) -> ResultArray:
        """Membrane voltage matrix indexed as ``(time, recorded_position)``."""

        cohort, row = self._cohort_row
        return cohort.Vm[row]

    @property
    def t(self) -> ResultArray:
        """Time vector in milliseconds."""

        cohort, _ = self._cohort_row
        return cohort.t

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Dispatch diagnostics for this row."""

        cohort, row = self._cohort_row
        return cohort.diagnostics[row]

    @property
    def recording(self) -> Recording | None:
        """Public recording policy used for this pool run, if any."""

        return self.parent.recording

    @property
    def recording_manifest(self) -> RecordingManifest:
        """Structured signal manifest for the parent pool result."""

        return self.parent.recording_manifest

    @property
    def record_indices(self) -> tuple[int, ...] | None:
        """Original compartment indices represented by ``Vm`` columns."""

        cohort, row = self._cohort_row
        return cohort.record_indices[row]

    @property
    def recordings(self) -> RecordingDict:
        """Recording dictionary compatible with ``SimResult.recordings``."""

        return {"Vm": self.Vm}

    @property
    def observations(self) -> ObservationDict | None:
        """Pool views do not currently carry compact solver observations."""

        return None

    def to_sim_result(self) -> SimResult:
        """Materialize this view as a standalone ``SimResult``."""

        return SimResult(
            axon=self.axon,
            Vm=self.Vm,
            t=self.t,
            diagnostics=dict(self.diagnostics),
            recording=self.recording,
            record_indices=self.record_indices,
            simulation=self.simulation,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate less common ``SimResult`` helpers without duplicating them."""

        return getattr(self.to_sim_result(), name)


class AxonSimulationResult(Sequence[AxonResultView]):
    """Canonical result returned by population simulations.

    Results are stored as one or more dense cohorts and exposed through
    lightweight per-axon views in the original input order.
    """

    def __init__(
        self,
        cohorts: Sequence[CohortResult],
        *,
        size: int,
        recording: Recording | None = None,
    ) -> None:
        self.cohorts = tuple(cohorts)
        self.size = int(size)
        self.recording = recording
        self.recording_manifest = RecordingManifest.from_cohorts(
            self.cohorts,
            policy=recording,
        )
        self._lookup = self._build_lookup()

    @classmethod
    def from_dispatch_results(
        cls,
        results: Sequence[DispatchResult],
        *,
        recording: Recording | None = None,
    ) -> AxonSimulationResult:
        """Build a cohort-backed public result from dispatcher rows."""

        rows = tuple(results)
        if not rows:
            raise ValueError("AxonSimulationResult requires at least one dispatch result.")

        groups: dict[tuple[Any, ...], list[DispatchResult]] = {}
        for row in rows:
            vm = np.asarray(row.Vm)
            t = np.asarray(row.t)
            record_indices = None if row.record_indices is None else tuple(row.record_indices)
            key = (vm.shape, t.shape, str(vm.dtype), str(t.dtype), record_indices)
            groups.setdefault(key, []).append(row)

        cohorts = []
        for grouped_rows in groups.values():
            dense_vm = np.stack([np.asarray(row.Vm) for row in grouped_rows], axis=0)
            first = grouped_rows[0]
            cohorts.append(
                CohortResult(
                    input_indices=tuple(int(row.index) for row in grouped_rows),
                    axons=tuple(row.axon for row in grouped_rows),
                    simulations=tuple(row.simulation for row in grouped_rows),
                    Vm=dense_vm,
                    t=np.asarray(first.t),
                    diagnostics=tuple(_dispatch_diagnostics(row) for row in grouped_rows),
                    record_indices=tuple(row.record_indices for row in grouped_rows),
                    recording=recording,
                )
            )

        return cls(cohorts, size=len(rows), recording=recording)

    def _build_lookup(self) -> tuple[tuple[int, int], ...]:
        lookup: list[tuple[int, int] | None] = [None] * self.size
        for cohort_index, cohort in enumerate(self.cohorts):
            for row_index, input_index in enumerate(cohort.input_indices):
                if input_index < 0 or input_index >= self.size:
                    raise ValueError(
                        "cohort input indices must cover pool rows 0..size-1."
                    )
                if lookup[input_index] is not None:
                    raise ValueError(f"duplicate pool result index {input_index}.")
                lookup[input_index] = (cohort_index, row_index)
        missing = [index for index, value in enumerate(lookup) if value is None]
        if missing:
            raise ValueError(f"missing pool result indices: {missing}.")
        return tuple(value for value in lookup if value is not None)

    def _cohort_row(self, index: int) -> tuple[CohortResult, int]:
        normalized = int(index)
        if normalized < 0:
            normalized += self.size
        if normalized < 0 or normalized >= self.size:
            raise IndexError(f"pool result index {index} is out of range.")
        cohort_index, row_index = self._lookup[normalized]
        return self.cohorts[cohort_index], row_index

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[AxonResultView]:
        for index in range(self.size):
            yield self.axon(index)

    @overload
    def __getitem__(self, index: int) -> AxonResultView:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[AxonResultView, ...]:
        ...

    def __getitem__(self, index: int | slice) -> AxonResultView | tuple[AxonResultView, ...]:
        if isinstance(index, slice):
            return tuple(self.axon(row) for row in range(*index.indices(self.size)))
        return self.axon(index)

    @property
    def single(self) -> AxonResultView:
        """Return the only result row, or raise if this is a real population."""

        if self.size != 1:
            raise ValueError(f"single requires exactly one row; got {self.size}.")
        return self.axon(0)

    @property
    def views(self) -> tuple[AxonResultView, ...]:
        """Return all per-axon views in input order."""

        return tuple(self)

    @property
    def axons(self) -> tuple[Axon, ...]:
        """Descriptive axons in input order."""

        return tuple(view.axon for view in self)

    @property
    def simulations(self) -> tuple[AxonInstance, ...]:
        """Executable axon instances in input order."""

        return tuple(view.simulation for view in self)

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Per-row dispatch diagnostics in input order."""

        return tuple(view.diagnostics for view in self)

    def axon(self, index: int) -> AxonResultView:
        """Return the per-axon result view at ``index``."""

        normalized = int(index)
        if normalized < 0:
            normalized += self.size
        self._cohort_row(normalized)
        return AxonResultView(self, normalized)

    def signal(self, signal: Any) -> ResultArray:
        """Return one recorded signal as a dense ``(axon, time, position)`` array."""

        entry = self.recording_manifest.signal(signal)
        if entry.result_key != "Vm":
            raise KeyError(
                f"pool results currently contain Vm only, not {entry.result_key!r}."
            )
        values = [np.asarray(view.Vm) for view in self]
        shapes = {value.shape for value in values}
        if len(shapes) != 1:
            raise ValueError(
                "signal is split across heterogeneous cohorts; use .cohorts "
                "or per-axon views instead."
            )
        return np.stack(values, axis=0)

    def analyze(self, *definitions: Any) -> Any:
        """Evaluate public analysis definitions on this population result."""

        from axonscope.analysis import analyze

        return analyze(self, *definitions)

    def report(self, *definitions: Any) -> Any:
        """Return an analysis report for one or more definitions."""

        from axonscope.analysis import AnalysisReport, analyze

        analyzed = analyze(self, *definitions)
        if isinstance(analyzed, AnalysisReport):
            return analyzed
        return AnalysisReport(
            simulation_result=self,
            analyses=(analyzed,),
        )


__all__ = [
    "RecordedSignal",
    "RecordingManifest",
    "CohortResult",
    "AxonResultView",
    "AxonSimulationResult",
]
