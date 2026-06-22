"""Cohort-backed pool simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Sequence, cast, overload

import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.analysis.core import AnalysisResult
from axonscope.results.axes import RecordedAxis
from axonscope.results.common import SingleAxonResultMixin
from axonscope.results.single import ObservationDict, RecordingDict, ResultArray
from axonscope.signals import MEMBRANE_VOLTAGE, Signal

if TYPE_CHECKING:
    from axonscope.dispatcher.results import DispatchCohortResult, DispatchRecord, DispatchResult
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


def _dispatch_cohort_diagnostics(result: DispatchCohortResult) -> tuple[dict[str, Any], ...]:
    """Return public diagnostics carried by one compact dispatched cohort."""

    return tuple(
        {
            "pool_index": int(index),
            "dispatch_group_id": result.group_id,
            "dispatch_method": result.method,
            "dispatch_group_size": result.group_size,
            "dispatch_batch_kind": result.batch_kind,
            "dispatch_geometry_shared": result.geometry_shared,
            "dispatch_has_padding": result.has_padding,
        }
        for index in result.indices
    )


def _require_signal(signal: Any) -> Signal[Any]:
    if isinstance(signal, str):
        raise TypeError("signal must use axonscope.signals values, not strings.")
    if not isinstance(signal, Signal):
        raise TypeError("signal must be an axonscope.Signal descriptor.")
    return signal


def _slice_analysis_result(result: AnalysisResult, row: int) -> AnalysisResult:
    """Return a one-row view of an ``AnalysisResult``."""

    return AnalysisResult(
        name=result.name,
        values=np.asarray(result.values)[row : row + 1],
        statuses=(result.statuses[row],),
        messages=(result.messages[row],),
        unit=result.unit,
        definition=result.definition,
        events=(result.events[row],),
        input_requirements=(result.input_requirements[row],),
    )


def _slice_observation_result(result: Any, row: int) -> Any:
    """Return a one-row view for supported solver-side observation objects."""

    if isinstance(result, AnalysisResult):
        return _slice_analysis_result(result, row)
    slice_batch = getattr(result, "slice_batch", None)
    if callable(slice_batch):
        return slice_batch(row)
    raise TypeError(f"unsupported observation result type: {type(result).__name__}")


def _merge_analysis_results(results: Sequence[AnalysisResult]) -> AnalysisResult:
    """Concatenate one-row analysis results along their population axis."""

    if not results:
        raise ValueError("at least one analysis result is required.")
    first = results[0]
    return AnalysisResult(
        name=first.name,
        values=np.concatenate([np.asarray(result.values) for result in results], axis=0),
        statuses=tuple(status for result in results for status in result.statuses),
        messages=tuple(message for result in results for message in result.messages),
        unit=first.unit,
        definition=first.definition,
        events=tuple(event for result in results for event in result.events),
        input_requirements=tuple(
            requirement
            for result in results
            for requirement in result.input_requirements
        ),
    )


def _merge_observation_results(results: Sequence[Any]) -> Any:
    """Merge one-row observation results along their population axis."""

    if not results:
        raise ValueError("at least one observation result is required.")
    first = results[0]
    if isinstance(first, AnalysisResult):
        return _merge_analysis_results(cast(Sequence[AnalysisResult], results))
    concat_batch = getattr(type(first), "concat_batch", None)
    if callable(concat_batch):
        return concat_batch(results)
    raise TypeError(f"unsupported observation result type: {type(first).__name__}")


def _merge_dispatch_observations(
    rows: Sequence[DispatchResult],
) -> ObservationDict | None:
    """Merge per-dispatch-row observations into one cohort dictionary."""

    row_observations = [row.observations for row in rows]
    if not any(row_observations):
        return None
    if any(observation is None for observation in row_observations):
        raise ValueError("dispatch rows must either all carry observations or none do.")
    if all(row.observations_are_batched for row in rows):
        first = row_observations[0]
        if not all(observation is first for observation in row_observations):
            raise ValueError("batched dispatch observations must share one object.")
        return first

    names = tuple(row_observations[0].keys())  # type: ignore[union-attr]
    merged: ObservationDict = {}
    for name in names:
        merged[name] = _merge_observation_results(
            [observation[name] for observation in row_observations if observation is not None]
        )
    return merged


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
    Vm: ResultArray | None
    t: ResultArray
    diagnostics: tuple[dict[str, Any], ...]
    record_indices: tuple[tuple[int, ...] | None, ...]
    recording: Recording | None = None
    observations: ObservationDict | None = None
    recordings: tuple[RecordingDict | None, ...] | None = None

    @property
    def size(self) -> int:
        """Number of axons stored in this dense cohort."""

        return len(self.input_indices)


def _cohort_from_dispatch_cohort(
    result: DispatchCohortResult,
    *,
    recording: Recording | None,
) -> CohortResult:
    """Convert one compact dispatch cohort to a public result cohort."""

    if len(result.record_indices) != len(result.indices):
        raise ValueError("cohort record_indices must align with cohort input indices.")
    return CohortResult(
        input_indices=tuple(int(index) for index in result.indices),
        axons=tuple(result.axons),
        simulations=tuple(result.simulations),
        Vm=None if result.Vm is None else np.asarray(result.Vm),
        t=np.asarray(result.t),
        diagnostics=_dispatch_cohort_diagnostics(result),
        record_indices=tuple(result.record_indices),
        recording=recording,
        observations=result.observations,
    )


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
        vm_cohort_indices = tuple(
            index for index, cohort in enumerate(cohorts) if cohort.Vm is not None
        )
        available: tuple[RecordedSignal, ...] = ()
        if vm_cohort_indices:
            available = (
                RecordedSignal(
                    signal=MEMBRANE_VOLTAGE,
                    result_key=MEMBRANE_VOLTAGE.result_key,
                    unit=MEMBRANE_VOLTAGE.unit,
                    cohort_indices=vm_cohort_indices,
                    cohort_shapes=tuple(
                        tuple(np.asarray(cohorts[index].Vm).shape)
                        for index in vm_cohort_indices
                    ),
                    cohort_dtypes=tuple(
                        str(np.asarray(cohorts[index].Vm).dtype)
                        for index in vm_cohort_indices
                    ),
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
class AxonResultView(SingleAxonResultMixin):
    """One-axon view into an ``AxonSimulationResult``.

    The view exposes the common one-axon result surface while keeping pool data
    stored in dense cohorts.
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
        if cohort.Vm is not None:
            return cohort.Vm[row]
        if cohort.recordings is not None:
            recordings = cohort.recordings[row]
            if recordings is not None and "Vm" in recordings:
                vm = recordings["Vm"]
                if isinstance(vm, dict):
                    raise TypeError("recordings['Vm'] must be an array, not a group.")
                return vm
        raise ValueError("this pool result row does not contain a Vm recording.")

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
    def recordings(self) -> RecordingDict | None:
        """Recording dictionary for this row."""

        cohort, row = self._cohort_row
        if cohort.recordings is not None:
            return cohort.recordings[row]
        if cohort.Vm is None:
            return None
        return {"Vm": self.Vm}

    @property
    def observations(self) -> ObservationDict | None:
        """Solver-side observations for this row, if requested."""

        cohort, row = self._cohort_row
        if cohort.observations is None:
            return None
        return {
            name: _slice_observation_result(observation, row)
            for name, observation in cohort.observations.items()
        }

    @property
    def recorded_axis(self) -> RecordedAxis:
        """Recorded intrinsic spatial axis for this row's ``Vm`` columns."""

        return RecordedAxis.from_result(self)

    def signal(self, signal: Any) -> ResultArray:
        """Return one recorded signal by public signal descriptor."""

        descriptor = _require_signal(signal)
        recordings = self.recordings
        if recordings is None or descriptor.result_key not in recordings:
            raise KeyError(f"signal {descriptor.id!s} is not available in this result.")
        value = recordings[descriptor.result_key]
        if isinstance(value, dict):
            raise TypeError(
                f"recordings[{descriptor.result_key!r}] is a grouped recording, not an array."
            )
        return value


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
        results: Sequence[DispatchRecord],
        *,
        recording: Recording | None = None,
    ) -> AxonSimulationResult:
        """Build a cohort-backed public result from dispatcher records."""

        records = tuple(results)
        if not records:
            raise ValueError("AxonSimulationResult requires at least one dispatch result.")

        cohorts = []
        groups: dict[tuple[Any, ...], list[DispatchResult]] = {}
        size = 0
        for record in records:
            if hasattr(record, "indices"):
                cohort_record = cast("DispatchCohortResult", record)
                cohorts.append(
                    _cohort_from_dispatch_cohort(cohort_record, recording=recording)
                )
                size += len(cohort_record.indices)
                continue
            row = cast("DispatchResult", record)
            vm = None if row.Vm is None else np.asarray(row.Vm)
            t = np.asarray(row.t)
            record_indices = None if row.record_indices is None else tuple(row.record_indices)
            key = (
                None if vm is None else vm.shape,
                t.shape,
                None if vm is None else str(vm.dtype),
                str(t.dtype),
                record_indices,
                tuple(sorted((row.observations or {}).keys())),
                id(row.observations) if row.observations_are_batched else None,
            )
            groups.setdefault(key, []).append(row)
            size += 1

        for grouped_rows in groups.values():
            dense_vm = (
                None
                if grouped_rows[0].Vm is None
                else np.stack([np.asarray(row.Vm) for row in grouped_rows], axis=0)
            )
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
                    observations=_merge_dispatch_observations(grouped_rows),
                )
            )

        return cls(cohorts, size=size, recording=recording)

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

    @property
    def observations(self) -> ObservationDict | None:
        """Solver-side observations in input order, if requested."""

        if len(self.cohorts) == 1:
            cohort = self.cohorts[0]
            if cohort.input_indices == tuple(range(self.size)):
                return cohort.observations

        row_observations = [view.observations for view in self]
        if not any(row_observations):
            return None
        if any(observation is None for observation in row_observations):
            raise ValueError("pool observation rows are incomplete.")
        names = tuple(row_observations[0].keys())  # type: ignore[union-attr]
        return {
            name: _merge_observation_results(
                [observation[name] for observation in row_observations if observation is not None]
            )
            for name in names
        }

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
