"""Pool simulation results exposed through one canonical public surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Sequence, TypeAlias, cast, overload

import numpy as np

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons.axon import Axon
from axonfleet.analysis.core import AnalysisResult
from axonfleet.results.axes import RecordedAxis
from axonfleet.results.common import SingleAxonResultMixin
from axonfleet.signals import MEMBRANE_VOLTAGE, Signal

if TYPE_CHECKING:
    from axonfleet.dispatcher._records import (
        DispatchCohortRecord,
        DispatchRecord,
        DispatchRowRecord,
    )
    from axonfleet.recording import Recording


ResultArray: TypeAlias = Any
RecordingDict: TypeAlias = dict[str, Any]
ObservationDict: TypeAlias = dict[str, Any]


def _dispatch_diagnostics(result: DispatchRowRecord) -> dict[str, Any]:
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


def _dispatch_cohort_diagnostics(result: DispatchCohortRecord) -> tuple[dict[str, Any], ...]:
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


def _recording_value_signature(value: Any) -> tuple[Any, ...]:
    """Return a grouping signature for one recorded value."""

    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (key, _recording_value_signature(value[key]))
                for key in sorted(value)
            ),
        )
    array = np.asarray(value)
    return ("array", tuple(array.shape), str(array.dtype))


def _recordings_signature(recordings: RecordingDict | None) -> tuple[Any, ...] | None:
    """Return a grouping signature for a per-row recording dictionary."""

    if recordings is None:
        return None
    return tuple(
        (key, _recording_value_signature(recordings[key]))
        for key in sorted(recordings)
    )


def _require_signal(signal: Any) -> Signal[Any]:
    if isinstance(signal, str):
        raise TypeError("signal must use axonfleet.signals values, not strings.")
    if not isinstance(signal, Signal):
        raise TypeError("signal must be an axonfleet.Signal descriptor.")
    return signal


def _slice_analysis_result(result: AnalysisResult, row: int) -> AnalysisResult:
    """Return a one-row view of an ``AnalysisResult``."""

    return AnalysisResult(
        name=result.name,
        values=np.asarray(result.values)[row : row + 1],
        statuses=(result.statuses[row],),
        messages=(result.messages[row],),
        unit=result.unit,
        row_labels=(result.row_labels[row],),
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
        row_labels=tuple(label for result in results for label in result.row_labels),
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
    rows: Sequence[DispatchRowRecord],
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
class _ResultBlock:
    """Internal dense result block for compatible pool rows.

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
    observations: ObservationDict | None = None
    recordings: tuple[RecordingDict | None, ...] | None = None
    final_states: tuple[Any | None, ...] | None = None

def _cohort_from_dispatch_cohort(
    result: DispatchCohortRecord,
) -> _ResultBlock:
    """Convert one compact dispatch cohort to a public result cohort."""

    if len(result.record_indices) != len(result.indices):
        raise ValueError("cohort record_indices must align with cohort input indices.")
    final_states = result.final_states
    if final_states is None:
        final_states = tuple(None for _ in result.indices)
    if len(final_states) != len(result.indices):
        raise ValueError("cohort final_states must align with cohort input indices.")
    return _ResultBlock(
        input_indices=tuple(int(index) for index in result.indices),
        axons=tuple(result.axons),
        simulations=tuple(result.simulations),
        Vm=None if result.Vm is None else np.asarray(result.Vm),
        t=np.asarray(result.t),
        diagnostics=_dispatch_cohort_diagnostics(result),
        record_indices=tuple(result.record_indices),
        observations=result.observations,
        recordings=result.recordings,
        final_states=tuple(final_states),
    )


@dataclass(frozen=True)
class RecordingManifest:
    """Structured description of requested and available result signals."""

    requested_signals: tuple[Signal[Any], ...]
    available_signals: tuple[Signal[Any], ...]
    policy: Recording | None = None

    @classmethod
    def from_cohorts(
        cls,
        cohorts: Sequence[_ResultBlock],
        *,
        policy: Recording | None = None,
    ) -> RecordingManifest:
        """Build the recording manifest from dense result cohorts."""

        requested = policy.signals if policy is not None else (MEMBRANE_VOLTAGE,)
        available: list[Signal[Any]] = []
        if any(cohort.Vm is not None for cohort in cohorts):
            available.append(MEMBRANE_VOLTAGE)
        if policy is not None:
            for signal in policy.signals:
                if signal.result_key == MEMBRANE_VOLTAGE.result_key:
                    continue
                has_signal = any(
                    cohort.recordings is not None
                    and any(
                        recordings is not None and signal.result_key in recordings
                        for recordings in cohort.recordings
                    )
                    for cohort in cohorts
                )
                if has_signal:
                    available.append(signal)
        return cls(
            requested_signals=tuple(requested),
            available_signals=tuple(available),
            policy=policy,
        )


@dataclass(frozen=True)
class AxonResultView(SingleAxonResultMixin):
    """One-axon view into an ``AxonSimulationResult``.

    The view exposes the common one-axon result surface while keeping pool data
    stored in dense cohorts.
    """

    parent: AxonSimulationResult
    index: int

    @property
    def _cohort_row(self) -> tuple[_ResultBlock, int]:
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
    def final_state(self) -> Any | None:
        """Final solver state for this row, when retained by the backend."""

        cohort, row = self._cohort_row
        if cohort.final_states is None:
            return None
        return cohort.final_states[row]

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
        return recordings[descriptor.result_key]


class AxonSimulationResult(Sequence[AxonResultView]):
    """Canonical result returned by population simulations.

    Results may use dense storage blocks internally, but the public data path is
    the sequence of lightweight per-axon views in the original input order.
    """

    def __init__(
        self,
        _cohorts: Sequence[_ResultBlock],
        *,
        size: int,
        recording: Recording | None = None,
    ) -> None:
        self._cohorts = tuple(_cohorts)
        self.size = int(size)
        self.recording = recording
        self.recording_manifest = RecordingManifest.from_cohorts(
            self._cohorts,
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
        """Build the public result container from dispatcher records."""

        records = tuple(results)
        if not records:
            raise ValueError("AxonSimulationResult requires at least one dispatch result.")

        cohorts = []
        groups: dict[tuple[Any, ...], list[DispatchRowRecord]] = {}
        size = 0
        for record in records:
            if hasattr(record, "indices"):
                cohort_record = cast("DispatchCohortRecord", record)
                cohorts.append(
                    _cohort_from_dispatch_cohort(cohort_record)
                )
                size += len(cohort_record.indices)
                continue
            row = cast("DispatchRowRecord", record)
            vm = None if row.Vm is None else np.asarray(row.Vm)
            t = np.asarray(row.t)
            record_indices = None if row.record_indices is None else tuple(row.record_indices)
            key = (
                None if vm is None else vm.shape,
                t.shape,
                None if vm is None else str(vm.dtype),
                str(t.dtype),
                record_indices,
                _recordings_signature(row.recordings),
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
                _ResultBlock(
                    input_indices=tuple(int(row.index) for row in grouped_rows),
                    axons=tuple(row.axon for row in grouped_rows),
                    simulations=tuple(row.simulation for row in grouped_rows),
                    Vm=dense_vm,
                    t=np.asarray(first.t),
                    diagnostics=tuple(_dispatch_diagnostics(row) for row in grouped_rows),
                    record_indices=tuple(row.record_indices for row in grouped_rows),
                    observations=_merge_dispatch_observations(grouped_rows),
                    recordings=(
                        tuple(row.recordings for row in grouped_rows)
                        if any(row.recordings is not None for row in grouped_rows)
                        else None
                    ),
                    final_states=tuple(row.final_state for row in grouped_rows),
                )
            )

        return cls(cohorts, size=size, recording=recording)

    def _build_lookup(self) -> tuple[tuple[int, int], ...]:
        lookup: list[tuple[int, int] | None] = [None] * self.size
        for cohort_index, cohort in enumerate(self._cohorts):
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

    def _cohort_row(self, index: int) -> tuple[_ResultBlock, int]:
        normalized = int(index)
        if normalized < 0:
            normalized += self.size
        if normalized < 0 or normalized >= self.size:
            raise IndexError(f"pool result index {index} is out of range.")
        cohort_index, row_index = self._lookup[normalized]
        return self._cohorts[cohort_index], row_index

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[AxonResultView]:
        for index in range(self.size):
            yield AxonResultView(self, index)

    @overload
    def __getitem__(self, index: int) -> AxonResultView:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[AxonResultView, ...]:
        ...

    def __getitem__(self, index: int | slice) -> AxonResultView | tuple[AxonResultView, ...]:
        if isinstance(index, slice):
            return tuple(self[row] for row in range(*index.indices(self.size)))
        normalized = int(index)
        if normalized < 0:
            normalized += self.size
        self._cohort_row(normalized)
        return AxonResultView(self, normalized)

    @property
    def single(self) -> AxonResultView:
        """Return the only result row, or raise if this is a real population."""

        if self.size != 1:
            raise ValueError(f"single requires exactly one row; got {self.size}.")
        return self[0]

    @property
    def recordings(self) -> tuple[RecordingDict | None, ...]:
        """Per-row recording dictionaries in input order."""

        return tuple(view.recordings for view in self)

    @property
    def final_states(self) -> tuple[Any | None, ...]:
        """Per-row final solver states in input order, when retained."""

        return tuple(view.final_state for view in self)

    @property
    def observations(self) -> ObservationDict | None:
        """Solver-side observations in input order, if requested."""

        if len(self._cohorts) == 1:
            cohort = self._cohorts[0]
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

    def signal(self, signal: Any) -> ResultArray:
        """Return one recorded signal as a dense ``(axon, time, position)`` array."""

        descriptor = _require_signal(signal)
        if not any(
            available.id == descriptor.id
            for available in self.recording_manifest.available_signals
        ):
            raise KeyError(f"signal {descriptor.id!s} is not available in this result.")
        if descriptor.result_key != "Vm":
            values = [view.signal(descriptor) for view in self]
            if isinstance(values[0], dict):
                keys = tuple(values[0])
                if any(not isinstance(value, dict) or tuple(value) != keys for value in values):
                    raise ValueError(
                        "grouped signal is heterogeneous across result rows; "
                        "use per-axon views instead."
                    )
                stacked: dict[str, ResultArray] = {}
                for key in keys:
                    arrays = [np.asarray(value[key]) for value in values]
                    shapes = {array.shape for array in arrays}
                    if len(shapes) != 1:
                        raise ValueError(
                            "grouped signal is heterogeneous across result rows; "
                            "use per-axon views instead."
                        )
                    stacked[key] = np.stack(arrays, axis=0)
                return stacked
            shapes = {value.shape for value in values}
            if len(shapes) != 1:
                raise ValueError(
                    "signal is heterogeneous across result rows; use per-axon views instead."
                )
            return np.stack(values, axis=0)
        values = [np.asarray(view.Vm) for view in self]
        shapes = {value.shape for value in values}
        if len(shapes) != 1:
            raise ValueError(
                "signal is heterogeneous across result rows; use per-axon views instead."
            )
        return np.stack(values, axis=0)

    def analyze(self, *definitions: Any) -> Any:
        """Evaluate public analysis definitions on this population result."""

        from axonfleet.analysis import analyze

        return analyze(self, *definitions)

    def report(self, *definitions: Any) -> Any:
        """Return an analysis report for one or more definitions."""

        from axonfleet.analysis import AnalysisReport, analyze

        analyzed = analyze(self, *definitions)
        if isinstance(analyzed, AnalysisReport):
            return analyzed
        return AnalysisReport(analyses=(analyzed,))

    def plot_traces(
        self,
        ax: Any | None = None,
        *,
        position: Any | None = None,
        index: int | None = None,
        labels: Sequence[str] | None = None,
        time_unit: Any = "millisecond",
        voltage_unit: Any = "millivolt",
        title: str = "Population Vm traces",
        grid: bool = True,
        legend: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot one membrane-voltage trace per population row."""

        from axonfleet.results.views import plot_population_traces

        return plot_population_traces(
            self,
            ax=ax,
            position=position,
            index=index,
            labels=labels,
            time_unit=time_unit,
            voltage_unit=voltage_unit,
            title=title,
            grid=grid,
            legend=legend,
            **plot_kwargs,
        )


__all__ = [
    "RecordingManifest",
    "AxonResultView",
    "AxonSimulationResult",
]
