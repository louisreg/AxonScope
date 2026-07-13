"""Public post-hoc analysis definition objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope.analysis.core import (
    AnalysisNotApplicableError,
    AnalysisRequirements,
    AnalysisResult,
    AnalysisStatus,
    MissingAnalysisInputError,
)
from axonscope.analysis.activation import ActivationCriterion, ActivationEvent
from axonscope.analysis.posthoc import conduction_velocity, rasterize
from axonscope.positions import ALL, DISTAL, PositionSelector
from axonscope.results.pool import AxonSimulationResult
from axonscope.signals import MEMBRANE_VOLTAGE, Signal
from axonscope.utils import units


_ANY_MYELINATION = ("unmyelinated", "myelinated")
_ANY_FORMULATION = ("single-cable", "double-cable")
_VM_RECORDING_HINT = (
    "Run the simulation with a Recording that includes axs.signals.Vm at the "
    "positions required by this analysis."
)


def _result_rows(result: Any) -> tuple[Any, ...]:
    if isinstance(result, AxonSimulationResult):
        return tuple(result)
    return (result,)


def _result_row_labels(result: Any) -> tuple[Any, ...]:
    rows = _result_rows(result)
    return tuple(getattr(row, "index", index) for index, row in enumerate(rows))


def _is_missing_input_error(exc: ValueError) -> bool:
    text = str(exc).lower()
    markers = (
        "record",
        "recording",
        "position",
        "positions",
        "indices",
        "vm",
        "time",
        "signal",
    )
    return any(marker in text for marker in markers)


def _missing(
    message: str,
    *,
    signal: Signal[Any],
    fields: tuple[str, ...] = ("Vm",),
    positions: tuple[Any, ...] = (),
) -> MissingAnalysisInputError:
    return MissingAnalysisInputError(
        message,
        required_signals=(signal,),
        required_result_fields=fields,
        required_positions=positions,
        recording_hint=_VM_RECORDING_HINT,
    )


def _require_membrane_voltage(row: Any, signal: Signal[Any]) -> np.ndarray:
    if not isinstance(signal, Signal):
        raise TypeError("analysis signal must be an axonscope.Signal descriptor.")
    if signal.id != MEMBRANE_VOLTAGE.id:
        raise AnalysisNotApplicableError(
            "this analysis currently supports membrane voltage only."
        )
    try:
        values = row.voltage_values(unit="millivolt")
    except (AttributeError, KeyError, TypeError) as exc:
        raise _missing(
            "analysis requires a membrane-voltage recording.",
            signal=signal,
        ) from exc

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"membrane-voltage recording must be 2D, got {values.shape}.")
    return values


def _time_values(row: Any) -> np.ndarray:
    try:
        time_ms = row.time_values(unit="millisecond")
    except (AttributeError, KeyError, TypeError) as exc:
        raise _missing(
            "analysis requires a result time vector.",
            signal=MEMBRANE_VOLTAGE,
            fields=("t",),
        ) from exc
    time_ms = np.asarray(time_ms, dtype=float)
    if time_ms.ndim != 1:
        raise ValueError(f"result time vector must be 1D, got {time_ms.shape}.")
    return time_ms


def _selected_columns(
    row: Any,
    *,
    target: PositionSelector,
    vm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(target, PositionSelector):
        raise TypeError("target must be an axonscope.positions.PositionSelector.")
    try:
        positions_um = np.asarray(row.position_values(unit="micrometer"), dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _missing(
            "analysis requires recorded position metadata.",
            signal=MEMBRANE_VOLTAGE,
            fields=("positions", "record_indices"),
            positions=(target,),
        ) from exc
    if positions_um.shape != (vm.shape[1],):
        raise MissingAnalysisInputError(
            "recorded positions must match membrane-voltage columns.",
            required_signals=(MEMBRANE_VOLTAGE,),
            required_result_fields=("Vm", "positions", "record_indices"),
            required_positions=(target,),
            recording_hint=_VM_RECORDING_HINT,
        )
    if row.record_indices is None:
        original_indices = np.arange(vm.shape[1], dtype=int)
    else:
        original_indices = np.asarray(row.record_indices, dtype=int)
    try:
        columns = target.columns(
            positions_um=positions_um,
            original_indices=original_indices,
        )
    except ValueError as exc:
        raise _missing(
            str(exc),
            signal=MEMBRANE_VOLTAGE,
            fields=("Vm", "record_indices"),
            positions=(target,),
        ) from exc
    return columns, original_indices[columns], positions_um[columns]


def _evaluate_rows(
    definition: Any,
    result: Any,
    evaluator: Callable[[Any], tuple[Any, AnalysisStatus, str, Any | None]],
    *,
    unit: Any | None,
) -> AnalysisResult:
    values: list[Any] = []
    statuses: list[AnalysisStatus] = []
    messages: list[str] = []
    events: list[Any | None] = []
    input_requirements: list[Any | None] = []

    for row in _result_rows(result):
        try:
            value, status, message, event = evaluator(row)
        except MissingAnalysisInputError as exc:
            value, status, message, event = np.nan, AnalysisStatus.MISSING_INPUT, str(exc), None
            input_requirement = exc.requirement
        except AnalysisNotApplicableError as exc:
            value, status, message, event = (
                np.nan,
                AnalysisStatus.NOT_APPLICABLE,
                str(exc),
                None,
            )
            input_requirement = None
        except ValueError as exc:
            if _is_missing_input_error(exc):
                value, status, message, event = (
                    np.nan,
                    AnalysisStatus.MISSING_INPUT,
                    str(exc),
                    None,
                )
                input_requirement = MissingAnalysisInputError(
                    str(exc),
                    required_signals=definition.requirements.required_signals,
                    required_result_fields=definition.requirements.required_result_fields,
                    required_positions=definition.requirements.required_positions,
                    recording_hint=definition.requirements.recording_hint,
                ).requirement
            else:
                value, status, message, event = (
                    np.nan,
                    AnalysisStatus.NUMERICAL_FAILURE,
                    str(exc),
                    None,
                )
                input_requirement = None
        except FloatingPointError as exc:
            value, status, message, event = (
                np.nan,
                AnalysisStatus.NUMERICAL_FAILURE,
                str(exc),
                None,
            )
            input_requirement = None
        else:
            input_requirement = None
        values.append(value)
        statuses.append(status)
        messages.append(message)
        events.append(event)
        input_requirements.append(input_requirement)

    return AnalysisResult(
        name=definition.name,
        values=np.asarray(values),
        statuses=tuple(statuses),
        messages=tuple(messages),
        unit=unit,
        row_labels=_result_row_labels(result),
        definition=definition,
        events=tuple(events),
        input_requirements=tuple(input_requirements),
    )


def _activation_event(
    row: Any,
    *,
    threshold: Any,
    blanking: Any,
    target: PositionSelector,
) -> ActivationEvent:
    try:
        return ActivationCriterion(
            threshold=threshold,
            blanking=blanking,
            target=target,
        ).evaluate(row)
    except (AttributeError, KeyError, TypeError) as exc:
        raise _missing(
            "activation requires membrane voltage, time, and position recordings.",
            signal=MEMBRANE_VOLTAGE,
        ) from exc


def _activation_fast_population(
    definition: Any,
    result: Any,
) -> AnalysisResult | None:
    """Vectorized Activation fast path for dense population result cohorts."""

    if not isinstance(result, AxonSimulationResult):
        return None
    signal = getattr(definition, "signal", MEMBRANE_VOLTAGE)
    if not isinstance(signal, Signal) or signal.id != MEMBRANE_VOLTAGE.id:
        return None
    target = getattr(definition, "target", None)
    if not isinstance(target, PositionSelector):
        return None

    try:
        threshold_mV = units.to_mV(getattr(definition, "threshold"))
        blanking_ms = units.to_ms(getattr(definition, "blanking"))
    except (TypeError, ValueError):
        return None
    if blanking_ms < 0.0:
        return None

    values = np.zeros(result.size, dtype=bool)
    statuses = [AnalysisStatus.VALID] * result.size
    messages = [""] * result.size
    events: list[ActivationEvent | None] = [None] * result.size
    requirements: list[Any | None] = [None] * result.size

    cohorts = tuple(getattr(result, "_cohorts", ()))
    if not cohorts:
        return None
    try:
        for cohort in cohorts:
            if getattr(cohort, "Vm", None) is None:
                return None
            _evaluate_activation_cohort_fast(
                cohort,
                threshold_mV=threshold_mV,
                blanking_ms=blanking_ms,
                target=target,
                values=values,
                events=events,
            )
    except (AttributeError, TypeError, ValueError, FloatingPointError):
        return None

    return AnalysisResult(
        name=definition.name,
        values=values,
        statuses=tuple(statuses),
        messages=tuple(messages),
        unit=None,
        row_labels=_result_row_labels(result),
        definition=definition,
        events=tuple(events),
        input_requirements=tuple(requirements),
    )


def _evaluate_activation_cohort_fast(
    cohort: Any,
    *,
    threshold_mV: float,
    blanking_ms: float,
    target: PositionSelector,
    values: np.ndarray,
    events: list[ActivationEvent | None],
) -> None:
    vm = np.asarray(cohort.Vm)
    if vm.ndim != 3:
        raise ValueError(f"population Vm must be 3D, got {vm.shape}.")
    time_ms = np.asarray(cohort.t, dtype=float)
    if time_ms.ndim != 1 or time_ms.shape[0] != vm.shape[1]:
        raise ValueError("cohort time vector must match Vm time axis.")
    if time_ms.size == 0 or vm.shape[2] == 0:
        raise ValueError("cohort Vm must include time and position samples.")

    groups = _activation_cohort_groups(cohort, width=vm.shape[2], target=target)
    eligible_start = int(np.searchsorted(time_ms, float(blanking_ms), side="left"))
    has_eligible_times = eligible_start < time_ms.shape[0]
    time_slice = slice(eligible_start, None) if has_eligible_times else slice(None)
    window_time_ms = time_ms[time_slice]

    for group in groups:
        rows = np.asarray(group["rows"], dtype=int)
        input_indices = np.asarray(group["input_indices"], dtype=int)
        selected = np.asarray(group["selected"], dtype=int)
        original_indices = np.asarray(group["original_indices"], dtype=int)
        positions_um = np.asarray(group["positions_um"], dtype=float)
        window = _activation_window_view(
            vm,
            rows=rows,
            time_slice=time_slice,
            selected=selected,
        )
        flat = window.reshape(window.shape[0], -1)
        peak_flat = np.argmax(flat, axis=1)
        peak_time = peak_flat // selected.size
        peak_col = peak_flat % selected.size

        if has_eligible_times:
            crossing = window >= float(threshold_mV)
            active = np.any(crossing, axis=(1, 2))
            crossing_flat = crossing.reshape(crossing.shape[0], -1)
            first_flat = np.argmax(crossing_flat, axis=1)
        else:
            active = np.zeros(rows.shape[0], dtype=bool)
            first_flat = np.zeros(rows.shape[0], dtype=int)

        for local_row, input_index in enumerate(input_indices):
            activated = bool(active[local_row])
            if activated:
                first_time = int(first_flat[local_row]) // selected.size
                first_col = int(first_flat[local_row]) % selected.size
                first_time_ms = float(window_time_ms[first_time])
                first_position_um = float(positions_um[first_col])
                first_index = int(original_indices[first_col])
            else:
                first_time_ms = None
                first_position_um = None
                first_index = None
            values[int(input_index)] = activated
            events[int(input_index)] = ActivationEvent(
                activated=activated,
                first_time_ms=first_time_ms,
                first_position_um=first_position_um,
                first_index=first_index,
                peak_mV=float(flat[local_row, peak_flat[local_row]]),
                peak_time_ms=float(window_time_ms[peak_time[local_row]]),
                peak_index=int(original_indices[peak_col[local_row]]),
            )


def _activation_cohort_groups(
    cohort: Any,
    *,
    width: int,
    target: PositionSelector,
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row, (axon, record_indices) in enumerate(
        zip(cohort.axons, cohort.record_indices, strict=True)
    ):
        full_positions_um = np.asarray(
            axon.layout.position_values(unit="micrometer"),
            dtype=float,
        )
        if record_indices is None:
            if full_positions_um.shape != (int(width),):
                raise ValueError("recorded positions must match Vm columns.")
            original_indices = np.arange(int(width), dtype=int)
            positions_um = full_positions_um
        else:
            original_indices = np.asarray(record_indices, dtype=int)
            if original_indices.shape != (int(width),):
                raise ValueError("record_indices must match Vm columns.")
            if np.any(original_indices < 0) or np.any(original_indices >= full_positions_um.size):
                raise ValueError("record_indices contains values outside axon positions.")
            positions_um = full_positions_um[original_indices]
        selected = target.columns(
            positions_um=positions_um,
            original_indices=original_indices,
        )
        if selected.size == 0:
            raise ValueError("activation target selects no positions.")
        selected = np.asarray(selected, dtype=int)
        key = (
            tuple(int(value) for value in selected),
            tuple(int(value) for value in original_indices[selected]),
            tuple(float(value) for value in positions_um[selected]),
        )
        group = grouped.setdefault(
            key,
            {
                "rows": [],
                "input_indices": [],
                "selected": selected,
                "original_indices": original_indices[selected],
                "positions_um": positions_um[selected],
            },
        )
        group["rows"].append(row)
        group["input_indices"].append(int(cohort.input_indices[row]))
    return tuple(grouped.values())


def _activation_window_view(
    vm: np.ndarray,
    *,
    rows: np.ndarray,
    time_slice: slice,
    selected: np.ndarray,
) -> np.ndarray:
    all_rows = rows.shape == (vm.shape[0],) and np.array_equal(
        rows,
        np.arange(vm.shape[0]),
    )
    all_columns = selected.shape == (vm.shape[2],) and np.array_equal(
        selected,
        np.arange(vm.shape[2]),
    )
    column_index: Any = slice(None) if all_columns else selected
    if all_rows:
        return np.asarray(vm[:, time_slice, column_index])
    return np.asarray(vm[rows][:, time_slice, :][:, :, column_index])


@dataclass(frozen=True)
class Activation:
    """Detect whether a membrane-voltage trace crosses a threshold."""

    threshold: Any = -20.0
    blanking: Any = 0.0
    target: PositionSelector = ALL
    signal: Signal[Any] = MEMBRANE_VOLTAGE
    name: str = "activation"
    algorithm_version: str = "activation_threshold_v1"

    @property
    def requirements(self) -> AnalysisRequirements:
        return AnalysisRequirements(
            required_signals=(self.signal,),
            required_result_fields=("Vm", "t", "positions"),
            required_positions=(self.target,),
            supported_myelination=_ANY_MYELINATION,
            supported_formulations=_ANY_FORMULATION,
            required_capabilities=("membrane_voltage_trace",),
            online_supported=True,
            algorithm_version=self.algorithm_version,
            recording_hint=_VM_RECORDING_HINT,
        )

    def evaluate(self, result: Any) -> AnalysisResult:
        fast = _activation_fast_population(self, result)
        if fast is not None:
            return fast
        return _evaluate_rows(self, result, self._evaluate_one, unit=None)

    def online_observer(
        self,
        *,
        positions: Any,
        original_indices: Any | None = None,
    ) -> Any:
        """Create an online observer for streamed Vm chunks."""

        from axonscope.analysis.observers import ActivationObserver

        return ActivationObserver(
            self,
            positions=positions,
            original_indices=original_indices,
        )

    def _evaluate_one(self, row: Any) -> tuple[bool, AnalysisStatus, str, ActivationEvent]:
        _require_membrane_voltage(row, self.signal)
        event = _activation_event(
            row,
            threshold=self.threshold,
            blanking=self.blanking,
            target=self.target,
        )
        return bool(event.activated), AnalysisStatus.VALID, "", event


@dataclass(frozen=True)
class Latency:
    """Return the first threshold-crossing time for a target position set."""

    threshold: Any = -20.0
    blanking: Any = 0.0
    target: PositionSelector = DISTAL
    signal: Signal[Any] = MEMBRANE_VOLTAGE
    name: str = "latency"
    algorithm_version: str = "activation_latency_v1"

    @property
    def requirements(self) -> AnalysisRequirements:
        return AnalysisRequirements(
            required_signals=(self.signal,),
            required_result_fields=("Vm", "t", "positions"),
            required_positions=(self.target,),
            supported_myelination=_ANY_MYELINATION,
            supported_formulations=_ANY_FORMULATION,
            required_capabilities=("membrane_voltage_trace",),
            algorithm_version=self.algorithm_version,
            recording_hint=_VM_RECORDING_HINT,
        )

    def evaluate(self, result: Any) -> AnalysisResult:
        return _evaluate_rows(self, result, self._evaluate_one, unit="millisecond")

    def _evaluate_one(self, row: Any) -> tuple[float, AnalysisStatus, str, ActivationEvent]:
        _require_membrane_voltage(row, self.signal)
        event = _activation_event(
            row,
            threshold=self.threshold,
            blanking=self.blanking,
            target=self.target,
        )
        if event.first_time_ms is None:
            return (
                np.nan,
                AnalysisStatus.UNDETERMINED,
                "threshold was not crossed at the requested target.",
                event,
            )
        return float(event.first_time_ms), AnalysisStatus.VALID, "", event


@dataclass(frozen=True)
class ConductionBlock:
    """Report whether the requested distal activation target failed to activate."""

    threshold: Any = -20.0
    blanking: Any = 0.0
    target: PositionSelector = DISTAL
    signal: Signal[Any] = MEMBRANE_VOLTAGE
    name: str = "conduction_block"
    algorithm_version: str = "conduction_block_v1"

    @property
    def requirements(self) -> AnalysisRequirements:
        return AnalysisRequirements(
            required_signals=(self.signal,),
            required_result_fields=("Vm", "t", "positions"),
            required_positions=(self.target,),
            supported_myelination=_ANY_MYELINATION,
            supported_formulations=_ANY_FORMULATION,
            required_capabilities=("membrane_voltage_trace",),
            algorithm_version=self.algorithm_version,
            recording_hint=_VM_RECORDING_HINT,
        )

    def evaluate(self, result: Any) -> AnalysisResult:
        return _evaluate_rows(self, result, self._evaluate_one, unit=None)

    def _evaluate_one(self, row: Any) -> tuple[bool, AnalysisStatus, str, ActivationEvent]:
        _require_membrane_voltage(row, self.signal)
        event = _activation_event(
            row,
            threshold=self.threshold,
            blanking=self.blanking,
            target=self.target,
        )
        return not bool(event.activated), AnalysisStatus.VALID, "", event


@dataclass(frozen=True)
class PeakVoltage:
    """Return the maximum membrane voltage over selected recorded positions."""

    target: PositionSelector = ALL
    signal: Signal[Any] = MEMBRANE_VOLTAGE
    name: str = "peak_voltage"
    algorithm_version: str = "peak_voltage_v1"

    @property
    def requirements(self) -> AnalysisRequirements:
        return AnalysisRequirements(
            required_signals=(self.signal,),
            required_result_fields=("Vm", "positions"),
            required_positions=(self.target,),
            supported_myelination=_ANY_MYELINATION,
            supported_formulations=_ANY_FORMULATION,
            required_capabilities=("membrane_voltage_trace",),
            online_supported=True,
            algorithm_version=self.algorithm_version,
            recording_hint=_VM_RECORDING_HINT,
        )

    def evaluate(self, result: Any) -> AnalysisResult:
        return _evaluate_rows(self, result, self._evaluate_one, unit="millivolt")

    def online_observer(
        self,
        *,
        positions: Any,
        original_indices: Any | None = None,
    ) -> Any:
        """Raise because peak voltage is a post-hoc analysis."""

        raise NotImplementedError(
            "PeakVoltage is post-hoc on recorded Vm. Use result.analyze("
            "axs.analysis.PeakVoltage(...)) or axs.analysis.peak_voltage(result)."
        )

    def _evaluate_one(self, row: Any) -> tuple[float, AnalysisStatus, str, None]:
        vm = _require_membrane_voltage(row, self.signal)
        columns, _, _ = _selected_columns(row, target=self.target, vm=vm)
        if columns.size == 0:
            raise MissingAnalysisInputError(
                "no recorded columns are available for the requested target.",
                required_signals=(self.signal,),
                required_result_fields=("Vm", "positions", "record_indices"),
                required_positions=(self.target,),
                recording_hint=_VM_RECORDING_HINT,
            )
        return float(np.max(vm[:, columns])), AnalysisStatus.VALID, "", None


@dataclass(frozen=True)
class SpikeCount:
    """Count detected action-potential peaks in a voltage recording."""

    threshold: Any = -20.0
    min_distance: Any = 0.5
    peak_height: Any | None = None
    min_width: Any | None = 0.1
    signal: Signal[Any] = MEMBRANE_VOLTAGE
    name: str = "spike_count"
    algorithm_version: str = "spike_count_v1"

    @property
    def requirements(self) -> AnalysisRequirements:
        return AnalysisRequirements(
            required_signals=(self.signal,),
            required_result_fields=("Vm", "t", "positions"),
            required_positions=(ALL,),
            supported_myelination=_ANY_MYELINATION,
            supported_formulations=_ANY_FORMULATION,
            required_capabilities=("membrane_voltage_trace",),
            algorithm_version=self.algorithm_version,
            recording_hint=_VM_RECORDING_HINT,
        )

    def evaluate(self, result: Any) -> AnalysisResult:
        return _evaluate_rows(self, result, self._evaluate_one, unit=None)

    def _evaluate_one(self, row: Any) -> tuple[int, AnalysisStatus, str, None]:
        _require_membrane_voltage(row, self.signal)
        spike_times_ms, _ = rasterize(
            row,
            threshold_mV=self.threshold,
            min_distance_ms=self.min_distance,
            peak_height_mV=self.peak_height,
            min_width_ms=self.min_width,
        )
        return int(spike_times_ms.shape[0]), AnalysisStatus.VALID, "", None


@dataclass(frozen=True)
class ConductionVelocity:
    """Estimate propagation velocity from detected action-potential peaks."""

    threshold: Any | None = None
    min_distance: Any = 0.5
    peak_height: Any | None = (-20.0, 70.0)
    min_width: Any | None = 0.1
    spatial_filter: str = "nodes_if_available"
    target: PositionSelector = ALL
    signal: Signal[Any] = MEMBRANE_VOLTAGE
    name: str = "conduction_velocity"
    algorithm_version: str = "conduction_velocity_v1"

    @property
    def requirements(self) -> AnalysisRequirements:
        roles = ("node",) if self.spatial_filter in {"nodes", "nodes_if_available"} else ()
        return AnalysisRequirements(
            required_signals=(self.signal,),
            required_result_fields=("Vm", "t", "positions"),
            required_positions=(ALL,),
            required_compartment_roles=roles,
            supported_myelination=_ANY_MYELINATION,
            supported_formulations=_ANY_FORMULATION,
            required_capabilities=("membrane_voltage_trace", "spike_timing"),
            algorithm_version=self.algorithm_version,
            recording_hint=_VM_RECORDING_HINT,
        )

    def evaluate(self, result: Any) -> AnalysisResult:
        return _evaluate_rows(self, result, self._evaluate_one, unit="meter / second")

    def _evaluate_one(self, row: Any) -> tuple[float, AnalysisStatus, str, None]:
        try:
            _require_membrane_voltage(row, self.signal)
        except MissingAnalysisInputError:
            value = self._evaluate_one_from_vm_raster(row)
        except ValueError as exc:
            if not _is_missing_input_error(exc):
                raise
            value = self._evaluate_one_from_vm_raster(row)
        else:
            value = float(
                conduction_velocity(
                    row,
                    threshold_mV=self.threshold,
                    min_distance_ms=self.min_distance,
                    peak_height_mV=self.peak_height,
                    min_width_ms=self.min_width,
                    spatial_filter=self.spatial_filter,
                )
            )
        if value <= 0.0:
            return (
                np.nan,
                AnalysisStatus.UNDETERMINED,
                "fewer than two propagated spikes were detected.",
                None,
            )
        return value, AnalysisStatus.VALID, "", None

    def _evaluate_one_from_vm_raster(self, row: Any) -> float:
        from axonscope.results.vm_raster import (
            VM_RASTER_OBSERVATION_KEY,
            conduction_velocity_values_from_vm_raster,
        )

        observations = getattr(row, "observations", None)
        if observations is None or VM_RASTER_OBSERVATION_KEY not in observations:
            raise _missing(
                "analysis requires a membrane-voltage recording or VmRaster observation.",
                signal=self.signal,
            )
        values = conduction_velocity_values_from_vm_raster(
            observations[VM_RASTER_OBSERVATION_KEY],
            self,
        )
        if np.asarray(values).shape != (1,):
            raise ValueError("row VmRaster observation must contain exactly one batch row.")
        return float(np.asarray(values, dtype=float)[0])


__all__ = [
    "Activation",
    "ConductionBlock",
    "ConductionVelocity",
    "Latency",
    "PeakVoltage",
    "SpikeCount",
]
