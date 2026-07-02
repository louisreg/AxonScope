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
        _require_membrane_voltage(row, self.signal)
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


__all__ = [
    "Activation",
    "ConductionBlock",
    "ConductionVelocity",
    "Latency",
    "PeakVoltage",
    "SpikeCount",
]
