"""Single-axon simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from axonscope.axons.axon import Axon
from axonscope.utils import units

if TYPE_CHECKING:
    from axonscope.axon_instance import AxonInstance
    from axonscope.recording import Recording


ResultArray: TypeAlias = Any
RecordingValue: TypeAlias = ResultArray | dict[str, ResultArray]
RecordingDict: TypeAlias = dict[str, RecordingValue]
ObservationDict: TypeAlias = dict[str, Any]


_UNIT_DISPLAY = {
    "micrometer": "um",
    "millimeter": "mm",
    "millisecond": "ms",
    "millivolt": "mV",
    "second": "s",
    "volt": "V",
}


def _unit_display(unit: Any) -> str:
    label = units.unit_label(unit)
    if label is None:
        return ""
    return _UNIT_DISPLAY.get(label, label)


def _normalize_recordings(
    recordings: RecordingDict | None,
    Vm: ResultArray | None,
) -> RecordingDict | None:
    normalized: RecordingDict = {}
    if recordings is not None:
        normalized.update(recordings)
    if Vm is not None:
        normalized["Vm"] = Vm
    return normalized or None


@dataclass(init=False)
class SimResult:
    """Recorded traces and optional observations returned by simulations.

    Recorded traces live in ``recordings``. ``recordings["Vm"]`` is the membrane
    voltage matrix indexed as ``(time, compartment)``. The ``Vm`` property is a
    convenience alias for existing notebook and test code. Compact solver-side
    reductions will live in ``observations``.
    """

    axon: Axon
    t: ResultArray
    diagnostics: dict[str, Any] | None = None
    recordings: RecordingDict | None = None
    observations: ObservationDict | None = None
    recording: Recording | None = None
    record_indices: tuple[int, ...] | None = None
    simulation: AxonInstance | None = None

    def __init__(
        self,
        axon: Axon,
        Vm: ResultArray | None = None,
        t: ResultArray | None = None,
        *,
        diagnostics: dict[str, Any] | None = None,
        recordings: RecordingDict | None = None,
        observations: ObservationDict | None = None,
        recording: Recording | None = None,
        record_indices: tuple[int, ...] | None = None,
        simulation: AxonInstance | None = None,
    ) -> None:
        if t is None:
            raise TypeError("SimResult requires a time vector `t`.")
        self.axon = axon
        self.t = t
        self.diagnostics = diagnostics
        self.recordings = _normalize_recordings(recordings, Vm)
        self.observations = observations
        self.recording = recording
        self.record_indices = record_indices
        self.simulation = simulation

    @property
    def Vm(self) -> ResultArray:
        """Membrane voltage recording, equivalent to ``recordings["Vm"]``."""

        if self.recordings is None or "Vm" not in self.recordings:
            raise AttributeError("this SimResult does not contain a Vm recording.")
        vm = self.recordings["Vm"]
        if isinstance(vm, dict):
            raise TypeError("recordings['Vm'] must be an array, not a group.")
        return vm

    @Vm.setter
    def Vm(self, value: ResultArray) -> None:
        """Update the membrane-voltage recording in place."""

        recordings = {} if self.recordings is None else dict(self.recordings)
        recordings["Vm"] = value
        self.recordings = recordings

    def time_values(self, *, unit: Any = "millisecond") -> np.ndarray:
        """Return result times as plain values in `unit`."""

        unit_label = units.unit_label(unit) or "millisecond"
        return units.to_array(
            units.Q_(np.asarray(self.t, dtype=float), "millisecond"),
            unit_label,
            dtype=float,
        )

    def position_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return recorded axon positions as plain values in `unit`."""

        from axonscope.analysis.posthoc import recorded_positions_um

        unit_label = units.unit_label(unit) or "micrometer"
        positions_um = recorded_positions_um(self)
        return units.to_array(units.Q_(positions_um, "micrometer"), unit_label, dtype=float)

    def voltage_values(self, *, unit: Any = "millivolt") -> np.ndarray:
        """Return membrane voltages as plain values in `unit`."""

        unit_label = units.unit_label(unit) or "millivolt"
        vm = np.asarray(self.Vm, dtype=float)
        return units.to_array(units.Q_(vm, "millivolt"), unit_label, dtype=float)

    def peak_voltage_values(self, *, unit: Any = "millivolt") -> np.ndarray:
        """Return the peak membrane voltage for each recorded column."""

        vm = self.voltage_values(unit=unit)
        if vm.ndim != 2:
            raise ValueError(
                f"result.Vm must be 2D (time, position), got shape {vm.shape}."
            )
        return np.max(vm, axis=0)

    def analyze(self, *definitions: Any) -> Any:
        """Evaluate public analysis definitions on this result."""

        from axonscope.analysis import analyze

        return analyze(self, *definitions)

    def report(self, *definitions: Any) -> Any:
        """Return an analysis report for one or more definitions."""

        from axonscope.analysis import analyze

        analyzed = analyze(self, *definitions)
        if hasattr(analyzed, "analyses"):
            return analyzed
        from axonscope.analysis import AnalysisReport

        return AnalysisReport(
            simulation_result=self,
            analyses=(analyzed,),
        )

    def nearest_position_index(self, position: Any) -> int:
        """Return the recorded column nearest to `position`."""

        positions_um = self.position_values(unit="micrometer")
        target_um = units.to_um(position)
        return int(np.argmin(np.abs(positions_um - target_um)))

    def trace_values(
        self,
        *,
        position: Any | None = None,
        index: int | None = None,
        time_unit: Any = "millisecond",
        voltage_unit: Any = "millivolt",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one voltage trace as ``(time, Vm)`` arrays."""

        vm = self.voltage_values(unit=voltage_unit)
        if vm.ndim != 2:
            raise ValueError(
                f"result.Vm must be 2D (time, position), got shape {vm.shape}."
            )
        if index is not None and position is not None:
            raise ValueError("Provide either `index` or `position`, not both.")
        if index is None:
            index = (
                self.nearest_position_index(position)
                if position is not None
                else vm.shape[1] // 2
            )
        if index < 0 or index >= vm.shape[1]:
            raise IndexError(
                f"index {index} is outside Vm columns 0..{vm.shape[1] - 1}."
            )

        t = self.time_values(unit=time_unit)
        if t.shape[0] != vm.shape[0]:
            raise ValueError("result.t length must match result.Vm time dimension.")
        return t, vm[:, index]

    def plot_trace(
        self,
        ax: Any | None = None,
        *,
        position: Any | None = None,
        index: int | None = None,
        time_unit: Any = "millisecond",
        voltage_unit: Any = "millivolt",
        label: str | None = None,
        title: str | None = None,
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot one membrane-voltage trace."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        if index is None:
            vm = self.voltage_values(unit=voltage_unit)
            if vm.ndim != 2:
                raise ValueError(
                    f"result.Vm must be 2D (time, position), got shape {vm.shape}."
                )
            resolved_index = (
                self.nearest_position_index(position)
                if position is not None
                else vm.shape[1] // 2
            )
        else:
            resolved_index = index
        t, trace = self.trace_values(
            position=position,
            index=index,
            time_unit=time_unit,
            voltage_unit=voltage_unit,
        )
        positions = self.position_values(unit="micrometer")

        plot_kwargs.setdefault("linewidth", 2.0)
        if label is not None:
            plot_kwargs.setdefault("label", label)
        ax.plot(t, trace, **plot_kwargs)
        ax.set_xlabel(f"Time [{_unit_display(time_unit)}]")
        ax.set_ylabel(f"Vm [{_unit_display(voltage_unit)}]")
        ax.set_title(title or f"Vm at x={positions[resolved_index]:g} um")
        if label is not None:
            ax.legend()
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    def plot_map(
        self,
        ax: Any | None = None,
        *,
        time_unit: Any = "millisecond",
        position_unit: Any = "micrometer",
        voltage_unit: Any = "millivolt",
        cmap: str = "viridis",
        colorbar: bool = True,
        title: str = "Vm propagation",
        **imshow_kwargs: Any,
    ) -> Any:
        """Plot the membrane-voltage space-time map."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        vm = self.voltage_values(unit=voltage_unit)
        if vm.ndim != 2:
            raise ValueError(
                f"result.Vm must be 2D (time, position), got shape {vm.shape}."
            )
        t = self.time_values(unit=time_unit)
        x = self.position_values(unit=position_unit)
        if t.shape[0] != vm.shape[0]:
            raise ValueError("result.t length must match result.Vm time dimension.")
        if x.shape[0] != vm.shape[1]:
            raise ValueError("recorded positions must match result.Vm position dimension.")

        imshow_kwargs.setdefault("aspect", "auto")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("cmap", cmap)
        image = ax.imshow(
            vm.T,
            extent=[t[0], t[-1], x[0], x[-1]],
            **imshow_kwargs,
        )
        ax.set_xlabel(f"Time [{_unit_display(time_unit)}]")
        ax.set_ylabel(f"Axon position x [{_unit_display(position_unit)}]")
        ax.set_title(f"{title} [{_unit_display(voltage_unit)}]")
        if colorbar:
            ax.figure.colorbar(
                image,
                ax=ax,
                label=f"Vm [{_unit_display(voltage_unit)}]",
            )
        return ax

    plot_voltage_trace = plot_trace
    plot_voltage_map = plot_map


__all__ = ["RecordingDict", "ResultArray", "SimResult"]
