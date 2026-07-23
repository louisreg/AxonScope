"""Typed extracellular footprint and drive contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from axonfleet.identifiers import AxonId, DriveId
from axonfleet.stimulation.stimuli import Stimulus
from axonfleet.utils import units


def _freeze_array(value: np.ndarray) -> np.ndarray:
    """Return a read-only NumPy view of `value`."""

    arr = np.asarray(value)
    arr.setflags(write=False)
    return arr


def _unit_pair(voltage_unit: Any, current_unit: Any) -> str:
    voltage = units.unit_label(voltage_unit) or "volt"
    current = units.unit_label(current_unit) or "ampere"
    return f"{voltage} / {current}"


def _unit_text(unit: Any) -> str:
    label = units.unit_label(unit) or str(unit)
    return units.short_unit_label(label) or label


@dataclass(frozen=True, kw_only=True)
class ExtracellularFootprint:
    """Static extracellular transfer profile.

    A footprint stores spatial voltage-per-current samples. It has no time
    axis and no stimulus amplitude; temporal drive happens in
    `ExtracellularDrive`.
    """

    values: Any
    positions: Any
    axon_ids: Sequence[AxonId] | None = None
    voltage_unit: Any = "volt"
    current_unit: Any = "ampere"
    interpolation: str = "sampled"
    source_id: str | None = None
    reference: str = "intrinsic axon coordinates"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions_um = units.require_length_array_um(
            self.positions,
            name="positions",
            dtype=float,
        )
        positions_um = np.atleast_1d(positions_um)
        if positions_um.ndim != 1:
            raise ValueError("ExtracellularFootprint positions must be a 1D length array.")
        if positions_um.size == 0:
            raise ValueError("ExtracellularFootprint requires at least one position.")

        unit_label = _unit_pair(self.voltage_unit, self.current_unit)
        values_V_per_A = units.to_array(self.values, unit_label, dtype=float)
        values_V_per_A = np.asarray(
            units.to_array(units.Q_(values_V_per_A, unit_label), "volt / ampere"),
            dtype=float,
        )

        ids = None if self.axon_ids is None else tuple(self.axon_ids)
        if ids is None:
            values_V_per_A = np.atleast_1d(values_V_per_A)
            if values_V_per_A.ndim != 1:
                raise ValueError("Shared ExtracellularFootprint values must be 1D.")
            if values_V_per_A.shape[0] != positions_um.shape[0]:
                raise ValueError("Shared footprint values must match the position count.")
        else:
            if not ids:
                raise ValueError("axon_ids must be non-empty when provided.")
            if any(not isinstance(value, AxonId) for value in ids):
                raise TypeError("axon_ids must contain AxonId values.")
            if len(set(ids)) != len(ids):
                raise ValueError("axon_ids must be unique.")
            values_V_per_A = np.asarray(values_V_per_A, dtype=float)
            if values_V_per_A.ndim == 1 and len(ids) == 1:
                values_V_per_A = values_V_per_A[None, :]
            if values_V_per_A.ndim != 2:
                raise ValueError("Per-axon ExtracellularFootprint values must be 2D.")
            expected_shape = (len(ids), positions_um.shape[0])
            if values_V_per_A.shape != expected_shape:
                raise ValueError(
                    "Per-axon footprint values must have shape "
                    f"{expected_shape}, got {values_V_per_A.shape}."
                )

        interpolation = str(self.interpolation).strip()
        if not interpolation:
            raise ValueError("interpolation must be a non-empty string.")

        object.__setattr__(self, "positions_um", _freeze_array(positions_um))
        object.__setattr__(self, "values_V_per_A", _freeze_array(values_V_per_A))
        object.__setattr__(self, "axon_ids", ids)
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def shared(
        cls,
        *,
        values: Any,
        positions: Any,
        voltage_unit: Any = "volt",
        current_unit: Any = "ampere",
        interpolation: str = "sampled",
        source_id: str | None = None,
        reference: str = "intrinsic axon coordinates",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExtracellularFootprint":
        """Build a footprint shared by every compatible axon row."""

        return cls(
            values=values,
            positions=positions,
            axon_ids=None,
            voltage_unit=voltage_unit,
            current_unit=current_unit,
            interpolation=interpolation,
            source_id=source_id,
            reference=reference,
            metadata={} if metadata is None else metadata,
        )

    def position_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return intrinsic positions in `unit`."""

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(units.Q_(self.positions_um, "micrometer"), unit_label, dtype=float)

    def value_values(
        self,
        *,
        voltage_unit: Any = "volt",
        current_unit: Any = "ampere",
        axon_id: AxonId | None = None,
    ) -> np.ndarray:
        """Return footprint values in `voltage_unit / current_unit`."""

        values = self.values_for_axon(axon_id)
        output_unit = _unit_pair(voltage_unit, current_unit)
        return units.to_array(units.Q_(values, "volt / ampere"), output_unit, dtype=float)

    def values_for_axon(self, axon_id: AxonId | None = None) -> np.ndarray:
        """Return canonical V/A samples for one axon or a shared footprint."""

        if self.axon_ids is None:
            return np.asarray(self.values_V_per_A, dtype=float)
        if axon_id is None:
            if len(self.axon_ids) == 1:
                return np.asarray(self.values_V_per_A[0], dtype=float)
            raise ValueError("axon_id is required for a multi-axon footprint.")
        if not isinstance(axon_id, AxonId):
            raise TypeError("axon_id must be an AxonId.")
        try:
            index = self.axon_ids.index(axon_id)
        except ValueError as exc:
            raise KeyError(f"Unknown axon_id: {axon_id!r}.") from exc
        return np.asarray(self.values_V_per_A[index], dtype=float)

    def _has_same_positions(self, other: "ExtracellularFootprint") -> bool:
        """Return whether two footprints share the same spatial support."""

        if not isinstance(other, ExtracellularFootprint):
            return False
        return (
            self.positions_um.shape == other.positions_um.shape
            and np.allclose(self.positions_um, other.positions_um)
        )

    def plot(
        self,
        ax: Any | None = None,
        *,
        position_unit: Any = "micrometer",
        voltage_unit: Any = "millivolt",
        current_unit: Any = "microampere",
        axon_id: AxonId | None = None,
        label: str | None = None,
        title: str = "Extracellular footprint",
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot this static voltage-per-current footprint."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        x = self.position_values(unit=position_unit)
        y = self.value_values(
            voltage_unit=voltage_unit,
            current_unit=current_unit,
            axon_id=axon_id,
        )
        if label is None:
            label = self.source_id
        if label is not None:
            plot_kwargs.setdefault("label", label)
        plot_kwargs.setdefault("linewidth", 2.0)
        ax.plot(x, y, **plot_kwargs)
        ax.set_title(title)
        ax.set_xlabel(f"Position [{_unit_text(position_unit)}]")
        ax.set_ylabel(f"Footprint [{_unit_text(voltage_unit)}/{_unit_text(current_unit)}]")
        if grid:
            ax.grid(True, alpha=0.3)
        if label is not None:
            ax.legend(frameon=False)
        return ax


@dataclass(frozen=True, kw_only=True)
class ExtracellularDrive:
    """One extracellular contribution: static footprint times stimulus."""

    id: DriveId
    footprint: ExtracellularFootprint
    stimulus: Stimulus
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, DriveId):
            raise TypeError("id must be a DriveId.")
        if not isinstance(self.footprint, ExtracellularFootprint):
            raise TypeError("footprint must be an ExtracellularFootprint.")
        if not isinstance(self.stimulus, Stimulus):
            raise TypeError("stimulus must be an axonfleet.stimulation.Stimulus.")

        object.__setattr__(self, "stimulus", self.stimulus.as_unit("ampere"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def _evaluate(
        self,
        t: Any,
        *,
        axon_id: AxonId | None = None,
        voltage_unit: Any = "volt",
    ) -> np.ndarray:
        """Materialize this drive's Vext contribution for teaching/inspection."""

        current_A = np.atleast_1d(np.asarray(self.stimulus.evaluate(t, unit="ampere"), dtype=float))
        footprint = self.footprint.values_for_axon(axon_id)
        values_V = current_A[:, None] * footprint[None, :]
        unit_label = units.unit_label(voltage_unit) or "volt"
        return units.to_array(units.Q_(values_V, "volt"), unit_label, dtype=float)

@dataclass(frozen=True)
class ExtracellularStimulation:
    """Immutable collection of extracellular drives."""

    drives: Sequence[ExtracellularDrive]

    def __post_init__(self) -> None:
        drives = tuple(self.drives)
        if not drives:
            raise ValueError("ExtracellularStimulation requires at least one drive.")
        for drive in drives:
            if not isinstance(drive, ExtracellularDrive):
                raise TypeError("drives must contain ExtracellularDrive objects.")
        ids = tuple(drive.id for drive in drives)
        if len(set(ids)) != len(ids):
            raise ValueError("ExtracellularDrive ids must be unique.")

        reference = drives[0].footprint
        incompatible = [
            drive.id
            for drive in drives[1:]
            if not reference._has_same_positions(drive.footprint)
        ]
        if incompatible:
            joined = ", ".join(str(value) for value in incompatible)
            raise ValueError(f"Drive footprints use incompatible position supports: {joined}.")

        object.__setattr__(self, "drives", drives)

    @property
    def names(self) -> tuple[DriveId, ...]:
        """Drive ids in collection order."""

        return tuple(drive.id for drive in self.drives)

    @property
    def positions_um(self) -> np.ndarray:
        """Shared intrinsic position support in micrometers."""

        return self.drives[0].footprint.positions_um

    def __len__(self) -> int:
        return len(self.drives)

    def __iter__(self):
        return iter(self.drives)

    def __getitem__(self, drive_id: DriveId) -> ExtracellularDrive:
        if not isinstance(drive_id, DriveId):
            raise TypeError("drive_id must be a DriveId.")
        for drive in self.drives:
            if drive.id == drive_id:
                return drive
        raise KeyError(f"Unknown extracellular drive: {drive_id!r}.")

    def replace_drive(
        self,
        drive_id: DriveId,
        *,
        footprint: ExtracellularFootprint | None = None,
        stimulus: Stimulus | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExtracellularStimulation":
        """Return a new collection with one drive updated."""

        if not isinstance(drive_id, DriveId):
            raise TypeError("drive_id must be a DriveId.")
        updated = []
        found = False
        for drive in self.drives:
            if drive.id != drive_id:
                updated.append(drive)
                continue
            found = True
            updated.append(
                replace(
                    drive,
                    footprint=drive.footprint if footprint is None else footprint,
                    stimulus=drive.stimulus if stimulus is None else stimulus,
                    metadata=drive.metadata if metadata is None else metadata,
                )
            )
        if not found:
            raise KeyError(f"Unknown extracellular drive: {drive_id!r}.")
        return ExtracellularStimulation(tuple(updated))

    def evaluate(
        self,
        t: Any,
        *,
        axon_id: AxonId | None = None,
        voltage_unit: Any = "volt",
    ) -> np.ndarray:
        """Materialize the summed Vext field for teaching/inspection."""

        values_V = None
        for drive in self.drives:
            contribution = drive._evaluate(t, axon_id=axon_id, voltage_unit="volt")
            values_V = contribution if values_V is None else values_V + contribution
        if values_V is None:
            raise ValueError("ExtracellularStimulation requires at least one drive.")
        unit_label = units.unit_label(voltage_unit) or "volt"
        return units.to_array(units.Q_(values_V, "volt"), unit_label, dtype=float)

    def potential(
        self,
        t: Any,
        *,
        axon_id: AxonId | None = None,
        voltage_unit: Any = "volt",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExtracellularPotential":
        """Explicitly materialize a dense potential object."""

        return ExtracellularPotential(
            values=self.evaluate(t, axon_id=axon_id, voltage_unit=voltage_unit),
            t=t,
            positions=units.Q_(self.positions_um, "micrometer"),
            voltage_unit=voltage_unit,
            axon_ids=None if axon_id is None else (axon_id,),
            metadata={} if metadata is None else metadata,
        )

    def plot_footprints(
        self,
        ax: Any | None = None,
        *,
        position_unit: Any = "micrometer",
        voltage_unit: Any = "millivolt",
        current_unit: Any = "microampere",
        axon_id: AxonId | None = None,
        title: str = "Extracellular footprints",
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot all static footprints in this stimulation."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        for drive in self.drives:
            drive.footprint.plot(
                ax=ax,
                position_unit=position_unit,
                voltage_unit=voltage_unit,
                current_unit=current_unit,
                axon_id=axon_id,
                label=str(drive.id),
                title=title,
                grid=grid,
                **plot_kwargs,
            )
        ax.set_title(title)
        return ax

    def plot_potential(
        self,
        t: Any,
        ax: Any | None = None,
        *,
        axon_id: AxonId | None = None,
        time_unit: Any = "millisecond",
        position_unit: Any = "micrometer",
        voltage_unit: Any = "millivolt",
        title: str = "Extracellular potential",
        colorbar: bool = True,
        **imshow_kwargs: Any,
    ) -> Any:
        """Materialize and plot the summed extracellular potential."""

        return self.potential(t, axon_id=axon_id, voltage_unit=voltage_unit).plot(
            ax=ax,
            time_unit=time_unit,
            position_unit=position_unit,
            voltage_unit=voltage_unit,
            title=title,
            colorbar=colorbar,
            **imshow_kwargs,
        )


@dataclass(frozen=True, kw_only=True)
class ExtracellularPotential:
    """Explicit dense Vext materialization for inspection and examples."""

    values: Any
    t: Any
    positions: Any
    axon_ids: Sequence[AxonId] | None = None
    voltage_unit: Any = "volt"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t_ms = units.require_time_array_ms(self.t, name="t", dtype=float)
        t_ms = np.atleast_1d(t_ms)
        positions_um = units.require_length_array_um(self.positions, name="positions", dtype=float)
        positions_um = np.atleast_1d(positions_um)
        unit_label = units.unit_label(self.voltage_unit) or "volt"
        values_V = units.to_array(self.values, unit_label, dtype=float)
        values_V = units.to_array(units.Q_(values_V, unit_label), "volt", dtype=float)

        ids = None if self.axon_ids is None else tuple(self.axon_ids)
        if ids is not None and any(not isinstance(value, AxonId) for value in ids):
            raise TypeError("axon_ids must contain AxonId values.")
        if ids is None:
            expected = (t_ms.shape[0], positions_um.shape[0])
            if values_V.shape != expected:
                raise ValueError(f"ExtracellularPotential values must have shape {expected}.")
        else:
            if values_V.ndim == 2 and len(ids) == 1:
                values_V = values_V[None, :, :]
            expected = (len(ids), t_ms.shape[0], positions_um.shape[0])
            if values_V.shape != expected:
                raise ValueError(f"ExtracellularPotential values must have shape {expected}.")

        object.__setattr__(self, "t_ms", _freeze_array(t_ms))
        object.__setattr__(self, "positions_um", _freeze_array(positions_um))
        object.__setattr__(self, "values_V", _freeze_array(values_V))
        object.__setattr__(self, "axon_ids", ids)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def time_values(self, *, unit: Any = "millisecond") -> np.ndarray:
        """Return time samples in `unit`."""

        unit_label = units.unit_label(unit) or "millisecond"
        return units.to_array(units.Q_(self.t_ms, "millisecond"), unit_label, dtype=float)

    def position_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return spatial samples in `unit`."""

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(units.Q_(self.positions_um, "micrometer"), unit_label, dtype=float)

    def value_values(self, *, voltage_unit: Any = "volt") -> np.ndarray:
        """Return dense Vext values in `voltage_unit`."""

        unit_label = units.unit_label(voltage_unit) or "volt"
        return units.to_array(units.Q_(self.values_V, "volt"), unit_label, dtype=float)

    def plot(
        self,
        ax: Any | None = None,
        *,
        axon_index: int = 0,
        time_unit: Any = "millisecond",
        position_unit: Any = "micrometer",
        voltage_unit: Any = "millivolt",
        title: str = "Extracellular potential",
        colorbar: bool = True,
        **imshow_kwargs: Any,
    ) -> Any:
        """Plot this dense Vext materialization as a time-space map."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        values = self.value_values(voltage_unit=voltage_unit)
        if values.ndim == 3:
            axon_index = int(axon_index)
            if axon_index < 0 or axon_index >= values.shape[0]:
                raise IndexError(
                    f"axon_index {axon_index} is outside potential rows 0..{values.shape[0] - 1}."
                )
            values = values[axon_index]
        if values.ndim != 2:
            raise ValueError("ExtracellularPotential values must be 2D or 3D.")

        t = self.time_values(unit=time_unit)
        x = self.position_values(unit=position_unit)
        imshow_kwargs.setdefault("aspect", "auto")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("cmap", "coolwarm")
        image = ax.imshow(
            values.T,
            extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
            **imshow_kwargs,
        )
        ax.set_title(title)
        ax.set_xlabel(f"Time [{_unit_text(time_unit)}]")
        ax.set_ylabel(f"Position [{_unit_text(position_unit)}]")
        if colorbar:
            ax.figure.colorbar(image, ax=ax, label=f"Vext [{_unit_text(voltage_unit)}]")
        return ax


__all__ = [
    "ExtracellularDrive",
    "ExtracellularFootprint",
    "ExtracellularPotential",
    "ExtracellularStimulation",
]
