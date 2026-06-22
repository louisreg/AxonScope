"""Physical stimulation contexts attached to simulation protocols.

Contexts are lightweight, backend-independent descriptions. Intracellular
contexts bind a current waveform to one axonal position. Extracellular contexts
group stimulated electrodes or precomputed spatial transfer data; specific
subclasses decide how electrode currents become extracellular potentials.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from axonscope.identifiers import AxonId
from axonscope.stimulation.stimuli import ArrayLike, Stimulus
from axonscope.utils import units

if TYPE_CHECKING:
    from axonscope.stimulation.electrodes import AnalyticalElectrode, Electrode


_UNIT_DISPLAY = {
    "ampere": "A",
    "microampere": "uA",
    "millivolt": "mV",
    "micrometer": "um",
    "nanoampere": "nA",
    "second": "s",
    "siemens / meter": "S/m",
    "volt": "V",
    "millisecond": "ms",
}


def _unit_display(unit: Any) -> str:
    """Return a short display label for a Pint-like unit."""

    label = units.unit_label(unit)
    if label is None:
        return ""
    return _UNIT_DISPLAY.get(label, label)


@dataclass(frozen=True, kw_only=True)
class IntracellularContext(ABC):
    """Base class for intracellular stimulation descriptions.

    Intracellular contexts describe current injection protocols attached to an
    axon. Runtime compilers lower concrete subclasses to solver arrays and
    functions; solver kernels do not inspect these public objects directly.
    """


@dataclass(frozen=True, kw_only=True, init=False)
class IntracellularCurrentClamp(IntracellularContext):
    """Descriptive intracellular current clamp attached to an axon position.

    `position` must carry length units and is stored internally as
    `position_um`. Plain waveform amplitudes are interpreted as nanoamperes.
    Pint quantities are converted to nanoamperes at construction time.
    """

    position_um: Any
    """Axial clamp position in micrometers after normalization."""

    current: Stimulus
    """Current waveform normalized to nanoamperes."""

    def __init__(self, *, position: Any, current: Stimulus) -> None:
        """Create an intracellular current clamp at axial `position`."""

        object.__setattr__(self, "position_um", position)
        object.__setattr__(self, "current", current)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate the current waveform and normalize public units."""

        if not isinstance(self.current, Stimulus):
            raise TypeError("current must be an axonscope.stimulation.Stimulus.")
        object.__setattr__(
            self,
            "position_um",
            units.require_length_um(self.position_um, name="position"),
        )
        object.__setattr__(self, "current", self.current.as_unit("nanoampere"))


@dataclass(frozen=True, kw_only=True)
class ExtracellularContext:
    """Base extracellular stimulation environment.

    The base class validates stimulated electrodes and defines the interface
    runtime code uses to request V/A footprints. Concrete subclasses implement
    `footprint_for_electrode`, which is the solver-facing contract: the solver
    does not need to know whether the footprint came from an analytical model,
    FEM, or another backend.
    """

    electrodes: Sequence["Electrode"]
    """Stimulated electrodes in the context's coordinate frame."""

    def __post_init__(self) -> None:
        """Validate and freeze the electrode collection as a tuple."""

        from axonscope.stimulation.electrodes import Electrode

        electrodes = tuple(self.electrodes)
        if not electrodes:
            raise ValueError("ExtracellularContext requires at least one electrode.")
        for electrode in electrodes:
            if not isinstance(electrode, Electrode):
                raise TypeError(
                    "electrodes must contain axonscope.stimulation.Electrode instances."
                )
            if getattr(electrode, "stimulus", None) is None:
                raise ValueError("Each extracellular electrode must have an attached stimulus.")
        object.__setattr__(self, "electrodes", electrodes)

    def with_electrodes(self, electrodes: Sequence["Electrode"]) -> "ExtracellularContext":
        """Return a copy-like context containing `electrodes`.

        Subclasses that carry medium-specific state should override this method
        to preserve that state.
        """

        return type(self)(electrodes=electrodes)

    def footprint_for_electrode(
        self,
        electrode: "Electrode",
        x_positions_m: ArrayLike,
        *,
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Return V/A footprint samples for one electrode and one axon."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement extracellular footprints."
        )

    def position_values(
        self,
        x_positions: ArrayLike,
        *,
        unit: Any = "micrometer",
    ) -> np.ndarray:
        """Return position coordinates as plain numeric values in `unit`."""

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(x_positions, unit_label, dtype=float)

    def evaluate(
        self,
        x_positions: ArrayLike,
        t: ArrayLike,
        *,
        voltage_unit: Any | None = None,
        position_unit: Any = "micrometer",
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Evaluate summed Vext for all context electrodes.

        Values are returned with shape `(n_times, n_positions)` in volts unless
        `voltage_unit` is provided. This method is generic and relies only on
        the `footprint_for_electrode` contract implemented by subclasses.
        """

        position_unit_label = units.unit_label(position_unit) or "micrometer"
        x_axis = self.position_values(x_positions, unit=position_unit_label)
        x_m = units.to_array(units.Q_(x_axis, position_unit_label), "meter", dtype=float)
        t_ms = units.to_ms_array(t, dtype=float)
        values = np.zeros((np.atleast_1d(t_ms).size, np.atleast_1d(x_m).size), dtype=float)
        for electrode in self.electrodes:
            stimulus = getattr(electrode, "stimulus", None)
            if stimulus is None:
                raise ValueError("Each extracellular electrode must have an attached stimulus.")
            fp = np.atleast_1d(
                np.asarray(
                    self.footprint_for_electrode(
                        electrode,
                        x_m,
                        axon_y_um=axon_y_um,
                        axon_z_um=axon_z_um,
                    ),
                    dtype=float,
                )
            )
            current_A = np.atleast_1d(
                np.asarray(stimulus.evaluate(t_ms, unit="ampere"), dtype=float)
            )
            values += current_A[:, None] * fp[None, :]
        if voltage_unit is not None:
            values = units.to_array(units.Q_(values, "volt"), voltage_unit, dtype=float)
        return values


@dataclass(frozen=True, kw_only=True)
class AnalyticalExtracellularContext(ExtracellularContext):
    """Homogeneous analytical extracellular context.

    This context evaluates analytical electrodes in an infinite homogeneous
    conductive medium. The conductivity `sigma` belongs here rather than on the
    electrodes, because it describes the extracellular medium shared by all
    electrodes in the context.
    """

    sigma: Any | None = None
    """Homogeneous medium conductivity. If omitted, defaults to 0.3 S/m."""

    def __post_init__(self) -> None:
        """Validate analytical electrodes and normalize conductivity."""

        super().__post_init__()
        from axonscope.stimulation.electrodes import AnalyticalElectrode

        for electrode in self.electrodes:
            if not isinstance(electrode, AnalyticalElectrode):
                raise TypeError(
                    "AnalyticalExtracellularContext requires analytical electrodes."
                )
        sigma = (
            0.3
            if self.sigma is None
            else units.require_conductivity_S_per_m(self.sigma, name="sigma")
        )
        object.__setattr__(self, "sigma", sigma)

    @property
    def sigma_S_m(self) -> float:
        """Conductivity in S/m."""

        return float(self.sigma)

    def with_electrodes(self, electrodes: Sequence["Electrode"]) -> "AnalyticalExtracellularContext":
        """Return an analytical context with the same conductivity."""

        sigma = units.Q_(self.sigma, "siemens / meter")
        return AnalyticalExtracellularContext(electrodes=electrodes, sigma=sigma)

    def footprint_for_electrode(
        self,
        electrode: "Electrode",
        x_positions_m: ArrayLike,
        *,
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Return V/A footprint samples for one analytical electrode."""

        from axonscope.stimulation.electrodes import AnalyticalElectrode

        if not isinstance(electrode, AnalyticalElectrode):
            raise TypeError("electrode must be an AnalyticalElectrode.")
        return electrode.footprint_for_axon(
            x_positions_m,
            sigma_S_m=self.sigma_S_m,
            axon_y_um=axon_y_um,
            axon_z_um=axon_z_um,
        )

    def footprint_per_current(
        self,
        electrode: "AnalyticalElectrode",
        x_positions: ArrayLike,
        *,
        voltage_unit: Any = "volt",
        current_unit: Any = "ampere",
        position_unit: Any = "micrometer",
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Return one electrode footprint expressed as voltage per current."""

        position_unit_label = units.unit_label(position_unit) or "micrometer"
        voltage_unit_label = units.unit_label(voltage_unit) or "volt"
        current_unit_label = units.unit_label(current_unit) or "ampere"
        x_axis = self.position_values(x_positions, unit=position_unit_label)
        x_m = units.to_array(units.Q_(x_axis, position_unit_label), "meter", dtype=float)
        footprint = units.Q_(
            self.footprint_for_electrode(
                electrode,
                x_m,
                axon_y_um=axon_y_um,
                axon_z_um=axon_z_um,
            ),
            "volt / ampere",
        )
        return units.to_array(
            footprint,
            f"{voltage_unit_label} / {current_unit_label}",
            dtype=float,
        )

    def build_footprint(
        self,
        electrode: "AnalyticalElectrode",
        positions: ArrayLike,
        *,
        axon_y: Any | None = None,
        axon_z: Any | None = None,
        source_id: str | None = None,
        axon_id: AxonId | None = None,
    ):
        """Build a static `ExtracellularFootprint` for one analytical electrode."""

        from axonscope.stimulation.extracellular import ExtracellularFootprint

        positions_um = units.require_length_array_um(
            positions,
            name="positions",
            dtype=float,
        )
        values = self.footprint_for_electrode(
            electrode,
            positions_um * 1e-6,
            axon_y_um=0.0 if axon_y is None else axon_y,
            axon_z_um=0.0 if axon_z is None else axon_z,
        )
        if axon_id is not None and not isinstance(axon_id, AxonId):
            raise TypeError("axon_id must be an AxonId.")
        axon_ids = None if axon_id is None else (axon_id,)
        return ExtracellularFootprint(
            values=values,
            positions=units.Q_(positions_um, "micrometer"),
            axon_ids=axon_ids,
            source_id=source_id,
            metadata={
                "builder": "AnalyticalExtracellularContext.build_footprint",
                "sigma_S_m": self.sigma_S_m,
            },
        )

    def activation_function(
        self,
        electrode: "AnalyticalElectrode",
        x_positions: ArrayLike,
        *,
        voltage_unit: Any = "millivolt",
        current_unit: Any = "microampere",
        position_unit: Any = "micrometer",
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
        edge_order: int = 2,
    ) -> np.ndarray:
        """Return Rattay's activation function, d2(footprint)/dx2."""

        x_axis = self.position_values(x_positions, unit=position_unit)
        footprint = self.footprint_per_current(
            electrode,
            x_positions,
            voltage_unit=voltage_unit,
            current_unit=current_unit,
            position_unit=position_unit,
            axon_y_um=axon_y_um,
            axon_z_um=axon_z_um,
        )
        if x_axis.size < 3:
            raise ValueError("At least three positions are required for a second derivative.")
        gradient_order = min(edge_order, x_axis.size - 1)
        first = np.gradient(footprint, x_axis, edge_order=gradient_order)
        return np.gradient(first, x_axis, edge_order=gradient_order)

    def plot_footprint(
        self,
        x_positions: ArrayLike,
        ax: Any | None = None,
        *,
        electrode: "AnalyticalElectrode | None" = None,
        voltage_unit: Any = "millivolt",
        current_unit: Any = "microampere",
        position_unit: Any = "micrometer",
        show_electrode: bool = True,
        title: str | None = None,
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot analytical footprint curves for one or all electrodes."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        electrodes = (electrode,) if electrode is not None else tuple(self.electrodes)
        x_axis = self.position_values(x_positions, unit=position_unit)
        for item in electrodes:
            y = self.footprint_per_current(
                item,
                x_positions,
                voltage_unit=voltage_unit,
                current_unit=current_unit,
                position_unit=position_unit,
            )
            kwargs = dict(plot_kwargs)
            kwargs.setdefault("linewidth", 2.0)
            ax.plot(x_axis, y, **kwargs)
            self._decorate_spatial_axis(
                ax,
                item,
                position_unit=position_unit,
                show_electrode=show_electrode,
            )
        ax.set_ylabel(f"Footprint [{_unit_display(voltage_unit)}/{_unit_display(current_unit)}]")
        ax.set_title(title or self._default_spatial_title(electrodes, position_unit=position_unit))
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    def plot_evaluation(
        self,
        x_positions: ArrayLike,
        t: ArrayLike,
        ax: Any | None = None,
        *,
        voltage_unit: Any = "millivolt",
        position_unit: Any = "micrometer",
        time_unit: Any = "millisecond",
        cmap: str = "coolwarm",
        colorbar: bool = True,
        title: str = "Vext(x, t)",
        **imshow_kwargs: Any,
    ) -> Any:
        """Plot summed extracellular potential for this context."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        x_axis = self.position_values(x_positions, unit=position_unit)
        time_unit_label = units.unit_label(time_unit) or "millisecond"
        t_axis = units.to_array(t, time_unit_label, dtype=float)
        t_for_eval = units.Q_(t_axis, time_unit_label)
        values = self.evaluate(
            x_positions,
            t_for_eval,
            voltage_unit=voltage_unit,
            position_unit=position_unit,
        )
        imshow_kwargs.setdefault("aspect", "auto")
        imshow_kwargs.setdefault("origin", "lower")
        imshow_kwargs.setdefault("cmap", cmap)
        image = ax.imshow(
            values.T,
            extent=[t_axis[0], t_axis[-1], x_axis[0], x_axis[-1]],
            **imshow_kwargs,
        )
        ax.set_xlabel(f"Time [{_unit_display(time_unit)}]")
        ax.set_ylabel(f"Axon position x [{_unit_display(position_unit)}]")
        ax.set_title(f"{title} [{_unit_display(voltage_unit)}]")
        if colorbar:
            ax.figure.colorbar(image, ax=ax, label=f"Vext [{_unit_display(voltage_unit)}]")
        return ax

    def plot_activation_function(
        self,
        x_positions: ArrayLike,
        ax: Any | None = None,
        *,
        electrode: "AnalyticalElectrode | None" = None,
        voltage_unit: Any = "millivolt",
        current_unit: Any = "microampere",
        position_unit: Any = "micrometer",
        show_electrode: bool = True,
        title: str = "Rattay activation function",
        grid: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot Rattay's activation function for one or all electrodes."""

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        electrodes = (electrode,) if electrode is not None else tuple(self.electrodes)
        x_axis = self.position_values(x_positions, unit=position_unit)
        for item in electrodes:
            y = self.activation_function(
                item,
                x_positions,
                voltage_unit=voltage_unit,
                current_unit=current_unit,
                position_unit=position_unit,
            )
            kwargs = dict(plot_kwargs)
            kwargs.setdefault("linewidth", 2.0)
            ax.plot(x_axis, y, **kwargs)
            self._decorate_spatial_axis(
                ax,
                item,
                position_unit=position_unit,
                show_electrode=show_electrode,
            )
        ax.set_ylabel(
            "d2V/dx2 "
            f"[{_unit_display(voltage_unit)}/{_unit_display(current_unit)}/"
            f"{_unit_display(position_unit)}^2]"
        )
        ax.set_title(title)
        if grid:
            ax.grid(True, alpha=0.3)
        return ax

    def _decorate_spatial_axis(
        self,
        ax: Any,
        electrode: "Electrode",
        *,
        position_unit: Any,
        show_electrode: bool,
    ) -> None:
        """Apply common labels and optional electrode-position markers."""

        ax.set_xlabel(f"Axon position x [{_unit_display(position_unit)}]")
        if show_electrode and hasattr(electrode, "x_um"):
            position_unit_label = units.unit_label(position_unit) or "micrometer"
            x_source = units.to_scalar(units.Q_(electrode.x_um, "micrometer"), position_unit_label)
            ax.axvline(x_source, color="black", linestyle="--", linewidth=1.0, label="electrode x")
            ax.legend()

    def _default_spatial_title(
        self,
        electrodes: Sequence["Electrode"],
        *,
        position_unit: Any,
    ) -> str:
        """Return a compact default title for spatial context plots."""

        if len(electrodes) == 1 and hasattr(electrodes[0], "z_um"):
            position_unit_label = units.unit_label(position_unit) or "micrometer"
            z_value = units.to_scalar(units.Q_(electrodes[0].z_um, "micrometer"), position_unit_label)
            return f"Point source z={z_value:g} {_unit_display(position_unit)}"
        return f"Analytical context ({len(electrodes)} electrodes)"


@dataclass(frozen=True, kw_only=True)
class NRVExtracellularContext(ExtracellularContext):
    """Extracellular context placeholder for NRV-managed FEM fields.

    This class reserves the public shape for future NRV FEM integration:
    electrodes remain AxonScope objects with attached stimuli, while `medium`,
    `fem_model`, and `metadata` carry NRV/FEM configuration. Footprint
    generation is intentionally not implemented yet.
    """

    medium: Any | None = None
    """NRV medium/material description, such as an endoneurium preset."""

    fem_model: Any | None = None
    """NRV FEM model or handle used to compute transfer functions."""

    backend: str = "nrv"
    """Backend identifier reserved for dispatch and diagnostics."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Free-form immutable metadata for future FEM integration."""

    def __post_init__(self) -> None:
        """Validate electrodes and freeze NRV-specific metadata."""

        super().__post_init__()
        backend = str(self.backend).strip()
        if not backend:
            raise ValueError("backend must be a non-empty string.")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def with_electrodes(self, electrodes: Sequence["Electrode"]) -> "NRVExtracellularContext":
        """Return an NRV context with the same FEM configuration."""

        return NRVExtracellularContext(
            electrodes=electrodes,
            medium=self.medium,
            fem_model=self.fem_model,
            backend=self.backend,
            metadata=dict(self.metadata),
        )

    def footprint_for_electrode(
        self,
        electrode: "Electrode",
        x_positions_m: ArrayLike,
        *,
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Raise until NRV FEM footprint evaluation is implemented."""

        raise NotImplementedError(
            "NRVExtracellularContext is a configuration placeholder; "
            "NRV/FEM footprint evaluation is not implemented yet."
        )
