"""Simulation protocol attached to a descriptive axon."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from axonscope.axons.axon import Axon
from axonscope.stimulation import (
    ExtracellularContext,
    IntracellularContext,
    IntracellularCurrentClamp,
    Stimulus,
)
from axonscope.utils import units


class AxonSimulation:
    """An axon plus the protocol needed to simulate it.

    `Axon` objects remain purely descriptive: geometry, layout, cable
    formulation, and membrane models. `AxonSimulation` is the user-facing
    object that adds spatial placement and stimulation contexts.

    Parameters
    ----------
    axon:
        Descriptive axon model.
    x_offset_um, y_um, z_um:
        Axon origin in the global simulation frame. Plain numbers are
        interpreted as micrometers; Pint quantities are converted.
    """

    def __init__(
        self,
        axon: Axon,
        *,
        x_offset_um: Any = 0.0,
        y_um: Any = 0.0,
        z_um: Any = 0.0,
    ) -> None:
        if not isinstance(axon, Axon):
            raise TypeError("axon must be an axonscope.axons.Axon instance.")
        self.axon = axon
        self.intracellular_contexts: list[IntracellularContext] = []
        self.intracellular_clamps = self.intracellular_contexts
        self.extracellular_context: ExtracellularContext | None = None
        self.x_offset_um = units.to_um(x_offset_um)
        self.y_um = units.to_um(y_um)
        self.z_um = units.to_um(z_um)
        self.Veinit = units.to_mV(0.0)
        self._use_extracellular_override: bool | None = None
        self._xraxial_override: np.ndarray | None = None
        self._xg_override: np.ndarray | None = None
        self._xc_override: np.ndarray | None = None

    @property
    def dtype(self) -> np.dtype:
        """Numerical dtype derived from the wrapped axon's membrane layout."""

        return np.dtype(self.axon.layout.sections[0].membrane.dtype)

    def __getattr__(self, name: str) -> Any:
        """Delegate descriptive axon attributes to the wrapped axon."""

        return getattr(self.axon, name)

    @property
    def use_extracellular(self) -> bool:
        """Whether the solver should include extracellular handling."""

        if self._use_extracellular_override is not None:
            return bool(self._use_extracellular_override)
        return self.axon.resolved_formulation == "double-cable" or self.extracellular_context is not None

    @use_extracellular.setter
    def use_extracellular(self, value: bool) -> None:
        """Force-enable or force-disable extracellular solver handling."""

        self._use_extracellular_override = bool(value)

    def set_position(
        self,
        *,
        x_offset_um: Any = 0.0,
        y_um: Any = 0.0,
        z_um: Any = 0.0,
    ) -> None:
        """Set the axon's spatial offset in the global simulation frame.

        Parameters
        ----------
        x_offset_um, y_um, z_um:
            Axon origin in micrometers. Plain numbers are interpreted as
            micrometers; Pint quantities are converted.
        """

        self.x_offset_um = units.to_um(x_offset_um)
        self.y_um = units.to_um(y_um)
        self.z_um = units.to_um(z_um)

    def add_intracellular_context(
        self,
        *,
        context: IntracellularContext,
    ) -> None:
        """Attach an intracellular stimulation context.

        Parameters
        ----------
        context:
            Intracellular context describing where and how current is injected.
            Use `IntracellularCurrentClamp` for point current injection.
        """

        if not isinstance(context, IntracellularContext):
            raise TypeError(
                "context must be an axonscope.stimulation.IntracellularContext."
            )
        self.intracellular_contexts.append(context)

    def add_current_clamp(
        self,
        *,
        position_um: Any,
        current: Stimulus,
    ) -> None:
        """Convenience wrapper for adding an `IntracellularCurrentClamp`.

        Plain numeric positions are interpreted as micrometers, and plain
        stimulus amplitudes are interpreted as nanoamperes.
        """

        self.add_intracellular_context(
            context=IntracellularCurrentClamp(position_um=position_um, current=current)
        )

    def clear_intracellular_contexts(self) -> None:
        """Remove all intracellular stimulation contexts."""

        self.intracellular_contexts.clear()

    def add_extracellular_context(
        self,
        *,
        context: ExtracellularContext,
        replace: bool = False,
        enable: bool = True,
    ) -> None:
        """Attach an extracellular stimulation context.

        Parameters
        ----------
        context:
            Extracellular context containing one or more stimulated electrodes.
        replace:
            Replace the existing context. If false, adding a second
            extracellular context raises an error; put multiple electrodes in
            one `ExtracellularContext` instead.
        enable:
            Force-enable extracellular solver handling.
        """

        if not isinstance(context, ExtracellularContext):
            raise TypeError("context must be an axonscope.stimulation.ExtracellularContext.")

        if self.extracellular_context is not None and not replace:
            raise ValueError(
                "AxonSimulation accepts one extracellular context. "
                "Use ExtracellularContext(electrodes=[...]) for multiple electrodes, "
                "or pass replace=True."
            )
        self.extracellular_context = context
        if enable:
            self._use_extracellular_override = True

    @property
    def extracellular_contexts(self) -> tuple[ExtracellularContext, ...]:
        """Return the optional extracellular context as a runtime tuple."""

        if self.extracellular_context is None:
            return ()
        return (self.extracellular_context,)

    def clear_extracellular_contexts(self) -> None:
        """Remove the extracellular stimulation context."""

        self.extracellular_context = None
        self._use_extracellular_override = None

    def _validate_layer_array(self, value: Any, name: str, unit: str) -> np.ndarray:
        arr = units.to_array(value, unit, dtype=self.dtype)
        if arr.shape != (self.axon.n_compartments,):
            raise ValueError(
                f"{name} must have shape ({self.axon.n_compartments},), got {arr.shape}"
            )
        return arr

    def set_extracellular_layer(
        self,
        *,
        xraxial_MOhm_per_cm: Optional[Any] = None,
        xg_S_per_cm2: Optional[Any] = None,
        xc_uF_per_cm2: Optional[Any] = None,
        use_extracellular: Optional[bool] = None,
        Veinit: Optional[Any] = None,
    ) -> None:
        """Override periaxonal arrays for advanced double-cable simulations.

        This belongs to the simulation protocol rather than to `Axon`: it
        changes how this run treats extracellular state without changing the
        descriptive axon layout.

        Parameters
        ----------
        xraxial_MOhm_per_cm:
            Per-compartment periaxonal axial resistance density.
        xg_S_per_cm2:
            Per-compartment periaxonal radial conductance density.
        xc_uF_per_cm2:
            Per-compartment periaxonal radial capacitance density.
        use_extracellular:
            Optional explicit solver flag for extracellular handling.
        Veinit:
            Initial extracellular potential in millivolts.
        """

        if xraxial_MOhm_per_cm is not None:
            self._xraxial_override = self._validate_layer_array(
                xraxial_MOhm_per_cm,
                "xraxial",
                "megaohm / centimeter",
            )
        if xg_S_per_cm2 is not None:
            self._xg_override = self._validate_layer_array(
                xg_S_per_cm2,
                "xg",
                "siemens / centimeter ** 2",
            )
        if xc_uF_per_cm2 is not None:
            self._xc_override = self._validate_layer_array(
                xc_uF_per_cm2,
                "xc",
                "microfarad / centimeter ** 2",
            )
        if use_extracellular is not None:
            self._use_extracellular_override = bool(use_extracellular)
        if Veinit is not None:
            self.Veinit = units.to_mV(Veinit)

    def extracellular_potential_mV(self, t_ms: Any) -> np.ndarray:
        """Return imposed extracellular potential in mV at time `t_ms`.

        `t_ms` accepts plain milliseconds or a Pint time quantity.
        """

        t_value_ms = units.to_ms(t_ms)
        x_positions_m = (
            np.asarray(self.axon.layout.position_values(unit="micrometer"), dtype=float)
            + self.x_offset_um
        ) * 1e-6
        vext = np.zeros((self.axon.n_compartments,), dtype=self.dtype)
        for ctx in self.extracellular_contexts:
            sample_mV = ctx.evaluate(
                x_positions_m,
                [t_value_ms],
                voltage_unit="millivolt",
                position_unit="meter",
                axon_y_um=self.y_um,
                axon_z_um=self.z_um,
            )[0]
            vext = vext + sample_mV.astype(vext.dtype)
        return vext


def as_axon_simulation(value: Axon | AxonSimulation) -> AxonSimulation:
    """Return `value` as an `AxonSimulation`.

    Passing a pure `Axon` creates a no-stimulation protocol around it. This
    keeps low-level solver calls convenient while preserving `Axon` as a pure
    descriptive object.
    """

    if isinstance(value, AxonSimulation):
        return value
    if isinstance(value, Axon):
        return AxonSimulation(value)
    raise TypeError(f"expected Axon or AxonSimulation, got {type(value)!r}.")


__all__ = ["AxonSimulation", "as_axon_simulation"]
