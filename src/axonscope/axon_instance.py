"""Concrete axon instance used by simulation execution."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from axonscope.axons.axon import Axon
from axonscope.stimulation import (
    ExtracellularStimulation,
    IntracellularContext,
    IntracellularCurrentClamp,
    Stimulus,
)
from axonscope.utils import units


class AxonInstance:
    """One concrete occurrence of a descriptive axon.

    `Axon` objects remain purely descriptive: geometry, layout, cable
    formulation, and membrane models. `AxonInstance` binds one descriptive axon
    to local stimulation contexts for a concrete run. It does not own
    anatomical/world placement; external geometry should be converted to
    sampled footprints/drives before attachment.

    Parameters
    ----------
    axon:
        Descriptive axon model.
    """

    def __init__(
        self,
        axon: Axon,
    ) -> None:
        if not isinstance(axon, Axon):
            raise TypeError("axon must be an axonscope.axons.Axon instance.")
        self.axon = axon
        self.intracellular_contexts: list[IntracellularContext] = []
        self.extracellular_stimulation: ExtracellularStimulation | None = None
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
        return (
            self.axon.resolved_formulation == "double-cable"
            or self.extracellular_stimulation is not None
        )

    @use_extracellular.setter
    def use_extracellular(self, value: bool) -> None:
        """Force-enable or force-disable extracellular solver handling."""

        self._use_extracellular_override = bool(value)

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
        position: Any,
        current: Stimulus,
    ) -> None:
        """Convenience wrapper for adding an `IntracellularCurrentClamp`.

        `position` must carry length units. Plain stimulus amplitudes are
        interpreted as nanoamperes.
        """

        self.add_intracellular_context(
            context=IntracellularCurrentClamp(position=position, current=current)
        )

    def clear_intracellular_contexts(self) -> None:
        """Remove all intracellular stimulation contexts."""

        self.intracellular_contexts.clear()

    def add_extracellular_stimulation(
        self,
        *,
        stimulation: ExtracellularStimulation,
        replace: bool = False,
        enable: bool = True,
    ) -> None:
        """Attach sampled extracellular stimulation.

        This is the canonical high-level extracellular path: helpers or
        external tools build `ExtracellularFootprint`/`ExtracellularDrive`
        objects, then the simulation receives the resulting immutable
        `ExtracellularStimulation`.
        """

        if not isinstance(stimulation, ExtracellularStimulation):
            raise TypeError(
                "stimulation must be an axonscope.stimulation.ExtracellularStimulation."
            )
        if self.extracellular_stimulation is not None and not replace:
            raise ValueError(
                "AxonInstance accepts one extracellular stimulation. "
                "Use ExtracellularStimulation([...]) for multiple drives, "
                "or pass replace=True."
            )
        self.extracellular_stimulation = stimulation
        if enable:
            self._use_extracellular_override = True

    @property
    def extracellular_stimulations(self) -> tuple[ExtracellularStimulation, ...]:
        """Return the optional extracellular stimulation as a runtime tuple."""

        if self.extracellular_stimulation is None:
            return ()
        return (self.extracellular_stimulation,)

    def clear_extracellular_stimulation(self) -> None:
        """Remove the extracellular stimulation."""

        self.extracellular_stimulation = None
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
            * 1e-6
        )
        vext = np.zeros((self.axon.n_compartments,), dtype=self.dtype)
        for stimulation in self.extracellular_stimulations:
            sample_mV = stimulation.evaluate(
                [t_value_ms],
                voltage_unit="millivolt",
            )[0]
            if sample_mV.shape != x_positions_m.shape:
                x_um = x_positions_m * 1e6
                sample_mV = np.interp(
                    x_um,
                    np.asarray(stimulation.positions_um, dtype=float),
                    np.asarray(sample_mV, dtype=float),
                )
            vext = vext + sample_mV.astype(vext.dtype)
        return vext


def as_axon_instance(value: Axon | AxonInstance) -> AxonInstance:
    """Return `value` as an `AxonInstance`.

    Passing a pure `Axon` creates a no-stimulation instance around it. This
    keeps low-level solver calls convenient while preserving `Axon` as a pure
    descriptive object.
    """

    if isinstance(value, AxonInstance):
        return value
    if isinstance(value, Axon):
        return AxonInstance(value)
    raise TypeError(f"expected Axon or AxonInstance, got {type(value)!r}.")


__all__ = ["AxonInstance", "as_axon_instance"]
