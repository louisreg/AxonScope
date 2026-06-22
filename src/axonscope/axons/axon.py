"""Descriptive axon model built from a section layout."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonscope.utils import units
from axonscope.axons.formulation import (
    CableFormulation,
    Formulation,
    normalize_formulation,
    resolve_layout_formulation,
)
from axonscope.axons.layout import Layout


_DEFAULT_V_INIT = units.Q_(-70.0, "millivolt")
_DEFAULT_TEMPERATURE = units.Q_(37.0, "degree_Celsius")


class Axon:
    """Descriptive axon: section layout, formulation, and membrane state.

    `Axon` owns user-facing model description only: a section `Layout`, the
    optional cable formulation, and global membrane state such as initial
    voltage and temperature. Numerical solver arrays are built separately at
    the solver boundary.

    Physical values at this boundary must carry units so user-facing model
    state is explicit.
    """

    def __init__(
        self,
        *,
        layout: Layout,
        formulation: CableFormulation | None = None,
        diameter: Any | None = None,
        v_init: units.voltage_t = _DEFAULT_V_INIT,
        temperature: units.temperature_t = _DEFAULT_TEMPERATURE,
    ) -> None:
        """Create an axon from a descriptive section layout.

        Parameters
        ----------
        layout:
            Conceptual spatial section layout.
        formulation:
            Optional cable formulation. When omitted, AxonScope infers it from
            the presence or absence of periaxonal layers.
        diameter:
            Optional nominal axon diameter, with units. Templates pass the
            constructor diameter here so it remains easy to inspect even when
            their internal layout has section-specific diameters.
        v_init:
            Initial membrane potential, with units convertible to millivolts.
        temperature:
            Model temperature, with units convertible to degrees Celsius.
        """

        if not isinstance(layout, Layout):
            raise TypeError("layout must be an axonscope.axons.Layout.")
        self._layout: Layout
        self.layout = layout
        normalize_formulation(formulation)
        self.formulation = formulation
        self._diameter_um = (
            None
            if diameter is None
            else units.require_length_um(diameter, name="diameter")
        )
        self.v_init = units.require_voltage_mV(v_init, name="v_init")
        self.temperature = units.require_temperature_degC(temperature, name="temperature")

    @property
    def layout(self) -> Layout:
        """Descriptive section layout owned by this axon."""

        return self._layout

    @layout.setter
    def layout(self, value: Layout) -> None:
        """Set the descriptive section layout."""

        if not isinstance(value, Layout):
            raise TypeError("layout must be an axonscope.axons.Layout.")
        self._layout = value

    @property
    def resolved_formulation(self) -> Formulation:
        """Cable formulation after validating or inferring from sections."""

        return resolve_layout_formulation(self.layout, self.formulation)

    @property
    def n_compartments(self) -> int:
        """Number of numerical compartments."""

        return self.layout.compartments

    @property
    def length(self) -> float:
        """Total axon length in micrometers."""

        return self.layout.length_um

    @property
    def diameter(self) -> float:
        """Nominal axon diameter in micrometers.

        Template axons return the diameter passed to their constructor. Custom
        layouts without an explicit nominal diameter return the shared diameter
        when all compartments are uniform. For genuinely heterogeneous layouts,
        use `diameter_values(...)` to inspect per-compartment values.
        """

        if self._diameter_um is not None:
            return float(self._diameter_um)
        diameters_um = self.diameter_values(unit="micrometer")
        reference = float(diameters_um[0])
        if np.allclose(diameters_um, reference, rtol=1e-9, atol=1e-12):
            return reference
        raise ValueError(
            "axon has non-uniform diameters; use axon.diameter_values(unit=...) "
            "for per-compartment values."
        )

    def diameter_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return compartment diameters as plain values in `unit`."""

        return self.layout.diameter_values(unit=unit)

    def compartment_position(self, index: int, *, unit: Any = "micrometer") -> units.length_t:
        """Return one compartment-center position as a unit-bearing quantity."""

        return self.layout.compartment_position(index, unit=unit)


__all__ = ["Axon"]
