"""Unmyelinated axon family and membrane templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope import membranes
from axonscope.axons.axon import Axon
from axonscope.axons.formulation import Formulation
from axonscope.axons.layout import Layout
from axonscope.axons.section import Section
from axonscope.utils import units
from axonscope.utils.units import (
    axial_resistivity_t,
    capacitance_density_t,
    length_t,
    temperature_t,
    voltage_t,
)
from axonscope.utils.validation import normalize_positive_int


_DEFAULT_Ra = units.Q_(100.0, "ohm * centimeter")
_DEFAULT_Cm = units.Q_(1.0, "microfarad / centimeter ** 2")
_DEFAULT_HH_Ra = units.Q_(200.0, "ohm * centimeter")
_DEFAULT_SCHILD_Cm = units.Q_(1.326291192, "microfarad / centimeter ** 2")


def _quantity_um(value: float | np.ndarray) -> length_t:
    return units.Q_(value, "micrometer")


def _quantity_degC(value: temperature_t, *, name: str) -> temperature_t:
    return units.Q_(units.require_temperature_degC(value, name=name), "degree_Celsius")


def _quantity_mV(value: voltage_t, *, name: str) -> voltage_t:
    return units.Q_(units.require_voltage_mV(value, name=name), "millivolt")


@dataclass(frozen=True)
class _GeometrySpec:
    length: length_t | None
    diameter: length_t
    compartments: int | None
    x: length_t | None


def _validate_geometry(
    *,
    length: length_t | None = None,
    diameter: length_t,
    compartments: int | None = None,
    x: length_t | None = None,
) -> _GeometrySpec:
    if x is not None and (length is not None or compartments is not None):
        raise ValueError("Provide either x, or both length and compartments, not both.")
    if x is None and (length is None or compartments is None):
        raise ValueError("Provide x or both length and compartments.")
    compartment_count = None
    if compartments is not None:
        compartment_count = normalize_positive_int(compartments, name="compartments")
        if compartment_count < 2:
            raise ValueError(f"compartments must be >= 2, got {compartment_count}.")

    length_value = None if length is None else units.require_length_um(length, name="length")
    diameter_value = units.require_length_um(diameter, name="diameter")
    x_value = None if x is None else units.require_length_array_um(x, name="x", dtype=np.float32)
    return _GeometrySpec(
        length=None if length_value is None else _quantity_um(length_value),
        diameter=_quantity_um(diameter_value),
        compartments=compartment_count,
        x=None if x_value is None else _quantity_um(x_value),
    )


def _passive_leak(g_pas: Any, e_pas: Any) -> membranes.MembraneModel:
    conductance = units.to_S_per_cm2(g_pas)
    if conductance <= 0.0:
        raise ValueError("g_pas must be strictly positive when include_passive_leak=True.")
    return membranes.Passive(Rm=1.0 / conductance, EL=e_pas)


def _single_section_model(
    *,
    membrane: membranes.MembraneModel,
    length: length_t | None,
    diameter: length_t,
    compartments: int | None,
    x: length_t | None,
    Ra: axial_resistivity_t | None,
    Cm: capacitance_density_t | None,
) -> Layout:
    section = Section(
        "axon",
        membrane=membrane,
        diameter=diameter,
        Ra=Ra,
        Cm=Cm,
        tags=("unmyelinated",),
    )
    if x is not None:
        return Layout.single_non_uniform(section, x=x)
    assert length is not None
    assert compartments is not None
    return Layout.single_uniform(
        section,
        length=length,
        compartments=compartments,
    )


class Unmyelinated(Axon):
    """Base class for unmyelinated axon descriptions.

    A user may either provide a ready-made `Layout`, or provide one membrane
    model plus geometry. The latter creates a one-section single-cable layout,
    which is conceptually the one-compartment case of the general multi-section
    representation.
    """

    def __init__(
        self,
        *,
        layout: Layout | None = None,
        membrane: membranes.MembraneModel | None = None,
        formulation: Formulation | None = "single-cable",
        length: length_t | None = None,
        diameter: length_t | None = None,
        compartments: int | None = None,
        x: length_t | None = None,
        Ra: axial_resistivity_t | None = None,
        Cm: capacitance_density_t | None = None,
        v_init: voltage_t = units.Q_(-70.0, "millivolt"),
        temperature: temperature_t = units.Q_(37.0, "degree_Celsius"),
    ) -> None:
        """Create an unmyelinated axon from a layout or one membrane section.

        Parameters
        ----------
        layout:
            Ready-made section layout. Mutually exclusive with `membrane`.
        membrane:
            Membrane model used for a single-section axon.
        formulation:
            Cable formulation, normally `"single-cable"` for unmyelinated
            models.
        length:
            Total axon length, with units. Required with `compartments`
            unless `x` is provided.
        diameter:
            Axon diameter, with units.
        compartments:
            Number of numerical compartments for a uniform layout.
        x:
            Explicit compartment-center coordinates, with units.
        Ra:
            Axial resistivity in ohm * centimeter.
        Cm:
            Membrane capacitance density in microfarad / centimeter^2.
        v_init:
            Initial membrane potential in millivolts.
        temperature:
            Model temperature in degrees Celsius.
        """

        if layout is None:
            if membrane is None:
                raise ValueError("Provide either layout or membrane with geometry.")
            if diameter is None:
                raise ValueError("diameter is required.")
            geometry = _validate_geometry(
                length=length,
                diameter=diameter,
                compartments=compartments,
                x=x,
            )
            layout = _single_section_model(
                membrane=membrane,
                length=geometry.length,
                diameter=geometry.diameter,
                compartments=geometry.compartments,
                x=geometry.x,
                Ra=Ra,
                Cm=Cm,
            )
        elif membrane is not None:
            raise ValueError("Provide either layout or membrane, not both.")
        elif any(value is not None for value in (length, diameter, compartments, x, Ra, Cm)):
            raise ValueError(
                "Section-building arguments are only valid when building from a membrane."
            )

        super().__init__(
            layout=layout,
            formulation=formulation,
            v_init=v_init,
            temperature=temperature,
        )


class HodgkinHuxley(Unmyelinated):
    """Hodgkin-Huxley unmyelinated template."""

    def __init__(
        self,
        *,
        diameter: length_t,
        length: length_t | None = None,
        compartments: int | None = None,
        x: length_t | None = None,
        Ra: axial_resistivity_t = _DEFAULT_HH_Ra,
        Cm: capacitance_density_t = _DEFAULT_Cm,
        v_init: voltage_t = units.Q_(-67.5, "millivolt"),
        gnabar: Any = 0.12,
        gkbar: Any = 0.036,
        gl: Any = 0.0003,
        el: Any = -54.3,
        ena: Any = 50.0,
        ek: Any = -77.0,
        celsius: temperature_t = units.Q_(6.3, "degree_Celsius"),
        include_passive_leak: bool = False,
        g_pas: Any = 0.001,
        e_pas: Any = -70.0,
        formulation: Formulation | None = "single-cable",
    ) -> None:
        """Create a Hodgkin-Huxley unmyelinated axon.

        Parameters
        ----------
        diameter:
            Axon diameter, with units.
        length:
            Axon length, with units. Required with `compartments` unless
            explicit `x` coordinates are provided.
        compartments:
            Number of numerical compartments for a uniform layout.
        x:
            Explicit compartment-center coordinates, with units.
        Ra:
            Axial resistivity in ohm * centimeter.
        Cm:
            Membrane capacitance density in microfarad / centimeter^2.
        v_init:
            Initial membrane potential in millivolts.
        gnabar, gkbar, gl:
            Sodium, potassium, and leak conductance densities in S/cm^2.
        el, ena, ek:
            Reversal potentials in millivolts.
        celsius:
            Model temperature in degrees Celsius.
        include_passive_leak:
            Add an additional passive leak membrane in parallel.
        g_pas, e_pas:
            Additional passive leak conductance density and reversal potential.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        celsius = _quantity_degC(celsius, name="celsius")
        hh_model = membranes.HodgkinHuxley(
            gnabar=gnabar,
            gkbar=gkbar,
            gl=gl,
            el=el,
            ena=ena,
            ek=ek,
            celsius=celsius,
        )
        membrane = (
            membranes.Composite([hh_model, _passive_leak(g_pas, e_pas)])
            if include_passive_leak
            else hh_model
        )
        super().__init__(
            membrane=membrane,
            length=length,
            diameter=diameter,
            compartments=compartments,
            x=x,
            Ra=Ra,
            Cm=Cm,
            v_init=v_init,
            temperature=celsius,
            formulation=formulation,
        )


class RattayAberham(Unmyelinated):
    """Rattay-Aberham unmyelinated template."""

    def __init__(
        self,
        *,
        diameter: length_t,
        length: length_t | None = None,
        compartments: int | None = None,
        x: length_t | None = None,
        Cm: capacitance_density_t = _DEFAULT_Cm,
        Ra: axial_resistivity_t = _DEFAULT_Ra,
        v_init: voltage_t = units.Q_(-70.0, "millivolt"),
        gnabar: Any = 0.12,
        gkbar: Any = 0.036,
        gl: Any = 0.0003,
        el: Any = -59.4,
        ena: Any = 50.0,
        ek: Any = -82.0,
        celsius: temperature_t = units.Q_(37.0, "degree_Celsius"),
        include_passive_leak: bool = True,
        g_pas: Any = 0.001,
        e_pas: Any = -70.0,
        formulation: Formulation | None = "single-cable",
    ) -> None:
        """Create a Rattay-Aberham unmyelinated axon.

        Parameters
        ----------
        diameter, length, compartments, x:
            Geometry values with explicit length units.
        Cm:
            Membrane capacitance density in uF/cm^2.
        Ra:
            Axial resistivity in ohm * centimeter.
        v_init:
            Initial membrane potential in millivolts.
        gnabar, gkbar, gl:
            Sodium, potassium, and leak conductance densities in S/cm^2.
        el, ena, ek:
            Reversal potentials in millivolts.
        celsius:
            Model temperature in degrees Celsius.
        include_passive_leak:
            Add an additional passive leak membrane in parallel.
        g_pas, e_pas:
            Additional passive leak conductance density and reversal potential.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        celsius = _quantity_degC(celsius, name="celsius")
        rattay = membranes.RattayAberham(
            gnabar=gnabar,
            gkbar=gkbar,
            gl=gl,
            el=el,
            ena=ena,
            ek=ek,
            celsius=celsius,
        )
        membrane = (
            membranes.Composite([rattay, _passive_leak(g_pas, e_pas)])
            if include_passive_leak
            else rattay
        )
        super().__init__(
            membrane=membrane,
            length=length,
            diameter=diameter,
            compartments=compartments,
            x=x,
            Ra=Ra,
            Cm=Cm,
            v_init=v_init,
            temperature=celsius,
            formulation=formulation,
        )


class Sundt(Unmyelinated):
    """Sundt unmyelinated composite membrane template."""

    def __init__(
        self,
        *,
        diameter: length_t,
        length: length_t | None = None,
        compartments: int | None = None,
        x: length_t | None = None,
        Cm: capacitance_density_t = _DEFAULT_Cm,
        Ra: axial_resistivity_t = _DEFAULT_Ra,
        v_init: voltage_t = units.Q_(-60.0, "millivolt"),
        celsius: temperature_t = units.Q_(37.0, "degree_Celsius"),
        gnabar: Any = 0.04,
        gkdrbar: Any = 0.04,
        ena: Any = 45.0,
        ek: Any = -90.0,
        Rm: Any = 10000.0,
        El: Any = -70.0,
        formulation: Formulation | None = "single-cable",
    ) -> None:
        """Create a Sundt unmyelinated axon.

        Parameters
        ----------
        diameter, length, compartments, x:
            Geometry values with explicit length units.
        Cm:
            Membrane capacitance density in uF/cm^2.
        Ra:
            Axial resistivity in ohm * centimeter.
        v_init:
            Initial membrane potential in millivolts.
        celsius:
            Model temperature in degrees Celsius.
        gnabar, gkdrbar:
            Sodium and delayed-rectifier potassium conductance densities in S/cm^2.
        ena, ek, El:
            Reversal potentials in millivolts.
        Rm:
            Passive membrane resistance in ohm * cm^2.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        celsius = _quantity_degC(celsius, name="celsius")
        super().__init__(
            membrane=membranes.Sundt(
                celsius=celsius,
                gnabar=gnabar,
                gkdrbar=gkdrbar,
                ena=ena,
                ek=ek,
                Rm=Rm,
                El=El,
            ),
            length=length,
            diameter=diameter,
            compartments=compartments,
            x=x,
            Ra=Ra,
            Cm=Cm,
            v_init=v_init,
            temperature=celsius,
            formulation=formulation,
        )


class Tigerholm(Unmyelinated):
    """Tigerholm et al. 2014 mammalian C-fiber template."""

    def __init__(
        self,
        *,
        diameter: length_t,
        length: length_t | None = None,
        compartments: int | None = None,
        x: length_t | None = None,
        Cm: capacitance_density_t = _DEFAULT_Cm,
        Ra: axial_resistivity_t = units.Q_(35.5, "ohm * centimeter"),
        v_init: voltage_t = units.Q_(-62.0, "millivolt"),
        celsius: temperature_t = units.Q_(37.0, "degree_Celsius"),
        ena: Any = 71.5,
        ek: Any = -87.0,
        gbar_nav17: Any = 0.10664,
        gbar_nav18: Any = 0.24271,
        gbar_nav19: Any = 9.4779e-05,
        gbar_ks: Any = 0.0069733,
        gbar_kf: Any = 0.012756,
        gbar_kdr: Any = 0.018002,
        gbar_h: Any = 0.0025377,
        gbar_kna: Any = 0.00042,
        nai_fixed: Any = 11.4,
        pump_smalla: Any = -0.0047891,
        pump_ko: Any = 5.6,
        formulation: Formulation | None = "single-cable",
    ) -> None:
        """Create a Tigerholm et al. C-fiber axon.

        Parameters
        ----------
        diameter, length, compartments, x:
            Geometry values with explicit length units.
        Cm:
            Membrane capacitance density in uF/cm^2.
        Ra:
            Axial resistivity in ohm * centimeter.
        v_init:
            Initial membrane potential in millivolts.
        celsius:
            Model temperature in degrees Celsius.
        ena, ek:
            Sodium and potassium reversal potentials in millivolts.
        gbar_nav17, gbar_nav18, gbar_nav19, gbar_ks, gbar_kf, gbar_kdr, gbar_h, gbar_kna:
            Channel conductance densities in S/cm^2.
        nai_fixed, pump_ko:
            Concentrations in millimolar.
        pump_smalla:
            Na/K pump current-density coefficient in mA/cm^2.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        celsius = _quantity_degC(celsius, name="celsius")
        diameter_um = units.require_length_um(diameter, name="diameter")
        super().__init__(
            membrane=membranes.Tigerholm(
                diameter_um=diameter_um,
                celsius=celsius,
                ena=ena,
                ek=ek,
                gbar_nav17=gbar_nav17,
                gbar_nav18=gbar_nav18,
                gbar_nav19=gbar_nav19,
                gbar_ks=gbar_ks,
                gbar_kf=gbar_kf,
                gbar_kdr=gbar_kdr,
                gbar_h=gbar_h,
                gbar_kna=gbar_kna,
                nai_fixed=nai_fixed,
                pump_smalla=pump_smalla,
                pump_ko=pump_ko,
            ),
            length=length,
            diameter=diameter,
            compartments=compartments,
            x=x,
            Ra=Ra,
            Cm=Cm,
            v_init=v_init,
            temperature=celsius,
            formulation=formulation,
        )


class Schild94(Unmyelinated):
    """Schild et al. 1994 DRG C-fiber template."""

    def __init__(
        self,
        *,
        diameter: length_t,
        length: length_t | None = None,
        compartments: int | None = None,
        x: length_t | None = None,
        Ra: axial_resistivity_t = _DEFAULT_Ra,
        Cm: capacitance_density_t = _DEFAULT_SCHILD_Cm,
        v_init: voltage_t = units.Q_(-48.0, "millivolt"),
        temperature: temperature_t = units.Q_(37.0, "degree_Celsius"),
        formulation: Formulation | None = "single-cable",
    ) -> None:
        """Create a Schild et al. 1994 DRG C-fiber axon.

        Parameters
        ----------
        diameter, length, compartments, x:
            Geometry values with explicit length units.
        Ra:
            Axial resistivity in ohm * centimeter.
        Cm:
            Membrane capacitance density in uF/cm^2.
        v_init:
            Initial membrane potential in millivolts.
        temperature:
            Model temperature in degrees Celsius.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        temperature = _quantity_degC(temperature, name="temperature")
        v_init = _quantity_mV(v_init, name="v_init")
        diameter_um = units.require_length_um(diameter, name="diameter")
        super().__init__(
            membrane=membranes.Schild94(
                diameter_um=diameter_um,
                temp_c=temperature,
                vinit_mV=v_init,
            ),
            length=length,
            diameter=diameter,
            compartments=compartments,
            x=x,
            Ra=Ra,
            Cm=Cm,
            v_init=v_init,
            temperature=temperature,
            formulation=formulation,
        )


class Schild97(Unmyelinated):
    """Schild and Bhatt 1997 DRG C-fiber template."""

    def __init__(
        self,
        *,
        diameter: length_t,
        length: length_t | None = None,
        compartments: int | None = None,
        x: length_t | None = None,
        Ra: axial_resistivity_t = _DEFAULT_Ra,
        Cm: capacitance_density_t = _DEFAULT_SCHILD_Cm,
        v_init: voltage_t = units.Q_(-48.0, "millivolt"),
        temperature: temperature_t = units.Q_(37.0, "degree_Celsius"),
        formulation: Formulation | None = "single-cable",
    ) -> None:
        """Create a Schild and Bhatt 1997 DRG C-fiber axon.

        Parameters
        ----------
        diameter, length, compartments, x:
            Geometry values with explicit length units.
        Ra:
            Axial resistivity in ohm * centimeter.
        Cm:
            Membrane capacitance density in uF/cm^2.
        v_init:
            Initial membrane potential in millivolts.
        temperature:
            Model temperature in degrees Celsius.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        temperature = _quantity_degC(temperature, name="temperature")
        v_init = _quantity_mV(v_init, name="v_init")
        diameter_um = units.require_length_um(diameter, name="diameter")
        super().__init__(
            membrane=membranes.Schild97(
                diameter_um=diameter_um,
                temp_c=temperature,
                vinit_mV=v_init,
            ),
            length=length,
            diameter=diameter,
            compartments=compartments,
            x=x,
            Ra=Ra,
            Cm=Cm,
            v_init=v_init,
            temperature=temperature,
            formulation=formulation,
        )


__all__ = [
    "Unmyelinated",
    "HodgkinHuxley",
    "RattayAberham",
    "Sundt",
    "Tigerholm",
    "Schild94",
    "Schild97",
]
