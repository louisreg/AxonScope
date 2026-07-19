"""Unmyelinated axon family and membrane templates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from axonscope.axons.axon import Axon
from axonscope.axons.diameters import round_axon_diameter_um
from axonscope.axons.formulation import CableFormulation
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
from axonscope.membranes.model import MembraneModel, Model
from .. import membranes


_DEFAULT_Ra = units.Q_(100.0, "ohm * centimeter")
_DEFAULT_Cm = units.Q_(1.0, "microfarad / centimeter ** 2")
_DEFAULT_HH_Ra = units.Q_(200.0, "ohm * centimeter")
_DEFAULT_SCHILD_Cm = units.Q_(1.326291192, "microfarad / centimeter ** 2")
_UNSET = object()


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
    diameter_value = round_axon_diameter_um(
        units.require_length_um(diameter, name="diameter")
    )
    x_value = None if x is None else units.require_length_array_um(x, name="x", dtype=np.float32)
    return _GeometrySpec(
        length=None if length_value is None else _quantity_um(length_value),
        diameter=_quantity_um(diameter_value),
        compartments=compartment_count,
        x=None if x_value is None else _quantity_um(x_value),
    )


def _passive_leak(g_pas: Any, e_pas: Any) -> Model:
    conductance = units.to_S_per_cm2(g_pas)
    if conductance <= 0.0:
        raise ValueError("g_pas must be strictly positive when include_passive_leak=True.")
    return _cached_builtin_model(
        membranes.Passive,
        Rm=1.0 / conductance,
        EL=e_pas,
    )


def _provided_model_params(**params: Any) -> dict[str, Any]:
    return {name: value for name, value in params.items() if value is not _UNSET}


def _cached_builtin_model(model_type: type[Model], **params: Any) -> Model:
    items = tuple(sorted(params.items()))
    try:
        hash((model_type, items))
    except TypeError:
        return model_type(**params)
    return _cached_builtin_model_from_items(model_type, items)


@lru_cache(maxsize=512)
def _cached_builtin_model_from_items(
    model_type: type[Model],
    items: tuple[tuple[str, Any], ...],
) -> Model:
    return model_type(**dict(items))


def _single_section_model(
    *,
    membrane: Model | MembraneModel,
    length: length_t | None,
    diameter: length_t,
    compartments: int | None,
    x: length_t | None,
    Ra: axial_resistivity_t | None,
    Cm: capacitance_density_t | None,
) -> Layout:
    if x is None and isinstance(membrane, MembraneModel):
        assert length is not None
        assert compartments is not None
        return _cached_single_uniform_layout(
            membrane,
            units.require_length_um(length, name="length"),
            round_axon_diameter_um(
                units.require_length_um(diameter, name="diameter")
            ),
            normalize_positive_int(compartments, name="compartments"),
            units.require_axial_resistivity_ohm_cm(
                _DEFAULT_Ra if Ra is None else Ra,
                name="Ra",
            ),
            units.require_capacitance_density_uF_per_cm2(
                _DEFAULT_Cm if Cm is None else Cm,
                name="Cm",
            ),
        )

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


@lru_cache(maxsize=512)
def _cached_single_uniform_layout(
    membrane: MembraneModel,
    length_um: float,
    diameter_um: float,
    compartments: int,
    Ra_ohm_cm: float,
    Cm_uF_cm2: float,
) -> Layout:
    section = Section(
        "axon",
        membrane=membrane,
        diameter=units.Q_(diameter_um, "micrometer"),
        Ra=units.Q_(Ra_ohm_cm, "ohm * centimeter"),
        Cm=units.Q_(Cm_uF_cm2, "microfarad / centimeter ** 2"),
        tags=("unmyelinated",),
    )
    return Layout.single_uniform(
        section,
        length=units.Q_(length_um, "micrometer"),
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
        membrane: Model | MembraneModel | None = None,
        formulation: CableFormulation | None = CableFormulation.SINGLE_CABLE,
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

        nominal_diameter = None
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
            nominal_diameter = geometry.diameter
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
            diameter=nominal_diameter,
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
        gnabar: Any = _UNSET,
        gkbar: Any = _UNSET,
        gl: Any = _UNSET,
        el: Any = _UNSET,
        ena: Any = _UNSET,
        ek: Any = _UNSET,
        celsius: temperature_t = units.Q_(32.0, "degree_Celsius"),
        include_passive_leak: bool = True,
        g_pas: Any = 0.001,
        e_pas: Any = -70.0,
        formulation: CableFormulation | None = CableFormulation.SINGLE_CABLE,
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
        gnabar, gkbar, gl, el, ena, ek:
            Optional Hodgkin-Huxley model overrides forwarded to
            `membranes/models/hodgkin_huxley.py`.
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
        hh_model = _cached_builtin_model(
            membranes.HodgkinHuxley,
            **_provided_model_params(
                gnabar=gnabar,
                gkbar=gkbar,
                gl=gl,
                el=el,
                ena=ena,
                ek=ek,
            ),
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
        gnabar: Any = _UNSET,
        gkbar: Any = _UNSET,
        gl: Any = _UNSET,
        el: Any = _UNSET,
        ena: Any = _UNSET,
        ek: Any = _UNSET,
        celsius: temperature_t = units.Q_(37.0, "degree_Celsius"),
        include_passive_leak: bool = True,
        g_pas: Any = 0.001,
        e_pas: Any = -70.0,
        formulation: CableFormulation | None = CableFormulation.SINGLE_CABLE,
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
        gnabar, gkbar, gl, el, ena, ek:
            Optional Rattay-Aberham model overrides forwarded to
            `membranes/models/rattay_aberham.py`.
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
        rattay = _cached_builtin_model(
            membranes.RattayAberham,
            **_provided_model_params(
                gnabar=gnabar,
                gkbar=gkbar,
                gl=gl,
                el=el,
                ena=ena,
                ek=ek,
            ),
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
        gnabar: Any = _UNSET,
        gkdrbar: Any = _UNSET,
        ena: Any = _UNSET,
        ek: Any = _UNSET,
        Rm: Any = _UNSET,
        El: Any = _UNSET,
        formulation: CableFormulation | None = CableFormulation.SINGLE_CABLE,
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
        gnabar, gkdrbar, ena, ek, Rm, El:
            Optional Sundt model overrides forwarded to
            `membranes/models/sundt.py`.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        celsius = _quantity_degC(celsius, name="celsius")
        super().__init__(
            membrane=_cached_builtin_model(
                membranes.Sundt,
                **_provided_model_params(
                    gnabar=gnabar,
                    gkdrbar=gkdrbar,
                    ena=ena,
                    ek=ek,
                    Rm=Rm,
                    El=El,
                ),
                celsius=celsius,
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
        ena: Any = _UNSET,
        ek: Any = _UNSET,
        gbar_nav17: Any = _UNSET,
        gbar_nav18: Any = _UNSET,
        gbar_nav19: Any = _UNSET,
        gbar_ks: Any = _UNSET,
        gbar_kf: Any = _UNSET,
        gbar_kdr: Any = _UNSET,
        gbar_h: Any = _UNSET,
        gbar_kna: Any = _UNSET,
        nai_fixed: Any = _UNSET,
        pump_smalla: Any = _UNSET,
        pump_ko: Any = _UNSET,
        formulation: CableFormulation | None = CableFormulation.SINGLE_CABLE,
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
        ena, ek, gbar_nav17, gbar_nav18, gbar_nav19, gbar_ks, gbar_kf, gbar_kdr,
        gbar_h, gbar_kna, nai_fixed, pump_smalla, pump_ko:
            Optional Tigerholm model overrides forwarded to
            `membranes/models/tigerholm.py`.
        formulation:
            Cable formulation, normally `"single-cable"`.
        """

        celsius = _quantity_degC(celsius, name="celsius")
        super().__init__(
            membrane=_cached_builtin_model(
                membranes.Tigerholm,
                **_provided_model_params(
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
                diameter_um=diameter,
                celsius=celsius,
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
        formulation: CableFormulation | None = CableFormulation.SINGLE_CABLE,
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
        super().__init__(
            membrane=_cached_builtin_model(
                membranes.Schild94,
                diameter_um=diameter,
                celsius=temperature,
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
        formulation: CableFormulation | None = CableFormulation.SINGLE_CABLE,
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
        super().__init__(
            membrane=_cached_builtin_model(
                membranes.Schild97,
                diameter_um=diameter,
                celsius=temperature,
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
