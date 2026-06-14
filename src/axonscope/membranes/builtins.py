"""Built-in public membrane model templates."""

from __future__ import annotations

from typing import Any

from axonscope.utils import units
from axonscope.membranes.model import MembraneModel


def _conductance(value: Any) -> float:
    return units.to_S_per_cm2(value)


def _voltage(value: Any) -> float:
    return units.to_mV(value)


def _temperature(value: Any) -> float:
    return units.to_degC(value)


def Passive(*, Rm: Any = 1e4, EL: Any = -70.0) -> MembraneModel:
    """Passive leak membrane model description.

    Plain `Rm` values are interpreted as ohm * cm^2 and `EL` as mV.
    Pint quantities are converted to those canonical units.
    """

    return MembraneModel(
        "passive",
        {
            "Rm": units.to_ohm_cm2(Rm),
            "EL": _voltage(EL),
        },
    )


def HodgkinHuxley(
    *,
    gnabar: Any = 0.12,
    gkbar: Any = 0.036,
    gl: Any = 0.0003,
    el: Any = -54.3,
    ena: Any = 50.0,
    ek: Any = -77.0,
    celsius: Any = 6.3,
) -> MembraneModel:
    """Hodgkin-Huxley membrane model description.

    Conductances are stored in S/cm^2, voltages in mV, and temperature in
    degrees Celsius.
    """

    return MembraneModel(
        "hodgkin_huxley",
        {
            "gnabar": _conductance(gnabar),
            "gkbar": _conductance(gkbar),
            "gl": _conductance(gl),
            "el": _voltage(el),
            "ena": _voltage(ena),
            "ek": _voltage(ek),
            "celsius": _temperature(celsius),
        },
    )


def RattayAberham(
    *,
    gnabar: Any = 0.12,
    gkbar: Any = 0.036,
    gl: Any = 0.0003,
    el: Any = -59.4,
    ena: Any = 50.0,
    ek: Any = -82.0,
    celsius: Any = 37.0,
) -> MembraneModel:
    """Rattay-Aberham membrane model description."""

    return MembraneModel(
        "rattay_aberham",
        {
            "gnabar": _conductance(gnabar),
            "gkbar": _conductance(gkbar),
            "gl": _conductance(gl),
            "el": _voltage(el),
            "ena": _voltage(ena),
            "ek": _voltage(ek),
            "celsius": _temperature(celsius),
        },
    )


def Sundt(
    *,
    celsius: Any = 37.0,
    gnabar: Any = 0.04,
    gkdrbar: Any = 0.04,
    ena: Any = 45.0,
    ek: Any = -90.0,
    Rm: Any = 10000.0,
    El: Any = -70.0,
) -> MembraneModel:
    """Sundt-style composite membrane model description."""

    return MembraneModel(
        "sundt",
        {
            "celsius": _temperature(celsius),
            "gnabar": _conductance(gnabar),
            "gkdrbar": _conductance(gkdrbar),
            "ena": _voltage(ena),
            "ek": _voltage(ek),
            "Rm": units.to_ohm_cm2(Rm),
            "El": _voltage(El),
        },
    )


def Tigerholm(
    *,
    diameter: Any,
    celsius: Any = 37.0,
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
) -> MembraneModel:
    """Tigerholm C-fiber membrane model description."""

    return MembraneModel(
        "tigerholm",
        {
            "diameter_um": units.require_length_um(diameter, name="diameter"),
            "celsius": _temperature(celsius),
            "ena": _voltage(ena),
            "ek": _voltage(ek),
            "gbar_nav17": _conductance(gbar_nav17),
            "gbar_nav18": _conductance(gbar_nav18),
            "gbar_nav19": _conductance(gbar_nav19),
            "gbar_ks": _conductance(gbar_ks),
            "gbar_kf": _conductance(gbar_kf),
            "gbar_kdr": _conductance(gbar_kdr),
            "gbar_h": _conductance(gbar_h),
            "gbar_kna": _conductance(gbar_kna),
            "nai_fixed": units.to_mM(nai_fixed),
            "pump_smalla": units.to_mA_per_cm2(pump_smalla),
            "pump_ko": units.to_mM(pump_ko),
        },
    )


def Schild94(
    *,
    diameter: Any,
    temp_c: Any = 37.0,
    vinit_mV: Any = -48.0,
) -> MembraneModel:
    """Schild 1994 membrane model description."""

    return MembraneModel(
        "schild94",
        {
            "diameter_um": units.require_length_um(diameter, name="diameter"),
            "temp_c": _temperature(temp_c),
            "vinit_mV": _voltage(vinit_mV),
        },
    )


def Schild97(
    *,
    diameter: Any,
    temp_c: Any = 37.0,
    vinit_mV: Any = -48.0,
) -> MembraneModel:
    """Schild 1997 membrane model description."""

    return MembraneModel(
        "schild97",
        {
            "diameter_um": units.require_length_um(diameter, name="diameter"),
            "temp_c": _temperature(temp_c),
            "vinit_mV": _voltage(vinit_mV),
        },
    )


def AxNode(
    *,
    gnapbar_S_cm2: Any = 0.01,
    gnabar_S_cm2: Any = 3.0,
    gkbar_S_cm2: Any = 0.08,
    gl_S_cm2: Any = 0.007,
    ena_mV: Any = 50.0,
    ek_mV: Any = -90.0,
    el_mV: Any = -90.0,
    celsius: Any = 37.0,
) -> MembraneModel:
    """MRG-like active node membrane model description."""

    return MembraneModel(
        "axnode",
        {
            "gnapbar_S_cm2": _conductance(gnapbar_S_cm2),
            "gnabar_S_cm2": _conductance(gnabar_S_cm2),
            "gkbar_S_cm2": _conductance(gkbar_S_cm2),
            "gl_S_cm2": _conductance(gl_S_cm2),
            "ena_mV": _voltage(ena_mV),
            "ek_mV": _voltage(ek_mV),
            "el_mV": _voltage(el_mV),
            "celsius": _temperature(celsius),
        },
    )


__all__ = [
    "AxNode",
    "HodgkinHuxley",
    "Passive",
    "RattayAberham",
    "Schild94",
    "Schild97",
    "Sundt",
    "Tigerholm",
]
