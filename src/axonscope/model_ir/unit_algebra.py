"""Unit normalization and algebra for compiled membrane equations."""

from __future__ import annotations

from typing import Any

from axonscope.utils.units import (
    CONCENTRATION_MM,
    CONCENTRATION_PER_CURRENT_DENSITY_TIME,
    CONCENTRATION_RATE_MM_PER_MS,
    CONDUCTANCE_DENSITY_MS_CM2,
    CURRENT_DENSITY_UA_CM2,
    DIMENSIONLESS,
    RATE_PER_MS,
    RATE_PER_MS_PER_MM,
    RATE_PER_MS_PER_MV,
    RESISTANCE_AREA_OHM_CM2,
    TEMPERATURE_DEGC,
    TIME_MS,
    VOLTAGE_MV,
)


IR_TO_PINT_UNIT = {
    DIMENSIONLESS: "",
    "": "",
    "dimensionless": "",
    VOLTAGE_MV: "millivolt",
    TIME_MS: "millisecond",
    CONDUCTANCE_DENSITY_MS_CM2: "millisiemens / centimeter ** 2",
    CURRENT_DENSITY_UA_CM2: "microampere / centimeter ** 2",
    RESISTANCE_AREA_OHM_CM2: "ohm * centimeter ** 2",
    TEMPERATURE_DEGC: "degree_Celsius",
    RATE_PER_MS: "1 / millisecond",
    RATE_PER_MS_PER_MV: "1 / millisecond / millivolt",
    RATE_PER_MS_PER_MM: "1 / millisecond / millimolar",
    CONCENTRATION_MM: "millimolar",
    CONCENTRATION_PER_CURRENT_DENSITY_TIME:
        "millimolar / (microampere / centimeter ** 2) / millisecond",
    "mS": "millisiemens",
    "uA": "microampere",
    "ohm": "ohm",
    "cm2": "centimeter ** 2",
    "ms*mV": "millisecond * millivolt",
    "ms*mM": "millisecond * millimolar",
}

PUBLIC_UNIT_TO_IR = {
    "": DIMENSIONLESS,
    "1": DIMENSIONLESS,
    "dimensionless": DIMENSIONLESS,
    VOLTAGE_MV: VOLTAGE_MV,
    "millivolt": VOLTAGE_MV,
    TIME_MS: TIME_MS,
    "millisecond": TIME_MS,
    CONDUCTANCE_DENSITY_MS_CM2: CONDUCTANCE_DENSITY_MS_CM2,
    "mS / cm ** 2": CONDUCTANCE_DENSITY_MS_CM2,
    "millisiemens / centimeter ** 2": CONDUCTANCE_DENSITY_MS_CM2,
    CURRENT_DENSITY_UA_CM2: CURRENT_DENSITY_UA_CM2,
    "uA / cm ** 2": CURRENT_DENSITY_UA_CM2,
    "microampere / centimeter ** 2": CURRENT_DENSITY_UA_CM2,
    RESISTANCE_AREA_OHM_CM2: RESISTANCE_AREA_OHM_CM2,
    "ohm * centimeter ** 2": RESISTANCE_AREA_OHM_CM2,
    "ohm*centimeter**2": RESISTANCE_AREA_OHM_CM2,
    "ohm_cm2": RESISTANCE_AREA_OHM_CM2,
    TEMPERATURE_DEGC: TEMPERATURE_DEGC,
    "degree_Celsius": TEMPERATURE_DEGC,
    RATE_PER_MS: RATE_PER_MS,
    "1 / ms": RATE_PER_MS,
    "1 / millisecond": RATE_PER_MS,
    RATE_PER_MS_PER_MV: RATE_PER_MS_PER_MV,
    "1 / (ms * mV)": RATE_PER_MS_PER_MV,
    "1 / millisecond / millivolt": RATE_PER_MS_PER_MV,
    "1 / millivolt / millisecond": RATE_PER_MS_PER_MV,
    RATE_PER_MS_PER_MM: RATE_PER_MS_PER_MM,
    "1 / (ms * mM)": RATE_PER_MS_PER_MM,
    "1 / millisecond / millimolar": RATE_PER_MS_PER_MM,
    "1 / millimolar / millisecond": RATE_PER_MS_PER_MM,
    CONCENTRATION_MM: CONCENTRATION_MM,
    "millimolar": CONCENTRATION_MM,
    CONCENTRATION_PER_CURRENT_DENSITY_TIME: CONCENTRATION_PER_CURRENT_DENSITY_TIME,
    "millimolar / (microampere / centimeter ** 2) / millisecond":
        CONCENTRATION_PER_CURRENT_DENSITY_TIME,
    "mS": "mS",
    "millisiemens": "mS",
    "uA": "uA",
    "microampere": "uA",
    "ohm": "ohm",
    "cm2": "cm2",
    "centimeter ** 2": "cm2",
    "ms*mV": "ms*mV",
    "ms * mV": "ms*mV",
    "millisecond * millivolt": "ms*mV",
    "ms*mM": "ms*mM",
    "ms * mM": "ms*mM",
    "millisecond * millimolar": "ms*mM",
}


def normalize_unit(unit: Any) -> str:
    """Return AxonScope's canonical compact spelling for a public unit."""

    candidates = [str(unit)]
    try:
        from axonscope.utils.units import unit_label

        label = unit_label(unit)
    except Exception:
        label = None
    if label is not None:
        candidates.append(label)
    candidates.extend(candidate.replace(" ", "") for candidate in tuple(candidates))
    for candidate in candidates:
        if candidate in PUBLIC_UNIT_TO_IR:
            return PUBLIC_UNIT_TO_IR[candidate]
    raise ValueError(f"Unsupported membrane equation unit {unit!r}.")


def quantity_literal(value: Any) -> tuple[int | float | bool, str] | None:
    """Return `(magnitude, canonical_unit)` if `value` is a unit quantity."""

    if not (hasattr(value, "to") and hasattr(value, "magnitude")):
        return None
    unit = normalize_unit(value)
    from axonscope.utils.units import to_scalar

    return float(to_scalar(value, IR_TO_PINT_UNIT.get(unit, unit))), unit


def is_dimensionless(unit: str) -> bool:
    return unit in {"", DIMENSIONLESS, "dimensionless"}


def product_unit(left: str, right: str) -> str:
    if is_dimensionless(left):
        return right
    if is_dimensionless(right):
        return left
    if {
        left,
        right,
    } == {CONDUCTANCE_DENSITY_MS_CM2, VOLTAGE_MV}:
        return CURRENT_DENSITY_UA_CM2
    if {left, right} == {RATE_PER_MS_PER_MV, VOLTAGE_MV}:
        return RATE_PER_MS
    if {left, right} == {RATE_PER_MS_PER_MM, CONCENTRATION_MM}:
        return RATE_PER_MS
    if {left, right} == {RATE_PER_MS, TIME_MS}:
        return DIMENSIONLESS
    if {left, right} == {"ohm", "cm2"}:
        return RESISTANCE_AREA_OHM_CM2
    if {left, right} == {TIME_MS, VOLTAGE_MV}:
        return "ms*mV"
    if {left, right} == {TIME_MS, CONCENTRATION_MM}:
        return "ms*mM"
    if {left, right} == {CONCENTRATION_MM, RATE_PER_MS}:
        return CONCENTRATION_RATE_MM_PER_MS
    if {left, right} == {
        CURRENT_DENSITY_UA_CM2,
        CONCENTRATION_PER_CURRENT_DENSITY_TIME,
    }:
        return CONCENTRATION_RATE_MM_PER_MS
    if {left, right} == {CONCENTRATION_RATE_MM_PER_MS, TIME_MS}:
        return CONCENTRATION_MM
    if left == RATE_PER_MS and is_dimensionless(right):
        return RATE_PER_MS
    if right == RATE_PER_MS and is_dimensionless(left):
        return RATE_PER_MS
    return f"({left})*({right})"


def quotient_unit(left: str, right: str) -> str:
    if is_dimensionless(right):
        return left
    if left == right:
        return DIMENSIONLESS
    if is_dimensionless(left) and right == TIME_MS:
        return RATE_PER_MS
    if is_dimensionless(left) and right == RATE_PER_MS:
        return TIME_MS
    if is_dimensionless(left) and right == RESISTANCE_AREA_OHM_CM2:
        return CONDUCTANCE_DENSITY_MS_CM2
    if left == "mS" and right == "cm2":
        return CONDUCTANCE_DENSITY_MS_CM2
    if left == "uA" and right == "cm2":
        return CURRENT_DENSITY_UA_CM2
    if is_dimensionless(left) and right == "ms*mV":
        return RATE_PER_MS_PER_MV
    if is_dimensionless(left) and right == "ms*mM":
        return RATE_PER_MS_PER_MM
    if left == CONCENTRATION_MM and right == TIME_MS:
        return CONCENTRATION_RATE_MM_PER_MS
    if is_dimensionless(left):
        return f"1/({right})"
    return f"({left})/({right})"
