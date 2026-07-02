"""Unit helpers for the public AxonScope API.

AxonScope stores canonical plain floats internally, but user-facing interfaces
can choose one of two boundary policies:

- `to_*` helpers accept plain numbers in the canonical unit and convert
  Pint-like quantities.
- `require_*` helpers reject plain numbers and require units with the expected
  physical dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, TypeAlias

import numpy as np


class UnitSupportError(ImportError):
    """Raised when Pint-specific helpers are used without Pint installed."""


@dataclass(frozen=True)
class UnitSpec:
    """Runtime unit contract for a public argument."""

    dimension: str
    unit: str
    example: str


@dataclass(frozen=True)
class UnitAlias:
    """Tiny unit token used by plain-Python membrane source files."""

    label: str

    def __mul__(self, other: Any) -> "UnitExpression":
        if isinstance(other, int | float):
            return UnitExpression(float(other), self.label)
        if isinstance(other, UnitAlias):
            return UnitExpression(1.0, f"{self.label}*{other.label}")
        if isinstance(other, UnitExpression):
            return UnitExpression(other.magnitude, f"{self.label}*{other.unit}")
        return NotImplemented

    def __rmul__(self, other: Any) -> "UnitExpression":
        if isinstance(other, int | float):
            return UnitExpression(float(other), self.label)
        if isinstance(other, UnitExpression):
            return UnitExpression(other.magnitude, f"{other.unit}*{self.label}")
        return NotImplemented

    def __truediv__(self, other: Any) -> "UnitExpression":
        if isinstance(other, UnitAlias):
            return UnitExpression(1.0, f"{self.label}/{other.label}")
        if isinstance(other, UnitExpression):
            return UnitExpression(1.0 / other.magnitude, f"{self.label}/({other.unit})")
        if isinstance(other, int | float):
            return UnitExpression(1.0 / float(other), self.label)
        return NotImplemented

    def __rtruediv__(self, other: Any) -> "UnitExpression":
        if isinstance(other, int | float):
            return UnitExpression(float(other), f"1/{self.label}")
        if isinstance(other, UnitExpression):
            return UnitExpression(other.magnitude, f"{other.unit}/{self.label}")
        return NotImplemented

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class UnitExpression:
    """Runtime value produced by source-unit arithmetic in model files."""

    magnitude: float
    unit: str

    def __mul__(self, other: Any) -> "UnitExpression":
        if isinstance(other, int | float):
            return UnitExpression(self.magnitude * float(other), self.unit)
        if isinstance(other, UnitAlias):
            return UnitExpression(self.magnitude, f"{self.unit}*{other.label}")
        if isinstance(other, UnitExpression):
            return UnitExpression(self.magnitude * other.magnitude, f"{self.unit}*{other.unit}")
        return NotImplemented

    def __rmul__(self, other: Any) -> "UnitExpression":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "UnitExpression":
        if isinstance(other, int | float):
            return UnitExpression(self.magnitude / float(other), self.unit)
        if isinstance(other, UnitAlias):
            return UnitExpression(self.magnitude, f"{self.unit}/{other.label}")
        if isinstance(other, UnitExpression):
            return UnitExpression(self.magnitude / other.magnitude, f"{self.unit}/({other.unit})")
        return NotImplemented

    def __rtruediv__(self, other: Any) -> "UnitExpression":
        if isinstance(other, int | float):
            return UnitExpression(float(other) / self.magnitude, f"1/({self.unit})")
        return NotImplemented


class QuantityLike(Protocol):
    """Minimal unit-bearing scalar/array protocol accepted by AxonScope."""

    magnitude: Any

    def to(self, unit: str) -> "QuantityLike":
        """Return this quantity converted to `unit`."""


# Canonical compact labels used by membrane equations and compiler validation.
DIMENSIONLESS = "1"
VOLTAGE_MV = "mV"
TIME_MS = "ms"
CONDUCTANCE_DENSITY_MS_CM2 = "mS/cm2"
CURRENT_DENSITY_UA_CM2 = "uA/cm2"
RESISTANCE_AREA_OHM_CM2 = "ohm*cm2"
TEMPERATURE_DEGC = "degC"
RATE_PER_MS = "1/ms"
RATE_PER_MS_PER_MV = "1/(ms*mV)"
RATE_PER_MS_PER_MM = "1/(ms*mM)"
CONCENTRATION_MM = "mM"
CONCENTRATION_RATE_MM_PER_MS = "mM/ms"
CONCENTRATION_PER_CURRENT_DENSITY_TIME = "mM/(uA/cm2*ms)"

# Short unit aliases used by plain-Python membrane source equations.
dimensionless = UnitAlias(DIMENSIONLESS)
mV = UnitAlias(VOLTAGE_MV)
ms = UnitAlias(TIME_MS)
mS = UnitAlias("mS")
uA = UnitAlias("uA")
ohm = UnitAlias("ohm")
cm2 = UnitAlias("cm2")
mS_per_cm2 = UnitAlias(CONDUCTANCE_DENSITY_MS_CM2)
uA_per_cm2 = UnitAlias(CURRENT_DENSITY_UA_CM2)
ohm_cm2 = UnitAlias(RESISTANCE_AREA_OHM_CM2)
degC = UnitAlias(TEMPERATURE_DEGC)
per_ms = UnitAlias(RATE_PER_MS)
per_ms_per_mV = UnitAlias(RATE_PER_MS_PER_MV)
per_ms_per_mM = UnitAlias(RATE_PER_MS_PER_MM)
mM = UnitAlias(CONCENTRATION_MM)
mM_per_uA_cm2_ms = UnitAlias(CONCENTRATION_PER_CURRENT_DENSITY_TIME)
gate = UnitAlias(DIMENSIONLESS)
um = UnitAlias("micrometer")


LENGTH = UnitSpec("length", "micrometer", "100.0 * axs.um or 1.0 * axs.mm")
TIME = UnitSpec("time", "millisecond", "1.0 * axs.ms")
VOLTAGE = UnitSpec("voltage", "millivolt", "-70.0 * axs.mV")
CURRENT = UnitSpec("current", "ampere", "1.0 * axs.uA")
TEMPERATURE = UnitSpec("temperature", "degree_Celsius", "37.0 * axs.degC")
CAPACITANCE = UnitSpec("capacitance", "farad", "1.0 * axs.uF")
CAPACITANCE_DENSITY = UnitSpec(
    "capacitance density",
    "microfarad / centimeter ** 2",
    "1.0 * axs.uF_per_cm2",
)
CAPACITANCE_PER_LENGTH = UnitSpec(
    "capacitance per length",
    "farad / meter",
    "1.0 * axs.uF / axs.m",
)
CONDUCTANCE_DENSITY = UnitSpec(
    "conductance density",
    "siemens / centimeter ** 2",
    "1e-3 * axs.S_per_cm2",
)
AXIAL_RESISTIVITY = UnitSpec(
    "axial resistivity",
    "ohm * centimeter",
    "100.0 * axs.ohm_cm",
)
AXOPLASMIC_RESISTIVITY = UnitSpec(
    "axoplasmic resistivity",
    "ohm * micrometer",
    "0.7e6 * axs.ohm_um",
)
PERIAXONAL_RESISTANCE = UnitSpec(
    "resistance per length",
    "megaohm / centimeter",
    "1e8 * axs.MOhm_per_cm",
)
CONDUCTIVITY = UnitSpec(
    "conductivity",
    "siemens / meter",
    "0.3 * axs.S_per_m",
)


UnitScalar: TypeAlias = QuantityLike

# These aliases document the expected physical dimension at public boundaries.
# Runtime validation is performed by the `require_*` helpers below; keeping the
# aliases as plain `QuantityLike` avoids Pylance/Pyright treating UnitSpec
# metadata objects as invalid type expressions.
length_t: TypeAlias = QuantityLike
meter_type: TypeAlias = length_t
time_t: TypeAlias = QuantityLike
voltage_t: TypeAlias = QuantityLike
current_t: TypeAlias = QuantityLike
temperature_t: TypeAlias = QuantityLike
capacitance_t: TypeAlias = QuantityLike
farad_type: TypeAlias = capacitance_t
capacitance_density_t: TypeAlias = QuantityLike
farad_per_area_type: TypeAlias = capacitance_density_t
capacitance_per_length_t: TypeAlias = QuantityLike
farad_per_meter_t: TypeAlias = capacitance_per_length_t
conductance_density_t: TypeAlias = QuantityLike
siemens_per_area_type: TypeAlias = conductance_density_t
axial_resistivity_t: TypeAlias = QuantityLike
ohm_meter_type: TypeAlias = axial_resistivity_t
axoplasmic_resistivity_t: TypeAlias = QuantityLike
periaxonal_resistance_t: TypeAlias = QuantityLike
resistance_per_length_type: TypeAlias = periaxonal_resistance_t
conductivity_t: TypeAlias = QuantityLike
siemens_per_meter_type: TypeAlias = conductivity_t


def _quantity_like(value: Any) -> bool:
    return hasattr(value, "to") and hasattr(value, "magnitude")


def is_quantity_like(value: Any) -> bool:
    """Return whether `value` looks like a Pint-compatible quantity."""

    return _quantity_like(value)


def quantity_unit(value: Any) -> str | None:
    """Return the unit string carried by a Pint-like quantity, if present."""

    if _quantity_like(value) and hasattr(value, "units"):
        return str(value.units)
    return None


def unit_label(unit: Any | None) -> str | None:
    """Return a normalized Pint unit label for strings, units, or quantities."""

    if unit is None:
        return None
    label = quantity_unit(unit) if _quantity_like(unit) else str(unit)
    if label is None:
        return None
    try:
        return str(Q_(1.0, label).units)
    except Exception:
        return label


def short_unit_label(unit: Any | None) -> str | None:
    """Return a compact display label for a unit when Pint can format it."""

    label = unit_label(unit)
    if label is None:
        return None
    try:
        return f"{Q_(1.0, label).units:~P}"
    except Exception:
        return label


def _convert_quantity_like(value: Any, unit: str) -> Any:
    if _quantity_like(value):
        return value.to(unit).magnitude
    if isinstance(value, tuple):
        return tuple(_convert_quantity_like(item, unit) for item in value)
    if isinstance(value, list):
        return [_convert_quantity_like(item, unit) for item in value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_convert_quantity_like(item, unit) for item in value.tolist()]
    return value


def to_scalar(value: Any, unit: str) -> float:
    """Return `value` as a float expressed in `unit`.

    Plain numbers are assumed to already use `unit`; Pint quantities are
    converted. This function intentionally returns Python floats so descriptive
    models remain backend-independent.
    """

    converted = _convert_quantity_like(value, unit)
    arr = np.asarray(converted, dtype=float)
    if arr.ndim != 0:
        raise ValueError(f"Expected a scalar value convertible to {unit!r}.")
    return float(arr)


def require_quantity(value: Any, spec: UnitSpec, *, name: str) -> float:
    """Return a unit-bearing scalar converted according to `spec`.

    Unlike `to_scalar`, this helper rejects plain numbers and reports the
    expected physical dimension. Use it for public interfaces where AxonScope
    intentionally requires explicit units.
    """

    if not _quantity_like(value):
        raise TypeError(
            f"{name} must include units compatible with {spec.dimension} "
            f"(for example {spec.example})."
        )
    try:
        return to_scalar(value, spec.unit)
    except Exception as exc:
        raise TypeError(
            f"{name} must have units compatible with {spec.dimension}; "
            f"expected units convertible to {spec.unit!r} "
            f"(for example {spec.example})."
        ) from exc


def require_quantity_array(value: Any, spec: UnitSpec, *, name: str, dtype: Any = float) -> np.ndarray:
    """Return a unit-bearing array converted according to `spec`."""

    if not _quantity_like(value):
        raise TypeError(
            f"{name} must include units compatible with {spec.dimension} "
            f"(for example {spec.example})."
        )
    try:
        return to_array(value, spec.unit, dtype=dtype)
    except Exception as exc:
        raise TypeError(
            f"{name} must have units compatible with {spec.dimension}; "
            f"expected units convertible to {spec.unit!r} "
            f"(for example {spec.example})."
        ) from exc


def require_scalar_quantity(value: Any, unit: str, *, name: str) -> float:
    """Return a unit-bearing scalar converted to `unit`.

    This generic compatibility helper rejects plain numbers but does not carry
    a domain-specific dimension label. Prefer the typed `require_*` helpers for
    new public APIs.
    """

    return require_quantity(
        value,
        UnitSpec(
            dimension=f"units convertible to {unit!r}",
            unit=unit,
            example=f"`value * units.ureg({unit!r})`",
        ),
        name=name,
    )


def require_length_um(value: length_t, *, name: str) -> float:
    """Return a unit-bearing length as micrometers."""

    return require_quantity(value, LENGTH, name=name)


def require_length_array_um(value: length_t, *, name: str, dtype: Any = np.float32) -> np.ndarray:
    """Return a unit-bearing length array as micrometers."""

    return require_quantity_array(value, LENGTH, name=name, dtype=dtype)


def require_time_ms(value: time_t, *, name: str) -> float:
    """Return a unit-bearing time as milliseconds."""

    return require_quantity(value, TIME, name=name)


def require_time_array_ms(value: time_t, *, name: str, dtype: Any = float) -> np.ndarray:
    """Return a unit-bearing time array as milliseconds."""

    return require_quantity_array(value, TIME, name=name, dtype=dtype)


def require_voltage_mV(value: voltage_t, *, name: str) -> float:
    """Return a unit-bearing voltage as millivolts."""

    return require_quantity(value, VOLTAGE, name=name)


def require_current_A(value: current_t, *, name: str) -> float:
    """Return a unit-bearing current as amperes."""

    return require_quantity(value, CURRENT, name=name)


def require_current_uA(value: current_t, *, name: str) -> float:
    """Return a unit-bearing current as microamperes."""

    return require_current_A(value, name=name) * 1e6


def require_current_array_uA(value: current_t, *, name: str, dtype: Any = float) -> np.ndarray:
    """Return a unit-bearing current array as microamperes."""

    return require_quantity_array(value, CURRENT, name=name, dtype=dtype) * 1e6


def require_temperature_degC(value: temperature_t, *, name: str) -> float:
    """Return a unit-bearing temperature as degrees Celsius."""

    return require_quantity(value, TEMPERATURE, name=name)


def require_capacitance_density_uF_per_cm2(
    value: capacitance_density_t,
    *,
    name: str,
) -> float:
    """Return a unit-bearing capacitance density as uF/cm^2."""

    return require_quantity(value, CAPACITANCE_DENSITY, name=name)


def require_conductance_density_S_per_cm2(
    value: conductance_density_t,
    *,
    name: str,
) -> float:
    """Return a unit-bearing conductance density as S/cm^2."""

    return require_quantity(value, CONDUCTANCE_DENSITY, name=name)


def require_axial_resistivity_ohm_cm(
    value: axial_resistivity_t,
    *,
    name: str,
) -> float:
    """Return a unit-bearing axial resistivity as ohm * cm."""

    return require_quantity(value, AXIAL_RESISTIVITY, name=name)


def require_axoplasmic_resistivity_ohm_um(
    value: axoplasmic_resistivity_t,
    *,
    name: str,
) -> float:
    """Return a unit-bearing axoplasmic resistivity as ohm * um."""

    return require_quantity(value, AXOPLASMIC_RESISTIVITY, name=name)


def require_periaxonal_resistance_MOhm_per_cm(
    value: periaxonal_resistance_t,
    *,
    name: str,
) -> float:
    """Return a unit-bearing periaxonal resistance per length as MOhm/cm."""

    return require_quantity(value, PERIAXONAL_RESISTANCE, name=name)


def require_conductivity_S_per_m(value: conductivity_t, *, name: str) -> float:
    """Return a unit-bearing conductivity as S/m."""

    return require_quantity(value, CONDUCTIVITY, name=name)


def to_array(value: Any, unit: str, *, dtype: Any = float) -> np.ndarray:
    """Return `value` as a NumPy array expressed in `unit`."""

    return np.asarray(_convert_quantity_like(value, unit), dtype=dtype)


def to_um(value: Any) -> float:
    """Return a scalar length in micrometers."""

    return to_scalar(value, "micrometer")


def to_um_array(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
    """Return a length array in micrometers."""

    return to_array(value, "micrometer", dtype=dtype)


def to_ms(value: Any) -> float:
    """Return a scalar duration in milliseconds."""

    return to_scalar(value, "millisecond")


def to_ms_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return a time array in milliseconds."""

    return to_array(value, "millisecond", dtype=dtype)


def to_m(value: Any) -> float:
    """Return a scalar length in meters."""

    return to_scalar(value, "meter")


def to_m_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return a length array in meters."""

    return to_array(value, "meter", dtype=dtype)


def to_mV(value: Any) -> float:
    """Return a scalar voltage in millivolts."""

    return to_scalar(value, "millivolt")


def to_mV_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return a voltage array in millivolts."""

    return to_array(value, "millivolt", dtype=dtype)


def to_degC(value: Any) -> float:
    """Return a scalar temperature in degrees Celsius."""

    return to_scalar(value, "degree_Celsius")


def to_A(value: Any) -> float:
    """Return a scalar current in amperes."""

    return to_scalar(value, "ampere")


def to_A_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return a current array in amperes."""

    return to_array(value, "ampere", dtype=dtype)


def to_nA(value: Any) -> float:
    """Return a scalar current in nanoamperes."""

    return to_scalar(value, "nanoampere")


def to_nA_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return a current array in nanoamperes."""

    return to_array(value, "nanoampere", dtype=dtype)


def to_uA(value: Any) -> float:
    """Return a scalar current in microamperes."""

    return to_scalar(value, "microampere")


def to_uA_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Return a current array in microamperes."""

    return to_array(value, "microampere", dtype=dtype)


def to_ohm_cm(value: Any) -> float:
    """Return axial resistivity in ohm * centimeter."""

    return to_scalar(value, "ohm * centimeter")


def to_ohm_cm2(value: Any) -> float:
    """Return membrane specific resistance in ohm * centimeter**2."""

    return to_scalar(value, "ohm * centimeter ** 2")


def to_uF_per_cm2(value: Any) -> float:
    """Return membrane capacitance density in microfarad / centimeter**2."""

    return to_scalar(value, "microfarad / centimeter ** 2")


def to_S_per_cm2(value: Any) -> float:
    """Return conductance density in siemens / centimeter**2."""

    return to_scalar(value, "siemens / centimeter ** 2")


def to_S_per_m(value: Any) -> float:
    """Return conductivity in siemens / meter."""

    return to_scalar(value, "siemens / meter")


def to_mA_per_cm2(value: Any) -> float:
    """Return current density in milliampere / centimeter**2."""

    return to_scalar(value, "milliampere / centimeter ** 2")


def to_mM(value: Any) -> float:
    """Return concentration in millimolar."""

    return to_scalar(value, "millimolar")


def to_MOhm_per_cm(value: Any) -> float:
    """Return periaxonal axial resistance density in megaohm / centimeter."""

    return to_scalar(value, "megaohm / centimeter")


def _define_unit_aliases(registry: Any) -> None:
    aliases = (
        "ohm_centimeter = ohm * centimeter = ohm_cm",
        "megaohm_per_centimeter = megaohm / centimeter = Mohm_per_cm = MOhm_per_cm",
        "microfarad_per_square_centimeter = microfarad / centimeter ** 2 = uF_per_cm2",
        "siemens_per_square_centimeter = siemens / centimeter ** 2 = S_per_cm2",
        "siemens_per_meter = siemens / meter = S_per_m",
        "ohm_square_centimeter = ohm * centimeter ** 2 = ohm_cm2",
        "milliampere_per_square_centimeter = milliampere / centimeter ** 2 = mA_per_cm2",
    )
    for alias in aliases:
        try:
            registry.define(alias)
        except Exception:
            # Pint raises when an alias already exists. Existing definitions are
            # fine; these aliases are only ergonomic sugar for public examples.
            pass


@lru_cache(maxsize=1)
def get_unit_registry() -> Any:
    """Return AxonScope's Pint unit registry.

    The registry is created lazily so importing AxonScope still works in an
    environment that has not installed Pint yet. Pint is declared as a project
    dependency; this fallback keeps local development and partial installs from
    failing before unit-aware code is actually used.
    """

    try:
        import pint
    except ModuleNotFoundError as exc:
        raise UnitSupportError(
            "Pint is required for unit construction. Install the project "
            "dependencies, or pass plain numeric values in AxonScope's "
            "canonical public units."
        ) from exc
    registry: Any = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    _define_unit_aliases(registry)
    return registry


class _UnitRegistryProxy:
    """Lazy proxy so `axonscope.units.ureg` is import-safe without Pint."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_unit_registry(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return get_unit_registry()(*args, **kwargs)


ureg = _UnitRegistryProxy()


def Q_(value: Any, units: str | None = None) -> Any:
    """Construct a Pint quantity with AxonScope's unit registry."""

    registry = get_unit_registry()
    if units is None:
        return registry.Quantity(value)
    return registry.Quantity(value, units)


def has_pint() -> bool:
    """Return whether Pint can be imported in the current environment."""

    try:
        get_unit_registry()
    except UnitSupportError:
        return False
    return True


def __getattr__(name: str) -> Any:
    """Forward common unit attribute access to the lazy Pint registry."""

    return getattr(get_unit_registry(), name)


__all__ = [
    "CONCENTRATION_MM",
    "CONCENTRATION_PER_CURRENT_DENSITY_TIME",
    "CONCENTRATION_RATE_MM_PER_MS",
    "CONDUCTANCE_DENSITY_MS_CM2",
    "CURRENT_DENSITY_UA_CM2",
    "DIMENSIONLESS",
    "Q_",
    "RATE_PER_MS",
    "RATE_PER_MS_PER_MM",
    "RATE_PER_MS_PER_MV",
    "RESISTANCE_AREA_OHM_CM2",
    "TEMPERATURE_DEGC",
    "TIME_MS",
    "UnitAlias",
    "UnitExpression",
    "UnitSupportError",
    "VOLTAGE_MV",
    "degC",
    "dimensionless",
    "get_unit_registry",
    "gate",
    "has_pint",
    "is_quantity_like",
    "cm2",
    "mM",
    "mM_per_uA_cm2_ms",
    "mS",
    "mS_per_cm2",
    "mV",
    "ms",
    "ohm",
    "ohm_cm2",
    "per_ms",
    "per_ms_per_mM",
    "per_ms_per_mV",
    "quantity_unit",
    "require_axoplasmic_resistivity_ohm_um",
    "require_axial_resistivity_ohm_cm",
    "require_capacitance_density_uF_per_cm2",
    "require_conductance_density_S_per_cm2",
    "require_current_A",
    "require_current_array_uA",
    "require_current_uA",
    "require_length_array_um",
    "require_length_um",
    "require_periaxonal_resistance_MOhm_per_cm",
    "require_scalar_quantity",
    "require_temperature_degC",
    "require_time_array_ms",
    "require_time_ms",
    "require_voltage_mV",
    "short_unit_label",
    "to_A",
    "to_A_array",
    "to_MOhm_per_cm",
    "to_S_per_cm2",
    "to_S_per_m",
    "to_array",
    "to_degC",
    "to_m",
    "to_mV",
    "to_mV_array",
    "to_m_array",
    "to_mA_per_cm2",
    "to_mM",
    "to_ms",
    "to_ms_array",
    "to_nA",
    "to_nA_array",
    "to_ohm_cm",
    "to_ohm_cm2",
    "to_scalar",
    "to_uF_per_cm2",
    "to_uA",
    "to_uA_array",
    "to_um",
    "to_um_array",
    "uA",
    "uA_per_cm2",
    "um",
    "unit_label",
    "ureg",
]
