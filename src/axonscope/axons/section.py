"""Local, descriptive axon sections.

`section.py` owns the smallest anatomical/modeling building blocks in the axon
package:

- `Section`: one named piece of cable with a membrane model, local geometry,
  cable material parameters, and tags.
- `PeriaxonalLayer`: optional local double-cable material around a section.

The public constructor API intentionally uses short physical names such as
`diameter`, `Ra`, and `Cm`. Those values must carry explicit units; plain
numbers are rejected here. Constructor validation checks both presence of units
and physical dimension through `axonscope.utils.units.require_*` helpers.

The stored attributes stay explicit canonical floats (`diameter_um`,
`Ra_ohm_cm`, `Cm_uF_cm2`, ...). This keeps descriptive objects
backend-independent. Spatial length and compartment counts live in `Layout`.

This module must remain free of stimulation protocols, electrode placement,
solver-facing aliases, runtime state, and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from axonscope.utils import units
from axonscope.utils.validation import (
    normalize_non_empty_string,
    normalize_string_tuple,
    require_non_negative,
    require_positive,
)
from axonscope.membranes.model import MembraneModel, Model, ensure_membrane_model


def _default_Ra() -> units.axial_resistivity_t:
    return units.Q_(100.0, "ohm * centimeter")


def _default_Cm() -> units.capacitance_density_t:
    return units.Q_(1.0, "microfarad / centimeter ** 2")


@dataclass(frozen=True, init=False)
class PeriaxonalLayer:
    """Local periaxonal layer material for double-cable sections.

    Parameters are local material properties around one section. They are not an
    extracellular stimulation context and they do not describe an electrode.

    Attributes
    ----------
    radial_conductance_S_cm2:
        Radial conductance density through the periaxonal/myelin layer in
        S/cm^2.
    radial_capacitance_uF_cm2:
        Radial capacitance density through the periaxonal/myelin layer in
        uF/cm^2.
    axial_resistance_MOhm_per_cm:
        Longitudinal periaxonal resistance density in MOhm/cm.
    """

    radial_conductance_S_cm2: float
    radial_capacitance_uF_cm2: float
    axial_resistance_MOhm_per_cm: float

    def __init__(
        self,
        *,
        radial_conductance: units.conductance_density_t,
        radial_capacitance: units.capacitance_density_t,
        axial_resistance: units.periaxonal_resistance_t,
    ) -> None:
        """Create a periaxonal layer.

        Parameters
        ----------
        radial_conductance:
            Radial conductance density through the periaxonal/myelin layer, with
            units convertible to S/cm^2.
        radial_capacitance:
            Radial capacitance density through the periaxonal/myelin layer, with
            units convertible to uF/cm^2.
        axial_resistance:
            Longitudinal periaxonal resistance density, with units convertible
            to MOhm/cm.
        """

        radial_conductance_value = units.require_conductance_density_S_per_cm2(
            radial_conductance,
            name="radial_conductance",
        )
        radial_capacitance_value = units.require_capacitance_density_uF_per_cm2(
            radial_capacitance,
            name="radial_capacitance",
        )
        axial_resistance_value = units.require_periaxonal_resistance_MOhm_per_cm(
            axial_resistance,
            name="axial_resistance",
        )
        object.__setattr__(
            self,
            "radial_conductance_S_cm2",
            require_non_negative(radial_conductance_value, name="radial_conductance"),
        )
        object.__setattr__(
            self,
            "radial_capacitance_uF_cm2",
            require_non_negative(radial_capacitance_value, name="radial_capacitance"),
        )
        object.__setattr__(
            self,
            "axial_resistance_MOhm_per_cm",
            require_positive(axial_resistance_value, name="axial_resistance"),
        )


@dataclass(frozen=True)
class Section:
    """Conceptual local section of an axon.

    A section has one membrane description and one local material/diameter
    description. Spatial length and compartment count are assigned by `Layout`.

    Attributes
    ----------
    name:
        Human-readable section name, for example `"node"`, `"internode"`, or
        `"axon"`.
    membrane:
        Runtime-independent membrane model applied to all numerical
        compartments in this conceptual section.
    diameter_um:
        Section diameter in micrometers.
    Ra_ohm_cm:
        Intracellular axial resistivity in ohm * centimeter.
    Cm_uF_cm2:
        Membrane capacitance density in uF/cm^2.
    periaxonal:
        Optional section-local double-cable periaxonal layer.
    tags:
        Optional descriptive labels copied to flattened compartments.
    """

    name: str
    membrane: MembraneModel | Model
    diameter_um: float
    Ra_ohm_cm: float = field(init=False)
    Cm_uF_cm2: float = field(init=False)
    periaxonal: PeriaxonalLayer | None = None
    tags: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        membrane: object,
        diameter: units.length_t,
        Ra: units.axial_resistivity_t | None = None,
        Cm: units.capacitance_density_t | None = None,
        periaxonal: PeriaxonalLayer | None = None,
        tags: Sequence[str] | str = (),
    ) -> None:
        """Create one conceptual axon section.

        Parameters
        ----------
        name:
            Human-readable section name, for example `"node"` or `"axon"`.
        membrane:
            Membrane model assigned to every numerical compartment in this
            section.
        diameter:
            Section diameter, with units convertible to micrometers.
        Ra:
            Intracellular axial resistivity, with units convertible to
            ohm * centimeter. Defaults to `100.0 * axs.ohm_cm`.
        Cm:
            Membrane capacitance density, with units convertible to uF/cm^2.
            Defaults to `1.0 * axs.uF_per_cm2`.
        periaxonal:
            Optional double-cable periaxonal layer parameters.
        tags:
            Optional descriptive tags copied to flattened compartments.
        """

        diameter_value = units.require_length_um(
            diameter,
            name="diameter",
        )
        Ra_value = units.require_axial_resistivity_ohm_cm(
            _default_Ra() if Ra is None else Ra,
            name="Ra",
        )
        Cm_value = units.require_capacitance_density_uF_per_cm2(
            _default_Cm() if Cm is None else Cm,
            name="Cm",
        )
        if periaxonal is not None and not isinstance(periaxonal, PeriaxonalLayer):
            raise TypeError("periaxonal must be a PeriaxonalLayer or None.")
        if isinstance(membrane, type) and issubclass(membrane, Model):
            membrane = membrane()
        ensure_membrane_model(membrane)
        object.__setattr__(self, "name", normalize_non_empty_string(name, name="Section name"))
        object.__setattr__(self, "membrane", membrane)
        object.__setattr__(self, "diameter_um", require_positive(diameter_value, name="diameter"))
        object.__setattr__(self, "Ra_ohm_cm", require_positive(Ra_value, name="Ra"))
        object.__setattr__(self, "Cm_uF_cm2", require_positive(Cm_value, name="Cm"))
        object.__setattr__(self, "periaxonal", periaxonal)
        object.__setattr__(self, "tags", normalize_string_tuple(tags, name="Section tags"))


__all__ = ["PeriaxonalLayer", "Section"]
