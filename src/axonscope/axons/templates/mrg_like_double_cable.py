"""MRG-like double-cable section-layout template.

This module expands the McIntyre-Richardson-Grill morphology into descriptive
node/MYSA/FLUT/STIN sections. Public entry points require unit-bearing physical
values and return `Layout` objects; solver arrays are derived elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from axonscope.utils import units
from axonscope.utils.units import (
    axoplasmic_resistivity_t,
    capacitance_density_t,
    conductance_density_t,
    length_t,
    temperature_t,
    voltage_t,
)
from axonscope.utils.validation import (
    normalize_positive_int,
    require_non_negative,
    require_positive,
)
from axonscope.axons.diameters import round_axon_diameter_um
from axonscope.axons.layout import Layout, LayoutElement
from axonscope.axons.section import PeriaxonalLayer, Section
from axonscope.axons.templates._mrg_morphology import (
    get_mrg_length_node_spacing,
    get_mrg_morphology,
)
from axonscope.membranes import SectionLayout
from ... import membranes as membrane_specs


_GEOMETRY_STOP_ATOL_UM = 1e-9
_DEFAULT_NODE_LENGTH = units.Q_(1.0, "micrometer")
_DEFAULT_MYSA_LENGTH = units.Q_(3.0, "micrometer")
_DEFAULT_AXOPLASMIC_RESISTIVITY = units.Q_(0.7e6, "ohm * micrometer")
_DEFAULT_MYELIN_CAPACITANCE = units.Q_(0.1, "microfarad / centimeter ** 2")
_DEFAULT_MYELIN_CONDUCTANCE = units.Q_(0.001, "siemens / centimeter ** 2")
_DEFAULT_NODE_SPACE = units.Q_(0.002, "micrometer")
_DEFAULT_FLUT_SPACE = units.Q_(0.004, "micrometer")
_DEFAULT_STIN_SPACE = units.Q_(0.004, "micrometer")
_DEFAULT_NODE_ENA = units.Q_(50.0, "millivolt")
_DEFAULT_NODE_EK = units.Q_(-90.0, "millivolt")
_DEFAULT_NODE_EL = units.Q_(-90.0, "millivolt")
_DEFAULT_INTERNODE_EL = units.Q_(-80.0, "millivolt")
_DEFAULT_TEMPERATURE = units.Q_(37.0, "degree_Celsius")

# Either one compartment count for every MRG section, or a per-section mapping
# such as {"node": 1, "MYSA": 1, "FLUT": 2, "STIN": 4}.
SectionCompartments: TypeAlias = int | Mapping[str, int]


@dataclass(frozen=True)
class MRGLikeDoubleCableGeometry:
    """Expanded MRG-like section geometry in canonical micrometer-based floats.

    This is derived template data: it stores one entry per placed section before
    layout compartment subdivision. User-facing constructors should receive
    unit-bearing values and usually return a `Layout` instead of this object.
    """

    fiber_d_um: float
    lengths_um: tuple[float, ...]
    diam_um: tuple[float, ...]
    Ra_ohm_cm: tuple[float, ...]
    Cm_uF_cm2: tuple[float, ...]
    leak_mS_cm2: tuple[float, ...]
    is_node: tuple[bool, ...]
    section_names: tuple[str, ...]
    periaxonal_layers: tuple[PeriaxonalLayer, ...]


def mrg_like_section_sequence() -> tuple[str, ...]:
    """Return the canonical node/MYSA/FLUT/STIN MRG-like sequence."""

    return (
        "node",
        "MYSA",
        "FLUT",
        "STIN",
        "STIN",
        "STIN",
        "STIN",
        "STIN",
        "STIN",
        "FLUT",
        "MYSA",
    )


def _normalize_nodes(nodes: int) -> int:
    node_count = normalize_positive_int(nodes, name="nodes")
    if node_count < 2:
        raise ValueError(f"nodes must be >= 2, got {node_count}.")
    return node_count


def _normalize_section_compartments(value: SectionCompartments) -> dict[str, int] | int:
    valid_sections = {name.lower() for name in mrg_like_section_sequence()}
    if isinstance(value, Mapping):
        normalized: dict[str, int] = {}
        for key, count in value.items():
            section_name = str(key).strip().lower()
            if section_name not in valid_sections:
                choices = ", ".join(sorted(valid_sections))
                raise ValueError(
                    f"Unknown MRG section {key!r} in compartments; expected one of: {choices}."
                )
            normalized[section_name] = normalize_positive_int(
                count,
                name=f"compartments[{key!r}]",
            )
        return normalized
    return normalize_positive_int(value, name="compartments")


def _compartments_for_section(value: dict[str, int] | int, section_name: str) -> int:
    if isinstance(value, int):
        return value
    return value.get(section_name.lower(), 1)


def mrg_like_node_spacing(diameter: length_t, *, fit_all: bool = False) -> float:
    """Return the MRG-like center-to-center node spacing in micrometers."""

    diameter_um = round_axon_diameter_um(
        units.require_length_um(diameter, name="diameter")
    )
    return float(get_mrg_morphology(diameter_um, fit_all=fit_all).deltax)


def mrg_like_length_from_nodes(
    diameter: length_t,
    nodes: int,
    *,
    x_shift: length_t | None = None,
    fit_all: bool = False,
) -> float:
    """Return the NRV-compatible MRG-like length for a requested node count.

    Parameters
    ----------
    diameter:
        Fiber diameter, with units.
    nodes:
        Requested number of Ranvier nodes.
    x_shift:
        Optional intrinsic distance from the axon start to the first node
        start. This phases the repeated MRG motif without assigning world
        coordinates.
    """

    nodes = _normalize_nodes(nodes)
    diameter_um = round_axon_diameter_um(
        units.require_length_um(diameter, name="diameter")
    )
    deltax = get_mrg_length_node_spacing(diameter_um, fit_all=fit_all)
    shift_um = 0.0 if x_shift is None else units.require_length_um(x_shift, name="x_shift")
    phase_um = float(shift_um) % float(deltax)
    return float(math.ceil(phase_um + deltax * (nodes - 1)))


def mrg_like_nodes_from_length(
    diameter: length_t,
    length: length_t,
    *,
    x_shift: length_t | None = None,
    fit_all: bool = False,
) -> int:
    """Return the approximate node count for a requested MRG-like length.

    Parameters
    ----------
    diameter:
        Fiber diameter, with units.
    length:
        Requested axon length, with units.
    x_shift:
        Optional intrinsic distance from the axon start to the first node
        start.
    """

    length_um = units.require_length_um(length, name="length")
    if length_um <= 0:
        raise ValueError(f"length must be positive, got {length}.")
    deltax = mrg_like_node_spacing(diameter, fit_all=fit_all)
    shift_um = 0.0 if x_shift is None else units.require_length_um(x_shift, name="x_shift")
    phase_um = float(shift_um) % float(deltax)
    if length_um <= phase_um:
        return 0
    return int(math.floor((length_um - phase_um) / deltax)) + 1


def build_mrg_like_geometry(
    *,
    diameter: length_t,
    nodes: int,
    length: length_t | None = None,
    x_shift: length_t | None = None,
    fit_all: bool = False,
    mysa_length: length_t = _DEFAULT_MYSA_LENGTH,
    node_length: length_t = _DEFAULT_NODE_LENGTH,
    axoplasmic_resistivity: axoplasmic_resistivity_t = _DEFAULT_AXOPLASMIC_RESISTIVITY,
    myelin_capacitance: capacitance_density_t = _DEFAULT_MYELIN_CAPACITANCE,
    myelin_conductance: conductance_density_t = _DEFAULT_MYELIN_CONDUCTANCE,
    node_space: length_t = _DEFAULT_NODE_SPACE,
    flut_space: length_t = _DEFAULT_FLUT_SPACE,
    stin_space: length_t = _DEFAULT_STIN_SPACE,
) -> MRGLikeDoubleCableGeometry:
    """Build an expanded MRG-like double-cable section sequence.

    Parameters
    ----------
    diameter:
        Fiber diameter, with units.
    nodes:
        Number of Ranvier nodes to include.
    length:
        Optional nominal length, with units. If omitted, MRG internode spacing determines
        the length from `nodes`.
    x_shift:
        Intrinsic phase shift along the MRG motif: distance from the axon
        start to the first node start. Values are wrapped modulo the node
        spacing.
    fit_all:
        Use polynomial morphology fits even for tabulated MRG diameters.
    mysa_length, node_length:
        MYSA and node section lengths, with units.
    axoplasmic_resistivity:
        Axoplasmic resistivity, with units convertible to ohm * micrometer.
    myelin_capacitance:
        Myelin capacitance density, with units convertible to uF/cm^2.
    myelin_conductance:
        Myelin conductance density, with units convertible to S/cm^2.
    node_space, flut_space, stin_space:
        Periaxonal spaces around node/MYSA, FLUT, and STIN sections.
    """

    nodes = _normalize_nodes(nodes)
    diameter_um = round_axon_diameter_um(
        units.require_length_um(diameter, name="diameter")
    )
    length_um = None if length is None else units.require_length_um(length, name="length")
    mysa_length_um = require_positive(
        units.require_length_um(mysa_length, name="mysa_length"),
        name="mysa_length",
    )
    node_length_um = require_positive(
        units.require_length_um(node_length, name="node_length"),
        name="node_length",
    )
    rhoa_ohm_um = require_positive(
        units.require_axoplasmic_resistivity_ohm_um(
            axoplasmic_resistivity,
            name="axoplasmic_resistivity",
        ),
        name="axoplasmic_resistivity",
    )
    mycm_uF_cm2 = require_positive(
        units.require_capacitance_density_uF_per_cm2(
            myelin_capacitance,
            name="myelin_capacitance",
        ),
        name="myelin_capacitance",
    )
    mygm_S_cm2 = require_non_negative(
        units.require_conductance_density_S_per_cm2(
            myelin_conductance,
            name="myelin_conductance",
        ),
        name="myelin_conductance",
    )
    node_space_um = require_positive(
        units.require_length_um(node_space, name="node_space"),
        name="node_space",
    )
    flut_space_um = require_positive(
        units.require_length_um(flut_space, name="flut_space"),
        name="flut_space",
    )
    stin_space_um = require_positive(
        units.require_length_um(stin_space, name="stin_space"),
        name="stin_space",
    )
    morph = get_mrg_morphology(float(diameter_um), fit_all=fit_all)
    x_shift_um = 0.0 if x_shift is None else units.require_length_um(x_shift, name="x_shift")
    phase_um = float(x_shift_um) % float(morph.deltax)
    if length_um is None:
        length_um = (
            mrg_like_length_from_nodes(
                units.Q_(diameter_um, "micrometer"),
                nodes,
                x_shift=units.Q_(phase_um, "micrometer"),
                fit_all=fit_all,
            )
            + 1.0
        )

    paralength1 = float(mysa_length_um)
    nodelength = float(node_length_um)
    paralength2 = float(morph.paralength2)
    interlength = float(
        (morph.deltax - nodelength - 2.0 * paralength1 - 2.0 * paralength2) / 6.0
    )

    rhoa = float(rhoa_ohm_um)
    mycm = float(mycm_uF_cm2)
    mygm = float(mygm_S_cm2)
    space_p1 = float(node_space_um)
    space_p2 = float(flut_space_um)
    space_i = float(stin_space_um)

    Rpn0 = (rhoa * 0.01) / (
        math.pi * ((((morph.nodeD / 2.0) + space_p1) ** 2) - ((morph.nodeD / 2.0) ** 2))
    )
    Rpn1 = (rhoa * 0.01) / (
        math.pi * ((((morph.paraD1 / 2.0) + space_p1) ** 2) - ((morph.paraD1 / 2.0) ** 2))
    )
    Rpn2 = (rhoa * 0.01) / (
        math.pi * ((((morph.paraD2 / 2.0) + space_p2) ** 2) - ((morph.paraD2 / 2.0) ** 2))
    )
    Rpx = (rhoa * 0.01) / (
        math.pi * ((((morph.axonD / 2.0) + space_i) ** 2) - ((morph.axonD / 2.0) ** 2))
    )

    seq = mrg_like_section_sequence()
    motif_lengths = {
        "node": nodelength,
        "MYSA": paralength1,
        "FLUT": paralength2,
        "STIN": interlength,
    }
    seq_lengths = [motif_lengths[kind] for kind in seq]
    seq_boundaries = [0.0]
    for seq_length in seq_lengths:
        seq_boundaries.append(seq_boundaries[-1] + float(seq_length))
    motif_length = float(seq_boundaries[-1])
    if not math.isclose(motif_length, float(morph.deltax), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(
            "MRG motif length does not match node spacing; "
            f"motif={motif_length} um, deltax={morph.deltax} um."
        )
    start_coordinate = (motif_length - phase_um) % motif_length
    if math.isclose(start_coordinate, motif_length, rel_tol=0.0, abs_tol=1e-9):
        start_coordinate = 0.0
    seq_index = int(np.searchsorted(np.asarray(seq_boundaries[1:]), start_coordinate, side="right"))
    lengths: list[float] = []
    diam: list[float] = []
    Ra: list[float] = []
    Cm: list[float] = []
    leak: list[float] = []
    is_node: list[bool] = []
    kinds: list[str] = []
    periaxonal: list[PeriaxonalLayer] = []

    x0 = 0.0
    first_section = True
    while True:
        remaining_um = float(length_um) - x0
        if remaining_um <= _GEOMETRY_STOP_ATOL_UM:
            break
        kind = seq[seq_index % len(seq)]
        natural_length = motif_lengths[kind]
        if first_section:
            natural_length = float(seq_boundaries[seq_index + 1] - start_coordinate)
            first_section = False
        if kind == "node":
            Lk = natural_length
            dk = float(morph.nodeD)
            ratio = 1.0
            Rak = rhoa / 10000.0
            Cmk = 2.0
            leakk_mS = 0.0
            xr, xg, xc = Rpn0, 1e10, 0.0
        elif kind == "MYSA":
            Lk = natural_length
            dk = float(morph.fiberD)
            ratio = float(morph.paraD1 / morph.fiberD)
            Rak = rhoa * (1.0 / ratio**2) / 10000.0
            Cmk = 2.0 * ratio
            leakk_mS = 0.001 * ratio * 1e3
            xr, xg, xc = Rpn1, mygm / (morph.nl * 2.0), mycm / (morph.nl * 2.0)
        elif kind == "FLUT":
            Lk = natural_length
            dk = float(morph.fiberD)
            ratio = float(morph.paraD2 / morph.fiberD)
            Rak = rhoa * (1.0 / ratio**2) / 10000.0
            Cmk = 2.0 * ratio
            leakk_mS = 0.0001 * ratio * 1e3
            xr, xg, xc = Rpn2, mygm / (morph.nl * 2.0), mycm / (morph.nl * 2.0)
        else:
            Lk = natural_length
            dk = float(morph.fiberD)
            ratio = float(morph.axonD / morph.fiberD)
            Rak = rhoa * (1.0 / ratio**2) / 10000.0
            Cmk = 2.0 * ratio
            leakk_mS = 0.0001 * ratio * 1e3
            xr, xg, xc = Rpx, mygm / (morph.nl * 2.0), mycm / (morph.nl * 2.0)

        Lk = min(float(Lk), remaining_um)
        if Lk <= _GEOMETRY_STOP_ATOL_UM:
            break
        lengths.append(Lk)
        diam.append(dk)
        Ra.append(Rak)
        Cm.append(Cmk)
        leak.append(leakk_mS)
        is_node.append(kind == "node")
        kinds.append(kind)
        periaxonal.append(
            PeriaxonalLayer(
                radial_conductance=units.Q_(float(xg), "siemens / centimeter ** 2"),
                radial_capacitance=units.Q_(float(xc), "microfarad / centimeter ** 2"),
                axial_resistance=units.Q_(float(xr), "megaohm / centimeter"),
            )
        )
        x0 += Lk
        seq_index += 1

    if sum(1 for value in is_node if value) < 2:
        raise ValueError("Generated MRG-like layout has fewer than 2 nodes.")

    return MRGLikeDoubleCableGeometry(
        fiber_d_um=float(diameter_um),
        lengths_um=tuple(lengths),
        diam_um=tuple(diam),
        Ra_ohm_cm=tuple(Ra),
        Cm_uF_cm2=tuple(Cm),
        leak_mS_cm2=tuple(leak),
        is_node=tuple(is_node),
        section_names=tuple(kinds),
        periaxonal_layers=tuple(periaxonal),
    )


def _leak_for_section(geometry: MRGLikeDoubleCableGeometry, section: str) -> float:
    key = section.lower()
    for kind, leak in zip(geometry.section_names, geometry.leak_mS_cm2, strict=True):
        if kind.lower() == key:
            return float(leak)
    raise KeyError(f"section {section!r} is not present in this geometry.")


def _passive_from_leak(leak_mS_per_cm2: float, e_rev_mV: voltage_t):
    leak = max(float(leak_mS_per_cm2), 1e-12)
    return membrane_specs.Passive(Rm=1e3 / leak, EL=e_rev_mV)


def default_mrg_like_membranes(
    geometry: MRGLikeDoubleCableGeometry,
    *,
    ena: voltage_t = _DEFAULT_NODE_ENA,
    ek: voltage_t = _DEFAULT_NODE_EK,
    node_el: voltage_t = _DEFAULT_NODE_EL,
    internode_el: voltage_t = _DEFAULT_INTERNODE_EL,
    temperature: temperature_t = _DEFAULT_TEMPERATURE,
) -> SectionLayout:
    """Return default membranes for an MRG-like double-cable layout.

    Parameters
    ----------
    geometry:
        Expanded MRG-like geometry.
    ena, ek:
        Sodium and potassium reversal potentials.
    node_el, internode_el:
        Passive leak reversal potentials for node/internode sections.
    temperature:
        Membrane temperature in degrees Celsius.
    """

    temperature = units.Q_(
        units.require_temperature_degC(temperature, name="temperature"),
        "degree_Celsius",
    )
    ena = units.Q_(units.require_voltage_mV(ena, name="ena"), "millivolt")
    ek = units.Q_(units.require_voltage_mV(ek, name="ek"), "millivolt")
    node_el = units.Q_(units.require_voltage_mV(node_el, name="node_el"), "millivolt")
    internode_el = units.Q_(
        units.require_voltage_mV(internode_el, name="internode_el"),
        "millivolt",
    )
    return SectionLayout(
        node=membrane_specs.AxNode(
            ena=ena,
            ek=ek,
            el=node_el,
            temperature=temperature,
        ),
        mysa=_passive_from_leak(_leak_for_section(geometry, "MYSA"), internode_el),
        flut=_passive_from_leak(_leak_for_section(geometry, "FLUT"), internode_el),
        stin=_passive_from_leak(_leak_for_section(geometry, "STIN"), internode_el),
    )


def layout_from_mrg_like_geometry(
    geometry: MRGLikeDoubleCableGeometry,
    *,
    membranes: SectionLayout,
    compartments: SectionCompartments = 1,
) -> Layout:
    """Build a Layout from pre-expanded MRG-like geometry and membranes.

    Parameters
    ----------
    geometry:
        Expanded MRG-like section geometry.
    membranes:
        Section-to-membrane assignment.
    compartments:
        Either one compartment count for every placed section, or a mapping
        from section name to compartment count. Missing mapping entries default
        to one compartment.
    """

    section_compartments = _normalize_section_compartments(compartments)
    sections = [
        Section(
            kind,
            membrane=membranes.membrane_for(kind),
            diameter=units.Q_(diam, "micrometer"),
            Ra=units.Q_(Ra, "ohm * centimeter"),
            Cm=units.Q_(Cm, "microfarad / centimeter ** 2"),
            periaxonal=periaxonal,
            tags=("myelinated", "node") if is_node else ("myelinated", kind.lower()),
        )
        for kind, diam, Ra, Cm, periaxonal, is_node in zip(
            geometry.section_names,
            geometry.diam_um,
            geometry.Ra_ohm_cm,
            geometry.Cm_uF_cm2,
            geometry.periaxonal_layers,
            geometry.is_node,
            strict=True,
        )
    ]
    return Layout(
        [
            LayoutElement(
                section,
                length=units.Q_(length_um, "micrometer"),
                compartments=_compartments_for_section(section_compartments, section.name),
            )
            for section, length_um in zip(sections, geometry.lengths_um, strict=True)
        ],
    )


def mrg_like_layout(
    *,
    diameter: length_t,
    nodes: int,
    membranes: SectionLayout | None = None,
    length: length_t | None = None,
    compartments: SectionCompartments = 1,
    x_shift: length_t | None = None,
    temperature: temperature_t = _DEFAULT_TEMPERATURE,
    fit_all: bool = False,
    mysa_length: length_t = _DEFAULT_MYSA_LENGTH,
    node_length: length_t = _DEFAULT_NODE_LENGTH,
    axoplasmic_resistivity: axoplasmic_resistivity_t = _DEFAULT_AXOPLASMIC_RESISTIVITY,
    myelin_capacitance: capacitance_density_t = _DEFAULT_MYELIN_CAPACITANCE,
    myelin_conductance: conductance_density_t = _DEFAULT_MYELIN_CONDUCTANCE,
    node_space: length_t = _DEFAULT_NODE_SPACE,
    flut_space: length_t = _DEFAULT_FLUT_SPACE,
    stin_space: length_t = _DEFAULT_STIN_SPACE,
) -> Layout:
    """Build a reusable MRG-like double-cable section layout.

    Parameters are the same as `build_mrg_like_geometry`; `membranes` may
    override the default MRG section membrane assignment. `compartments` can
    be an integer or a mapping such as `{"node": 1, "FLUT": 2, "STIN": 4}`.
    `x_shift` phases the repeated MRG motif without assigning world
    coordinates; it is the distance from the axon start to the first node
    start.
    """

    geometry = build_mrg_like_geometry(
        diameter=diameter,
        nodes=nodes,
        length=length,
        x_shift=x_shift,
        fit_all=fit_all,
        mysa_length=mysa_length,
        node_length=node_length,
        axoplasmic_resistivity=axoplasmic_resistivity,
        myelin_capacitance=myelin_capacitance,
        myelin_conductance=myelin_conductance,
        node_space=node_space,
        flut_space=flut_space,
        stin_space=stin_space,
    )
    section_membranes = membranes or default_mrg_like_membranes(
        geometry,
        temperature=temperature,
    )
    return layout_from_mrg_like_geometry(
        geometry,
        membranes=section_membranes,
        compartments=compartments,
    )


@dataclass(frozen=True)
class MRGLikeDoubleCableTemplate:
    """Reusable node/MYSA/FLUT/STIN double-cable section template.

    All physical fields require explicit units.
    """

    diameter: length_t
    nodes: int
    length: length_t | None = None
    compartments: SectionCompartments = 1
    x_shift: length_t | None = None
    fit_all: bool = False
    mysa_length: length_t = _DEFAULT_MYSA_LENGTH
    node_length: length_t = _DEFAULT_NODE_LENGTH
    axoplasmic_resistivity: axoplasmic_resistivity_t = _DEFAULT_AXOPLASMIC_RESISTIVITY
    myelin_capacitance: capacitance_density_t = _DEFAULT_MYELIN_CAPACITANCE
    myelin_conductance: conductance_density_t = _DEFAULT_MYELIN_CONDUCTANCE
    node_space: length_t = _DEFAULT_NODE_SPACE
    flut_space: length_t = _DEFAULT_FLUT_SPACE
    stin_space: length_t = _DEFAULT_STIN_SPACE

    @property
    def nominal_length_um(self) -> float:
        """Requested template length in micrometers."""

        if self.length is not None:
            return units.require_length_um(self.length, name="length")
        return mrg_like_length_from_nodes(
            self.diameter,
            self.nodes,
            x_shift=self.x_shift,
            fit_all=self.fit_all,
        )

    def geometry(self) -> MRGLikeDoubleCableGeometry:
        """Return the expanded MRG-like section geometry."""

        return build_mrg_like_geometry(
            diameter=self.diameter,
            nodes=self.nodes,
            length=self.length,
            x_shift=self.x_shift,
            fit_all=self.fit_all,
            mysa_length=self.mysa_length,
            node_length=self.node_length,
            axoplasmic_resistivity=self.axoplasmic_resistivity,
            myelin_capacitance=self.myelin_capacitance,
            myelin_conductance=self.myelin_conductance,
            node_space=self.node_space,
            flut_space=self.flut_space,
            stin_space=self.stin_space,
        )

    def default_membranes(
        self,
        *,
        temperature: temperature_t = _DEFAULT_TEMPERATURE,
    ) -> SectionLayout:
        """Return the default MRG membrane assignment for this geometry."""

        return default_mrg_like_membranes(self.geometry(), temperature=temperature)

    def layout(
        self,
        *,
        membranes: SectionLayout | None = None,
        temperature: temperature_t = _DEFAULT_TEMPERATURE,
    ) -> Layout:
        """Build a descriptive `Layout` from this template.

        Parameters
        ----------
        membranes:
            Optional section-to-membrane assignment.
        temperature:
            Temperature for the default membrane assignment.
        """

        geometry = self.geometry()
        section_membranes = membranes or default_mrg_like_membranes(
            geometry,
            temperature=temperature,
        )
        return layout_from_mrg_like_geometry(
            geometry,
            membranes=section_membranes,
            compartments=self.compartments,
        )


__all__ = [
    "MRGLikeDoubleCableGeometry",
    "MRGLikeDoubleCableTemplate",
    "SectionCompartments",
    "build_mrg_like_geometry",
    "default_mrg_like_membranes",
    "layout_from_mrg_like_geometry",
    "mrg_like_layout",
    "mrg_like_length_from_nodes",
    "mrg_like_nodes_from_length",
    "mrg_like_node_spacing",
    "mrg_like_section_sequence",
]
