"""Myelinated axon descriptions and concrete myelinated templates.

This module owns high-level descriptive constructors such as `MRG`. Template
helpers build section layouts, while `Myelinated` adds node-oriented inspection
helpers on top of the base `Axon` description.
"""

from __future__ import annotations

from typing import Any

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
from axonscope.axons.axon import Axon
from axonscope.axons.formulation import CableFormulation
from axonscope.axons.layout import Layout
from axonscope.axons.templates.mrg_like_double_cable import (
    MRGLikeDoubleCableTemplate,
    SectionCompartments,
    gaines_motor_membranes,
    gaines_sensory_membranes,
    mrg_like_layout,
    mrg_like_length_from_nodes,
    mrg_like_nodes_from_length,
)
from axonscope.membranes import SectionLayout


_DEFAULT_V_INIT = units.Q_(-80.0, "millivolt")
_DEFAULT_TEMPERATURE = units.Q_(37.0, "degree_Celsius")
_DEFAULT_NODE_LENGTH = units.Q_(1.0, "micrometer")
_DEFAULT_MYSA_LENGTH = units.Q_(3.0, "micrometer")
_DEFAULT_AXOPLASMIC_RESISTIVITY = units.Q_(0.7e6, "ohm * micrometer")
_DEFAULT_MYELIN_CAPACITANCE = units.Q_(0.1, "microfarad / centimeter ** 2")
_DEFAULT_MYELIN_CONDUCTANCE = units.Q_(0.001, "siemens / centimeter ** 2")
_DEFAULT_NODE_SPACE = units.Q_(0.002, "micrometer")
_DEFAULT_FLUT_SPACE = units.Q_(0.004, "micrometer")
_DEFAULT_STIN_SPACE = units.Q_(0.004, "micrometer")


class Myelinated(Axon):
    """Base class for myelinated axon descriptions.

    Myelinated axons usually expose Ranvier-node helpers and use the
    `double-cable` formulation by default, but they still share the same
    `Layout -> Axon` descriptive pipeline as unmyelinated axons.
    """

    def __init__(
        self,
        *,
        layout: Layout,
        formulation: CableFormulation | None = CableFormulation.DOUBLE_CABLE,
        diameter: length_t | None = None,
        v_init: voltage_t = _DEFAULT_V_INIT,
        temperature: temperature_t = _DEFAULT_TEMPERATURE,
    ) -> None:
        """Create a myelinated axon from a descriptive layout.

        Parameters
        ----------
        layout:
            Section layout, usually with periaxonal data on every section.
        formulation:
            Cable formulation, normally `"double-cable"`.
        diameter:
            Optional nominal fiber diameter, with units.
        v_init:
            Initial membrane potential in millivolts.
        temperature:
            Model temperature in degrees Celsius.
        """

        super().__init__(
            layout=layout,
            formulation=formulation,
            diameter=diameter,
            v_init=v_init,
            temperature=temperature,
        )

    @property
    def node_mask(self) -> np.ndarray:
        """Boolean mask selecting Ranvier-node compartments."""

        mask: list[bool] = []
        for element in self.layout.elements:
            tags = {tag.lower() for tag in element.section.tags}
            is_node = element.section.name.lower() == "node" or "node" in tags
            mask.extend([is_node] * element.compartments)
        return np.asarray(mask, dtype=bool)

    @property
    def node_indices(self) -> np.ndarray:
        """Indices of Ranvier-node compartments."""

        return np.where(self.node_mask)[0].astype(np.int32)

    @property
    def nodes(self) -> int:
        """Number of Ranvier-node compartments."""

        return int(self.node_indices.shape[0])

    @property
    def x_nodes_um(self) -> np.ndarray:
        """Ranvier-node positions along the local fiber axis in micrometers."""

        return self.layout.position_values(unit="micrometer")[self.node_indices]

    def node_position_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return Ranvier-node positions as plain values in `unit`.

        Parameters
        ----------
        unit:
            Target length unit. Plain strings and Pint units are accepted.
        """

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(units.Q_(self.x_nodes_um, "micrometer"), unit_label, dtype=float)

    def node_index(self, index: int | str = 0) -> int:
        """Return the compartment index of one Ranvier node."""

        node_indices = self.node_indices
        resolved = _resolve_node_selector(index, int(node_indices.shape[0]))
        return int(node_indices[resolved])

    def node_position(
        self,
        index: int | str = 0,
        *,
        unit: Any = "micrometer",
    ) -> length_t:
        """Return one Ranvier-node position as a unit-bearing quantity.

        `index` accepts integer node ordinals or the names `"first"`/`"proximal"`,
        `"center"`, and `"last"`/`"distal"`.
        """

        values = self.node_position_values(unit=unit)
        resolved = _resolve_node_selector(index, int(values.shape[0]))
        unit_label = units.unit_label(unit) or "micrometer"
        return units.Q_(float(values[resolved]), unit_label)


def _resolve_node_selector(index: int | str, count: int) -> int:
    if count <= 0:
        raise ValueError("myelinated axon has no Ranvier nodes.")

    if isinstance(index, str):
        key = index.strip().lower()
        if key in {"first", "proximal"}:
            return 0
        if key == "center":
            return count // 2
        if key in {"last", "distal"}:
            return count - 1
        raise ValueError(
            "node selector must be an integer, 'first'/'proximal', "
            "'center', or 'last'/'distal'."
        )

    resolved = int(index)
    if resolved < 0:
        resolved += count
    if resolved < 0 or resolved >= count:
        raise IndexError(f"node index {index} is out of range for {count} nodes.")
    return resolved


class MRG(Myelinated):
    """Concrete MRG myelinated axon model.

    `MRG` uses the reusable MRG-like node/MYSA/FLUT/STIN double-cable layout
    template plus the default MRG membrane assignment. Future myelinated
    models may reuse the same layout template with different membranes.
    """

    def __init__(
        self,
        *,
        diameter: length_t,
        nodes: int,
        length: length_t | None = None,
        compartments: SectionCompartments = 1,
        x_shift: length_t | None = None,
        membranes: SectionLayout | None = None,
        formulation: CableFormulation | None = CableFormulation.DOUBLE_CABLE,
        v_init: voltage_t = _DEFAULT_V_INIT,
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
    ) -> None:
        """Create an MRG myelinated axon.

        Parameters
        ----------
        diameter:
            Fiber diameter, with units.
        nodes:
            Number of Ranvier nodes to generate.
        length:
            Optional nominal axon length, with units. If omitted, the length
            is inferred from the MRG internode spacing and `nodes`.
        compartments:
            Either a single compartment count for every placed section, or a
            mapping from section name (`"node"`, `"MYSA"`, `"FLUT"`, `"STIN"`)
            to compartment count. Missing mapping entries default to one
            compartment.
        x_shift:
            Optional local phase shift of the MRG motif. It is the intrinsic
            distance from the axon start to the first node start, useful when
            importing NRV node-shifted fiber tables. It does not assign world
            coordinates to the axon.
        membranes:
            Optional section-to-membrane assignment. Defaults to MRG node and
            passive internode membranes.
        formulation:
            Cable formulation, normally `"double-cable"` for MRG.
        v_init:
            Initial intracellular membrane potential in millivolts.
        temperature:
            Model temperature in degrees Celsius.
        fit_all:
            If True, use polynomial morphology fits even for tabulated fiber
            diameters.
        mysa_length, node_length:
            MYSA and node lengths.
        axoplasmic_resistivity:
            Axoplasmic resistivity used by the MRG morphology equations.
        myelin_capacitance:
            Myelin capacitance density.
        myelin_conductance:
            Myelin conductance density.
        node_space, flut_space, stin_space:
            Periaxonal spaces around node/MYSA, FLUT, and STIN sections.
        """

        layout = mrg_like_layout(
            diameter=diameter,
            nodes=nodes,
            length=length,
            compartments=compartments,
            x_shift=x_shift,
            membranes=membranes,
            temperature=temperature,
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
        super().__init__(
            layout=layout,
            formulation=formulation,
            diameter=diameter,
            v_init=v_init,
            temperature=temperature,
        )


class _Gaines(Myelinated):
    """Shared constructor for the Gaines motor and sensory axon families."""

    _default_v_init: voltage_t
    _default_membranes = staticmethod(gaines_motor_membranes)

    def __init__(
        self,
        *,
        diameter: length_t,
        nodes: int,
        length: length_t | None = None,
        compartments: SectionCompartments = 1,
        x_shift: length_t | None = None,
        membranes: SectionLayout | None = None,
        formulation: CableFormulation | None = CableFormulation.DOUBLE_CABLE,
        v_init: voltage_t | None = None,
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
    ) -> None:
        if membranes is None:
            membranes = self._default_membranes(temperature=temperature)
        layout = mrg_like_layout(
            diameter=diameter,
            nodes=nodes,
            length=length,
            compartments=compartments,
            x_shift=x_shift,
            membranes=membranes,
            temperature=temperature,
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
        super().__init__(
            layout=layout,
            formulation=formulation,
            diameter=diameter,
            v_init=self._default_v_init if v_init is None else v_init,
            temperature=temperature,
        )


class GainesMotor(_Gaines):
    """Gaines et al. motor myelinated axon model."""

    _default_v_init = units.Q_(-85.9411, "millivolt")
    _default_membranes = staticmethod(gaines_motor_membranes)


class GainesSensory(_Gaines):
    """Gaines et al. sensory myelinated axon model."""

    _default_v_init = units.Q_(-79.3565, "millivolt")
    _default_membranes = staticmethod(gaines_sensory_membranes)


__all__ = [
    "GainesMotor",
    "GainesSensory",
    "MRG",
    "Myelinated",
    "MRGLikeDoubleCableTemplate",
    "mrg_like_layout",
    "mrg_like_length_from_nodes",
    "mrg_like_nodes_from_length",
]
