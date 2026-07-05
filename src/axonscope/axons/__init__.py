"""Descriptive axon, section layout, and template API."""

from axonscope.axons.axon import Axon
from axonscope.axons.diameters import (
    round_axon_diameter_um,
    round_axon_diameter_values_um,
)
from axonscope.axons.flattened import FlattenedLayout, flatten_layout
from axonscope.axons.formulation import (
    CableFormulation,
    Formulation,
    infer_formulation,
    resolve_formulation,
)
from axonscope.axons.layout import Layout, LayoutElement
from axonscope.axons.myelinated import MRG, Myelinated
from axonscope.axons.plotting import plot_layout
from axonscope.axons.section import PeriaxonalLayer, Section
from axonscope.axons.templates import (
    MRGLikeDoubleCableGeometry,
    MRGLikeDoubleCableTemplate,
    SectionCompartments,
    build_mrg_like_geometry,
    default_mrg_like_membranes,
    layout_from_mrg_like_geometry,
    mrg_like_layout,
    mrg_like_length_from_nodes,
    mrg_like_node_spacing,
    mrg_like_nodes_from_length,
    mrg_like_section_sequence,
)
from axonscope.axons.unmyelinated import (
    HodgkinHuxley,
    RattayAberham,
    Schild94,
    Schild97,
    Sundt,
    Tigerholm,
    Unmyelinated,
)

__all__ = [
    "Axon",
    "round_axon_diameter_um",
    "round_axon_diameter_values_um",
    "Section",
    "PeriaxonalLayer",
    "Layout",
    "LayoutElement",
    "FlattenedLayout",
    "flatten_layout",
    "infer_formulation",
    "resolve_formulation",
    "plot_layout",
    "CableFormulation",
    "Formulation",
    "Unmyelinated",
    "HodgkinHuxley",
    "RattayAberham",
    "Sundt",
    "Tigerholm",
    "Schild94",
    "Schild97",
    "Myelinated",
    "MRG",
    "MRGLikeDoubleCableGeometry",
    "MRGLikeDoubleCableTemplate",
    "SectionCompartments",
    "build_mrg_like_geometry",
    "default_mrg_like_membranes",
    "layout_from_mrg_like_geometry",
    "mrg_like_layout",
    "mrg_like_length_from_nodes",
    "mrg_like_node_spacing",
    "mrg_like_nodes_from_length",
    "mrg_like_section_sequence",
]
