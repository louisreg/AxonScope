"""Descriptive axon, section layout, and template API."""

from axonfleet.axons.axon import Axon
from axonfleet.axons.diameters import (
    round_axon_diameter_um,
    round_axon_diameter_values_um,
)
from axonfleet.axons.formulation import CableFormulation
from axonfleet.axons.layout import Layout, LayoutElement
from axonfleet.axons.myelinated import GainesMotor, GainesSensory, MRG, Myelinated
from axonfleet.axons.section import PeriaxonalLayer, Section
from axonfleet.axons.templates import (
    MRGLikeDoubleCableGeometry,
    MRGLikeDoubleCableTemplate,
)
from axonfleet.axons.unmyelinated import (
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
    "CableFormulation",
    "Unmyelinated",
    "HodgkinHuxley",
    "RattayAberham",
    "Sundt",
    "Tigerholm",
    "Schild94",
    "Schild97",
    "Myelinated",
    "GainesMotor",
    "GainesSensory",
    "MRG",
    "MRGLikeDoubleCableGeometry",
    "MRGLikeDoubleCableTemplate",
]
