"""Derived per-compartment geometry from descriptive layouts.

`flattened.py` is the bridge between user-facing layout descriptions and
solver-facing arrays. It should not be used to author an axon model; build
`Section` and `Layout` objects first, then derive a `FlattenedLayout` when code
needs one value per numerical compartment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from axonscope.axons.section import PeriaxonalLayer
from axonscope.membranes import MembraneModel

if TYPE_CHECKING:
    from axonscope.axons.layout import Layout


def _control_lengths_um(x_um: np.ndarray, *, total_length_um: float | None = None) -> np.ndarray:
    x = np.asarray(x_um, dtype=np.float64)
    dx = np.diff(x)
    if dx.shape[0] == 0:
        return np.ones_like(x, dtype=np.float64)
    mean_dx = float(np.mean(dx))
    if total_length_um is not None and np.allclose(dx, mean_dx, rtol=1e-5, atol=1e-6):
        return np.full(x.shape, float(total_length_um) / float(x.shape[0]), dtype=np.float64)
    lengths = np.zeros_like(x, dtype=np.float64)
    lengths[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    lengths[0] = dx[0]
    lengths[-1] = dx[-1]
    return lengths


@dataclass(frozen=True)
class FlattenedLayout:
    """Canonical per-compartment arrays produced from a `Layout`.

    This object is a derived representation, not a user-facing modeling object.
    It keeps solver-bound geometry and section metadata in canonical float
    arrays.
    """

    x_um: np.ndarray
    edges_um: np.ndarray
    lengths_um: np.ndarray
    diam_um: np.ndarray
    Ra_ohm_cm: np.ndarray
    Cm_uF_cm2: np.ndarray
    membrane_models: tuple[MembraneModel, ...]
    section_names: tuple[str, ...]
    section_indices: np.ndarray
    section_tags: tuple[tuple[str, ...], ...]
    periaxonal_layers: tuple[PeriaxonalLayer | None, ...]
    total_length_um: float | None = None

    def __post_init__(self) -> None:
        x_um = np.asarray(self.x_um, dtype=np.float32)
        edges_um = np.asarray(self.edges_um, dtype=np.float32)
        lengths_um = np.asarray(self.lengths_um, dtype=np.float32)
        diam_um = np.asarray(self.diam_um, dtype=np.float32)
        Ra_ohm_cm = np.asarray(self.Ra_ohm_cm, dtype=np.float32)
        Cm_uF_cm2 = np.asarray(self.Cm_uF_cm2, dtype=np.float32)
        section_indices = np.asarray(self.section_indices, dtype=np.int32)
        membrane_models = tuple(self.membrane_models)
        section_names = tuple(self.section_names)
        section_tags = tuple(tuple(tags) for tags in self.section_tags)
        periaxonal_layers = tuple(self.periaxonal_layers)

        arrays = (x_um, lengths_um, diam_um, Ra_ohm_cm, Cm_uF_cm2, section_indices)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("FlattenedLayout arrays must be one-dimensional.")
        n = int(x_um.shape[0])
        if n < 1:
            raise ValueError("FlattenedLayout requires at least one compartment.")
        if any(array.shape != (n,) for array in arrays):
            raise ValueError("FlattenedLayout arrays must all have the same length.")
        if edges_um.ndim != 1 or edges_um.shape != (n + 1,):
            raise ValueError("edges_um must have one more entry than compartment arrays.")
        if len(membrane_models) != n:
            raise ValueError("membrane_models must have one entry per compartment.")
        if len(section_names) != n:
            raise ValueError("section_names must have one entry per compartment.")
        if len(section_tags) != n:
            raise ValueError("section_tags must have one entry per compartment.")
        if len(periaxonal_layers) != n:
            raise ValueError("periaxonal_layers must have one entry per compartment.")
        if n > 1 and not np.all(np.diff(x_um.astype(float)) > 0.0):
            raise ValueError("x_um must be strictly increasing.")
        if not np.all(np.diff(edges_um.astype(float)) > 0.0):
            raise ValueError("edges_um must be strictly increasing.")
        if np.any(
            (x_um.astype(float) <= edges_um[:-1].astype(float))
            | (x_um.astype(float) >= edges_um[1:].astype(float))
        ):
            raise ValueError("x_um entries must lie inside their compartment edges.")
        if np.any(lengths_um.astype(float) <= 0.0):
            raise ValueError("lengths_um must be strictly positive.")
        if np.any(diam_um.astype(float) <= 0.0):
            raise ValueError("diam_um must be strictly positive.")
        if np.any(Ra_ohm_cm.astype(float) <= 0.0):
            raise ValueError("Ra_ohm_cm must be strictly positive.")
        if np.any(Cm_uF_cm2.astype(float) <= 0.0):
            raise ValueError("Cm_uF_cm2 must be strictly positive.")

        object.__setattr__(self, "x_um", x_um)
        object.__setattr__(self, "edges_um", edges_um)
        object.__setattr__(self, "lengths_um", lengths_um)
        object.__setattr__(self, "diam_um", diam_um)
        object.__setattr__(self, "Ra_ohm_cm", Ra_ohm_cm)
        object.__setattr__(self, "Cm_uF_cm2", Cm_uF_cm2)
        object.__setattr__(self, "section_indices", section_indices)
        object.__setattr__(self, "membrane_models", membrane_models)
        object.__setattr__(self, "section_names", section_names)
        object.__setattr__(self, "section_tags", section_tags)
        object.__setattr__(self, "periaxonal_layers", periaxonal_layers)

    @property
    def Nx(self) -> int:
        """Number of flattened numerical compartments."""

        return int(self.x_um.shape[0])

    @property
    def length_um(self) -> float:
        """Total flattened layout length in micrometers."""

        if self.total_length_um is not None:
            return float(self.total_length_um)
        return float(np.sum(self.lengths_um))


def flatten_layout(layout: "Layout") -> FlattenedLayout:
    """Derive canonical per-compartment arrays from a descriptive layout."""

    if layout.x_centers_um is not None:
        return _flatten_x_centers(layout)
    return _flatten_elements(layout)


def _flatten_x_centers(layout: "Layout") -> FlattenedLayout:
    element = layout.elements[0]
    section = element.section
    x_um = np.asarray(layout.x_centers_um, dtype=np.float32)
    lengths_um = _control_lengths_um(x_um, total_length_um=layout.length_um).astype(np.float32)
    if x_um.shape[0] == 1:
        half_width = 0.5 * float(lengths_um[0])
        edges_um = np.asarray(
            [float(x_um[0]) - half_width, float(x_um[0]) + half_width],
            dtype=np.float32,
        )
    else:
        edges_um = np.empty((x_um.shape[0] + 1,), dtype=np.float32)
        edges_um[1:-1] = 0.5 * (x_um[:-1] + x_um[1:])
        edges_um[0] = float(x_um[0]) - 0.5 * float(x_um[1] - x_um[0])
        edges_um[-1] = float(x_um[-1]) + 0.5 * float(x_um[-1] - x_um[-2])
    count = int(x_um.shape[0])
    return FlattenedLayout(
        x_um=x_um,
        edges_um=edges_um,
        lengths_um=lengths_um,
        diam_um=np.full((count,), section.diameter_um, dtype=np.float32),
        Ra_ohm_cm=np.full((count,), section.Ra_ohm_cm, dtype=np.float32),
        Cm_uF_cm2=np.full((count,), section.Cm_uF_cm2, dtype=np.float32),
        membrane_models=tuple(section.membrane for _ in range(count)),
        section_names=tuple(section.name for _ in range(count)),
        section_indices=np.zeros((count,), dtype=np.int32),
        section_tags=tuple(section.tags for _ in range(count)),
        periaxonal_layers=tuple(section.periaxonal for _ in range(count)),
        total_length_um=layout.length_um,
    )


def _flatten_elements(layout: "Layout") -> FlattenedLayout:
    x_positions: list[float] = []
    edges: list[float] = [0.0]
    lengths: list[float] = []
    diam: list[float] = []
    Ra: list[float] = []
    Cm: list[float] = []
    membranes: list[MembraneModel] = []
    names: list[str] = []
    section_indices: list[int] = []
    tags: list[tuple[str, ...]] = []
    periaxonal: list[PeriaxonalLayer | None] = []

    cursor = 0.0
    for section_index, element in enumerate(layout.elements):
        section = element.section
        compartment_length = element.length_um / element.compartments
        centers = cursor + (np.arange(element.compartments, dtype=np.float64) + 0.5) * compartment_length
        compartment_edges = cursor + np.arange(1, element.compartments + 1, dtype=np.float64) * compartment_length
        x_positions.extend(float(value) for value in centers)
        edges.extend(float(value) for value in compartment_edges)
        lengths.extend(float(compartment_length) for _ in range(element.compartments))
        diam.extend(float(section.diameter_um) for _ in range(element.compartments))
        Ra.extend(float(section.Ra_ohm_cm) for _ in range(element.compartments))
        Cm.extend(float(section.Cm_uF_cm2) for _ in range(element.compartments))
        membranes.extend(section.membrane for _ in range(element.compartments))
        names.extend(section.name for _ in range(element.compartments))
        section_indices.extend(section_index for _ in range(element.compartments))
        tags.extend(section.tags for _ in range(element.compartments))
        periaxonal.extend(section.periaxonal for _ in range(element.compartments))
        cursor += element.length_um

    return FlattenedLayout(
        x_um=np.asarray(x_positions, dtype=np.float32),
        edges_um=np.asarray(edges, dtype=np.float32),
        lengths_um=np.asarray(lengths, dtype=np.float32),
        diam_um=np.asarray(diam, dtype=np.float32),
        Ra_ohm_cm=np.asarray(Ra, dtype=np.float32),
        Cm_uF_cm2=np.asarray(Cm, dtype=np.float32),
        membrane_models=tuple(membranes),
        section_names=tuple(names),
        section_indices=np.asarray(section_indices, dtype=np.int32),
        section_tags=tuple(tags),
        periaxonal_layers=tuple(periaxonal),
        total_length_um=layout.length_um,
    )


__all__ = ["FlattenedLayout", "flatten_layout"]
