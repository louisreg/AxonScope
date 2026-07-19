"""Descriptive spatial layout of axon sections.

`layout.py` owns the user-facing spatial description:

- `LayoutElement`: one placed section with a length and compartment count.
- `Layout`: an ordered sequence of placed sections.

`Section` describes what a piece of cable is. `LayoutElement` describes where
that section lives in the one-dimensional axon and how many numerical
compartments it is split into. Flattened arrays are produced separately in
`axonscope.axons.flattened`.

This module does not own solver arrays, membrane compilation, stimulation
protocols, electrode placement, or solver runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from axonscope.axons.section import Section
from axonscope.utils import units
from axonscope.utils.validation import normalize_positive_int, require_positive


_REPEAT_RTOL = 1e-9
_REPEAT_ATOL = 1e-9


def _normalize_compartments(value: int | Sequence[int], *, count: int) -> tuple[int, ...]:
    if isinstance(value, Integral):
        return tuple(
            normalize_positive_int(value, name="compartments")
            for _ in range(count)
        )
    values = tuple(normalize_positive_int(item, name="compartments") for item in value)
    if len(values) != count:
        raise ValueError(f"compartments must have {count} entries, got {len(values)}.")
    return values


def _length_array_um(value: units.length_t, *, name: str, count: int) -> np.ndarray:
    lengths_um = units.require_length_array_um(value, name=name, dtype=np.float64)
    if lengths_um.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional length array.")
    if lengths_um.shape[0] != count:
        raise ValueError(f"{name} must have {count} entries, got {lengths_um.shape[0]}.")
    if np.any(lengths_um <= 0.0):
        raise ValueError(f"{name} entries must be positive.")
    return lengths_um


def _center_control_length_sum_um(centers_um: np.ndarray) -> float:
    spacings = np.diff(centers_um.astype(np.float64))
    if spacings.shape[0] == 0:
        raise ValueError("centers must contain at least two points.")
    return float(np.sum(spacings) + 0.5 * (spacings[0] + spacings[-1]))


def _normalize_sections(sections: Sequence[Section]) -> tuple[Section, ...]:
    frozen = tuple(sections)
    if not frozen:
        raise ValueError("sections cannot be empty.")
    if any(not isinstance(section, Section) for section in frozen):
        raise TypeError("sections must contain only Section objects.")
    return frozen


def _repeat_count_from_lengths(*, lengths_um: float, motif_length_um: float) -> int:
    repeat_float = lengths_um / motif_length_um
    repeat_int = int(round(repeat_float))
    if repeat_int < 1 or not np.isclose(
        repeat_float,
        repeat_int,
        rtol=_REPEAT_RTOL,
        atol=_REPEAT_ATOL,
    ):
        raise ValueError(
            "lengths must be an integer multiple of sum(section_lengths); "
            f"got lengths={lengths_um} um and motif length={motif_length_um} um."
        )
    return repeat_int


def _phase_shifted_motif_elements(
    *,
    sections: Sequence[Section],
    section_lengths_um: np.ndarray,
    compartment_counts: Sequence[int],
    total_length_um: float,
    phase_shift_um: float,
) -> list["LayoutElement"]:
    motif_length_um = float(np.sum(section_lengths_um))
    phase_um = float(phase_shift_um) % motif_length_um
    start_coordinate_um = (motif_length_um - phase_um) % motif_length_um
    boundaries_um = np.concatenate(
        [np.asarray([0.0], dtype=np.float64), np.cumsum(section_lengths_um)]
    )
    if np.isclose(start_coordinate_um, motif_length_um, rtol=0.0, atol=1e-9):
        start_coordinate_um = 0.0
    start_index = int(np.searchsorted(boundaries_um[1:], start_coordinate_um, side="right"))

    elements: list[LayoutElement] = []
    cursor_um = 0.0
    index = start_index
    first = True
    while cursor_um < total_length_um - 1e-9:
        base_index = index % len(sections)
        if first:
            natural_length_um = float(boundaries_um[base_index + 1] - start_coordinate_um)
            if natural_length_um <= 1e-9:
                index += 1
                continue
            first = False
        else:
            natural_length_um = float(section_lengths_um[base_index])
        remaining_um = float(total_length_um) - cursor_um
        length_um = min(natural_length_um, remaining_um)
        if length_um > 1e-9:
            elements.append(
                LayoutElement(
                    sections[base_index],
                    length=units.Q_(length_um, "micrometer"),
                    compartments=compartment_counts[base_index],
                )
            )
            cursor_um += length_um
        index += 1
    return elements


@dataclass(frozen=True, init=False)
class LayoutElement:
    """One section placed along the axon.

    Parameters
    ----------
    section:
        Descriptive section prototype.
    length:
        Spatial length occupied by this section.
    compartments:
        Number of numerical compartments used inside this placed section.
    """

    section: Section
    length_um: float
    compartments: int

    def __init__(
        self,
        section: Section,
        *,
        length: units.length_t,
        compartments: int = 1,
    ) -> None:
        if not isinstance(section, Section):
            raise TypeError("section must be a Section.")
        length_um = require_positive(units.require_length_um(length, name="length"), name="length")
        object.__setattr__(self, "section", section)
        object.__setattr__(self, "length_um", length_um)
        object.__setattr__(
            self,
            "compartments",
            normalize_positive_int(compartments, name="compartments"),
        )


class Layout:
    """Ordered sequence of placed axon sections.

    A layout is still descriptive: it knows which `Section` occupies each
    interval, how long that interval is, and how many compartments represent it.
    Per-compartment arrays are derived later by `flatten_layout`.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_template_frozen", False):
            raise AttributeError(
                "Layout descriptions are immutable; use with_x_shift() or build "
                "a new Layout."
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        elements: Sequence[LayoutElement],
        *,
        total_length: units.length_t | None = None,
        x_centers: units.length_t | None = None,
        x_shift: units.length_t | None = None,
    ) -> None:
        """Create a descriptive spatial layout.

        Parameters
        ----------
        elements:
            Ordered placed sections.
        total_length:
            Optional nominal total length. When omitted, the sum of element
            lengths is used.
        x_centers:
            Advanced single-section layout: explicit compartment-center
            positions. Normal layouts should use element `compartments`.
        x_shift:
            Optional translation of the local compartment centers along the
            axon's x-axis. This is an intrinsic one-dimensional layout shift,
            not an electrode/world-coordinate placement.
        """

        frozen = tuple(elements)
        if not frozen:
            raise ValueError("Layout requires at least one element.")
        if any(not isinstance(element, LayoutElement) for element in frozen):
            raise TypeError("Layout elements must contain only LayoutElement objects.")
        x_centers_um = None
        if x_centers is not None:
            if len(frozen) != 1:
                raise ValueError("x_centers requires a single layout element.")
            x_centers_um = units.require_length_array_um(
                x_centers,
                name="x_centers",
                dtype=np.float32,
            )
            if x_centers_um.ndim != 1 or x_centers_um.shape[0] < 2:
                raise ValueError(
                    "x_centers must be a one-dimensional length array with at least two points."
                )
            if not np.all(np.diff(x_centers_um.astype(float)) > 0.0):
                raise ValueError("x_centers must be strictly increasing.")
            x_centers_um = np.array(x_centers_um, copy=True)
            x_centers_um.setflags(write=False)
            control_length_um = _center_control_length_sum_um(x_centers_um)
            if frozen[0].compartments != int(x_centers_um.shape[0]) or not np.isclose(
                frozen[0].length_um,
                control_length_um,
            ):
                frozen = (
                    LayoutElement(
                        frozen[0].section,
                        length=units.Q_(control_length_um, "micrometer"),
                        compartments=int(x_centers_um.shape[0]),
                    ),
                )
        self.elements = frozen
        self.sections = tuple(element.section for element in frozen)
        self.x_centers_um = x_centers_um
        self.x_shift_um = (
            0.0
            if x_shift is None
            else units.require_length_um(x_shift, name="x_shift")
        )
        self._total_length_um = (
            None
            if total_length is None
            else require_positive(
                units.require_length_um(total_length, name="total_length"),
                name="total_length",
            )
        )
        self._flattened_cache = None
        self._translation_template_token = object()
        self._template_frozen = True

    @classmethod
    def single_uniform(
        cls,
        section: Section,
        *,
        length: units.length_t,
        compartments: int = 1,
        x_shift: units.length_t | None = None,
    ) -> "Layout":
        """Build one section with uniformly spaced compartments."""

        return cls(
            [LayoutElement(section, length=length, compartments=compartments)],
            x_shift=x_shift,
        )

    @classmethod
    def single_non_uniform(
        cls,
        section: Section,
        *,
        x: units.length_t,
        x_shift: units.length_t | None = None,
    ) -> "Layout":
        """Build one section from explicit compartment-center positions."""

        centers_um = units.require_length_array_um(x, name="x", dtype=np.float32)
        if centers_um.ndim != 1 or centers_um.shape[0] < 2:
            raise ValueError("x must be a one-dimensional length array with at least two points.")
        if not np.all(np.diff(centers_um.astype(float)) > 0.0):
            raise ValueError("x must be strictly increasing.")
        length_um = _center_control_length_sum_um(centers_um)
        return cls(
            [
                LayoutElement(
                    section,
                    length=units.Q_(length_um, "micrometer"),
                    compartments=int(centers_um.shape[0]),
                )
            ],
            x_centers=units.Q_(centers_um, "micrometer"),
            x_shift=x_shift,
        )

    @classmethod
    def sequence(
        cls,
        sections: Sequence[Section],
        *,
        section_lengths: units.length_t,
        compartments: int | Sequence[int] = 1,
        lengths: units.length_t,
        phase_shift: units.length_t | None = None,
        x_shift: units.length_t | None = None,
    ) -> "Layout":
        """Build a repeated section motif over a requested total length.

        `section_lengths` gives the length of one motif, one entry per
        `sections` item. `lengths` is the requested full layout length.
        Without `phase_shift`, `lengths` must be an integer multiple of the
        motif length. With `phase_shift`, the motif is cyclically rotated and
        cropped to `lengths`; this is useful for node-phase examples.
        `compartments` can be one count applied to every section type, or one
        count per section type.
        """

        base_sections = _normalize_sections(sections)
        section_lengths_um = _length_array_um(
            section_lengths,
            name="section_lengths",
            count=len(base_sections),
        )
        motif_length_um = float(np.sum(section_lengths_um))
        lengths_um = require_positive(
            units.require_length_um(lengths, name="lengths"),
            name="lengths",
        )
        compartment_counts = _normalize_compartments(compartments, count=len(base_sections))
        if phase_shift is not None:
            phase_shift_um = units.require_length_um(phase_shift, name="phase_shift")
            return cls(
                _phase_shifted_motif_elements(
                    sections=base_sections,
                    section_lengths_um=section_lengths_um,
                    compartment_counts=compartment_counts,
                    total_length_um=lengths_um,
                    phase_shift_um=phase_shift_um,
                ),
                total_length=units.Q_(lengths_um, "micrometer"),
                x_shift=x_shift,
            )
        repeat_count = _repeat_count_from_lengths(
            lengths_um=lengths_um,
            motif_length_um=motif_length_um,
        )
        section_tuple = tuple(section for _ in range(repeat_count) for section in base_sections)
        length_tuple = tuple(float(value) for _ in range(repeat_count) for value in section_lengths_um)
        compartment_tuple = compartment_counts * repeat_count
        return cls(
            [
                LayoutElement(
                    section,
                    length=units.Q_(length_um, "micrometer"),
                    compartments=compartment_count,
                )
                for section, length_um, compartment_count in zip(
                    section_tuple,
                    length_tuple,
                    compartment_tuple,
                    strict=True,
                )
            ],
            total_length=units.Q_(lengths_um, "micrometer"),
            x_shift=x_shift,
        )

    @property
    def length_um(self) -> float:
        """Nominal layout length in micrometers."""

        if self._total_length_um is not None:
            return float(self._total_length_um)
        return float(sum(element.length_um for element in self.elements))

    @property
    def length(self) -> float:
        """Nominal layout length in micrometers."""

        return self.length_um

    @property
    def compartments(self) -> int:
        """Total number of numerical compartments."""

        return int(sum(element.compartments for element in self.elements))

    @property
    def n_compartments(self) -> int:
        """Total number of numerical compartments."""

        return self.compartments

    def with_x_shift(self, x_shift: units.length_t | None) -> "Layout":
        """Return the same layout description with a different local x-shift."""

        shifted = Layout(
            self.elements,
            total_length=(
                None
                if self._total_length_um is None
                else units.Q_(self._total_length_um, "micrometer")
            ),
            x_centers=(
                None
                if self.x_centers_um is None
                else units.Q_(self.x_centers_um, "micrometer")
            ),
            x_shift=x_shift,
        )
        object.__setattr__(
            shifted,
            "_translation_template_token",
            self._translation_template_token,
        )
        return shifted

    def _flattened(self):
        """Return solver-facing per-compartment arrays derived from this layout."""

        from axonscope.axons.flattened import flatten_layout

        return flatten_layout(self)

    def position_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return compartment-center positions as plain values in `unit`."""

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(
            units.Q_(self._flattened().x_um, "micrometer"),
            unit_label,
            dtype=float,
        )

    def compartment_position(self, index: int, *, unit: Any = "micrometer") -> units.length_t:
        """Return one compartment-center position as a unit-bearing quantity."""

        positions = self.position_values(unit=unit)
        resolved_index = _resolve_compartment_index(index, positions.shape[0])
        unit_label = units.unit_label(unit) or "micrometer"
        return units.Q_(float(positions[resolved_index]), unit_label)

    def diameter_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return compartment diameters as plain values in `unit`."""

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(
            units.Q_(self._flattened().diam_um, "micrometer"),
            unit_label,
            dtype=float,
        )

    def compartment_length_values(self, *, unit: Any = "micrometer") -> np.ndarray:
        """Return compartment control-volume lengths as plain values in `unit`."""

        unit_label = units.unit_label(unit) or "micrometer"
        return units.to_array(
            units.Q_(self._flattened().lengths_um, "micrometer"),
            unit_label,
            dtype=float,
        )

    def plot(self, ax=None, **kwargs):
        """Plot sections and compartments in this layout."""

        from axonscope.axons.plotting import plot_layout

        return plot_layout(self, ax=ax, **kwargs)


def _resolve_compartment_index(index: int, count: int) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved += int(count)
    if resolved < 0 or resolved >= int(count):
        raise IndexError(
            f"compartment index {index} is out of range for {count} compartments."
        )
    return resolved


__all__ = ["Layout", "LayoutElement"]
