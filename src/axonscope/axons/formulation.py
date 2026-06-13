"""Cable-formulation helpers for descriptive axon layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, cast

Formulation: TypeAlias = Literal["single-cable", "double-cable"]
_FORMULATIONS: frozenset[str] = frozenset({"single-cable", "double-cable"})

if TYPE_CHECKING:
    from axonscope.axons.flattened import FlattenedLayout
    from axonscope.axons.layout import Layout


def normalize_formulation(value: Formulation | str | None) -> Formulation | None:
    """Validate and normalize a cable formulation name."""

    if value is None:
        return None
    text = str(value)
    if text not in _FORMULATIONS:
        choices = ", ".join(sorted(_FORMULATIONS))
        raise ValueError(f"Unknown axon formulation {value!r}; expected one of: {choices}.")
    return cast(Formulation, text)


def infer_layout_formulation(layout: "Layout") -> Formulation:
    """Infer the cable formulation from placed section descriptions."""

    has_periaxonal = tuple(element.section.periaxonal is not None for element in layout.elements)
    if all(has_periaxonal):
        return "double-cable"
    if not any(has_periaxonal):
        return "single-cable"
    raise ValueError(
        "Mixed single- and double-cable sections are not supported yet. "
        "Either provide periaxonal data on every section, or on none."
    )


def _validate_layout_formulation(layout: "Layout", formulation: Formulation) -> None:
    if formulation == "double-cable" and any(
        element.section.periaxonal is None for element in layout.elements
    ):
        raise ValueError("Double-cable axons require periaxonal data on every section.")


def resolve_layout_formulation(
    layout: "Layout",
    formulation: Formulation | str | None,
) -> Formulation:
    """Return explicit or inferred cable formulation for a descriptive layout."""

    normalized = normalize_formulation(formulation)
    if normalized is None:
        return infer_layout_formulation(layout)
    _validate_layout_formulation(layout, normalized)
    return normalized


def infer_formulation(flat: "FlattenedLayout") -> Formulation:
    """Infer the cable formulation from flattened section data."""

    has_periaxonal = tuple(layer is not None for layer in flat.periaxonal_layers)
    if all(has_periaxonal):
        return "double-cable"
    if not any(has_periaxonal):
        return "single-cable"
    raise ValueError(
        "Mixed single- and double-cable sections are not supported yet. "
        "Either provide periaxonal data on every section, or on none."
    )


def _validate_formulation(flat: "FlattenedLayout", formulation: Formulation) -> None:
    if formulation == "double-cable" and any(layer is None for layer in flat.periaxonal_layers):
        raise ValueError("Double-cable axons require periaxonal data on every section.")


def resolve_formulation(
    flat: "FlattenedLayout",
    formulation: Formulation | str | None,
) -> Formulation:
    """Return explicit or inferred cable formulation for a flattened layout."""

    normalized = normalize_formulation(formulation)
    if normalized is None:
        return infer_formulation(flat)
    _validate_formulation(flat, normalized)
    return normalized


__all__ = [
    "Formulation",
    "infer_formulation",
    "infer_layout_formulation",
    "normalize_formulation",
    "resolve_formulation",
    "resolve_layout_formulation",
]
