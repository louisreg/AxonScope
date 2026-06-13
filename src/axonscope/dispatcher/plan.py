from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from axonscope.axon_simulation import AxonSimulation, as_axon_simulation
from axonscope.axons.axon import Axon
from axonscope.solvers.axon_runtime import SolverAxon, build_solver_axon


_CableMode = Literal["single", "double"]


@dataclass(frozen=True)
class DispatchItem:
    """Normalized internal row for one input axon simulation."""

    index: int
    simulation: AxonSimulation
    solver_axon: SolverAxon


@dataclass(frozen=True)
class DispatchGroup:
    """A compatibility group selected by the dispatcher."""

    group_id: int
    items: tuple[DispatchItem, ...]
    signature: tuple[Any, ...]
    mode: _CableMode
    nx: int
    geometry_shared: bool = True

    @property
    def pool_indices(self) -> tuple[int, ...]:
        """Input-order indices represented by this group."""

        return tuple(item.index for item in self.items)

    @property
    def size(self) -> int:
        """Number of axons in this dispatch group."""

        return len(self.items)

    @property
    def batch_kind(self) -> str:
        """Return the intended batch family for this group."""

        if self.size < 2:
            return "scalar"
        prefix = "strict" if self.geometry_shared else "parameter"
        return f"{prefix}-{self.mode}-cable"

    @property
    def has_padding(self) -> bool:
        """Whether rows in this group require spatial padding to share one shape."""

        return any(
            int(item.solver_axon.n_compartments) != int(self.nx)
            for item in self.items
        )


@dataclass(frozen=True)
class DispatchPlan:
    """Dispatcher plan built from a pool of axons."""

    items: tuple[DispatchItem, ...]
    groups: tuple[DispatchGroup, ...]


def build_dispatch_plan(axons: Sequence[Axon | AxonSimulation]) -> DispatchPlan:
    """Normalize and group axon simulations before execution."""

    items = _normalize_dispatch_items(axons)
    groups_by_signature: dict[tuple[Any, ...], list[list[DispatchItem]]] = {}
    for item in items:
        signature = _dispatch_signature(item)
        compatible_groups = groups_by_signature.setdefault(signature, [])
        for group_items in compatible_groups:
            candidate = [*group_items, item]
            if _items_can_share_batch_runtime(candidate):
                group_items.append(item)
                break
        else:
            compatible_groups.append([item])

    groups_list: list[DispatchGroup] = []
    for signature, signature_groups in groups_by_signature.items():
        for group_items in signature_groups:
            groups_list.append(
                DispatchGroup(
                    group_id=len(groups_list),
                    items=tuple(group_items),
                    signature=signature,
                    mode=_resolve_mode(group_items[0].solver_axon),
                    nx=max(int(item.solver_axon.n_compartments) for item in group_items),
                    geometry_shared=_group_has_shared_geometry(group_items),
                )
            )
    groups = tuple(groups_list)
    return DispatchPlan(items=items, groups=groups)


def _normalize_dispatch_items(axons: Sequence[Axon | AxonSimulation]) -> tuple[DispatchItem, ...]:
    """Validate public pool items and preserve input order."""

    items = []
    for index, axon in enumerate(axons):
        simulation = as_axon_simulation(axon)
        items.append(
            DispatchItem(
                index=index,
                simulation=simulation,
                solver_axon=build_solver_axon(simulation),
            )
        )
    return tuple(items)


def _resolve_mode(axon: SolverAxon) -> _CableMode:
    """Return the cable mode implied by an axon description."""

    if axon.formulation == "double-cable":
        return "double"
    return "single"


def _dispatch_signature(item: DispatchItem) -> tuple[Any, ...]:
    """Return the compatibility signature used for grouping."""

    simulation = item.simulation
    solver_axon = item.solver_axon
    mode = _resolve_mode(solver_axon)
    membrane_signature = (
        _double_cable_membrane_family_signature(solver_axon)
        if mode == "double"
        else _axon_membrane_signature(solver_axon)
    )
    nx_signature = None if mode == "double" else int(solver_axon.n_compartments)
    return (
        mode,
        solver_axon.dtype.str,
        membrane_signature,
        nx_signature,
        float(getattr(simulation, "v_init", 0.0)),
        float(getattr(simulation, "Veinit", 0.0)),
        float(getattr(simulation, "temperature", 0.0)),
    )


def _items_can_share_batch_runtime(items: Sequence[DispatchItem]) -> bool:
    """Return whether items can share one compiled batch runtime."""

    if not items:
        return False
    mode = _resolve_mode(items[0].solver_axon)
    if mode == "single":
        nx_values = {int(item.solver_axon.n_compartments) for item in items}
        membrane_signatures = {
            _axon_membrane_signature(item.solver_axon)
            for item in items
        }
        return len(nx_values) == 1 and len(membrane_signatures) == 1
    return _double_cable_membranes_are_padding_compatible(
        item.solver_axon for item in items
    )


def _double_cable_membranes_are_padding_compatible(
    axons: Iterable[SolverAxon],
) -> bool:
    """Return whether shorter double-cable rows match a longer membrane prefix."""

    signatures = tuple(_axon_membrane_family_sequence(axon) for axon in axons)
    if not signatures:
        return False
    longest = max(signatures, key=len)
    return all(longest[: len(signature)] == signature for signature in signatures)


def _double_cable_membrane_family_signature(axon: SolverAxon) -> Any:
    """Return a coarse membrane signature used before padding compatibility."""

    unique: list[Any] = []
    for signature in _axon_membrane_family_sequence(axon):
        if signature not in unique:
            unique.append(signature)
    return tuple(unique)


def _model_family_signature(model: Any) -> Any:
    """Return a structural membrane signature that ignores numeric parameters."""

    kind = getattr(model, "kind", None)
    if kind is not None:
        component_families = tuple(
            _model_family_signature(component)
            for component in getattr(model, "components", ())
        )
        return ("membrane", kind, component_families)
    implementation = getattr(model, "_implementation", None)
    if implementation is not None:
        return _model_family_signature(implementation)
    return (model.__class__.__module__, model.__class__.__qualname__)


def _axon_membrane_family_sequence(axon: SolverAxon) -> Any:
    """Return per-compartment membrane families without parameter values."""

    return tuple(_model_family_signature(model) for model in axon.membrane_models)


def _group_has_shared_geometry(items: Sequence[DispatchItem]) -> bool:
    """Return whether all rows share exact cable/periaxonal arrays."""

    signatures = {_axon_cable_signature(item.solver_axon) for item in items}
    return len(signatures) == 1


def _model_signature(model: Any) -> Any:
    """Return a stable-enough signature for a membrane model description."""

    static_signature = getattr(model, "_static_signature", None)
    if callable(static_signature):
        return static_signature()
    return repr(model)


def _axon_membrane_signature(axon: SolverAxon) -> Any:
    """Return the membrane component of an axon compatibility signature."""

    return tuple(_model_signature(model) for model in axon.membrane_models)


def _axon_cable_signature(axon: SolverAxon) -> Any:
    """Return the shared cable/periaxonal arrays required by batch kernels."""

    return (
        _array_signature(axon.x_um),
        _array_signature(axon.compartment_lengths_um),
        _array_signature(axon.diam_um),
        _array_signature(axon.Ra_ohm_cm),
        _array_signature(axon.Cm_uF_cm2),
        _array_signature(axon.xraxial_MOhm_per_cm),
        _array_signature(axon.xg_S_cm2),
        _array_signature(axon.xc_uF_cm2),
    )


def _array_signature(values: Any) -> tuple[tuple[int, ...], str, str]:
    """Return a compact hashable signature for numeric arrays."""

    arr = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha1(arr.view(np.uint8)).hexdigest()
    return arr.shape, arr.dtype.str, digest


__all__ = [
    "DispatchPlan",
    "build_dispatch_plan",
]
