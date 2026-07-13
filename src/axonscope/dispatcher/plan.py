from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.axons.axon import Axon
from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.runtime.solver_axon import SolverAxon, build_solver_axon


_CableMode = Literal["single", "double"]


@dataclass(frozen=True)
class DispatchItem:
    """Normalized internal row for one input axon simulation."""

    index: int
    simulation: AxonInstance
    solver_axon: SolverAxon
    signature: tuple[Any, ...]
    mode: _CableMode
    membrane_signature: tuple[Any, ...]
    cable_signature: tuple[Any, ...]


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
        """Return the intended batch route for this group."""

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


@dataclass
class _PendingDispatchGroup:
    """Mutable grouping state used while building a dispatch plan."""

    items: list[DispatchItem]

    @classmethod
    def from_item(cls, item: DispatchItem) -> "_PendingDispatchGroup":
        return cls(items=[item])

    def can_accept(self, item: DispatchItem) -> bool:
        return item.signature == self.items[0].signature

    def append(self, item: DispatchItem) -> None:
        self.items.append(item)


@dataclass(frozen=True)
class _SolverDispatchMetadata:
    mode: _CableMode
    dtype_str: str
    nx_signature: int | None
    membrane_signature: tuple[Any, ...]
    membrane_group_signature: tuple[Any, ...]
    membrane_structure_sequence: tuple[Any, ...]
    cable_signature: tuple[Any, ...]


_DISPATCH_PLAN_CACHE: OrderedDict[tuple[Any, ...], DispatchPlan] = OrderedDict()
_DISPATCH_PLAN_CACHE_MAX_SIZE = 64


def build_dispatch_plan(axons: Sequence[Axon | AxonInstance]) -> DispatchPlan:
    """Normalize and group axon simulations before execution."""

    simulations = tuple(as_axon_instance(axon) for axon in axons)
    with benchmark_span("dispatch.build_plan", pool_size=len(simulations)):
        cache_key = _dispatch_plan_cache_key(simulations)
        cached = _DISPATCH_PLAN_CACHE.get(cache_key)
        if cached is not None:
            _DISPATCH_PLAN_CACHE.move_to_end(cache_key)
            record_benchmark_metadata(
                dispatch_plan_cache="hit",
                item_count=len(cached.items),
                group_count=len(cached.groups),
                group_sizes=[group.size for group in cached.groups],
                group_modes=[group.mode for group in cached.groups],
                group_nx=[group.nx for group in cached.groups],
            )
            return cached

        record_benchmark_metadata(dispatch_plan_cache="miss")
        items = _normalize_dispatch_items(simulations)
        groups_by_signature: dict[tuple[Any, ...], list[_PendingDispatchGroup]] = {}
        for item in items:
            signature = item.signature
            compatible_groups = groups_by_signature.setdefault(signature, [])
            target_group: _PendingDispatchGroup | None = None
            if item.mode == "single":
                target_group = compatible_groups[0] if compatible_groups else None
            else:
                for group in compatible_groups:
                    if group.can_accept(item):
                        target_group = group
                        break
            if target_group is None:
                target_group = _PendingDispatchGroup.from_item(item)
                compatible_groups.append(target_group)
            else:
                target_group.append(item)

        groups_list: list[DispatchGroup] = []
        for signature, signature_groups in groups_by_signature.items():
            for pending_group in signature_groups:
                group_items = pending_group.items
                groups_list.append(
                    DispatchGroup(
                        group_id=len(groups_list),
                        items=tuple(group_items),
                        signature=signature,
                        mode=group_items[0].mode,
                        nx=max(int(item.solver_axon.n_compartments) for item in group_items),
                        geometry_shared=_group_has_shared_geometry(group_items),
                    )
                )
        groups = tuple(groups_list)
        record_benchmark_metadata(
            item_count=len(items),
            group_count=len(groups),
            group_sizes=[group.size for group in groups],
            group_modes=[group.mode for group in groups],
            group_nx=[group.nx for group in groups],
        )
        plan = DispatchPlan(items=items, groups=groups)
        _store_dispatch_plan_cache(cache_key, plan)
        return plan


def dispatch_plan_identity_key(
    axons: Sequence[Axon | AxonInstance],
) -> tuple[Any, ...]:
    """Return a lightweight identity key for reusing an already-built plan.

    This key follows object replacement rather than numeric waveform values.
    Iterative curve protocols may mutate a shared stimulus amplitude while the
    dispatch compatibility stays unchanged, but replacing a row, axon,
    stimulation, drive, or stimulus invalidates the per-simulation plan.
    """

    simulations = tuple(as_axon_instance(axon) for axon in axons)
    stimulus_signature_cache: dict[int, tuple[Any, ...]] = {}
    return (
        "dispatch_plan_identity_v1",
        tuple(
            _dispatch_plan_row_identity_key(
                simulation,
                stimulus_signature_cache=stimulus_signature_cache,
            )
            for simulation in simulations
        ),
    )


def _dispatch_plan_cache_key(
    simulations: Sequence[AxonInstance],
) -> tuple[Any, ...]:
    """Return the stable execution-layout key for a pool.

    The key is intentionally tied to `AxonInstance` identity. This makes the
    cache useful for iterative protocols that mutate only stimuli on a stable
    pool, while avoiding accidental reuse when callers pass fresh bare `Axon`
    objects or rebuild simulation rows.
    """

    stimulus_signature_cache: dict[int, tuple[Any, ...]] = {}
    return (
        "dispatch_plan_v2",
        tuple(
            _dispatch_plan_row_cache_key(
                simulation,
                stimulus_signature_cache=stimulus_signature_cache,
            )
            for simulation in simulations
        ),
    )


def _dispatch_plan_row_cache_key(
    simulation: AxonInstance,
    *,
    stimulus_signature_cache: dict[int, tuple[Any, ...]],
) -> tuple[Any, ...]:
    return (
        id(simulation),
        _solver_axon_cache_key(simulation),
        _stimulation_temporal_signature(simulation, stimulus_signature_cache),
        float(getattr(simulation, "v_init", 0.0)),
        float(getattr(simulation, "Veinit", 0.0)),
        float(getattr(simulation, "temperature", 0.0)),
    )


def _dispatch_plan_row_identity_key(
    simulation: AxonInstance,
    *,
    stimulus_signature_cache: dict[int, tuple[Any, ...]],
) -> tuple[Any, ...]:
    return (
        id(simulation),
        id(simulation.axon),
        id(getattr(simulation, "_xraxial_override", None)),
        id(getattr(simulation, "_xg_override", None)),
        id(getattr(simulation, "_xc_override", None)),
        _stimulation_temporal_signature(simulation, stimulus_signature_cache),
        float(getattr(simulation, "v_init", 0.0)),
        float(getattr(simulation, "Veinit", 0.0)),
        float(getattr(simulation, "temperature", 0.0)),
    )


def _store_dispatch_plan_cache(key: tuple[Any, ...], plan: DispatchPlan) -> None:
    _DISPATCH_PLAN_CACHE[key] = plan
    _DISPATCH_PLAN_CACHE.move_to_end(key)
    while len(_DISPATCH_PLAN_CACHE) > _DISPATCH_PLAN_CACHE_MAX_SIZE:
        _DISPATCH_PLAN_CACHE.popitem(last=False)


def _normalize_dispatch_items(axons: Sequence[Axon | AxonInstance]) -> tuple[DispatchItem, ...]:
    """Validate public pool items and preserve input order."""

    items: list[DispatchItem] = []
    solver_cache: dict[tuple[Any, ...], SolverAxon] = {}
    metadata_cache: dict[tuple[Any, ...], _SolverDispatchMetadata] = {}
    model_signature_cache: dict[int, Any] = {}
    model_structure_cache: dict[int, Any] = {}
    stimulus_signature_cache: dict[int, tuple[Any, ...]] = {}
    for index, axon in enumerate(axons):
        simulation = as_axon_instance(axon)
        cache_key = _solver_axon_cache_key(simulation)
        solver_axon = solver_cache.get(cache_key)
        if solver_axon is None:
            solver_axon = build_solver_axon(simulation)
            solver_cache[cache_key] = solver_axon
        metadata = metadata_cache.get(cache_key)
        if metadata is None:
            metadata = _solver_dispatch_metadata(
                solver_axon,
                model_signature_cache=model_signature_cache,
                model_structure_cache=model_structure_cache,
            )
            metadata_cache[cache_key] = metadata
        items.append(
            _make_dispatch_item(
                index=index,
                simulation=simulation,
                solver_axon=solver_axon,
                metadata=metadata,
                stimulation_temporal_signature=_stimulation_temporal_signature(
                    simulation,
                    stimulus_signature_cache,
                ),
            )
        )
    return tuple(items)


def _solver_axon_cache_key(simulation: AxonInstance) -> tuple[Any, ...]:
    """Return the part of an instance that changes its flattened cable arrays."""

    return (
        id(simulation.axon),
        _optional_array_signature(getattr(simulation, "_xraxial_override", None)),
        _optional_array_signature(getattr(simulation, "_xg_override", None)),
        _optional_array_signature(getattr(simulation, "_xc_override", None)),
    )


def _optional_array_signature(values: Any | None) -> tuple[tuple[int, ...], str, str] | None:
    if values is None:
        return None
    return _array_signature(values)


def _solver_dispatch_metadata(
    solver_axon: SolverAxon,
    *,
    model_signature_cache: dict[int, Any],
    model_structure_cache: dict[int, Any],
) -> _SolverDispatchMetadata:
    mode = _resolve_mode(solver_axon)
    membrane_signature = _axon_membrane_signature(
        solver_axon,
        model_signature_cache=model_signature_cache,
    )
    membrane_structure_sequence = _axon_membrane_structure_sequence(
        solver_axon,
        model_structure_cache=model_structure_cache,
    )
    cable_signature = _axon_cable_signature(solver_axon)
    membrane_group_signature = (
        _unique_membrane_structures(membrane_structure_sequence)
        if mode == "double"
        else membrane_signature
    )
    nx_signature = None if mode == "double" else int(solver_axon.n_compartments)
    return _SolverDispatchMetadata(
        mode=mode,
        dtype_str=solver_axon.dtype.str,
        nx_signature=nx_signature,
        membrane_signature=membrane_signature,
        membrane_group_signature=membrane_group_signature,
        membrane_structure_sequence=membrane_structure_sequence,
        cable_signature=cable_signature,
    )


def _make_dispatch_item(
    index: int,
    simulation: AxonInstance,
    solver_axon: SolverAxon,
    *,
    metadata: _SolverDispatchMetadata,
    stimulation_temporal_signature: tuple[Any, ...],
) -> DispatchItem:
    signature = (
        metadata.mode,
        metadata.dtype_str,
        metadata.membrane_group_signature,
        metadata.nx_signature,
        stimulation_temporal_signature,
        float(getattr(simulation, "v_init", 0.0)),
        float(getattr(simulation, "Veinit", 0.0)),
        float(getattr(simulation, "temperature", 0.0)),
    )
    return DispatchItem(
        index=index,
        simulation=simulation,
        solver_axon=solver_axon,
        signature=signature,
        mode=metadata.mode,
        membrane_signature=metadata.membrane_signature,
        cable_signature=metadata.cable_signature,
    )


def _resolve_mode(axon: SolverAxon) -> _CableMode:
    """Return the cable mode implied by an axon description."""

    if axon.formulation == "double-cable":
        return "double"
    return "single"


def _dispatch_signature(item: DispatchItem) -> tuple[Any, ...]:
    """Return the compatibility signature used for grouping."""

    return item.signature


def _unique_membrane_structures(signatures: Iterable[Any]) -> tuple[Any, ...]:
    """Return an order-independent membrane-structure set signature."""

    return tuple(sorted(set(signatures), key=repr))


def _model_structure_signature(model: Any, cache: dict[int, Any]) -> Any:
    """Return a structural membrane signature that ignores numeric parameters."""

    cache_key = id(model)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    kind = getattr(model, "kind", None)
    if kind is not None:
        component_structures = tuple(
            _model_structure_signature(component, cache)
            for component in getattr(model, "components", ())
        )
        signature = ("membrane", kind, component_structures)
    else:
        implementation = getattr(model, "_implementation", None)
        if implementation is not None:
            signature = _model_structure_signature(implementation, cache)
        else:
            signature = (model.__class__.__module__, model.__class__.__qualname__)
    cache[cache_key] = signature
    return signature


def _axon_membrane_structure_sequence(
    axon: SolverAxon,
    *,
    model_structure_cache: dict[int, Any],
) -> Any:
    """Return per-compartment membrane structures without parameter values."""

    return tuple(
        _model_structure_signature(model, model_structure_cache)
        for model in axon.membrane_models
    )


def _group_has_shared_geometry(items: Sequence[DispatchItem]) -> bool:
    """Return whether all rows share exact cable/periaxonal arrays."""

    signatures = {item.cable_signature for item in items}
    return len(signatures) == 1


def _model_signature(model: Any) -> Any:
    """Return a stable-enough signature for a membrane model description."""

    static_signature = getattr(model, "_static_signature", None)
    if callable(static_signature):
        return static_signature()
    return repr(model)


def _axon_membrane_signature(
    axon: SolverAxon,
    *,
    model_signature_cache: dict[int, Any],
) -> Any:
    """Return the membrane component of an axon compatibility signature."""

    signatures = []
    for model in axon.membrane_models:
        cache_key = id(model)
        signature = model_signature_cache.get(cache_key)
        if signature is None:
            signature = _model_signature(model)
            model_signature_cache[cache_key] = signature
        signatures.append(signature)
    return tuple(signatures)


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


def _stimulation_temporal_signature(
    simulation: AxonInstance,
    stimulus_signature_cache: dict[int, tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Return the temporal stimulation compatibility signature for grouping.

    Footprints are deliberately excluded: a runtime batch may contain distinct
    extracellular footprints, but rows should only share a group when their
    temporal drive waveform shapes are compatible. Amplitude scale is a runtime
    numeric payload and should not split threshold-style groups.
    """

    rows = tuple(getattr(simulation, "extracellular_stimulations", ()))
    if not rows:
        return ()
    return tuple(
        tuple(
            _stimulus_temporal_signature(
                getattr(drive, "stimulus", None),
                stimulus_signature_cache,
            )
            for drive in stimulation.drives
        )
        for stimulation in rows
    )


def _stimulus_temporal_signature(
    stimulus: Any,
    stimulus_signature_cache: dict[int, tuple[Any, ...]],
) -> tuple[Any, ...]:
    if stimulus is None:
        return ("none",)
    cache_key = id(stimulus)
    cached = stimulus_signature_cache.get(cache_key)
    if cached is not None:
        return cached
    signature = (
        "stimulus_waveform_shape",
        type(stimulus),
        _array_signature(getattr(stimulus, "t", ())),
        _stimulus_y_shape_signature(getattr(stimulus, "y", ())),
        getattr(stimulus, "mode", None),
        getattr(stimulus, "y_unit", None),
    )
    stimulus_signature_cache[cache_key] = signature
    return signature


def _stimulus_y_shape_signature(values: Any) -> tuple[Any, ...]:
    """Return an amplitude-scale-invariant waveform signature when possible."""

    try:
        y = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return ("raw", _array_signature(values))
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        return ("raw", _array_signature(values))
    nonzero = np.flatnonzero(np.abs(y) > 0.0)
    if len(nonzero) == 0:
        normalized = np.zeros_like(y, dtype=float)
    else:
        scale = float(y[int(nonzero[0])])
        if scale == 0.0:
            return ("raw", _array_signature(values))
        normalized = np.asarray(y / scale, dtype=float)
    return ("scaled", _array_signature(normalized))


__all__ = [
    "DispatchPlan",
    "build_dispatch_plan",
    "dispatch_plan_identity_key",
]
