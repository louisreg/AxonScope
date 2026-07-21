from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.axons.axon import Axon
from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.dispatcher.numeric_axis import NumericAxisInput
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


@dataclass(frozen=True, slots=True)
class DispatchGroupStructure:
    """Versioned content signature for one planned group row layout."""

    schema_version: int
    spatial_rows: tuple[Any, ...]
    runtime_rows: tuple[Any, ...]

    @classmethod
    def from_items(cls, items: tuple[DispatchItem, ...]) -> "DispatchGroupStructure":
        if not items:
            raise ValueError("dispatch group structure requires at least one item.")
        return cls(
            schema_version=1,
            spatial_rows=(
                "dispatch_spatial_rows_v1",
                len(items),
                _digest_group_spatial_items(items),
            ),
            runtime_rows=(
                "dispatch_runtime_rows_v1",
                len(items),
                _digest_group_runtime_items(items),
            ),
        )

    def repeated(self, count: int) -> "DispatchGroupStructure":
        repeats = int(count)
        if repeats < 1:
            raise ValueError("dispatch group row repeat count must be positive.")
        if repeats == 1:
            return self
        return DispatchGroupStructure(
            schema_version=self.schema_version,
            spatial_rows=("dispatch_repeat_rows_v1", self.spatial_rows, repeats),
            runtime_rows=("dispatch_repeat_rows_v1", self.runtime_rows, repeats),
        )

    def padded_with_last(
        self,
        item: DispatchItem,
        count: int,
    ) -> "DispatchGroupStructure":
        padding = int(count)
        if padding < 0:
            raise ValueError("dispatch group row padding must be non-negative.")
        if padding == 0:
            return self
        return DispatchGroupStructure(
            schema_version=self.schema_version,
            spatial_rows=(
                "dispatch_pad_last_v1",
                self.spatial_rows,
                _digest_spatial_item(item),
                padding,
            ),
            runtime_rows=(
                "dispatch_pad_last_v1",
                self.runtime_rows,
                _digest_runtime_item(item),
                padding,
            ),
        )


@dataclass(frozen=True)
class DispatchGroup:
    """A compatibility group selected by the dispatcher."""

    group_id: int
    items: tuple[DispatchItem, ...]
    signature: tuple[Any, ...]
    mode: _CableMode
    nx: int
    structure: DispatchGroupStructure
    geometry_shared: bool = True
    numeric_axis: NumericAxisInput | None = None
    numeric_axis_source_size: int | None = None

    @cached_property
    def pool_indices(self) -> tuple[int, ...]:
        """Input-order indices represented by this group."""

        return tuple(item.index for item in self.items)

    @cached_property
    def axons(self) -> tuple[Axon, ...]:
        """Descriptive axons represented by this group."""

        return tuple(item.simulation.axon for item in self.items)

    @cached_property
    def simulations(self) -> tuple[AxonInstance, ...]:
        """Executable axon instances represented by this group."""

        return tuple(item.simulation for item in self.items)

    @cached_property
    def empty_record_indices(self) -> tuple[None, ...]:
        """Shared no-recording row metadata for compact cohort results."""

        return (None,) * len(self.items)

    @property
    def size(self) -> int:
        """Number of axons in this dispatch group."""

        return len(self.items)

    @cached_property
    def batch_kind(self) -> str:
        """Return the intended batch route for this group."""

        prefix = "strict" if self.geometry_shared else "parameter"
        return f"{prefix}-{self.mode}-cable"

    @cached_property
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


def expand_dispatch_plan_for_numeric_axis(
    plan: DispatchPlan,
    axis_input: NumericAxisInput,
) -> DispatchPlan:
    """Expand only dispatch indices/references for a numeric execution axis.

    Axon, stimulation, membrane, and solver descriptions remain shared. The
    concrete runtime lowers these logical rows into numerical batch arrays.
    """

    axis_size = int(axis_input.size)
    if axis_size < 1:
        raise ValueError("numeric axis must contain at least one value.")
    source_size = len(plan.items)
    declared_source_size = getattr(axis_input, "source_size", source_size)
    if int(declared_source_size) != source_size:
        raise ValueError(
            "numeric-axis source size does not match the dispatch plan: "
            f"got {declared_source_size}, expected {source_size}."
        )
    expanded_items: list[DispatchItem] = []
    expanded_groups: list[DispatchGroup] = []
    for group in plan.groups:
        select_sources = getattr(axis_input, "for_source_indices", None)
        group_axis_input = (
            select_sources(group.pool_indices)
            if callable(select_sources)
            else axis_input
        )
        group_items = tuple(
            replace(
                item,
                index=axis_index * source_size + item.index,
            )
            for axis_index in range(axis_size)
            for item in group.items
        )
        expanded_items.extend(group_items)
        expanded_groups.append(
            DispatchGroup(
                group_id=group.group_id,
                items=group_items,
                signature=(
                    group.signature,
                    "numeric_axis_v1",
                    group_axis_input.dispatch_signature,
                ),
                mode=group.mode,
                nx=group.nx,
                structure=group.structure.repeated(axis_size),
                geometry_shared=group.geometry_shared,
                numeric_axis=group_axis_input,
                numeric_axis_source_size=group.size,
            )
        )
    return DispatchPlan(
        items=tuple(sorted(expanded_items, key=lambda item: item.index)),
        groups=tuple(expanded_groups),
    )


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


@dataclass(frozen=True, slots=True)
class _PreparedDispatchRow:
    """Stable row signatures shared by cache lookup and normalization."""

    simulation: AxonInstance
    solver_axon_cache_key: tuple[Any, ...]
    stimulation_temporal_signature: tuple[Any, ...]


_DISPATCH_PLAN_CACHE: OrderedDict[tuple[Any, ...], DispatchPlan] = OrderedDict()
_DISPATCH_PLAN_CACHE_MAX_SIZE = 64


def build_dispatch_plan(axons: Sequence[Axon | AxonInstance]) -> DispatchPlan:
    """Normalize and group axon simulations before execution."""

    simulations = tuple(as_axon_instance(axon) for axon in axons)
    with benchmark_span("dispatch.build_plan", pool_size=len(simulations)):
        with benchmark_span("dispatch.build_plan.cache_key"):
            prepared_rows = _prepare_dispatch_rows(simulations)
            cache_key = _dispatch_plan_cache_key(prepared_rows)
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
        with benchmark_span("dispatch.build_plan.normalize_items"):
            items = _normalize_dispatch_items(prepared_rows)
        with benchmark_span("dispatch.build_plan.group_items"):
            groups_by_signature: dict[
                tuple[Any, ...], list[_PendingDispatchGroup]
            ] = {}
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

        with benchmark_span("dispatch.build_plan.materialize_groups"):
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
                            nx=max(
                                int(item.solver_axon.n_compartments)
                                for item in group_items
                            ),
                            structure=DispatchGroupStructure.from_items(
                                tuple(group_items)
                            ),
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
    prepared_rows: Sequence[_PreparedDispatchRow],
) -> tuple[Any, ...]:
    """Return the stable execution-layout key for a pool.

    The key is intentionally tied to `AxonInstance` identity. This makes the
    cache useful for iterative protocols that mutate only stimuli on a stable
    pool, while avoiding accidental reuse when callers pass fresh bare `Axon`
    objects or rebuild simulation rows.
    """

    return (
        "dispatch_plan_v2",
        tuple(
            _dispatch_plan_row_cache_key(prepared_row)
            for prepared_row in prepared_rows
        ),
    )


def _dispatch_plan_row_cache_key(
    prepared_row: _PreparedDispatchRow,
) -> tuple[Any, ...]:
    simulation = prepared_row.simulation
    return (
        id(simulation),
        prepared_row.solver_axon_cache_key,
        prepared_row.stimulation_temporal_signature,
        float(getattr(simulation, "v_init", 0.0)),
        float(getattr(simulation, "Veinit", 0.0)),
        float(getattr(simulation, "temperature", 0.0)),
    )


def _prepare_dispatch_rows(
    simulations: Sequence[AxonInstance],
) -> tuple[_PreparedDispatchRow, ...]:
    stimulus_signature_cache: dict[int, tuple[Any, ...]] = {}
    return tuple(
        _PreparedDispatchRow(
            simulation=simulation,
            solver_axon_cache_key=_solver_axon_cache_key(simulation),
            stimulation_temporal_signature=_stimulation_temporal_signature(
                simulation,
                stimulus_signature_cache,
            ),
        )
        for simulation in simulations
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


def _normalize_dispatch_items(
    prepared_rows: Sequence[_PreparedDispatchRow],
) -> tuple[DispatchItem, ...]:
    """Validate public pool items and preserve input order."""

    items: list[DispatchItem] = []
    solver_cache: dict[tuple[Any, ...], SolverAxon] = {}
    metadata_cache: dict[tuple[Any, ...], _SolverDispatchMetadata] = {}
    model_signature_cache: dict[int, Any] = {}
    model_structure_cache: dict[int, Any] = {}
    for index, prepared_row in enumerate(prepared_rows):
        simulation = prepared_row.simulation
        cache_key = prepared_row.solver_axon_cache_key
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
                stimulation_temporal_signature=(
                    prepared_row.stimulation_temporal_signature
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
        else membrane_structure_sequence
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
        components = tuple(getattr(model, "components", ()))
        if components:
            component_structures = tuple(
                _model_structure_signature(component, cache)
                for component in components
            )
            signature = ("membrane_composite", component_structures)
        else:
            source_path = getattr(model, "source_path", None)
            source_class = getattr(model, "source_class", None)
            if source_path is not None and source_class is not None:
                signature = (
                    "membrane_source",
                    str(source_path),
                    str(source_class),
                )
            else:
                signature = ("membrane", kind)
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


def _digest_group_spatial_items(items: tuple[DispatchItem, ...]) -> str:
    token_cache: dict[int, str] = {}
    hasher = hashlib.blake2b(digest_size=16)
    for item in items:
        hasher.update(_digest_signature_value(item.cable_signature, token_cache).encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _digest_group_runtime_items(items: tuple[DispatchItem, ...]) -> str:
    token_cache: dict[int, str] = {}
    hasher = hashlib.blake2b(digest_size=16)
    for item in items:
        hasher.update(_digest_signature_value(item.membrane_signature, token_cache).encode())
        hasher.update(b"\0")
        hasher.update(_digest_signature_value(item.cable_signature, token_cache).encode())
        hasher.update(b"\0")
        simulation = item.simulation
        _update_digest_float(hasher, float(getattr(simulation, "v_init", 0.0)))
        _update_digest_float(hasher, float(getattr(simulation, "Veinit", 0.0)))
        _update_digest_float(hasher, float(getattr(simulation, "temperature", 0.0)))
    return hasher.hexdigest()


def _digest_spatial_item(item: DispatchItem) -> str:
    return _digest_signature_value(item.cable_signature, {})


def _digest_runtime_item(item: DispatchItem) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    token_cache: dict[int, str] = {}
    hasher.update(_digest_signature_value(item.membrane_signature, token_cache).encode())
    hasher.update(b"\0")
    hasher.update(_digest_signature_value(item.cable_signature, token_cache).encode())
    hasher.update(b"\0")
    simulation = item.simulation
    _update_digest_float(hasher, float(getattr(simulation, "v_init", 0.0)))
    _update_digest_float(hasher, float(getattr(simulation, "Veinit", 0.0)))
    _update_digest_float(hasher, float(getattr(simulation, "temperature", 0.0)))
    return hasher.hexdigest()


def _digest_signature_value(value: Any, cache: dict[int, str]) -> str:
    cache_key = id(value)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PickleError, TypeError, AttributeError):
        payload = repr(value).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
    cache[cache_key] = digest
    return digest


def _update_digest_float(hasher: Any, value: float) -> None:
    hasher.update(repr(float(value)).encode("ascii"))
    hasher.update(b"\0")


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
        _stimulus_y_shape_signature(stimulus),
        getattr(stimulus, "mode", None),
        getattr(stimulus, "y_unit", None),
    )
    stimulus_signature_cache[cache_key] = signature
    return signature


def _stimulus_y_shape_signature(stimulus: Any) -> tuple[Any, ...]:
    """Return an amplitude-scale-invariant waveform signature when possible."""

    declared = getattr(stimulus, "_scale_shape", None)
    if declared is not None:
        return ("declared", tuple(declared))
    values = getattr(stimulus, "y", ())
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
    "expand_dispatch_plan_for_numeric_axis",
]
