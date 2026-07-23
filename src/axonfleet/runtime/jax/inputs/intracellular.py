"""JAX materialization of intracellular input tensors."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons.axon import Axon
from axonfleet.runtime.inputs.payloads import SparseIntracellularCurrentDensityBatch
from axonfleet.runtime.jax.types import SolverRuntime
from axonfleet.runtime.solver_axon import SolverAxon
from axonfleet.stimulation import (
    IntracellularCurrentClamp,
    Stimulus,
)

Array = Any
AxonLike = Axon | AxonInstance
_ZERO_SPARSE_INTRACELLULAR_CACHE_MAX = 32
_ZERO_SPARSE_INTRACELLULAR_CACHE: OrderedDict[
    tuple[int, int, int, str, str],
    SparseIntracellularCurrentDensityBatch,
] = OrderedDict()


def build_intracellular_current_density_batch(
    axons: Sequence[AxonLike],
    runtime: SolverRuntime,
    *,
    solver_axons: Sequence[SolverAxon] | None = None,
    target_nx: int | None = None,
) -> Array:
    """Build ``Iinj[B, Nt, Nx]`` from axon-attached intracellular contexts."""

    if solver_axons is not None and len(solver_axons) != len(axons):
        raise ValueError("solver_axons must contain one row per axon.")
    dtype = runtime.membrane.dtype
    target_width = runtime.membrane.Nx if target_nx is None else int(target_nx)
    t_mid = (
        jnp.arange(runtime.grid.Nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(runtime.grid.dt_ms, dtype=dtype)
    resolved_solver_axons = tuple(
        runtime.axon if solver_axons is None else solver_axons[index]
        for index in range(len(axons))
    )
    if not _can_build_intracellular_rows_from_clamps(axons):
        raise TypeError("intracellular batches support current clamps only.")
    return _build_intracellular_current_density_batch_from_clamps(
        axons,
        resolved_solver_axons,
        t_mid,
        target_nx=target_width,
        dtype_local=dtype,
    )


def can_build_sparse_intracellular_current_density_batch(
    axons: Sequence[AxonLike],
) -> bool:
    """Return whether axon rows contain only point current clamps."""

    return _can_build_intracellular_rows_from_clamps(axons)


def build_sparse_intracellular_current_density_batch(
    axons: Sequence[AxonLike],
    runtime: SolverRuntime,
    *,
    solver_axons: Sequence[SolverAxon] | None = None,
    target_nx: int | None = None,
) -> SparseIntracellularCurrentDensityBatch:
    """Build sparse ``Iinj`` data from point current clamps.

    This keeps the time axis but removes the dense compartment axis. It is
    intended for observer-only kernels where the current density can be
    scattered into the solver state inside each time step.
    """

    if not _can_build_intracellular_rows_from_clamps(axons):
        raise TypeError("sparse intracellular batches currently support current clamps only.")
    if solver_axons is not None and len(solver_axons) != len(axons):
        raise ValueError("solver_axons must contain one row per axon.")
    dtype = runtime.membrane.dtype
    target_width = runtime.membrane.Nx if target_nx is None else int(target_nx)
    t_mid = (
        jnp.arange(runtime.grid.Nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(runtime.grid.dt_ms, dtype=dtype)
    resolved_solver_axons = tuple(
        runtime.axon if solver_axons is None else solver_axons[index]
        for index in range(len(axons))
    )
    return _build_sparse_intracellular_current_density_batch_from_clamps(
        axons,
        resolved_solver_axons,
        t_mid,
        target_nx=target_width,
        dtype_local=dtype,
    )


def build_zero_sparse_intracellular_current_density_batch(
    *,
    batch_size: int,
    step_count: int,
    target_nx: int,
    dtype_local: jnp.dtype,
) -> SparseIntracellularCurrentDensityBatch:
    """Build an empty sparse current-density payload for no-clamp cohorts."""

    rows = int(batch_size)
    steps = int(step_count)
    nx = int(target_nx)
    dtype = np.dtype(dtype_local)
    cache_key = (rows, steps, nx, dtype.str, _default_jax_device_key())
    cached = _ZERO_SPARSE_INTRACELLULAR_CACHE.get(cache_key)
    if cached is not None:
        _ZERO_SPARSE_INTRACELLULAR_CACHE.move_to_end(cache_key)
        return cached
    batch = SparseIntracellularCurrentDensityBatch(
        density_mid=np.zeros((rows, steps, 0), dtype=dtype),
        indices=np.zeros((rows, 0), dtype=np.int32),
        mask=np.zeros((rows, 0), dtype=bool),
        target_nx=nx,
    )
    _ZERO_SPARSE_INTRACELLULAR_CACHE[cache_key] = batch
    if len(_ZERO_SPARSE_INTRACELLULAR_CACHE) > _ZERO_SPARSE_INTRACELLULAR_CACHE_MAX:
        _ZERO_SPARSE_INTRACELLULAR_CACHE.popitem(last=False)
    return batch


def _default_jax_device_key() -> str:
    try:
        return str(jax.devices()[0])
    except Exception:
        return "unknown"


def _can_build_intracellular_rows_from_clamps(axons: Sequence[AxonLike]) -> bool:
    for axon in axons:
        for context in getattr(axon, "intracellular_contexts", ()):
            if not isinstance(context, IntracellularCurrentClamp):
                return False
    return True


def _build_intracellular_current_density_batch_from_clamps(
    axons: Sequence[AxonLike],
    solver_axons: Sequence[SolverAxon],
    t_ms: Array,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
) -> Array:
    np_dtype = np.dtype(dtype_local)
    t = np.asarray(t_ms, dtype=np_dtype)
    values = np.zeros((len(axons), int(t.shape[0]), int(target_nx)), dtype=np_dtype)
    current_cache: dict[Any, np.ndarray] = {}

    for row_index, (axon, solver_axon) in enumerate(zip(axons, solver_axons, strict=True)):
        if int(solver_axon.n_compartments) > int(target_nx):
            raise ValueError(
                f"target_nx must be >= array width, got target_nx={target_nx}, "
                f"width={solver_axon.n_compartments}."
            )
        contexts = tuple(getattr(axon, "intracellular_contexts", ()))
        if not contexts:
            continue
        x_um = np.asarray(solver_axon.x_um, dtype=float)
        area_cm2 = _compartment_surface_area_cm2_numpy(solver_axon)
        for context in contexts:
            idx = int(np.argmin(np.abs(x_um - float(context.position_um))))
            cache_key = id(context.current)
            current_nA = current_cache.get(cache_key)
            if current_nA is None:
                current_nA = np.asarray(
                    context.current.evaluate(t, unit="nanoampere"),
                    dtype=np_dtype,
                )
                current_cache[cache_key] = current_nA
            values[row_index, :, idx] += current_nA * (
                np.asarray(1e-3, dtype=np_dtype) / area_cm2[idx]
            )
    return jnp.asarray(values, dtype=dtype_local)


def _build_sparse_intracellular_current_density_batch_from_clamps(
    axons: Sequence[AxonLike],
    solver_axons: Sequence[SolverAxon],
    t_ms: Array,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
) -> SparseIntracellularCurrentDensityBatch:
    np_dtype = np.dtype(dtype_local)
    t = np.asarray(t_ms, dtype=np_dtype)
    fast_pulse_batch = _try_build_single_pulse_sparse_current_density_batch(
        axons,
        solver_axons,
        t,
        target_nx=target_nx,
        dtype_local=dtype_local,
        np_dtype=np_dtype,
    )
    if fast_pulse_batch is not None:
        return fast_pulse_batch

    max_contexts = max(
        (len(tuple(getattr(axon, "intracellular_contexts", ()))) for axon in axons),
        default=0,
    )
    density_mid = np.zeros(
        (len(axons), int(t.shape[0]), int(max_contexts)),
        dtype=np_dtype,
    )
    indices = np.zeros((len(axons), int(max_contexts)), dtype=np.int32)
    mask = np.zeros((len(axons), int(max_contexts)), dtype=bool)
    current_cache: dict[Any, np.ndarray] = {}

    for row_index, (axon, solver_axon) in enumerate(zip(axons, solver_axons, strict=True)):
        if int(solver_axon.n_compartments) > int(target_nx):
            raise ValueError(
                f"target_nx must be >= array width, got target_nx={target_nx}, "
                f"width={solver_axon.n_compartments}."
            )
        contexts = tuple(getattr(axon, "intracellular_contexts", ()))
        if not contexts:
            continue
        x_um = np.asarray(solver_axon.x_um, dtype=float)
        area_cm2 = _compartment_surface_area_cm2_numpy(solver_axon)
        for context_index, context in enumerate(contexts):
            idx = int(np.argmin(np.abs(x_um - float(context.position_um))))
            cache_key = id(context.current)
            current_nA = current_cache.get(cache_key)
            if current_nA is None:
                current_nA = np.asarray(
                    context.current.evaluate(t, unit="nanoampere"),
                    dtype=np_dtype,
                )
                current_cache[cache_key] = current_nA
            density_mid[row_index, :, context_index] = current_nA * (
                np.asarray(1e-3, dtype=np_dtype) / area_cm2[idx]
            )
            indices[row_index, context_index] = idx
            mask[row_index, context_index] = True

    return SparseIntracellularCurrentDensityBatch(
        density_mid=jnp.asarray(density_mid, dtype=dtype_local),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        mask=jnp.asarray(mask, dtype=bool),
        target_nx=int(target_nx),
    )


def _try_build_single_pulse_sparse_current_density_batch(
    axons: Sequence[AxonLike],
    solver_axons: Sequence[SolverAxon],
    t: np.ndarray,
    *,
    target_nx: int,
    dtype_local: jnp.dtype,
    np_dtype: np.dtype[Any],
) -> SparseIntracellularCurrentDensityBatch | None:
    """Vectorize the common one-rectangular-pulse current-clamp case."""

    if not axons:
        return None

    contexts_by_row = tuple(
        tuple(getattr(axon, "intracellular_contexts", ())) for axon in axons
    )
    if not all(len(contexts) == 1 for contexts in contexts_by_row):
        return None
    if not all(
        _is_three_point_hold_stimulus(contexts[0].current)
        for contexts in contexts_by_row
    ):
        return None

    pulse_times = np.stack(
        [
            np.asarray(contexts[0].current.t, dtype=np_dtype)
            for contexts in contexts_by_row
        ],
        axis=0,
    )
    pulse_values = np.stack(
        [
            np.asarray(contexts[0].current.y, dtype=np_dtype)
            for contexts in contexts_by_row
        ],
        axis=0,
    )

    indices = np.zeros((len(axons), 1), dtype=np.int32)
    scales = np.zeros((len(axons),), dtype=np_dtype)
    geometry_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row_index, (contexts, solver_axon) in enumerate(
        zip(contexts_by_row, solver_axons, strict=True)
    ):
        if int(solver_axon.n_compartments) > int(target_nx):
            raise ValueError(
                f"target_nx must be >= array width, got target_nx={target_nx}, "
                f"width={solver_axon.n_compartments}."
            )
        cache_key = id(solver_axon)
        cached = geometry_cache.get(cache_key)
        if cached is None:
            cached = (
                np.asarray(solver_axon.x_um, dtype=float),
                _compartment_surface_area_cm2_numpy(solver_axon),
            )
            geometry_cache[cache_key] = cached
        x_um, area_cm2 = cached
        context = contexts[0]
        index = int(np.argmin(np.abs(x_um - float(context.position_um))))
        indices[row_index, 0] = index
        scales[row_index] = np.asarray(1e-3, dtype=np_dtype) / area_cm2[index]

    # This mirrors Stimulus.evaluate(..., mode="hold") for 3 sample points:
    # baseline before t1, pulse value from t1 to t2, baseline after t2.
    t_grid = t[None, :]
    current_nA = np.where(
        t_grid < pulse_times[:, 1:2],
        pulse_values[:, 0:1],
        np.where(
            t_grid < pulse_times[:, 2:3],
            pulse_values[:, 1:2],
            pulse_values[:, 2:3],
        ),
    )
    density_mid = current_nA[:, :, None] * scales[:, None, None]
    mask = np.ones((len(axons), 1), dtype=bool)

    return SparseIntracellularCurrentDensityBatch(
        density_mid=jnp.asarray(density_mid, dtype=dtype_local),
        indices=jnp.asarray(indices, dtype=jnp.int32),
        mask=jnp.asarray(mask, dtype=bool),
        target_nx=int(target_nx),
    )


def _is_three_point_hold_stimulus(stimulus: Stimulus) -> bool:
    """Return whether a stimulus can use the vectorized pulse path."""

    return stimulus.mode == "hold" and len(stimulus.t) == 3 and len(stimulus.y) == 3


def _compartment_surface_area_cm2_numpy(solver_axon: SolverAxon) -> np.ndarray:
    diam_cm = np.asarray(solver_axon.diam_um, dtype=float) * 1e-4
    length_cm = np.asarray(solver_axon.compartment_lengths_um, dtype=float) * 1e-4
    return np.pi * diam_cm * length_cm
