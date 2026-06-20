"""JAX materialization of batched solver input tensors.

The solver kernels operate on JAX arrays. This module turns prepared host rows,
extracellular context rows, intracellular contexts, and static footprints into
backend arrays for batch execution.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking.hotpaths import record_benchmark_metadata
from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.stimulation import (
    AnalyticalExtracellularContext,
    ExtracellularContext,
    IntracellularCurrentClamp,
    PointSourceElectrode,
    Stimulus,
)
from axonscope.stimulation.runtime import (
    compile_extracellular_context,
    compile_intracellular_contexts,
    compile_stimulus,
)
from axonscope.solvers.batch_inputs import (
    FactorizedExtracellularPotentialBatch,
    SparseIntracellularCurrentDensityBatch,
)
from axonscope.solvers.runtime import SolverRuntime
from axonscope.solvers.axon_runtime import SolverAxon
from axonscope.solvers.common import simulation_step_count

Array = Any
AxonLike = Axon | AxonInstance
ContextBatchRow = ExtracellularContext | Sequence[ExtracellularContext] | None
FootprintEngine = Literal["numpy", "jax"]

_POINT_SOURCE_FOOTPRINT_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def build_vstim_midpoint_batch(
    axon: object,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    tsim_ms: float,
    dt_ms: float,
    x_positions_m: Array | None = None,
    axon_y_um: Array | None = None,
    axon_z_um: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build imposed extracellular potentials at solver midpoints.

    Returns ``Vstim[B, Nt, Nx]`` in millivolts. Each row can contain one
    ``ExtracellularContext``, several contexts that are summed, or ``None`` for
    a zero-field control row.
    """

    dtype = _resolve_dtype(axon, dtype_local)
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        jnp.arange(nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(dt_ms, dtype=dtype)
    return build_vstim_batch(
        axon,
        contexts_batch,
        t_ms=t_mid_ms,
        x_positions_m=x_positions_m,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        dtype_local=dtype,
    )


def build_factorized_vstim_midpoint_batch(
    axon: object,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    tsim_ms: float,
    dt_ms: float,
    x_positions_m: Array | None = None,
    axon_y_um: Array | None = None,
    axon_z_um: Array | None = None,
    dtype_local: jnp.dtype | None = None,
    include_initial_previous: bool = False,
) -> FactorizedExtracellularPotentialBatch | None:
    """Build a factorized midpoint ``Vstim`` batch when the contexts allow it.

    The first supported case is the common shared point-source extracellular
    context. It keeps the static spatial footprint separate from the temporal
    stimulus so observer-only kernels can avoid materializing
    ``Vstim[B, Nt, Nx]``.
    """

    rows = tuple(_normalize_context_row(row) for row in contexts_batch)
    if not rows or not any(rows):
        return None
    if not _can_build_context_rows_from_footprints(rows):
        return None

    dtype = _resolve_dtype(axon, dtype_local)
    np_dtype = np.dtype(dtype)
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        np.arange(nt, dtype=np_dtype) + np.asarray(0.5, dtype=np_dtype)
    ) * np.asarray(dt_ms, dtype=np_dtype)
    t_initial_previous_ms = (
        np.asarray([-0.5 * dt_ms], dtype=np_dtype) if include_initial_previous else None
    )
    x_rows_np = _resolve_x_positions_m_numpy(
        axon,
        x_positions_m,
        batch_size=len(rows),
    )
    y_rows_np, z_rows_np = _resolve_axon_transverse_um_numpy(
        axon,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        batch_size=len(rows),
    )
    return _try_build_shared_point_source_factorized_vstim_batch(
        rows,
        t_mid_ms,
        t_initial_previous_ms=t_initial_previous_ms,
        x_rows=x_rows_np,
        axon_y_um=y_rows_np,
        axon_z_um=z_rows_np,
        np_dtype=np_dtype,
        dtype_local=dtype,
    )


def build_vstim_initial_previous_batch(
    axon: object,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    dt_ms: float,
    x_positions_m: Array | None = None,
    axon_y_um: Array | None = None,
    axon_z_um: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build the ``t=-dt/2`` imposed field required by double-cable batches.

    Returns ``Vstim[B, Nx]`` in millivolts.
    """

    dtype = _resolve_dtype(axon, dtype_local)
    samples = build_vstim_batch(
        axon,
        contexts_batch,
        t_ms=jnp.asarray([-0.5 * dt_ms], dtype=dtype),
        x_positions_m=x_positions_m,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        dtype_local=dtype,
    )
    return samples[:, 0, :]


def build_vstim_midpoint_and_initial_previous_batch(
    axon: object,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    tsim_ms: float,
    dt_ms: float,
    x_positions_m: Array | None = None,
    axon_y_um: Array | None = None,
    axon_z_um: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> tuple[Array, Array]:
    """Build double-cable midpoint and previous imposed fields together."""

    dtype = _resolve_dtype(axon, dtype_local)
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        jnp.arange(nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(dt_ms, dtype=dtype)
    t_all_ms = jnp.concatenate(
        [jnp.asarray([-0.5 * dt_ms], dtype=dtype), t_mid_ms],
        axis=0,
    )
    samples = build_vstim_batch(
        axon,
        contexts_batch,
        t_ms=t_all_ms,
        x_positions_m=x_positions_m,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        dtype_local=dtype,
    )
    return samples[:, 1:, :], samples[:, 0, :]


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
    if _can_build_intracellular_rows_from_clamps(axons):
        return _build_intracellular_current_density_batch_from_clamps(
            axons,
            resolved_solver_axons,
            t_mid,
            target_nx=target_width,
            dtype_local=dtype,
        )
    return jnp.stack(
        [
            _pad_time_space_array(
                _build_intracellular_current_density_row(
                    axon,
                    t_mid,
                    runtime=runtime,
                    solver_axon=resolved_solver_axons[index],
                    dtype_local=dtype,
                ),
                target_nx=target_width,
            )
            for index, axon in enumerate(axons)
        ],
        axis=0,
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
    current_cache: dict[int, np.ndarray] = {}

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
    current_cache: dict[int, np.ndarray] = {}

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


def _build_intracellular_current_density_row(
    axon: AxonLike,
    t_ms: Array,
    *,
    runtime: SolverRuntime,
    solver_axon: SolverAxon,
    dtype_local: jnp.dtype,
) -> Array:
    """Sample one row of compiled intracellular contexts."""

    compiled = compile_intracellular_contexts(
        axon,
        dtype_local=dtype_local,
        solver_axon=solver_axon,
    )
    return jax.vmap(compiled)(t_ms)


def build_footprint_vstim_midpoint_batch(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    tsim_ms: float,
    dt_ms: float,
    amplitude_scale: float | Array = 1.0,
    dtype_local: jnp.dtype | None = None,
    engine: FootprintEngine = "numpy",
) -> Array:
    """Build midpoint ``Vstim`` from precomputed electrode footprints.

    ``footprint_V_per_A`` has shape ``(Nx,)`` or ``(B, Nx)`` and is expressed in
    volts per ampere. The returned array has shape ``(B, Nt, Nx)`` and units of
    millivolts.
    """

    dtype = jnp.float32 if dtype_local is None else dtype_local
    nt = simulation_step_count(tsim_ms, dt_ms)
    t_mid_ms = (
        jnp.arange(nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(dt_ms, dtype=dtype)
    return build_footprint_vstim_batch(
        stimulus=stimulus,
        footprint_V_per_A=footprint_V_per_A,
        t_ms=t_mid_ms,
        amplitude_scale=amplitude_scale,
        dtype_local=dtype,
        engine=engine,
    )


def build_footprint_vstim_initial_previous_batch(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    dt_ms: float,
    amplitude_scale: float | Array = 1.0,
    dtype_local: jnp.dtype | None = None,
    engine: FootprintEngine = "numpy",
) -> Array:
    """Build the previous imposed field from precomputed footprints.

    Returns ``Vstim[B, Nx]`` sampled at ``t=-dt/2`` in millivolts.
    """

    dtype = jnp.float32 if dtype_local is None else dtype_local
    samples = build_footprint_vstim_batch(
        stimulus=stimulus,
        footprint_V_per_A=footprint_V_per_A,
        t_ms=jnp.asarray([-0.5 * dt_ms], dtype=dtype),
        amplitude_scale=amplitude_scale,
        dtype_local=dtype,
        engine=engine,
    )
    return samples[:, 0, :]


def build_footprint_vstim_batch(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    t_ms: Array,
    amplitude_scale: float | Array = 1.0,
    dtype_local: jnp.dtype | None = None,
    engine: FootprintEngine = "numpy",
) -> Array:
    """Build ``Vstim[B, Nt, Nx]`` from static footprints and one stimulus.

    The default NumPy path keeps data preparation outside JAX tracing and
    returns a JAX array for solver consumption. Use ``engine="jax"`` only when
    JAX-side multiplication is explicitly useful.
    """

    dtype = jnp.float32 if dtype_local is None else dtype_local
    if engine == "numpy":
        return _build_footprint_vstim_batch_numpy(
            stimulus=stimulus,
            footprint_V_per_A=footprint_V_per_A,
            t_ms=t_ms,
            amplitude_scale=amplitude_scale,
            dtype_local=dtype,
        )
    if engine != "jax":
        raise ValueError(f"engine must be 'numpy' or 'jax', got {engine!r}.")

    t = jnp.asarray(t_ms, dtype=dtype)
    if t.ndim != 1:
        raise ValueError(f"t_ms must have shape (Nt,), got {t.shape}.")

    batch_size = _infer_footprint_batch_size(footprint_V_per_A, (amplitude_scale,))
    footprint = _as_footprint_batch(
        "footprint_V_per_A",
        footprint_V_per_A,
        batch_size=batch_size,
        dtype_local=dtype,
    )
    scale = _as_batch_vector(
        "amplitude_scale",
        amplitude_scale,
        batch_size=batch_size,
        dtype_local=dtype,
    )
    current_A = jax.vmap(compile_stimulus(stimulus, dtype_local=dtype))(t)
    return (
        scale[:, None, None]
        * current_A[None, :, None]
        * footprint[:, None, :]
        * jnp.asarray(1e3, dtype=dtype)
    )


def build_vstim_batch(
    axon: object,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    t_ms: Array,
    x_positions_m: Array | None = None,
    axon_y_um: Array | None = None,
    axon_z_um: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build imposed extracellular samples for a batch of context rows.

    ``x_positions_m`` can have shape ``(Nx,)`` for shared positions or
    ``(B, Nx)`` when each row is spatially shifted relative to the electrode.
    """

    rows = tuple(_normalize_context_row(row) for row in contexts_batch)
    if not rows:
        raise ValueError("contexts_batch must contain at least one row.")

    dtype = _resolve_dtype(axon, dtype_local)
    t = jnp.asarray(t_ms, dtype=dtype)
    if t.ndim != 1:
        raise ValueError(f"t_ms must have shape (Nt,), got {t.shape}.")

    if not any(rows):
        nx = _resolve_x_positions_width(axon, x_positions_m, batch_size=len(rows))
        return jnp.zeros(
            (len(rows), int(t.shape[0]), nx),
            dtype=dtype,
        )

    if _can_build_context_rows_from_footprints(rows):
        x_rows_np = _resolve_x_positions_m_numpy(
            axon,
            x_positions_m,
            batch_size=len(rows),
        )
        y_rows_np, z_rows_np = _resolve_axon_transverse_um_numpy(
            axon,
            axon_y_um=axon_y_um,
            axon_z_um=axon_z_um,
            batch_size=len(rows),
        )
        return _build_vstim_batch_from_footprints(
            rows,
            t,
            x_rows=x_rows_np,
            axon_y_um=y_rows_np,
            axon_z_um=z_rows_np,
            dtype_local=dtype,
        )

    x_rows = _resolve_x_positions_m(
        axon,
        x_positions_m,
        batch_size=len(rows),
        dtype_local=dtype,
    )
    y_rows, z_rows = _resolve_axon_transverse_um(
        axon,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        batch_size=len(rows),
        dtype_local=dtype,
    )
    vstim_rows = [
        _build_vstim_row(
            row,
            t,
            x_positions_row_m=x_rows[i],
            axon_y_um=float(y_rows[i]),
            axon_z_um=float(z_rows[i]),
            dtype_local=dtype,
        )
        for i, row in enumerate(rows)
    ]
    return jnp.stack(vstim_rows, axis=0)


def _can_build_context_rows_from_footprints(
    rows: Sequence[tuple[ExtracellularContext, ...]],
) -> bool:
    for row in rows:
        for ctx in row:
            if not hasattr(ctx, "footprint_for_electrode"):
                return False
            for electrode in ctx.electrodes:
                if getattr(electrode, "stimulus", None) is None:
                    return False
    return True


def _build_vstim_batch_from_footprints(
    rows: Sequence[tuple[ExtracellularContext, ...]],
    t_ms: Array,
    *,
    x_rows: Array,
    axon_y_um: Array,
    axon_z_um: Array,
    dtype_local: jnp.dtype,
) -> Array:
    np_dtype = np.dtype(dtype_local)
    t = np.asarray(t_ms, dtype=np_dtype)
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(axon_y_um, dtype=float)
    z = np.asarray(axon_z_um, dtype=float)
    fast_point_source = _try_build_shared_point_source_vstim_batch(
        rows,
        t,
        x_rows=x,
        axon_y_um=y,
        axon_z_um=z,
        np_dtype=np_dtype,
        dtype_local=dtype_local,
    )
    if fast_point_source is not None:
        return fast_point_source

    values = np.zeros((len(rows), int(t.shape[0]), int(x.shape[1])), dtype=np_dtype)
    current_cache: dict[int, np.ndarray] = {}
    mV_per_V = np.asarray(1e3, dtype=np_dtype)

    for row_index, row in enumerate(rows):
        for ctx in row:
            for electrode in ctx.electrodes:
                stimulus = getattr(electrode, "stimulus", None)
                if stimulus is None:
                    raise ValueError(
                        "Each extracellular electrode must have an attached stimulus."
                    )
                cache_key = id(stimulus)
                current_A = current_cache.get(cache_key)
                if current_A is None:
                    current_A = np.asarray(
                        stimulus.evaluate(t, unit="ampere"),
                        dtype=np_dtype,
                    )
                    current_cache[cache_key] = current_A
                footprint = np.asarray(
                    ctx.footprint_for_electrode(
                        electrode,
                        x[row_index],
                        axon_y_um=float(y[row_index]),
                        axon_z_um=float(z[row_index]),
                    ),
                    dtype=np_dtype,
                )
                values[row_index] += current_A[:, None] * footprint[None, :] * mV_per_V
    return jnp.asarray(values, dtype=dtype_local)


def _try_build_shared_point_source_vstim_batch(
    rows: Sequence[tuple[ExtracellularContext, ...]],
    t_ms: np.ndarray,
    *,
    x_rows: np.ndarray,
    axon_y_um: np.ndarray,
    axon_z_um: np.ndarray,
    np_dtype: np.dtype[Any],
    dtype_local: jnp.dtype,
) -> Array | None:
    if not rows or any(len(row) != 1 for row in rows):
        return None
    first_context = rows[0][0]
    if not isinstance(first_context, AnalyticalExtracellularContext):
        return None
    if any(row[0] is not first_context for row in rows):
        return None
    if len(first_context.electrodes) != 1:
        return None
    electrode = first_context.electrodes[0]
    if not isinstance(electrode, PointSourceElectrode):
        return None
    stimulus = getattr(electrode, "stimulus", None)
    if stimulus is None:
        return None

    current_A = np.asarray(stimulus.evaluate(t_ms, unit="ampere"), dtype=np_dtype)
    cache_key = _point_source_footprint_cache_key(
        first_context,
        electrode,
        x_rows=x_rows,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        np_dtype=np_dtype,
    )
    footprint = _POINT_SOURCE_FOOTPRINT_CACHE.get(cache_key)
    if footprint is None:
        x_rel_m = x_rows - float(electrode.x0_m)
        y_rel_m = (float(electrode.y_um) - axon_y_um)[:, None] * 1e-6
        z_rel_m = (float(electrode.z_um) - axon_z_um)[:, None] * 1e-6
        radius_m = np.sqrt(x_rel_m**2 + y_rel_m**2 + z_rel_m**2)
        radius_m = np.maximum(radius_m, float(electrode.min_distance_m))
        footprint = np.asarray(
            1.0 / (4.0 * np.pi * float(first_context.sigma_S_m) * radius_m),
            dtype=np_dtype,
        )
        _POINT_SOURCE_FOOTPRINT_CACHE[cache_key] = footprint
        footprint_cache_status = "miss"
    else:
        footprint_cache_status = "hit"
    record_benchmark_metadata(
        vstim_footprint_cache=footprint_cache_status,
        vstim_footprint_cache_nbytes=int(footprint.nbytes),
    )
    values = (
        current_A[None, :, None]
        * footprint[:, None, :]
        * np.asarray(1e3, dtype=np_dtype)
    )
    return jnp.asarray(values, dtype=dtype_local)


def _try_build_shared_point_source_factorized_vstim_batch(
    rows: Sequence[tuple[ExtracellularContext, ...]],
    t_ms: np.ndarray,
    *,
    t_initial_previous_ms: np.ndarray | None = None,
    x_rows: np.ndarray,
    axon_y_um: np.ndarray,
    axon_z_um: np.ndarray,
    np_dtype: np.dtype[Any],
    dtype_local: jnp.dtype,
) -> FactorizedExtracellularPotentialBatch | None:
    if not rows or any(len(row) != 1 for row in rows):
        return None
    first_context = rows[0][0]
    if not isinstance(first_context, AnalyticalExtracellularContext):
        return None
    if any(row[0] is not first_context for row in rows):
        return None
    if len(first_context.electrodes) != 1:
        return None
    electrode = first_context.electrodes[0]
    if not isinstance(electrode, PointSourceElectrode):
        return None
    stimulus = getattr(electrode, "stimulus", None)
    if stimulus is None:
        return None

    current_A = np.asarray(stimulus.evaluate(t_ms, unit="ampere"), dtype=np_dtype)
    current_initial_previous_A = None
    if t_initial_previous_ms is not None:
        current_initial_previous_A = np.asarray(
            stimulus.evaluate(t_initial_previous_ms, unit="ampere"),
            dtype=np_dtype,
        ).reshape(-1)[0]
    cache_key = _point_source_footprint_cache_key(
        first_context,
        electrode,
        x_rows=x_rows,
        axon_y_um=axon_y_um,
        axon_z_um=axon_z_um,
        np_dtype=np_dtype,
    )
    footprint_V_per_A = _POINT_SOURCE_FOOTPRINT_CACHE.get(cache_key)
    if footprint_V_per_A is None:
        x_rel_m = x_rows - float(electrode.x0_m)
        y_rel_m = (float(electrode.y_um) - axon_y_um)[:, None] * 1e-6
        z_rel_m = (float(electrode.z_um) - axon_z_um)[:, None] * 1e-6
        radius_m = np.sqrt(x_rel_m**2 + y_rel_m**2 + z_rel_m**2)
        radius_m = np.maximum(radius_m, float(electrode.min_distance_m))
        footprint_V_per_A = np.asarray(
            1.0 / (4.0 * np.pi * float(first_context.sigma_S_m) * radius_m),
            dtype=np_dtype,
        )
        _POINT_SOURCE_FOOTPRINT_CACHE[cache_key] = footprint_V_per_A
        footprint_cache_status = "miss"
    else:
        footprint_cache_status = "hit"

    footprint_mV_per_A = footprint_V_per_A * np.asarray(1e3, dtype=np_dtype)
    dense_equivalent_nbytes = (
        int(len(rows)) * int(t_ms.shape[0]) * int(x_rows.shape[1]) * int(np_dtype.itemsize)
    )
    previous_nbytes = (
        0
        if current_initial_previous_A is None
        else int(np.asarray(current_initial_previous_A).nbytes)
    )
    factorized_nbytes = (
        int(current_A.nbytes) + int(footprint_mV_per_A.nbytes) + previous_nbytes
    )
    record_benchmark_metadata(
        vstim_footprint_cache=footprint_cache_status,
        vstim_footprint_cache_nbytes=int(footprint_V_per_A.nbytes),
        vstim_input_format="factorized_point_source",
        vstim_factorized_current_nbytes=int(current_A.nbytes),
        vstim_factorized_initial_previous_nbytes=previous_nbytes,
        vstim_factorized_footprint_nbytes=int(footprint_mV_per_A.nbytes),
        vstim_factorized_total_nbytes=factorized_nbytes,
        vstim_dense_equivalent_nbytes=dense_equivalent_nbytes,
        vstim_factorized_dense_ratio=(
            factorized_nbytes / float(dense_equivalent_nbytes)
            if dense_equivalent_nbytes
            else 0.0
        ),
    )
    return FactorizedExtracellularPotentialBatch(
        current_mid_A=jnp.asarray(current_A, dtype=dtype_local),
        footprint_mV_per_A=jnp.asarray(footprint_mV_per_A, dtype=dtype_local),
        target_nx=int(x_rows.shape[1]),
        current_initial_previous_A=(
            None
            if current_initial_previous_A is None
            else jnp.asarray(current_initial_previous_A, dtype=dtype_local)
        ),
    )


def _point_source_footprint_cache_key(
    context: AnalyticalExtracellularContext,
    electrode: PointSourceElectrode,
    *,
    x_rows: np.ndarray,
    axon_y_um: np.ndarray,
    axon_z_um: np.ndarray,
    np_dtype: np.dtype[Any],
) -> tuple[Any, ...]:
    return (
        "point_source_footprint",
        str(np_dtype),
        float(context.sigma_S_m),
        float(electrode.x0_m),
        float(electrode.y_um),
        float(electrode.z_um),
        float(electrode.min_distance_m),
        _array_content_key(x_rows),
        _array_content_key(axon_y_um),
        _array_content_key(axon_z_um),
    )


def _array_content_key(values: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    arr = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return tuple(int(dim) for dim in arr.shape), arr.dtype.str, digest


def _build_vstim_row(
    contexts: tuple[ExtracellularContext, ...],
    t_ms: Array,
    *,
    x_positions_row_m: Array,
    axon_y_um: float,
    axon_z_um: float,
    dtype_local: jnp.dtype,
) -> Array:
    nt = int(t_ms.shape[0])
    nx = int(x_positions_row_m.shape[0])
    vstim = jnp.zeros((nt, nx), dtype=dtype_local)
    for ctx in contexts:
        compiled = compile_extracellular_context(
            ctx,
            x_positions_row_m,
            dtype_local=dtype_local,
            axon_y_um=axon_y_um,
            axon_z_um=axon_z_um,
        )
        vstim = vstim + jax.vmap(compiled)(t_ms)
    return vstim * jnp.asarray(1e3, dtype=dtype_local)


def _pad_time_space_array(values: Array, *, target_nx: int) -> Array:
    """Pad a ``(Nt, Nx)`` array with trailing zero compartments."""

    arr = jnp.asarray(values)
    pad_count = int(target_nx) - int(arr.shape[-1])
    if pad_count < 0:
        raise ValueError(
            f"target_nx must be >= array width, got target_nx={target_nx}, "
            f"width={arr.shape[-1]}."
        )
    if pad_count == 0:
        return arr
    return jnp.pad(arr, ((0, 0), (0, pad_count)), mode="constant")


def _normalize_context_row(row: ContextBatchRow) -> tuple[ExtracellularContext, ...]:
    if row is None:
        return ()
    if isinstance(row, ExtracellularContext):
        return (row,)
    return tuple(row)


def _resolve_dtype(axon: object, dtype_local: jnp.dtype | None) -> jnp.dtype:
    if dtype_local is not None:
        return dtype_local
    for attr in ("dtype",):
        dtype = getattr(axon, attr, None)
        if dtype is not None:
            return _jax_scalar_dtype(dtype)
    membrane_models = getattr(axon, "membrane_models", None)
    if membrane_models:
        return _jax_scalar_dtype(membrane_models[0].dtype)
    return jnp.float32


def _jax_scalar_dtype(dtype_like: object) -> jnp.dtype:
    return jnp.float64 if np.dtype(cast(Any, dtype_like)).name == "float64" else jnp.float32


def _resolve_x_positions_m(
    axon: object,
    x_positions_m: Array | None,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    if x_positions_m is None:
        layout = getattr(axon, "layout", None)
        if layout is None:
            raise AttributeError("axon must expose a layout for spatial sampling.")
        x_um = layout.position_values(unit="micrometer")
        x = jnp.asarray(x_um, dtype=dtype_local) * jnp.asarray(1e-6, dtype=dtype_local)
    else:
        x = jnp.asarray(x_positions_m, dtype=dtype_local)

    if x.ndim == 1:
        return jnp.broadcast_to(x[jnp.newaxis, :], (batch_size, x.shape[0]))
    if x.ndim == 2:
        if x.shape[0] != batch_size:
            raise ValueError(
                f"x_positions_m has batch size {x.shape[0]}, expected {batch_size}."
            )
        return x
    raise ValueError(f"x_positions_m must have shape (Nx,) or (B, Nx), got {x.shape}.")


def _resolve_x_positions_width(
    axon: object,
    x_positions_m: Array | None,
    *,
    batch_size: int,
) -> int:
    if x_positions_m is None:
        layout = getattr(axon, "layout", None)
        if layout is None:
            raise AttributeError("axon must expose a layout for spatial sampling.")
        return int(np.asarray(layout.position_values(unit="micrometer")).shape[0])
    shape, ndim = _shape_and_ndim(x_positions_m)
    if ndim == 1:
        return int(shape[0])
    if ndim == 2:
        if int(shape[0]) != batch_size:
            raise ValueError(
                f"x_positions_m has batch size {shape[0]}, expected {batch_size}."
            )
        return int(shape[1])
    raise ValueError(f"x_positions_m must have shape (Nx,) or (B, Nx), got {shape}.")


def _resolve_x_positions_m_numpy(
    axon: object,
    x_positions_m: Array | None,
    *,
    batch_size: int,
) -> np.ndarray:
    if x_positions_m is None:
        layout = getattr(axon, "layout", None)
        if layout is None:
            raise AttributeError("axon must expose a layout for spatial sampling.")
        x_um = layout.position_values(unit="micrometer")
        x = np.asarray(x_um, dtype=float) * 1e-6
    else:
        x = np.asarray(x_positions_m, dtype=float)

    if x.ndim == 1:
        return np.broadcast_to(x[None, :], (batch_size, x.shape[0]))
    if x.ndim == 2:
        if x.shape[0] != batch_size:
            raise ValueError(
                f"x_positions_m has batch size {x.shape[0]}, expected {batch_size}."
            )
        return x
    raise ValueError(f"x_positions_m must have shape (Nx,) or (B, Nx), got {x.shape}.")


def _resolve_axon_transverse_um(
    axon: object,
    *,
    axon_y_um: Array | None,
    axon_z_um: Array | None,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array]:
    y = _as_batch_vector(
        "axon_y_um",
        float(getattr(axon, "y_um", 0.0)) if axon_y_um is None else axon_y_um,
        batch_size=batch_size,
        dtype_local=dtype_local,
    )
    z = _as_batch_vector(
        "axon_z_um",
        float(getattr(axon, "z_um", 0.0)) if axon_z_um is None else axon_z_um,
        batch_size=batch_size,
        dtype_local=dtype_local,
    )
    return y, z


def _resolve_axon_transverse_um_numpy(
    axon: object,
    *,
    axon_y_um: Array | None,
    axon_z_um: Array | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = _as_batch_vector_numpy(
        "axon_y_um",
        float(getattr(axon, "y_um", 0.0)) if axon_y_um is None else axon_y_um,
        batch_size=batch_size,
        dtype_local=np.dtype(float),
    )
    z = _as_batch_vector_numpy(
        "axon_z_um",
        float(getattr(axon, "z_um", 0.0)) if axon_z_um is None else axon_z_um,
        batch_size=batch_size,
        dtype_local=np.dtype(float),
    )
    return y, z


def _build_footprint_vstim_batch_numpy(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    t_ms: Array,
    amplitude_scale: float | Array,
    dtype_local: jnp.dtype,
) -> Array:
    np_dtype = np.dtype(dtype_local)
    t = np.asarray(t_ms, dtype=np_dtype)
    if t.ndim != 1:
        raise ValueError(f"t_ms must have shape (Nt,), got {t.shape}.")

    batch_size = _infer_footprint_batch_size(footprint_V_per_A, (amplitude_scale,))
    footprint = _as_footprint_batch_numpy(
        "footprint_V_per_A",
        footprint_V_per_A,
        batch_size=batch_size,
        dtype_local=np_dtype,
    )
    scale = _as_batch_vector_numpy(
        "amplitude_scale",
        amplitude_scale,
        batch_size=batch_size,
        dtype_local=np_dtype,
    )
    current_A = np.asarray(stimulus.evaluate(t, unit="ampere"), dtype=np_dtype)
    vstim_mV = scale[:, None, None] * current_A[None, :, None] * footprint[:, None, :]
    vstim_mV = vstim_mV * np.asarray(1e3, dtype=np_dtype)
    return jnp.asarray(vstim_mV, dtype=dtype_local)


def _infer_footprint_batch_size(footprint_V_per_A: Array, params: Sequence[object]) -> int:
    candidates: list[int] = []
    footprint_shape, footprint_ndim = _shape_and_ndim(footprint_V_per_A)
    if footprint_ndim == 2:
        candidates.append(int(footprint_shape[0]))
    elif footprint_ndim != 1:
        raise ValueError(
            "footprint_V_per_A must have shape (Nx,) or (B, Nx), "
            f"got {footprint_shape}."
        )

    for values in params:
        shape, ndim = _shape_and_ndim(values)
        if ndim == 1 and shape[0] != 1:
            candidates.append(int(shape[0]))
        elif ndim > 1:
            raise ValueError(f"batched parameter must be scalar or shape (B,), got {shape}.")

    if not candidates:
        return 1
    batch_size = candidates[0]
    if any(candidate != batch_size for candidate in candidates):
        raise ValueError(f"batched inputs disagree on batch size: {candidates}.")
    return batch_size


def _as_footprint_batch(
    name: str,
    values: Array,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        return jnp.broadcast_to(arr[jnp.newaxis, :], (batch_size, arr.shape[0]))
    if arr.ndim == 2:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (batch_size, arr.shape[1]))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")


def _as_footprint_batch_numpy(
    name: str,
    values: Array,
    *,
    batch_size: int,
    dtype_local: np.dtype,
) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        return np.broadcast_to(arr[None, :], (batch_size, arr.shape[0]))
    if arr.ndim == 2:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.broadcast_to(arr, (batch_size, arr.shape[1]))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")


def _as_batch_vector(
    name: str,
    values: float | Array,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return jnp.full((batch_size,), arr, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (batch_size,))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must be scalar or have shape (B,), got {arr.shape}.")


def _as_batch_vector_numpy(
    name: str,
    values: float | Array,
    *,
    batch_size: int,
    dtype_local: np.dtype,
) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return np.full((batch_size,), arr, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.broadcast_to(arr, (batch_size,))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must be scalar or have shape (B,), got {arr.shape}.")


def _shape_and_ndim(values: object) -> tuple[tuple[int, ...], int]:
    shape = getattr(values, "shape", None)
    ndim = getattr(values, "ndim", None)
    if shape is not None and ndim is not None:
        return tuple(int(dim) for dim in shape), int(ndim)
    arr = np.asarray(values)
    return arr.shape, arr.ndim


__all__ = [
    "ContextBatchRow",
    "FootprintEngine",
    "build_intracellular_current_density_batch",
    "build_sparse_intracellular_current_density_batch",
    "build_footprint_vstim_batch",
    "build_footprint_vstim_initial_previous_batch",
    "build_footprint_vstim_midpoint_batch",
    "build_vstim_batch",
    "build_vstim_initial_previous_batch",
    "build_vstim_midpoint_and_initial_previous_batch",
    "build_vstim_midpoint_batch",
    "can_build_sparse_intracellular_current_density_batch",
]
