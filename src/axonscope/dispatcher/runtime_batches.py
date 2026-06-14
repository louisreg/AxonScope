"""Dispatcher-owned assembly of batched solver input tensors.

The solver kernels operate on numeric arrays. This module is the boundary that
turns public stimulation descriptions, electrode footprints, intracellular
contexts, and per-axon positions into arrays for pool or batch execution.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.stimulation import ExtracellularContext, Stimulus
from axonscope.stimulation.runtime import (
    compile_extracellular_context,
    compile_intracellular_contexts,
    compile_stimulus,
)
from axonscope.solvers.runtime import SolverRuntime
from axonscope.solvers.axon_runtime import SolverAxon
from axonscope.solvers.common import simulation_step_count

Array = Any
AxonLike = Axon | AxonInstance
ContextBatchRow = ExtracellularContext | Sequence[ExtracellularContext] | None
FootprintEngine = Literal["numpy", "jax"]


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
    t_mid = (
        jnp.arange(runtime.grid.Nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(runtime.grid.dt_ms, dtype=dtype)
    return jnp.stack(
        [
            _pad_time_space_array(
                _build_intracellular_current_density_row(
                    axon,
                    t_mid,
                    runtime=runtime,
                    solver_axon=(
                        runtime.axon if solver_axons is None else solver_axons[index]
                    ),
                    dtype_local=dtype,
                ),
                target_nx=runtime.membrane.Nx if target_nx is None else target_nx,
            )
            for index, axon in enumerate(axons)
        ],
        axis=0,
    )


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


def extracellular_context_rows(
    axons: Sequence[AxonLike],
) -> tuple[tuple[ExtracellularContext, ...], ...]:
    """Return one enabled extracellular-context row per axon."""

    return tuple(
        tuple(axon.extracellular_contexts)
        if bool(getattr(axon, "use_extracellular", False))
        else ()
        for axon in axons
    )


def x_positions_batch_m(
    axons: Sequence[AxonLike],
    *,
    target_nx: int | None = None,
) -> np.ndarray:
    """Return batched axial positions in meters, including x offsets."""

    rows = []
    for axon in axons:
        x_offset_um = float(getattr(axon, "x_offset_um", 0.0))
        x_row = (
            np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
            + x_offset_um * 1e-6
        )
        if target_nx is not None:
            x_row = _pad_numpy_space_array(x_row, target_nx=target_nx)
        rows.append(x_row)
    return np.stack(rows, axis=0)


def axon_transverse_positions_um(axons: Sequence[AxonLike]) -> tuple[np.ndarray, np.ndarray]:
    """Return batched axon transverse y/z positions in micrometers."""

    y = np.asarray([float(getattr(axon, "y_um", 0.0)) for axon in axons], dtype=float)
    z = np.asarray([float(getattr(axon, "z_um", 0.0)) for axon in axons], dtype=float)
    return y, z


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


def scale_extracellular_contexts(
    contexts: Sequence[ExtracellularContext],
    scale: float,
) -> tuple[ExtracellularContext, ...]:
    """Return contexts with their current amplitudes scaled by ``scale``."""

    return tuple(
        ctx.with_electrodes(
            tuple(electrode.with_scaled_stimulus(scale) for electrode in ctx.electrodes)
        )
        for ctx in contexts
    )


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


def _pad_numpy_space_array(values: np.ndarray, *, target_nx: int) -> np.ndarray:
    """Pad one spatial vector by repeating the final position."""

    arr = np.asarray(values)
    pad_count = int(target_nx) - int(arr.shape[-1])
    if pad_count < 0:
        raise ValueError(
            f"target_nx must be >= array width, got target_nx={target_nx}, "
            f"width={arr.shape[-1]}."
        )
    if pad_count == 0:
        return arr
    if arr.shape[-1] == 0:
        raise ValueError("cannot pad an empty spatial row.")
    return np.pad(arr, (0, pad_count), mode="edge")


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
    return jnp.float64 if np.dtype(dtype_like).name == "float64" else jnp.float32


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
    "build_footprint_vstim_batch",
    "build_footprint_vstim_initial_previous_batch",
    "build_footprint_vstim_midpoint_batch",
    "build_vstim_batch",
    "build_vstim_initial_previous_batch",
    "build_vstim_midpoint_batch",
    "axon_transverse_positions_um",
    "extracellular_context_rows",
    "scale_extracellular_contexts",
    "x_positions_batch_m",
]
