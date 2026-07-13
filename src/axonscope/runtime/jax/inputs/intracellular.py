"""JAX materialization of intracellular input tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.runtime.input_payloads import SparseIntracellularCurrentDensityBatch
from axonscope.runtime.jax.inputs.stimulus import JaxStimulus, compile_stimulus
from axonscope.runtime.jax.types import SolverRuntime
from axonscope.runtime.solver_axon import SolverAxon
from axonscope.stimulation import (
    IntracellularContext,
    IntracellularCurrentClamp,
    Stimulus,
)

Array = Any
AxonLike = Axon | AxonInstance


@dataclass(frozen=True)
class CompiledIntracellularContexts:
    """JAX-ready intracellular current-density compiler output."""

    n_compartments: int
    dtype_local: Any
    basis: jnp.ndarray
    nA_to_mA_per_cm2: jnp.ndarray
    stimuli: tuple[JaxStimulus, ...]

    def __call__(self, t_ms):
        """Return injected current density in mA/cm^2 at one time."""

        if not self.stimuli:
            return jnp.zeros((self.n_compartments,), dtype=self.dtype_local)
        amps_nA = jnp.asarray([stim(t_ms) for stim in self.stimuli], dtype=self.dtype_local)
        densities = amps_nA * self.nA_to_mA_per_cm2
        return jnp.sum(densities[:, None] * self.basis, axis=0)


def compile_intracellular_contexts(
    axon: AxonLike,
    dtype_local: jnp.dtype | None = None,
    *,
    solver_axon: SolverAxon | None = None,
) -> CompiledIntracellularContexts:
    """Compile intracellular contexts to a current-density callable."""

    solver_data = _resolve_solver_axon(axon, solver_axon)
    if dtype_local is None:
        dtype_local = _axon_dtype(axon, solver_axon=solver_data)
    contexts = _intracellular_contexts_from_axon(axon)
    nx = solver_data.n_compartments

    if not contexts:
        return CompiledIntracellularContexts(
            n_compartments=nx,
            dtype_local=dtype_local,
            basis=jnp.zeros((0, nx), dtype=dtype_local),
            nA_to_mA_per_cm2=jnp.zeros((0,), dtype=dtype_local),
            stimuli=(),
        )

    x = jnp.asarray(solver_data.x_um, dtype=dtype_local)
    area_cm2 = compartment_surface_area_cm2(
        axon,
        dtype_local,
        solver_axon=solver_data,
    )

    idxs = []
    nA_to_mA_per_cm2 = []
    compiled_stimuli = []
    for context in contexts:
        if not isinstance(context, IntracellularCurrentClamp):
            raise NotImplementedError(
                "Only IntracellularCurrentClamp is currently supported by the "
                "intracellular runtime compiler."
            )
        idx = int(jnp.argmin(jnp.abs(x - dtype_local(context.position_um))))
        idxs.append(idx)
        nA_to_mA_per_cm2.append(dtype_local(1e-3) / area_cm2[idx])
        compiled_stimuli.append(compile_stimulus(context.current, dtype_local=dtype_local))

    return CompiledIntracellularContexts(
        n_compartments=nx,
        dtype_local=dtype_local,
        basis=jnp.eye(nx, dtype=dtype_local)[jnp.asarray(idxs, dtype=jnp.int32)],
        nA_to_mA_per_cm2=jnp.asarray(nA_to_mA_per_cm2, dtype=dtype_local),
        stimuli=tuple(compiled_stimuli),
    )


def build_intracellular_current_density_fn(
    axon: AxonLike,
    *,
    solver_axon: SolverAxon | None = None,
) -> CompiledIntracellularContexts:
    """Compile intracellular clamps into a current-density function."""

    return compile_intracellular_contexts(axon, solver_axon=solver_axon)


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
    return SparseIntracellularCurrentDensityBatch(
        density_mid=jnp.zeros((rows, steps, 0), dtype=dtype_local),
        indices=jnp.zeros((rows, 0), dtype=jnp.int32),
        mask=jnp.zeros((rows, 0), dtype=bool),
        target_nx=nx,
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


def compartment_surface_area_cm2(
    axon: AxonLike,
    dtype_local: jnp.dtype,
    *,
    solver_axon: SolverAxon | None = None,
) -> jnp.ndarray:
    """Return per-compartment membrane surface area in cm^2."""

    solver_data = _resolve_solver_axon(axon, solver_axon)
    diam_um = jnp.asarray(solver_data.diam_um, dtype=dtype_local)
    length_cm = (
        jnp.asarray(solver_data.compartment_lengths_um, dtype=dtype_local)
        * dtype_local(1e-4)
    )
    return jnp.pi * (diam_um * dtype_local(1e-4)) * length_cm


def _resolve_solver_axon(
    axon: AxonLike,
    solver_axon: SolverAxon | None,
) -> SolverAxon:
    """Return an existing solver axon or build one from a public axon object."""

    if solver_axon is not None:
        return solver_axon
    from axonscope.runtime.solver_axon import build_solver_axon

    return build_solver_axon(axon)


def _axon_dtype(
    axon: AxonLike,
    *,
    solver_axon: SolverAxon | None = None,
) -> jnp.dtype:
    """Return the JAX scalar dtype associated with an axon-like object."""

    if solver_axon is not None:
        return _jax_scalar_dtype(solver_axon.dtype)
    if hasattr(axon, "dtype"):
        return _jax_scalar_dtype(axon.dtype)
    layout = getattr(axon, "layout", None)
    if layout is not None:
        return _jax_scalar_dtype(layout.sections[0].membrane.dtype)
    return jnp.float32


def _jax_scalar_dtype(dtype_like: object) -> jnp.dtype:
    """Normalize NumPy-like dtype inputs to the supported JAX float dtype."""

    name = np.dtype(dtype_like).name
    if name == "float64":
        return jnp.float64
    return jnp.float32


def _intracellular_contexts_from_axon(
    axon: AxonLike,
) -> tuple[IntracellularContext, ...]:
    """Return intracellular contexts from a simulation-like object."""

    return tuple(getattr(axon, "intracellular_contexts", ()))


def _compartment_surface_area_cm2_numpy(solver_axon: SolverAxon) -> np.ndarray:
    diam_cm = np.asarray(solver_axon.diam_um, dtype=float) * 1e-4
    length_cm = np.asarray(solver_axon.compartment_lengths_um, dtype=float) * 1e-4
    return np.pi * diam_cm * length_cm


def _build_intracellular_current_density_row(
    axon: AxonLike,
    t_ms: Array,
    *,
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


__all__ = [
    "CompiledIntracellularContexts",
    "build_intracellular_current_density_batch",
    "build_intracellular_current_density_fn",
    "build_sparse_intracellular_current_density_batch",
    "build_zero_sparse_intracellular_current_density_batch",
    "can_build_sparse_intracellular_current_density_batch",
    "compile_intracellular_contexts",
    "compartment_surface_area_cm2",
]
