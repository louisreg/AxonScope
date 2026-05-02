from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from axonscope.axons.base import AxonBase

from .common import (
    Array,
    compartment_area_cm2,
    diffusion_operator_coeffs,
    extracellular_absolute_arrays,
    initial_voltage,
)
from .recording import membrane_observable_names
from .stimulus_runtime import (
    build_extracellular_potential_fn,
    build_intracellular_current_density_fn,
)


@dataclass(frozen=True)
class SimulationGrid:
    """Time discretization arrays owned by the solver runtime."""

    tsim_ms: float
    dt_ms: float
    Nt: int
    t_vec_ms: Array


@dataclass(frozen=True)
class MembraneRuntime:
    """Membrane backend and initial states extracted from an axon description."""

    backend: Any
    membrane: Any
    dtype: jnp.dtype
    Nx: int
    Vm0_mV: Array
    gates0: Array
    state0: tuple[Array, ...]
    background_current: Array
    observable_names: dict[str, tuple[str, ...]]
    diagnostic_names: tuple[str, ...]


@dataclass(frozen=True)
class CableRuntime:
    """Single-cable geometry/operator arrays used by solver kernels."""

    lower: Array
    diag: Array
    upper: Array
    area_cm2: Array


@dataclass(frozen=True)
class StimulationRuntime:
    """Compiled solver-side stimulation functions.

    The current fields are callables consumed by the single-axon solvers. The
    next batch-oriented step is to replace or supplement them with precomputed
    tensors such as Vstim[Nt, Nx] or Vstim[B, Nt, Nx].
    """

    intracellular_current_density: Callable[[float], Array]
    extracellular_potential_mV: Callable[[float], Array]
    has_driven_extracellular: bool
    intracellular_current_density_mid: Array | None = None
    extracellular_potential_mid_mV: Array | None = None
    extracellular_potential_initial_previous_mV: Array | None = None


@dataclass(frozen=True)
class ExtracellularRuntime:
    """Double-cable extracellular arrays for full Vi/Vperi reference solvers."""

    Cm_abs: Array
    Cx_abs: Array
    Gx_abs: Array
    Gax_e: Array
    Gax_i: Array
    left_i: Array
    right_i: Array
    left_e: Array
    right_e: Array


@dataclass(frozen=True)
class SolverRuntime:
    """Prepared single-axon runtime data consumed by solver kernels."""

    grid: SimulationGrid
    membrane: MembraneRuntime
    cable: CableRuntime
    stimulation: StimulationRuntime
    extracellular: ExtracellularRuntime | None = None


def prepare_simulation_grid(tsim_ms: float, dt_ms: float, dtype_local: jnp.dtype) -> SimulationGrid:
    Nt = int(jnp.ceil(tsim_ms / dt_ms))
    t_vec = (jnp.arange(Nt, dtype=dtype_local) + dtype_local(1.0)) * dt_ms
    return SimulationGrid(
        tsim_ms=float(tsim_ms),
        dt_ms=float(dt_ms),
        Nt=Nt,
        t_vec_ms=t_vec,
    )


def prepare_membrane_runtime(axon: AxonBase) -> MembraneRuntime:
    backend = axon.build_icm_backend()
    membrane = axon.ion_channel
    dtype_local = backend.dtype
    Nx = int(axon.Nx)
    Vm0 = initial_voltage(axon, Nx, dtype_local)
    gates0 = backend.init_gates(V0_mV=Vm0)
    state0 = membrane.init_membrane_state(Nx=Nx, dtype_local=dtype_local, V0_mV=Vm0)
    return MembraneRuntime(
        backend=backend,
        membrane=membrane,
        dtype=dtype_local,
        Nx=Nx,
        Vm0_mV=Vm0,
        gates0=gates0,
        state0=tuple(state0),
        background_current=backend.background_current(),
        observable_names=membrane_observable_names(membrane),
        diagnostic_names=membrane.diagnostic_names(),
    )


def prepare_cable_runtime(
    axon: AxonBase,
    dtype_local: jnp.dtype,
    *,
    include_area: bool = True,
) -> CableRuntime:
    lower, diag, upper = diffusion_operator_coeffs(axon, dtype_local)
    area = (
        compartment_area_cm2(axon, dtype_local)
        if include_area
        else jnp.zeros((axon.Nx,), dtype=dtype_local)
    )
    return CableRuntime(lower=lower, diag=diag, upper=upper, area_cm2=area)


def prepare_stimulation_runtime(
    axon: AxonBase,
    dtype_local: jnp.dtype,
    *,
    grid: SimulationGrid | None = None,
    precompute_intracellular: bool = False,
    precompute_extracellular: bool = False,
) -> StimulationRuntime:
    use_extracellular = bool(getattr(axon, "use_extracellular", False))
    inj_fun = build_intracellular_current_density_fn(axon)
    vext_fun = build_extracellular_potential_fn(axon)
    iinj_mid = None
    vext_mid = None
    vext_initial_previous = None
    if grid is not None and (precompute_intracellular or precompute_extracellular):
        t_mid = (jnp.arange(grid.Nt, dtype=dtype_local) + dtype_local(0.5)) * dtype_local(grid.dt_ms)
        if precompute_intracellular:
            iinj_mid = sample_intracellular_current_density(
                inj_fun,
                t_mid,
                dtype_local=dtype_local,
            )
    if precompute_extracellular and grid is not None:
        vext_mid = sample_extracellular_potential_mV(vext_fun, t_mid, dtype_local=dtype_local)
        vext_initial_previous = sample_extracellular_potential_mV(
            vext_fun,
            jnp.asarray([-0.5 * grid.dt_ms], dtype=dtype_local),
            dtype_local=dtype_local,
        )[0]
    return StimulationRuntime(
        intracellular_current_density=inj_fun,
        extracellular_potential_mV=vext_fun,
        has_driven_extracellular=bool(
            use_extracellular and getattr(axon, "extracellular_contexts", ())
        ),
        intracellular_current_density_mid=iinj_mid,
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_initial_previous,
    )


def prepare_extracellular_runtime(
    axon: AxonBase,
    dtype_local: jnp.dtype,
    cable: CableRuntime,
) -> ExtracellularRuntime:
    Cm_abs, Cx_abs, Gx_abs, Gax_e = extracellular_absolute_arrays(axon, dtype_local)
    Gax_i = 0.5 * (cable.upper[:-1] * Cm_abs[:-1] + cable.lower[1:] * Cm_abs[1:])
    left_i = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_i])
    right_i = jnp.concatenate([Gax_i, jnp.zeros((1,), dtype=dtype_local)])
    left_e = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_e])
    right_e = jnp.concatenate([Gax_e, jnp.zeros((1,), dtype=dtype_local)])
    return ExtracellularRuntime(
        Cm_abs=Cm_abs,
        Cx_abs=Cx_abs,
        Gx_abs=Gx_abs,
        Gax_e=Gax_e,
        Gax_i=Gax_i,
        left_i=left_i,
        right_i=right_i,
        left_e=left_e,
        right_e=right_e,
    )


def prepare_solver_runtime(
    axon: AxonBase,
    tsim_ms: float,
    dt_ms: float,
    *,
    include_extracellular: bool | None = None,
    include_area: bool | None = None,
    precompute_intracellular: bool = False,
    precompute_extracellular: bool | None = None,
) -> SolverRuntime:
    membrane = prepare_membrane_runtime(axon)
    grid = prepare_simulation_grid(tsim_ms, dt_ms, membrane.dtype)
    if include_extracellular is None:
        include_extracellular = bool(getattr(axon, "use_extracellular", False))
    if precompute_extracellular is None:
        precompute_extracellular = include_extracellular
    if include_area is None:
        include_area = True
    cable = prepare_cable_runtime(axon, membrane.dtype, include_area=include_area)
    stimulation = prepare_stimulation_runtime(
        axon,
        membrane.dtype,
        grid=grid,
        precompute_intracellular=precompute_intracellular,
        precompute_extracellular=precompute_extracellular,
    )
    extracellular = (
        prepare_extracellular_runtime(axon, membrane.dtype, cable)
        if include_extracellular
        else None
    )
    return SolverRuntime(
        grid=grid,
        membrane=membrane,
        cable=cable,
        stimulation=stimulation,
        extracellular=extracellular,
    )


def precompute_extracellular_potential_mV(
    axon: AxonBase,
    t_ms: Array,
    *,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Sample imposed Vstim on a time grid, returning shape (Nt, Nx).

    This utility is intentionally solver-side: the axon remains a descriptive
    object, while solvers decide whether they want callable stimulation or a
    precomputed tensor suitable for batching.
    """
    if dtype_local is None:
        runtime = prepare_membrane_runtime(axon)
        dtype_local = runtime.dtype
    t = jnp.asarray(t_ms, dtype=dtype_local)
    vext_fun = build_extracellular_potential_fn(axon)
    return sample_extracellular_potential_mV(vext_fun, t, dtype_local=dtype_local)


def precompute_intracellular_current_density(
    axon: AxonBase,
    t_ms: Array,
    *,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Sample intracellular current density on a time grid, returning shape (Nt, Nx)."""
    if dtype_local is None:
        runtime = prepare_membrane_runtime(axon)
        dtype_local = runtime.dtype
    t = jnp.asarray(t_ms, dtype=dtype_local)
    inj_fun = build_intracellular_current_density_fn(axon)
    return sample_intracellular_current_density(inj_fun, t, dtype_local=dtype_local)


def sample_intracellular_current_density(
    current_density_fn: Callable[[float], Array],
    t_ms: Array,
    *,
    dtype_local: jnp.dtype,
) -> Array:
    """Sample a compiled intracellular current-density function on a time grid."""
    t = jnp.asarray(t_ms, dtype=dtype_local)
    return jax.vmap(current_density_fn)(t)


def sample_extracellular_potential_mV(
    potential_fn: Callable[[float], Array],
    t_ms: Array,
    *,
    dtype_local: jnp.dtype,
) -> Array:
    """Sample a compiled Vstim function on a time grid, returning shape (Nt, Nx)."""
    t = jnp.asarray(t_ms, dtype=dtype_local)
    return jax.vmap(potential_fn)(t)
