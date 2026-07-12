from __future__ import annotations

import hashlib
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.axons.axon import Axon
from axonscope.runtime.jax.membranes.backend import (
    HeterogeneousMembraneBackend,
    MembraneBackend,
)
from axonscope.runtime.jax.membranes.compile import (
    backend_from_membrane,
    compile_axon_membrane,
)
from axonscope.runtime.solver_axon import SolverAxon, build_solver_axon
from axonscope.timebase import simulation_step_count

from axonscope.runtime.jax.cable_geometry import (
    Array,
    compartment_area_cm2,
    diffusion_operator_coeffs,
    extracellular_absolute_arrays,
    initial_voltage,
)
from axonscope.solvers.options import SolverOptions
from axonscope.runtime.jax.inputs.extracellular import build_extracellular_potential_fn
from axonscope.runtime.jax.inputs.intracellular import build_intracellular_current_density_fn
from axonscope.runtime.jax.types import (
    CableRuntime,
    ExtracellularRuntime,
    MembraneRuntime,
    SimulationGrid,
    SolverRuntime,
    StimulationRuntime,
)


def membrane_observable_names(membrane: Any) -> dict[str, tuple[str, ...]]:
    return {
        "gates": membrane.gate_names(),
        "currents": membrane.current_names(),
        "conductances": membrane.conductance_names(),
        "states": membrane.membrane_state_names(),
    }


_MEMBRANE_RUNTIME_CACHE: dict[tuple[Any, ...], MembraneRuntime] = {}
_CABLE_RUNTIME_CACHE: dict[tuple[Any, ...], CableRuntime] = {}
_EXTRACELLULAR_RUNTIME_CACHE: dict[tuple[Any, ...], ExtracellularRuntime] = {}
_SOLVER_RUNTIME_CACHE: dict[tuple[Any, ...], SolverRuntime] = {}


def _resolve_solver_options(options: SolverOptions | None) -> SolverOptions:
    return SolverOptions() if options is None else options


def _solver_options_cache_key(options: SolverOptions) -> tuple[Any, ...]:
    _ = options
    return ("solver_options",)


def _membrane_runtime_cache_key(
    axon: Axon,
    solver_data: SolverAxon,
    options: SolverOptions,
    *,
    membrane_signatures: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    return (
        "membrane_runtime",
        (
            tuple(model._static_signature() for model in solver_data.membrane_models)
            if membrane_signatures is None
            else membrane_signatures
        ),
        _solver_options_cache_key(options),
        int(solver_data.n_compartments),
        solver_data.dtype.str,
        float(getattr(axon, "v_init", 0.0)),
        float(getattr(axon, "temperature", 0.0)),
    )


def _array_cache_key(values: Any) -> tuple[tuple[int, ...], str, str]:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest()
    return tuple(array.shape), array.dtype.str, digest


def _solver_cable_cache_key(axon: SolverAxon) -> tuple[Any, ...]:
    return (
        "solver_cable",
        axon.formulation,
        axon.dtype.str,
        int(axon.n_compartments),
        bool(axon.has_heterogeneous_cable_properties),
        _array_cache_key(axon.compartment_lengths_um),
        _array_cache_key(axon.h_cm),
        _array_cache_key(axon.diam_um),
        _array_cache_key(axon.Ra_ohm_cm),
        _array_cache_key(axon.Cm_uF_cm2),
    )


def _cable_runtime_cache_key(
    axon: SolverAxon,
    dtype_local: jnp.dtype,
    *,
    include_area: bool,
) -> tuple[Any, ...]:
    return (
        "cable_runtime",
        _solver_cable_cache_key(axon),
        np.dtype(dtype_local).str,
        bool(include_area),
    )


def _extracellular_runtime_cache_key(
    axon: SolverAxon,
    dtype_local: jnp.dtype,
) -> tuple[Any, ...]:
    return (
        "extracellular_runtime",
        _solver_cable_cache_key(axon),
        np.dtype(dtype_local).str,
        _array_cache_key(axon.dx_cm),
        _array_cache_key(axon.xraxial_MOhm_per_cm),
        _array_cache_key(axon.xg_S_cm2),
        _array_cache_key(axon.xc_uF_cm2),
    )


def _solver_runtime_cache_key(
    axon: Axon,
    solver_axon: SolverAxon,
    options: SolverOptions,
    membrane: MembraneRuntime,
    *,
    tsim_ms: float,
    dt_ms: float,
    include_extracellular: bool,
    include_area: bool,
) -> tuple[Any, ...]:
    return (
        "solver_runtime",
        _membrane_runtime_cache_key(axon, solver_axon, options),
        _solver_cable_cache_key(solver_axon),
        float(tsim_ms),
        float(dt_ms),
        np.dtype(membrane.dtype).str,
        bool(include_extracellular),
        bool(include_area),
        bool(getattr(axon, "use_extracellular", False)),
        bool(getattr(axon, "extracellular_stimulations", ())),
    )


def _can_cache_solver_runtime(
    *,
    compile_stimulation: bool,
    precompute_intracellular: bool,
    precompute_extracellular: bool,
) -> bool:
    """Return whether the whole runtime is independent of drive amplitudes."""

    return (
        not bool(compile_stimulation)
        and not bool(precompute_intracellular)
        and not bool(precompute_extracellular)
    )


def prepare_simulation_grid(tsim_ms: float, dt_ms: float, dtype_local: jnp.dtype) -> SimulationGrid:
    Nt = simulation_step_count(tsim_ms, dt_ms)
    t_vec = (
        jnp.arange(Nt, dtype=dtype_local) + jnp.asarray(1.0, dtype=dtype_local)
    ) * jnp.asarray(dt_ms, dtype=dtype_local)
    return SimulationGrid(
        tsim_ms=float(tsim_ms),
        dt_ms=float(dt_ms),
        Nt=Nt,
        t_vec_ms=t_vec,
    )


def prepare_membrane_runtime(
    axon: Axon,
    *,
    solver_axon: SolverAxon | None = None,
    solver_options: SolverOptions | None = None,
) -> MembraneRuntime:
    options = _resolve_solver_options(solver_options)
    solver_data = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane_signatures = tuple(model._static_signature() for model in solver_data.membrane_models)
    cache_key = _membrane_runtime_cache_key(
        axon,
        solver_data,
        options,
        membrane_signatures=membrane_signatures,
    )
    cached = _MEMBRANE_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        record_benchmark_metadata(membrane_runtime_cache="hit")
        return cached
    record_benchmark_metadata(membrane_runtime_cache="miss")
    with benchmark_span(
        "runtime.prepare.membrane_compile",
        nx=int(solver_data.n_compartments),
    ):
        membrane = compile_axon_membrane(
            axon,
            solver_axon=solver_data,
            solver_options=options,
            membrane_signatures=membrane_signatures,
        )
    with benchmark_span(
        "runtime.prepare.membrane_backend",
        nx=int(solver_data.n_compartments),
    ):
        backend = backend_from_membrane(membrane, int(solver_data.n_compartments))
    dtype_local = backend.dtype
    Nx = int(solver_data.n_compartments)
    with benchmark_span("runtime.prepare.membrane_init", nx=Nx):
        Vm0, gates0, state0, background_current = _prepare_membrane_initial_arrays(
            axon,
            membrane,
            backend,
            nx=Nx,
            dtype_local=dtype_local,
        )
    runtime = MembraneRuntime(
        backend=backend,
        membrane=membrane,
        dtype=dtype_local,
        Nx=Nx,
        Vm0_mV=Vm0,
        gates0=gates0,
        state0=tuple(state0),
        background_current=background_current,
        observable_names=membrane_observable_names(membrane),
        diagnostic_names=membrane.diagnostic_names(),
    )
    _MEMBRANE_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _prepare_membrane_initial_arrays(
    axon: Axon,
    membrane: Any,
    backend: MembraneBackend,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array, tuple[Array, ...], Array]:
    """Prepare initial membrane arrays without unnecessary eager JAX scatter."""

    if isinstance(backend, HeterogeneousMembraneBackend) and not membrane.membrane_state_specs():
        record_benchmark_metadata(membrane_init_source="heterogeneous_numpy")
        return _prepare_heterogeneous_membrane_initial_arrays(
            axon,
            backend,
            nx=nx,
            dtype_local=dtype_local,
        )
    record_benchmark_metadata(membrane_init_source="backend_jax")
    Vm0 = initial_voltage(axon, nx, dtype_local)
    gates0 = backend.init_gates(V0_mV=Vm0)
    state0 = membrane.init_membrane_state(Nx=nx, dtype_local=dtype_local, V0_mV=Vm0)
    background_current = backend.background_current()
    return Vm0, gates0, tuple(state0), background_current


def _prepare_heterogeneous_membrane_initial_arrays(
    axon: Axon,
    backend: HeterogeneousMembraneBackend,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array, tuple[Array, ...], Array]:
    """Host-side initial arrays for heterogeneous compartment membranes."""

    np_dtype = np.dtype(dtype_local)
    vm0_np = np.full((nx,), float(getattr(axon, "v_init", 0.0)), dtype=np_dtype)
    gates_np = np.zeros((nx, backend.n_gates_max), dtype=np_dtype)
    background_np = np.zeros((nx,), dtype=np_dtype)
    for group in backend.groups:
        indices = np.asarray(group.indices, dtype=np.int64)
        if group.gate_size:
            local_v = jnp.asarray([vm0_np[indices[0]]], dtype=dtype_local)
            local_gates = np.asarray(
                group.model.init_gates(local_v),
                dtype=np_dtype,
            )
            gates_np[indices, : group.gate_size] = np.asarray(
                np.broadcast_to(local_gates, (len(indices), group.gate_size)),
                dtype=np_dtype,
            )
        background_np[indices] = np.asarray(
            group.model.I_background(len(indices)),
            dtype=np_dtype,
        )
    return (
        jnp.asarray(vm0_np, dtype=dtype_local),
        jnp.asarray(gates_np, dtype=dtype_local),
        (),
        jnp.asarray(background_np, dtype=dtype_local),
    )


def prepare_cable_runtime(
    axon: SolverAxon,
    dtype_local: jnp.dtype,
    *,
    include_area: bool = True,
) -> CableRuntime:
    cache_key = _cable_runtime_cache_key(axon, dtype_local, include_area=include_area)
    cached = _CABLE_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached
    lower, diag, upper = diffusion_operator_coeffs(axon, dtype_local)
    area = (
        compartment_area_cm2(axon, dtype_local)
        if include_area
        else jnp.zeros((axon.n_compartments,), dtype=dtype_local)
    )
    runtime = CableRuntime(lower=lower, diag=diag, upper=upper, area_cm2=area)
    _CABLE_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def prepare_stimulation_runtime(
    axon: Axon,
    solver_axon: SolverAxon,
    dtype_local: jnp.dtype,
    *,
    grid: SimulationGrid | None = None,
    precompute_intracellular: bool = False,
    precompute_extracellular: bool = False,
    compile_callables: bool = True,
) -> StimulationRuntime:
    use_extracellular = bool(getattr(axon, "use_extracellular", False))
    if compile_callables:
        inj_fun = build_intracellular_current_density_fn(axon, solver_axon=solver_axon)
        vext_fun = build_extracellular_potential_fn(axon, solver_axon=solver_axon)
    else:
        nx = int(solver_axon.n_compartments)

        def inj_fun(_: float) -> Array:
            return jnp.zeros((nx,), dtype=dtype_local)

        def vext_fun(_: float) -> Array:
            return jnp.zeros((nx,), dtype=dtype_local)

    iinj_mid = None
    vext_mid = None
    vext_initial_previous = None
    if grid is not None and (precompute_intracellular or precompute_extracellular):
        t_mid = (
            jnp.arange(grid.Nt, dtype=dtype_local) + jnp.asarray(0.5, dtype=dtype_local)
        ) * jnp.asarray(grid.dt_ms, dtype=dtype_local)
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
            use_extracellular and getattr(axon, "extracellular_stimulations", ())
        ),
        intracellular_current_density_mid=iinj_mid,
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_initial_previous,
    )


def prepare_extracellular_runtime(
    axon: SolverAxon,
    dtype_local: jnp.dtype,
    cable: CableRuntime,
) -> ExtracellularRuntime:
    cache_key = _extracellular_runtime_cache_key(axon, dtype_local)
    cached = _EXTRACELLULAR_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached
    Cm_abs, Cx_abs, Gx_abs, Gax_e = extracellular_absolute_arrays(axon, dtype_local)
    Gax_i = 0.5 * (cable.upper[:-1] * Cm_abs[:-1] + cable.lower[1:] * Cm_abs[1:])
    left_i = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_i])
    right_i = jnp.concatenate([Gax_i, jnp.zeros((1,), dtype=dtype_local)])
    left_e = jnp.concatenate([jnp.zeros((1,), dtype=dtype_local), Gax_e])
    right_e = jnp.concatenate([Gax_e, jnp.zeros((1,), dtype=dtype_local)])
    runtime = ExtracellularRuntime(
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
    _EXTRACELLULAR_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def prepare_solver_runtime(
    axon: Axon,
    tsim_ms: float,
    dt_ms: float,
    *,
    solver_axon: SolverAxon | None = None,
    include_extracellular: bool | None = None,
    include_area: bool | None = None,
    precompute_intracellular: bool = False,
    precompute_extracellular: bool | None = None,
    compile_stimulation: bool = True,
    solver_options: SolverOptions | None = None,
) -> SolverRuntime:
    solver_axon = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane = prepare_membrane_runtime(
        axon,
        solver_axon=solver_axon,
        solver_options=solver_options,
    )
    if include_extracellular is None:
        include_extracellular = bool(getattr(axon, "use_extracellular", False))
    if precompute_extracellular is None:
        precompute_extracellular = include_extracellular
    if include_area is None:
        include_area = True
    options = _resolve_solver_options(solver_options)
    cache_key = None
    if _can_cache_solver_runtime(
        compile_stimulation=compile_stimulation,
        precompute_intracellular=precompute_intracellular,
        precompute_extracellular=precompute_extracellular,
    ):
        cache_key = _solver_runtime_cache_key(
            axon,
            solver_axon,
            options,
            membrane,
            tsim_ms=tsim_ms,
            dt_ms=dt_ms,
            include_extracellular=include_extracellular,
            include_area=include_area,
        )
        cached = _SOLVER_RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            return cached

    grid = prepare_simulation_grid(tsim_ms, dt_ms, membrane.dtype)
    cable = prepare_cable_runtime(solver_axon, membrane.dtype, include_area=include_area)
    stimulation = prepare_stimulation_runtime(
        axon,
        solver_axon,
        membrane.dtype,
        grid=grid,
        precompute_intracellular=precompute_intracellular,
        precompute_extracellular=precompute_extracellular,
        compile_callables=compile_stimulation,
    )
    extracellular = (
        prepare_extracellular_runtime(solver_axon, membrane.dtype, cable)
        if include_extracellular
        else None
    )
    runtime = SolverRuntime(
        axon=solver_axon,
        grid=grid,
        membrane=membrane,
        cable=cable,
        stimulation=stimulation,
        extracellular=extracellular,
    )
    if cache_key is not None:
        _SOLVER_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def precompute_extracellular_potential_mV(
    axon: Axon,
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
        solver_axon = build_solver_axon(axon)
        runtime = prepare_membrane_runtime(axon, solver_axon=solver_axon)
        dtype_local = runtime.dtype
    else:
        solver_axon = build_solver_axon(axon)
    t = jnp.asarray(t_ms, dtype=dtype_local)
    vext_fun = build_extracellular_potential_fn(axon, solver_axon=solver_axon)
    return sample_extracellular_potential_mV(vext_fun, t, dtype_local=dtype_local)


def sample_intracellular_current_density(
    current_density_fn: Callable[[float], Array],
    t_ms: Array,
    *,
    dtype_local: jnp.dtype,
) -> Array:
    """Sample a compiled intracellular current-density function on a time grid."""
    t = jnp.asarray(t_ms, dtype=dtype_local)
    sampled_fn = cast(Callable[[Array], Array], current_density_fn)
    return jax.vmap(sampled_fn)(t)


def sample_extracellular_potential_mV(
    potential_fn: Callable[[float], Array],
    t_ms: Array,
    *,
    dtype_local: jnp.dtype,
) -> Array:
    """Sample a compiled Vstim function on a time grid, returning shape (Nt, Nx)."""
    t = jnp.asarray(t_ms, dtype=dtype_local)
    sampled_fn = cast(Callable[[Array], Array], potential_fn)
    return jax.vmap(sampled_fn)(t)
