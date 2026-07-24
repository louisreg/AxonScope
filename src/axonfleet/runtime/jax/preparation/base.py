from __future__ import annotations

import hashlib
from typing import Any

import jax.numpy as jnp
import numpy as np

from axonfleet.benchmarking import benchmark_span, record_benchmark_metadata
from axonfleet.axons.axon import Axon
from axonfleet.runtime.jax.membranes.backend import (
    HeterogeneousMembraneBackend,
    MembraneBackend,
    UniformMembraneBackend,
)
from axonfleet.runtime.jax.membranes.compile import (
    backend_from_membrane,
    compile_axon_membrane,
)
from axonfleet.runtime.jax.membranes.program import JaxMembraneProgram
from axonfleet.runtime.solver_axon import SolverAxon, build_solver_axon
from axonfleet.runtime.host_preparation import (
    compartment_area_cm2_numpy,
    diffusion_operator_coeffs_numpy,
    extracellular_runtime_numpy,
)
from axonfleet.runtime.timebase import simulation_step_count

from axonfleet.runtime.jax.cable_geometry import Array
from axonfleet.runtime.jax.types import (
    CableRuntime,
    ExtracellularRuntime,
    MembraneRuntime,
    SimulationGrid,
    SolverRuntime,
)


def membrane_observable_names(membrane: Any) -> dict[str, tuple[str, ...]]:
    return {
        "gates": membrane.gate_names(),
        "occupancies": membrane.occupancy_names(),
        "currents": membrane.current_names(),
        "conductances": membrane.conductance_names(),
        "states": membrane.membrane_state_names(),
    }


_MEMBRANE_RUNTIME_CACHE: dict[tuple[Any, ...], MembraneRuntime] = {}
_CABLE_RUNTIME_CACHE: dict[tuple[Any, ...], CableRuntime] = {}
_EXTRACELLULAR_RUNTIME_CACHE: dict[tuple[Any, ...], ExtracellularRuntime] = {}
_SOLVER_RUNTIME_CACHE: dict[tuple[Any, ...], SolverRuntime] = {}


def _membrane_runtime_cache_key(
    axon: Axon,
    solver_data: SolverAxon,
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
    membrane: MembraneRuntime,
    *,
    tsim_ms: float,
    dt_ms: float,
    include_extracellular: bool,
    include_area: bool,
) -> tuple[Any, ...]:
    return (
        "solver_runtime",
        _membrane_runtime_cache_key(axon, solver_axon),
        _solver_cable_cache_key(solver_axon),
        float(tsim_ms),
        float(dt_ms),
        np.dtype(membrane.dtype).str,
        bool(include_extracellular),
        bool(include_area),
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
) -> MembraneRuntime:
    solver_data = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane_signatures = tuple(model._static_signature() for model in solver_data.membrane_models)
    cache_key = _membrane_runtime_cache_key(
        axon,
        solver_data,
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

    if isinstance(backend, UniformMembraneBackend) and isinstance(
        membrane, JaxMembraneProgram
    ) and not membrane.membrane_state_specs():
        record_benchmark_metadata(membrane_init_source="uniform_numpy")
        return _prepare_uniform_model_ir_initial_arrays(
            axon,
            membrane,
            backend,
            nx=nx,
            dtype_local=dtype_local,
        )
    if isinstance(backend, HeterogeneousMembraneBackend) and not membrane.membrane_state_specs():
        source = (
            "heterogeneous_model_ir_numpy"
            if all(isinstance(group.model, JaxMembraneProgram) for group in backend.groups)
            else "heterogeneous_numpy"
        )
        record_benchmark_metadata(membrane_init_source=source)
        return _prepare_heterogeneous_membrane_initial_arrays(
            axon,
            backend,
            nx=nx,
            dtype_local=dtype_local,
        )
    record_benchmark_metadata(membrane_init_source="backend_jax")
    Vm0 = jnp.full((nx,), axon.v_init, dtype=dtype_local)
    gates0 = backend.init_gates(V0_mV=Vm0)
    state0 = membrane.init_membrane_state(Nx=nx, dtype_local=dtype_local, V0_mV=Vm0)
    background_current = backend.background_current()
    return Vm0, gates0, tuple(state0), background_current


def _prepare_uniform_model_ir_initial_arrays(
    axon: Axon,
    membrane: JaxMembraneProgram,
    backend: UniformMembraneBackend,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array, tuple[Array, ...], Array]:
    """Host-side initial arrays for uniform stateless Model IR membranes."""

    _ = backend
    np_dtype = np.dtype(dtype_local)
    v_init = float(getattr(axon, "v_init", 0.0))
    vm0_np = np.full((nx,), v_init, dtype=np_dtype)
    gate_row = membrane.init_gates_host(
        np.asarray([v_init], dtype=np_dtype),
        dtype_local=np_dtype,
    )[0]
    gates_np = np.array(
        np.broadcast_to(gate_row, (nx, gate_row.shape[0])),
        dtype=np_dtype,
        copy=True,
    )
    background_np = np.zeros((nx,), dtype=np_dtype)
    return (
        jnp.asarray(vm0_np, dtype=dtype_local),
        jnp.asarray(gates_np, dtype=dtype_local),
        (),
        jnp.asarray(background_np, dtype=dtype_local),
    )


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
            local_v = np.asarray([vm0_np[indices[0]]], dtype=np_dtype)
            local_gates = _initial_gates_for_heterogeneous_group(
                group.model,
                local_v,
                dtype_local=dtype_local,
                np_dtype=np_dtype,
            )
            gates_np[indices, : group.gate_size] = np.asarray(
                np.broadcast_to(local_gates, (len(indices), group.gate_size)),
                dtype=np_dtype,
            )
        background_np[indices] = _background_current_for_heterogeneous_group(
            group.model,
            len(indices),
            dtype_local=dtype_local,
            np_dtype=np_dtype,
        )
    return (
        jnp.asarray(vm0_np, dtype=dtype_local),
        jnp.asarray(gates_np, dtype=dtype_local),
        (),
        jnp.asarray(background_np, dtype=dtype_local),
    )


def _initial_gates_for_heterogeneous_group(
    model: Any,
    local_v_np: np.ndarray,
    *,
    dtype_local: jnp.dtype,
    np_dtype: np.dtype,
) -> np.ndarray:
    if isinstance(model, JaxMembraneProgram):
        return model.init_gates_host(local_v_np, dtype_local=np_dtype)
    local_v = jnp.asarray(local_v_np, dtype=dtype_local)
    return np.asarray(model.init_gates(local_v), dtype=np_dtype)


def _background_current_for_heterogeneous_group(
    model: Any,
    count: int,
    *,
    dtype_local: jnp.dtype,
    np_dtype: np.dtype,
) -> np.ndarray:
    if isinstance(model, JaxMembraneProgram):
        return np.zeros((count,), dtype=np_dtype)
    return np.asarray(model.I_background(count), dtype=np_dtype)


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
    np_dtype = np.dtype(dtype_local)
    lower, diag, upper = diffusion_operator_coeffs_numpy(axon, dtype=np_dtype)
    area = (
        compartment_area_cm2_numpy(axon, dtype=np_dtype)
        if include_area
        else np.zeros((axon.n_compartments,), dtype=np_dtype)
    )
    record_benchmark_metadata(cable_runtime_source="numpy")
    runtime = CableRuntime(
        lower=jnp.asarray(lower, dtype=dtype_local),
        diag=jnp.asarray(diag, dtype=dtype_local),
        upper=jnp.asarray(upper, dtype=dtype_local),
        area_cm2=jnp.asarray(area, dtype=dtype_local),
    )
    _CABLE_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def prepare_extracellular_runtime(
    axon: SolverAxon,
    dtype_local: jnp.dtype,
    cable: CableRuntime,
) -> ExtracellularRuntime:
    cache_key = _extracellular_runtime_cache_key(axon, dtype_local)
    cached = _EXTRACELLULAR_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _ = cable
    host_arrays = extracellular_runtime_numpy(
        axon,
        dtype=np.dtype(dtype_local),
        target_nx=int(axon.n_compartments),
    )
    record_benchmark_metadata(extracellular_runtime_source="numpy")
    runtime = ExtracellularRuntime(
        Cm_abs=jnp.asarray(host_arrays.Cm_abs, dtype=dtype_local),
        Cx_abs=jnp.asarray(host_arrays.Cx_abs, dtype=dtype_local),
        Gx_abs=jnp.asarray(host_arrays.Gx_abs, dtype=dtype_local),
        Gax_e=jnp.asarray(host_arrays.Gax_e, dtype=dtype_local),
        Gax_i=jnp.asarray(host_arrays.Gax_i, dtype=dtype_local),
        left_i=jnp.asarray(host_arrays.left_i, dtype=dtype_local),
        right_i=jnp.asarray(host_arrays.right_i, dtype=dtype_local),
        left_e=jnp.asarray(host_arrays.left_e, dtype=dtype_local),
        right_e=jnp.asarray(host_arrays.right_e, dtype=dtype_local),
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
) -> SolverRuntime:
    solver_axon = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane = prepare_membrane_runtime(
        axon,
        solver_axon=solver_axon,
    )
    if include_extracellular is None:
        include_extracellular = bool(getattr(axon, "use_extracellular", False))
    if include_area is None:
        include_area = True
    cache_key = _solver_runtime_cache_key(
        axon,
        solver_axon,
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
        extracellular=extracellular,
    )
    _SOLVER_RUNTIME_CACHE[cache_key] = runtime
    return runtime
