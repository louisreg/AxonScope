from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.benchmarking.hotpaths import benchmark_span, record_benchmark_metadata
from axonscope.axons.axon import Axon
from axonscope.channel_models.axnode import AxnodeICM
from axonscope.channel_models.base_channel_model import CompositeICM, IonChannelModelBase
from axonscope.channel_models.borg_kdr import BorgKDRICM
from axonscope.channel_models.composite_models import (
    Schild94CompositeICM,
    Schild97CompositeICM,
    TigerholmCompositeICM,
)
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM
from axonscope.channel_models.na_hh import NaHHICM
from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.rattay_aberham import RattayAberhamICM
from axonscope.channel_models.rate_tables import enable_rate_tables
from axonscope.icm import (
    CompartmentMembraneLayout,
    HeterogeneousICMBackend,
    ICMBackend,
    UniformICMBackend,
)
from axonscope.membranes import MembraneModel
from axonscope.solvers.axon_runtime import SolverAxon, build_solver_axon
from axonscope.timebase import simulation_step_count

from .common import (
    Array,
    compartment_area_cm2,
    diffusion_operator_coeffs,
    extracellular_absolute_arrays,
    initial_voltage,
)
from axonscope.solvers.options import SolverOptions
from .observables import membrane_observable_names
from axonscope.backends.jax.stimulation_runtime import (
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

    axon: SolverAxon
    grid: SimulationGrid
    membrane: MembraneRuntime
    cable: CableRuntime
    stimulation: StimulationRuntime
    extracellular: ExtracellularRuntime | None = None


_COMPILED_MEMBRANE_CACHE: dict[tuple[Any, ...], IonChannelModelBase] = {}
_BACKEND_CACHE: dict[tuple[Any, ...], ICMBackend] = {}
_MEMBRANE_RUNTIME_CACHE: dict[tuple[Any, ...], MembraneRuntime] = {}
_CABLE_RUNTIME_CACHE: dict[tuple[Any, ...], CableRuntime] = {}
_EXTRACELLULAR_RUNTIME_CACHE: dict[tuple[Any, ...], ExtracellularRuntime] = {}
_SOLVER_RUNTIME_CACHE: dict[tuple[Any, ...], SolverRuntime] = {}


def _with_rate_tables(
    model: IonChannelModelBase,
    options: SolverOptions,
) -> IonChannelModelBase:
    if options.rate_table_config is not None:
        enable_rate_tables(model, config=options.rate_table_config)
    return model


def _resolve_solver_options(options: SolverOptions | None) -> SolverOptions:
    return SolverOptions() if options is None else options


def _solver_options_cache_key(options: SolverOptions) -> tuple[Any, ...]:
    return ("solver_options", repr(options))


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
        bool(getattr(axon, "extracellular_contexts", ())),
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


def compile_membrane_model(
    model: Any,
    *,
    solver_options: SolverOptions | None = None,
) -> IonChannelModelBase:
    """Compile a public membrane description to the current compute model.

    Public axons carry `MembraneModel` values so they stay independent from the
    JAX/channel-model implementation. This function is the narrow bridge from
    that descriptive layer to the solver runtime.
    """

    options = _resolve_solver_options(solver_options)
    if isinstance(model, IonChannelModelBase):
        return _with_rate_tables(model, options)
    if not isinstance(model, MembraneModel):
        implementation = getattr(model, "_implementation", None)
        if implementation is not None:
            return compile_membrane_model(
                implementation,
                solver_options=options,
            )
        raise TypeError(f"Unsupported membrane model description: {model!r}")

    if model.kind == "legacy":
        if model._implementation is None:
            raise ValueError("Legacy membrane descriptions must carry an implementation.")
        return compile_membrane_model(
            model._implementation,
            solver_options=options,
        )

    params = dict(model.params)
    if model.kind == "passive":
        return _with_rate_tables(PassiveICM(**params), options)
    if model.kind == "hodgkin_huxley":
        return _with_rate_tables(HodgkinHuxleyICM(**params), options)
    if model.kind == "rattay_aberham":
        return _with_rate_tables(RattayAberhamICM(**params), options)
    if model.kind == "sundt":
        return _with_rate_tables(
            CompositeICM(
                [
                    NaHHICM(
                        gnabar=params["gnabar"],
                        ena=params["ena"],
                        celsius=params["celsius"],
                        mshift=-6.0,
                        hshift=6.0,
                    ),
                    BorgKDRICM(
                        gkdrbar=params["gkdrbar"],
                        ek=params["ek"],
                        celsius=params["celsius"],
                    ),
                    PassiveICM(Rm=params["Rm"], EL=params["El"]),
                ]
            ),
            options,
        )
    if model.kind == "tigerholm":
        return _with_rate_tables(TigerholmCompositeICM(**params), options)
    if model.kind == "schild94":
        return _with_rate_tables(Schild94CompositeICM(**params), options)
    if model.kind == "schild97":
        return _with_rate_tables(Schild97CompositeICM(**params), options)
    if model.kind == "axnode":
        return _with_rate_tables(AxnodeICM(**params), options)
    if model.kind == "composite":
        return _with_rate_tables(
            CompositeICM(
                [
                    compile_membrane_model(component, solver_options=options)
                    for component in model.components
                ]
            ),
            options,
        )

    raise ValueError(f"Unknown membrane model kind: {model.kind!r}")


def compile_axon_membrane(
    axon: Axon,
    *,
    solver_axon: SolverAxon | None = None,
    solver_options: SolverOptions | None = None,
    membrane_signatures: tuple[Any, ...] | None = None,
) -> IonChannelModelBase:
    """Compile the membrane description carried by an axon."""

    options = _resolve_solver_options(solver_options)
    solver_data = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane_models = solver_data.membrane_models
    if len(membrane_models) == 0:
        raise ValueError("Axon membrane_models cannot be empty.")
    if membrane_signatures is None:
        membrane_signatures = tuple(model._static_signature() for model in membrane_models)
    cache_key = (
        "axon_membrane",
        membrane_signatures,
        _solver_options_cache_key(options),
    )
    cached = _COMPILED_MEMBRANE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    first_signature = membrane_signatures[0]
    if all(signature == first_signature for signature in membrane_signatures):
        compiled = compile_membrane_model(
            membrane_models[0],
            solver_options=options,
        )
    else:
        compiled_by_signature: dict[tuple[Any, ...], IonChannelModelBase] = {}
        for model, signature in zip(membrane_models, membrane_signatures, strict=True):
            compiled_component = compiled_by_signature.get(signature)
            if compiled_component is None:
                compiled_component = compile_membrane_model(
                    model,
                    solver_options=options,
                )
                compiled_by_signature[signature] = compiled_component
        compiled_components = tuple(
            compiled_by_signature[signature]
            for signature in membrane_signatures
        )
        compiled = CompartmentMembraneLayout(compiled_components).as_membrane_model()
    _COMPILED_MEMBRANE_CACHE[cache_key] = compiled
    return compiled


def _backend_from_compiled_membrane(membrane: IonChannelModelBase, nx: int) -> ICMBackend:
    cache_key = ("backend", membrane._static_signature(), int(nx))
    cached = _BACKEND_CACHE.get(cache_key)
    if cached is not None:
        return cached
    build_backend = getattr(membrane, "build_backend", None)
    if callable(build_backend):
        backend = build_backend()
    else:
        backend = UniformICMBackend.from_model(membrane, int(nx))
    _BACKEND_CACHE[cache_key] = backend
    return backend


def build_icm_backend_from_axon(
    axon: Axon,
    *,
    solver_options: SolverOptions | None = None,
) -> ICMBackend:
    """Build the solver-side membrane backend for an axon description."""

    solver_axon = build_solver_axon(axon)
    membrane = compile_axon_membrane(
        axon,
        solver_axon=solver_axon,
        solver_options=solver_options,
    )
    return _backend_from_compiled_membrane(membrane, int(solver_axon.n_compartments))


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
        backend = _backend_from_compiled_membrane(membrane, int(solver_data.n_compartments))
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
    membrane: IonChannelModelBase,
    backend: ICMBackend,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array, tuple[Array, ...], Array]:
    """Prepare initial membrane arrays without unnecessary eager JAX scatter."""

    rattay_initial = _try_prepare_uniform_rattay_initial_arrays(
        axon,
        membrane,
        backend,
        nx=nx,
        dtype_local=dtype_local,
    )
    if rattay_initial is not None:
        record_benchmark_metadata(membrane_init_source="rattay_numpy")
        return rattay_initial
    if isinstance(backend, HeterogeneousICMBackend) and not membrane.membrane_state_specs():
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


def _try_prepare_uniform_rattay_initial_arrays(
    axon: Axon,
    membrane: IonChannelModelBase,
    backend: ICMBackend,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> tuple[Array, Array, tuple[Array, ...], Array] | None:
    """Prepare the common Rattay/Rattay+passive initial state on the host."""

    if not isinstance(backend, UniformICMBackend):
        return None
    if int(backend.Nx) != int(nx) or membrane.membrane_state_specs():
        return None
    rattay = _uniform_rattay_component(membrane)
    if rattay is None or _membrane_uses_rate_table(membrane):
        return None

    np_dtype = np.dtype(dtype_local)
    vm0_value = float(getattr(axon, "v_init", 0.0))
    vm0_np = np.full((int(nx),), vm0_value, dtype=np_dtype)
    gates_row = _rattay_initial_gates_numpy(
        rattay,
        vm0_mV=vm0_value,
        dtype=np_dtype,
    )
    gates_np = np.broadcast_to(gates_row, (int(nx), gates_row.shape[0])).copy()
    background_np = np.zeros((int(nx),), dtype=np_dtype)
    return (
        jnp.asarray(vm0_np, dtype=dtype_local),
        jnp.asarray(gates_np, dtype=dtype_local),
        (),
        jnp.asarray(background_np, dtype=dtype_local),
    )


def _uniform_rattay_component(membrane: IonChannelModelBase) -> RattayAberhamICM | None:
    if isinstance(membrane, RattayAberhamICM):
        return membrane
    if not isinstance(membrane, CompositeICM):
        return None
    rattay_components: list[RattayAberhamICM] = []
    for component in membrane.models:
        if isinstance(component, RattayAberhamICM):
            rattay_components.append(component)
        elif not isinstance(component, PassiveICM):
            return None
        elif component.gate_names():
            return None
    if len(rattay_components) != 1:
        return None
    rattay = rattay_components[0]
    if tuple(membrane.gate_names()) != tuple(rattay.gate_names()):
        return None
    return rattay


def _membrane_uses_rate_table(membrane: IonChannelModelBase) -> bool:
    if getattr(membrane, "_rate_table", None) is not None:
        return True
    if isinstance(membrane, CompositeICM):
        return any(_membrane_uses_rate_table(component) for component in membrane.models)
    return False


def _rattay_initial_gates_numpy(
    model: RattayAberhamICM,
    *,
    vm0_mV: float,
    dtype: np.dtype,
) -> np.ndarray:
    """NumPy equivalent of RattayAberhamICM.init_gates for scalar Vm0."""

    v = dtype.type(vm0_mV)

    def vtrap(x_value: np.generic, y_value: float) -> np.generic:
        y = dtype.type(y_value)
        z = x_value / y
        if abs(float(z)) < 1e-6:
            return y * (dtype.type(1.0) - z / dtype.type(2.0))
        return x_value / (dtype.type(np.exp(z)) - dtype.type(1.0))

    alpha_m = vtrap(
        dtype.type(2.5) - dtype.type(0.1) * (v + dtype.type(70.0)),
        1.0,
    )
    alpha_h = dtype.type(0.07) * dtype.type(
        np.exp(-(v + dtype.type(70.0)) / dtype.type(20.0))
    )
    alpha_n = dtype.type(0.1) * vtrap(
        dtype.type(1.0) - dtype.type(0.1) * (v + dtype.type(70.0)),
        1.0,
    )
    beta_m = dtype.type(4.0) * dtype.type(
        np.exp(-(v + dtype.type(70.0)) / dtype.type(18.0))
    )
    beta_h = dtype.type(1.0) / (
        dtype.type(np.exp(dtype.type(3.0) - dtype.type(0.1) * (v + dtype.type(70.0))))
        + dtype.type(1.0)
    )
    beta_n = dtype.type(0.125) * dtype.type(
        np.exp(-(v + dtype.type(70.0)) / dtype.type(80.0))
    )
    alpha = np.asarray((alpha_m, alpha_h, alpha_n), dtype=dtype)
    beta = np.asarray((beta_m, beta_h, beta_n), dtype=dtype)
    return alpha / np.maximum(alpha + beta, dtype.type(1e-12))


def _prepare_heterogeneous_membrane_initial_arrays(
    axon: Axon,
    backend: HeterogeneousICMBackend,
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
            use_extracellular and getattr(axon, "extracellular_contexts", ())
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


def precompute_intracellular_current_density(
    axon: Axon,
    t_ms: Array,
    *,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Sample intracellular current density on a time grid, returning shape (Nt, Nx)."""
    if dtype_local is None:
        solver_axon = build_solver_axon(axon)
        runtime = prepare_membrane_runtime(axon, solver_axon=solver_axon)
        dtype_local = runtime.dtype
    else:
        solver_axon = build_solver_axon(axon)
    t = jnp.asarray(t_ms, dtype=dtype_local)
    inj_fun = build_intracellular_current_density_fn(axon, solver_axon=solver_axon)
    return sample_intracellular_current_density(inj_fun, t, dtype_local=dtype_local)


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
