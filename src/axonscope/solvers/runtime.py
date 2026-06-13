from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

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
    ICMBackend,
    UniformICMBackend,
)
from axonscope.membranes import MembraneModel
from axonscope.solvers.axon_runtime import SolverAxon, build_solver_axon

from .common import (
    Array,
    compartment_area_cm2,
    diffusion_operator_coeffs,
    extracellular_absolute_arrays,
    initial_voltage,
    simulation_step_count,
)
from .options import SolverOptions
from .observables import membrane_observable_names
from axonscope.stimulation.runtime import (
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


def _with_rate_tables(
    model: IonChannelModelBase,
    options: SolverOptions,
) -> IonChannelModelBase:
    if options.rate_table_config is not None:
        enable_rate_tables(model, config=options.rate_table_config)
    return model


def _resolve_solver_options(options: SolverOptions | None) -> SolverOptions:
    return SolverOptions() if options is None else options


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
) -> IonChannelModelBase:
    """Compile the membrane description carried by an axon."""

    options = _resolve_solver_options(solver_options)
    solver_data = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane_models = solver_data.membrane_models
    if len(membrane_models) == 0:
        raise ValueError("Axon membrane_models cannot be empty.")
    first_signature = membrane_models[0]._static_signature()
    if all(model._static_signature() == first_signature for model in membrane_models):
        return compile_membrane_model(
            membrane_models[0],
            solver_options=options,
        )
    compiled = tuple(
        compile_membrane_model(model, solver_options=options)
        for model in membrane_models
    )
    return CompartmentMembraneLayout(compiled).as_membrane_model()


def _backend_from_compiled_membrane(membrane: IonChannelModelBase, nx: int) -> ICMBackend:
    build_backend = getattr(membrane, "build_backend", None)
    if callable(build_backend):
        return build_backend()
    return UniformICMBackend.from_model(membrane, int(nx))


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
    t_vec = (jnp.arange(Nt, dtype=dtype_local) + dtype_local(1.0)) * dt_ms
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
    solver_data = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane = compile_axon_membrane(
        axon,
        solver_axon=solver_data,
        solver_options=solver_options,
    )
    backend = _backend_from_compiled_membrane(membrane, int(solver_data.n_compartments))
    dtype_local = backend.dtype
    Nx = int(solver_data.n_compartments)
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
    axon: SolverAxon,
    dtype_local: jnp.dtype,
    *,
    include_area: bool = True,
) -> CableRuntime:
    lower, diag, upper = diffusion_operator_coeffs(axon, dtype_local)
    area = (
        compartment_area_cm2(axon, dtype_local)
        if include_area
        else jnp.zeros((axon.n_compartments,), dtype=dtype_local)
    )
    return CableRuntime(lower=lower, diag=diag, upper=upper, area_cm2=area)


def prepare_stimulation_runtime(
    axon: Axon,
    solver_axon: SolverAxon,
    dtype_local: jnp.dtype,
    *,
    grid: SimulationGrid | None = None,
    precompute_intracellular: bool = False,
    precompute_extracellular: bool = False,
) -> StimulationRuntime:
    use_extracellular = bool(getattr(axon, "use_extracellular", False))
    inj_fun = build_intracellular_current_density_fn(axon, solver_axon=solver_axon)
    vext_fun = build_extracellular_potential_fn(axon, solver_axon=solver_axon)
    iinj_mid = None
    vext_mid = None
    vext_initial_previous = None
    if grid is not None and (precompute_intracellular or precompute_extracellular):
        t_mid = (
            jnp.arange(grid.Nt, dtype=dtype_local) + dtype_local(0.5)
        ) * dtype_local(grid.dt_ms)
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
    axon: Axon,
    tsim_ms: float,
    dt_ms: float,
    *,
    solver_axon: SolverAxon | None = None,
    include_extracellular: bool | None = None,
    include_area: bool | None = None,
    precompute_intracellular: bool = False,
    precompute_extracellular: bool | None = None,
    solver_options: SolverOptions | None = None,
) -> SolverRuntime:
    solver_axon = build_solver_axon(axon) if solver_axon is None else solver_axon
    membrane = prepare_membrane_runtime(
        axon,
        solver_axon=solver_axon,
        solver_options=solver_options,
    )
    grid = prepare_simulation_grid(tsim_ms, dt_ms, membrane.dtype)
    if include_extracellular is None:
        include_extracellular = bool(getattr(axon, "use_extracellular", False))
    if precompute_extracellular is None:
        precompute_extracellular = include_extracellular
    if include_area is None:
        include_area = True
    cable = prepare_cable_runtime(solver_axon, membrane.dtype, include_area=include_area)
    stimulation = prepare_stimulation_runtime(
        axon,
        solver_axon,
        membrane.dtype,
        grid=grid,
        precompute_intracellular=precompute_intracellular,
        precompute_extracellular=precompute_extracellular,
    )
    extracellular = (
        prepare_extracellular_runtime(solver_axon, membrane.dtype, cable)
        if include_extracellular
        else None
    )
    return SolverRuntime(
        axon=solver_axon,
        grid=grid,
        membrane=membrane,
        cable=cable,
        stimulation=stimulation,
        extracellular=extracellular,
    )


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
