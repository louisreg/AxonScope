from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp

from axonscope.runtime.jax.cable_geometry import Array
from axonscope.runtime.solver_axon import SolverAxon


@dataclass(frozen=True)
class SimulationGrid:
    """Time discretization arrays owned by the JAX solver runtime."""

    tsim_ms: float
    dt_ms: float
    Nt: int
    t_vec_ms: Array


@dataclass(frozen=True)
class MembraneRuntime:
    """Compiled membrane backend and initial state arrays."""

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
    parameter_rows: dict[str, Array] | None = None


@dataclass(frozen=True)
class CableRuntime:
    """Single-cable geometry/operator arrays used by JAX kernels."""

    lower: Array
    diag: Array
    upper: Array
    area_cm2: Array


@dataclass(frozen=True)
class StimulationRuntime:
    """Compiled and optionally precomputed stimulation arrays."""

    intracellular_current_density: Callable[[float], Array]
    extracellular_potential_mV: Callable[[float], Array]
    has_driven_extracellular: bool
    intracellular_current_density_mid: Array | None = None
    extracellular_potential_mid_mV: Array | None = None
    extracellular_potential_initial_previous_mV: Array | None = None


@dataclass(frozen=True)
class ExtracellularRuntime:
    """Double-cable extracellular arrays for JAX kernels."""

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
    """Prepared single-axon JAX runtime payload."""

    axon: SolverAxon
    grid: SimulationGrid
    membrane: MembraneRuntime
    cable: CableRuntime
    stimulation: StimulationRuntime
    extracellular: ExtracellularRuntime | None = None
