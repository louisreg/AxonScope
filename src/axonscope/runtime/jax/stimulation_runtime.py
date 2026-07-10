"""Runtime compilation of stimulation descriptions for JAX solvers.

Public stimulation objects are descriptive and unit-aware. This module is the
solver boundary: it converts stimuli, current clamps, and sampled extracellular
stimulation into small JAX callables and precomputed arrays with explicit
numeric units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import numpy as np

from axonscope.stimulation import (
    ExtracellularStimulation,
    IntracellularContext,
    IntracellularCurrentClamp,
)
from axonscope.stimulation import ArrayLike, Stimulus

if TYPE_CHECKING:
    from axonscope.solvers.axon_runtime import SolverAxon


def _resolve_solver_axon(axon, solver_axon: "SolverAxon | None") -> "SolverAxon":
    """Return an existing solver axon or build one from a public axon object."""

    if solver_axon is not None:
        return solver_axon
    from axonscope.solvers.axon_runtime import build_solver_axon

    return build_solver_axon(axon)


def _axon_dtype(axon, *, solver_axon: "SolverAxon | None" = None) -> jnp.dtype:
    """Return the JAX scalar dtype associated with an axon-like object."""

    if solver_axon is not None:
        return _jax_scalar_dtype(solver_axon.dtype)
    if hasattr(axon, "dtype"):
        return _jax_scalar_dtype(axon.dtype)
    layout = getattr(axon, "layout", None)
    if layout is not None:
        return _jax_scalar_dtype(layout.sections[0].membrane.dtype)
    return jnp.float32


def _jax_scalar_dtype(dtype_like) -> jnp.dtype:
    """Normalize NumPy-like dtype inputs to the supported JAX float dtype."""

    name = np.dtype(dtype_like).name
    if name == "float64":
        return jnp.float64
    return jnp.float32


def _intracellular_contexts_from_axon(axon) -> tuple[IntracellularContext, ...]:
    """Return intracellular contexts from a simulation-like object."""

    return tuple(getattr(axon, "intracellular_contexts", ()))


def _extracellular_stimulations_from_axon(axon) -> tuple[ExtracellularStimulation, ...]:
    """Return extracellular stimulations from a simulation-like object."""

    return tuple(getattr(axon, "extracellular_stimulations", ()))


@dataclass(frozen=True)
class JaxStimulus:
    """JAX-ready stimulus representation used by runtime code.

    `t` is stored in milliseconds and `y` is stored in the numeric unit already
    required by the consuming solver term.
    """

    t: jnp.ndarray
    y: jnp.ndarray
    mode: Literal["hold", "linear"] = "hold"

    def __call__(self, tq):
        """Evaluate the stimulus at one scalar time in milliseconds."""

        if self.mode == "linear":
            return jnp.interp(tq, self.t, self.y, left=self.y[0], right=self.y[-1])

        idx = jnp.searchsorted(self.t, tq, side="right") - 1
        idx = jnp.clip(idx, 0, self.y.shape[0] - 1)
        return self.y[idx]


def compile_stimulus(stimulus: Stimulus, dtype_local: jnp.dtype | None = None) -> JaxStimulus:
    """Compile a descriptive stimulus to a JAX-ready callable.

    The stimulus is assumed to already be expressed in the physical unit needed
    by its consumer, such as nanoamperes for clamps or amperes for electrodes.
    """
    if dtype_local is None:
        dtype_local = jnp.float32
    return JaxStimulus(
        t=jnp.asarray(stimulus.t, dtype=dtype_local),
        y=jnp.asarray(stimulus.y, dtype=dtype_local),
        mode=stimulus.mode,
    )


@dataclass(frozen=True)
class CompiledExtracellularDrive:
    """JAX-ready extracellular drive with a precomputed spatial footprint."""

    footprint_V_per_A: jnp.ndarray
    stimulus: JaxStimulus

    def __call__(self, t_ms):
        """Return this drive's Vext contribution in volts at `t_ms`."""

        return self.stimulus(t_ms) * self.footprint_V_per_A


@dataclass(frozen=True)
class CompiledExtracellularStimulation:
    """JAX-ready sampled extracellular stimulation."""

    drives: tuple[CompiledExtracellularDrive, ...]

    def __call__(self, t_ms):
        """Return summed extracellular potential in volts at `t_ms`."""

        if not self.drives:
            raise ValueError("CompiledExtracellularStimulation requires at least one drive.")
        total = jnp.zeros_like(self.drives[0].footprint_V_per_A)
        for drive in self.drives:
            total = total + drive(t_ms)
        return total


@dataclass(frozen=True)
class CompiledExtracellularStimulations:
    """JAX-ready collection of extracellular stimulations for one axon."""

    n_compartments: int
    dtype_local: Any
    stimulations: tuple[CompiledExtracellularStimulation, ...]

    def __call__(self, t_ms):
        """Return summed extracellular potential in millivolts at `t_ms`."""

        vext = jnp.zeros((self.n_compartments,), dtype=self.dtype_local)
        for stimulation in self.stimulations:
            vext = (
                vext
                + stimulation(t_ms).astype(self.dtype_local) * self.dtype_local(1e3)
            )
        return vext


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


def compile_extracellular_stimulation(
    stimulation: ExtracellularStimulation,
    x_positions_m: ArrayLike,
    dtype_local: jnp.dtype | None = None,
) -> CompiledExtracellularStimulation:
    """Precompute all drive footprints for one axon.

    Parameters
    ----------
    stimulation:
        Sampled extracellular stimulation containing one or more drives.
    x_positions_m:
        Intrinsic axial sample positions in meters.
    dtype_local:
        JAX dtype used for compiled arrays.
    """
    if dtype_local is None:
        dtype_local = jnp.float32
    compiled_drives = []
    for drive in stimulation.drives:
        stimulus = getattr(drive, "stimulus", None)
        if stimulus is None:
            raise ValueError("Each extracellular drive must have an attached stimulus.")
        fp = _drive_footprint_for_positions(drive, x_positions_m)
        compiled_drives.append(
            CompiledExtracellularDrive(
                footprint_V_per_A=jnp.asarray(fp, dtype=dtype_local),
                stimulus=compile_stimulus(stimulus, dtype_local=dtype_local),
            )
        )
    return CompiledExtracellularStimulation(
        drives=tuple(compiled_drives),
    )


def compartment_surface_area_cm2(
    axon,
    dtype_local: jnp.dtype,
    *,
    solver_axon: "SolverAxon | None" = None,
) -> jnp.ndarray:
    """Return per-compartment membrane surface area in cm^2."""
    solver_data = _resolve_solver_axon(axon, solver_axon)
    diam_um = jnp.asarray(solver_data.diam_um, dtype=dtype_local)
    length_cm = jnp.asarray(solver_data.compartment_lengths_um, dtype=dtype_local) * dtype_local(1e-4)
    return jnp.pi * (diam_um * dtype_local(1e-4)) * length_cm


def compile_intracellular_contexts(
    axon,
    dtype_local: jnp.dtype | None = None,
    *,
    solver_axon: "SolverAxon | None" = None,
) -> CompiledIntracellularContexts:
    """Compile intracellular contexts to a current-density callable."""

    solver_data = _resolve_solver_axon(axon, solver_axon)
    if dtype_local is None:
        dtype_local = _axon_dtype(axon, solver_axon=solver_data)
    contexts = _intracellular_contexts_from_axon(axon)
    Nx = solver_data.n_compartments

    if not contexts:
        return CompiledIntracellularContexts(
            n_compartments=Nx,
            dtype_local=dtype_local,
            basis=jnp.zeros((0, Nx), dtype=dtype_local),
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
        n_compartments=Nx,
        dtype_local=dtype_local,
        basis=jnp.eye(Nx, dtype=dtype_local)[jnp.asarray(idxs, dtype=jnp.int32)],
        nA_to_mA_per_cm2=jnp.asarray(nA_to_mA_per_cm2, dtype=dtype_local),
        stimuli=tuple(compiled_stimuli),
    )


def _drive_footprint_for_positions(drive: Any, x_positions_m: ArrayLike) -> np.ndarray:
    footprint = drive.footprint
    values = np.asarray(footprint.values_for_axon(), dtype=float)
    x_um = np.asarray(x_positions_m, dtype=float) * 1e6
    support_um = np.asarray(footprint.positions_um, dtype=float)
    if x_um.shape == support_um.shape and np.allclose(x_um, support_um):
        return values
    if footprint.interpolation not in {"sampled", "linear"}:
        raise NotImplementedError(
            "Only sampled/linear footprint interpolation is supported by "
            "scalar extracellular lowering."
        )
    if np.any(np.diff(support_um) < 0.0):
        raise ValueError("Footprint positions must be sorted for interpolation.")
    return np.asarray(np.interp(x_um, support_um, values), dtype=float)


def compile_extracellular_stimulations(
    axon,
    dtype_local: jnp.dtype | None = None,
    *,
    solver_axon: "SolverAxon | None" = None,
) -> CompiledExtracellularStimulations:
    """Compile all extracellular stimulations attached to one axon."""

    solver_data = _resolve_solver_axon(axon, solver_axon)
    if dtype_local is None:
        dtype_local = _axon_dtype(axon, solver_axon=solver_data)
    stimulations = _extracellular_stimulations_from_axon(axon)
    Nx = solver_data.n_compartments

    if not stimulations:
        return CompiledExtracellularStimulations(
            n_compartments=Nx,
            dtype_local=dtype_local,
            stimulations=(),
        )

    x_positions_m = jnp.asarray(solver_data.x_um, dtype=dtype_local) * dtype_local(1e-6)
    compiled_stimulations = tuple(
        compile_extracellular_stimulation(
            stimulation,
            x_positions_m,
            dtype_local=dtype_local,
        )
        for stimulation in stimulations
    )
    return CompiledExtracellularStimulations(
        n_compartments=Nx,
        dtype_local=dtype_local,
        stimulations=compiled_stimulations,
    )


def build_intracellular_current_density_fn(
    axon,
    *,
    solver_axon: "SolverAxon | None" = None,
):
    """Compile intracellular clamps into a current-density function.

    The returned callable maps time in milliseconds to an array of injected
    current density in mA/cm^2, one value per compartment.
    """
    return compile_intracellular_contexts(axon, solver_axon=solver_axon)


def build_extracellular_potential_fn(
    axon,
    *,
    solver_axon: "SolverAxon | None" = None,
):
    """Compile extracellular stimulation into an imposed-potential function.

    The returned callable maps time in milliseconds to Vext in millivolts, one
    value per compartment. If no extracellular stimulation is attached, the
    callable returns zeros.
    """
    return compile_extracellular_stimulations(axon, solver_axon=solver_axon)


__all__ = [
    "CompiledExtracellularDrive",
    "CompiledExtracellularStimulations",
    "CompiledExtracellularStimulation",
    "CompiledIntracellularContexts",
    "JaxStimulus",
    "build_extracellular_potential_fn",
    "build_intracellular_current_density_fn",
    "compile_extracellular_stimulation",
    "compile_extracellular_stimulations",
    "compile_intracellular_contexts",
    "compile_stimulus",
    "compartment_surface_area_cm2",
]
