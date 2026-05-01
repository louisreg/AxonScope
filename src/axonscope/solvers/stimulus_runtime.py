from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from axonscope.stimulation import ExtracellularContext, IntracellularCurrentClamp
from axonscope.stimulus import ArrayLike, Stimulus


def _axon_dtype(axon) -> jnp.dtype:
    ion_channel = getattr(axon, "ion_channel", None)
    if ion_channel is not None and hasattr(ion_channel, "dtype"):
        return ion_channel.dtype
    if hasattr(axon, "dtype"):
        return axon.dtype
    return jnp.float32


@dataclass(frozen=True)
class JaxStimulus:
    """JAX-ready stimulus representation used inside solver kernels."""

    t: jnp.ndarray
    y: jnp.ndarray
    mode: Literal["hold", "linear"] = "hold"

    def __call__(self, tq):
        if self.mode == "linear":
            return jnp.interp(tq, self.t, self.y, left=self.y[0], right=self.y[-1])

        idx = jnp.searchsorted(self.t, tq, side="right") - 1
        idx = jnp.clip(idx, 0, self.y.shape[0] - 1)
        return self.y[idx]


def compile_stimulus(stimulus: Stimulus, dtype_local: jnp.dtype | None = None) -> JaxStimulus:
    """Compile a descriptive stimulus to a JAX-ready callable."""
    if dtype_local is None:
        dtype_local = jnp.float32
    return JaxStimulus(
        t=jnp.asarray(stimulus.t, dtype=dtype_local),
        y=jnp.asarray(stimulus.y, dtype=dtype_local),
        mode=stimulus.mode,
    )


@dataclass(frozen=True)
class CompiledExtracellularContext:
    """JAX-ready extracellular context used by extracellular solvers."""

    footprint_V_per_A: jnp.ndarray
    stimulus: JaxStimulus

    def __call__(self, t_ms):
        return self.stimulus(t_ms) * self.footprint_V_per_A


def compile_extracellular_context(
    ctx: ExtracellularContext,
    x_positions_m: ArrayLike,
    dtype_local: jnp.dtype | None = None,
) -> CompiledExtracellularContext:
    """Precompute the spatial footprint for one extracellular context."""
    if dtype_local is None:
        dtype_local = jnp.float32
    fp = ctx.electrode.footprint(x_positions_m)
    return CompiledExtracellularContext(
        footprint_V_per_A=jnp.asarray(fp, dtype=dtype_local),
        stimulus=compile_stimulus(ctx.stimulus, dtype_local=dtype_local),
    )


def compartment_surface_area_cm2(axon, dtype_local: jnp.dtype) -> jnp.ndarray:
    """Return per-compartment membrane surface area in cm^2."""
    if hasattr(axon, "diam_vec"):
        diam_um = jnp.asarray(axon.diam_vec, dtype=dtype_local)
    else:
        diam_um = jnp.full((axon.Nx,), dtype_local(axon.d), dtype=dtype_local)
    if hasattr(axon, "compartment_lengths_um"):
        length_cm = jnp.asarray(axon.compartment_lengths_um, dtype=dtype_local) * dtype_local(1e-4)
    else:
        length_cm = jnp.asarray(axon.dx_cm, dtype=dtype_local)
    return jnp.pi * (diam_um * dtype_local(1e-4)) * length_cm


def build_intracellular_current_density_fn(axon):
    """Compile descriptive intracellular clamps into a JAX-ready density function."""
    dtype_local = _axon_dtype(axon)
    Nx = axon.Nx
    clamps: tuple[IntracellularCurrentClamp, ...] = tuple(
        getattr(axon, "intracellular_clamps", ())
    )

    if not clamps:
        return lambda t_ms: jnp.zeros((Nx,), dtype=dtype_local)

    x = jnp.asarray(axon.x, dtype=dtype_local)
    area_cm2 = compartment_surface_area_cm2(axon, dtype_local)

    idxs = []
    amp_to_density = []
    compiled_stimuli = []
    for clamp in clamps:
        idx = int(jnp.argmin(jnp.abs(x - dtype_local(clamp.position_um))))
        idxs.append(idx)
        amp_to_density.append(dtype_local(1e-3) / area_cm2[idx])
        compiled_stimuli.append(compile_stimulus(clamp.stimulus, dtype_local=dtype_local))

    basis = jnp.eye(Nx, dtype=dtype_local)[jnp.asarray(idxs, dtype=jnp.int32)]
    amp_to_density = jnp.asarray(amp_to_density, dtype=dtype_local)
    compiled_stimuli = tuple(compiled_stimuli)

    def inj_fun(t_ms):
        amps_nA = jnp.asarray([stim(t_ms) for stim in compiled_stimuli], dtype=dtype_local)
        densities = amps_nA * amp_to_density
        return jnp.sum(densities[:, None] * basis, axis=0)

    return inj_fun


def build_extracellular_potential_fn(axon):
    """Compile descriptive extracellular contexts into a JAX-ready Vext function."""
    dtype_local = _axon_dtype(axon)
    Nx = axon.Nx
    contexts: tuple[ExtracellularContext, ...] = tuple(
        getattr(axon, "extracellular_contexts", ())
    )

    if not contexts:
        return lambda t_ms: jnp.zeros((Nx,), dtype=dtype_local)

    x_positions_m = jnp.asarray(axon.x, dtype=dtype_local) * dtype_local(1e-6)
    compiled_contexts = tuple(
        compile_extracellular_context(ctx, x_positions_m, dtype_local=dtype_local)
        for ctx in contexts
    )

    def vext_fun(t_ms):
        vext = jnp.zeros((Nx,), dtype=dtype_local)
        for ctx in compiled_contexts:
            vext = vext + ctx(t_ms).astype(dtype_local) * dtype_local(1e3)
        return vext

    return vext_fun
