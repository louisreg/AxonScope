from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .common import Array, apply_diffusion_operator
from .kernels import _run_single_cable_vstim_vm_scan
from .runtime import SolverRuntime


@dataclass(frozen=True)
class BatchKernelResult:
    """Raw batched solver-kernel output before packaging public simulations."""

    Vm: Array
    t: Array


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
    ),
)
def _run_single_cable_vstim_batch_vm_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    lower: Array,
    diag: Array,
    upper: Array,
    dl: Array,
    d_static: Array,
    du: Array,
    Cm_uF_cm2: Array,
    I_background: Array,
    Vm0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    dt_ms: Array,
) -> Array:
    """Run the imposed-Vstim single-cable scan over a leading batch axis."""

    def one_batch(Iinj_mid: Array, vext_mid: Array) -> Array:
        vstim_forcing_mid = jax.vmap(
            lambda values: apply_diffusion_operator(values, lower, diag, upper)
        )(vext_mid)
        return _run_single_cable_vstim_vm_scan(
            backend=backend,
            membrane=membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            lower=lower,
            diag=diag,
            upper=upper,
            dl=dl,
            d_static=d_static,
            du=du,
            Cm_uF_cm2=Cm_uF_cm2,
            I_background=I_background,
            Vm0_mV=Vm0_mV,
            gates0=gates0,
            state0=state0,
            intracellular_current_density_mid=Iinj_mid,
            extracellular_diffusion_forcing_mid=vstim_forcing_mid,
            dt_ms=dt_ms,
        )

    return jax.vmap(one_batch)(
        intracellular_current_density_mid,
        extracellular_potential_mid_mV,
    )


@dataclass(frozen=True)
class SingleCableVStimBatchKernel:
    """Batch-oriented imposed-field kernel for homogeneous single-cable axons.

    The batch axis represents independent extracellular fields sharing the same
    axon geometry, membrane model, initial state, and time grid. This is the
    first GPU-friendly shape: ``Vstim[B, Nt, Nx] -> Vm[B, Nt, Nx]``.
    """

    runtime: SolverRuntime
    Cm_uF_cm2: Array
    has_driven_extracellular: bool | None = None

    def run(
        self,
        *,
        extracellular_potential_mid_mV: Array | None = None,
        intracellular_current_density_mid: Array | None = None,
    ) -> BatchKernelResult:
        runtime = self.runtime
        if runtime.extracellular is not None:
            raise ValueError(
                "SingleCableVStimBatchKernel expects a scalar single-cable runtime; "
                "prepare it with include_extracellular=False."
            )

        membrane_runtime = runtime.membrane
        grid = runtime.grid
        cable = runtime.cable
        dtype_local = membrane_runtime.dtype

        vext_mid = (
            runtime.stimulation.extracellular_potential_mid_mV
            if extracellular_potential_mid_mV is None
            else extracellular_potential_mid_mV
        )
        if vext_mid is None:
            raise ValueError("extracellular_potential_mid_mV is required for Vstim batching.")

        vext_batch = _as_batched_time_space_array(
            "extracellular_potential_mid_mV",
            vext_mid,
            nt=grid.Nt,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
        )
        batch_size = int(vext_batch.shape[0])

        iinj_mid = (
            runtime.stimulation.intracellular_current_density_mid
            if intracellular_current_density_mid is None
            else intracellular_current_density_mid
        )
        if iinj_mid is None:
            raise ValueError("intracellular_current_density_mid is required for Vstim batching.")
        iinj_batch = _as_batched_time_space_array(
            "intracellular_current_density_mid",
            iinj_mid,
            nt=grid.Nt,
            nx=membrane_runtime.Nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )

        dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
        lower, diag, upper = cable.lower, cable.diag, cable.upper
        out = _run_single_cable_vstim_batch_vm_scan(
            backend=membrane_runtime.backend,
            membrane=membrane_runtime.membrane,
            has_driven_extracellular=(
                runtime.stimulation.has_driven_extracellular
                if self.has_driven_extracellular is None
                else bool(self.has_driven_extracellular)
            ),
            stateless_vm_only=bool(
                membrane_runtime.membrane.supports_stateless_vm_only_fast_path()
            ),
            lower=lower,
            diag=diag,
            upper=upper,
            dl=-dt * lower,
            d_static=jnp.ones_like(diag) - dt * diag,
            du=-dt * upper,
            Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
            I_background=membrane_runtime.background_current,
            Vm0_mV=membrane_runtime.Vm0_mV,
            gates0=membrane_runtime.gates0,
            state0=membrane_runtime.state0,
            intracellular_current_density_mid=iinj_batch,
            extracellular_potential_mid_mV=vext_batch,
            dt_ms=dt,
        )
        return BatchKernelResult(Vm=out, t=grid.t_vec_ms)


def _as_batched_time_space_array(
    name: str,
    values: Array,
    *,
    nt: int,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int | None = None,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 2:
        if arr.shape != (nt, nx):
            raise ValueError(
                f"{name} must have shape (Nt, Nx)=({nt}, {nx}) "
                f"or (B, Nt, Nx), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[1:] != (nt, nx):
            raise ValueError(
                f"{name} must have trailing shape (Nt, Nx)=({nt}, {nx}), "
                f"got {arr.shape}."
            )
    else:
        raise ValueError(
            f"{name} must have shape (Nt, Nx) or (B, Nt, Nx), got {arr.shape}."
        )

    if batch_size is None:
        return arr
    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, nt, nx))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")


__all__ = [
    "BatchKernelResult",
    "SingleCableVStimBatchKernel",
]
