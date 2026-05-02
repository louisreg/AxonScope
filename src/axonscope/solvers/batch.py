from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from axonscope.stimulation import ExtracellularContext

from .common import Array, apply_diffusion_operator
from .kernels import _run_double_cable_vm_scan, _run_single_cable_vstim_vm_scan
from .runtime import SolverRuntime
from .stimulus_runtime import compile_extracellular_context


ContextBatchRow = ExtracellularContext | Sequence[ExtracellularContext] | None


@dataclass(frozen=True)
class BatchKernelResult:
    """Raw batched solver-kernel output before packaging public simulations."""

    Vm: Array
    t: Array


def build_vstim_midpoint_batch(
    axon,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    tsim_ms: float,
    dt_ms: float,
    x_positions_m: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build imposed extracellular samples on solver midpoints.

    Returns ``Vstim[B, Nt, Nx]`` in mV. Each batch row may contain one
    ``ExtracellularContext``, multiple contexts that are summed, or ``None`` for
    a zero imposed field.
    """

    dtype = _resolve_dtype(axon, dtype_local)
    nt = int(jnp.ceil(tsim_ms / dt_ms))
    t_mid_ms = (
        jnp.arange(nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(dt_ms, dtype=dtype)
    return build_vstim_batch(
        axon,
        contexts_batch,
        t_ms=t_mid_ms,
        x_positions_m=x_positions_m,
        dtype_local=dtype,
    )


def build_vstim_initial_previous_batch(
    axon,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    dt_ms: float,
    x_positions_m: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build the initial previous imposed field used by double-cable batches.

    Returns ``Vstim[B, Nx]`` sampled at ``t = -dt/2`` in mV. This pairs with
    ``build_vstim_midpoint_batch`` for full double-cable kernels.
    """

    dtype = _resolve_dtype(axon, dtype_local)
    samples = build_vstim_batch(
        axon,
        contexts_batch,
        t_ms=jnp.asarray([-0.5 * dt_ms], dtype=dtype),
        x_positions_m=x_positions_m,
        dtype_local=dtype,
    )
    return samples[:, 0, :]


def build_vstim_batch(
    axon,
    contexts_batch: Sequence[ContextBatchRow],
    *,
    t_ms: Array,
    x_positions_m: Array | None = None,
    dtype_local: jnp.dtype | None = None,
) -> Array:
    """Build imposed extracellular samples for a batch of context rows.

    Parameters
    ----------
    axon
        Axon providing the default compartment positions and dtype.
    contexts_batch
        Sequence of batch rows. A row can be one ``ExtracellularContext``, a
        sequence of contexts to sum, or ``None``/empty for a zero field.
    t_ms
        Time samples in ms, usually solver midpoints, shape ``(Nt,)``.
    x_positions_m
        Optional spatial samples in meters. Shape ``(Nx,)`` shares positions
        across the batch; shape ``(B, Nx)`` uses per-row positions.
    dtype_local
        Optional JAX dtype override.

    Returns
    -------
    Array
        ``Vstim`` in mV with shape ``(B, Nt, Nx)``.
    """

    rows = tuple(_normalize_context_row(row) for row in contexts_batch)
    if not rows:
        raise ValueError("contexts_batch must contain at least one row.")

    dtype = _resolve_dtype(axon, dtype_local)
    t = jnp.asarray(t_ms, dtype=dtype)
    if t.ndim != 1:
        raise ValueError(f"t_ms must have shape (Nt,), got {t.shape}.")

    x_rows = _resolve_x_positions_m(
        axon,
        x_positions_m,
        batch_size=len(rows),
        dtype_local=dtype,
    )
    vstim_rows = [
        _build_vstim_row(row, t, x_positions_row_m=x_rows[i], dtype_local=dtype)
        for i, row in enumerate(rows)
    ]
    return jnp.stack(vstim_rows, axis=0)


def scale_extracellular_contexts(
    contexts: Sequence[ExtracellularContext],
    scale: float,
) -> tuple[ExtracellularContext, ...]:
    """Return contexts with their stimulus amplitudes scaled by ``scale``."""

    return tuple(
        ExtracellularContext(electrode=ctx.electrode, stimulus=ctx.stimulus.scaled(scale))
        for ctx in contexts
    )


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


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
    ),
)
def _run_double_cable_batch_vm_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    Vi0_mV: Array,
    Ve0_mV: Array,
    gates0: Array,
    state0: tuple[Array, ...],
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
    dt_ms: Array,
) -> Array:
    """Run the full double-cable scan over a leading batch axis."""

    def one_batch(Iinj_mid: Array, vext_mid: Array, vext_previous: Array) -> Array:
        return _run_double_cable_vm_scan(
            backend=backend,
            membrane=membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            Vi0_mV=Vi0_mV,
            Ve0_mV=Ve0_mV,
            gates0=gates0,
            state0=state0,
            area_cm2=area_cm2,
            Cm_abs=Cm_abs,
            Cx_abs=Cx_abs,
            Gx_abs=Gx_abs,
            Gax_e=Gax_e,
            Gax_i=Gax_i,
            left_i=left_i,
            right_i=right_i,
            left_e=left_e,
            right_e=right_e,
            I_background=I_background,
            intracellular_current_density_mid=Iinj_mid,
            extracellular_potential_mid_mV=vext_mid,
            extracellular_potential_initial_previous_mV=vext_previous,
            dt_ms=dt_ms,
        )

    return jax.vmap(one_batch)(
        intracellular_current_density_mid,
        extracellular_potential_mid_mV,
        extracellular_potential_initial_previous_mV,
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


@dataclass(frozen=True)
class DoubleCableBatchKernel:
    """Batch-oriented full double-cable kernel with shared axon structure.

    This intentionally keeps the first population constraint simple: all batch
    rows share geometry, membrane model, extracellular parameters, initial
    state, and time grid. Only imposed ``Vstim`` and optional ``Iinj`` vary.
    """

    runtime: SolverRuntime
    Veinit_mV: float = 0.0
    has_driven_extracellular: bool | None = None

    def run(
        self,
        *,
        extracellular_potential_mid_mV: Array | None = None,
        extracellular_potential_initial_previous_mV: Array | None = None,
        intracellular_current_density_mid: Array | None = None,
    ) -> BatchKernelResult:
        runtime = self.runtime
        extracellular = runtime.extracellular
        if extracellular is None:
            raise ValueError(
                "DoubleCableBatchKernel requires extracellular runtime arrays; "
                "prepare it with include_extracellular=True."
            )

        membrane_runtime = runtime.membrane
        grid = runtime.grid
        cable = runtime.cable
        dtype_local = membrane_runtime.dtype
        nx = membrane_runtime.Nx

        vext_mid = (
            runtime.stimulation.extracellular_potential_mid_mV
            if extracellular_potential_mid_mV is None
            else extracellular_potential_mid_mV
        )
        if vext_mid is None:
            raise ValueError(
                "extracellular_potential_mid_mV is required for double-cable batching."
            )
        vext_batch = _as_batched_time_space_array(
            "extracellular_potential_mid_mV",
            vext_mid,
            nt=grid.Nt,
            nx=nx,
            dtype_local=dtype_local,
        )
        batch_size = int(vext_batch.shape[0])

        vext_previous = (
            runtime.stimulation.extracellular_potential_initial_previous_mV
            if extracellular_potential_initial_previous_mV is None
            else extracellular_potential_initial_previous_mV
        )
        if vext_previous is None:
            raise ValueError(
                "extracellular_potential_initial_previous_mV is required for double-cable batching."
            )
        vext_previous_batch = _as_batched_space_array(
            "extracellular_potential_initial_previous_mV",
            vext_previous,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )

        iinj_mid = (
            runtime.stimulation.intracellular_current_density_mid
            if intracellular_current_density_mid is None
            else intracellular_current_density_mid
        )
        if iinj_mid is None:
            raise ValueError(
                "intracellular_current_density_mid is required for double-cable batching."
            )
        iinj_batch = _as_batched_time_space_array(
            "intracellular_current_density_mid",
            iinj_mid,
            nt=grid.Nt,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )

        Ve0 = jnp.full((nx,), dtype_local(self.Veinit_mV), dtype=dtype_local)
        Vm0 = membrane_runtime.Vm0_mV
        out = _run_double_cable_batch_vm_scan(
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
            Vi0_mV=Vm0 + Ve0,
            Ve0_mV=Ve0,
            gates0=membrane_runtime.gates0,
            state0=membrane_runtime.state0,
            area_cm2=cable.area_cm2,
            Cm_abs=extracellular.Cm_abs,
            Cx_abs=extracellular.Cx_abs,
            Gx_abs=extracellular.Gx_abs,
            Gax_e=extracellular.Gax_e,
            Gax_i=extracellular.Gax_i,
            left_i=extracellular.left_i,
            right_i=extracellular.right_i,
            left_e=extracellular.left_e,
            right_e=extracellular.right_e,
            I_background=membrane_runtime.background_current,
            intracellular_current_density_mid=iinj_batch,
            extracellular_potential_mid_mV=vext_batch,
            extracellular_potential_initial_previous_mV=vext_previous_batch,
            dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
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


def _as_batched_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int | None = None,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape != (nx,):
            raise ValueError(
                f"{name} must have shape (Nx,)=({nx},) or (B, Nx), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1:] != (nx,):
            raise ValueError(
                f"{name} must have trailing shape (Nx,)=({nx},), got {arr.shape}."
            )
    else:
        raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")

    if batch_size is None:
        return arr
    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, nx))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")


def _build_vstim_row(
    contexts: tuple[ExtracellularContext, ...],
    t_ms: Array,
    *,
    x_positions_row_m: Array,
    dtype_local: jnp.dtype,
) -> Array:
    nt = int(t_ms.shape[0])
    nx = int(x_positions_row_m.shape[0])
    vstim = jnp.zeros((nt, nx), dtype=dtype_local)
    for ctx in contexts:
        compiled = compile_extracellular_context(
            ctx,
            x_positions_row_m,
            dtype_local=dtype_local,
        )
        current_A = jax.vmap(compiled.stimulus)(t_ms)
        vstim = vstim + current_A[:, None] * compiled.footprint_V_per_A[None, :]
    return vstim * jnp.asarray(1e3, dtype=dtype_local)


def _normalize_context_row(row: ContextBatchRow) -> tuple[ExtracellularContext, ...]:
    if row is None:
        return ()
    if isinstance(row, ExtracellularContext):
        return (row,)
    return tuple(row)


def _resolve_dtype(axon, dtype_local: jnp.dtype | None) -> jnp.dtype:
    if dtype_local is not None:
        return dtype_local
    ion_channel = getattr(axon, "ion_channel", None)
    if ion_channel is not None and hasattr(ion_channel, "dtype"):
        return ion_channel.dtype
    return jnp.float32


def _resolve_x_positions_m(
    axon,
    x_positions_m: Array | None,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    if x_positions_m is None:
        x = jnp.asarray(axon.x, dtype=dtype_local) * jnp.asarray(1e-6, dtype=dtype_local)
    else:
        x = jnp.asarray(x_positions_m, dtype=dtype_local)

    if x.ndim == 1:
        return jnp.broadcast_to(x[jnp.newaxis, :], (batch_size, x.shape[0]))
    if x.ndim == 2:
        if x.shape[0] != batch_size:
            raise ValueError(
                f"x_positions_m has batch size {x.shape[0]}, expected {batch_size}."
            )
        return x
    raise ValueError(f"x_positions_m must have shape (Nx,) or (B, Nx), got {x.shape}.")


__all__ = [
    "BatchKernelResult",
    "build_vstim_batch",
    "build_vstim_initial_previous_batch",
    "build_vstim_midpoint_batch",
    "scale_extracellular_contexts",
    "SingleCableVStimBatchKernel",
    "DoubleCableBatchKernel",
]
