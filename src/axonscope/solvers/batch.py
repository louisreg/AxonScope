from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from axonscope.stimulation import ExtracellularContext
from axonscope.stimulus import Stimulus
from axonscope.stimulus_eval import evaluate_stimulus_numpy

from .batch_options import BatchOptions, BatchRecording
from .common import Array, apply_diffusion_operator, solve_block_tridiagonal_2x2_scalar
from .kernels import _run_double_cable_vm_scan, _run_single_cable_vstim_vm_scan
from .runtime import SolverRuntime
from .stimulus_runtime import compile_extracellular_context, compile_stimulus


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


def build_footprint_vstim_midpoint_batch(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    tsim_ms: float,
    dt_ms: float,
    amplitude_scale: float | Array = 1.0,
    dtype_local: jnp.dtype | None = None,
    engine: Literal["numpy", "jax"] = "numpy",
) -> Array:
    """Build midpoint Vstim from precomputed electrode footprints.

    This is the preferred population fast path for production use. The
    footprint can come from an analytic electrode, FEM interpolation, or any
    external field model, with shape ``(Nx,)`` or ``(B, Nx)`` in V/A.
    """

    dtype = jnp.float32 if dtype_local is None else dtype_local
    nt = int(jnp.ceil(tsim_ms / dt_ms))
    t_mid_ms = (
        jnp.arange(nt, dtype=dtype) + jnp.asarray(0.5, dtype=dtype)
    ) * jnp.asarray(dt_ms, dtype=dtype)
    return build_footprint_vstim_batch(
        stimulus=stimulus,
        footprint_V_per_A=footprint_V_per_A,
        t_ms=t_mid_ms,
        amplitude_scale=amplitude_scale,
        dtype_local=dtype,
        engine=engine,
    )


def build_footprint_vstim_initial_previous_batch(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    dt_ms: float,
    amplitude_scale: float | Array = 1.0,
    dtype_local: jnp.dtype | None = None,
    engine: Literal["numpy", "jax"] = "numpy",
) -> Array:
    """Build the previous imposed field from precomputed footprints.

    Returns ``Vstim[B, Nx]`` sampled at ``t = -dt/2`` in mV.
    """

    dtype = jnp.float32 if dtype_local is None else dtype_local
    samples = build_footprint_vstim_batch(
        stimulus=stimulus,
        footprint_V_per_A=footprint_V_per_A,
        t_ms=jnp.asarray([-0.5 * dt_ms], dtype=dtype),
        amplitude_scale=amplitude_scale,
        dtype_local=dtype,
        engine=engine,
    )
    return samples[:, 0, :]


def build_footprint_vstim_batch(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    t_ms: Array,
    amplitude_scale: float | Array = 1.0,
    dtype_local: jnp.dtype | None = None,
    engine: Literal["numpy", "jax"] = "numpy",
) -> Array:
    """Build Vstim samples from batched electrode footprints.

    ``footprint_V_per_A`` is the static spatial field per ampere. The returned
    array has shape ``(B, Nt, Nx)`` and units mV.
    """

    dtype = jnp.float32 if dtype_local is None else dtype_local
    if engine == "numpy":
        return _build_footprint_vstim_batch_numpy(
            stimulus=stimulus,
            footprint_V_per_A=footprint_V_per_A,
            t_ms=t_ms,
            amplitude_scale=amplitude_scale,
            dtype_local=dtype,
        )
    if engine != "jax":
        raise ValueError(f"engine must be 'numpy' or 'jax', got {engine!r}.")

    t = jnp.asarray(t_ms, dtype=dtype)
    if t.ndim != 1:
        raise ValueError(f"t_ms must have shape (Nt,), got {t.shape}.")

    batch_size = _infer_footprint_batch_size(footprint_V_per_A, (amplitude_scale,))
    footprint = _as_footprint_batch(
        "footprint_V_per_A",
        footprint_V_per_A,
        batch_size=batch_size,
        dtype_local=dtype,
    )
    scale = _as_batch_vector(
        "amplitude_scale",
        amplitude_scale,
        batch_size=batch_size,
        dtype_local=dtype,
    )
    current_A = jax.vmap(compile_stimulus(stimulus, dtype_local=dtype))(t)
    return (
        scale[:, None, None]
        * current_A[None, :, None]
        * footprint[:, None, :]
        * jnp.asarray(1e3, dtype=dtype)
    )


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


@partial(
    jax.jit,
    static_argnames=(
        "backend",
        "membrane",
        "has_driven_extracellular",
        "stateless_vm_only",
        "record_full",
    ),
)
def _run_single_cable_vstim_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
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
    record_indices: Array,
    dt_ms: Array,
) -> tuple[Array, Array, tuple[Array, ...], Array]:
    """Run one time chunk and return final batch state plus recorded Vm."""

    def one_batch(Vm0_row, gates0_row, state0_row, Iinj_mid, vext_mid):
        vstim_forcing_mid = jax.vmap(
            lambda values: apply_diffusion_operator(values, lower, diag, upper)
        )(vext_mid)

        def step(carry, step_inputs):
            Iinj, vstim_force = step_inputs
            Vm, gates, *extra = carry
            extra = tuple(extra)

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)
            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Gm, GE = backend.membrane_conductance_terms(linearization_gates)
            d = d_static + (dt_ms / Cm_uF_cm2) * Gm
            rhs = (
                Vm
                + dt_ms * vstim_force
                + (dt_ms / Cm_uF_cm2)
                * (
                    GE
                    + Iinj
                    - explicit_outward_current
                    - correction_current
                )
            )
            Vm_new = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]

            if stateless_vm_only:
                output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
                return (Vm_new, gates_pred, *extra), output

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
            return (Vm_new, gates_new, *state_new), output

        final_carry, trace = jax.lax.scan(
            step,
            (Vm0_row, gates0_row, *state0_row),
            (Iinj_mid, vstim_forcing_mid),
        )
        return final_carry[0], final_carry[1], tuple(final_carry[2:]), trace

    state_axes = tuple(0 for _ in state0)
    return jax.vmap(one_batch, in_axes=(0, 0, state_axes, 0, 0))(
        Vm0_mV,
        gates0,
        state0,
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
        "record_full",
    ),
)
def _run_double_cable_batch_stateful_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    record_full: bool,
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
    record_indices: Array,
    dt_ms: Array,
) -> tuple[Array, Array, Array, tuple[Array, ...], Array]:
    """Run one double-cable time chunk and return final batch state."""

    cm_over_dt = Cm_abs / dt_ms
    cx_over_dt = Cx_abs / dt_ms
    intracellular_current_abs_mid = intracellular_current_density_mid * area_cm2[None, None, :]
    off_i = -Gax_i
    off_e = -Gax_e

    def one_batch(Vi0_row, Ve0_row, gates0_row, state0_row, Iinj_abs_mid, vext_mid, vext_prev0):
        vext_previous_mV = jnp.concatenate([vext_prev0[None, :], vext_mid[:-1]], axis=0)
        extracellular_rhs_drive = (
            (cx_over_dt + Gx_abs)[None, :] * vext_mid
            - cx_over_dt[None, :] * vext_previous_mV
        )

        def solve_vi_vperi(
            Vi: Array,
            Ve: Array,
            gates_new: Array,
            Iinj_abs: Array,
            I_outward_den: Array,
            I_corr_den: Array,
            extracellular_drive_abs: Array,
        ) -> tuple[Array, Array]:
            Gm_den, GE_den = backend.membrane_conductance_terms(gates_new)
            Gm_abs = Gm_den * area_cm2
            GE_abs = GE_den * area_cm2

            I_outward_abs = I_outward_den * area_cm2
            I_corr_abs = I_corr_den * area_cm2
            Vm = Vi - Ve

            a00 = cm_over_dt + Gm_abs + left_i + right_i
            a01 = -(cm_over_dt + Gm_abs)
            rhs0 = (
                cm_over_dt * Vm
                + GE_abs
                + Iinj_abs
                - I_outward_abs
                - I_corr_abs
            )

            a10 = a01
            a11 = cm_over_dt + Gm_abs + cx_over_dt + Gx_abs + left_e + right_e
            rhs1 = (
                -cm_over_dt * Vm
                - GE_abs
                + cx_over_dt * Ve
                + extracellular_drive_abs
                + I_outward_abs
                + I_corr_abs
            )

            return solve_block_tridiagonal_2x2_scalar(
                a00,
                a01,
                a10,
                a11,
                off_i,
                off_e,
                rhs0,
                rhs1,
            )

        def step(carry, step_inputs):
            Iinj_abs, extracellular_drive_abs = step_inputs
            Vi, Ve, gates, *extra = carry
            extra = tuple(extra)
            Vm = Vi - Ve

            gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt_ms)

            if stateless_vm_only:
                linearization_gates = gates if has_driven_extracellular else gates_pred
                explicit_outward_current = I_background
                correction_current = jnp.zeros_like(Vm)
            else:
                Iion_pred = backend.currents(V_mV=Vm, gates=gates_pred)
                step_plan_pred = membrane.prepare_membrane_step(
                    V_mV=Vm,
                    gates_prev=gates,
                    gates_new=gates_pred,
                    state=extra,
                    dt=dt_ms,
                    I_ion=Iion_pred,
                    I_background=I_background,
                )
                linearization_gates = step_plan_pred.linearization_gates
                if has_driven_extracellular:
                    linearization_gates = gates
                explicit_outward_current = step_plan_pred.explicit_outward_current
                correction_current = step_plan_pred.correction_current

            Vi_new, Ve_new = solve_vi_vperi(
                Vi=Vi,
                Ve=Ve,
                gates_new=linearization_gates,
                Iinj_abs=Iinj_abs,
                I_outward_den=explicit_outward_current,
                I_corr_den=correction_current,
                extracellular_drive_abs=extracellular_drive_abs,
            )
            Vm_new = Vi_new - Ve_new

            if stateless_vm_only:
                output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
                return (Vi_new, Ve_new, gates_pred, *extra), output

            gates_new = membrane.final_gate_update(
                gates_prev=gates,
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                dt=dt_ms,
                gates_predictor=gates_pred,
            )
            Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
            step_plan = membrane.prepare_membrane_step(
                V_mV=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state=extra,
                dt=dt_ms,
                I_ion=Iion_new,
                I_background=I_background,
            )
            state_new = membrane.finalize_membrane_step(
                V_mV_prev=Vm,
                V_mV_new=Vm_new,
                gates_prev=gates,
                gates_new=gates_new,
                state_prev=extra,
                step_plan=step_plan,
                dt=dt_ms,
            )
            output = Vm_new if record_full else jnp.take(Vm_new, record_indices, axis=0)
            return (Vi_new, Ve_new, gates_new, *state_new), output

        final_carry, trace = jax.lax.scan(
            step,
            (Vi0_row, Ve0_row, gates0_row, *state0_row),
            (Iinj_abs_mid, extracellular_rhs_drive),
        )
        return final_carry[0], final_carry[1], final_carry[2], tuple(final_carry[3:]), trace

    state_axes = tuple(0 for _ in state0)
    return jax.vmap(one_batch, in_axes=(0, 0, 0, state_axes, 0, 0, 0))(
        Vi0_mV,
        Ve0_mV,
        gates0,
        state0,
        intracellular_current_abs_mid,
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
        options: BatchOptions | None = None,
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
        options = _normalize_batch_options(options)
        record_idx, record_full = _resolve_recording(options.recording, nx=membrane_runtime.Nx)
        chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
        has_driven_extracellular = (
            runtime.stimulation.has_driven_extracellular
            if self.has_driven_extracellular is None
            else bool(self.has_driven_extracellular)
        )
        stateless_vm_only = bool(
            membrane_runtime.membrane.supports_stateless_vm_only_fast_path()
        )
        if record_full and chunk_steps is None:
            out = _run_single_cable_vstim_batch_vm_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
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
        else:
            out = _run_single_cable_vstim_batch_array_chunks(
                runtime=runtime,
                Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=dtype_local),
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                intracellular_current_density_mid=iinj_batch,
                extracellular_potential_mid_mV=vext_batch,
                record_indices=record_idx,
                record_full=record_full,
                time_chunk_steps=chunk_steps,
            )
        return BatchKernelResult(Vm=out, t=grid.t_vec_ms)

    def run_footprint(
        self,
        *,
        stimulus: Stimulus,
        footprint_V_per_A: Array,
        amplitude_scale: float | Array = 1.0,
        options: BatchOptions | None = None,
    ) -> BatchKernelResult:
        """Run a footprint-driven population without materializing full Vstim."""

        runtime = self.runtime
        if runtime.extracellular is not None:
            raise ValueError(
                "SingleCableVStimBatchKernel expects a scalar single-cable runtime; "
                "prepare it with include_extracellular=False."
            )
        if runtime.stimulation.intracellular_current_density_mid is None:
            raise ValueError("precomputed intracellular current density is required.")
        out = _run_single_cable_footprint_chunks(
            runtime=runtime,
            Cm_uF_cm2=jnp.asarray(self.Cm_uF_cm2, dtype=runtime.membrane.dtype),
            stimulus=stimulus,
            footprint_V_per_A=footprint_V_per_A,
            amplitude_scale=amplitude_scale,
            has_driven_extracellular=(
                runtime.stimulation.has_driven_extracellular
                if self.has_driven_extracellular is None
                else bool(self.has_driven_extracellular)
            ),
            stateless_vm_only=bool(
                runtime.membrane.membrane.supports_stateless_vm_only_fast_path()
            ),
            options=_normalize_batch_options(options),
        )
        return BatchKernelResult(Vm=out, t=runtime.grid.t_vec_ms)


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
        options: BatchOptions | None = None,
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

        options = _normalize_batch_options(options)
        record_idx, record_full = _resolve_recording(options.recording, nx=nx)
        chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
        has_driven_extracellular = (
            runtime.stimulation.has_driven_extracellular
            if self.has_driven_extracellular is None
            else bool(self.has_driven_extracellular)
        )
        stateless_vm_only = bool(
            membrane_runtime.membrane.supports_stateless_vm_only_fast_path()
        )
        if record_full and chunk_steps is None:
            Ve0 = jnp.full((nx,), dtype_local(self.Veinit_mV), dtype=dtype_local)
            Vm0 = membrane_runtime.Vm0_mV
            out = _run_double_cable_batch_vm_scan(
                backend=membrane_runtime.backend,
                membrane=membrane_runtime.membrane,
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
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
        else:
            out = _run_double_cable_batch_array_chunks(
                runtime=runtime,
                Veinit_mV=float(self.Veinit_mV),
                has_driven_extracellular=has_driven_extracellular,
                stateless_vm_only=stateless_vm_only,
                intracellular_current_density_mid=iinj_batch,
                extracellular_potential_mid_mV=vext_batch,
                extracellular_potential_initial_previous_mV=vext_previous_batch,
                record_indices=record_idx,
                record_full=record_full,
                time_chunk_steps=chunk_steps,
            )
        return BatchKernelResult(Vm=out, t=grid.t_vec_ms)

    def run_footprint(
        self,
        *,
        stimulus: Stimulus,
        footprint_V_per_A: Array,
        amplitude_scale: float | Array = 1.0,
        options: BatchOptions | None = None,
    ) -> BatchKernelResult:
        """Run a footprint-driven double-cable population in time chunks."""

        runtime = self.runtime
        if runtime.extracellular is None:
            raise ValueError(
                "DoubleCableBatchKernel requires extracellular runtime arrays; "
                "prepare it with include_extracellular=True."
            )
        if runtime.stimulation.intracellular_current_density_mid is None:
            raise ValueError("precomputed intracellular current density is required.")
        out = _run_double_cable_footprint_chunks(
            runtime=runtime,
            Veinit_mV=float(self.Veinit_mV),
            stimulus=stimulus,
            footprint_V_per_A=footprint_V_per_A,
            amplitude_scale=amplitude_scale,
            has_driven_extracellular=(
                runtime.stimulation.has_driven_extracellular
                if self.has_driven_extracellular is None
                else bool(self.has_driven_extracellular)
            ),
            stateless_vm_only=bool(
                runtime.membrane.membrane.supports_stateless_vm_only_fast_path()
            ),
            options=_normalize_batch_options(options),
        )
        return BatchKernelResult(Vm=out, t=runtime.grid.t_vec_ms)


def _run_single_cable_vstim_batch_array_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    record_indices: Array,
    record_full: bool,
    time_chunk_steps: int | None,
) -> Array:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    batch_size = int(extracellular_potential_mid_mV.shape[0])
    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    chunks = []

    for start, stop in _time_chunks(grid.Nt, time_chunk_steps):
        Vm, gates, state, trace = _run_single_cable_vstim_batch_stateful_scan(
            backend=membrane_runtime.backend,
            membrane=membrane_runtime.membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            record_full=record_full,
            lower=cable.lower,
            diag=cable.diag,
            upper=cable.upper,
            dl=-dt * cable.lower,
            d_static=jnp.ones_like(cable.diag) - dt * cable.diag,
            du=-dt * cable.upper,
            Cm_uF_cm2=Cm_uF_cm2,
            I_background=membrane_runtime.background_current,
            Vm0_mV=Vm,
            gates0=gates,
            state0=state,
            intracellular_current_density_mid=intracellular_current_density_mid[:, start:stop],
            extracellular_potential_mid_mV=extracellular_potential_mid_mV[:, start:stop],
            record_indices=record_indices,
            dt_ms=dt,
        )
        chunks.append(trace)

    return _concat_trace_chunks(chunks)


def _run_single_cable_footprint_chunks(
    *,
    runtime: SolverRuntime,
    Cm_uF_cm2: Array,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    amplitude_scale: float | Array,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    options: BatchOptions,
) -> Array:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    cable = runtime.cable
    dtype_local = membrane_runtime.dtype
    nx = membrane_runtime.Nx
    _validate_footprint_width(footprint_V_per_A, nx=nx)
    batch_size = _infer_footprint_batch_size(footprint_V_per_A, (amplitude_scale,))
    record_idx, record_full = _resolve_recording(options.recording, nx=nx)
    chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
    dt = jnp.asarray(grid.dt_ms, dtype=dtype_local)
    iinj_mid = runtime.stimulation.intracellular_current_density_mid
    if iinj_mid is None:
        raise ValueError("precomputed intracellular current density is required.")

    Vm, gates, state = _initial_single_cable_batch_state(runtime, batch_size)
    chunks = []
    for start, stop in _time_chunks(grid.Nt, chunk_steps):
        t_mid = _midpoint_times_for_chunk(start, stop, dt_ms=grid.dt_ms, dtype_local=dtype_local)
        vext_chunk = build_footprint_vstim_batch(
            stimulus=stimulus,
            footprint_V_per_A=footprint_V_per_A,
            t_ms=t_mid,
            amplitude_scale=amplitude_scale,
            dtype_local=dtype_local,
        )
        iinj_chunk = _as_batched_time_space_array(
            "intracellular_current_density_mid",
            iinj_mid[start:stop],
            nt=stop - start,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        Vm, gates, state, trace = _run_single_cable_vstim_batch_stateful_scan(
            backend=membrane_runtime.backend,
            membrane=membrane_runtime.membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            record_full=record_full,
            lower=cable.lower,
            diag=cable.diag,
            upper=cable.upper,
            dl=-dt * cable.lower,
            d_static=jnp.ones_like(cable.diag) - dt * cable.diag,
            du=-dt * cable.upper,
            Cm_uF_cm2=Cm_uF_cm2,
            I_background=membrane_runtime.background_current,
            Vm0_mV=Vm,
            gates0=gates,
            state0=state,
            intracellular_current_density_mid=iinj_chunk,
            extracellular_potential_mid_mV=vext_chunk,
            record_indices=record_idx,
            dt_ms=dt,
        )
        chunks.append(trace)

    return _concat_trace_chunks(chunks)


def _run_double_cable_batch_array_chunks(
    *,
    runtime: SolverRuntime,
    Veinit_mV: float,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    intracellular_current_density_mid: Array,
    extracellular_potential_mid_mV: Array,
    extracellular_potential_initial_previous_mV: Array,
    record_indices: Array,
    record_full: bool,
    time_chunk_steps: int | None,
) -> Array:
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    dtype_local = membrane_runtime.dtype
    batch_size = int(extracellular_potential_mid_mV.shape[0])
    Vi, Ve, gates, state = _initial_double_cable_batch_state(runtime, batch_size, Veinit_mV)
    previous = extracellular_potential_initial_previous_mV
    chunks = []

    for start, stop in _time_chunks(grid.Nt, time_chunk_steps):
        vext_chunk = extracellular_potential_mid_mV[:, start:stop]
        Vi, Ve, gates, state, trace = _run_double_cable_batch_stateful_scan(
            backend=membrane_runtime.backend,
            membrane=membrane_runtime.membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            record_full=record_full,
            Vi0_mV=Vi,
            Ve0_mV=Ve,
            gates0=gates,
            state0=state,
            area_cm2=runtime.cable.area_cm2,
            Cm_abs=runtime.extracellular.Cm_abs,
            Cx_abs=runtime.extracellular.Cx_abs,
            Gx_abs=runtime.extracellular.Gx_abs,
            Gax_e=runtime.extracellular.Gax_e,
            Gax_i=runtime.extracellular.Gax_i,
            left_i=runtime.extracellular.left_i,
            right_i=runtime.extracellular.right_i,
            left_e=runtime.extracellular.left_e,
            right_e=runtime.extracellular.right_e,
            I_background=membrane_runtime.background_current,
            intracellular_current_density_mid=intracellular_current_density_mid[:, start:stop],
            extracellular_potential_mid_mV=vext_chunk,
            extracellular_potential_initial_previous_mV=previous,
            record_indices=record_indices,
            dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
        )
        previous = vext_chunk[:, -1]
        chunks.append(trace)

    return _concat_trace_chunks(chunks)


def _run_double_cable_footprint_chunks(
    *,
    runtime: SolverRuntime,
    Veinit_mV: float,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    amplitude_scale: float | Array,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    options: BatchOptions,
) -> Array:
    if runtime.extracellular is None:
        raise ValueError("Double-cable footprint chunks require extracellular runtime arrays.")
    membrane_runtime = runtime.membrane
    grid = runtime.grid
    dtype_local = membrane_runtime.dtype
    nx = membrane_runtime.Nx
    _validate_footprint_width(footprint_V_per_A, nx=nx)
    batch_size = _infer_footprint_batch_size(footprint_V_per_A, (amplitude_scale,))
    record_idx, record_full = _resolve_recording(options.recording, nx=nx)
    chunk_steps = _normalize_time_chunk_steps(options.time_chunk_steps, nt=grid.Nt)
    iinj_mid = runtime.stimulation.intracellular_current_density_mid
    if iinj_mid is None:
        raise ValueError("precomputed intracellular current density is required.")

    Vi, Ve, gates, state = _initial_double_cable_batch_state(runtime, batch_size, Veinit_mV)
    previous = build_footprint_vstim_initial_previous_batch(
        stimulus=stimulus,
        footprint_V_per_A=footprint_V_per_A,
        amplitude_scale=amplitude_scale,
        dt_ms=grid.dt_ms,
        dtype_local=dtype_local,
    )
    chunks = []

    for start, stop in _time_chunks(grid.Nt, chunk_steps):
        t_mid = _midpoint_times_for_chunk(start, stop, dt_ms=grid.dt_ms, dtype_local=dtype_local)
        vext_chunk = build_footprint_vstim_batch(
            stimulus=stimulus,
            footprint_V_per_A=footprint_V_per_A,
            t_ms=t_mid,
            amplitude_scale=amplitude_scale,
            dtype_local=dtype_local,
        )
        iinj_chunk = _as_batched_time_space_array(
            "intracellular_current_density_mid",
            iinj_mid[start:stop],
            nt=stop - start,
            nx=nx,
            dtype_local=dtype_local,
            batch_size=batch_size,
        )
        Vi, Ve, gates, state, trace = _run_double_cable_batch_stateful_scan(
            backend=membrane_runtime.backend,
            membrane=membrane_runtime.membrane,
            has_driven_extracellular=has_driven_extracellular,
            stateless_vm_only=stateless_vm_only,
            record_full=record_full,
            Vi0_mV=Vi,
            Ve0_mV=Ve,
            gates0=gates,
            state0=state,
            area_cm2=runtime.cable.area_cm2,
            Cm_abs=runtime.extracellular.Cm_abs,
            Cx_abs=runtime.extracellular.Cx_abs,
            Gx_abs=runtime.extracellular.Gx_abs,
            Gax_e=runtime.extracellular.Gax_e,
            Gax_i=runtime.extracellular.Gax_i,
            left_i=runtime.extracellular.left_i,
            right_i=runtime.extracellular.right_i,
            left_e=runtime.extracellular.left_e,
            right_e=runtime.extracellular.right_e,
            I_background=membrane_runtime.background_current,
            intracellular_current_density_mid=iinj_chunk,
            extracellular_potential_mid_mV=vext_chunk,
            extracellular_potential_initial_previous_mV=previous,
            record_indices=record_idx,
            dt_ms=jnp.asarray(grid.dt_ms, dtype=dtype_local),
        )
        previous = vext_chunk[:, -1]
        chunks.append(trace)

    return _concat_trace_chunks(chunks)


def _initial_single_cable_batch_state(
    runtime: SolverRuntime,
    batch_size: int,
) -> tuple[Array, Array, tuple[Array, ...]]:
    membrane_runtime = runtime.membrane
    Vm = _broadcast_batch_leading(membrane_runtime.Vm0_mV, batch_size)
    gates = _broadcast_batch_leading(membrane_runtime.gates0, batch_size)
    state = tuple(_broadcast_batch_leading(values, batch_size) for values in membrane_runtime.state0)
    return Vm, gates, state


def _initial_double_cable_batch_state(
    runtime: SolverRuntime,
    batch_size: int,
    Veinit_mV: float,
) -> tuple[Array, Array, Array, tuple[Array, ...]]:
    membrane_runtime = runtime.membrane
    dtype_local = membrane_runtime.dtype
    nx = membrane_runtime.Nx
    Ve0 = jnp.full((nx,), dtype_local(Veinit_mV), dtype=dtype_local)
    Vi0 = membrane_runtime.Vm0_mV + Ve0
    Vi = _broadcast_batch_leading(Vi0, batch_size)
    Ve = _broadcast_batch_leading(Ve0, batch_size)
    gates = _broadcast_batch_leading(membrane_runtime.gates0, batch_size)
    state = tuple(_broadcast_batch_leading(values, batch_size) for values in membrane_runtime.state0)
    return Vi, Ve, gates, state


def _broadcast_batch_leading(values: Array, batch_size: int) -> Array:
    arr = jnp.asarray(values)
    return jnp.broadcast_to(arr, (batch_size, *arr.shape))


def _normalize_batch_options(options: BatchOptions | None) -> BatchOptions:
    return BatchOptions.full() if options is None else options


def _resolve_recording(recording: BatchRecording, *, nx: int) -> tuple[Array, bool]:
    indices = recording.indices_for(nx)
    if indices is None:
        return jnp.arange(nx, dtype=jnp.int32), True
    return jnp.asarray(indices, dtype=jnp.int32), False


def _normalize_time_chunk_steps(time_chunk_steps: int | None, *, nt: int) -> int | None:
    if time_chunk_steps is None:
        return None
    steps = int(time_chunk_steps)
    if steps < 1:
        raise ValueError("time_chunk_steps must be >= 1.")
    if steps >= nt:
        return None
    return steps


def _time_chunks(nt: int, time_chunk_steps: int | None):
    chunk_steps = nt if time_chunk_steps is None else time_chunk_steps
    for start in range(0, nt, chunk_steps):
        yield start, min(start + chunk_steps, nt)


def _concat_trace_chunks(chunks: list[Array]) -> Array:
    if not chunks:
        raise ValueError("at least one time chunk is required.")
    if len(chunks) == 1:
        return chunks[0]
    return jnp.concatenate(chunks, axis=1)


def _midpoint_times_for_chunk(
    start: int,
    stop: int,
    *,
    dt_ms: float,
    dtype_local: jnp.dtype,
) -> Array:
    return (
        jnp.arange(start, stop, dtype=dtype_local) + jnp.asarray(0.5, dtype=dtype_local)
    ) * jnp.asarray(dt_ms, dtype=dtype_local)


def _validate_footprint_width(footprint_V_per_A: Array, *, nx: int) -> None:
    shape, ndim = _shape_and_ndim(footprint_V_per_A)
    if ndim not in (1, 2):
        raise ValueError(
            "footprint_V_per_A must have shape (Nx,) or (B, Nx), "
            f"got {shape}."
        )
    if shape[-1] != nx:
        raise ValueError(f"footprint_V_per_A trailing size must be Nx={nx}, got {shape}.")


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


def _build_footprint_vstim_batch_numpy(
    *,
    stimulus: Stimulus,
    footprint_V_per_A: Array,
    t_ms: Array,
    amplitude_scale: float | Array,
    dtype_local: jnp.dtype,
) -> Array:
    np_dtype = np.dtype(dtype_local)
    t = np.asarray(t_ms, dtype=np_dtype)
    if t.ndim != 1:
        raise ValueError(f"t_ms must have shape (Nt,), got {t.shape}.")

    batch_size = _infer_footprint_batch_size(footprint_V_per_A, (amplitude_scale,))
    footprint = _as_footprint_batch_numpy(
        "footprint_V_per_A",
        footprint_V_per_A,
        batch_size=batch_size,
        dtype_local=np_dtype,
    )
    scale = _as_batch_vector_numpy(
        "amplitude_scale",
        amplitude_scale,
        batch_size=batch_size,
        dtype_local=np_dtype,
    )
    current_A = evaluate_stimulus_numpy(stimulus, t).astype(np_dtype, copy=False)
    vstim_mV = scale[:, None, None] * current_A[None, :, None] * footprint[:, None, :]
    vstim_mV = vstim_mV * np.asarray(1e3, dtype=np_dtype)
    return jnp.asarray(vstim_mV, dtype=dtype_local)


def _infer_footprint_batch_size(footprint_V_per_A: Array, params: Sequence[object]) -> int:
    candidates = []
    footprint_shape, footprint_ndim = _shape_and_ndim(footprint_V_per_A)
    if footprint_ndim == 2:
        candidates.append(int(footprint_shape[0]))
    elif footprint_ndim != 1:
        raise ValueError(
            "footprint_V_per_A must have shape (Nx,) or (B, Nx), "
            f"got {footprint_shape}."
        )

    for values in params:
        shape, ndim = _shape_and_ndim(values)
        if ndim == 1 and shape[0] != 1:
            candidates.append(int(shape[0]))
        elif ndim > 1:
            raise ValueError(f"batched parameter must be scalar or shape (B,), got {shape}.")

    if not candidates:
        return 1
    batch_size = candidates[0]
    if any(candidate != batch_size for candidate in candidates):
        raise ValueError(f"batched inputs disagree on batch size: {candidates}.")
    return batch_size


def _as_footprint_batch(
    name: str,
    values: Array,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        return jnp.broadcast_to(arr[jnp.newaxis, :], (batch_size, arr.shape[0]))
    if arr.ndim == 2:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (batch_size, arr.shape[1]))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")


def _as_footprint_batch_numpy(
    name: str,
    values: Array,
    *,
    batch_size: int,
    dtype_local: np.dtype,
) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        return np.broadcast_to(arr[None, :], (batch_size, arr.shape[0]))
    if arr.ndim == 2:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.broadcast_to(arr, (batch_size, arr.shape[1]))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")


def _as_batch_vector(
    name: str,
    values: float | Array,
    *,
    batch_size: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return jnp.full((batch_size,), arr, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (batch_size,))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must be scalar or have shape (B,), got {arr.shape}.")


def _as_batch_vector_numpy(
    name: str,
    values: float | Array,
    *,
    batch_size: int,
    dtype_local: np.dtype,
) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return np.full((batch_size,), arr, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.broadcast_to(arr, (batch_size,))
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must be scalar or have shape (B,), got {arr.shape}.")


def _shape_and_ndim(values: object) -> tuple[tuple[int, ...], int]:
    shape = getattr(values, "shape", None)
    ndim = getattr(values, "ndim", None)
    if shape is not None and ndim is not None:
        return tuple(int(dim) for dim in shape), int(ndim)
    arr = np.asarray(values)
    return arr.shape, arr.ndim


__all__ = [
    "BatchKernelResult",
    "BatchOptions",
    "BatchRecording",
    "build_footprint_vstim_batch",
    "build_footprint_vstim_initial_previous_batch",
    "build_footprint_vstim_midpoint_batch",
    "build_vstim_batch",
    "build_vstim_initial_previous_batch",
    "build_vstim_midpoint_batch",
    "scale_extracellular_contexts",
    "SingleCableVStimBatchKernel",
    "DoubleCableBatchKernel",
]
