"""Runnable series-reduction pseudo-double validation path."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax.linalg import tridiagonal_solve

import axonscope as axs
from axonscope.axon_instance import AxonInstance
from axonscope.results import AxonSimulationResult
from axonscope.results.pool import CohortResult
from axonscope.backends.jax.runtime import prepare_solver_runtime

from benchmark.pseudo_double.reductions import (
    series_equivalent,
    tridiagonal_edges_to_jax,
)


@dataclass(frozen=True)
class PseudoDoubleSeriesConfig:
    """Experimental local RC-series reduction parameters."""

    vext_scale: float = 1.0
    capacitance_floor_fraction: float = 0.02
    conductance_floor_fraction: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "vext_scale": float(self.vext_scale),
            "capacitance_floor_fraction": float(self.capacitance_floor_fraction),
            "conductance_floor_fraction": float(self.conductance_floor_fraction),
        }


def _apply_edge_operator(values, edges):
    """Apply an absolute-conductance edge diffusion operator."""

    out = jnp.zeros_like(values)
    if values.shape[0] >= 2:
        out = out.at[0].set(edges[0] * (values[1] - values[0]))
        out = out.at[-1].set(edges[-1] * (values[-2] - values[-1]))
    if values.shape[0] > 2:
        out = out.at[1:-1].set(
            edges[:-1] * (values[:-2] - values[1:-1])
            + edges[1:] * (values[2:] - values[1:-1])
        )
    return out


def _series_or_axolemma(axolemma, myelin, *, floor_fraction, eps):
    """Use a series equivalent where the myelin branch exists, else axolemma."""

    raw = series_equivalent(axolemma, myelin, eps=eps)
    floored = jnp.maximum(raw, floor_fraction * axolemma)
    return jnp.where(myelin > eps, floored, axolemma)


@partial(
    jax.jit,
    static_argnames=("backend", "membrane", "has_driven_extracellular", "stateless_vm_only"),
)
def _run_series_vm_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    Vm0_mV,
    gates0,
    state0,
    area_cm2,
    Cm_abs,
    Cx_abs,
    Gx_abs,
    Gax_i,
    left_i,
    right_i,
    I_background,
    intracellular_current_density_mid,
    extracellular_potential_mid_mV,
    dt_ms,
    vext_scale,
    capacitance_floor_fraction,
    conductance_floor_fraction,
):
    eps = jnp.asarray(1e-12, dtype=Cm_abs.dtype)
    dt = jnp.asarray(dt_ms, dtype=Cm_abs.dtype)
    Ceff_abs = _series_or_axolemma(
        Cm_abs,
        Cx_abs,
        floor_fraction=capacitance_floor_fraction,
        eps=eps,
    )
    c_over_dt = Ceff_abs / dt
    vext_mid_scaled = extracellular_potential_mid_mV * vext_scale
    vstim_force_abs = jax.vmap(lambda values: _apply_edge_operator(values, Gax_i))(
        vext_mid_scaled
    )
    intracellular_current_abs_mid = intracellular_current_density_mid * area_cm2[None, :]

    def step(carry, step_inputs):
        Iinj_abs, vstim_force_abs_step = step_inputs
        Vm, gates, *extra = carry
        extra = tuple(extra)

        gates_pred = backend.cn_gate_update(g_prev=gates, V_mV=Vm, dt=dt)
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
                dt=dt,
                I_ion=Iion_pred,
                I_background=I_background,
            )
            linearization_gates = step_plan_pred.linearization_gates
            if has_driven_extracellular:
                linearization_gates = gates
            explicit_outward_current = step_plan_pred.explicit_outward_current
            correction_current = step_plan_pred.correction_current

        Gm_den, GE_den = backend.membrane_conductance_terms(linearization_gates)
        Gm_abs = Gm_den * area_cm2
        GE_abs = GE_den * area_cm2
        Gm_eff_abs = _series_or_axolemma(
            Gm_abs,
            Gx_abs,
            floor_fraction=conductance_floor_fraction,
            eps=eps,
        )
        source_ratio = jnp.where(jnp.abs(Gm_abs) > eps, Gm_eff_abs / Gm_abs, 1.0)

        I_outward_abs = explicit_outward_current * area_cm2 * source_ratio
        I_corr_abs = correction_current * area_cm2 * source_ratio
        GE_eff_abs = GE_abs * source_ratio

        diag_abs = c_over_dt + Gm_eff_abs + left_i + right_i
        dl, d, du = tridiagonal_edges_to_jax(-Gax_i, diag_abs, -Gax_i)
        rhs = (
            c_over_dt * Vm
            + vstim_force_abs_step
            + GE_eff_abs
            + Iinj_abs
            - I_outward_abs
            - I_corr_abs
        )
        Vm_new = tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]

        if stateless_vm_only:
            return (Vm_new, gates_pred, *extra), Vm_new

        gates_new = membrane.final_gate_update(
            gates_prev=gates,
            V_mV_prev=Vm,
            V_mV_new=Vm_new,
            dt=dt,
            gates_predictor=gates_pred,
        )
        Iion_new = backend.currents(V_mV=Vm_new, gates=gates_new)
        step_plan = membrane.prepare_membrane_step(
            V_mV=Vm_new,
            gates_prev=gates,
            gates_new=gates_new,
            state=extra,
            dt=dt,
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
            dt=dt,
        )
        return (Vm_new, gates_new, *state_new), Vm_new

    init_carry = (Vm0_mV, gates0, *state0)
    _, vm_trace = jax.lax.scan(
        step,
        init_carry,
        (intracellular_current_abs_mid, vstim_force_abs),
    )
    return vm_trace


def _record_indices(width: int, *, recording: str, probe_count: int) -> tuple[int, ...] | None:
    if recording == "full":
        return None
    if recording == "center":
        return (int(width // 2),)
    if recording == "probes":
        count = max(1, min(int(probe_count), int(width)))
        return tuple(int(value) for value in np.unique(np.linspace(0, width - 1, count, dtype=int)))
    raise ValueError("recording must be 'full', 'center', or 'probes'.")


def run_series_population(
    instances: Sequence[AxonInstance],
    *,
    duration_ms: float,
    dt_ms: float,
    recording: str,
    probe_count: int,
    config: PseudoDoubleSeriesConfig,
) -> AxonSimulationResult:
    """Run one pseudo-double RC-series validation population."""

    rows = []
    t_vec = None
    for index, instance in enumerate(instances):
        runtime = prepare_solver_runtime(
            instance,
            duration_ms,
            dt_ms,
            include_extracellular=True,
            include_area=True,
            precompute_intracellular=True,
            precompute_extracellular=True,
        )
        extracellular = runtime.extracellular
        if extracellular is None:
            raise ValueError("Series pseudo-double requires extracellular arrays.")
        dtype = runtime.membrane.dtype
        vm = _run_series_vm_scan(
            backend=runtime.membrane.backend,
            membrane=runtime.membrane.membrane,
            has_driven_extracellular=runtime.stimulation.has_driven_extracellular,
            stateless_vm_only=bool(
                runtime.membrane.membrane.supports_stateless_vm_only_fast_path()
            ),
            Vm0_mV=runtime.membrane.Vm0_mV,
            gates0=runtime.membrane.gates0,
            state0=runtime.membrane.state0,
            area_cm2=runtime.cable.area_cm2,
            Cm_abs=extracellular.Cm_abs,
            Cx_abs=extracellular.Cx_abs,
            Gx_abs=extracellular.Gx_abs,
            Gax_i=extracellular.Gax_i,
            left_i=extracellular.left_i,
            right_i=extracellular.right_i,
            I_background=runtime.membrane.background_current,
            intracellular_current_density_mid=runtime.stimulation.intracellular_current_density_mid,
            extracellular_potential_mid_mV=runtime.stimulation.extracellular_potential_mid_mV,
            dt_ms=jnp.asarray(runtime.grid.dt_ms, dtype=dtype),
            vext_scale=jnp.asarray(config.vext_scale, dtype=dtype),
            capacitance_floor_fraction=jnp.asarray(
                config.capacitance_floor_fraction,
                dtype=dtype,
            ),
            conductance_floor_fraction=jnp.asarray(
                config.conductance_floor_fraction,
                dtype=dtype,
            ),
        )
        vm_np = np.asarray(vm)
        indices = _record_indices(vm_np.shape[1], recording=recording, probe_count=probe_count)
        if indices is not None:
            vm_np = vm_np[:, indices]
        rows.append(
            {
                "index": index,
                "instance": instance,
                "vm": vm_np,
                "record_indices": indices,
                "diagnostics": {
                    "dispatch_method": "pseudo_double_series",
                    "dispatch_group_id": "pseudo_double_series",
                    "dispatch_group_size": 1,
                    "dispatch_batch_kind": "validation_custom",
                    "dispatch_geometry_shared": False,
                    "dispatch_has_padding": False,
                },
            }
        )
        if t_vec is None:
            t_vec = np.asarray(runtime.grid.t_vec_ms)

    if t_vec is None:
        raise ValueError("at least one instance is required.")
    cohort = CohortResult(
        input_indices=tuple(row["index"] for row in rows),
        axons=tuple(row["instance"].axon for row in rows),
        simulations=tuple(row["instance"] for row in rows),
        Vm=np.stack([row["vm"] for row in rows], axis=0),
        t=t_vec,
        diagnostics=tuple(row["diagnostics"] for row in rows),
        record_indices=tuple(row["record_indices"] for row in rows),
        recording=_recording_policy_for_result(recording, probe_count=probe_count),
    )
    return AxonSimulationResult((cohort,), size=len(rows), recording=cohort.recording)


def _recording_policy_for_result(recording: str, *, probe_count: int) -> axs.Recording:
    if recording == "full":
        return axs.Recording.voltage()
    if recording == "center":
        return axs.Recording.center(axs.signals.Vm)
    if recording == "probes":
        return axs.Recording.probes(axs.signals.Vm, count=probe_count)
    raise ValueError("recording must be 'full', 'center', or 'probes'.")


__all__ = [
    "PseudoDoubleSeriesConfig",
    "run_series_population",
]
