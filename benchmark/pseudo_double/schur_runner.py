"""Runnable Schur-local pseudo-double validation path."""

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
    DoubleCableBlockCoefficients,
    schur_local_v1,
    tridiagonal_edges_to_jax,
)


@dataclass(frozen=True)
class PseudoDoubleSchurLocalConfig:
    """Experimental diagonal-App Schur-local reduction parameters."""

    vext_scale: float = 1.0
    app_inverse_scale: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "vext_scale": float(self.vext_scale),
            "app_inverse_scale": float(self.app_inverse_scale),
        }


@partial(
    jax.jit,
    static_argnames=("backend", "membrane", "has_driven_extracellular", "stateless_vm_only"),
)
def _run_schur_local_vm_scan(
    *,
    backend,
    membrane,
    has_driven_extracellular: bool,
    stateless_vm_only: bool,
    Vi0_mV,
    Ve0_mV,
    gates0,
    state0,
    area_cm2,
    Cm_abs,
    Cx_abs,
    Gx_abs,
    Gax_e,
    Gax_i,
    left_i,
    right_i,
    left_e,
    right_e,
    I_background,
    intracellular_current_density_mid,
    extracellular_potential_mid_mV,
    extracellular_potential_initial_previous_mV,
    dt_ms,
    vext_scale,
    app_inverse_scale,
):
    cm_over_dt = Cm_abs / dt_ms
    cx_over_dt = Cx_abs / dt_ms
    vext_mid_scaled = extracellular_potential_mid_mV * vext_scale
    vext_previous_mV = jnp.concatenate(
        [
            (extracellular_potential_initial_previous_mV * vext_scale)[None, :],
            vext_mid_scaled[:-1],
        ],
        axis=0,
    )
    extracellular_rhs_drive = (
        (cx_over_dt + Gx_abs)[None, :] * vext_mid_scaled
        - cx_over_dt[None, :] * vext_previous_mV
    )
    intracellular_current_abs_mid = intracellular_current_density_mid * area_cm2[None, :]

    def solve_vi_schur(
        Vi,
        Ve,
        gates_new,
        Iinj_abs,
        I_outward_den,
        I_corr_den,
        extracellular_drive_abs,
    ):
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

        coeffs = DoubleCableBlockCoefficients(
            aii_lower=-Gax_i,
            aii_diag=a00,
            aii_upper=-Gax_i,
            app_lower=-Gax_e,
            app_diag=a11,
            app_upper=-Gax_e,
            aip_diag=a01,
            api_diag=a10,
        )
        reduced = schur_local_v1(
            coeffs,
            rhs0,
            rhs1,
            app_inverse_scale=app_inverse_scale,
        )
        dl, d, du = tridiagonal_edges_to_jax(
            reduced.lower,
            reduced.diag,
            reduced.upper,
        )
        Vi_new = tridiagonal_solve(dl, d, du, reduced.rhs[:, None])[:, 0]
        inv_app = app_inverse_scale / a11
        Ve_new = inv_app * (rhs1 - a10 * Vi_new)
        return Vi_new, Ve_new

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

        Vi_new, Ve_new = solve_vi_schur(
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
            return (Vi_new, Ve_new, gates_pred, *extra), Vm_new

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
        return (Vi_new, Ve_new, gates_new, *state_new), Vm_new

    init_carry = (Vi0_mV, Ve0_mV, gates0, *state0)
    _, vm_trace = jax.lax.scan(
        step,
        init_carry,
        (intracellular_current_abs_mid, extracellular_rhs_drive),
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


def run_schur_local_population(
    instances: Sequence[AxonInstance],
    *,
    duration_ms: float,
    dt_ms: float,
    recording: str,
    probe_count: int,
    config: PseudoDoubleSchurLocalConfig,
) -> AxonSimulationResult:
    """Run one pseudo-double Schur-local validation population."""

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
            raise ValueError("Schur-local pseudo-double requires extracellular arrays.")
        dtype = runtime.membrane.dtype
        nx = int(runtime.membrane.Nx)
        ve0 = jnp.zeros((nx,), dtype=dtype)
        vi0 = runtime.membrane.Vm0_mV + ve0
        vm = _run_schur_local_vm_scan(
            backend=runtime.membrane.backend,
            membrane=runtime.membrane.membrane,
            has_driven_extracellular=runtime.stimulation.has_driven_extracellular,
            stateless_vm_only=bool(
                runtime.membrane.membrane.supports_stateless_vm_only_fast_path()
            ),
            Vi0_mV=vi0,
            Ve0_mV=ve0,
            gates0=runtime.membrane.gates0,
            state0=runtime.membrane.state0,
            area_cm2=runtime.cable.area_cm2,
            Cm_abs=extracellular.Cm_abs,
            Cx_abs=extracellular.Cx_abs,
            Gx_abs=extracellular.Gx_abs,
            Gax_e=extracellular.Gax_e,
            Gax_i=extracellular.Gax_i,
            left_i=extracellular.left_i,
            right_i=extracellular.right_i,
            left_e=extracellular.left_e,
            right_e=extracellular.right_e,
            I_background=runtime.membrane.background_current,
            intracellular_current_density_mid=runtime.stimulation.intracellular_current_density_mid,
            extracellular_potential_mid_mV=runtime.stimulation.extracellular_potential_mid_mV,
            extracellular_potential_initial_previous_mV=(
                runtime.stimulation.extracellular_potential_initial_previous_mV
            ),
            dt_ms=jnp.asarray(runtime.grid.dt_ms, dtype=dtype),
            vext_scale=jnp.asarray(config.vext_scale, dtype=dtype),
            app_inverse_scale=jnp.asarray(config.app_inverse_scale, dtype=dtype),
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
                    "dispatch_method": "pseudo_double_schur_local",
                    "dispatch_group_id": "pseudo_double_schur_local",
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
    "PseudoDoubleSchurLocalConfig",
    "run_schur_local_population",
]
