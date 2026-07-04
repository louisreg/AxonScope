"""JAX execution contract for backend-neutral membrane programs."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from axonscope.backends.jax.membrane_backend import MembraneStateSpec, MembraneStepPlan
from axonscope.backends.jax.rate_tables import RateTable, make_rate_table_config
from axonscope.model_ir import ModelIR, MembraneProgram, membrane_program_from_model_ir
from axonscope.model_ir.interpreter import NumpyModelInterpreter
from axonscope.solvers.rate_tables import RateTableConfig
from axonscope.utils.settings import dtype

from .model_ir_lowering import JaxModelIRLowering


class JaxMembraneProgram:
    """JAX-lowered executable membrane program.

    This is the backend contract consumed by JAX membrane backends. It
    intentionally does not subclass the legacy compiled-membrane base class.
    """

    def __init__(
        self,
        program: MembraneProgram,
        *,
        dtype_local: jnp.dtype | None = None,
        generated_module: Any | None = None,
    ) -> None:
        self.program = program
        self.model_ir = program.model
        self.dtype = normalize_jax_dtype(dtype_local)
        self.lowering = JaxModelIRLowering(
            self.model_ir,
            dtype=self.dtype,
            generated_module=generated_module,
        )
        self.q10 = dtype(self._representative_q10())
        self._rate_table: RateTable | None = None

    @classmethod
    def from_model_ir(
        cls,
        model_ir: ModelIR,
        *,
        dtype_local: jnp.dtype | None = None,
        generated_module: Any | None = None,
    ) -> "JaxMembraneProgram":
        return cls(
            membrane_program_from_model_ir(model_ir),
            dtype_local=dtype_local,
            generated_module=generated_module,
        )

    @property
    def uses_generated_model_step(self) -> bool:
        return self.lowering.generated_model_step_available

    def static_signature(self) -> tuple[Any, ...]:
        rate_table = self.rate_table_config
        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            str(self.dtype),
            self.program.structural_hash,
            self.program.parameterized_hash,
            self.program.final_gate_update_mode,
            None if rate_table is None else repr(rate_table),
        )

    @property
    def source_provenance(self) -> dict[str, Any]:
        return dict(self.program.source_provenance)

    @property
    def codegen_cache(self) -> dict[str, Any]:
        return dict(self.program.codegen_cache)

    @property
    def g_bar(self) -> jnp.ndarray:
        fallback = None
        values = []
        for index, name in enumerate(self.program.conductance_parameter_names):
            if name is not None and name in self.lowering.parameters:
                values.append(self.lowering.parameters[name])
                continue
            if fallback is None:
                fallback = NumpyModelInterpreter(
                    self.model_ir,
                    dtype=np.float32,
                ).conductances(
                    np.zeros((1, len(self.program.gate_state_names)), dtype=np.float32)
                )[0]
            values.append(jnp.asarray(fallback[index], dtype=self.dtype))
        if not values:
            return jnp.zeros((0,), dtype=self.dtype)
        return jnp.stack(values).astype(self.dtype)

    @property
    def E_rev(self) -> jnp.ndarray:
        return self.lowering.reversal_values()

    def exact_rate_constants(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.lowering.rate_constants(V_mV)

    def rate_constants(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._rate_table is None:
            return self.exact_rate_constants(V_mV)
        return self._rate_table.interpolate(V_mV, dtype_local=self.dtype)

    def gating_inf_tau(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha, beta = self.rate_constants(V_mV)
        sum_ab = jnp.maximum(alpha + beta, dtype(1e-12))
        g_inf = alpha / sum_ab
        tau = dtype(1.0) / (self.q10 * sum_ab)
        return g_inf, tau

    def enable_rate_table(
        self,
        *,
        config: RateTableConfig | None = None,
        v_min_mV: float = -120.0,
        v_max_mV: float = 80.0,
        step_mV: float = 0.05,
        clamp: bool = True,
    ) -> "JaxMembraneProgram":
        resolved = make_rate_table_config(
            config,
            v_min_mV=v_min_mV,
            v_max_mV=v_max_mV,
            step_mV=step_mV,
            clamp=clamp,
        )
        self._rate_table = RateTable.build(
            resolved,
            dtype_local=self.dtype,
            exact_rate_constants=self.exact_rate_constants,
        )
        return self

    def disable_rate_table(self) -> "JaxMembraneProgram":
        self._rate_table = None
        return self

    @property
    def has_rate_table(self) -> bool:
        return self._rate_table is not None

    @property
    def rate_table_config(self) -> RateTableConfig | None:
        return None if self._rate_table is None else self._rate_table.config

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        return self.lowering.init_gates(V0_mV)

    def cn_gate_update(self, g_prev: jnp.ndarray, V_mV: jnp.ndarray, dt: float) -> jnp.ndarray:
        gates = jnp.asarray(g_prev, dtype=self.dtype)
        if gates.shape[-1] == 0:
            return gates
        alpha, beta = self.rate_constants(V_mV)
        q10 = self.lowering.q10_factors(V_mV)
        alpha = q10 * alpha
        beta = q10 * beta
        sum_ab = jnp.maximum(alpha + beta, jnp.asarray(1e-12, dtype=self.dtype))
        dt_local = jnp.asarray(dt, dtype=self.dtype)
        if all(gate.update.value == "crank_nicolson" for gate in self.model_ir.gates):
            denom = jnp.maximum(
                1.0 / dt_local + 0.5 * sum_ab,
                jnp.asarray(1e-12, dtype=self.dtype),
            )
            return alpha / denom + ((1.0 / dt_local) - 0.5 * sum_ab) / denom * gates
        g_inf = alpha / sum_ab
        tau = jnp.asarray(1.0, dtype=self.dtype) / sum_ab
        return g_inf - (g_inf - gates) * jnp.exp(-dt_local / tau)

    def final_gate_update(
        self,
        gates_prev: jnp.ndarray,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        dt: float,
        gates_predictor: jnp.ndarray,
    ) -> jnp.ndarray:
        _ = V_mV_prev
        if self.program.final_gate_update_mode == "post_solve_voltage":
            return self.cn_gate_update(g_prev=gates_prev, V_mV=V_mV_new, dt=dt)
        return gates_predictor

    def supports_stateless_vm_only_fast_path(self) -> bool:
        return (
            self.membrane_state_specs() == ()
            and self.program.final_gate_update_mode != "post_solve_voltage"
            and self.model_ir.step_program is None
        )

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        overrides = {
            name: g_bar[index]
            for index, name in enumerate(self.program.conductance_parameter_names)
            if name is not None and index < int(g_bar.shape[0])
        }
        return self.lowering.conductances(gates, parameters=overrides)

    def conductances(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        return self.lowering.conductances(gates, state=state)

    def currents(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        return self.lowering.currents(V_mV, gates, state=state)

    def total_conductance(self, gates: jnp.ndarray) -> jnp.ndarray:
        Gm, _ = self.membrane_conductance_terms(gates)
        return Gm

    def membrane_conductance_terms(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.lowering.membrane_conductance_terms(gates, state=state)

    def gate_names(self) -> tuple[str, ...]:
        return self.program.gate_names

    def conductance_names(self) -> tuple[str, ...]:
        return self.program.conductance_names

    def current_names(self) -> tuple[str, ...]:
        return self.program.current_names

    def membrane_state_specs(self) -> tuple[MembraneStateSpec, ...]:
        return tuple(
            MembraneStateSpec(name)
            for name in self.program.membrane_state_display_names
        )

    def membrane_state_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.membrane_state_specs())

    def membrane_state_trace_matrix(self, state: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        if not state:
            return jnp.zeros((0, 0), dtype=dtype)
        return jnp.stack(state, axis=1)

    def init_membrane_state(
        self,
        Nx: int,
        dtype_local: jnp.dtype,
        V0_mV: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        V0 = jnp.asarray(V0_mV, dtype=self.dtype)
        if V0.ndim == 0:
            V0 = jnp.full((int(Nx),), V0, dtype=self.dtype)
        return tuple(
            jnp.asarray(value, dtype=dtype_local)
            for value in self.lowering.init_membrane_state(V0)
        )

    def conductance_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        return aggregate_columns(
            self.conductances(gates, state),
            self.program.conductance_groups,
        )

    def gate_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        base = jnp.asarray(gates, dtype=self.dtype)
        if not self.program.gate_trace_observable_names:
            return base
        extras = [
            self.lowering.observable_matrix(name, base, state=state)
            for name in self.program.gate_trace_observable_names
        ]
        return jnp.concatenate([base, jnp.stack(extras, axis=1)], axis=1)

    def ionic_current_trace_matrix(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        raw_currents = self.lowering.current_matrix(V_mV, gates, state=state)
        return aggregate_columns(raw_currents, self.program.current_groups)

    def prepare_membrane_step(
        self,
        V_mV: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state: tuple[jnp.ndarray, ...],
        dt: float,
        I_ion: jnp.ndarray,
        I_background: jnp.ndarray,
    ) -> MembraneStepPlan:
        (
            prepared_state,
            linearization_gates,
            total_outward_current,
            explicit_outward_current,
            correction_current,
        ) = self.lowering.prepare_membrane_step(
            V_mV,
            gates_prev,
            gates_new,
            state,
            dt,
            I_ion,
            I_background,
        )
        return MembraneStepPlan(
            state=prepared_state,
            linearization_gates=linearization_gates,
            total_outward_current=total_outward_current,
            explicit_outward_current=explicit_outward_current,
            correction_current=correction_current,
        )

    def finalize_membrane_step(
        self,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state_prev: tuple[jnp.ndarray, ...],
        step_plan: MembraneStepPlan,
        dt: float,
    ) -> tuple[jnp.ndarray, ...]:
        return self.lowering.finalize_membrane_step(
            V_mV_prev,
            V_mV_new,
            gates_prev,
            gates_new,
            state_prev,
            step_plan.state,
            dt,
        )

    def diagnostic_names(self) -> tuple[str, ...]:
        return self.program.diagnostic_names

    def compute_step_diagnostics(
        self,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state_prev: tuple[jnp.ndarray, ...],
        state_new: tuple[jnp.ndarray, ...],
        step_plan: MembraneStepPlan,
        I_ion: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        return self.lowering.compute_step_diagnostics(
            V_mV_prev,
            V_mV_new,
            gates_prev,
            gates_new,
            state_prev,
            state_new,
            step_plan.state,
            I_ion,
        )

    def I_background(self, Nx: int) -> jnp.ndarray:
        return jnp.zeros((int(Nx),), dtype=self.dtype)

    def _representative_q10(self) -> float:
        if not self.model_ir.gates:
            return 1.0
        q10 = NumpyModelInterpreter(self.model_ir).q10_factors([0.0])
        if q10.shape[1] == 0:
            return 1.0
        first = float(q10[0, 0])
        if not all(abs(float(value) - first) < 1e-12 for value in q10[0]):
            raise ValueError("JaxMembraneProgram requires a common q10 across gates.")
        return first


def is_jax_membrane_program_kind(model: Any, kind: str) -> bool:
    return isinstance(model, JaxMembraneProgram) and model.program.name == kind


def normalize_jax_dtype(dtype_local: Any | None) -> jnp.dtype:
    if dtype_local is None:
        return dtype
    resolved = np.dtype(dtype_local)
    if resolved == np.dtype("float64"):
        return jnp.float64
    return jnp.float32


def aggregate_columns(
    values: jnp.ndarray,
    groups: tuple[tuple[int, ...], ...],
) -> jnp.ndarray:
    cols = []
    for indices in groups:
        if len(indices) == 1:
            cols.append(values[:, indices[0]])
        else:
            cols.append(jnp.sum(values[:, jnp.array(indices)], axis=1))
    if not cols:
        return jnp.zeros((values.shape[0], 0), dtype=values.dtype)
    return jnp.stack(cols, axis=1)


__all__ = [
    "JaxMembraneProgram",
    "aggregate_columns",
    "is_jax_membrane_program_kind",
    "normalize_jax_dtype",
]
