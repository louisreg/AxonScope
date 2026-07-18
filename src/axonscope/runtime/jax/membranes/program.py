"""JAX execution contract for backend-neutral membrane programs."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from axonscope.runtime.jax.membranes.backend import MembraneStateSpec, MembraneStepPlan
from axonscope.model_ir import ModelIR, MembraneProgram, membrane_program_from_model_ir
from axonscope.model_ir.interpreter import NumpyModelInterpreter, parameter_defaults
from axonscope.utils.settings import dtype

from .model_ir_lowering import JaxModelIRLowering
from .generated_lowering import GeneratedJaxMembraneLowering
from .generated_contract import (
    GeneratedMembraneContract,
    load_generated_membrane_contract,
)


class JaxMembraneProgram:
    """JAX-lowered executable membrane program.

    This is the backend contract consumed by JAX membrane backends. It
    intentionally does not subclass the legacy compiled-membrane base class.
    """

    def __init__(
        self,
        program: MembraneProgram | None,
        *,
        dtype_local: jnp.dtype | None = None,
        generated_module: Any | None = None,
        host_module: Any | None = None,
        parameter_overrides: dict[str, Any] | None = None,
        codegen_cache: dict[str, Any] | None = None,
    ) -> None:
        if program is None and generated_module is None:
            raise ValueError("A membrane program or generated module is required.")
        self.program = program
        self.model_ir = None if program is None else program.model
        self.dtype = normalize_jax_dtype(dtype_local)
        self.generated_contract: GeneratedMembraneContract | None = None
        self._host_module = host_module
        self._parameter_values: dict[str, Any] = {}
        if generated_module is not None:
            self.generated_contract = load_generated_membrane_contract(
                generated_module
            )
            if program is not None:
                _validate_generated_contract(self.generated_contract, program)
            model_parameters = (
                parameter_defaults(program.model)
                if parameter_overrides is None and program is not None
                else parameter_overrides
            )
            parameter_values = self.generated_contract.parameter_values(
                model_parameters
            )
            self._parameter_values = dict(parameter_values)
            self.lowering = GeneratedJaxMembraneLowering(
                generated_module,
                self.generated_contract,
                dtype=self.dtype,
                parameter_values=parameter_values,
            )
        else:
            assert program is not None
            self._parameter_values = parameter_defaults(program.model)
            self.lowering = JaxModelIRLowering(program.model, dtype=self.dtype)
        self._codegen_cache = dict(codegen_cache or {})
        self.q10 = dtype(self._representative_q10())
        self._static_signature_cache: tuple[Any, ...] | None = None
        self._g_bar_cache: jnp.ndarray | None = None
        self._e_rev_cache: jnp.ndarray | None = None
        self._membrane_state_specs_cache: tuple[MembraneStateSpec, ...] | None = None

    @classmethod
    def from_model_ir(
        cls,
        model_ir: ModelIR,
        *,
        dtype_local: jnp.dtype | None = None,
        generated_module: Any | None = None,
        host_module: Any | None = None,
    ) -> "JaxMembraneProgram":
        return cls(
            membrane_program_from_model_ir(model_ir),
            dtype_local=dtype_local,
            generated_module=generated_module,
            host_module=host_module,
        )

    @classmethod
    def from_generated_module(
        cls,
        generated_module: Any,
        *,
        parameter_overrides: dict[str, Any],
        dtype_local: jnp.dtype | None = None,
        host_module: Any | None = None,
        codegen_cache: dict[str, Any] | None = None,
    ) -> "JaxMembraneProgram":
        """Build directly from an autonomous generated runtime artifact."""

        return cls(
            None,
            dtype_local=dtype_local,
            generated_module=generated_module,
            host_module=host_module,
            parameter_overrides=parameter_overrides,
            codegen_cache=codegen_cache,
        )

    @property
    def uses_generated_model_step(self) -> bool:
        return self.lowering.generated_model_step_available

    def static_signature(self) -> tuple[Any, ...]:
        cached = self._static_signature_cache
        if cached is not None:
            return cached
        signature = (
            self.__class__.__module__,
            self.__class__.__qualname__,
            str(self.dtype),
            self._structural_hash,
            self._parameterized_hash,
            tuple(
                (name, _parameter_signature_value(value))
                for name, value in sorted(self._parameter_values.items())
            ),
            self._final_gate_update_mode,
        )
        self._static_signature_cache = signature
        return signature

    @property
    def source_provenance(self) -> dict[str, Any]:
        if self.generated_contract is not None:
            return dict(self.generated_contract.source_provenance)
        assert self.program is not None
        return dict(self.program.source_provenance)

    @property
    def codegen_cache(self) -> dict[str, Any]:
        if self._codegen_cache:
            return dict(self._codegen_cache)
        assert self.program is not None
        return dict(self.program.codegen_cache)

    @property
    def g_bar(self) -> jnp.ndarray:
        cached = self._g_bar_cache
        if cached is not None:
            return cached
        fallback = None
        values = []
        for index, name in enumerate(self._conductance_parameter_names):
            if name is not None and name in self.lowering.parameters:
                values.append(self.lowering.parameters[name])
                continue
            if fallback is None:
                if self.model_ir is None:
                    gates = jnp.zeros(
                        (1, len(self.generated_contract.gate_state_names)),
                        dtype=self.dtype,
                    )
                    state = self.lowering.init_membrane_state(
                        jnp.zeros((1,), dtype=self.dtype)
                    )
                    fallback = np.asarray(
                        self.lowering.conductances(gates, state=state)[0]
                    )
                else:
                    fallback = NumpyModelInterpreter(
                        self.model_ir,
                        dtype=np.float32,
                    ).conductances(
                        np.zeros((1, len(self.program.gate_state_names)), dtype=np.float32)
                    )[0]
            values.append(jnp.asarray(fallback[index], dtype=self.dtype))
        if not values:
            out = jnp.zeros((0,), dtype=self.dtype)
        else:
            out = jnp.stack(values).astype(self.dtype)
        self._g_bar_cache = out
        return out

    @property
    def E_rev(self) -> jnp.ndarray:
        cached = self._e_rev_cache
        if cached is not None:
            return cached
        out = self.lowering.reversal_values()
        self._e_rev_cache = out
        return out

    def exact_rate_constants(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.lowering.rate_constants(V_mV)

    def rate_constants(self, V_mV: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.exact_rate_constants(V_mV)

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def init_gates(self, V0_mV: jnp.ndarray) -> jnp.ndarray:
        return self.lowering.init_gates(V0_mV)

    def init_gates_host(
        self,
        V0_mV: np.ndarray,
        *,
        dtype_local: np.dtype,
    ) -> np.ndarray:
        """Initialize gates through the generated NumPy companion artifact."""

        if self.generated_contract is None or self._host_module is None:
            if self.model_ir is None:
                raise ValueError("Generated NumPy membrane module is unavailable.")
            return NumpyModelInterpreter(
                self.model_ir,
                dtype=dtype_local,
            ).init_gates(V0_mV)
        V = np.atleast_1d(np.asarray(V0_mV, dtype=dtype_local))
        if not self.generated_contract.gate_state_names:
            return np.zeros((V.shape[0], 0), dtype=dtype_local)
        spec = self.generated_contract.function("gate_terms")
        env = {
            name: np.asarray(value, dtype=dtype_local)
            for name, value in self._parameter_values.items()
        }
        env["Vm"] = V
        missing = tuple(name for name in spec.args if name not in env)
        if missing:
            raise ValueError(
                f"Generated host gate arguments are unavailable: {missing!r}."
            )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            raw = self._host_module.gate_terms(*(env[name] for name in spec.args))
        values = raw if isinstance(raw, tuple) else (raw,)
        if len(values) != len(spec.outputs):
            raise ValueError("Generated host gate output count is inconsistent.")
        alpha = np.stack(
            [np.broadcast_to(value, V.shape) for value in values[0::3]], axis=1
        ).astype(dtype_local)
        beta = np.stack(
            [np.broadcast_to(value, V.shape) for value in values[1::3]], axis=1
        ).astype(dtype_local)
        return alpha / np.maximum(alpha + beta, np.asarray(1e-12, dtype=dtype_local))

    def cn_gate_update(self, g_prev: jnp.ndarray, V_mV: jnp.ndarray, dt: float) -> jnp.ndarray:
        gates = jnp.asarray(g_prev, dtype=self.dtype)
        if gates.shape[-1] == 0:
            return gates
        alpha, beta, q10 = self.lowering.gate_terms(V_mV)
        alpha = q10 * alpha
        beta = q10 * beta
        sum_ab = jnp.maximum(alpha + beta, jnp.asarray(1e-12, dtype=self.dtype))
        dt_local = jnp.asarray(dt, dtype=self.dtype)
        if all(mode == "crank_nicolson" for mode in self._gate_update_modes):
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
        if self._final_gate_update_mode == "post_solve_voltage":
            return self.cn_gate_update(g_prev=gates_prev, V_mV=V_mV_new, dt=dt)
        return gates_predictor

    def supports_stateless_vm_only_fast_path(self) -> bool:
        return (
            self.membrane_state_specs() == ()
            and self._final_gate_update_mode != "post_solve_voltage"
            and not self._has_step_program
        )

    def g_funcs(self, gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        overrides = {
            name: g_bar[index]
            for index, name in enumerate(self._conductance_parameter_names)
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
        if self.generated_contract is not None:
            return self.generated_contract.gate_names
        return self.program.gate_names

    def conductance_names(self) -> tuple[str, ...]:
        if self.generated_contract is not None:
            return self.generated_contract.conductance_names
        return self.program.conductance_names

    def current_names(self) -> tuple[str, ...]:
        if self.generated_contract is not None:
            return self.generated_contract.current_names
        return self.program.current_names

    def membrane_state_specs(self) -> tuple[MembraneStateSpec, ...]:
        cached = self._membrane_state_specs_cache
        if cached is not None:
            return cached
        specs = tuple(
            MembraneStateSpec(name)
            for name in self._membrane_state_display_names
        )
        self._membrane_state_specs_cache = specs
        return specs

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
            self._conductance_groups,
        )

    def gate_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        base = jnp.asarray(gates, dtype=self.dtype)
        if not self._gate_trace_observable_names:
            return base
        extras = [
            self.lowering.observable_matrix(name, base, state=state)
            for name in self._gate_trace_observable_names
        ]
        return jnp.concatenate([base, jnp.stack(extras, axis=1)], axis=1)

    def ionic_current_trace_matrix(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        raw_currents = self.lowering.current_matrix(V_mV, gates, state=state)
        return aggregate_columns(raw_currents, self._current_groups)

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
        if self.generated_contract is not None:
            return self.generated_contract.diagnostic_names
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
        if self.generated_contract is not None:
            if not self.generated_contract.gate_state_names:
                return 1.0
            if self._host_module is not None:
                spec = self.generated_contract.function("gate_terms")
                np_dtype = np.dtype(self.dtype)
                env = {
                    name: np.asarray(value, dtype=np_dtype)
                    for name, value in self._parameter_values.items()
                }
                env["Vm"] = np.asarray([0.0], dtype=np_dtype)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    raw = self._host_module.gate_terms(
                        *(env[name] for name in spec.args)
                    )
                values = raw if isinstance(raw, tuple) else (raw,)
                q10 = np.stack(
                    [np.broadcast_to(value, (1,)) for value in values[2::3]],
                    axis=1,
                )
            else:
                q10 = np.asarray(
                    self.lowering.q10_factors(
                        jnp.asarray([0.0], dtype=self.dtype)
                    )
                )
        elif self.model_ir is None or not self.model_ir.gates:
            return 1.0
        else:
            q10 = NumpyModelInterpreter(self.model_ir).q10_factors([0.0])
        if q10.shape[1] == 0:
            return 1.0
        first = float(q10[0, 0])
        if not all(abs(float(value) - first) < 1e-12 for value in q10[0]):
            raise ValueError("JaxMembraneProgram requires a common q10 across gates.")
        return first

    @property
    def _conductance_parameter_names(self) -> tuple[str | None, ...]:
        if self.generated_contract is not None:
            return self.generated_contract.conductance_parameter_names
        return self.program.conductance_parameter_names

    @property
    def _gate_update_modes(self) -> tuple[str, ...]:
        if self.generated_contract is not None:
            return self.generated_contract.gate_update_modes
        return tuple(gate.update.value for gate in self.model_ir.gates)

    @property
    def _final_gate_update_mode(self) -> str:
        if self.generated_contract is not None:
            return self.generated_contract.final_gate_update_mode
        return self.program.final_gate_update_mode

    @property
    def _has_step_program(self) -> bool:
        if self.generated_contract is not None:
            return self.generated_contract.has_step_program
        return self.model_ir.step_program is not None

    @property
    def _membrane_state_display_names(self) -> tuple[str, ...]:
        if self.generated_contract is not None:
            return self.generated_contract.membrane_state_display_names
        return self.program.membrane_state_display_names

    @property
    def _structural_hash(self) -> str:
        if self.generated_contract is not None:
            return self.generated_contract.structural_hash
        assert self.program is not None
        return self.program.structural_hash

    @property
    def _parameterized_hash(self) -> str:
        if self.generated_contract is not None:
            return self.generated_contract.parameterized_hash
        assert self.program is not None
        return self.program.parameterized_hash

    @property
    def _conductance_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.generated_contract is not None:
            return self.generated_contract.conductance_groups
        assert self.program is not None
        return self.program.conductance_groups

    @property
    def _current_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.generated_contract is not None:
            return self.generated_contract.current_groups
        assert self.program is not None
        return self.program.current_groups

    @property
    def _gate_trace_observable_names(self) -> tuple[str, ...]:
        if self.generated_contract is not None:
            return self.generated_contract.gate_trace_observable_names
        assert self.program is not None
        return self.program.gate_trace_observable_names


def is_jax_membrane_program_kind(model: Any, kind: str) -> bool:
    if not isinstance(model, JaxMembraneProgram):
        return False
    if model.generated_contract is not None:
        return model.generated_contract.model_name == kind
    return model.program is not None and model.program.name == kind


def _validate_generated_contract(
    contract: GeneratedMembraneContract,
    program: MembraneProgram,
) -> None:
    expected = {
        "model_name": program.name,
        "structural_hash": program.structural_hash,
        "gate_state_names": program.gate_state_names,
        "membrane_state_names": program.membrane_state_names,
        "gate_names": program.gate_names,
        "current_names": program.current_names,
        "current_groups": program.current_groups,
        "conductance_names": program.conductance_names,
        "conductance_groups": program.conductance_groups,
        "conductance_parameter_names": program.conductance_parameter_names,
        "diagnostic_names": program.diagnostic_names,
        "final_gate_update_mode": program.final_gate_update_mode,
    }
    mismatches = tuple(
        name
        for name, value in expected.items()
        if getattr(contract, name) != value
    )
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(
            "Generated JAX membrane contract does not match compiled model: "
            f"{names}."
        )


def normalize_jax_dtype(dtype_local: Any | None) -> jnp.dtype:
    if dtype_local is None:
        return dtype
    resolved = np.dtype(dtype_local)
    if resolved == np.dtype("float64"):
        return jnp.float64
    return jnp.float32


def _parameter_signature_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return (str(array.dtype), tuple(array.shape), array.tobytes())


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
