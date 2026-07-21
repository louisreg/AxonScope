"""JAX lowering for backend-neutral Model IR membrane expressions."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .kinetics import (
    dense_kinetic_matrix,
    solve_kinetic_transitions,
)

from axonscope.model_ir.expressions import BinaryOp, Call, Expression, Literal, Symbol, UnaryOp
from axonscope.model_ir.interpreter import parameter_defaults
from axonscope.model_ir.program import membrane_program_from_model_ir
from axonscope.model_ir.schema import (
    GateUpdateKind,
    KineticInitialization,
    LinearizationGateSource,
    ModelIR,
    StateUpdate,
)
from axonscope.runtime.jax.membranes.generated_contract import (
    GeneratedMembraneContract,
)


def evaluate_expression_jax(expr: Expression, env: dict[str, Any], *, dtype: jnp.dtype) -> Any:
    """Evaluate a Model IR expression with JAX operations."""

    if isinstance(expr, Literal):
        return jnp.asarray(expr.value, dtype=dtype)
    if isinstance(expr, Symbol):
        try:
            return env[expr.name]
        except KeyError as exc:
            raise KeyError(f"Missing Model IR symbol {expr.name!r}.") from exc
    if isinstance(expr, UnaryOp):
        value = evaluate_expression_jax(expr.operand, env, dtype=dtype)
        if expr.op == "neg":
            return -value
    if isinstance(expr, BinaryOp):
        left = evaluate_expression_jax(expr.left, env, dtype=dtype)
        right = evaluate_expression_jax(expr.right, env, dtype=dtype)
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "div":
            return left / right
        if expr.op == "pow":
            return jnp.power(left, right)
        if expr.op == "lt":
            return left < right
        if expr.op == "le":
            return left <= right
        if expr.op == "gt":
            return left > right
        if expr.op == "ge":
            return left >= right
    if isinstance(expr, Call):
        args = [evaluate_expression_jax(arg, env, dtype=dtype) for arg in expr.args]
        return _call_intrinsic(expr.intrinsic, args, dtype=dtype)
    raise TypeError(f"Unsupported Model IR expression {type(expr).__name__}.")


class JaxModelIRLowering:
    """JAX evaluator for the first Model IR membrane-step contract."""

    def __init__(
        self,
        model: ModelIR,
        *,
        dtype: jnp.dtype,
        generated_module: Any | None = None,
        generated_contract: GeneratedMembraneContract | None = None,
        parameter_values: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.dtype = dtype
        resolved_parameters = (
            parameter_defaults(model)
            if parameter_values is None
            else parameter_values
        )
        self.parameters = {
            key: jnp.asarray(value, dtype=dtype)
            for key, value in resolved_parameters.items()
        }
        self.program = membrane_program_from_model_ir(model)
        self.gate_state_names = self.program.gate_state_names
        self.hh_gate_state_names = self.program.hh_gate_state_names
        self.kinetic_state_names = self.program.kinetic_state_names
        self.membrane_states = self.program.membrane_states
        self.membrane_state_names = self.program.membrane_state_names
        self.state_names = self.gate_state_names
        self.generated_module = generated_module
        self.generated_contract = generated_contract
        self.generated_arg_names = _generated_names(generated_module, "ARG_NAMES")
        self.generated_output_names = _generated_names(generated_module, "OUTPUT_NAMES")
        self.generated_output_index = {
            name: index for index, name in enumerate(self.generated_output_names)
        }
        self.generated_gate_arg_names = _generated_names(
            generated_module, "GATE_ARG_NAMES"
        )
        self.generated_gate_output_names = _generated_names(
            generated_module, "GATE_OUTPUT_NAMES"
        )
        self.generated_membrane_arg_names = _generated_names(
            generated_module, "MEMBRANE_ARG_NAMES"
        )
        self.generated_membrane_output_names = _generated_names(
            generated_module, "MEMBRANE_OUTPUT_NAMES"
        )
        self.source_current_output_names = _source_output_names(model, "currents")

    @property
    def generated_model_step_available(self) -> bool:
        return (
            self.generated_module is not None
            and bool(self.generated_arg_names)
            and bool(self.generated_output_names)
        )

    @property
    def generated_gate_terms_available(self) -> bool:
        return (
            self.generated_module is not None
            and callable(getattr(self.generated_module, "gate_terms", None))
            and len(self.generated_gate_output_names) == 3 * len(self.model.gates)
        )

    @property
    def generated_membrane_terms_available(self) -> bool:
        return (
            self.generated_module is not None
            and callable(getattr(self.generated_module, "membrane_terms", None))
            and len(self.generated_membrane_output_names) == 2 * len(self.model.currents)
        )

    def with_parameters(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(self.parameters)
        if overrides:
            params.update({key: jnp.asarray(value, dtype=self.dtype) for key, value in overrides.items()})
        return params

    def rate_constants(
        self,
        V_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha, beta, _ = self.gate_terms(V_mV, parameters=parameters)
        return alpha, beta

    def q10_factors(
        self,
        V_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        _, _, q10 = self.gate_terms(V_mV, parameters=parameters)
        return q10

    def gate_terms(
        self,
        V_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        V = jnp.atleast_1d(jnp.asarray(V_mV, dtype=self.dtype))
        env = self._base_env(V, parameters=parameters)
        generated = self._generated_term_outputs(
            function_name="gate_terms",
            arg_names=self.generated_gate_arg_names,
            output_names=self.generated_gate_output_names,
            env=env,
            node_count=V.shape[0],
        )
        if generated is not None:
            alpha = list(generated[0::3])
            beta = list(generated[1::3])
            factors = list(generated[2::3])
            return self._combined_gate_terms(alpha, beta, factors, V.shape[0])

        alpha = []
        beta = []
        factors = []
        for gate in self.model.gates:
            alpha.append(
                _as_node_vector(
                    evaluate_expression_jax(gate.alpha, env, dtype=self.dtype),
                    V.shape[0],
                    self.dtype,
                )
            )
            beta.append(
                _as_node_vector(
                    evaluate_expression_jax(gate.beta, env, dtype=self.dtype),
                    V.shape[0],
                    self.dtype,
                )
            )
            factors.append(
                jnp.ones((V.shape[0],), dtype=self.dtype)
                if gate.q10 is None
                else _as_node_vector(
                    evaluate_expression_jax(gate.q10, env, dtype=self.dtype),
                    V.shape[0],
                    self.dtype,
                )
            )
        return self._combined_gate_terms(alpha, beta, factors, V.shape[0])

    def init_gates(
        self,
        V0_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        V = jnp.atleast_1d(jnp.asarray(V0_mV, dtype=self.dtype))
        if not self.gate_state_names:
            return jnp.zeros((V.shape[0], 0), dtype=self.dtype)
        alpha, beta = self.rate_constants(V, parameters=parameters)
        hh_count = len(self.hh_gate_state_names)
        denom = jnp.maximum(
            alpha[:, :hh_count] + beta[:, :hh_count],
            jnp.asarray(1e-12, dtype=self.dtype),
        )
        hh = alpha[:, :hh_count] / denom
        kinetics = self._init_kinetic_states(V, parameters=parameters)
        return jnp.concatenate((hh, kinetics), axis=1)

    def init_membrane_state(
        self,
        V0_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        V = jnp.atleast_1d(jnp.asarray(V0_mV, dtype=self.dtype))
        env = self._base_env(V, parameters=parameters)
        generated = self._generated_runtime_outputs(
            "init_state",
            env=env,
            node_count=V.shape[0],
        )
        if generated is not None:
            return tuple(generated[name] for name in self.membrane_state_names)
        values: list[jnp.ndarray] = []
        for state in self.membrane_states:
            if state.initial is None:
                value = jnp.zeros((V.shape[0],), dtype=self.dtype)
            else:
                value = _as_node_vector(
                    evaluate_expression_jax(state.initial, env, dtype=self.dtype),
                    V.shape[0],
                    self.dtype,
                )
            values.append(value)
            env[state.name] = value
        return tuple(values)

    def gate_update(
        self,
        gates_prev: Any,
        V_mV: Any,
        dt_ms: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        gates = jnp.asarray(gates_prev, dtype=self.dtype)
        V = jnp.atleast_1d(jnp.asarray(V_mV, dtype=self.dtype))
        if gates.shape[-1] == 0:
            return gates
        hh_count = len(self.hh_gate_state_names)
        alpha, beta, q10 = self.gate_terms(V, parameters=parameters)
        alpha = alpha[:, :hh_count]
        beta = beta[:, :hh_count]
        q10 = q10[:, :hh_count]
        alpha = q10 * alpha
        beta = q10 * beta
        sum_ab = jnp.maximum(alpha + beta, jnp.asarray(1e-12, dtype=self.dtype))
        g_inf = alpha / sum_ab
        dt = jnp.asarray(dt_ms, dtype=self.dtype)
        hh_previous = gates[:, :hh_count]
        if all(gate.update is GateUpdateKind.CRANK_NICOLSON for gate in self.model.gates):
            denom = jnp.maximum(1.0 / dt + 0.5 * sum_ab, jnp.asarray(1e-12, dtype=self.dtype))
            hh = alpha / denom + ((1.0 / dt) - 0.5 * sum_ab) / denom * hh_previous
        else:
            tau = jnp.asarray(1.0, dtype=self.dtype) / sum_ab
            hh = g_inf - (g_inf - hh_previous) * jnp.exp(-dt / tau)
        kinetics = self._update_kinetic_states(
            gates[:, hh_count:],
            V,
            dt,
            parameters=parameters,
        )
        return jnp.concatenate((hh, kinetics), axis=1)

    def _combined_gate_terms(
        self,
        alpha: list[jnp.ndarray],
        beta: list[jnp.ndarray],
        factors: list[jnp.ndarray],
        node_count: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        alpha.extend(jnp.zeros((node_count,), dtype=self.dtype) for _ in self.kinetic_state_names)
        beta.extend(jnp.zeros((node_count,), dtype=self.dtype) for _ in self.kinetic_state_names)
        factors.extend(jnp.ones((node_count,), dtype=self.dtype) for _ in self.kinetic_state_names)
        return (
            _stack_columns(alpha, node_count, self.dtype),
            _stack_columns(beta, node_count, self.dtype),
            _stack_columns(factors, node_count, self.dtype),
        )

    def _init_kinetic_states(
        self,
        V: jnp.ndarray,
        *,
        parameters: dict[str, Any] | None,
    ) -> jnp.ndarray:
        env = self._base_env(V, parameters=parameters)
        values: list[jnp.ndarray] = []
        states = {state.name: state for state in self.model.states}
        for block in self.model.kinetics:
            if block.initialization is KineticInitialization.STATIONARY:
                matrix = self._kinetic_matrix(block, env, V.shape[0])
                system = matrix.at[:, -1, :].set(jnp.asarray(1.0, dtype=self.dtype))
                rhs = jnp.zeros((V.shape[0], len(block.states)), dtype=self.dtype)
                rhs = rhs.at[:, -1].set(jnp.asarray(1.0, dtype=self.dtype))
                block_values = jnp.linalg.solve(system, rhs[..., None])[..., 0]
            else:
                block_values = _stack_columns(
                    [
                        jnp.zeros((V.shape[0],), dtype=self.dtype)
                        if states[name].initial is None
                        else _as_node_vector(
                            evaluate_expression_jax(states[name].initial, env, dtype=self.dtype),
                            V.shape[0],
                            self.dtype,
                        )
                        for name in block.states
                    ],
                    V.shape[0],
                    self.dtype,
                )
            if block.conserve_probability:
                block_values = _normalize_probabilities_jax(block_values, self.dtype)
            values.extend(block_values[:, index] for index in range(block_values.shape[1]))
        return _stack_columns(values, V.shape[0], self.dtype)

    def _update_kinetic_states(
        self,
        previous: jnp.ndarray,
        V: jnp.ndarray,
        dt: jnp.ndarray,
        *,
        parameters: dict[str, Any] | None,
    ) -> jnp.ndarray:
        env = self._base_env(V, parameters=parameters)
        values: list[jnp.ndarray] = []
        offset = 0
        for block in self.model.kinetics:
            width = len(block.states)
            block_values = solve_kinetic_transitions(
                width=width,
                transitions=self._kinetic_transitions(block, env, V.shape[0]),
                previous=previous[:, offset : offset + width],
                dt=dt,
                node_count=V.shape[0],
                dtype=self.dtype,
                conserve_probability=block.conserve_probability,
            )
            values.extend(block_values[:, index] for index in range(width))
            offset += width
        return _stack_columns(values, V.shape[0], self.dtype)

    def _kinetic_matrix(self, block: Any, env: dict[str, Any], node_count: int) -> jnp.ndarray:
        width = len(block.states)
        return dense_kinetic_matrix(
            width=width,
            transitions=self._kinetic_transitions(block, env, node_count),
            node_count=node_count,
            dtype=self.dtype,
        )

    def _kinetic_transitions(
        self,
        block: Any,
        env: dict[str, Any],
        node_count: int,
    ) -> tuple[tuple[int, int, jnp.ndarray], ...]:
        indices = {name: index for index, name in enumerate(block.states)}
        transitions: list[tuple[int, int, jnp.ndarray]] = []
        for transition in block.transitions:
            source = indices[transition.source]
            target = indices[transition.target]
            rate = _as_node_vector(
                evaluate_expression_jax(transition.rate, env, dtype=self.dtype),
                node_count,
                self.dtype,
            )
            transitions.append((source, target, rate))
        return tuple(transitions)

    def conductances(
        self,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        gates_arr = jnp.asarray(gates, dtype=self.dtype)
        node_count = int(gates_arr.shape[0]) if gates_arr.ndim else 1
        env = self._state_env(gates_arr, state=state, parameters=parameters)
        generated = self._generated_term_outputs(
            function_name="membrane_terms",
            arg_names=self.generated_membrane_arg_names,
            output_names=self.generated_membrane_output_names,
            env=env,
            node_count=node_count,
        )
        if generated is not None:
            return _stack_columns(list(generated[0::2]), node_count, self.dtype)
        cols = [
            _as_node_vector(
                evaluate_expression_jax(current.conductance, env, dtype=self.dtype),
                node_count,
                self.dtype,
            )
            for current in self.model.currents
        ]
        return _stack_columns(cols, node_count, self.dtype)

    def current_matrix(
        self,
        V_mV: Any,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        V = jnp.atleast_1d(jnp.asarray(V_mV, dtype=self.dtype))
        gates_arr = jnp.asarray(gates, dtype=self.dtype)
        generated = self._generated_output_matrix(
            self.source_current_output_names,
            V,
            gates_arr,
            state=state,
            parameters=parameters,
        )
        if generated is not None:
            return generated
        env = self._state_env(gates_arr, state=state, V=V, parameters=parameters)
        cols = [
            _as_node_vector(
                evaluate_expression_jax(current.current, env, dtype=self.dtype),
                V.shape[0],
                self.dtype,
            )
            for current in self.model.currents
        ]
        return _stack_columns(cols, V.shape[0], self.dtype)

    def currents(
        self,
        V_mV: Any,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        return jnp.sum(
            self.current_matrix(V_mV, gates, state=state, parameters=parameters),
            axis=1,
        )

    def membrane_conductance_terms(
        self,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gates_arr = jnp.asarray(gates, dtype=self.dtype)
        node_count = int(gates_arr.shape[0]) if gates_arr.ndim else 1
        env = self._state_env(gates_arr, state=state, parameters=parameters)
        generated = self._generated_term_outputs(
            function_name="membrane_terms",
            arg_names=self.generated_membrane_arg_names,
            output_names=self.generated_membrane_output_names,
            env=env,
            node_count=node_count,
        )
        if generated is not None:
            g = _stack_columns(list(generated[0::2]), node_count, self.dtype)
            e = _stack_columns(list(generated[1::2]), node_count, self.dtype)
            return jnp.sum(g, axis=1), jnp.sum(g * e, axis=1)
        conductances = []
        reversals = []
        for current in self.model.currents:
            conductances.append(
                _as_node_vector(
                    evaluate_expression_jax(current.conductance, env, dtype=self.dtype),
                    node_count,
                    self.dtype,
                )
            )
            reversals.append(
                _as_node_vector(
                    evaluate_expression_jax(current.reversal, env, dtype=self.dtype),
                    node_count,
                    self.dtype,
                )
            )
        g = _stack_columns(conductances, node_count, self.dtype)
        e = _stack_columns(reversals, node_count, self.dtype)
        return jnp.sum(g, axis=1), jnp.sum(g * e, axis=1)

    def _generated_term_outputs(
        self,
        *,
        function_name: str,
        arg_names: tuple[str, ...],
        output_names: tuple[str, ...],
        env: dict[str, Any],
        node_count: int,
    ) -> tuple[jnp.ndarray, ...] | None:
        if self.generated_module is None:
            return None
        missing = tuple(name for name in arg_names if name not in env)
        if missing:
            raise ValueError(
                f"Generated {function_name!r} arguments are unavailable: {missing!r}."
            )
        function = getattr(self.generated_module, function_name, None)
        if not callable(function):
            raise TypeError(
                f"Generated membrane module has no {function_name!r} function."
            )
        if not output_names:
            return None
        raw = function(*(env[name] for name in arg_names))
        values = raw if isinstance(raw, tuple) else (raw,)
        if len(values) != len(output_names):
            raise ValueError(
                f"Generated {function_name!r} returned {len(values)} values; "
                f"expected {len(output_names)}."
            )
        return tuple(
            _as_node_vector(value, node_count, self.dtype)
            for value in values
        )

    def prepare_membrane_step(
        self,
        V_mV: Any,
        gates_prev: Any,
        gates_new: Any,
        state: tuple[Any, ...],
        dt_ms: Any,
        I_ion: Any,
        I_background: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        V = jnp.atleast_1d(jnp.asarray(V_mV, dtype=self.dtype))
        gates_previous = jnp.asarray(gates_prev, dtype=self.dtype)
        gates_next = jnp.asarray(gates_new, dtype=self.dtype)
        ion = _as_node_vector(I_ion, V.shape[0], self.dtype)
        background = _as_node_vector(I_background, V.shape[0], self.dtype)
        step = self.model.step_program
        if step is None:
            return (
                tuple(jnp.asarray(value, dtype=self.dtype) for value in state),
                gates_next,
                background + ion,
                background,
                jnp.zeros((V.shape[0],), dtype=self.dtype),
            )

        prepare_gates = (
            gates_previous
            if step.prepare_gate_source is LinearizationGateSource.PREVIOUS
            else gates_next
        )
        prepare_required = self._generated_function_args("prepare_state")
        env = self._step_env(
            V,
            prepare_gates,
            state=state,
            dt_ms=dt_ms,
            I_ion=ion,
            I_background=background,
            parameters=parameters,
            required_names=prepare_required,
        )
        term_required = self._generated_function_args("step_current_terms")
        term_env = self._step_env(
            V,
            gates_next,
            state=state,
            dt_ms=dt_ms,
            I_ion=ion,
            I_background=background,
            parameters=parameters,
            required_names=term_required,
        )
        generated_state = self._generated_runtime_outputs(
            "prepare_state",
            env=env,
            node_count=V.shape[0],
        )
        prepared_state = (
            self._merge_state_updates(state, generated_state, node_count=V.shape[0])
            if generated_state is not None
            else self._apply_state_updates(
                state,
                step.prepare_state_updates,
                env,
                node_count=V.shape[0],
            )
        )
        generated_terms = self._generated_runtime_outputs(
            "step_current_terms",
            env=term_env,
            node_count=V.shape[0],
        )
        linearization_gates = (
            gates_previous
            if step.linearization_gate_source is LinearizationGateSource.PREVIOUS
            else gates_next
        )
        return (
            prepared_state,
            linearization_gates,
            self._generated_or_interpreted_step_term(
                "total_outward_current",
                generated_terms,
                step.total_outward_current,
                term_env,
                fallback=background + ion,
                node_count=V.shape[0],
            ),
            self._generated_or_interpreted_step_term(
                "explicit_outward_current",
                generated_terms,
                step.explicit_outward_current,
                term_env,
                fallback=background,
                node_count=V.shape[0],
            ),
            self._generated_or_interpreted_step_term(
                "correction_current",
                generated_terms,
                step.correction_current,
                term_env,
                fallback=jnp.zeros((V.shape[0],), dtype=self.dtype),
                node_count=V.shape[0],
            ),
        )

    def finalize_membrane_step(
        self,
        V_mV_prev: Any,
        V_mV_new: Any,
        gates_prev: Any,
        gates_new: Any,
        state_prev: tuple[Any, ...],
        prepared_state: tuple[Any, ...],
        dt_ms: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        step = self.model.step_program
        if step is None or not step.finalize_state_updates:
            return tuple(jnp.asarray(value, dtype=self.dtype) for value in prepared_state)
        V_new = jnp.atleast_1d(jnp.asarray(V_mV_new, dtype=self.dtype))
        finalize_required = self._generated_function_args("finalize_state")
        finalize_ion = (
            self.currents(V_new, gates_new, state=prepared_state)
            if finalize_required is None or "I_ion" in finalize_required
            else jnp.zeros((V_new.shape[0],), dtype=self.dtype)
        )
        env = self._step_env(
            V_new,
            jnp.asarray(gates_new, dtype=self.dtype),
            state=prepared_state,
            dt_ms=dt_ms,
            I_ion=finalize_ion,
            I_background=jnp.zeros((V_new.shape[0],), dtype=self.dtype),
            parameters=parameters,
            V_prev=jnp.atleast_1d(jnp.asarray(V_mV_prev, dtype=self.dtype)),
            required_names=finalize_required,
        )
        _ = gates_prev, state_prev
        generated = self._generated_runtime_outputs(
            "finalize_state",
            env=env,
            node_count=V_new.shape[0],
        )
        if generated is not None:
            return self._merge_state_updates(
                prepared_state,
                generated,
                node_count=V_new.shape[0],
            )
        return self._apply_state_updates(
            prepared_state,
            step.finalize_state_updates,
            env,
            node_count=V_new.shape[0],
        )

    def compute_step_diagnostics(
        self,
        V_mV_prev: Any,
        V_mV_new: Any,
        gates_prev: Any,
        gates_new: Any,
        state_prev: tuple[Any, ...],
        state_new: tuple[Any, ...],
        prepared_state: tuple[Any, ...],
        I_ion: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        step = self.model.step_program
        if step is None or not step.diagnostics:
            return ()
        V_prev = jnp.atleast_1d(jnp.asarray(V_mV_prev, dtype=self.dtype))
        V_new = jnp.atleast_1d(jnp.asarray(V_mV_new, dtype=self.dtype))
        diagnostic_required = self._generated_function_args("diagnostics")
        env = self._step_env(
            V_new,
            jnp.asarray(gates_new, dtype=self.dtype),
            state=prepared_state,
            dt_ms=0.0,
            I_ion=I_ion,
            I_background=jnp.zeros((V_new.shape[0],), dtype=self.dtype),
            parameters=parameters,
            V_prev=V_prev,
            required_names=diagnostic_required,
        )
        _ = gates_prev, state_prev, state_new
        env["Vm_prev"] = V_prev
        env["Vm_new"] = V_new
        generated = self._generated_runtime_outputs(
            "diagnostics",
            env=env,
            node_count=V_new.shape[0],
        )
        if generated is not None:
            return tuple(generated[name] for name in self.program.diagnostic_names)
        return tuple(
            _as_node_vector(
                evaluate_expression_jax(diagnostic.expression, env, dtype=self.dtype),
                V_new.shape[0],
                self.dtype,
            )
            for diagnostic in step.diagnostics
        )

    def reversal_values(self) -> jnp.ndarray:
        env = self.with_parameters()
        values = [
            jnp.ravel(jnp.asarray(evaluate_expression_jax(current.reversal, env, dtype=self.dtype)))[0]
            for current in self.model.currents
        ]
        if not values:
            return jnp.zeros((0,), dtype=self.dtype)
        return jnp.stack(values).astype(self.dtype)

    def observable_matrix(
        self,
        name: str,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        gates_arr = jnp.asarray(gates, dtype=self.dtype)
        node_count = int(gates_arr.shape[0]) if gates_arr.ndim else 1
        generated = self._generated_output_vector(
            name,
            gates_arr,
            state=state,
            parameters=parameters,
            node_count=node_count,
        )
        if generated is not None:
            return generated
        env = self._state_env(gates_arr, state=state, parameters=parameters)
        available = {observable.name: observable for observable in self.model.observables}
        observable = available[name]
        return _as_node_vector(
            evaluate_expression_jax(observable.expression, env, dtype=self.dtype),
            node_count,
            self.dtype,
        )

    def _base_env(
        self,
        V: jnp.ndarray | None = None,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        env = self.with_parameters(parameters)
        if V is not None:
            env["Vm"] = V
        return env

    def _generated_output_matrix(
        self,
        names: tuple[str, ...],
        V: jnp.ndarray,
        gates: jnp.ndarray,
        *,
        state: tuple[Any, ...],
        parameters: dict[str, Any] | None,
    ) -> jnp.ndarray | None:
        if (
            not self.generated_model_step_available
            or len(names) != len(self.model.currents)
            or any(name not in self.generated_output_index for name in names)
        ):
            return None
        outputs = self._generated_outputs(
            V,
            gates,
            state=state,
            parameters=parameters,
        )
        if outputs is None:
            return None
        return _stack_columns(
            [outputs[name] for name in names],
            int(V.shape[0]),
            self.dtype,
        )

    def _generated_output_vector(
        self,
        name: str,
        gates: jnp.ndarray,
        *,
        state: tuple[Any, ...],
        parameters: dict[str, Any] | None,
        node_count: int,
    ) -> jnp.ndarray | None:
        if (
            not self.generated_model_step_available
            or name not in self.generated_output_index
            or "Vm" in self.generated_arg_names
        ):
            return None
        V = jnp.zeros((node_count,), dtype=self.dtype)
        outputs = self._generated_outputs(
            V,
            gates,
            state=state,
            parameters=parameters,
        )
        if outputs is None:
            return None
        return outputs[name]

    def _generated_outputs(
        self,
        V: jnp.ndarray,
        gates: jnp.ndarray,
        *,
        state: tuple[Any, ...],
        parameters: dict[str, Any] | None,
    ) -> dict[str, jnp.ndarray] | None:
        if self.generated_module is None:
            return None
        env = self._state_env(gates, state=state, V=V, parameters=parameters)
        if any(name not in env for name in self.generated_arg_names):
            return None
        raw = self.generated_module.model_step(
            *(env[name] for name in self.generated_arg_names)
        )
        values = raw if isinstance(raw, tuple) else (raw,)
        if len(values) != len(self.generated_output_names):
            return None
        return {
            name: _as_node_vector(value, int(V.shape[0]), self.dtype)
            for name, value in zip(self.generated_output_names, values, strict=True)
        }

    def _state_env(
        self,
        gates: jnp.ndarray,
        *,
        state: tuple[Any, ...] = (),
        V: jnp.ndarray | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        env = self._base_env(V, parameters=parameters)
        for index, name in enumerate(self.gate_state_names):
            env[name] = gates[:, index]
        if state:
            if len(state) != len(self.membrane_state_names):
                raise ValueError(
                    "Expected "
                    f"{len(self.membrane_state_names)} Model IR membrane states, "
                    f"got {len(state)}."
                )
            for name, value in zip(self.membrane_state_names, state, strict=True):
                env[name] = jnp.asarray(value, dtype=self.dtype)
        return env

    def _step_env(
        self,
        V: jnp.ndarray,
        gates: jnp.ndarray,
        *,
        state: tuple[Any, ...],
        dt_ms: Any,
        I_ion: Any,
        I_background: Any,
        parameters: dict[str, Any] | None = None,
        V_prev: jnp.ndarray | None = None,
        required_names: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        env = self._state_env(gates, state=state, V=V, parameters=parameters)
        env["Vm_new"] = V
        env["Vm_prev"] = V if V_prev is None else V_prev
        env["dt"] = jnp.asarray(dt_ms, dtype=self.dtype)
        env["I_ion"] = _as_node_vector(I_ion, V.shape[0], self.dtype)
        env["I_background"] = _as_node_vector(I_background, V.shape[0], self.dtype)
        required_currents = (
            set(self.model_current_names)
            if required_names is None
            else set(self.model_current_names).intersection(required_names)
        )
        if required_currents:
            currents = self.current_matrix(V, gates, state=state, parameters=parameters)
            for index, name in enumerate(self.model_current_names):
                if name not in required_currents:
                    continue
                if name not in env:
                    env[name] = currents[:, index]
                else:
                    env[name] = env[name] + currents[:, index]
        return env

    @property
    def model_current_names(self) -> tuple[str, ...]:
        return self.program.raw_current_names

    def _apply_state_updates(
        self,
        state: tuple[Any, ...],
        updates: tuple[StateUpdate, ...],
        env: dict[str, Any],
        *,
        node_count: int,
    ) -> tuple[jnp.ndarray, ...]:
        state_values = {
            name: _as_node_vector(value, node_count, self.dtype)
            for name, value in zip(self.membrane_state_names, state, strict=True)
        }
        base_env = dict(env)
        evaluated: list[tuple[str, jnp.ndarray]] = []
        for update in updates:
            value = _as_node_vector(
                evaluate_expression_jax(update.expression, base_env, dtype=self.dtype),
                node_count,
                self.dtype,
            )
            evaluated.append((update.state, value))
        for state_name, value in evaluated:
            state_values[state_name] = value
            env[state_name] = value
        return tuple(state_values[name] for name in self.membrane_state_names)

    def _merge_state_updates(
        self,
        state: tuple[Any, ...],
        updates: dict[str, jnp.ndarray],
        *,
        node_count: int,
    ) -> tuple[jnp.ndarray, ...]:
        state_values = {
            name: _as_node_vector(value, node_count, self.dtype)
            for name, value in zip(self.membrane_state_names, state, strict=True)
        }
        state_values.update(updates)
        return tuple(state_values[name] for name in self.membrane_state_names)

    def _generated_runtime_outputs(
        self,
        function_name: str,
        *,
        env: dict[str, Any],
        node_count: int,
    ) -> dict[str, jnp.ndarray] | None:
        if self.generated_module is None and self.generated_contract is None:
            return None
        if self.generated_module is None or self.generated_contract is None:
            raise ValueError(
                "Generated membrane module and contract must be loaded together."
            )
        spec = self.generated_contract.function(function_name)
        missing = tuple(name for name in spec.args if name not in env)
        if missing:
            raise ValueError(
                f"Generated {function_name!r} arguments are unavailable: {missing!r}."
            )
        function = getattr(self.generated_module, function_name, None)
        if not callable(function):
            raise TypeError(
                f"Generated membrane module has no {function_name!r} function."
            )
        raw = function(*(env[name] for name in spec.args))
        values = raw if isinstance(raw, tuple) else (raw,)
        if not spec.outputs:
            return {}
        if len(values) != len(spec.outputs):
            raise ValueError(
                f"Generated {function_name!r} returned {len(values)} values; "
                f"expected {len(spec.outputs)}."
            )
        return {
            name: _as_node_vector(value, node_count, self.dtype)
            for name, value in zip(spec.outputs, values, strict=True)
        }

    def _generated_function_args(self, name: str) -> tuple[str, ...] | None:
        if self.generated_contract is None:
            return None
        return self.generated_contract.function(name).args

    def _generated_or_interpreted_step_term(
        self,
        name: str,
        generated: dict[str, jnp.ndarray] | None,
        expression: Expression | None,
        env: dict[str, Any],
        *,
        fallback: Any,
        node_count: int,
    ) -> jnp.ndarray:
        if generated is not None and name in generated:
            return generated[name]
        return self._step_current_term(
            expression,
            env,
            fallback=fallback,
            node_count=node_count,
        )

    def _step_current_term(
        self,
        expression: Expression | None,
        env: dict[str, Any],
        *,
        fallback: Any,
        node_count: int,
    ) -> jnp.ndarray:
        if expression is None:
            return _as_node_vector(fallback, node_count, self.dtype)
        return _as_node_vector(
            evaluate_expression_jax(expression, env, dtype=self.dtype),
            node_count,
            self.dtype,
        )


def _call_intrinsic(name: str, args: list[Any], *, dtype: jnp.dtype) -> Any:
    if name == "abs":
        return jnp.abs(args[0])
    if name == "clip":
        return jnp.clip(args[0], args[1], args[2])
    if name == "exp":
        return jnp.exp(args[0])
    if name == "expm1":
        return jnp.expm1(args[0])
    if name == "log":
        return jnp.log(args[0])
    if name == "log1p":
        return jnp.log1p(args[0])
    if name == "maximum":
        return jnp.maximum(args[0], args[1])
    if name == "minimum":
        return jnp.minimum(args[0], args[1])
    if name == "pow":
        return jnp.power(args[0], args[1])
    if name == "q10":
        return jnp.power(args[0], (args[1] - args[2]) / jnp.asarray(10.0, dtype=dtype))
    if name == "alpha_from_inf_tau":
        return args[0] / args[1]
    if name == "beta_from_inf_tau":
        return (jnp.asarray(1.0, dtype=dtype) - args[0]) / args[1]
    if name == "safe_exp":
        x = jnp.asarray(args[0], dtype=dtype)
        return jnp.where(x < jnp.asarray(-100.0, dtype=dtype), 0.0, jnp.exp(x))
    if name == "sigmoid":
        one = jnp.asarray(1.0, dtype=dtype)
        return one / (one + jnp.exp(-args[0]))
    if name == "sqrt":
        return jnp.sqrt(args[0])
    if name == "tanh":
        return jnp.tanh(args[0])
    if name == "vtrap":
        return _vtrap(args[0], args[1], dtype=dtype)
    if name == "where":
        return jnp.where(args[0], args[1], args[2])
    raise NotImplementedError(f"JAX Model IR intrinsic {name!r} is not implemented.")


def _vtrap(x: Any, y: Any, *, dtype: jnp.dtype) -> Any:
    x_arr = jnp.asarray(x, dtype=dtype)
    y_arr = jnp.asarray(y, dtype=dtype)
    z = x_arr / y_arr
    return jnp.where(
        jnp.abs(z) < jnp.asarray(1e-6, dtype=dtype),
        y_arr * (jnp.asarray(1.0, dtype=dtype) - z / jnp.asarray(2.0, dtype=dtype)),
        x_arr / (jnp.exp(z) - jnp.asarray(1.0, dtype=dtype)),
    )


def _as_node_vector(value: Any, node_count: int, dtype: jnp.dtype) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=dtype)
    if arr.shape == ():
        return jnp.full((node_count,), arr, dtype=dtype)
    if arr.shape == (node_count,):
        return arr
    return jnp.broadcast_to(arr, (node_count,)).astype(dtype)


def _stack_columns(values: list[jnp.ndarray], node_count: int, dtype: jnp.dtype) -> jnp.ndarray:
    if not values:
        return jnp.zeros((node_count, 0), dtype=dtype)
    return jnp.stack(values, axis=1).astype(dtype)


def _normalize_probabilities_jax(values: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    clipped = jnp.maximum(values, jnp.asarray(0.0, dtype=dtype))
    total = jnp.sum(clipped, axis=1, keepdims=True)
    return clipped / jnp.maximum(total, jnp.asarray(1e-12, dtype=dtype))


def _generated_names(module: Any | None, attr: str) -> tuple[str, ...]:
    if module is None:
        return ()
    values = getattr(module, attr, ())
    if not isinstance(values, tuple | list):
        return ()
    return tuple(str(value) for value in values)


def _source_output_names(model: ModelIR, section: str) -> tuple[str, ...]:
    outputs = model.metadata.get("source_outputs", {})
    if not isinstance(outputs, dict):
        return ()
    values = outputs.get(section, ())
    if not isinstance(values, tuple | list):
        return ()
    return tuple(str(value) for value in values)
