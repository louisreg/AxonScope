"""NumPy reference interpreter for Model IR membrane steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .expressions import BinaryOp, Call, Expression, Literal, Symbol, UnaryOp
from .program import membrane_program_from_model_ir
from .schema import GateUpdateKind, LinearizationGateSource, ModelIR, StateUpdate


def parameter_defaults(model: ModelIR) -> dict[str, float]:
    """Return dynamic parameter defaults carried by a Model IR."""

    return {
        parameter.name: float(parameter.default)
        for parameter in model.parameters
        if parameter.default is not None
    }


def evaluate_expression_np(expr: Expression, env: dict[str, Any], *, dtype: np.dtype) -> Any:
    """Evaluate a Model IR expression with NumPy operations."""

    if isinstance(expr, Literal):
        return np.asarray(expr.value, dtype=dtype)
    if isinstance(expr, Symbol):
        try:
            return env[expr.name]
        except KeyError as exc:
            raise KeyError(f"Missing Model IR symbol {expr.name!r}.") from exc
    if isinstance(expr, UnaryOp):
        value = evaluate_expression_np(expr.operand, env, dtype=dtype)
        if expr.op == "neg":
            return -value
    if isinstance(expr, BinaryOp):
        left = evaluate_expression_np(expr.left, env, dtype=dtype)
        right = evaluate_expression_np(expr.right, env, dtype=dtype)
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "div":
            return left / right
        if expr.op == "pow":
            return np.power(left, right)
        if expr.op == "lt":
            return left < right
        if expr.op == "le":
            return left <= right
        if expr.op == "gt":
            return left > right
        if expr.op == "ge":
            return left >= right
    if isinstance(expr, Call):
        args = [evaluate_expression_np(arg, env, dtype=dtype) for arg in expr.args]
        return _call_intrinsic(expr.intrinsic, args, dtype=dtype)
    raise TypeError(f"Unsupported Model IR expression {type(expr).__name__}.")


@dataclass(frozen=True, slots=True)
class NumpyModelStep:
    """One fully visible membrane-step result from Model IR semantics."""

    state: np.ndarray
    conductances: np.ndarray
    currents: np.ndarray
    total_outward_current: np.ndarray
    total_conductance: np.ndarray
    conductance_reversal_sum: np.ndarray
    observables: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class NumpyMembraneStepPlan:
    """Prepared membrane-step terms mirroring the solver-side step plan."""

    state: tuple[np.ndarray, ...]
    linearization_gates: np.ndarray
    total_outward_current: np.ndarray
    explicit_outward_current: np.ndarray
    correction_current: np.ndarray


class NumpyModelInterpreter:
    """Small deterministic interpreter for Model IR membrane equations."""

    def __init__(self, model: ModelIR, *, dtype: Any = np.float32) -> None:
        self.model = model
        self.dtype = np.dtype(dtype)
        self.parameters = parameter_defaults(model)
        self.program = membrane_program_from_model_ir(model)
        self.gate_state_names = self.program.gate_state_names
        self.membrane_states = self.program.membrane_states
        self.membrane_state_names = self.program.membrane_state_names
        self.state_names = self.gate_state_names
        self.current_names = self.program.raw_current_names
        self.conductance_names = self.program.raw_conductance_names

    def with_parameters(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(self.parameters)
        if overrides:
            params.update({key: self.dtype.type(value) for key, value in overrides.items()})
        return {key: np.asarray(value, dtype=self.dtype) for key, value in params.items()}

    def rate_constants(
        self,
        V_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        V = np.atleast_1d(np.asarray(V_mV, dtype=self.dtype))
        env = self._base_env(V, parameters=parameters)
        alpha = [
            _as_node_vector(
                evaluate_expression_np(gate.alpha, env, dtype=self.dtype),
                V.shape[0],
                self.dtype,
            )
            for gate in self.model.gates
        ]
        beta = [
            _as_node_vector(
                evaluate_expression_np(gate.beta, env, dtype=self.dtype),
                V.shape[0],
                self.dtype,
            )
            for gate in self.model.gates
        ]
        return _stack_columns(alpha, V.shape[0], self.dtype), _stack_columns(beta, V.shape[0], self.dtype)

    def q10_factors(
        self,
        V_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        V = np.atleast_1d(np.asarray(V_mV, dtype=self.dtype))
        env = self._base_env(V, parameters=parameters)
        factors = []
        for gate in self.model.gates:
            if gate.q10 is None:
                factors.append(np.ones((V.shape[0],), dtype=self.dtype))
            else:
                factors.append(
                    _as_node_vector(
                        evaluate_expression_np(gate.q10, env, dtype=self.dtype),
                        V.shape[0],
                        self.dtype,
                    )
                )
        return _stack_columns(factors, V.shape[0], self.dtype)

    def init_gates(
        self,
        V0_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        V = np.atleast_1d(np.asarray(V0_mV, dtype=self.dtype))
        if not self.model.gates:
            return np.zeros((V.shape[0], 0), dtype=self.dtype)
        alpha, beta = self.rate_constants(V, parameters=parameters)
        denom = np.maximum(alpha + beta, self.dtype.type(1e-12))
        return alpha / denom

    def init_membrane_state(
        self,
        V0_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, ...]:
        V = np.atleast_1d(np.asarray(V0_mV, dtype=self.dtype))
        env = self._base_env(V, parameters=parameters)
        values: list[np.ndarray] = []
        for state in self.membrane_states:
            if state.initial is None:
                value = np.zeros((V.shape[0],), dtype=self.dtype)
            else:
                value = _as_node_vector(
                    evaluate_expression_np(state.initial, env, dtype=self.dtype),
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
        dt_ms: float,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        gates = np.asarray(gates_prev, dtype=self.dtype)
        V = np.atleast_1d(np.asarray(V_mV, dtype=self.dtype))
        if gates.shape[-1] == 0:
            return gates
        alpha, beta = self.rate_constants(V, parameters=parameters)
        q10 = self.q10_factors(V, parameters=parameters)
        alpha = q10 * alpha
        beta = q10 * beta
        sum_ab = np.maximum(alpha + beta, self.dtype.type(1e-12))
        g_inf = alpha / sum_ab
        if all(gate.update is GateUpdateKind.CRANK_NICOLSON for gate in self.model.gates):
            denom = np.maximum(1.0 / dt_ms + 0.5 * sum_ab, self.dtype.type(1e-12))
            return alpha / denom + ((1.0 / dt_ms) - 0.5 * sum_ab) / denom * gates
        tau = self.dtype.type(1.0) / sum_ab
        return g_inf - (g_inf - gates) * np.exp(-self.dtype.type(dt_ms) / tau)

    def conductances(
        self,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        gates_arr = np.asarray(gates, dtype=self.dtype)
        node_count = int(gates_arr.shape[0]) if gates_arr.ndim else 1
        env = self._state_env(gates_arr, state=state, parameters=parameters)
        cols = [
            _as_node_vector(
                evaluate_expression_np(current.conductance, env, dtype=self.dtype),
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
    ) -> np.ndarray:
        V = np.atleast_1d(np.asarray(V_mV, dtype=self.dtype))
        gates_arr = np.asarray(gates, dtype=self.dtype)
        env = self._state_env(gates_arr, state=state, V=V, parameters=parameters)
        cols = [
            _as_node_vector(
                evaluate_expression_np(current.current, env, dtype=self.dtype),
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
    ) -> np.ndarray:
        return np.sum(
            self.current_matrix(V_mV, gates, state=state, parameters=parameters),
            axis=1,
        )

    def membrane_conductance_terms(
        self,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        gates_arr = np.asarray(gates, dtype=self.dtype)
        node_count = int(gates_arr.shape[0]) if gates_arr.ndim else 1
        env = self._state_env(gates_arr, state=state, parameters=parameters)
        conductances = []
        reversals = []
        for current in self.model.currents:
            conductances.append(
                _as_node_vector(
                    evaluate_expression_np(current.conductance, env, dtype=self.dtype),
                    node_count,
                    self.dtype,
                )
            )
            reversals.append(
                _as_node_vector(
                    evaluate_expression_np(current.reversal, env, dtype=self.dtype),
                    node_count,
                    self.dtype,
                )
            )
        g = _stack_columns(conductances, node_count, self.dtype)
        e = _stack_columns(reversals, node_count, self.dtype)
        return np.sum(g, axis=1), np.sum(g * e, axis=1)

    def prepare_membrane_step(
        self,
        V_mV: Any,
        gates_prev: Any,
        gates_new: Any,
        state: tuple[Any, ...],
        dt_ms: float,
        I_ion: Any,
        I_background: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> NumpyMembraneStepPlan:
        V = np.atleast_1d(np.asarray(V_mV, dtype=self.dtype))
        gates_previous = np.asarray(gates_prev, dtype=self.dtype)
        gates_next = np.asarray(gates_new, dtype=self.dtype)
        ion = _as_node_vector(I_ion, V.shape[0], self.dtype)
        background = _as_node_vector(I_background, V.shape[0], self.dtype)
        step = self.model.step_program
        if step is None:
            return NumpyMembraneStepPlan(
                state=tuple(np.asarray(value, dtype=self.dtype) for value in state),
                linearization_gates=gates_next,
                total_outward_current=background + ion,
                explicit_outward_current=background,
                correction_current=np.zeros((V.shape[0],), dtype=self.dtype),
            )

        prepare_gates = (
            gates_previous
            if step.prepare_gate_source is LinearizationGateSource.PREVIOUS
            else gates_next
        )
        env = self._step_env(
            V,
            prepare_gates,
            state=state,
            dt_ms=dt_ms,
            I_ion=ion,
            I_background=background,
            parameters=parameters,
        )
        term_env = self._step_env(
            V,
            gates_next,
            state=state,
            dt_ms=dt_ms,
            I_ion=ion,
            I_background=background,
            parameters=parameters,
        )
        prepared_state = self._apply_state_updates(
            state,
            step.prepare_state_updates,
            env,
            node_count=V.shape[0],
        )
        linearization_gates = (
            gates_previous
            if step.linearization_gate_source is LinearizationGateSource.PREVIOUS
            else gates_next
        )
        return NumpyMembraneStepPlan(
            state=prepared_state,
            linearization_gates=linearization_gates,
            total_outward_current=self._step_current_term(
                step.total_outward_current,
                term_env,
                fallback=background + ion,
                node_count=V.shape[0],
            ),
            explicit_outward_current=self._step_current_term(
                step.explicit_outward_current,
                term_env,
                fallback=background,
                node_count=V.shape[0],
            ),
            correction_current=self._step_current_term(
                step.correction_current,
                term_env,
                fallback=np.zeros((V.shape[0],), dtype=self.dtype),
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
        step_plan: NumpyMembraneStepPlan,
        dt_ms: float,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, ...]:
        step = self.model.step_program
        if step is None or not step.finalize_state_updates:
            return step_plan.state
        V_new = np.atleast_1d(np.asarray(V_mV_new, dtype=self.dtype))
        env = self._step_env(
            V_new,
            np.asarray(gates_new, dtype=self.dtype),
            state=step_plan.state,
            dt_ms=dt_ms,
            I_ion=self.currents(V_new, gates_new, state=step_plan.state),
            I_background=np.zeros((V_new.shape[0],), dtype=self.dtype),
            parameters=parameters,
            V_prev=np.atleast_1d(np.asarray(V_mV_prev, dtype=self.dtype)),
        )
        _ = gates_prev, state_prev
        return self._apply_state_updates(
            step_plan.state,
            step.finalize_state_updates,
            env,
            node_count=V_new.shape[0],
        )

    def diagnostic_names(self) -> tuple[str, ...]:
        step = self.model.step_program
        if step is None:
            return ()
        return tuple(diagnostic.name for diagnostic in step.diagnostics)

    def compute_step_diagnostics(
        self,
        V_mV_prev: Any,
        V_mV_new: Any,
        gates_prev: Any,
        gates_new: Any,
        state_prev: tuple[Any, ...],
        state_new: tuple[Any, ...],
        step_plan: NumpyMembraneStepPlan,
        I_ion: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, ...]:
        step = self.model.step_program
        if step is None or not step.diagnostics:
            return ()
        V_prev = np.atleast_1d(np.asarray(V_mV_prev, dtype=self.dtype))
        V_new = np.atleast_1d(np.asarray(V_mV_new, dtype=self.dtype))
        env = self._step_env(
            V_new,
            np.asarray(gates_new, dtype=self.dtype),
            state=step_plan.state,
            dt_ms=0.0,
            I_ion=I_ion,
            I_background=np.zeros((V_new.shape[0],), dtype=self.dtype),
            parameters=parameters,
            V_prev=V_prev,
        )
        _ = gates_prev, state_prev, state_new
        env["Vm_prev"] = V_prev
        env["Vm_new"] = V_new
        return tuple(
            _as_node_vector(
                evaluate_expression_np(diagnostic.expression, env, dtype=self.dtype),
                V_new.shape[0],
                self.dtype,
            )
            for diagnostic in step.diagnostics
        )

    def step(
        self,
        V_mV: Any,
        gates_prev: Any,
        dt_ms: float,
        *,
        parameters: dict[str, Any] | None = None,
        requested_observables: tuple[str, ...] = (),
    ) -> NumpyModelStep:
        gates_new = self.gate_update(
            gates_prev,
            V_mV,
            dt_ms,
            parameters=parameters,
        )
        currents = self.current_matrix(V_mV, gates_new, parameters=parameters)
        conductances = self.conductances(gates_new, parameters=parameters)
        total_g, ge = self.membrane_conductance_terms(gates_new, parameters=parameters)
        observables = self.observables(
            gates_new,
            parameters=parameters,
            requested=requested_observables,
        )
        return NumpyModelStep(
            state=gates_new,
            conductances=conductances,
            currents=currents,
            total_outward_current=np.sum(currents, axis=1),
            total_conductance=total_g,
            conductance_reversal_sum=ge,
            observables=observables,
        )

    def observables(
        self,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
        requested: tuple[str, ...] = (),
    ) -> dict[str, np.ndarray]:
        gates_arr = np.asarray(gates, dtype=self.dtype)
        node_count = int(gates_arr.shape[0]) if gates_arr.ndim else 1
        env = self._state_env(gates_arr, state=state, parameters=parameters)
        names = requested or tuple(observable.name for observable in self.model.observables)
        available = {observable.name: observable for observable in self.model.observables}
        out: dict[str, np.ndarray] = {}
        for name in names:
            observable = available[name]
            out[name] = _as_node_vector(
                evaluate_expression_np(observable.expression, env, dtype=self.dtype),
                node_count,
                self.dtype,
            )
        return out

    def _base_env(
        self,
        V: np.ndarray | None = None,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        env = self.with_parameters(parameters)
        if V is not None:
            env["Vm"] = V
        return env

    def _state_env(
        self,
        gates: np.ndarray,
        *,
        state: tuple[Any, ...] = (),
        V: np.ndarray | None = None,
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
                env[name] = np.asarray(value, dtype=self.dtype)
        return env

    def _step_env(
        self,
        V: np.ndarray,
        gates: np.ndarray,
        *,
        state: tuple[Any, ...],
        dt_ms: Any,
        I_ion: Any,
        I_background: Any,
        parameters: dict[str, Any] | None = None,
        V_prev: np.ndarray | None = None,
    ) -> dict[str, Any]:
        env = self._state_env(gates, state=state, V=V, parameters=parameters)
        env["Vm_new"] = V
        env["Vm_prev"] = V if V_prev is None else V_prev
        env["dt"] = np.asarray(dt_ms, dtype=self.dtype)
        env["I_ion"] = _as_node_vector(I_ion, V.shape[0], self.dtype)
        env["I_background"] = _as_node_vector(
            I_background,
            V.shape[0],
            self.dtype,
        )
        currents = self.current_matrix(V, gates, state=state, parameters=parameters)
        for index, name in enumerate(self.current_names):
            if name not in env:
                env[name] = currents[:, index]
            else:
                env[name] = env[name] + currents[:, index]
        return env

    def _apply_state_updates(
        self,
        state: tuple[Any, ...],
        updates: tuple[StateUpdate, ...],
        env: dict[str, Any],
        *,
        node_count: int,
    ) -> tuple[np.ndarray, ...]:
        state_values = {
            name: _as_node_vector(value, node_count, self.dtype)
            for name, value in zip(self.membrane_state_names, state, strict=True)
        }
        base_env = dict(env)
        evaluated: list[tuple[str, np.ndarray]] = []
        for update in updates:
            value = _as_node_vector(
                evaluate_expression_np(update.expression, base_env, dtype=self.dtype),
                node_count,
                self.dtype,
            )
            evaluated.append((update.state, value))
        for state_name, value in evaluated:
            state_values[state_name] = value
            env[state_name] = value
        return tuple(state_values[name] for name in self.membrane_state_names)

    def _step_current_term(
        self,
        expression: Expression | None,
        env: dict[str, Any],
        *,
        fallback: Any,
        node_count: int,
    ) -> np.ndarray:
        if expression is None:
            return _as_node_vector(fallback, node_count, self.dtype)
        return _as_node_vector(
            evaluate_expression_np(expression, env, dtype=self.dtype),
            node_count,
            self.dtype,
        )


def _call_intrinsic(name: str, args: list[Any], *, dtype: np.dtype) -> Any:
    if name == "abs":
        return np.abs(args[0])
    if name == "clip":
        return np.clip(args[0], args[1], args[2])
    if name == "exp":
        return np.exp(args[0])
    if name == "expm1":
        return np.expm1(args[0])
    if name == "log":
        return np.log(args[0])
    if name == "log1p":
        return np.log1p(args[0])
    if name == "maximum":
        return np.maximum(args[0], args[1])
    if name == "minimum":
        return np.minimum(args[0], args[1])
    if name == "pow":
        return np.power(args[0], args[1])
    if name == "q10":
        return np.power(args[0], (args[1] - args[2]) / dtype.type(10.0))
    if name == "alpha_from_inf_tau":
        return args[0] / args[1]
    if name == "beta_from_inf_tau":
        return (dtype.type(1.0) - args[0]) / args[1]
    if name == "safe_exp":
        x = np.asarray(args[0], dtype=dtype)
        return np.where(x < dtype.type(-100.0), dtype.type(0.0), np.exp(x))
    if name == "sigmoid":
        one = dtype.type(1.0)
        return one / (one + np.exp(-args[0]))
    if name == "sqrt":
        return np.sqrt(args[0])
    if name == "tanh":
        return np.tanh(args[0])
    if name == "vtrap":
        return _vtrap(args[0], args[1], dtype=dtype)
    if name == "where":
        return np.where(args[0], args[1], args[2])
    raise NotImplementedError(f"NumPy Model IR intrinsic {name!r} is not implemented.")


def _vtrap(x: Any, y: Any, *, dtype: np.dtype) -> Any:
    x_arr = np.asarray(x, dtype=dtype)
    y_arr = np.asarray(y, dtype=dtype)
    z = x_arr / y_arr
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.where(
            np.abs(z) < dtype.type(1e-6),
            y_arr * (dtype.type(1.0) - z / dtype.type(2.0)),
            x_arr / (np.exp(z) - dtype.type(1.0)),
        )


def _as_node_vector(value: Any, node_count: int, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.shape == ():
        return np.full((node_count,), arr.item(), dtype=dtype)
    if arr.shape == (node_count,):
        return arr
    return np.broadcast_to(arr, (node_count,)).astype(dtype, copy=False)


def _stack_columns(values: list[np.ndarray], node_count: int, dtype: np.dtype) -> np.ndarray:
    if not values:
        return np.zeros((node_count, 0), dtype=dtype)
    return np.stack(values, axis=1).astype(dtype, copy=False)
