"""Model-independent JAX adapter for generated membrane modules."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .generated_contract import GeneratedMembraneContract
from .kinetics import (
    dense_kinetic_matrix,
    solve_kinetic_transitions,
)


class GeneratedJaxMembraneLowering:
    """Execute one generated membrane contract without reconstructing Model IR."""

    def __init__(
        self,
        module: Any,
        contract: GeneratedMembraneContract,
        *,
        dtype: jnp.dtype,
        parameter_values: dict[str, Any],
    ) -> None:
        self.module = module
        self.contract = contract
        self.dtype = dtype
        self.parameters = {
            name: jnp.asarray(value, dtype=dtype)
            for name, value in parameter_values.items()
        }
        self.gate_state_names = contract.gate_state_names
        self.hh_gate_state_names = contract.hh_gate_state_names
        self.kinetic_state_names = contract.kinetic_state_names
        self.membrane_state_names = contract.membrane_state_names
        self.state_names = contract.gate_state_names

    @property
    def generated_model_step_available(self) -> bool:
        return True

    @property
    def generated_gate_terms_available(self) -> bool:
        return True

    @property
    def generated_membrane_terms_available(self) -> bool:
        return True

    def with_parameters(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        values = dict(self.parameters)
        if overrides:
            values.update(
                {
                    name: jnp.asarray(value, dtype=self.dtype)
                    for name, value in overrides.items()
                }
            )
        return values

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
        outputs = self._call(
            "gate_terms",
            self._base_env(V, parameters=parameters),
            node_count=V.shape[0],
        )
        names = self.contract.function("gate_terms").outputs
        values = tuple(outputs[name] for name in names)
        alpha = list(values[0::3])
        beta = list(values[1::3])
        factors = list(values[2::3])
        alpha.extend(jnp.zeros((V.shape[0],), dtype=self.dtype) for _ in self.kinetic_state_names)
        beta.extend(jnp.zeros((V.shape[0],), dtype=self.dtype) for _ in self.kinetic_state_names)
        factors.extend(jnp.ones((V.shape[0],), dtype=self.dtype) for _ in self.kinetic_state_names)
        return (
            _stack_columns(tuple(alpha), V.shape[0], self.dtype),
            _stack_columns(tuple(beta), V.shape[0], self.dtype),
            _stack_columns(tuple(factors), V.shape[0], self.dtype),
        )

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
        denominator = jnp.maximum(
            alpha[:, :hh_count] + beta[:, :hh_count],
            jnp.asarray(1e-12, dtype=self.dtype),
        )
        hh = alpha[:, :hh_count] / denominator
        kinetics = self._init_kinetic_states(V, parameters=parameters)
        return jnp.concatenate((hh, kinetics), axis=1)

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
        alpha = q10[:, :hh_count] * alpha[:, :hh_count]
        beta = q10[:, :hh_count] * beta[:, :hh_count]
        sum_ab = jnp.maximum(alpha + beta, jnp.asarray(1e-12, dtype=self.dtype))
        dt = jnp.asarray(dt_ms, dtype=self.dtype)
        previous_hh = gates[:, :hh_count]
        if all(
            mode == "crank_nicolson"
            for mode in self.contract.gate_update_modes[:hh_count]
        ):
            denominator = jnp.maximum(
                1.0 / dt + 0.5 * sum_ab,
                jnp.asarray(1e-12, dtype=self.dtype),
            )
            hh = alpha / denominator + (
                (1.0 / dt) - 0.5 * sum_ab
            ) / denominator * previous_hh
        else:
            equilibrium = alpha / sum_ab
            tau = jnp.asarray(1.0, dtype=self.dtype) / sum_ab
            hh = equilibrium - (equilibrium - previous_hh) * jnp.exp(-dt / tau)
        kinetics = self._update_kinetic_states(
            gates[:, hh_count:], V, dt, parameters=parameters
        )
        return jnp.concatenate((hh, kinetics), axis=1)

    def _kinetic_rates(
        self,
        V: jnp.ndarray,
        *,
        parameters: dict[str, Any] | None,
    ) -> dict[str, jnp.ndarray]:
        if not self.contract.kinetic_blocks:
            return {}
        return self._call(
            "kinetic_terms",
            self._base_env(V, parameters=parameters),
            node_count=V.shape[0],
        )

    def _kinetic_matrix(
        self,
        block: Any,
        rates: dict[str, jnp.ndarray],
        node_count: int,
    ) -> jnp.ndarray:
        width = len(block.states)
        return dense_kinetic_matrix(
            width=width,
            transitions=self._kinetic_transitions(block, rates),
            node_count=node_count,
            dtype=self.dtype,
        )

    @staticmethod
    def _kinetic_transitions(
        block: Any,
        rates: dict[str, jnp.ndarray],
    ) -> tuple[tuple[int, int, jnp.ndarray], ...]:
        return tuple(
            (transition.source, transition.target, rates[transition.output])
            for transition in block.transitions
        )

    def _init_kinetic_states(
        self,
        V: jnp.ndarray,
        *,
        parameters: dict[str, Any] | None,
    ) -> jnp.ndarray:
        if not self.contract.kinetic_blocks:
            return jnp.zeros((V.shape[0], 0), dtype=self.dtype)
        rates = self._kinetic_rates(V, parameters=parameters)
        declared = self._call(
            "kinetic_initials",
            self._base_env(V, parameters=parameters),
            node_count=V.shape[0],
        )
        values: list[jnp.ndarray] = []
        for block in self.contract.kinetic_blocks:
            if block.initialization == "stationary":
                matrix = self._kinetic_matrix(block, rates, V.shape[0])
                system = matrix.at[:, -1, :].set(jnp.asarray(1.0, dtype=self.dtype))
                rhs = jnp.zeros((V.shape[0], len(block.states)), dtype=self.dtype)
                rhs = rhs.at[:, -1].set(jnp.asarray(1.0, dtype=self.dtype))
                block_values = jnp.linalg.solve(system, rhs[..., None])[..., 0]
            else:
                block_values = _stack_columns(
                    tuple(declared[name] for name in block.states),
                    V.shape[0],
                    self.dtype,
                )
            if block.conserve_probability:
                block_values = _normalize_probabilities(block_values, self.dtype)
            values.extend(block_values[:, index] for index in range(len(block.states)))
        return _stack_columns(tuple(values), V.shape[0], self.dtype)

    def _update_kinetic_states(
        self,
        previous: jnp.ndarray,
        V: jnp.ndarray,
        dt: jnp.ndarray,
        *,
        parameters: dict[str, Any] | None,
    ) -> jnp.ndarray:
        rates = self._kinetic_rates(V, parameters=parameters)
        values: list[jnp.ndarray] = []
        offset = 0
        for block in self.contract.kinetic_blocks:
            width = len(block.states)
            block_values = solve_kinetic_transitions(
                width=width,
                transitions=self._kinetic_transitions(block, rates),
                previous=previous[:, offset : offset + width],
                dt=dt,
                node_count=V.shape[0],
                dtype=self.dtype,
                conserve_probability=block.conserve_probability,
            )
            values.extend(block_values[:, index] for index in range(width))
            offset += width
        return _stack_columns(tuple(values), V.shape[0], self.dtype)

    def init_membrane_state(
        self,
        V0_mV: Any,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        V = jnp.atleast_1d(jnp.asarray(V0_mV, dtype=self.dtype))
        outputs = self._call(
            "init_state",
            self._base_env(V, parameters=parameters),
            node_count=V.shape[0],
        )
        return tuple(outputs[name] for name in self.membrane_state_names)

    def conductances(
        self,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        gates_array = jnp.asarray(gates, dtype=self.dtype)
        node_count = int(gates_array.shape[0]) if gates_array.ndim else 1
        outputs = self._call(
            "membrane_terms",
            self._state_env(gates_array, state=state, parameters=parameters),
            node_count=node_count,
        )
        names = self.contract.function("membrane_terms").outputs
        return _stack_columns(
            tuple(outputs[name] for name in names[0::2]),
            node_count,
            self.dtype,
        )

    def current_matrix(
        self,
        V_mV: Any,
        gates: Any,
        *,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        V = jnp.atleast_1d(jnp.asarray(V_mV, dtype=self.dtype))
        gates_array = jnp.asarray(gates, dtype=self.dtype)
        outputs = self._call(
            "model_step",
            self._state_env(
                gates_array,
                state=state,
                V=V,
                parameters=parameters,
            ),
            node_count=V.shape[0],
        )
        return _stack_columns(
            tuple(outputs[name] for name in self.contract.current_output_names),
            V.shape[0],
            self.dtype,
        )

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
        gates_array = jnp.asarray(gates, dtype=self.dtype)
        node_count = int(gates_array.shape[0]) if gates_array.ndim else 1
        outputs = self._call(
            "membrane_terms",
            self._state_env(gates_array, state=state, parameters=parameters),
            node_count=node_count,
        )
        names = self.contract.function("membrane_terms").outputs
        conductance = _stack_columns(
            tuple(outputs[name] for name in names[0::2]), node_count, self.dtype
        )
        reversal = _stack_columns(
            tuple(outputs[name] for name in names[1::2]), node_count, self.dtype
        )
        return jnp.sum(conductance, axis=1), jnp.sum(
            conductance * reversal, axis=1
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
        previous = jnp.asarray(gates_prev, dtype=self.dtype)
        predictor = jnp.asarray(gates_new, dtype=self.dtype)
        ion = _as_node_vector(I_ion, V.shape[0], self.dtype)
        background = _as_node_vector(I_background, V.shape[0], self.dtype)
        if not self.contract.has_step_program:
            return (
                tuple(jnp.asarray(value, dtype=self.dtype) for value in state),
                predictor,
                background + ion,
                background,
                jnp.zeros((V.shape[0],), dtype=self.dtype),
            )

        prepare_gates = (
            previous
            if self.contract.prepare_gate_source == "previous"
            else predictor
        )
        prepare_env = self._step_env(
            V,
            prepare_gates,
            state=state,
            dt_ms=dt_ms,
            I_ion=ion,
            I_background=background,
            parameters=parameters,
            required_names=self.contract.function("prepare_state").args,
        )
        term_env = self._step_env(
            V,
            predictor,
            state=state,
            dt_ms=dt_ms,
            I_ion=ion,
            I_background=background,
            parameters=parameters,
            required_names=self.contract.function("step_current_terms").args,
        )
        prepared = self._merge_state(
            state,
            self._call("prepare_state", prepare_env, node_count=V.shape[0]),
            node_count=V.shape[0],
        )
        terms = self._call(
            "step_current_terms", term_env, node_count=V.shape[0]
        )
        linearization_gates = (
            previous
            if self.contract.linearization_gate_source == "previous"
            else predictor
        )
        return (
            prepared,
            linearization_gates,
            terms.get("total_outward_current", background + ion),
            terms.get("explicit_outward_current", background),
            terms.get(
                "correction_current",
                jnp.zeros((V.shape[0],), dtype=self.dtype),
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
        _ = gates_prev, state_prev
        spec = self.contract.function("finalize_state")
        if not spec.outputs:
            return tuple(jnp.asarray(value, dtype=self.dtype) for value in prepared_state)
        V_previous = jnp.atleast_1d(jnp.asarray(V_mV_prev, dtype=self.dtype))
        V_new = jnp.atleast_1d(jnp.asarray(V_mV_new, dtype=self.dtype))
        gates = jnp.asarray(gates_new, dtype=self.dtype)
        ion = (
            self.currents(V_new, gates, state=prepared_state)
            if "I_ion" in spec.args
            else jnp.zeros((V_new.shape[0],), dtype=self.dtype)
        )
        env = self._step_env(
            V_new,
            gates,
            state=prepared_state,
            dt_ms=dt_ms,
            I_ion=ion,
            I_background=jnp.zeros((V_new.shape[0],), dtype=self.dtype),
            parameters=parameters,
            V_prev=V_previous,
            required_names=spec.args,
        )
        return self._merge_state(
            prepared_state,
            self._call("finalize_state", env, node_count=V_new.shape[0]),
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
        _ = gates_prev, state_prev, state_new
        spec = self.contract.function("diagnostics")
        if not spec.outputs:
            return ()
        V_previous = jnp.atleast_1d(jnp.asarray(V_mV_prev, dtype=self.dtype))
        V_new = jnp.atleast_1d(jnp.asarray(V_mV_new, dtype=self.dtype))
        env = self._step_env(
            V_new,
            jnp.asarray(gates_new, dtype=self.dtype),
            state=prepared_state,
            dt_ms=0.0,
            I_ion=I_ion,
            I_background=jnp.zeros((V_new.shape[0],), dtype=self.dtype),
            parameters=parameters,
            V_prev=V_previous,
            required_names=spec.args,
        )
        outputs = self._call("diagnostics", env, node_count=V_new.shape[0])
        return tuple(outputs[name] for name in self.contract.diagnostic_names)

    def reversal_values(self) -> jnp.ndarray:
        outputs = self._call("reversal_terms", self.with_parameters(), node_count=1)
        names = self.contract.function("reversal_terms").outputs
        if not names:
            return jnp.zeros((0,), dtype=self.dtype)
        return jnp.stack([outputs[name][0] for name in names]).astype(self.dtype)

    def observable_matrix(
        self,
        name: str,
        gates: Any,
        *,
        V_mV: Any | None = None,
        state: tuple[Any, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        observable_names = tuple(value.name for value in self.contract.observables)
        try:
            index = observable_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        output_name = self.contract.observable_output_names[index]
        gates_array = jnp.asarray(gates, dtype=self.dtype)
        node_count = int(gates_array.shape[0]) if gates_array.ndim else 1
        spec = self.contract.function("model_step")
        if "Vm" in spec.args and V_mV is None:
            raise ValueError(f"Observable {name!r} requires membrane voltage.")
        outputs = self._call(
            "model_step",
            self._state_env(
                gates_array,
                state=state,
                V=V_mV,
                parameters=parameters,
            ),
            node_count=node_count,
        )
        return outputs[output_name]

    @property
    def model_current_names(self) -> tuple[str, ...]:
        return self.contract.raw_current_names

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
        if len(state) != len(self.membrane_state_names):
            raise ValueError(
                f"Expected {len(self.membrane_state_names)} membrane states, "
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
        parameters: dict[str, Any] | None,
        required_names: tuple[str, ...],
        V_prev: jnp.ndarray | None = None,
    ) -> dict[str, Any]:
        env = self._state_env(gates, state=state, V=V, parameters=parameters)
        env.update(
            Vm_new=V,
            Vm_prev=V if V_prev is None else V_prev,
            dt=jnp.asarray(dt_ms, dtype=self.dtype),
            I_ion=_as_node_vector(I_ion, V.shape[0], self.dtype),
            I_background=_as_node_vector(I_background, V.shape[0], self.dtype),
        )
        required_currents = set(self.model_current_names).intersection(required_names)
        if required_currents:
            currents = self.current_matrix(V, gates, state=state, parameters=parameters)
            for index, name in enumerate(self.model_current_names):
                if name not in required_currents:
                    continue
                env[name] = env.get(name, 0.0) + currents[:, index]
        return env

    def _merge_state(
        self,
        state: tuple[Any, ...],
        updates: dict[str, jnp.ndarray],
        *,
        node_count: int,
    ) -> tuple[jnp.ndarray, ...]:
        values = {
            name: _as_node_vector(value, node_count, self.dtype)
            for name, value in zip(self.membrane_state_names, state, strict=True)
        }
        values.update(updates)
        return tuple(values[name] for name in self.membrane_state_names)

    def _call(
        self,
        name: str,
        env: dict[str, Any],
        *,
        node_count: int,
    ) -> dict[str, jnp.ndarray]:
        spec = self.contract.function(name)
        missing = tuple(argument for argument in spec.args if argument not in env)
        if missing:
            raise ValueError(f"Generated {name!r} arguments are unavailable: {missing!r}.")
        function = getattr(self.module, name, None)
        if not callable(function):
            raise TypeError(f"Generated membrane module has no {name!r} function.")
        raw = function(*(env[argument] for argument in spec.args))
        values = raw if isinstance(raw, tuple) else (raw,)
        if len(values) != len(spec.outputs):
            raise ValueError(
                f"Generated {name!r} returned {len(values)} values; "
                f"expected {len(spec.outputs)}."
            )
        return {
            output: _as_node_vector(value, node_count, self.dtype)
            for output, value in zip(spec.outputs, values, strict=True)
        }


def _as_node_vector(value: Any, node_count: int, dtype: jnp.dtype) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape == ():
        return jnp.full((node_count,), array, dtype=dtype)
    if array.shape == (node_count,):
        return array
    return jnp.broadcast_to(array, (node_count,)).astype(dtype)


def _stack_columns(
    values: tuple[jnp.ndarray, ...],
    node_count: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if not values:
        return jnp.zeros((node_count, 0), dtype=dtype)
    return jnp.stack(values, axis=1).astype(dtype)


def _normalize_probabilities(values: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    clipped = jnp.maximum(values, jnp.asarray(0.0, dtype=dtype))
    total = jnp.sum(clipped, axis=1, keepdims=True)
    return clipped / jnp.maximum(total, jnp.asarray(1e-12, dtype=dtype))


__all__ = ["GeneratedJaxMembraneLowering"]
