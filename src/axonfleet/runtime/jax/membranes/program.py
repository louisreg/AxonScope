"""JAX execution contract for backend-neutral membrane programs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from axonfleet.runtime.jax.membranes.backend import MembraneStateSpec, MembraneStepPlan
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
        generated_module: Any,
        *,
        parameter_overrides: dict[str, Any],
        dtype_local: jnp.dtype | None = None,
        host_module: Any | None = None,
        codegen_cache: dict[str, Any] | None = None,
        public_model_name: str | None = None,
    ) -> None:
        self.dtype = _normalize_jax_dtype(dtype_local)
        self.generated_contract: GeneratedMembraneContract = (
            load_generated_membrane_contract(generated_module)
        )
        self._host_module = host_module
        parameter_values = self.generated_contract.parameter_values(
            parameter_overrides
        )
        self._parameter_values = dict(parameter_values)
        self.lowering = GeneratedJaxMembraneLowering(
            generated_module,
            self.generated_contract,
            dtype=self.dtype,
            parameter_values=parameter_values,
        )
        self._codegen_cache = dict(codegen_cache or {})
        source_model_name = self.generated_contract.model_name
        self.model_name = source_model_name if public_model_name is None else str(public_model_name)
        self._source_model_name = source_model_name
        self.q10 = self.dtype(self._representative_q10())
        self._static_signature_cache: tuple[Any, ...] | None = None
        self._g_bar_cache: jnp.ndarray | None = None
        self._e_rev_cache: jnp.ndarray | None = None
        self._membrane_state_specs_cache: tuple[MembraneStateSpec, ...] | None = None

    @classmethod
    def from_generated_module(
        cls,
        generated_module: Any,
        *,
        parameter_overrides: dict[str, Any],
        dtype_local: jnp.dtype | None = None,
        host_module: Any | None = None,
        codegen_cache: dict[str, Any] | None = None,
        public_model_name: str | None = None,
    ) -> "JaxMembraneProgram":
        """Build directly from an autonomous generated runtime artifact."""

        return cls(
            generated_module,
            dtype_local=dtype_local,
            host_module=host_module,
            parameter_overrides=parameter_overrides,
            codegen_cache=codegen_cache,
            public_model_name=public_model_name,
        )

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
            self.model_name,
            self._final_gate_update_mode,
        )
        self._static_signature_cache = signature
        return signature

    def execution_structure_signature(self) -> tuple[Any, ...]:
        """Return generated execution identity without numeric parameters."""

        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            str(self.dtype),
            self._structural_hash,
            self._source_model_name,
            self._final_gate_update_mode,
        )

    @property
    def source_provenance(self) -> dict[str, Any]:
        return dict(self.generated_contract.source_provenance)

    @property
    def codegen_cache(self) -> dict[str, Any]:
        return dict(self._codegen_cache)

    def generated_target_path(self, target: str) -> Path | None:
        """Return one available generated runtime artifact without loading it."""

        cache = self.codegen_cache
        directory = cache.get("directory")
        file_name = {
            "jax": "jax_model.py",
            "numpy": "numpy_model.py",
            "triton": "triton_model.py",
        }.get(str(target))
        if not isinstance(directory, str) or file_name is None:
            return None
        if file_name not in cache.get("files", ()):
            return None
        path = Path(directory) / file_name
        return path if path.is_file() else None

    @property
    def parameter_values(self) -> dict[str, Any]:
        """Return runtime parameter values for generated target adapters."""

        return dict(self._parameter_values)

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

    def alpha_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        alpha, _ = self.exact_rate_constants(V)
        return alpha

    def beta_funcs(self, V: jnp.ndarray) -> jnp.ndarray:
        _, beta = self.exact_rate_constants(V)
        return beta

    def init_gates(
        self,
        V0_mV: jnp.ndarray,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        return self.lowering.init_gates(V0_mV, parameters=parameters)

    def init_gates_host(
        self,
        V0_mV: np.ndarray,
        *,
        dtype_local: np.dtype,
        parameters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Initialize gates through the generated NumPy companion artifact."""

        if self._host_module is None:
            raise ValueError("Generated NumPy membrane module is unavailable.")
        V = np.atleast_1d(np.asarray(V0_mV, dtype=dtype_local))
        if not self.generated_contract.gate_state_names:
            return np.zeros((V.shape[0], 0), dtype=dtype_local)
        spec = self.generated_contract.function("gate_terms")
        parameter_values = self._parameter_values if parameters is None else parameters
        env = {
            name: np.asarray(value, dtype=dtype_local)
            for name, value in parameter_values.items()
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
        hh_count = len(self.generated_contract.hh_gate_state_names)
        if hh_count:
            alpha = np.stack(
                [np.broadcast_to(value, V.shape) for value in values[0::3]], axis=1
            ).astype(dtype_local)
            beta = np.stack(
                [np.broadcast_to(value, V.shape) for value in values[1::3]], axis=1
            ).astype(dtype_local)
            hh = alpha / np.maximum(
                alpha + beta,
                np.asarray(1e-12, dtype=dtype_local),
            )
        else:
            hh = np.zeros((V.shape[0], 0), dtype=dtype_local)
        kinetics = self._init_kinetic_states_host(V, env=env, dtype_local=dtype_local)
        return np.concatenate((hh, kinetics), axis=1)

    def _init_kinetic_states_host(
        self,
        V: np.ndarray,
        *,
        env: dict[str, np.ndarray],
        dtype_local: np.dtype,
    ) -> np.ndarray:
        contract = self.generated_contract
        if self._host_module is None or not contract.kinetic_blocks:
            return np.zeros((V.shape[0], 0), dtype=dtype_local)

        def outputs(function_name: str) -> dict[str, np.ndarray]:
            function_spec = contract.function(function_name)
            raw_values = getattr(self._host_module, function_name)(
                *(env[name] for name in function_spec.args)
            )
            values = raw_values if isinstance(raw_values, tuple) else (raw_values,)
            return {
                name: np.broadcast_to(value, V.shape).astype(dtype_local)
                for name, value in zip(function_spec.outputs, values, strict=True)
            }

        rates = outputs("kinetic_terms")
        declared = outputs("kinetic_initials")
        columns: list[np.ndarray] = []
        for block in contract.kinetic_blocks:
            width = len(block.states)
            if block.initialization == "stationary":
                matrix = np.zeros((V.shape[0], width, width), dtype=dtype_local)
                for transition in block.transitions:
                    rate = rates[transition.output]
                    matrix[:, transition.target, transition.source] += rate
                    matrix[:, transition.source, transition.source] -= rate
                matrix[:, -1, :] = np.asarray(1.0, dtype=dtype_local)
                rhs = np.zeros((V.shape[0], width), dtype=dtype_local)
                rhs[:, -1] = np.asarray(1.0, dtype=dtype_local)
                block_values = np.linalg.solve(matrix, rhs[..., None])[..., 0]
            else:
                block_values = np.stack(
                    [declared[name] for name in block.states], axis=1
                )
            if block.conserve_probability:
                block_values = np.maximum(block_values, 0.0)
                block_values /= np.maximum(
                    np.sum(block_values, axis=1, keepdims=True),
                    np.asarray(1e-12, dtype=dtype_local),
                )
            columns.extend(block_values[:, index] for index in range(width))
        return np.stack(columns, axis=1).astype(dtype_local)

    def cn_gate_update(
        self,
        g_prev: jnp.ndarray,
        V_mV: jnp.ndarray,
        dt: float,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray:
        return self.lowering.gate_update(
            g_prev,
            V_mV,
            dt,
            parameters=parameters,
        )

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
        *,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.lowering.membrane_conductance_terms(
            gates,
            state=state,
            parameters=parameters,
        )

    def gate_names(self) -> tuple[str, ...]:
        return self._public_names(self.generated_contract.gate_names)

    def conductance_names(self) -> tuple[str, ...]:
        return self.generated_contract.conductance_names

    def current_names(self) -> tuple[str, ...]:
        return self.generated_contract.current_names

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
            return jnp.zeros((0, 0), dtype=self.dtype)
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
        return _aggregate_columns(
            self.conductances(gates, state),
            self._conductance_groups,
        )

    def gate_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
        V_mV: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        base = jnp.asarray(gates, dtype=self.dtype)
        if not self._gate_trace_observable_names:
            return base
        extras = [
            self.lowering.observable_matrix(name, base, V_mV=V_mV, state=state)
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
        return _aggregate_columns(raw_currents, self._current_groups)

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
        return self.generated_contract.diagnostic_names

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
        if not self.generated_contract.hh_gate_state_names:
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
        if q10.shape[1] == 0:
            return 1.0
        first = float(q10[0, 0])
        if not all(abs(float(value) - first) < 1e-12 for value in q10[0]):
            raise ValueError("JaxMembraneProgram requires a common q10 across gates.")
        return first

    @property
    def _conductance_parameter_names(self) -> tuple[str | None, ...]:
        return self.generated_contract.conductance_parameter_names

    @property
    def _final_gate_update_mode(self) -> str:
        return self.generated_contract.final_gate_update_mode

    @property
    def _has_step_program(self) -> bool:
        return self.generated_contract.has_step_program

    @property
    def _membrane_state_display_names(self) -> tuple[str, ...]:
        return self._public_names(self.generated_contract.membrane_state_display_names)

    def _public_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        if self.model_name == self._source_model_name:
            return names
        prefix = f"{self._source_model_name}."
        return tuple(
            f"{self.model_name}.{name.removeprefix(prefix)}"
            if name.startswith(prefix)
            else name
            for name in names
        )

    @property
    def _structural_hash(self) -> str:
        return self.generated_contract.structural_hash

    @property
    def _parameterized_hash(self) -> str:
        return self.generated_contract.parameterized_hash

    @property
    def _conductance_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.generated_contract.conductance_groups

    @property
    def _current_groups(self) -> tuple[tuple[int, ...], ...]:
        return self.generated_contract.current_groups

    @property
    def _gate_trace_observable_names(self) -> tuple[str, ...]:
        return self.generated_contract.gate_trace_observable_names


def _normalize_jax_dtype(dtype_local: Any | None) -> jnp.dtype:
    if dtype_local is None:
        return jnp.float32
    resolved = np.dtype(dtype_local)
    if resolved == np.dtype("float64"):
        return jnp.float64
    return jnp.float32


def _parameter_signature_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return (str(array.dtype), tuple(array.shape), array.tobytes())


def _aggregate_columns(
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
