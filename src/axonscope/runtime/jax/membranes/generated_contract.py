"""Typed loader for target-specific generated JAX membrane contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


MEMBRANE_RUNTIME_CONTRACT_VERSION = "jax_membrane_runtime.v3"


@dataclass(frozen=True, slots=True)
class GeneratedParameterSpec:
    """One parameter entry embedded in a generated runtime module."""

    name: str
    unit: str
    dtype: str
    shape: tuple[int | str, ...]
    role: str
    variability: str
    default: int | float | bool | None


@dataclass(frozen=True, slots=True)
class GeneratedQuantitySpec:
    """Typed quantity metadata for one generated runtime value."""

    name: str
    unit: str
    dtype: str
    shape: tuple[int | str, ...]
    role: str


@dataclass(frozen=True, slots=True)
class GeneratedFunctionSpec:
    """Positional signature of one pure generated runtime entrypoint."""

    args: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratedMembraneContract:
    """Model-specific facts required by the JAX runtime after generation."""

    version: str
    model_name: str
    inputs: tuple[GeneratedQuantitySpec, ...]
    parameters: tuple[GeneratedParameterSpec, ...]
    states: tuple[GeneratedQuantitySpec, ...]
    currents: tuple[GeneratedQuantitySpec, ...]
    observables: tuple[GeneratedQuantitySpec, ...]
    diagnostics: tuple[GeneratedQuantitySpec, ...]
    functions: Mapping[str, GeneratedFunctionSpec]
    gate_state_names: tuple[str, ...]
    gate_update_modes: tuple[str, ...]
    membrane_state_names: tuple[str, ...]
    gate_trace_observable_names: tuple[str, ...]
    gate_names: tuple[str, ...]
    membrane_state_display_names: tuple[str, ...]
    observable_display_names: tuple[str, ...]
    raw_current_names: tuple[str, ...]
    current_output_names: tuple[str, ...]
    observable_output_names: tuple[str, ...]
    current_names: tuple[str, ...]
    current_groups: tuple[tuple[int, ...], ...]
    conductance_names: tuple[str, ...]
    conductance_groups: tuple[tuple[int, ...], ...]
    conductance_parameter_names: tuple[str | None, ...]
    diagnostic_names: tuple[str, ...]
    final_gate_update_mode: str
    has_step_program: bool
    prepare_gate_source: str | None
    linearization_gate_source: str | None
    prepare_state_update_names: tuple[str, ...]
    finalize_state_update_names: tuple[str, ...]
    structural_hash: str
    parameterized_hash: str
    source_provenance: Mapping[str, Any]

    def parameter_defaults(self) -> dict[str, int | float | bool]:
        """Return generated defaults keyed by canonical parameter name."""

        return {
            parameter.name: parameter.default
            for parameter in self.parameters
            if parameter.default is not None
        }

    def parameter_values(
        self,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return defaults with normalized per-instance overrides applied."""

        values: dict[str, Any] = self.parameter_defaults()
        if not overrides:
            return values
        known = {parameter.name for parameter in self.parameters}
        unknown = set(overrides).difference(known)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"Unknown generated membrane parameters: {names}.")
        values.update({str(name): value for name, value in overrides.items()})
        return values

    def function(self, name: str) -> GeneratedFunctionSpec:
        """Return one required generated entrypoint signature."""

        try:
            return self.functions[name]
        except KeyError as exc:
            raise ValueError(
                f"Generated membrane contract has no {name!r} function."
            ) from exc


def load_generated_membrane_contract(
    module: Any,
) -> GeneratedMembraneContract:
    """Load and validate the autonomous runtime contract from one module."""

    raw = getattr(module, "RUNTIME_CONTRACT", None)
    if not isinstance(raw, Mapping):
        raise TypeError("Generated JAX membrane module has no runtime contract.")
    version = str(raw.get("version", ""))
    declared_version = str(getattr(module, "RUNTIME_CONTRACT_VERSION", ""))
    if version != MEMBRANE_RUNTIME_CONTRACT_VERSION:
        raise ValueError(f"Unsupported generated membrane contract {version!r}.")
    if declared_version != version:
        raise ValueError("Generated membrane contract version metadata is inconsistent.")

    parameters = tuple(
        _parameter_spec(entry)
        for entry in _mapping_sequence(raw, "parameters")
    )
    contract = GeneratedMembraneContract(
        version=version,
        model_name=_required_string(raw, "model_name"),
        inputs=tuple(
            _quantity_spec(entry) for entry in _mapping_sequence(raw, "inputs")
        ),
        parameters=parameters,
        states=tuple(
            _quantity_spec(entry) for entry in _mapping_sequence(raw, "states")
        ),
        currents=tuple(
            _quantity_spec(entry) for entry in _mapping_sequence(raw, "currents")
        ),
        observables=tuple(
            _quantity_spec(entry) for entry in _mapping_sequence(raw, "observables")
        ),
        diagnostics=tuple(
            _quantity_spec(entry) for entry in _mapping_sequence(raw, "diagnostics")
        ),
        functions={
            str(name): GeneratedFunctionSpec(
                args=_string_tuple(spec, "args"),
                outputs=_string_tuple(spec, "outputs"),
            )
            for name, spec in _mapping(raw, "functions").items()
            if isinstance(spec, Mapping)
        },
        gate_state_names=_string_tuple(raw, "gate_state_names"),
        gate_update_modes=_string_tuple(raw, "gate_update_modes"),
        membrane_state_names=_string_tuple(raw, "membrane_state_names"),
        gate_trace_observable_names=_string_tuple(
            raw, "gate_trace_observable_names"
        ),
        gate_names=_string_tuple(raw, "gate_names"),
        membrane_state_display_names=_string_tuple(
            raw, "membrane_state_display_names"
        ),
        observable_display_names=_string_tuple(raw, "observable_display_names"),
        raw_current_names=_string_tuple(raw, "raw_current_names"),
        current_output_names=_string_tuple(raw, "current_output_names"),
        observable_output_names=_string_tuple(raw, "observable_output_names"),
        current_names=_string_tuple(raw, "current_names"),
        current_groups=_index_groups(raw, "current_groups"),
        conductance_names=_string_tuple(raw, "conductance_names"),
        conductance_groups=_index_groups(raw, "conductance_groups"),
        conductance_parameter_names=_optional_string_tuple(
            raw, "conductance_parameter_names"
        ),
        diagnostic_names=_string_tuple(raw, "diagnostic_names"),
        final_gate_update_mode=_required_string(raw, "final_gate_update_mode"),
        has_step_program=_required_bool(raw, "has_step_program"),
        prepare_gate_source=_optional_string(raw, "prepare_gate_source"),
        linearization_gate_source=_optional_string(
            raw, "linearization_gate_source"
        ),
        prepare_state_update_names=_string_tuple(
            raw, "prepare_state_update_names"
        ),
        finalize_state_update_names=_string_tuple(
            raw, "finalize_state_update_names"
        ),
        structural_hash=_required_string(raw, "structural_hash"),
        parameterized_hash=_required_string(raw, "parameterized_hash"),
        source_provenance=dict(_mapping(raw, "source_provenance")),
    )
    if len(contract.gate_state_names) != len(contract.gate_update_modes):
        raise ValueError("Generated membrane gate metadata has inconsistent lengths.")
    if len(contract.gate_names) != (
        len(contract.gate_state_names) + len(contract.gate_trace_observable_names)
    ):
        raise ValueError("Generated membrane recorded gate names are inconsistent.")
    if len(contract.raw_current_names) != len(
        contract.conductance_parameter_names
    ):
        raise ValueError("Generated membrane current metadata has inconsistent lengths.")
    required_functions = {
        "init_state",
        "gate_terms",
        "membrane_terms",
        "reversal_terms",
        "model_step",
        "prepare_state",
        "step_current_terms",
        "finalize_state",
        "diagnostics",
    }
    missing_functions = required_functions.difference(contract.functions)
    if missing_functions:
        names = ", ".join(sorted(missing_functions))
        raise ValueError(f"Generated membrane contract is missing functions: {names}.")
    state_names = tuple(value.name for value in contract.states)
    expected_state_names = set(contract.gate_state_names) | set(
        contract.membrane_state_names
    )
    if set(state_names) != expected_state_names or len(state_names) != len(
        expected_state_names
    ):
        raise ValueError("Generated membrane state metadata is inconsistent.")
    if tuple(value.name for value in contract.currents) != contract.raw_current_names:
        raise ValueError("Generated membrane current metadata is inconsistent.")
    if len(contract.current_output_names) != len(contract.currents):
        raise ValueError("Generated membrane current outputs are inconsistent.")
    if len(contract.observable_output_names) != len(contract.observables):
        raise ValueError("Generated membrane observable outputs are inconsistent.")
    if len(contract.observable_display_names) != len(contract.observables):
        raise ValueError("Generated membrane observable names are inconsistent.")
    if len(contract.membrane_state_display_names) != len(
        contract.membrane_state_names
    ):
        raise ValueError("Generated membrane state names are inconsistent.")
    observable_names = {value.name for value in contract.observables}
    if not set(contract.gate_trace_observable_names).issubset(observable_names):
        raise ValueError("Generated membrane gate observables are inconsistent.")
    if tuple(value.name for value in contract.diagnostics) != contract.diagnostic_names:
        raise ValueError("Generated membrane diagnostic metadata is inconsistent.")
    _validate_name_groups(
        contract.current_names,
        contract.current_groups,
        raw_count=len(contract.raw_current_names),
        label="current",
    )
    _validate_name_groups(
        contract.conductance_names,
        contract.conductance_groups,
        raw_count=len(contract.raw_current_names),
        label="conductance",
    )
    if contract.function("gate_terms").outputs != tuple(
        item
        for state_name in contract.gate_state_names
        for item in (
            f"alpha:{state_name}",
            f"beta:{state_name}",
            f"q10:{state_name}",
        )
    ):
        raise ValueError("Generated gate-term signature is inconsistent.")
    if len(contract.function("membrane_terms").outputs) != 2 * len(
        contract.currents
    ):
        raise ValueError("Generated membrane-term signature is inconsistent.")
    if len(contract.function("reversal_terms").outputs) != len(contract.currents):
        raise ValueError("Generated reversal-term signature is inconsistent.")
    if contract.function("model_step").outputs != (
        *contract.current_output_names,
        *contract.observable_output_names,
    ):
        raise ValueError("Generated model-step recording signature is inconsistent.")
    if contract.function("init_state").outputs != contract.membrane_state_names:
        raise ValueError("Generated initial-state signature is inconsistent.")
    if contract.function("prepare_state").outputs != contract.prepare_state_update_names:
        raise ValueError("Generated prepare-state signature is inconsistent.")
    if contract.function("finalize_state").outputs != contract.finalize_state_update_names:
        raise ValueError("Generated finalize-state signature is inconsistent.")
    if contract.function("diagnostics").outputs != contract.diagnostic_names:
        raise ValueError("Generated diagnostic signature is inconsistent.")
    return contract


def _validate_name_groups(
    names: tuple[str, ...],
    groups: tuple[tuple[int, ...], ...],
    *,
    raw_count: int,
    label: str,
) -> None:
    if len(names) != len(groups):
        raise ValueError(f"Generated membrane {label} groups are inconsistent.")
    indices = tuple(index for group in groups for index in group)
    if len(indices) != raw_count or sorted(indices) != list(range(raw_count)):
        raise ValueError(f"Generated membrane {label} groups are not a partition.")


def _parameter_spec(value: Mapping[str, Any]) -> GeneratedParameterSpec:
    return GeneratedParameterSpec(
        name=_required_string(value, "name"),
        unit=_required_string(value, "unit"),
        dtype=_required_string(value, "dtype"),
        shape=tuple(value.get("shape", ())),
        role=_required_string(value, "role"),
        variability=_required_string(value, "variability"),
        default=value.get("default"),
    )


def _quantity_spec(value: Mapping[str, Any]) -> GeneratedQuantitySpec:
    return GeneratedQuantitySpec(
        name=_required_string(value, "name"),
        unit=_required_string(value, "unit"),
        dtype=_required_string(value, "dtype"),
        shape=tuple(value.get("shape", ())),
        role=_required_string(value, "role"),
    )


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"Generated membrane contract requires {key!r}.")
    return result


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str) or not result:
        raise ValueError(f"Generated membrane contract {key!r} must be a string.")
    return result


def _required_bool(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"Generated membrane contract requires boolean {key!r}.")
    return result


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key, {})
    if not isinstance(result, Mapping):
        raise ValueError(f"Generated membrane contract {key!r} must be a mapping.")
    return result


def _mapping_sequence(
    value: Mapping[str, Any],
    key: str,
) -> tuple[Mapping[str, Any], ...]:
    result = tuple(value.get(key, ()))
    if not all(isinstance(entry, Mapping) for entry in result):
        raise ValueError(f"Generated membrane contract {key!r} must contain mappings.")
    return result


def _string_tuple(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    result = tuple(value.get(key, ()))
    if not all(isinstance(entry, str) for entry in result):
        raise ValueError(f"Generated membrane contract {key!r} must contain strings.")
    return result


def _optional_string_tuple(
    value: Mapping[str, Any],
    key: str,
) -> tuple[str | None, ...]:
    result = tuple(value.get(key, ()))
    if not all(entry is None or isinstance(entry, str) for entry in result):
        raise ValueError(
            f"Generated membrane contract {key!r} must contain strings or None."
        )
    return result


def _index_groups(
    value: Mapping[str, Any],
    key: str,
) -> tuple[tuple[int, ...], ...]:
    try:
        return tuple(tuple(int(index) for index in group) for group in value.get(key, ()))
    except TypeError as exc:
        raise ValueError(
            f"Generated membrane contract {key!r} must contain index groups."
        ) from exc


__all__ = [
    "GeneratedMembraneContract",
    "GeneratedFunctionSpec",
    "GeneratedParameterSpec",
    "GeneratedQuantitySpec",
    "MEMBRANE_RUNTIME_CONTRACT_VERSION",
    "load_generated_membrane_contract",
]
