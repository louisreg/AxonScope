"""Solver-facing contracts derived from Model IR."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import LinearizationGateSource, ModelIR


@dataclass(frozen=True, slots=True)
class OutputPruningPlan:
    """Outputs a backend must retain for one lowered model step."""

    retain_state: tuple[str, ...]
    retain_observables: tuple[str, ...]
    retain_currents: tuple[str, ...] = ()
    retain_conductances: tuple[str, ...] = ()
    retain_gates: tuple[str, ...] = ()
    retain_recorded_state: tuple[str, ...] = ()
    retain_diagnostics: tuple[str, ...] = ()
    solver_output_names: tuple[str, ...] = ()
    recording_output_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelStepContract:
    """Visible membrane terms required by fused cable/membrane programs."""

    state_updates: tuple[str, ...]
    total_outward_current: str
    total_conductance: str
    conductance_reversal_sum: str
    explicit_outward_current: str
    correction_current: str
    linearization_state: tuple[str, ...]
    observables: tuple[str, ...]
    pruning: OutputPruningPlan
    supports_single_cable_fusion: bool
    supports_double_cable_fusion: bool


def derive_model_step_contract(
    model: ModelIR,
    *,
    requested_observables: tuple[str, ...] = (),
    record_gates: bool = False,
    record_currents: bool = False,
    record_conductances: bool = False,
    record_state: bool = False,
    record_diagnostics: bool = False,
) -> ModelStepContract:
    observable_names = {observable.name for observable in model.observables}
    unknown = sorted(set(requested_observables).difference(observable_names))
    if unknown:
        raise ValueError(f"Unknown requested Model IR observables: {unknown!r}")

    state_names = tuple(state.name for state in model.states)
    gate_state_names = tuple(gate.state for gate in model.gates)
    gate_state_set = set(gate_state_names)
    membrane_state_names = tuple(
        state.name for state in model.states if state.name not in gate_state_set
    )
    current_names = tuple(current.name for current in model.currents)
    diagnostic_names = (
        ()
        if model.step_program is None
        else tuple(d.name for d in model.step_program.diagnostics)
    )
    step = model.step_program
    retained_observables = tuple(
        name for name in requested_observables if name in observable_names
    )
    retained_currents = current_names if record_currents else ()
    retained_conductances = current_names if record_conductances else ()
    retained_gates = gate_state_names if record_gates else ()
    retained_recorded_state = membrane_state_names if record_state else ()
    retained_diagnostics = diagnostic_names if record_diagnostics else ()
    solver_output_names = current_names
    recording_output_names = (
        *_grouped_output_names("gates", retained_gates),
        *_grouped_output_names("currents", retained_currents),
        *_grouped_output_names("conductances", retained_conductances),
        *_grouped_output_names("states", retained_recorded_state),
        *_grouped_output_names("observables", retained_observables),
        *_grouped_output_names("diagnostics", retained_diagnostics),
    )
    return ModelStepContract(
        state_updates=state_names,
        total_outward_current=(
            "step.total_outward_current"
            if step is not None and step.total_outward_current is not None
            else "sum(currents) + background_current"
        ),
        total_conductance="sum(current.conductance)",
        conductance_reversal_sum="sum(current.conductance * current.reversal)",
        explicit_outward_current=(
            "step.explicit_outward_current"
            if step is not None and step.explicit_outward_current is not None
            else "background_current"
        ),
        correction_current=(
            "step.correction_current"
            if step is not None and step.correction_current is not None
            else "0"
        ),
        linearization_state=(
            ("previous_gates",)
            if step is not None
            and step.linearization_gate_source is LinearizationGateSource.PREVIOUS
            else state_names
        ),
        observables=retained_observables,
        pruning=OutputPruningPlan(
            retain_state=state_names,
            retain_observables=retained_observables,
            retain_currents=retained_currents,
            retain_conductances=retained_conductances,
            retain_gates=retained_gates,
            retain_recorded_state=retained_recorded_state,
            retain_diagnostics=retained_diagnostics,
            solver_output_names=solver_output_names,
            recording_output_names=recording_output_names,
        ),
        supports_single_cable_fusion=bool(model.currents),
        supports_double_cable_fusion=bool(model.currents),
    )


def _grouped_output_names(group: str, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{group}.{name}" for name in names)
