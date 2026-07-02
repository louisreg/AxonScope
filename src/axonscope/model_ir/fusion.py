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
) -> ModelStepContract:
    observable_names = {observable.name for observable in model.observables}
    unknown = sorted(set(requested_observables).difference(observable_names))
    if unknown:
        raise ValueError(f"Unknown requested Model IR observables: {unknown!r}")

    state_names = tuple(state.name for state in model.states)
    current_names = tuple(current.name for current in model.currents)
    step = model.step_program
    retained_observables = tuple(
        name for name in requested_observables if name in observable_names
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
            retain_currents=(),
            retain_conductances=(),
        ),
        supports_single_cable_fusion=bool(model.currents),
        supports_double_cable_fusion=bool(model.currents),
    )
