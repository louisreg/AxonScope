"""Public analysis contracts, statuses, and result containers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

from axonscope.signals import Signal


class AnalysisStatus(str, Enum):
    """Per-axon status for one post-hoc or online analysis metric."""

    VALID = "VALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING_INPUT = "MISSING_INPUT"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True)
class AnalysisRequirements:
    """Declarative input contract for one analysis definition."""

    required_signals: tuple[Signal[Any], ...]
    required_result_fields: tuple[str, ...] = ("Vm", "t", "positions")
    supported_myelination: tuple[str, ...] = ("unmyelinated", "myelinated")
    supported_formulations: tuple[str, ...] = ("single-cable", "double-cable")
    required_capabilities: tuple[str, ...] = ()
    required_compartment_roles: tuple[str, ...] = ()
    required_positions: tuple[Any, ...] = ()
    posthoc_supported: bool = True
    online_supported: bool = False
    algorithm_version: str = "unspecified"
    recording_hint: str | None = None


@dataclass(frozen=True)
class AnalysisInputRequirement:
    """Concrete missing input requirement reported for one failed row."""

    message: str
    required_signals: tuple[Signal[Any], ...] = ()
    required_result_fields: tuple[str, ...] = ()
    required_positions: tuple[Any, ...] = ()
    recording_hint: str | None = None


@dataclass(frozen=True)
class AnalysisPopulation:
    """Population-level denominators for one analysis result."""

    n_total: int
    n_applicable: int
    n_valid: int
    n_failed: int

    @classmethod
    def from_statuses(cls, statuses: Sequence[AnalysisStatus]) -> "AnalysisPopulation":
        """Build denominators from per-axon statuses."""

        total = len(statuses)
        applicable = sum(status is not AnalysisStatus.NOT_APPLICABLE for status in statuses)
        valid = sum(status is AnalysisStatus.VALID for status in statuses)
        failed = sum(
            status in {AnalysisStatus.MISSING_INPUT, AnalysisStatus.NUMERICAL_FAILURE}
            for status in statuses
        )
        return cls(
            n_total=total,
            n_applicable=applicable,
            n_valid=valid,
            n_failed=failed,
        )


@dataclass(frozen=True)
class AnalysisResult:
    """Structured output for one analysis definition over one or more axons."""

    name: str
    values: Any
    statuses: Sequence[AnalysisStatus]
    messages: Sequence[str] = ()
    unit: Any | None = None
    definition: Any | None = None
    events: Sequence[Any | None] = ()
    input_requirements: Sequence[AnalysisInputRequirement | None] = ()

    def __post_init__(self) -> None:
        statuses = tuple(AnalysisStatus(status) for status in self.statuses)
        values = np.asarray(self.values)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.shape[0] != len(statuses):
            raise ValueError("values and statuses must have the same leading length.")

        messages = tuple(str(message) for message in self.messages)
        if not messages:
            messages = ("",) * len(statuses)
        if len(messages) != len(statuses):
            raise ValueError("messages must be empty or match statuses length.")

        events = tuple(self.events)
        if not events:
            events = (None,) * len(statuses)
        if len(events) != len(statuses):
            raise ValueError("events must be empty or match statuses length.")

        requirements = tuple(self.input_requirements)
        if not requirements:
            requirements = (None,) * len(statuses)
        if len(requirements) != len(statuses):
            raise ValueError("input_requirements must be empty or match statuses length.")

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "input_requirements", requirements)

    @property
    def population(self) -> AnalysisPopulation:
        """Population denominators for this analysis."""

        return AnalysisPopulation.from_statuses(self.statuses)

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask selecting valid rows."""

        return np.asarray([status is AnalysisStatus.VALID for status in self.statuses])

    @property
    def status(self) -> AnalysisStatus:
        """Status for a one-axon result."""

        if len(self.statuses) != 1:
            raise ValueError("status is only defined for one-axon analysis results.")
        return self.statuses[0]

    @property
    def value(self) -> Any:
        """Value for a one-axon result."""

        if self.values.shape[0] != 1:
            raise ValueError("value is only defined for one-axon analysis results.")
        value = self.values[0]
        return value.item() if hasattr(value, "item") else value

    @property
    def missing_input_requirements(self) -> tuple[AnalysisInputRequirement, ...]:
        """Structured requirements for rows that could not be analyzed."""

        return tuple(
            requirement
            for status, requirement in zip(self.statuses, self.input_requirements, strict=True)
            if status is AnalysisStatus.MISSING_INPUT and requirement is not None
        )


@dataclass(frozen=True)
class AnalysisReport:
    """Bundle of analysis results associated with one simulation result."""

    simulation_result: Any
    analyses: tuple[AnalysisResult, ...]

    def __len__(self) -> int:
        return len(self.analyses)

    def __iter__(self) -> Iterator[AnalysisResult]:
        return iter(self.analyses)

    def __getitem__(self, key: int | str) -> AnalysisResult:
        if isinstance(key, str):
            for result in self.analyses:
                if result.name == key:
                    return result
            raise KeyError(key)
        return self.analyses[key]

    @property
    def names(self) -> tuple[str, ...]:
        """Analysis names in report order."""

        return tuple(result.name for result in self.analyses)


class AnalysisDefinition(Protocol):
    """Protocol implemented by public analysis definition objects."""

    name: str
    requirements: AnalysisRequirements

    def evaluate(self, result: Any) -> AnalysisResult:
        """Evaluate this definition on a simulation result."""


class MissingAnalysisInputError(ValueError):
    """Raised when a post-hoc analysis lacks a required recorded signal."""

    def __init__(
        self,
        message: str,
        *,
        required_signals: Sequence[Signal[Any]] = (),
        required_result_fields: Sequence[str] = (),
        required_positions: Sequence[Any] = (),
        recording_hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.requirement = AnalysisInputRequirement(
            message=message,
            required_signals=tuple(required_signals),
            required_result_fields=tuple(required_result_fields),
            required_positions=tuple(required_positions),
            recording_hint=recording_hint,
        )

    @property
    def required_signals(self) -> tuple[Signal[Any], ...]:
        """Signals required to satisfy this error."""

        return self.requirement.required_signals


class AnalysisNotApplicableError(ValueError):
    """Raised when an analysis does not apply to a result row."""


def analyze(result: Any, *definitions: AnalysisDefinition) -> AnalysisResult | AnalysisReport:
    """Evaluate one or more analysis definitions on a simulation result."""

    if not definitions:
        raise ValueError("analyze requires at least one analysis definition.")

    analyses: list[AnalysisResult] = []
    for definition in definitions:
        if not hasattr(definition, "evaluate"):
            raise TypeError("analysis definitions must expose an evaluate(result) method.")
        analyses.append(definition.evaluate(result))

    if len(analyses) == 1:
        return analyses[0]
    return AnalysisReport(
        simulation_result=result,
        analyses=tuple(analyses),
    )


__all__ = [
    "AnalysisDefinition",
    "AnalysisInputRequirement",
    "AnalysisNotApplicableError",
    "AnalysisPopulation",
    "AnalysisReport",
    "AnalysisRequirements",
    "AnalysisResult",
    "AnalysisStatus",
    "MissingAnalysisInputError",
    "analyze",
]
