"""Result containers returned by high-level protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TextIO

import numpy as np

from axonfleet.analysis import AnalysisResult, AnalysisStatus
from axonfleet.protocols.types import ThresholdStatus
from axonfleet.utils import units


@dataclass(frozen=True)
class RecruitmentCurve:
    """Recruitment fraction versus stimulus amplitude."""

    amplitudes_uA: np.ndarray
    activated: np.ndarray
    row_labels: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        amplitudes = np.asarray(self.amplitudes_uA, dtype=float)
        activated = np.asarray(self.activated, dtype=bool)
        if activated.ndim != 2:
            raise ValueError("activated must have shape (value, row).")
        if amplitudes.shape != (activated.shape[0],):
            raise ValueError("amplitudes_uA must contain one value per activation row.")
        row_labels = tuple(self.row_labels)
        if not row_labels:
            row_labels = tuple(range(activated.shape[1]))
        if len(row_labels) != activated.shape[1]:
            raise ValueError("row_labels must be empty or match recruited row count.")
        object.__setattr__(self, "amplitudes_uA", amplitudes)
        object.__setattr__(self, "activated", activated)
        object.__setattr__(self, "row_labels", row_labels)

    @property
    def amplitudes(self) -> Any:
        """Sweep amplitudes as a Pint quantity array."""

        return units.Q_(self.amplitudes_uA, "microampere")

    @property
    def count(self) -> np.ndarray:
        """Number of activated fibers at each amplitude."""

        return np.sum(self.activated, axis=1)

    @property
    def fraction(self) -> np.ndarray:
        """Activated fraction at each amplitude."""

        if self.activated.shape[1] == 0:
            return np.zeros(self.activated.shape[0], dtype=float)
        return self.count / float(self.activated.shape[1])

    @property
    def first_activation_uA(self) -> np.ndarray:
        """First sampled activating amplitude for each row, or NaN."""

        values = np.full(self.activated.shape[1], np.nan, dtype=float)
        for row_index in range(self.activated.shape[1]):
            active_rows = np.flatnonzero(self.activated[:, row_index])
            if active_rows.size:
                values[row_index] = float(self.amplitudes_uA[active_rows[0]])
        return values

    def to_analysis_result(
        self,
        *,
        name: str = "first_activation_amplitude",
    ) -> AnalysisResult:
        """Return per-row first activation amplitudes as an analysis metric."""

        values = self.first_activation_uA
        statuses = tuple(
            AnalysisStatus.VALID if np.isfinite(value) else AnalysisStatus.UNDETERMINED
            for value in values
        )
        messages = tuple(
            "" if status is AnalysisStatus.VALID else "row never activated over sampled values."
            for status in statuses
        )
        return AnalysisResult(
            name=name,
            values=values,
            statuses=statuses,
            messages=messages,
            unit="microampere",
            row_labels=self.row_labels,
            definition=self,
        )

    def rows(self, *, unit: Any = "microampere") -> tuple[dict[str, Any], ...]:
        """Return row dictionaries for dataframe/text views."""

        from axonfleet.protocols.views import recruitment_curve_rows

        return recruitment_curve_rows(self, unit=unit)

    def format(self, *, unit: Any = "microampere") -> str:
        """Return a compact text representation."""

        from axonfleet.protocols.views import format_recruitment_curve

        return format_recruitment_curve(self, unit=unit)

    def print(self, file: TextIO | None = None, *, unit: Any = "microampere") -> None:
        """Print a compact text representation."""

        from axonfleet.protocols.views import print_recruitment_curve

        print_recruitment_curve(self, file=file, unit=unit)

    def to_dataframe(self, *, unit: Any = "microampere") -> Any:
        """Return a pandas DataFrame summary."""

        from axonfleet.protocols.views import recruitment_curve_to_dataframe

        return recruitment_curve_to_dataframe(self, unit=unit)

    def plot(
        self,
        ax: Any | None = None,
        *,
        unit: Any = "microampere",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot recruitment fraction versus amplitude."""

        from axonfleet.protocols.views import plot_recruitment_curve

        return plot_recruitment_curve(self, ax=ax, unit=unit, **plot_kwargs)

    def plot_groups(
        self,
        groups: Any,
        ax: Any | None = None,
        *,
        unit: Any = "microampere",
        include_total: bool = True,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot recruitment fractions for row groups."""

        from axonfleet.protocols.views import plot_recruitment_groups

        return plot_recruitment_groups(
            self,
            groups,
            ax=ax,
            unit=unit,
            include_total=include_total,
            **plot_kwargs,
        )


@dataclass(frozen=True)
class PoolSweepResult:
    """Generic per-row observations over a swept parameter.

    This result is intentionally agnostic to what was observed: activation,
    peak voltage, latency, charge, energy, or any other scalar/object returned
    by the user-provided observer.
    """

    values: tuple[Any, ...]
    observations: np.ndarray
    row_labels: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations)
        if observations.ndim < 2:
            raise ValueError("observations must have at least value and row axes.")
        if len(self.values) != observations.shape[0]:
            raise ValueError("values must contain one entry per observation value row.")
        row_labels = tuple(self.row_labels)
        if not row_labels:
            row_labels = tuple(range(observations.shape[1]))
        if len(row_labels) != observations.shape[1]:
            raise ValueError("row_labels must be empty or match observation row count.")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "row_labels", row_labels)

    @property
    def n_values(self) -> int:
        """Number of sampled parameter values."""

        return len(self.values)

    @property
    def n_rows(self) -> int:
        """Number of simulated pool rows."""

        if self.observations.ndim < 2:
            return 0
        return int(self.observations.shape[1])

    def value_values(self, *, unit: Any | None = None) -> np.ndarray:
        """Return swept values as floats when they are unit-bearing scalars."""

        if unit is None:
            return np.arange(len(self.values), dtype=float)
        unit_label = units.unit_label(unit) or str(unit)
        return units.to_array(list(self.values), unit_label, dtype=float)

    def rows(
        self,
        *,
        value_name: str = "value",
        value_unit: Any | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return row dictionaries for dataframe/text views."""

        from axonfleet.protocols.views import pool_sweep_rows

        return pool_sweep_rows(
            self,
            value_name=value_name,
            value_unit=value_unit,
        )

    def format(
        self,
        *,
        value_name: str = "value",
        value_unit: Any | None = None,
    ) -> str:
        """Return a compact text representation."""

        from axonfleet.protocols.views import format_pool_sweep

        return format_pool_sweep(
            self,
            value_name=value_name,
            value_unit=value_unit,
        )

    def print(
        self,
        file: TextIO | None = None,
        *,
        value_name: str = "value",
        value_unit: Any | None = None,
    ) -> None:
        """Print a compact text representation."""

        from axonfleet.protocols.views import print_pool_sweep

        print_pool_sweep(
            self,
            file=file,
            value_name=value_name,
            value_unit=value_unit,
        )

    def to_dataframe(
        self,
        *,
        value_name: str = "value",
        value_unit: Any | None = None,
    ) -> Any:
        """Return a pandas DataFrame summary."""

        from axonfleet.protocols.views import pool_sweep_to_dataframe

        return pool_sweep_to_dataframe(
            self,
            value_name=value_name,
            value_unit=value_unit,
        )

    def plot(
        self,
        ax: Any | None = None,
        *,
        value_unit: Any | None = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot scalar numeric observations by row."""

        from axonfleet.protocols.views import plot_pool_sweep

        return plot_pool_sweep(
            self,
            ax=ax,
            value_unit=value_unit,
            **plot_kwargs,
        )


@dataclass(frozen=True)
class ThresholdCurve:
    """Per-row thresholds estimated by batched binary search.

    ``status`` contains threshold-search outcomes, not ``AnalysisStatus``
    validity values.
    """

    row_labels: tuple[Any, ...]
    threshold_uA: np.ndarray
    lower_bound_uA: np.ndarray
    upper_bound_uA: np.ndarray
    status: tuple[ThresholdStatus, ...]
    tested_uA: tuple[np.ndarray, ...]
    satisfied: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        row_labels = tuple(self.row_labels)
        threshold_uA = np.asarray(self.threshold_uA, dtype=float)
        lower_bound_uA = np.asarray(self.lower_bound_uA, dtype=float)
        upper_bound_uA = np.asarray(self.upper_bound_uA, dtype=float)
        status = tuple(self.status)
        row_count = len(row_labels)
        if threshold_uA.shape != (row_count,):
            raise ValueError("threshold_uA must contain one value per row label.")
        if lower_bound_uA.shape != (row_count,) or upper_bound_uA.shape != (row_count,):
            raise ValueError("threshold bounds must contain one value per row label.")
        if len(status) != row_count:
            raise ValueError("status must contain one value per row label.")
        object.__setattr__(self, "row_labels", row_labels)
        object.__setattr__(self, "threshold_uA", threshold_uA)
        object.__setattr__(self, "lower_bound_uA", lower_bound_uA)
        object.__setattr__(self, "upper_bound_uA", upper_bound_uA)
        object.__setattr__(self, "status", status)

    @property
    def threshold(self) -> Any:
        """Threshold amplitudes as a Pint quantity array."""

        return units.Q_(self.threshold_uA, "microampere")

    @property
    def lower_bound(self) -> Any:
        """Final inactive lower bounds as a Pint quantity array."""

        return units.Q_(self.lower_bound_uA, "microampere")

    @property
    def upper_bound(self) -> Any:
        """Final active upper bounds as a Pint quantity array."""

        return units.Q_(self.upper_bound_uA, "microampere")

    @property
    def n_iterations(self) -> int:
        """Number of batched activation evaluations."""

        return len(self.tested_uA)

    def row_values(self, *, unit: Any | None = None) -> np.ndarray:
        """Return row labels as floats when they are unit-bearing scalars."""

        if unit is None:
            return np.arange(len(self.row_labels), dtype=float)
        unit_label = units.unit_label(unit) or str(unit)
        return units.to_array(list(self.row_labels), unit_label, dtype=float)

    def rows(
        self,
        *,
        row_name: str = "row",
        row_unit: Any | None = None,
        threshold_unit: Any = "microampere",
    ) -> tuple[dict[str, Any], ...]:
        """Return row dictionaries for dataframe/text views."""

        from axonfleet.protocols.views import threshold_curve_rows

        return threshold_curve_rows(
            self,
            row_name=row_name,
            row_unit=row_unit,
            threshold_unit=threshold_unit,
        )

    def to_analysis_result(
        self,
        *,
        name: str = "threshold",
    ) -> AnalysisResult:
        """Return per-row threshold amplitudes as an analysis metric."""

        converted = [
            _threshold_status_as_analysis(status, value)
            for status, value in zip(self.status, self.threshold_uA, strict=True)
        ]
        values = np.asarray([value for value, _, _ in converted], dtype=float)
        statuses = tuple(status for _, status, _ in converted)
        messages = tuple(message for _, _, message in converted)
        return AnalysisResult(
            name=name,
            values=values,
            statuses=statuses,
            messages=messages,
            unit="microampere",
            row_labels=self.row_labels,
            definition=self,
        )

    def format(
        self,
        *,
        row_name: str = "row",
        row_unit: Any | None = None,
        threshold_unit: Any = "microampere",
    ) -> str:
        """Return a compact text representation."""

        from axonfleet.protocols.views import format_threshold_curve

        return format_threshold_curve(
            self,
            row_name=row_name,
            row_unit=row_unit,
            threshold_unit=threshold_unit,
        )

    def print(
        self,
        file: TextIO | None = None,
        *,
        row_name: str = "row",
        row_unit: Any | None = None,
        threshold_unit: Any = "microampere",
    ) -> None:
        """Print a compact text representation."""

        from axonfleet.protocols.views import print_threshold_curve

        print_threshold_curve(
            self,
            file=file,
            row_name=row_name,
            row_unit=row_unit,
            threshold_unit=threshold_unit,
        )

    def to_dataframe(
        self,
        *,
        row_name: str = "row",
        row_unit: Any | None = None,
        threshold_unit: Any = "microampere",
    ) -> Any:
        """Return a pandas DataFrame summary when pandas is installed."""

        from axonfleet.protocols.views import threshold_curve_to_dataframe

        return threshold_curve_to_dataframe(
            self,
            row_name=row_name,
            row_unit=row_unit,
            threshold_unit=threshold_unit,
        )

    def plot(
        self,
        ax: Any | None = None,
        *,
        row_unit: Any | None = None,
        threshold_unit: Any = "microampere",
        **plot_kwargs: Any,
    ) -> Any:
        """Plot threshold versus row value."""

        from axonfleet.protocols.views import plot_threshold_curve

        return plot_threshold_curve(
            self,
            ax=ax,
            row_unit=row_unit,
            threshold_unit=threshold_unit,
            **plot_kwargs,
        )


__all__ = [
    "PoolSweepResult",
    "RecruitmentCurve",
    "ThresholdCurve",
]


def _threshold_status_as_analysis(
    status: ThresholdStatus,
    value_uA: float | None,
) -> tuple[float, AnalysisStatus, str]:
    if status == "threshold" and value_uA is not None and np.isfinite(value_uA):
        return float(value_uA), AnalysisStatus.VALID, ""
    if status == "below_range":
        return np.nan, AnalysisStatus.UNDETERMINED, "threshold is below supplied bounds."
    if status == "above_range":
        return np.nan, AnalysisStatus.UNDETERMINED, "threshold is above supplied bounds."
    return np.nan, AnalysisStatus.UNDETERMINED, f"threshold status {status!r} is not resolved."
