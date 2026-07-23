"""Progress rendering helpers for protocol execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonfleet.protocols.types import ProgressSummary
from axonfleet.utils import units
from axonfleet.utils.progress_reporting import (
    format_duration,
    memory_summary,
    progress_timestamp,
    runtime_snapshot,
    timing_summary,
)


class _OneShotProgress:
    """Return a progress option once, then disable it for subsequent runs."""

    def __init__(self, progress: bool | str) -> None:
        self.progress = progress
        self._used = False

    def consume(self) -> bool | str:
        if not self.progress or self._used:
            return False
        self._used = True
        return self.progress


def _format_row(row: Any) -> str:
    if units.is_quantity_like(row):
        try:
            return f"{float(row.magnitude):.4g} {row.units:~P}"
        except Exception:
            return str(row)
    return str(row)


def _format_sweep_value(value: Any) -> str:
    if units.is_quantity_like(value):
        try:
            return f"{float(value.magnitude):.4g} {value.units:~P}"
        except Exception:
            return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _activation_progress_summary(row: np.ndarray) -> str:
    activated = np.asarray(row, dtype=bool)
    total = int(activated.shape[0])
    count = int(np.sum(activated))
    fraction = 0.0 if total == 0 else count / float(total)
    return f"{count}/{total} ({fraction:.2f})"


class _ThresholdProgress:
    def __init__(self, progress: bool | str) -> None:
        self.progress = progress
        self.mode = "rich" if progress is True else str(progress)
        self._live: Any | None = None
        self._console: Any | None = None
        self._started = runtime_snapshot()
        self._last_update_s = self._started.perf_counter_s
        self._iteration_durations_s: list[float] = []

    def update(
        self,
        *,
        iteration: str,
        rows: tuple[Any, ...],
        tested_uA: np.ndarray,
        satisfied: np.ndarray,
        lower_bound_uA: np.ndarray,
        upper_bound_uA: np.ndarray,
        status: np.ndarray,
    ) -> None:
        if not self.progress:
            return
        now = runtime_snapshot()
        self._iteration_durations_s.append(
            max(now.perf_counter_s - self._last_update_s, 0.0)
        )
        self._last_update_s = now.perf_counter_s
        if self.mode != "plain":
            try:
                table = self._rich_table(
                    iteration=iteration,
                    rows=rows,
                    tested_uA=tested_uA,
                    satisfied=satisfied,
                    lower_bound_uA=lower_bound_uA,
                    upper_bound_uA=upper_bound_uA,
                    status=status,
                )
                if self._live is None:
                    from rich.console import Console
                    from rich.live import Live

                    self._console = Console()
                    self._live = Live(
                        table,
                        console=self._console,
                        refresh_per_second=8,
                        transient=False,
                    )
                    self._live.start()
                else:
                    self._live.update(table)
                return
            except Exception:
                self.mode = "plain"

        self._plain_update(
            iteration=iteration,
            rows=rows,
            tested_uA=tested_uA,
            satisfied=satisfied,
            lower_bound_uA=lower_bound_uA,
            upper_bound_uA=upper_bound_uA,
            status=status,
        )

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            if self._console is not None:
                self._console.print()
            self._live = None
        if self.progress:
            summary = self._summary(end=runtime_snapshot())
            if self.mode != "plain":
                try:
                    from rich.console import Console

                    console = self._console if self._console is not None else Console()
                    console.print(
                        f"[dim]{progress_timestamp()}[/dim] "
                        f"[bold]Threshold protocol completed[/bold] [dim]{summary}[/dim]"
                    )
                    return
                except Exception:
                    pass
            print(f"{progress_timestamp()} Threshold protocol completed: {summary}", flush=True)

    @staticmethod
    def _rich_table(
        *,
        iteration: str,
        rows: tuple[Any, ...],
        tested_uA: np.ndarray,
        satisfied: np.ndarray,
        lower_bound_uA: np.ndarray,
        upper_bound_uA: np.ndarray,
        status: np.ndarray,
    ) -> Any:
        from rich.table import Table

        table = Table(title=f"Threshold search iteration {iteration} ({progress_timestamp()})")
        table.add_column("row")
        table.add_column("low (uA)", justify="right")
        table.add_column("high (uA)", justify="right")
        table.add_column("test (uA)", justify="right")
        table.add_column("satisfied", justify="center")
        table.add_column("status")
        for row, low, high, tested, is_satisfied, state in zip(
            rows,
            lower_bound_uA,
            upper_bound_uA,
            tested_uA,
            satisfied,
            status,
            strict=True,
        ):
            table.add_row(
                _format_row(row),
                f"{float(low):.3g}",
                f"{float(high):.3g}",
                f"{float(tested):.3g}",
                "yes" if bool(is_satisfied) else "no",
                str(state),
            )
        return table

    @staticmethod
    def _plain_update(
        *,
        iteration: str,
        rows: tuple[Any, ...],
        tested_uA: np.ndarray,
        satisfied: np.ndarray,
        lower_bound_uA: np.ndarray,
        upper_bound_uA: np.ndarray,
        status: np.ndarray,
    ) -> None:
        print("\033[2J\033[H", end="")
        print(f"{progress_timestamp()} Threshold search iteration {iteration}")
        for row, low, high, tested, is_satisfied, state in zip(
            rows,
            lower_bound_uA,
            upper_bound_uA,
            tested_uA,
            satisfied,
            status,
            strict=True,
        ):
            print(
                f"{_format_row(row):>12s}: "
                f"low={float(low):.3g} uA "
                f"high={float(high):.3g} uA "
                f"test={float(tested):.3g} uA "
                f"satisfied={'yes' if bool(is_satisfied) else 'no'} "
                f"status={state}"
            )

    def _summary(self, *, end: Any) -> str:
        return timing_summary(
            start=self._started,
            end=end,
            iteration_durations_s=tuple(self._iteration_durations_s),
        )


class _SweepProgress:
    def __init__(self, progress: bool | str) -> None:
        self.progress = progress
        self.mode = "rich" if progress is True else str(progress)
        self._live: Any | None = None
        self._console: Any | None = None
        self._started = runtime_snapshot()
        self._last_update_s = self._started.perf_counter_s
        self._iteration_durations_s: list[float] = []
        self._batched_solver_elapsed_s: float | None = None
        self._batched_value_count: int | None = None
        self._running_index: int | None = None
        self._running_started_s: float | None = None

    def note_batched_solver(self, *, elapsed_s: float, value_count: int) -> None:
        """Record timing for one solver call that covers every sweep value."""

        self._batched_solver_elapsed_s = max(float(elapsed_s), 0.0)
        self._batched_value_count = max(int(value_count), 1)

    def begin(
        self,
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
    ) -> None:
        if not self.progress:
            return
        now = runtime_snapshot()
        self._running_index = int(current_index)
        self._running_started_s = now.perf_counter_s
        self._render(
            label=label,
            current_index=current_index,
            values=values,
            completed_rows=completed_rows,
            progress_summary=progress_summary,
        )

    def update(
        self,
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
        elapsed_s: float | None = None,
    ) -> None:
        if not self.progress:
            return
        if elapsed_s is None:
            now = runtime_snapshot()
            start_s = (
                self._running_started_s
                if self._running_started_s is not None
                else self._last_update_s
            )
            self._iteration_durations_s.append(max(now.perf_counter_s - start_s, 0.0))
            self._last_update_s = now.perf_counter_s
        else:
            self._iteration_durations_s.append(max(float(elapsed_s), 0.0))
        self._running_index = None
        self._running_started_s = None
        self._render(
            label=label,
            current_index=current_index,
            values=values,
            completed_rows=completed_rows,
            progress_summary=progress_summary,
        )

    def _render(
        self,
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
    ) -> None:
        if self.mode != "plain":
            try:
                table = self._rich_table(
                    label=label,
                    current_index=current_index,
                    values=values,
                    completed_rows=completed_rows,
                    progress_summary=progress_summary,
                    durations_s=tuple(self._iteration_durations_s),
                    running_index=self._running_index,
                    running_started_s=self._running_started_s,
                )
                if self._live is None:
                    from rich.console import Console
                    from rich.live import Live

                    self._console = Console()
                    self._live = Live(
                        table,
                        console=self._console,
                        refresh_per_second=8,
                        transient=False,
                    )
                    self._live.start()
                else:
                    self._live.update(table)
                return
            except Exception:
                self.mode = "plain"

        self._plain_update(
            label=label,
            current_index=current_index,
            values=values,
            completed_rows=completed_rows,
            progress_summary=progress_summary,
            durations_s=tuple(self._iteration_durations_s),
            running_index=self._running_index,
            running_started_s=self._running_started_s,
        )

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            if self._console is not None:
                self._console.print()
            self._live = None
        if self.progress:
            summary = self._summary(end=runtime_snapshot())
            if self.mode != "plain":
                try:
                    from rich.console import Console

                    console = self._console if self._console is not None else Console()
                    console.print(
                        f"[dim]{progress_timestamp()}[/dim] "
                        f"[bold]Protocol sweep completed[/bold] [dim]{summary}[/dim]"
                    )
                    return
                except Exception:
                    pass
            print(f"{progress_timestamp()} Protocol sweep completed: {summary}", flush=True)

    def _summary(self, *, end: Any) -> str:
        if self._batched_solver_elapsed_s is None:
            return timing_summary(
                start=self._started,
                end=end,
                iteration_durations_s=tuple(self._iteration_durations_s),
            )
        value_count = max(1, int(self._batched_value_count or 1))
        total = end.perf_counter_s - self._started.perf_counter_s
        per_value = self._batched_solver_elapsed_s / value_count
        return (
            f"total={format_duration(total)}, "
            f"cold_start={format_duration(self._batched_solver_elapsed_s)}, "
            f"warm=n/a, "
            f"per_iteration={format_duration(per_value)}, "
            f"{memory_summary(self._started, end)}"
        )

    @staticmethod
    def _rich_table(
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
        durations_s: tuple[float, ...],
        running_index: int | None,
        running_started_s: float | None,
    ) -> Any:
        from rich.table import Table

        current = _format_sweep_value(values[current_index])
        table = Table(title=f"{label}, current={current} ({progress_timestamp()})")
        table.add_column("value", justify="right")
        table.add_column("summary", justify="right")
        table.add_column("elapsed", justify="right")
        table.add_column("status", justify="right")
        completed = len(completed_rows)
        now_s = runtime_snapshot().perf_counter_s
        for index, value in enumerate(values):
            if index < completed:
                row = completed_rows[index]
                summary = (
                    progress_summary(row)
                    if progress_summary is not None
                    else f"{int(row.shape[0])} rows"
                )
                table.add_row(
                    _format_sweep_value(value),
                    summary,
                    _format_optional_duration(
                        durations_s[index] if index < len(durations_s) else None
                    ),
                    "done",
                )
            elif index == running_index:
                elapsed = (
                    None
                    if running_started_s is None
                    else max(now_s - running_started_s, 0.0)
                )
                table.add_row(
                    _format_sweep_value(value),
                    "-",
                    _format_optional_duration(elapsed),
                    "running",
                )
            else:
                table.add_row(_format_sweep_value(value), "-", "-", "pending")
        return table

    @staticmethod
    def _plain_update(
        *,
        label: str,
        current_index: int,
        values: tuple[Any, ...],
        completed_rows: list[np.ndarray],
        progress_summary: ProgressSummary | None,
        durations_s: tuple[float, ...],
        running_index: int | None,
        running_started_s: float | None,
    ) -> None:
        print("\033[2J\033[H", end="", flush=True)
        print(
            f"{progress_timestamp()} {label}, current={_format_sweep_value(values[current_index])}",
            flush=True,
        )
        completed = len(completed_rows)
        now_s = runtime_snapshot().perf_counter_s
        for index, value in enumerate(values):
            if index < completed:
                row = completed_rows[index]
                summary = (
                    progress_summary(row)
                    if progress_summary is not None
                    else f"{int(row.shape[0])} rows"
                )
                elapsed = _format_optional_duration(
                    durations_s[index] if index < len(durations_s) else None
                )
                print(
                    f"{_format_sweep_value(value):>12s}: {summary} "
                    f"elapsed={elapsed} done",
                    flush=True,
                )
            elif index == running_index:
                elapsed = (
                    None
                    if running_started_s is None
                    else max(now_s - running_started_s, 0.0)
                )
                print(
                    f"{_format_sweep_value(value):>12s}: running "
                    f"elapsed={_format_optional_duration(elapsed)}",
                    flush=True,
                )
            else:
                print(f"{_format_sweep_value(value):>12s}: pending", flush=True)


def _format_optional_duration(value_s: float | None) -> str:
    if value_s is None:
        return "-"
    return format_duration(float(value_s))


__all__ = [
    "_OneShotProgress",
    "_SweepProgress",
    "_ThresholdProgress",
    "_activation_progress_summary",
]
