"""Optional progress reporting for pool dispatch and batch solver execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Mapping, TypeAlias

from axonfleet.utils.progress_reporting import (
    progress_timestamp,
    runtime_snapshot,
    timing_summary,
)


ProgressMode: TypeAlias = Literal["auto", "rich", "plain"]
ProgressOption: TypeAlias = bool | ProgressMode
ProgressStage: TypeAlias = Literal[
    "dispatch",
    "route",
    "prepare",
    "batch",
    "lowering",
    "kernel",
    "result",
]
KernelProgressCallback: TypeAlias = Callable[..., None]


_STAGE_LABELS: Mapping[ProgressStage, str] = {
    "dispatch": "plan",
    "route": "route",
    "prepare": "prepare",
    "batch": "batch",
    "lowering": "lower",
    "kernel": "kernel",
    "result": "result",
}

_STAGE_STYLES: Mapping[ProgressStage, str] = {
    "dispatch": "cyan",
    "route": "magenta",
    "prepare": "blue",
    "batch": "blue",
    "lowering": "yellow",
    "kernel": "green",
    "result": "cyan",
}


def _normalize_progress_mode(progress: ProgressOption) -> ProgressMode | None:
    if progress is False:
        return None
    if progress is True:
        return "auto"
    if progress in {"auto", "rich", "plain"}:
        return progress
    raise ValueError("progress must be False, True, 'auto', 'rich', or 'plain'.")


def _format_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _short_batch_kind(value: Any) -> str:
    text = str(value)
    for suffix in ("-single-cable", "-double-cable"):
        if text.endswith(suffix):
            return text.removesuffix(suffix)
    return text


def _short_route(value: Any) -> str:
    text = str(value)
    if text == "scalar":
        return "scalar"
    if text.startswith("parameter-batch-"):
        return "parameter batch"
    if text.startswith("batch-"):
        return "batch"
    return text


def _rich_event_details(event: ProgressEvent) -> str:
    details = event.details or {}
    parts: list[str] = []

    if event.stage == "dispatch":
        if event.rows is not None:
            parts.append(f"rows={event.rows}")
    elif event.stage == "route":
        if event.route:
            parts.append(_short_route(event.route))
        if details.get("batch_kind"):
            parts.append(_short_batch_kind(details["batch_kind"]))
        if "padding" in details:
            parts.append(f"padding={_format_bool(details['padding'])}")
    elif event.stage == "prepare":
        if details.get("mode"):
            parts.append(f"mode={details['mode']}")
    elif event.stage == "batch":
        if details.get("recording"):
            parts.append(f"recording={details['recording']}")
        if "observers" in details:
            parts.append(f"observers={details['observers']}")
    elif event.stage == "lowering":
        intracellular = details.get("intracellular")
        extracellular = details.get("extracellular")
        if intracellular or extracellular:
            parts.append(f"{intracellular or '-'} -> {extracellular or '-'}")
        if "stimulations" in details:
            parts.append(f"stimulations={details['stimulations']}")
    elif event.stage == "kernel":
        if details.get("recording"):
            parts.append(f"recording={details['recording']}")
        if details.get("time_chunk_steps"):
            parts.append(f"chunk={details['time_chunk_steps']} steps")
        if details.get("block_solver"):
            parts.append(f"solver={details['block_solver']}")
        if event.completed is not None and event.total is not None and event.total > 1:
            parts.append(f"{event.completed}/{event.total}")
    elif event.stage == "result":
        if details.get("output"):
            parts.append(f"output={details['output']}")

    return ", ".join(str(part) for part in parts)


def _rich_event_text(event: ProgressEvent) -> Any:
    from rich.text import Text

    style = _STAGE_STYLES.get(event.stage, "white")
    label = _STAGE_LABELS.get(event.stage, event.stage)
    text = Text()
    text.append(f"{progress_timestamp()} ", style="dim")
    text.append(f"{label:<7}", style=f"bold {style}")

    if event.group_id is not None:
        group = f"g{event.group_id}"
        if event.group_index is not None and event.group_count is not None:
            group += f" {event.group_index}/{event.group_count}"
        text.append(f" {group:<7}", style="bold")
    else:
        text.append(" ")

    if event.message:
        text.append(f" {event.message}", style="default")

    details = _rich_event_details(event)
    if details:
        text.append(f"  ({details})", style="dim")
    return text


def _rich_group_text(group: Any, *, group_index: int, group_count: int) -> Any:
    from rich.text import Text

    text = Text()
    text.append(f"{progress_timestamp()} ", style="dim")
    text.append("group   ", style="bold cyan")
    text.append(f"g{group.group_id} {group_index}/{group_count}", style="bold")
    text.append(f"  {group.dispatch_method}", style="default")
    meta = f"rows={group.size}, Nx={group.nx}"
    if group.has_padding:
        meta += ", padded"
    text.append(f"  ({meta})", style="dim")
    return text


def _plain_event_text(event: ProgressEvent) -> str:
    label = _STAGE_LABELS.get(event.stage, event.stage)
    parts = [progress_timestamp(), f"{label:<7}"]
    if event.group_id is not None:
        group = f"g{event.group_id}"
        if event.group_index is not None and event.group_count is not None:
            group += f" {event.group_index}/{event.group_count}"
        parts.append(f"{group:<7}")
    if event.message:
        parts.append(event.message)
    details = _rich_event_details(event)
    line = " ".join(part for part in parts if part)
    if details:
        line += f" ({details})"
    return line


def _plain_group_text(group: Any, *, group_index: int, group_count: int) -> str:
    meta = f"rows={group.size}, Nx={group.nx}"
    if group.has_padding:
        meta += ", padded"
    return (
        f"{progress_timestamp()} group   g{group.group_id} {group_index}/{group_count} "
        f"{group.dispatch_method} ({meta})"
    )


def _should_render_plain_chunk_progress(done: int, total: int) -> bool:
    if total <= 12:
        return True
    if done <= 1 or done >= total:
        return True
    interval = max(1, total // 10)
    return done % interval == 0


def emit_initial_progress(
    progress: ProgressOption,
    *,
    rows: int,
    message: str,
) -> None:
    """Emit a progress line before a dispatch plan exists."""

    mode = _normalize_progress_mode(progress)
    if mode is None:
        return

    event = ProgressEvent(stage="dispatch", rows=int(rows), message=message)
    if mode in {"auto", "rich"}:
        try:
            from rich.console import Console

            Console().print(_rich_event_text(event))
            return
        except ImportError:
            if mode == "rich":
                raise
    print(_plain_event_text(event), flush=True)


@dataclass(frozen=True)
class ProgressEvent:
    """Structured progress event emitted by dispatch and backend execution."""

    stage: ProgressStage
    group_id: int | None = None
    group_index: int | None = None
    group_count: int | None = None
    rows: int | None = None
    nx: int | None = None
    route: str | None = None
    message: str = ""
    completed: int | None = None
    total: int | None = None
    details: Mapping[str, Any] | None = None

    def plain_text(self) -> str:
        """Return a compact one-line representation for plain progress."""

        prefix = self.stage
        if self.group_id is not None:
            prefix += f" group={self.group_id}"
        parts: list[str] = []
        if self.route:
            parts.append(f"route={self.route}")
        if self.rows is not None:
            parts.append(f"rows={self.rows}")
        if self.nx is not None:
            parts.append(f"Nx={self.nx}")
        if self.completed is not None and self.total is not None:
            parts.append(f"{self.completed}/{self.total}")
        if self.message:
            parts.append(self.message)
        if self.details:
            parts.extend(
                f"{key}={value}"
                for key, value in self.details.items()
                if value is not None
            )
        return f"{prefix}: " + " ".join(str(part) for part in parts)


@dataclass
class DispatchProgress:
    """Context manager used by the dispatcher to report execution progress."""

    progress: ProgressOption
    plan: Any

    def __post_init__(self) -> None:
        self._mode = _normalize_progress_mode(self.progress)
        self._rich = None
        self._group_task = None
        self._kernel_task = None
        self._group_index = 0
        self._current_group_index: dict[int, int] = {}
        self._use_plain = False
        self._started = runtime_snapshot()

    def __enter__(self) -> "DispatchProgress":
        if self._mode is None:
            return self
        if self._mode in {"auto", "rich"}:
            try:
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                self._rich = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    refresh_per_second=8,
                    transient=True,
                )
                self._rich.start()
                group_label = "group" if len(self.plan.groups) == 1 else "groups"
                self._rich.console.print(
                    f"[dim]{progress_timestamp()}[/dim] [bold]Dispatch progress[/bold] "
                    f"[dim]{len(self.plan.items)} rows, "
                    f"{len(self.plan.groups)} {group_label}[/dim]"
                )
                self._group_task = self._rich.add_task(
                    "dispatch groups",
                    total=len(self.plan.groups),
                )
                self._kernel_task = self._rich.add_task(
                    "kernel progress",
                    total=1,
                    visible=False,
                )
                return self
            except ImportError:
                if self._mode == "rich":
                    raise
        self._use_plain = True
        group_label = "group" if len(self.plan.groups) == 1 else "groups"
        print(
            f"{progress_timestamp()} Dispatch progress: {len(self.plan.items)} rows, "
            f"{len(self.plan.groups)} {group_label}",
            flush=True,
        )
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self._rich is not None:
            self._rich.stop()
        if self._mode is None:
            return
        status = "failed" if exc_type is not None else "completed"
        summary = timing_summary(start=self._started, end=runtime_snapshot())
        if self._rich is not None:
            self._rich.console.print(
                f"[dim]{progress_timestamp()}[/dim] "
                f"[bold]Simulation run {status}[/bold] [dim]{summary}[/dim]"
            )
        elif self._use_plain:
            print(f"{progress_timestamp()} Simulation run {status}: {summary}", flush=True)

    def start_group(self, group: Any) -> None:
        """Mark one dispatch group as running."""

        if self._mode is None:
            return
        self._group_index += 1
        self._current_group_index[int(group.group_id)] = self._group_index
        label = (
            f"group {self._group_index}/{len(self.plan.groups)} "
            f"g{group.group_id}"
        )
        if self._rich is not None:
            self._rich.console.print(
                _rich_group_text(
                    group,
                    group_index=self._group_index,
                    group_count=len(self.plan.groups),
                )
            )
            self._rich.update(self._group_task, description=label)
            self._rich.update(
                self._kernel_task,
                description=f"kernel g{group.group_id}",
                completed=0,
                total=1,
                visible=True,
            )
        elif self._use_plain:
            print(
                _plain_group_text(
                    group,
                    group_index=self._group_index,
                    group_count=len(self.plan.groups),
                ),
                flush=True,
            )

    def route_group(self, group: Any, *, route: str, reason: str) -> None:
        """Report the selected execution route for one group."""

        self.emit(
            ProgressEvent(
                stage="route",
                group_id=int(group.group_id),
                group_index=self._current_group_index.get(int(group.group_id)),
                group_count=len(self.plan.groups),
                rows=int(group.size),
                nx=int(group.nx),
                route=route,
                message=reason,
                details={
                    "mode": group.mode,
                    "batch_kind": group.batch_kind,
                    "padding": bool(group.has_padding),
                },
            )
        )

    def finish_group(self, group: Any) -> None:
        """Mark one dispatch group as complete."""

        if self._mode is None:
            return
        if self._rich is not None:
            self._rich.update(
                self._group_task,
                description=f"completed g{group.group_id}",
            )
            self._rich.advance(self._group_task, 1)
            self._rich.update(self._kernel_task, visible=False)
        elif self._use_plain:
            print(f"{progress_timestamp()} done    g{group.group_id}", flush=True)

    def kernel_callback(self, group: Any) -> KernelProgressCallback | None:
        """Return a callback for backend progress events and kernel progress."""

        if self._mode is None:
            return None

        def _callback(event_or_done: Any, total: int | None = None) -> None:
            if isinstance(event_or_done, ProgressEvent):
                self.emit(event_or_done)
                return
            total = max(1 if total is None else int(total), 1)
            done = min(max(int(event_or_done), 0), total)
            self.emit(
                ProgressEvent(
                    stage="kernel",
                    group_id=int(group.group_id),
                    group_index=self._current_group_index.get(int(group.group_id)),
                    group_count=len(self.plan.groups),
                    rows=int(group.size),
                    nx=int(group.nx),
                    message="solving time chunks",
                    completed=done,
                    total=total,
                )
            )

        return _callback

    def emit(self, event: ProgressEvent) -> None:
        """Render one structured progress event."""

        if self._mode is None:
            return
        event = self._with_group_position(event)
        if self._rich is not None:
            self._render_rich_event(event)
        elif self._use_plain:
            self._render_plain_event(event)

    def _with_group_position(self, event: ProgressEvent) -> ProgressEvent:
        if event.group_id is None or event.group_index is not None:
            return event
        group_index = self._current_group_index.get(int(event.group_id))
        if group_index is None:
            return event
        return replace(
            event,
            group_index=group_index,
            group_count=len(self.plan.groups),
        )

    def _render_rich_event(self, event: ProgressEvent) -> None:
        if self._rich is None:
            return
        if event.stage == "kernel":
            update_kwargs: dict[str, Any] = {
                "description": self._event_label(event),
                "visible": True,
            }
            if event.completed is not None and event.total is not None:
                update_kwargs["completed"] = event.completed
                update_kwargs["total"] = max(event.total, 1)
            elif event.message.startswith("completed"):
                update_kwargs["completed"] = 1
                update_kwargs["total"] = 1
            else:
                update_kwargs["completed"] = 0
                update_kwargs["total"] = 1
            self._rich.update(self._kernel_task, **update_kwargs)
            if event.completed is None:
                self._rich.console.print(_rich_event_text(event))
            return
        self._rich.update(self._group_task, description=self._event_label(event))
        if event.stage in {"route", "prepare", "batch", "lowering", "result"}:
            self._rich.console.print(_rich_event_text(event))

    def _render_plain_event(self, event: ProgressEvent) -> None:
        if (
            event.stage == "kernel"
            and event.message == "solving time chunks"
            and event.completed is not None
            and event.total is not None
        ):
            if event.total == 1:
                return
            if not _should_render_plain_chunk_progress(event.completed, event.total):
                return
        print(f"  {_plain_event_text(event)}", flush=True)

    def _event_label(self, event: ProgressEvent) -> str:
        label = _STAGE_LABELS.get(event.stage, event.stage)
        if event.group_id is not None:
            label += f" g{event.group_id}"
        if event.message:
            label += f": {event.message}"
        if event.completed is not None and event.total is not None:
            label += f" {event.completed}/{event.total}"
        return label


__all__ = [
    "DispatchProgress",
    "KernelProgressCallback",
    "ProgressEvent",
    "ProgressMode",
    "ProgressOption",
    "ProgressStage",
    "emit_initial_progress",
]
