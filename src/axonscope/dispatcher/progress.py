"""Optional progress reporting for pool dispatch and batch solver execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, TypeAlias


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


def _normalize_progress_mode(progress: ProgressOption) -> ProgressMode | None:
    if progress is False:
        return None
    if progress is True:
        return "auto"
    if progress in {"auto", "rich", "plain"}:
        return progress
    raise ValueError("progress must be False, True, 'auto', 'rich', or 'plain'.")


def _dispatch_method(group: Any) -> str:
    if group.size < 2:
        return "scalar"
    prefix = "batch" if group.geometry_shared else "parameter-batch"
    return f"{prefix}-{group.mode}-cable"


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

    def __enter__(self) -> "DispatchProgress":
        if self._mode is None:
            return self
        if self._mode in {"auto", "rich"}:
            try:
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                self._rich = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    transient=False,
                )
                self._rich.start()
                self._group_task = self._rich.add_task(
                    f"dispatch groups ({len(self.plan.items)} rows)",
                    total=len(self.plan.groups),
                )
                self._kernel_task = self._rich.add_task(
                    "kernel chunks",
                    total=1,
                    visible=False,
                )
                return self
            except ImportError:
                if self._mode == "rich":
                    raise
        self._use_plain = True
        print(
            f"Dispatch progress: {len(self.plan.items)} rows, {len(self.plan.groups)} groups",
            flush=True,
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._rich is not None:
            self._rich.stop()

    def start_group(self, group: Any) -> None:
        """Mark one dispatch group as running."""

        if self._mode is None:
            return
        self._group_index += 1
        self._current_group_index[int(group.group_id)] = self._group_index
        label = (
            f"group {group.group_id} {_dispatch_method(group)} "
            f"rows={group.size} Nx={group.nx}"
            f"{' padded' if group.has_padding else ''}"
        )
        if self._rich is not None:
            self._rich.update(self._group_task, description=label)
            self._rich.update(
                self._kernel_task,
                description=f"kernel {group.group_id}",
                completed=0,
                total=1,
                visible=True,
            )
        elif self._use_plain:
            print(f"[{self._group_index}/{len(self.plan.groups)}] {label}", flush=True)

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
            self._rich.advance(self._group_task, 1)
            self._rich.update(self._kernel_task, visible=False)
        elif self._use_plain:
            print(f"done group {group.group_id}", flush=True)

    def kernel_callback(self, group: Any) -> KernelProgressCallback | None:
        """Return a callback for backend progress events and kernel chunks."""

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
                    message="chunks",
                    completed=done,
                    total=total,
                )
            )

        return _callback

    def emit(self, event: ProgressEvent) -> None:
        """Render one structured progress event."""

        if self._mode is None:
            return
        if self._rich is not None:
            self._render_rich_event(event)
        elif self._use_plain:
            self._render_plain_event(event)

    def _render_rich_event(self, event: ProgressEvent) -> None:
        if self._rich is None:
            return
        if event.stage == "kernel" and event.completed is not None and event.total is not None:
            self._rich.update(
                self._kernel_task,
                description=self._event_label(event),
                completed=event.completed,
                total=max(event.total, 1),
                visible=True,
            )
            return
        self._rich.update(self._group_task, description=self._event_label(event))
        if event.stage in {"route", "prepare", "lowering", "result"}:
            self._rich.console.print(f"[dim]{event.plain_text()}[/dim]")

    def _render_plain_event(self, event: ProgressEvent) -> None:
        if event.stage == "kernel" and event.total == 1:
            return
        print(f"  {event.plain_text()}", flush=True)

    def _event_label(self, event: ProgressEvent) -> str:
        label = event.stage
        if event.group_id is not None:
            label += f" group {event.group_id}"
        if event.route:
            label += f" {event.route}"
        if event.message:
            label += f" {event.message}"
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
]
