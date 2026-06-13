"""Optional progress reporting for pool dispatch and batch solver execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeAlias


ProgressMode: TypeAlias = Literal["auto", "rich", "plain"]
ProgressOption: TypeAlias = bool | ProgressMode
KernelProgressCallback: TypeAlias = Callable[[int, int], None]


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
                    "dispatch groups",
                    total=len(self.plan.groups),
                )
                self._kernel_task = self._rich.add_task(
                    "kernel",
                    total=1,
                    visible=False,
                )
                return self
            except ImportError:
                if self._mode == "rich":
                    raise
        self._use_plain = True
        print(f"Dispatch progress: {len(self.plan.items)} rows, {len(self.plan.groups)} groups")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._rich is not None:
            self._rich.stop()

    def start_group(self, group: Any) -> None:
        """Mark one dispatch group as running."""

        if self._mode is None:
            return
        self._group_index += 1
        label = (
            f"group {group.group_id} {_dispatch_method(group)} "
            f"B={group.size} Nx={group.nx}"
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
            print(f"[{self._group_index}/{len(self.plan.groups)}] {label}")

    def finish_group(self, group: Any) -> None:
        """Mark one dispatch group as complete."""

        if self._mode is None:
            return
        if self._rich is not None:
            self._rich.advance(self._group_task, 1)
            self._rich.update(self._kernel_task, visible=False)
        elif self._use_plain:
            print(f"done group {group.group_id}")

    def kernel_callback(self, group: Any) -> KernelProgressCallback | None:
        """Return a callback for chunked solver kernels."""

        if self._mode is None:
            return None

        def _callback(done: int, total: int) -> None:
            total = max(int(total), 1)
            done = min(max(int(done), 0), total)
            if self._rich is not None:
                self._rich.update(
                    self._kernel_task,
                    description=f"kernel {group.group_id} chunks",
                    completed=done,
                    total=total,
                    visible=True,
                )
            elif self._use_plain and total > 1:
                print(f"  kernel chunks {done}/{total}")

        return _callback


__all__ = [
    "DispatchProgress",
    "KernelProgressCallback",
    "ProgressMode",
    "ProgressOption",
]
