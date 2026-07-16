"""Write per-run-pool timing details from AxonScope benchmark events."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


RUN_POOL_DETAIL_STAGES = (
    "dispatch.build_plan",
    "runtime.prepare",
    "inputs.positions",
    "inputs.extracellular",
    "kernel.enqueue",
    "kernel.dispatch_jax",
    "kernel.wait",
    "kernel.finalize_observer",
    "results.split_batch",
    "results.to_public",
)


def write_run_pool_detail(
    run_dir: Path,
    *,
    amplitudes: Sequence[float],
) -> None:
    """Write one all/single/double timing row per run-pool invocation."""

    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {int(event["event_id"]): event for event in events}
    run_pool_events = [event for event in events if event.get("name") == "simulation.run_pool"]
    rows: list[dict[str, Any]] = []
    completed_value_count = 0
    amplitude_values = tuple(float(value) for value in amplitudes)
    for unit_index, run_pool_event in enumerate(run_pool_events):
        descendants = [
            event
            for event in events
            if _is_descendant(event, int(run_pool_event["event_id"]), by_id)
        ]
        chunk_span = _nearest_ancestor_named(
            run_pool_event,
            "protocol.sweep.amplitude_chunk",
            by_id,
        )
        value_count = int((chunk_span or {}).get("metadata", {}).get("value_count", 1))
        unit_amplitudes = amplitude_values[
            completed_value_count : completed_value_count + value_count
        ]

        run_pool_ms = float(run_pool_event.get("duration_ms", 0.0))
        base = {
            "unit_index": unit_index,
            "amplitude_count": value_count,
            "amplitudes_uA": " ".join(f"{value:g}" for value in unit_amplitudes),
            "run_pool_ms": run_pool_ms,
            "kernel_wait_ms": _sum_stage(descendants, "kernel.wait"),
            "kernel_wait_pct_run_pool": _percent(
                _sum_stage(descendants, "kernel.wait"),
                run_pool_ms,
            ),
        }
        for mode in ("all", "double", "single"):
            mode_events = (
                descendants
                if mode == "all"
                else [event for event in descendants if _event_mode(event, by_id) == mode]
            )
            row = dict(base)
            row["mode"] = mode
            group_ms = (
                run_pool_ms
                if mode == "all"
                else _sum_stage(mode_events, "dispatch.group.total")
            )
            row["group_ms"] = group_ms
            for stage in RUN_POOL_DETAIL_STAGES:
                row[f"{stage}_ms"] = _sum_stage(mode_events, stage)
            enqueue_ms = float(row["kernel.enqueue_ms"])
            wait_ms = float(row["kernel.wait_ms"])
            # dispatch_jax is nested in enqueue. On asynchronous backends the
            # enqueue call can also absorb solver work through queue backpressure.
            solver_ms = enqueue_ms + wait_ms
            row["kernel_solver_ms"] = solver_ms
            row["kernel_solver_pct_group"] = _percent(solver_ms, group_ms)
            row["kernel_wait_pct_group"] = _percent(wait_ms, group_ms)
            row["kernel_wait_pct_solver"] = _percent(wait_ms, solver_ms)
            rows.append(row)
        completed_value_count += value_count
    if not rows:
        return
    with (run_dir / "run_pool_detail.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _is_descendant(
    event: dict[str, Any],
    ancestor_id: int,
    by_id: dict[int, dict[str, Any]],
) -> bool:
    parent_id = event.get("parent_event_id")
    while parent_id is not None:
        if int(parent_id) == ancestor_id:
            return True
        parent = by_id.get(int(parent_id))
        if parent is None:
            return False
        parent_id = parent.get("parent_event_id")
    return False


def _nearest_ancestor_named(
    event: dict[str, Any],
    name: str,
    by_id: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    parent_id = event.get("parent_event_id")
    while parent_id is not None:
        parent = by_id.get(int(parent_id))
        if parent is None:
            return None
        if parent.get("name") == name:
            return parent
        parent_id = parent.get("parent_event_id")
    return None


def _event_mode(
    event: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
) -> str | None:
    current: dict[str, Any] | None = event
    while current is not None:
        mode = current.get("metadata", {}).get("mode")
        if mode in {"single", "double"}:
            return str(mode)
        parent_id = current.get("parent_event_id")
        current = None if parent_id is None else by_id.get(int(parent_id))
    return None


def _sum_stage(events: Sequence[dict[str, Any]], name: str) -> float:
    return sum(
        float(event.get("duration_ms", 0.0))
        for event in events
        if event.get("name") == name
    )


def _percent(value: float, total: float) -> float:
    return 0.0 if total <= 0.0 else 100.0 * value / total


__all__ = ["RUN_POOL_DETAIL_STAGES", "write_run_pool_detail"]
