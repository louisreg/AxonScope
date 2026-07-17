"""Shared JAX batch-kernel chunking and threshold-observer helpers."""

from __future__ import annotations

import jax.numpy as jnp

from axonscope.benchmarking import benchmark_span
from axonscope.runtime.jax.cable_geometry import Array
from axonscope.runtime.jax.recording.observer import (
    ObserverRetention,
    ThresholdObserverPlan,
    ThresholdObserverState,
    combine_threshold_observer_chunk_states,
    init_threshold_observer_state,
)


def _threshold_probe_tables_for_kernel(
    plan: ThresholdObserverPlan,
    *,
    batch_size: int,
) -> tuple[Array, Array]:
    with benchmark_span(
        "kernel.prepare_observer_tables",
        observer=plan.retention,
        group_size=batch_size,
        observer_definition_count=plan.definition_count,
        probe_count=plan.probe_count,
        row_aware=plan.row_aware,
    ):
        indices = jnp.asarray(plan.probe_indices)
        mask = jnp.asarray(plan.probe_mask)
        if indices.ndim == 2:
            shape = (int(batch_size),) + tuple(indices.shape)
            return (
                jnp.broadcast_to(indices[None, :, :], shape),
                jnp.broadcast_to(mask[None, :, :], shape),
            )
        return indices, mask


def _threshold_blanking_for_chunk(
    plan: ThresholdObserverPlan,
    *,
    start: int,
    dt_ms: Array,
    local_state: bool,
) -> Array:
    """Express absolute activation blanking in a chunk-local time frame."""

    if local_state and plan.retention == "activation":
        start_step = jnp.asarray(start, dtype=jnp.asarray(dt_ms).dtype)
        return plan.blanking_ms - start_step * dt_ms
    return plan.blanking_ms


def _init_local_threshold_chunk_template(
    plan: ThresholdObserverPlan,
    *,
    batch_size: int,
    chunk_ranges: tuple[tuple[int, int], ...],
    mode: str,
    variant: str,
    time_chunk_steps: int | None,
    enabled: bool,
) -> ThresholdObserverState | None:
    if not enabled:
        return None
    max_chunk_steps = max((stop - start for start, stop in chunk_ranges), default=0)
    if max_chunk_steps <= 0:
        return None
    with benchmark_span(
        "kernel.prepare_state",
        mode=mode,
        variant=variant,
        output="observer_only",
        observer=plan.retention,
        state="chunk_template",
        group_size=batch_size,
        chunk_steps=max_chunk_steps,
        chunk_count=len(chunk_ranges),
        time_chunk_steps=time_chunk_steps,
    ):
        return init_threshold_observer_state(plan, batch_size=batch_size, nt=max_chunk_steps)

def _resolve_threshold_observer_state_scope(
    scope: str | None,
    *,
    time_chunk_steps: int | None,
) -> str:
    text = "default" if scope in (None, "") else str(scope).strip().lower()
    if text == "default":
        return "full"
    if text not in {"chunk", "full"}:
        raise ValueError(
            "benchmark_observer_state_scope must be 'default', 'chunk', or 'full'."
        )
    if text == "chunk" and time_chunk_steps is None:
        return "full"
    return text

def _combine_threshold_observer_chunk_states(
    states: list[ThresholdObserverState],
    *,
    starts: list[int],
    lengths: list[int],
    nt: int,
    mode: str,
    variant: str,
    time_chunk_steps: int | None,
    retention: ObserverRetention = "vm_raster",
) -> ThresholdObserverState:
    with benchmark_span(
        "kernel.combine_observer_chunks",
        mode=mode,
        variant=variant,
        observer=retention,
        observer_state_scope="chunk",
        chunk_count=len(states),
        chunk_steps_min=min(lengths) if lengths else None,
        chunk_steps_max=max(lengths) if lengths else None,
        time_chunk_steps=time_chunk_steps,
        nt=nt,
    ):
        return combine_threshold_observer_chunk_states(
            states,
            starts=starts,
            lengths=lengths,
            nt=nt,
            retention=retention,
        )

def _normalize_time_chunk_steps(time_chunk_steps: int | None, *, nt: int) -> int | None:
    if time_chunk_steps is None:
        return None
    steps = int(time_chunk_steps)
    if steps < 1:
        raise ValueError("time_chunk_steps must be >= 1.")
    return min(steps, int(nt))

def _time_chunks(nt: int, time_chunk_steps: int | None):
    chunk_steps = nt if time_chunk_steps is None else time_chunk_steps
    for start in range(0, nt, chunk_steps):
        yield start, min(start + chunk_steps, nt)


def _pad_time_chunk(
    values: Array | None,
    *,
    target_steps: int,
    time_axis: int,
    edge: bool,
) -> Array | None:
    """Pad a final time chunk to an already-compiled static shape."""

    if values is None:
        return None
    arr = jnp.asarray(values)
    missing = int(target_steps) - int(arr.shape[time_axis])
    if missing <= 0:
        return arr
    widths = [(0, 0)] * arr.ndim
    widths[time_axis] = (0, missing)
    return jnp.pad(arr, widths, mode="edge" if edge else "constant")

def _concat_trace_chunks(chunks: list[Array]) -> Array:
    with benchmark_span(
        "kernel.concat_trace_chunks",
        chunk_count=len(chunks),
        single_chunk=len(chunks) == 1,
    ):
        if not chunks:
            raise ValueError("at least one time chunk is required.")
        if len(chunks) == 1:
            return chunks[0]
        return jnp.concatenate(chunks, axis=1)

__all__ = [
    "_combine_threshold_observer_chunk_states",
    "_concat_trace_chunks",
    "_init_local_threshold_chunk_template",
    "_normalize_time_chunk_steps",
    "_pad_time_chunk",
    "_resolve_threshold_observer_state_scope",
    "_time_chunks",
    "_threshold_probe_tables_for_kernel",
]
