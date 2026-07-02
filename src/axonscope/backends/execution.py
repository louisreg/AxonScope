"""Backend execution adapters used by public simulation orchestration.

This module is the backend boundary for public simulation entry points. It keeps
concrete JAX modules out of ``axonscope.simulation`` import time while still
centralizing the currently supported backend route.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from axonscope.performance import ExecutionPolicy
from axonscope.recording import Recording, RecordingPlan
from axonscope.solvers import BatchOptions, resolve_double_cable_block_solver


def execution_context(
    policy: ExecutionPolicy | None,
    *,
    instances: Sequence[Any],
):
    """Return the execution context for the currently supported backend."""

    from axonscope.backends.jax.execution_policy import jax_execution_context

    return jax_execution_context(policy, instances=instances)


def batch_options_from_recording(
    recording: Recording | RecordingPlan | None,
    *,
    batch_options: BatchOptions | None,
) -> BatchOptions | None:
    """Lower a public recording request to backend batch options."""

    from axonscope.backends.jax.recording import (
        batch_options_from_recording as jax_batch_options_from_recording,
    )

    return jax_batch_options_from_recording(recording, batch_options=batch_options)


def batch_options_for_execution_context(
    batch_options: BatchOptions | None,
    context: Any,
) -> BatchOptions | None:
    """Apply effective backend/device routing to batch-only solver options."""

    platform = getattr(context, "platform", None)
    if platform is None:
        return batch_options
    options = BatchOptions.full() if batch_options is None else batch_options
    if options.double_cable_block_solver != "auto":
        return options
    return replace(
        options,
        double_cable_block_solver=resolve_double_cable_block_solver(
            "auto",
            platform=platform,
        ),
    )


def run_batch_group(
    group: Any,
    *,
    tsim_ms: float,
    dt_ms: float,
    batch_options: BatchOptions,
    solver_options: Any | None,
    observers: Sequence[Any] | None,
    progress_callback: Any = None,
    backend_context: Any | None = None,
):
    """Execute a prepared dispatch group through the active concrete backend."""

    from axonscope.backends.jax.group_runner import run_jax_batch_group

    return run_jax_batch_group(
        group,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        batch_options=batch_options,
        solver_options=solver_options,
        observers=observers,
        progress_callback=progress_callback,
        backend_context=backend_context,
    )


__all__ = [
    "batch_options_for_execution_context",
    "batch_options_from_recording",
    "execution_context",
    "run_batch_group",
]
