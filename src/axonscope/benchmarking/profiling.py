"""Benchmark profiling interface.

Concrete profiler implementations live behind `axonscope.runtime.execution`.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path


def profile_trace(
    runtime: str,
    log_dir: Path | str,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
) -> object:
    """Return a runtime-owned profiler trace context."""

    from axonscope.runtime.execution import benchmark_profile_trace

    return benchmark_profile_trace(
        runtime,
        Path(log_dir),
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
    )


def jax_profile_trace(
    log_dir: Path | str,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
) -> object:
    """Return a JAX profiler trace context through the runtime boundary."""

    return profile_trace(
        "jax",
        log_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
    )


def trace_annotation(name: str):
    """Return a trace annotation context when a runtime can provide one."""

    try:
        from axonscope.runtime.execution import benchmark_trace_annotation

        return benchmark_trace_annotation(name)
    except Exception:
        return nullcontext()


__all__ = ["jax_profile_trace", "profile_trace", "trace_annotation"]
