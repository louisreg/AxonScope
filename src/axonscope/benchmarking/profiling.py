"""Benchmark profiling interface.

Concrete profiler implementations live behind `axonscope.backends.execution`.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path


def profile_trace(
    backend: str,
    log_dir: Path | str,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
) -> object:
    """Return a backend-owned profiler trace context."""

    from axonscope.backends.execution import benchmark_profile_trace

    return benchmark_profile_trace(
        backend,
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
    """Return a JAX profiler trace context through the backend boundary."""

    return profile_trace(
        "jax",
        log_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
    )


def trace_annotation(name: str):
    """Return a trace annotation context when a backend can provide one."""

    try:
        from axonscope.backends.execution import benchmark_trace_annotation

        return benchmark_trace_annotation(name)
    except Exception:
        return nullcontext()


__all__ = ["jax_profile_trace", "profile_trace", "trace_annotation"]
