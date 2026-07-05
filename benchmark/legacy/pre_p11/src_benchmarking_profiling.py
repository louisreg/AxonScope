from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator


@contextmanager
def jax_profile_trace(
    log_dir: Path | str,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
) -> Iterator[Path]:
    """Capture a JAX profiler trace around a benchmark block."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    from axonscope.backends.execution import benchmark_profile_trace

    with benchmark_profile_trace(
        "jax",
        path,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
    ):
        yield path


def trace_annotation(name: str):
    """Return a JAX trace annotation context when JAX is available."""
    try:
        from axonscope.backends.execution import benchmark_trace_annotation

        return benchmark_trace_annotation(name)
    except Exception:
        return nullcontext()
