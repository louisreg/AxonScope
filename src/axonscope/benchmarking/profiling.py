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
    import jax

    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(
        str(path),
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace,
    )
    try:
        yield path
    finally:
        jax.profiler.stop_trace()


def trace_annotation(name: str):
    """Return a JAX trace annotation context when JAX is available."""
    try:
        import jax

        return jax.profiler.TraceAnnotation(name)
    except Exception:
        return nullcontext()
