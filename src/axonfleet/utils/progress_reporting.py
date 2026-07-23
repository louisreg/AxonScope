"""Small timing and memory helpers for human-facing progress output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import resource
import sys
import time


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Process timing/memory snapshot used by progress summaries."""

    perf_counter_s: float
    rss_mib: float | None


def progress_timestamp() -> str:
    """Return a compact local timestamp for progress lines."""

    return datetime.now().strftime("%H:%M:%S")


def runtime_snapshot() -> RuntimeSnapshot:
    """Return current elapsed-clock and process RSS when available."""

    return RuntimeSnapshot(
        perf_counter_s=time.perf_counter(),
        rss_mib=current_rss_mib(),
    )


def current_rss_mib() -> float | None:
    """Return resident set size in MiB when the platform exposes it."""

    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0**2)
    except Exception:
        pass

    try:
        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return maxrss / (1024.0**2)
    return maxrss / 1024.0


def format_duration(seconds: float | None) -> str:
    """Format elapsed seconds for compact terminal summaries."""

    if seconds is None:
        return "n/a"
    seconds = max(float(seconds), 0.0)
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m {rem:.1f}s"


def memory_summary(start: RuntimeSnapshot, end: RuntimeSnapshot) -> str:
    """Return RSS and delta-RSS summary."""

    if end.rss_mib is None:
        return "memory=n/a"
    if start.rss_mib is None:
        return f"rss={end.rss_mib:.1f} MiB"
    delta = end.rss_mib - start.rss_mib
    return f"rss={end.rss_mib:.1f} MiB, delta={delta:+.1f} MiB"


def timing_summary(
    *,
    start: RuntimeSnapshot,
    end: RuntimeSnapshot,
    iteration_durations_s: tuple[float, ...] = (),
) -> str:
    """Return total/cold/warm/per-iteration timing text."""

    total = end.perf_counter_s - start.perf_counter_s
    if iteration_durations_s:
        cold = float(iteration_durations_s[0])
        warm_values = tuple(float(value) for value in iteration_durations_s[1:])
        warm = sum(warm_values) / len(warm_values) if warm_values else None
        per_iteration = sum(iteration_durations_s) / len(iteration_durations_s)
    else:
        cold = total
        warm = None
        per_iteration = None
    return (
        f"total={format_duration(total)}, "
        f"cold_start={format_duration(cold)}, "
        f"warm={format_duration(warm)}, "
        f"per_iteration={format_duration(per_iteration)}, "
        f"{memory_summary(start, end)}"
    )
