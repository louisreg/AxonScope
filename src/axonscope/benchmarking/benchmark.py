"""Public benchmark instrumentation facade.

The concrete session, memory, and report implementation lives in
`axonscope.runtime.benchmarking`. Keep this module as the user-facing import
surface only.
"""

from __future__ import annotations

from axonscope.runtime.benchmarking import (
    BenchmarkConfig,
    BenchmarkEvent,
    BenchmarkOptions,
    BenchmarkReport,
    BenchmarkSession,
    BenchmarkSummaryRow,
    active_benchmark_session,
    benchmark,
    benchmark_array_metadata,
    benchmark_report,
    benchmark_span,
    benchmark_wait,
    disable_benchmark,
    enable_benchmark,
    record_benchmark_metadata,
    reset_benchmark,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkEvent",
    "BenchmarkOptions",
    "BenchmarkReport",
    "BenchmarkSession",
    "BenchmarkSummaryRow",
    "active_benchmark_session",
    "benchmark",
    "benchmark_array_metadata",
    "benchmark_report",
    "benchmark_span",
    "benchmark_wait",
    "disable_benchmark",
    "enable_benchmark",
    "record_benchmark_metadata",
    "reset_benchmark",
]
