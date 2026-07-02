"""Top-level benchmark surface classification.

This registry is intentionally descriptive. Concrete suite arguments remain in
the runner-local modules that execute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BenchmarkStatus = Literal[
    "active",
    "validation-only",
    "experimental",
    "archive",
    "generated-output",
]


@dataclass(frozen=True)
class BenchmarkSurface:
    """One benchmark folder or entry-point family."""

    path: str
    status: BenchmarkStatus
    owner: str
    description: str
    entrypoints: tuple[str, ...] = ()
    docs: tuple[str, ...] = ()


BENCHMARK_SURFACES: tuple[BenchmarkSurface, ...] = (
    BenchmarkSurface(
        path="benchmark/runtime",
        status="active",
        owner="runtime",
        description=(
            "Named runtime suites for supported public execution paths, including "
            "class-based membrane model codegen/cache benchmarks."
        ),
        entrypoints=(
            "python benchmark/runtime/run.py --list",
            "python benchmark/runtime/run.py --suite model_codegen",
        ),
    ),
    BenchmarkSurface(
        path="benchmark/hotpaths",
        status="active",
        owner="runtime-diagnostics",
        description="Opt-in hotpath probes for dispatch, lowering, memory, and cold/warm spans.",
        entrypoints=("python benchmark/hotpaths/run.py --list",),
        docs=("benchmark/hotpaths/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/nrv_performance",
        status="active",
        owner="external-comparison",
        description="AxonScope-vs-NRV and realistic fascicle performance suites.",
        entrypoints=("python benchmark/nrv_performance/run.py --list",),
    ),
    BenchmarkSurface(
        path="benchmark/realistic_examples",
        status="active",
        owner="workflow",
        description="Workflow-level public-example benchmarks.",
        entrypoints=("python benchmark/realistic_examples/bench_basic_examples.py --help",),
        docs=("benchmark/realistic_examples/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/solvers",
        status="validation-only",
        owner="solver-validation",
        description="Retained double-cable solver timing, agreement, and trace matrices.",
        entrypoints=(
            "python benchmark/solvers/bench_double_cable_linear_solvers.py --dry-run",
            "python benchmark/solvers/bench_double_cable_end_to_end.py --dry-run",
            "python benchmark/solvers/validate_double_cable_solver_agreement.py --dry-run",
        ),
        docs=("benchmark/solvers/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/kaggle",
        status="active",
        owner="remote-gpu",
        description="Remote GPU wrapper for active runtime, realistic, and solver-validation suites.",
        entrypoints=("python benchmark/kaggle/run_kernel.py --help",),
        docs=("benchmark/kaggle/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/pseudo_double",
        status="experimental",
        owner="solver-research",
        description="Pseudo-double validation harness kept as standby evidence, not a public solver route.",
        docs=("benchmark/pseudo_double/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/archived_solver_spikes",
        status="archive",
        owner="solver-research",
        description="Archived Pallas/prototype solver spikes retained for evidence only.",
        docs=("benchmark/archived_solver_spikes/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/triton_solver",
        status="archive",
        owner="solver-research",
        description="Archived Triton solver candidate.",
        docs=("benchmark/triton_solver/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/jax_triton_solver",
        status="archive",
        owner="solver-research",
        description="Archived JAX-Triton solver candidate.",
        docs=("benchmark/jax_triton_solver/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/cuda_ffi_solver",
        status="archive",
        owner="solver-research",
        description="Archived CUDA FFI solver candidate.",
        docs=("benchmark/cuda_ffi_solver/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/cute_dsl",
        status="archive",
        owner="solver-research",
        description="Archived CuTe DSL smoke/candidate material.",
        docs=("benchmark/cute_dsl/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/notebooks",
        status="archive",
        owner="notebook-snapshots",
        description="Historical notebook snapshots; keep out of the active benchmark contract.",
    ),
    BenchmarkSurface(
        path="benchmark/reports",
        status="generated-output",
        owner="reports",
        description="Generated reports and figures. Ignored by git and not an architecture source of truth.",
    ),
    BenchmarkSurface(
        path="benchmark/results",
        status="generated-output",
        owner="outputs",
        description="Generated raw benchmark outputs. Ignored by git and never used as source code evidence.",
    ),
)


def surfaces_by_status(status: BenchmarkStatus) -> tuple[BenchmarkSurface, ...]:
    """Return benchmark surfaces with the requested lifecycle status."""

    return tuple(surface for surface in BENCHMARK_SURFACES if surface.status == status)


__all__ = [
    "BENCHMARK_SURFACES",
    "BenchmarkStatus",
    "BenchmarkSurface",
    "surfaces_by_status",
]
