"""Top-level benchmark surface classification.

This registry is intentionally descriptive. Concrete suite arguments remain in
the runner-local modules that execute them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BenchmarkCommandKind = Literal[
    "public-runtime",
    "hotpath-diagnostic",
    "model-codegen",
    "validation-only",
    "external-comparison",
    "remote-GPU",
    "archive",
    "generated-output",
]
BenchmarkStatus = Literal[
    "active",
    "validation-only",
    "experimental",
    "archive",
    "generated-output",
]


@dataclass(frozen=True)
class BenchmarkCommand:
    """One retained benchmark command with its evidence class."""

    command: str
    kind: BenchmarkCommandKind
    purpose: str


@dataclass(frozen=True)
class BenchmarkSurface:
    """One benchmark folder or entry-point family."""

    path: str
    status: BenchmarkStatus
    owner: str
    description: str
    entrypoints: tuple[str, ...] = ()
    commands: tuple[BenchmarkCommand, ...] = ()
    docs: tuple[str, ...] = ()

    @property
    def command_kinds(self) -> tuple[BenchmarkCommandKind, ...]:
        """Return unique command classes used by this surface."""

        return tuple(dict.fromkeys(command.kind for command in self.commands))


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
        commands=(
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite smoke",
                kind="public-runtime",
                purpose="Fast supported-runtime smoke before broader timing runs.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite full",
                kind="public-runtime",
                purpose="Default supported runtime matrix with warm repeats.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite profiled",
                kind="hotpath-diagnostic",
                purpose="Runtime matrix with a JAX profiler output directory.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite vstim_forcing",
                kind="public-runtime",
                purpose="Supported single-cable imposed-Vstim path comparison.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite vstim_batch",
                kind="hotpath-diagnostic",
                purpose="Batch-kernel diagnostic for imposed-Vstim input paths.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite double_cable_batch",
                kind="hotpath-diagnostic",
                purpose="Batch-kernel diagnostic for double-cable runtime paths.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite pool_memory",
                kind="hotpath-diagnostic",
                purpose="Pool memory/runtime probe for retained-output policies.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite model_codegen",
                kind="model-codegen",
                purpose="Built-in source/codegen cache and model-step smoke timing.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite model_codegen_simulations",
                kind="model-codegen",
                purpose="Tiny public AxonSimulation first/warm timings for class-based templates.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite model_codegen_all",
                kind="model-codegen",
                purpose="Built-in plus custom codegen, model-step, and template simulation timing.",
            ),
            BenchmarkCommand(
                command="python benchmark/runtime/run.py --suite reference_solvers",
                kind="validation-only",
                purpose="Focused optimized-vs-dense HH solver comparison.",
            ),
        ),
    ),
    BenchmarkSurface(
        path="benchmark/hotpaths",
        status="active",
        owner="runtime-diagnostics",
        description="Opt-in hotpath probes for dispatch, lowering, memory, and cold/warm spans.",
        entrypoints=("python benchmark/hotpaths/run.py --list",),
        commands=(
            BenchmarkCommand(
                command="python benchmark/hotpaths/run.py --workload hotpath_matrix --preset smoke",
                kind="hotpath-diagnostic",
                purpose="Compact stage coverage before deeper CPU/GPU profiling.",
            ),
            BenchmarkCommand(
                command=(
                    "python benchmark/hotpaths/run.py --workload cold_run_micro "
                    "--sizes 1 --duration 1.0 --dt 0.02 --warmups 0 "
                    "--memory-trace rss --prefix cold_run_micro"
                ),
                kind="hotpath-diagnostic",
                purpose=(
                    "Short local P9 cold-run baseline covering retained Vm, "
                    "VmRaster observer-only, and point-source extracellular paths."
                ),
            ),
            BenchmarkCommand(
                command=(
                    "python benchmark/hotpaths/run.py --workload path_comparison_matrix "
                    "--sizes 1 --jax-log-compiles --prefix cold_path_probe"
                ),
                kind="hotpath-diagnostic",
                purpose="Cold-start and compile-log evidence for first-call claims.",
            ),
            BenchmarkCommand(
                command=(
                    "python benchmark/hotpaths/run.py --workload hotpath_matrix "
                    "--preset smoke --memory-trace all --memory-top-n 10 "
                    "--jax-device-memory-profile --prefix memory_map_smoke"
                ),
                kind="hotpath-diagnostic",
                purpose="Per-stage time+memory map for optimization targeting.",
            ),
            BenchmarkCommand(
                command=(
                    "python benchmark/hotpaths/run.py --workload double_cable_observer "
                    "--sizes 100 300 600 2000 --duration 10.0 --dt 0.01 "
                    "--compartments 51 --warmups 1 --double-cable-block-solver auto"
                ),
                kind="hotpath-diagnostic",
                purpose="MRG double-cable VmRaster compact-output scaling probe.",
            ),
        ),
        docs=("benchmark/hotpaths/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/nrv_performance",
        status="active",
        owner="external-comparison",
        description="AxonScope-vs-NRV and realistic fascicle performance suites.",
        entrypoints=("python benchmark/nrv_performance/run.py --list",),
        commands=(
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite smoke --dry-run",
                kind="external-comparison",
                purpose="Expand the smallest AxonScope-vs-NRV performance grid.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite full",
                kind="external-comparison",
                purpose="Full HH/MRG AxonScope-vs-NRV performance grid.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite mrg_extracellular_perf",
                kind="external-comparison",
                purpose="Focused MRG extracellular warm-runtime comparison.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite population_cold_path_smoke",
                kind="hotpath-diagnostic",
                purpose="AxonScope-only cold/warm point-source timing with hotpath reports.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite population_tsim",
                kind="external-comparison",
                purpose="Point-source population AxonScope-vs-NRV timing.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite population_tsim_gpu",
                kind="public-runtime",
                purpose="Synthetic AxonScope population timing with explicit GPU execution policy.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite population_tsim_gpu_1000",
                kind="public-runtime",
                purpose="Large synthetic AxonScope GPU population timing.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite realistic_fascicle_smoke",
                kind="external-comparison",
                purpose="Small NRV LIFE/FEM handoff and AxonScope recruitment profile.",
            ),
            BenchmarkCommand(
                command="python benchmark/nrv_performance/run.py --suite realistic_fascicle_synthetic_full",
                kind="external-comparison",
                purpose="Full-size synthetic NRV LIFE/FEM handoff profile.",
            ),
        ),
    ),
    BenchmarkSurface(
        path="benchmark/realistic_examples",
        status="active",
        owner="workflow",
        description="Workflow-level public-example benchmarks.",
        entrypoints=("python benchmark/realistic_examples/bench_basic_examples.py --help",),
        commands=(
            BenchmarkCommand(
                command="python benchmark/realistic_examples/bench_basic_examples.py --preset smoke --repeats 1",
                kind="public-runtime",
                purpose="Workflow-level public-example smoke timing.",
            ),
            BenchmarkCommand(
                command=(
                    "python benchmark/realistic_examples/bench_basic_examples.py "
                    "--preset stress --platforms cpu gpu --profile"
                ),
                kind="public-runtime",
                purpose="CPU/GPU public-workflow stress pass with hotpath profiles.",
            ),
        ),
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
        commands=(
            BenchmarkCommand(
                command="python benchmark/solvers/bench_double_cable_linear_solvers.py --dry-run",
                kind="validation-only",
                purpose="Retained double-cable linear solver timing matrix preview.",
            ),
            BenchmarkCommand(
                command="python benchmark/solvers/bench_double_cable_end_to_end.py --dry-run",
                kind="validation-only",
                purpose="Retained double-cable end-to-end timing matrix preview.",
            ),
            BenchmarkCommand(
                command="python benchmark/solvers/validate_double_cable_solver_agreement.py --dry-run",
                kind="validation-only",
                purpose="Agreement harness for retained solver-route changes.",
            ),
            BenchmarkCommand(
                command="python benchmark/solvers/profile_double_cable_linear_solvers.py --help",
                kind="validation-only",
                purpose="Focused trace helper for retained linear-solver diagnostics.",
            ),
        ),
        docs=("benchmark/solvers/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/kaggle",
        status="active",
        owner="remote-gpu",
        description="Remote GPU wrapper for active runtime, realistic, and solver-validation suites.",
        entrypoints=("python benchmark/kaggle/run_kernel.py --help",),
        commands=(
            BenchmarkCommand(
                command=(
                    "python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME "
                    "--benchmark population_tsim_gpu --machine-shape NvidiaTeslaP100"
                ),
                kind="remote-GPU",
                purpose="Run the synthetic population GPU validation preset remotely.",
            ),
            BenchmarkCommand(
                command=(
                    "python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME "
                    "--benchmark realistic_fascicle_nrv_gpu --machine-shape NvidiaTeslaP100"
                ),
                kind="remote-GPU",
                purpose="Run the NRV LIFE/FEM handoff smoke on a remote GPU.",
            ),
            BenchmarkCommand(
                command=(
                    "python benchmark/kaggle/prepare_kernel_metadata.py "
                    "--username YOUR_KAGGLE_USERNAME --benchmark smoke"
                ),
                kind="generated-output",
                purpose="Generate Kaggle metadata/config files before remote submission.",
            ),
        ),
        docs=("benchmark/kaggle/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/archived_solver_spikes",
        status="archive",
        owner="solver-research",
        description="Archived Pallas/prototype solver spikes retained for evidence only.",
        commands=(
            BenchmarkCommand(
                command="benchmark/archived_solver_spikes/*",
                kind="archive",
                purpose="Historical prototype code only; do not use for fresh claims.",
            ),
        ),
        docs=("benchmark/archived_solver_spikes/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/triton_solver",
        status="archive",
        owner="solver-research",
        description="Archived Triton solver candidate.",
        commands=(
            BenchmarkCommand(
                command="python benchmark/triton_solver/bench_double_cable_triton.py --help",
                kind="archive",
                purpose="Historical Triton candidate only.",
            ),
        ),
        docs=("benchmark/triton_solver/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/jax_triton_solver",
        status="archive",
        owner="solver-research",
        description="Archived JAX-Triton solver candidate.",
        commands=(
            BenchmarkCommand(
                command="benchmark/jax_triton_solver/*",
                kind="archive",
                purpose="Historical JAX-Triton candidate snapshot only.",
            ),
        ),
        docs=("benchmark/jax_triton_solver/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/cuda_ffi_solver",
        status="archive",
        owner="solver-research",
        description="Archived CUDA FFI solver candidate.",
        commands=(
            BenchmarkCommand(
                command="python benchmark/cuda_ffi_solver/bench_double_cable_cuda_ffi.py --help",
                kind="archive",
                purpose="Historical CUDA FFI candidate only.",
            ),
        ),
        docs=("benchmark/cuda_ffi_solver/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/cute_dsl",
        status="archive",
        owner="solver-research",
        description="Archived CuTe DSL smoke/candidate material.",
        commands=(
            BenchmarkCommand(
                command="python benchmark/cute_dsl/run_cute_dsl_smoke.py --help",
                kind="archive",
                purpose="Historical CuTe DSL smoke material only.",
            ),
        ),
        docs=("benchmark/cute_dsl/README.md",),
    ),
    BenchmarkSurface(
        path="benchmark/notebooks",
        status="archive",
        owner="notebook-snapshots",
        description="Historical notebook snapshots; keep out of the active benchmark contract.",
        commands=(
            BenchmarkCommand(
                command="benchmark/notebooks/*.ipynb",
                kind="archive",
                purpose="Historical notebook snapshots only.",
            ),
        ),
    ),
    BenchmarkSurface(
        path="benchmark/reports",
        status="generated-output",
        owner="reports",
        description=(
            "Generated reports and figures. Some retained summaries are tracked; "
            "new generated reports are ignored and are not architecture source of truth."
        ),
        commands=(
            BenchmarkCommand(
                command="benchmark/reports/*",
                kind="generated-output",
                purpose="Generated summaries; cite tracked summaries only after fresh review.",
            ),
        ),
    ),
    BenchmarkSurface(
        path="benchmark/results",
        status="generated-output",
        owner="outputs",
        description="Generated raw benchmark outputs. Ignored by git and never used as source code evidence.",
        commands=(
            BenchmarkCommand(
                command="benchmark/results/*",
                kind="generated-output",
                purpose="Ignored raw benchmark outputs; never edit as architecture evidence.",
            ),
        ),
    ),
)


def surfaces_by_status(status: BenchmarkStatus) -> tuple[BenchmarkSurface, ...]:
    """Return benchmark surfaces with the requested lifecycle status."""

    return tuple(surface for surface in BENCHMARK_SURFACES if surface.status == status)


__all__ = [
    "BENCHMARK_SURFACES",
    "BenchmarkCommand",
    "BenchmarkCommandKind",
    "BenchmarkStatus",
    "BenchmarkSurface",
    "surfaces_by_status",
]
