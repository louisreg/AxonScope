from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSuite:
    """Named runtime benchmark invocation."""

    name: str
    description: str
    runner: str
    args: tuple[str, ...]


RUNTIME_SUITES: dict[str, RuntimeSuite] = {
    "smoke": RuntimeSuite(
        name="smoke",
        description="Fast HH runtime smoke benchmark.",
        runner="benchmark_solver",
        args=("--cases", "hh_intracellular_small", "--repeats", "1", "--warmups", "0"),
    ),
    "full": RuntimeSuite(
        name="full",
        description="All default solver workloads with warm repeats.",
        runner="benchmark_solver",
        args=("--cases", "all", "--repeats", "3", "--warmups", "1"),
    ),
    "profiled": RuntimeSuite(
        name="profiled",
        description="All default workloads with a JAX profiler trace.",
        runner="benchmark_solver",
        args=(
            "--cases",
            "all",
            "--repeats",
            "3",
            "--warmups",
            "1",
            "--jax-profile-dir",
            "benchmark/results/jax_profiles",
        ),
    ),
    "vstim_forcing": RuntimeSuite(
        name="vstim_forcing",
        description="HH imposed-Vstim single-cable prototype against the full extracellular solver.",
        runner="benchmark_solver",
        args=(
            "--cases",
            "hh_extracellular_small",
            "hh_extracellular_medium",
            "--solvers",
            "crank_nicholson",
            "crank_nicholson_vstim_forcing",
            "--repeats",
            "5",
            "--warmups",
            "1",
        ),
    ),
    "vstim_batch": RuntimeSuite(
        name="vstim_batch",
        description="Scalar-loop vs batched imposed-Vstim single-cable kernel benchmark.",
        runner="benchmark_vstim_batch",
        args=(
            "--batch-sizes",
            "1",
            "2",
            "4",
            "8",
            "16",
            "--nx",
            "41",
            "--tsim",
            "1.2",
            "--dt",
            "0.01",
            "--repeats",
            "5",
            "--warmups",
            "1",
        ),
    ),
    "reference_solvers": RuntimeSuite(
        name="reference_solvers",
        description="Focused HH benchmark for optimized CN against the dense reference solver.",
        runner="benchmark_solver",
        args=(
            "--cases",
            "hh_intracellular_small",
            "--solvers",
            "crank_nicholson",
            "crank_nicholson_dense_reference",
            "--repeats",
            "5",
            "--warmups",
            "1",
        ),
    ),
}
