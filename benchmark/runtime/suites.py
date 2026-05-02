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
    "experimental_solvers": RuntimeSuite(
        name="experimental_solvers",
        description="Focused HH benchmark for maintained experimental CN variants.",
        runner="benchmark_solver",
        args=(
            "--cases",
            "hh_intracellular_small",
            "--solvers",
            "crank_nicholson",
            "crank_nicholson_dense_reference",
            "crank_nicholson_semi_implicit",
            "crank_nicholson_implicit",
            "--repeats",
            "5",
            "--warmups",
            "1",
        ),
    ),
}
