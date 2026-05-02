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
}
