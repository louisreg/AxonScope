from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NrvPerformanceSuite:
    """Named AxonScope-vs-NRV performance benchmark invocation."""

    name: str
    description: str
    runner: str
    args: tuple[str, ...]


NRV_PERFORMANCE_SUITES: dict[str, NrvPerformanceSuite] = {
    "smoke": NrvPerformanceSuite(
        name="smoke",
        description="Fast HH AxonScope-vs-NRV performance smoke grid.",
        runner="nrv_axonscope_grid",
        args=("--profile", "smoke"),
    ),
    "full": NrvPerformanceSuite(
        name="full",
        description="Full HH/MRG AxonScope-vs-NRV performance grid.",
        runner="nrv_axonscope_grid",
        args=("--profile", "full"),
    ),
    "mrg_smoke": NrvPerformanceSuite(
        name="mrg_smoke",
        description="Small MRG intracellular/extracellular NRV performance comparison.",
        runner="nrv_axonscope_grid",
        args=(
            "--profile",
            "smoke",
            "--model",
            "mrg_intracellular",
            "mrg_extracellular",
            "--dt",
            "0.01",
            "--nodes",
            "5",
            "--tsim",
            "2.0",
        ),
    ),
    "mrg_extracellular_gates": NrvPerformanceSuite(
        name="mrg_extracellular_gates",
        description="Focused MRG extracellular performance run with m-gate diagnostics.",
        runner="nrv_axonscope_grid",
        args=(
            "--model",
            "mrg_extracellular",
            "--dt",
            "0.005",
            "0.01",
            "--nodes",
            "5",
            "9",
            "--tsim",
            "4.0",
            "--diameter",
            "10.0",
            "--record-gates",
        ),
    ),
}

SUITE_ALIASES = {
    "nrv_smoke": "smoke",
    "nrv_full": "full",
    "nrv_mrg_smoke": "mrg_smoke",
    "nrv_mrg_extracellular_gates": "mrg_extracellular_gates",
}


def suite_choices() -> tuple[str, ...]:
    return tuple(sorted((*NRV_PERFORMANCE_SUITES, *SUITE_ALIASES)))


def resolve_suite(name: str) -> NrvPerformanceSuite:
    canonical_name = SUITE_ALIASES.get(name, name)
    return NRV_PERFORMANCE_SUITES[canonical_name]
