from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationSuite:
    """Named validation benchmark invocation."""

    name: str
    description: str
    runner: str
    args: tuple[str, ...]


VALIDATION_SUITES: dict[str, ValidationSuite] = {
    "nrv_smoke": ValidationSuite(
        name="nrv_smoke",
        description="Fast HH AxonScope-vs-NRV smoke grid.",
        runner="nrv_axonscope_grid",
        args=("--profile", "smoke"),
    ),
    "nrv_full": ValidationSuite(
        name="nrv_full",
        description="Full HH/MRG AxonScope-vs-NRV grid.",
        runner="nrv_axonscope_grid",
        args=("--profile", "full"),
    ),
    "nrv_mrg_smoke": ValidationSuite(
        name="nrv_mrg_smoke",
        description="Small MRG intracellular and extracellular NRV comparison.",
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
    "nrv_mrg_extracellular_gates": ValidationSuite(
        name="nrv_mrg_extracellular_gates",
        description="Focused MRG extracellular comparison with m-gate diagnostics.",
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
