"""Internal JAX double-cable block-solver names."""

from __future__ import annotations

from typing import Literal, cast

DoubleCableBlockSolver = Literal["auto", "thomas", "pcr", "pcr_soa", "pcr_adaptive"]
ResolvedDoubleCableBlockSolver = Literal["thomas", "pcr", "pcr_soa", "pcr_adaptive"]

_GPU_PLATFORMS = {"cuda", "gpu", "metal", "rocm"}


def resolve_double_cable_block_solver(
    solver: DoubleCableBlockSolver | str,
    *,
    platform: str | None,
) -> ResolvedDoubleCableBlockSolver:
    """Resolve a backend-local double-cable block-solver request."""

    if solver == "auto":
        normalized = "" if platform is None else platform.lower()
        return "pcr_adaptive" if normalized in _GPU_PLATFORMS else "thomas"
    if solver in {"thomas", "pcr", "pcr_soa", "pcr_adaptive"}:
        return cast(ResolvedDoubleCableBlockSolver, solver)
    raise ValueError(
        "double_cable_block_solver must be 'auto', 'thomas', 'pcr', "
        "'pcr_soa', or 'pcr_adaptive'."
    )


__all__ = [
    "DoubleCableBlockSolver",
    "ResolvedDoubleCableBlockSolver",
    "resolve_double_cable_block_solver",
]
