"""Internal JAX solver-engine descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JaxSolverEngine:
    """Resolved backend-local solver engine.

    This is an internal routing object. Public code chooses typed
    ``ExecutionPolicy`` values; JAX lowering resolves them to this compact
    descriptor before launching kernels.
    """

    name: str
    platform: str
    double_cable_block_solver: str | None
    allow_internal_double_cable_block_solver: bool = False
    tiled_thomas_block_b: int | None = None


__all__ = ["JaxSolverEngine"]
