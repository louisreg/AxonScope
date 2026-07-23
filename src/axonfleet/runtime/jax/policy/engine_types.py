"""Internal JAX solver-engine descriptors."""

from __future__ import annotations

from dataclasses import dataclass


CPU_SINGLE_CABLE_SOLVER = "jax_tridiagonal"
GPU_SINGLE_CABLE_SOLVER = "jax_triton_tiled_thomas_xb"


@dataclass(frozen=True)
class JaxSolverEngine:
    """Resolved backend-local solver engine.

    This is an internal routing object. Public code chooses typed
    ``ExecutionPolicy`` values; JAX lowering resolves them to this compact
    descriptor before launching kernels.
    """

    name: str
    platform: str
    single_cable_solver: str
    double_cable_block_solver: str
    tiled_thomas_block_b: int | None = None
