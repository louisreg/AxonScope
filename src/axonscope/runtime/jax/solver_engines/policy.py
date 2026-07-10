"""JAX solver-engine policy resolution."""

from __future__ import annotations

from axonscope.runtime.jax.solver_engines.cpu import resolve_cpu_solver_engine
from axonscope.runtime.jax.solver_engines.gpu import resolve_gpu_solver_engine
from axonscope.runtime.jax.solver_engines.types import JaxSolverEngine
from axonscope.runtime import ExecutionPolicy


def resolve_jax_solver_engine(
    policy: ExecutionPolicy,
    *,
    platform: str | None,
) -> JaxSolverEngine | None:
    """Resolve public execution policy to an internal JAX solver engine."""

    if platform is None:
        return None
    solver_policy = policy.solver_policy
    if platform == "cpu":
        return resolve_cpu_solver_engine(solver_policy)
    if platform == "gpu":
        return resolve_gpu_solver_engine(solver_policy)
    return None


__all__ = ["resolve_jax_solver_engine"]
