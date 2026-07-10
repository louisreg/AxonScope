"""CPU JAX solver-engine resolution."""

from __future__ import annotations

from axonscope.runtime.jax.solver_engines.types import JaxSolverEngine
from axonscope.runtime.jax.policy import DoubleCableSolver, DoubleCableSolverKind
from axonscope.runtime.policy import SolverPolicy


def resolve_cpu_solver_engine(policy: SolverPolicy) -> JaxSolverEngine:
    """Resolve the public CPU solver policy to a JAX CPU engine descriptor."""

    double_cable = _resolve_double_cable_policy(policy)
    if double_cable.kind in {
        DoubleCableSolverKind.AUTO,
        DoubleCableSolverKind.THOMAS,
    }:
        return JaxSolverEngine(
            name="jax_cpu_thomas",
            platform="cpu",
            double_cable_block_solver="thomas",
        )
    raise ValueError(
        f"Unsupported CPU double-cable solver policy: {double_cable.kind!r}."
    )


def _resolve_double_cable_policy(policy: SolverPolicy) -> DoubleCableSolver:
    if policy.double_cable is None:
        return DoubleCableSolver.auto()
    if isinstance(policy.double_cable, DoubleCableSolver):
        return policy.double_cable
    raise ValueError(
        "JAX CPU execution requires a double-cable solver from "
        "axs.runtime.jax.cpu or axs.runtime.jax."
    )


__all__ = ["resolve_cpu_solver_engine"]
