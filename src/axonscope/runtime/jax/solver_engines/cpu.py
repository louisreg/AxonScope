"""CPU JAX solver-engine resolution."""

from __future__ import annotations

from axonscope.runtime.jax.solver_engines.types import JaxSolverEngine
from axonscope.runtime.jax.policy import (
    DoubleCableSolver,
    DoubleCableSolverKind,
    SingleCableSolver,
    SingleCableSolverKind,
)
from axonscope.runtime.policy import SolverPolicy


def resolve_cpu_solver_engine(policy: SolverPolicy) -> JaxSolverEngine:
    """Resolve the public CPU solver policy to a JAX CPU engine descriptor."""

    single_cable = _resolve_single_cable_policy(policy)
    double_cable = _resolve_double_cable_policy(policy)
    if double_cable.kind in {
        DoubleCableSolverKind.AUTO,
        DoubleCableSolverKind.THOMAS,
    }:
        return JaxSolverEngine(
            name="jax_cpu_thomas",
            platform="cpu",
            single_cable_solver=single_cable.value,
            double_cable_block_solver="thomas",
        )
    raise ValueError(
        "Unsupported CPU double-cable solver policy "
        f"{double_cable.kind!r}; CPU double-cable supports only auto/thomas."
    )


def _resolve_single_cable_policy(policy: SolverPolicy) -> SingleCableSolverKind:
    if policy.single_cable is None:
        return SingleCableSolverKind.JAX_TRIDIAGONAL
    if isinstance(policy.single_cable, SingleCableSolver):
        if policy.single_cable.kind in {
            SingleCableSolverKind.AUTO,
            SingleCableSolverKind.JAX_TRIDIAGONAL,
        }:
            return SingleCableSolverKind.JAX_TRIDIAGONAL
        raise ValueError(
            f"Unsupported CPU single-cable solver policy: {policy.single_cable.kind!r}."
        )
    raise ValueError(
        "JAX CPU execution requires a single-cable solver from "
        "axs.runtime.jax.cpu or axs.runtime.jax."
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
