"""Shared JAX solver-policy validation helpers."""

from __future__ import annotations

from axonfleet.runtime.jax.policy import (
    DoubleCableSolver,
)
from axonfleet.runtime.policy import SolverPolicy


def resolve_double_cable_policy(
    policy: SolverPolicy,
    *,
    platform: str,
) -> DoubleCableSolver:
    """Resolve a JAX double-cable solver request shared by CPU/GPU engines."""

    if policy.double_cable is None:
        return DoubleCableSolver.auto()
    if isinstance(policy.double_cable, DoubleCableSolver):
        return policy.double_cable
    raise ValueError(
        f"JAX {platform.upper()} execution requires a double-cable solver from "
        f"axs.runtime.jax.{platform} or axs.runtime.jax."
    )
