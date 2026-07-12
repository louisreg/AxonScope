"""Shared JAX solver-policy validation helpers."""

from __future__ import annotations

from axonscope.runtime.jax.policy import (
    DoubleCableSolver,
    SingleCableSolver,
    SingleCableSolverKind,
)
from axonscope.runtime.policy import SolverPolicy


def resolve_single_cable_policy(
    policy: SolverPolicy,
    *,
    platform: str,
) -> SingleCableSolverKind:
    """Resolve a JAX single-cable solver request shared by CPU/GPU engines."""

    if policy.single_cable is None:
        return SingleCableSolverKind.JAX_TRIDIAGONAL
    if isinstance(policy.single_cable, SingleCableSolver):
        if policy.single_cable.kind in {
            SingleCableSolverKind.AUTO,
            SingleCableSolverKind.JAX_TRIDIAGONAL,
        }:
            return SingleCableSolverKind.JAX_TRIDIAGONAL
        raise ValueError(
            f"Unsupported {platform.upper()} single-cable solver policy: "
            f"{policy.single_cable.kind!r}."
        )
    raise ValueError(
        f"JAX {platform.upper()} execution requires a single-cable solver from "
        f"axs.runtime.jax.{platform} or axs.runtime.jax."
    )


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


__all__ = [
    "resolve_double_cable_policy",
    "resolve_single_cable_policy",
]

