"""GPU JAX solver-engine resolution."""

from __future__ import annotations

from axonscope.runtime.jax.solver_engines.types import JaxSolverEngine
from axonscope.runtime.jax.policy import (
    DoubleCableSolver,
    DoubleCableSolverKind,
    SingleCableSolver,
    SingleCableSolverKind,
)
from axonscope.runtime.policy import SolverPolicy


def resolve_gpu_solver_engine(policy: SolverPolicy) -> JaxSolverEngine:
    """Resolve the public GPU solver policy to a JAX GPU engine descriptor."""

    single_cable = _resolve_single_cable_policy(policy)
    double_cable = _resolve_double_cable_policy(policy)
    if double_cable.kind is DoubleCableSolverKind.AUTO:
        return JaxSolverEngine(
            name="jax_gpu_auto",
            platform="gpu",
            single_cable_solver=single_cable.value,
            double_cable_block_solver="pcr_adaptive",
        )
    if double_cable.kind is DoubleCableSolverKind.THOMAS:
        return JaxSolverEngine(
            name="jax_gpu_thomas",
            platform="gpu",
            single_cable_solver=single_cable.value,
            double_cable_block_solver="thomas",
        )
    if double_cable.kind is DoubleCableSolverKind.JAX_PCR:
        return JaxSolverEngine(
            name="jax_gpu_pcr",
            platform="gpu",
            single_cable_solver=single_cable.value,
            double_cable_block_solver="pcr",
        )
    if double_cable.kind is DoubleCableSolverKind.JAX_PCR_SOA:
        return JaxSolverEngine(
            name="jax_gpu_pcr_soa",
            platform="gpu",
            single_cable_solver=single_cable.value,
            double_cable_block_solver="pcr_soa",
        )
    if double_cable.kind is DoubleCableSolverKind.TILED_THOMAS:
        return JaxSolverEngine(
            name="jax_gpu_tiled_thomas",
            platform="gpu",
            single_cable_solver=single_cable.value,
            double_cable_block_solver="jax_triton_loop_xb",
            allow_internal_double_cable_block_solver=True,
            tiled_thomas_block_b=double_cable.tiled_thomas_options.block_b,
        )
    raise ValueError(
        f"Unsupported GPU double-cable solver policy: {double_cable.kind!r}."
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
            f"Unsupported GPU single-cable solver policy: {policy.single_cable.kind!r}."
        )
    raise ValueError(
        "JAX GPU execution requires a single-cable solver from "
        "axs.runtime.jax.gpu or axs.runtime.jax."
    )


def _resolve_double_cable_policy(policy: SolverPolicy) -> DoubleCableSolver:
    if policy.double_cable is None:
        return DoubleCableSolver.auto()
    if isinstance(policy.double_cable, DoubleCableSolver):
        return policy.double_cable
    raise ValueError(
        "JAX GPU execution requires a double-cable solver from "
        "axs.runtime.jax.gpu or axs.runtime.jax."
    )


__all__ = ["resolve_gpu_solver_engine"]
