"""GPU JAX solver-engine resolution."""

from __future__ import annotations

from axonscope.runtime.jax.policy.engine_types import (
    GPU_SINGLE_CABLE_SOLVER,
    JaxSolverEngine,
)
from axonscope.runtime.jax.policy.engine_common import (
    resolve_double_cable_policy,
    resolve_single_cable_policy,
)
from axonscope.runtime.jax.policy import (
    DoubleCableSolverKind,
)
from axonscope.runtime.policy import SolverPolicy


def resolve_gpu_solver_engine(policy: SolverPolicy) -> JaxSolverEngine:
    """Resolve the public GPU solver policy to a JAX GPU engine descriptor."""

    single_cable = resolve_single_cable_policy(policy, platform="gpu")
    double_cable = resolve_double_cable_policy(policy, platform="gpu")
    if double_cable.kind is DoubleCableSolverKind.AUTO:
        return JaxSolverEngine(
            name="jax_gpu_tiled_thomas",
            platform="gpu",
            single_cable_solver=GPU_SINGLE_CABLE_SOLVER,
            double_cable_block_solver="jax_triton_loop_xb",
            allow_internal_double_cable_block_solver=True,
            tiled_thomas_block_b=double_cable.tiled_thomas_options.block_b,
        )
    if double_cable.kind is DoubleCableSolverKind.TILED_THOMAS:
        return JaxSolverEngine(
            name="jax_gpu_tiled_thomas",
            platform="gpu",
            single_cable_solver=GPU_SINGLE_CABLE_SOLVER,
            double_cable_block_solver="jax_triton_loop_xb",
            allow_internal_double_cable_block_solver=True,
            tiled_thomas_block_b=double_cable.tiled_thomas_options.block_b,
        )
    raise ValueError(
        f"Unsupported GPU double-cable solver policy: {double_cable.kind!r}."
    )


__all__ = ["resolve_gpu_solver_engine"]
