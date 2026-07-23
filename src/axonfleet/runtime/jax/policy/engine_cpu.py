"""CPU JAX solver-engine resolution."""

from __future__ import annotations

from axonfleet.runtime.jax.policy.engine_types import (
    CPU_SINGLE_CABLE_SOLVER,
    JaxSolverEngine,
)
from axonfleet.runtime.jax.policy.engine_common import (
    resolve_double_cable_policy,
)
from axonfleet.runtime.jax.policy import (
    DoubleCableSolverKind,
)
from axonfleet.runtime.policy import SolverPolicy


def resolve_cpu_solver_engine(policy: SolverPolicy) -> JaxSolverEngine:
    """Resolve the public CPU solver policy to a JAX CPU engine descriptor."""

    double_cable = resolve_double_cable_policy(policy, platform="cpu")
    if double_cable.kind in {
        DoubleCableSolverKind.AUTO,
        DoubleCableSolverKind.THOMAS,
    }:
        return JaxSolverEngine(
            name="jax_cpu_thomas",
            platform="cpu",
            single_cable_solver=CPU_SINGLE_CABLE_SOLVER,
            double_cable_block_solver="thomas",
        )
    raise ValueError(
        "Unsupported CPU double-cable solver policy "
        f"{double_cable.kind!r}; CPU double-cable supports only auto/thomas."
    )
