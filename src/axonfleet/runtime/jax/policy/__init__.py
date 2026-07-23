"""JAX runtime target and JAX-specific solver-policy values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from axonfleet.runtime.policy import (
    DoubleCableSolverRequest,
    RuntimeKind,
    RuntimeTarget,
)


class DoubleCableSolverKind(Enum):
    """JAX double-cable solver-family request."""

    AUTO = "auto"
    THOMAS = "thomas"
    TILED_THOMAS = "tiled_thomas"


@dataclass(frozen=True)
class TiledThomasSolverOptions:
    """Options for the tiled-Thomas JAX GPU solver."""

    block_b: int | None = None

    def __post_init__(self) -> None:
        if self.block_b is not None:
            if int(self.block_b) < 1:
                raise ValueError("block_b must be >= 1.")
            object.__setattr__(self, "block_b", int(self.block_b))


@dataclass(frozen=True)
class DoubleCableSolver(DoubleCableSolverRequest):
    """Typed JAX double-cable solver-family request."""

    kind: DoubleCableSolverKind = DoubleCableSolverKind.AUTO
    tiled_thomas_options: TiledThomasSolverOptions = field(
        default_factory=TiledThomasSolverOptions
    )

    @classmethod
    def auto(cls) -> "DoubleCableSolver":
        """Let the JAX runtime choose the supported double-cable route."""

        return cls(DoubleCableSolverKind.AUTO)

    @classmethod
    def thomas(cls) -> "DoubleCableSolver":
        """Request the exact Thomas-family double-cable route."""

        return cls(DoubleCableSolverKind.THOMAS)

    @classmethod
    def tiled_thomas(
        cls,
        *,
        block_b: int | None = None,
    ) -> "DoubleCableSolver":
        """Request the tiled-Thomas double-cable route."""

        return cls(
            DoubleCableSolverKind.TILED_THOMAS,
            tiled_thomas_options=TiledThomasSolverOptions(
                block_b=block_b,
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_double_cable_solver_kind(self.kind))
        if not isinstance(self.tiled_thomas_options, TiledThomasSolverOptions):
            raise TypeError(
                "double-cable tiled-Thomas options must be a TiledThomasSolverOptions value."
            )


class _JaxCpuNamespace:
    class DoubleCableSolver:
        """CPU-supported JAX double-cable solver policies."""

        auto = staticmethod(DoubleCableSolver.auto)
        thomas = staticmethod(DoubleCableSolver.thomas)


class _JaxGpuNamespace:
    class DoubleCableSolver:
        """GPU-supported JAX double-cable solver policies."""

        auto = staticmethod(DoubleCableSolver.auto)
        tiled_thomas = staticmethod(DoubleCableSolver.tiled_thomas)


def _coerce_double_cable_solver_kind(
    value: DoubleCableSolverKind,
) -> DoubleCableSolverKind:
    if isinstance(value, DoubleCableSolverKind):
        return value
    raise TypeError(
        "double-cable solver kind must be an "
        "axonfleet.runtime.jax.DoubleCableSolverKind value."
    )


runtime_target = RuntimeTarget(RuntimeKind.JAX)
kind = runtime_target.kind
value = runtime_target.value
cpu = _JaxCpuNamespace()
gpu = _JaxGpuNamespace()


__all__ = [
    "DoubleCableSolver",
    "DoubleCableSolverKind",
    "TiledThomasSolverOptions",
    "cpu",
    "gpu",
    "kind",
    "runtime_target",
    "value",
]
