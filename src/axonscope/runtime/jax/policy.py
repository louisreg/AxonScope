"""JAX runtime target and JAX-specific solver-policy values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from axonscope.runtime.policy import (
    DoubleCableSolverRequest,
    RuntimeKind,
    RuntimeTarget,
    SingleCableSolverRequest,
)


class SingleCableSolverKind(Enum):
    """JAX single-cable solver-family request."""

    AUTO = "auto"
    JAX_TRIDIAGONAL = "jax_tridiagonal"


class DoubleCableSolverKind(Enum):
    """JAX double-cable solver-family request."""

    AUTO = "auto"
    THOMAS = "thomas"
    JAX_PCR = "jax_pcr"
    JAX_PCR_SOA = "jax_pcr_soa"
    TILED_THOMAS = "tiled_thomas"


@dataclass(frozen=True)
class PcrSolverOptions:
    """Options for JAX PCR-family GPU solvers."""

    adaptive_threshold: int = 4096

    def __post_init__(self) -> None:
        if int(self.adaptive_threshold) < 1:
            raise ValueError("adaptive_threshold must be >= 1.")
        object.__setattr__(self, "adaptive_threshold", int(self.adaptive_threshold))


@dataclass(frozen=True)
class TiledThomasSolverOptions:
    """Options for the tiled-Thomas JAX GPU solver."""

    block_b: int | None = None
    allow_fallback: bool = False
    require_gpu: bool = True

    def __post_init__(self) -> None:
        if self.block_b is not None:
            if int(self.block_b) < 1:
                raise ValueError("block_b must be >= 1.")
            object.__setattr__(self, "block_b", int(self.block_b))
        object.__setattr__(self, "allow_fallback", bool(self.allow_fallback))
        object.__setattr__(self, "require_gpu", bool(self.require_gpu))


@dataclass(frozen=True)
class SingleCableSolver(SingleCableSolverRequest):
    """Typed JAX single-cable solver-family request."""

    kind: SingleCableSolverKind = SingleCableSolverKind.AUTO

    @classmethod
    def auto(cls) -> "SingleCableSolver":
        """Let the JAX runtime choose the supported single-cable route."""

        return cls(SingleCableSolverKind.AUTO)

    @classmethod
    def jax_tridiagonal(cls) -> "SingleCableSolver":
        """Request the current JAX tridiagonal single-cable route."""

        return cls(SingleCableSolverKind.JAX_TRIDIAGONAL)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_single_cable_solver_kind(self.kind))


@dataclass(frozen=True)
class DoubleCableSolver(DoubleCableSolverRequest):
    """Typed JAX double-cable solver-family request."""

    kind: DoubleCableSolverKind = DoubleCableSolverKind.AUTO
    pcr_options: PcrSolverOptions = field(default_factory=PcrSolverOptions)
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
    def pcr(cls, *, adaptive_threshold: int = 4096) -> "DoubleCableSolver":
        """Request the JAX PCR-family double-cable route."""

        return cls(
            DoubleCableSolverKind.JAX_PCR,
            pcr_options=PcrSolverOptions(adaptive_threshold=adaptive_threshold),
        )

    @classmethod
    def pcr_soa(cls, *, adaptive_threshold: int = 4096) -> "DoubleCableSolver":
        """Request the JAX PCR-SoA double-cable route."""

        return cls(
            DoubleCableSolverKind.JAX_PCR_SOA,
            pcr_options=PcrSolverOptions(adaptive_threshold=adaptive_threshold),
        )

    @classmethod
    def tiled_thomas(
        cls,
        *,
        block_b: int | None = None,
        allow_fallback: bool = False,
        require_gpu: bool = True,
    ) -> "DoubleCableSolver":
        """Request the tiled-Thomas double-cable route."""

        return cls(
            DoubleCableSolverKind.TILED_THOMAS,
            tiled_thomas_options=TiledThomasSolverOptions(
                block_b=block_b,
                allow_fallback=allow_fallback,
                require_gpu=require_gpu,
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_double_cable_solver_kind(self.kind))
        if not isinstance(self.pcr_options, PcrSolverOptions):
            raise TypeError("double-cable PCR options must be a PcrSolverOptions value.")
        if not isinstance(self.tiled_thomas_options, TiledThomasSolverOptions):
            raise TypeError(
                "double-cable tiled-Thomas options must be a TiledThomasSolverOptions value."
            )


class _JaxCpuNamespace:
    SingleCableSolver = SingleCableSolver

    class DoubleCableSolver:
        """CPU-supported JAX double-cable solver policies."""

        auto = staticmethod(DoubleCableSolver.auto)
        thomas = staticmethod(DoubleCableSolver.thomas)


class _JaxGpuNamespace:
    SingleCableSolver = SingleCableSolver
    DoubleCableSolver = DoubleCableSolver


def _coerce_single_cable_solver_kind(
    value: SingleCableSolverKind,
) -> SingleCableSolverKind:
    if isinstance(value, SingleCableSolverKind):
        return value
    raise TypeError(
        "single-cable solver kind must be an "
        "axonscope.runtime.jax.SingleCableSolverKind value."
    )


def _coerce_double_cable_solver_kind(
    value: DoubleCableSolverKind,
) -> DoubleCableSolverKind:
    if isinstance(value, DoubleCableSolverKind):
        return value
    raise TypeError(
        "double-cable solver kind must be an "
            "axonscope.runtime.jax.DoubleCableSolverKind value."
    )


runtime_target = RuntimeTarget(RuntimeKind.JAX)
kind = runtime_target.kind
value = runtime_target.value
cpu = _JaxCpuNamespace()
gpu = _JaxGpuNamespace()


__all__ = [
    "DoubleCableSolver",
    "DoubleCableSolverKind",
    "PcrSolverOptions",
    "SingleCableSolver",
    "SingleCableSolverKind",
    "TiledThomasSolverOptions",
    "cpu",
    "gpu",
    "kind",
    "runtime_target",
    "value",
]
