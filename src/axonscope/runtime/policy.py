"""Public runtime policy types shared by concrete runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


class RuntimeKind(Enum):
    """Known execution runtime family."""

    AUTO = "auto"
    JAX = "jax"
    NUMPY = "numpy"


@dataclass(frozen=True)
class RuntimeTarget:
    """Concrete runtime target used by execution policy."""

    kind: RuntimeKind

    @property
    def value(self) -> str:
        """Stable string label for reports and cache metadata."""

        return self.kind.value

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RuntimeKind):
            raise TypeError("runtime target kind must be a RuntimeKind value.")

    def __repr__(self) -> str:
        return f"axs.runtime.{self.value}"


DeviceKind = Literal["auto", "cpu", "gpu"]


@dataclass(frozen=True)
class Device:
    """Structured runtime device request."""

    kind: DeviceKind
    index: int | None = None

    @classmethod
    def auto(cls) -> "Device":
        """Let AxonScope or the backend choose a device."""

        return cls("auto")

    @classmethod
    def cpu(cls) -> "Device":
        """Request CPU execution."""

        return cls("cpu")

    @classmethod
    def gpu(cls, index: int = 0) -> "Device":
        """Request one GPU device by index."""

        return cls("gpu", int(index))

    def __post_init__(self) -> None:
        if self.kind not in {"auto", "cpu", "gpu"}:
            raise ValueError("Device kind must be 'auto', 'cpu', or 'gpu'.")
        if self.kind == "gpu":
            if self.index is None or int(self.index) < 0:
                raise ValueError("GPU device index must be >= 0.")
            object.__setattr__(self, "index", int(self.index))
        elif self.index is not None:
            raise ValueError("Only GPU devices accept an index.")


@dataclass(frozen=True)
class PrecisionPolicy:
    """Dtype policy used by estimators and runtime lowering."""

    state_dtype: str
    solver_dtype: str
    accumulation_dtype: str

    @classmethod
    def float32(cls) -> "PrecisionPolicy":
        """Use float32 for state, solver inputs, and reductions."""

        return cls("float32", "float32", "float32")

    @classmethod
    def float64(cls) -> "PrecisionPolicy":
        """Use float64 for state, solver inputs, and reductions."""

        return cls("float64", "float64", "float64")

    @classmethod
    def mixed(
        cls,
        *,
        state_dtype: Any = "float32",
        solver_dtype: Any = "float32",
        accumulation_dtype: Any = "float64",
    ) -> "PrecisionPolicy":
        """Build an explicit mixed-precision policy."""

        return cls(
            _dtype_name(state_dtype),
            _dtype_name(solver_dtype),
            _dtype_name(accumulation_dtype),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_dtype", _dtype_name(self.state_dtype))
        object.__setattr__(self, "solver_dtype", _dtype_name(self.solver_dtype))
        object.__setattr__(
            self,
            "accumulation_dtype",
            _dtype_name(self.accumulation_dtype),
        )


class SingleCableSolverRequest:
    """Base class for runtime-specific single-cable solver requests."""


class DoubleCableSolverRequest:
    """Base class for runtime-specific double-cable solver requests."""


auto = RuntimeTarget(RuntimeKind.AUTO)
numpy = RuntimeTarget(RuntimeKind.NUMPY)


@dataclass(frozen=True)
class SolverPolicy:
    """Typed per-cable solver-family request for executable simulations."""

    single_cable: SingleCableSolverRequest | None = None
    double_cable: DoubleCableSolverRequest | None = None

    def __post_init__(self) -> None:
        if self.single_cable is not None and not isinstance(
            self.single_cable,
            SingleCableSolverRequest,
        ):
            raise TypeError(
                "solver policy single_cable must be a runtime-specific "
                "single-cable solver request."
            )
        if self.double_cable is not None and not isinstance(
            self.double_cable,
            DoubleCableSolverRequest,
        ):
            raise TypeError(
                "solver policy double_cable must be a runtime-specific "
                "double-cable solver request."
            )


@dataclass(frozen=True)
class ExecutionPolicy:
    """Typed runtime, device, precision, and solver request."""

    runtime: Any = auto
    device: Device = field(default_factory=Device.auto)
    precision: PrecisionPolicy | None = None
    solvers: SolverPolicy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", coerce_runtime(self.runtime))
        if not isinstance(self.device, Device):
            raise TypeError("execution_policy.device must be an axonscope.Device value.")
        if self.precision is not None and not isinstance(self.precision, PrecisionPolicy):
            raise TypeError(
                "execution_policy.precision must be an axonscope.PrecisionPolicy value."
            )
        if self.solvers is not None and not isinstance(self.solvers, SolverPolicy):
            raise TypeError(
                "execution_policy.solvers must be an axonscope.SolverPolicy value."
            )

    @property
    def solver_policy(self) -> SolverPolicy:
        """Return the explicit or default solver policy."""

        return SolverPolicy() if self.solvers is None else self.solvers


def coerce_runtime(value: Any) -> Any:
    """Return the canonical runtime singleton for a public runtime request."""

    target = getattr(value, "runtime_target", None)
    if isinstance(target, RuntimeTarget):
        return value
    if not isinstance(value, RuntimeTarget):
        raise TypeError("runtime must be an axonscope.runtime target value.")
    if value.kind is RuntimeKind.AUTO:
        return auto
    if value.kind is RuntimeKind.JAX:
        return value
    if value.kind is RuntimeKind.NUMPY:
        return numpy
    raise ValueError(f"Unsupported runtime target: {value!r}.")


def _dtype_name(value: Any) -> str:
    return str(np.dtype(value))


__all__ = [
    "Device",
    "DoubleCableSolverRequest",
    "ExecutionPolicy",
    "PrecisionPolicy",
    "RuntimeKind",
    "RuntimeTarget",
    "SingleCableSolverRequest",
    "SolverPolicy",
    "auto",
    "coerce_runtime",
    "numpy",
]
