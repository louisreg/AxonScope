"""JAX execution-policy resolution.

This module is the backend-owned bridge from public typed execution requests to
JAX runtime controls. Public layers should pass an ``ExecutionPolicy`` here
instead of importing JAX directly.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import jax
import numpy as np

from axonscope.axon_instance import AxonInstance
from axonscope.performance import Device, ExecutionPolicy, PrecisionPolicy, Runtime


@dataclass(frozen=True)
class JaxExecutionContext:
    """Resolved JAX execution request used by orchestration code."""

    policy: ExecutionPolicy | None
    device: Any | None
    platform: str | None


@contextmanager
def jax_execution_context(
    policy: ExecutionPolicy | None,
    *,
    instances: Sequence[AxonInstance],
) -> Iterator[JaxExecutionContext]:
    """Apply a public execution policy to the JAX backend for one run."""

    if policy is None:
        yield JaxExecutionContext(policy=None, device=None, platform=None)
        return

    if not isinstance(policy, ExecutionPolicy):
        raise TypeError("execution_policy must be an axonscope.ExecutionPolicy value.")
    if policy.runtime == Runtime.NUMPY:
        raise NotImplementedError(
            "Runtime.NUMPY is not implemented for executable simulations yet; "
            "use Runtime.AUTO or Runtime.JAX."
        )
    if policy.runtime not in {Runtime.AUTO, Runtime.JAX}:
        raise ValueError(f"Unsupported runtime: {policy.runtime!r}.")

    _validate_precision(policy.precision, instances=instances)
    device = _resolve_device(policy.device)
    platform = _platform_for_device(policy.device, device)
    manager = jax.default_device(device) if device is not None else nullcontext()
    with manager:
        yield JaxExecutionContext(policy=policy, device=device, platform=platform)


def _resolve_device(request: Device) -> Any | None:
    if request.kind == "auto":
        return None
    try:
        devices = list(jax.devices(request.kind))
    except RuntimeError as exc:
        raise RuntimeError(f"Requested JAX device kind {request.kind!r} is unavailable.") from exc
    if not devices:
        raise RuntimeError(f"Requested JAX device kind {request.kind!r} is unavailable.")
    if request.kind == "cpu":
        return devices[0]
    assert request.index is not None
    if request.index >= len(devices):
        raise RuntimeError(
            f"Requested GPU index {request.index} but only {len(devices)} GPU device(s) are visible."
        )
    return devices[request.index]


def _platform_for_device(request: Device, device: Any | None) -> str | None:
    if request.kind in {"cpu", "gpu"}:
        return request.kind
    if device is None:
        return None
    platform = getattr(device, "platform", None)
    return None if platform is None else str(platform)


def _validate_precision(
    precision: PrecisionPolicy | None,
    *,
    instances: Sequence[AxonInstance],
) -> None:
    if precision is None:
        return

    requested = {
        "state_dtype": np.dtype(precision.state_dtype),
        "solver_dtype": np.dtype(precision.solver_dtype),
        "accumulation_dtype": np.dtype(precision.accumulation_dtype),
    }
    if any(dtype.itemsize > np.dtype("float32").itemsize for dtype in requested.values()):
        if not bool(jax.config.read("jax_enable_x64")):
            raise RuntimeError(
                "ExecutionPolicy requested float64 precision, but JAX was initialized "
                "with jax_enable_x64=False."
            )

    if len(set(requested.values())) != 1:
        raise NotImplementedError(
            "Mixed-precision ExecutionPolicy values are estimate-only for now; "
            "use PrecisionPolicy.float32() or PrecisionPolicy.float64() for execution."
        )

    solver_dtype = requested["solver_dtype"]
    mismatches = sorted(
        {
            str(dtype)
            for instance in instances
            for dtype in _instance_membrane_dtypes(instance)
            if dtype != solver_dtype
        }
    )
    if mismatches:
        raise ValueError(
            "ExecutionPolicy precision does not match the axon membrane dtype(s): "
            f"requested solver={solver_dtype}; found {mismatches}. "
            "AxonScope does not cast membrane/runtime precision implicitly; "
            "rebuild the axon/membrane with the requested dtype or omit precision."
        )


def _instance_membrane_dtypes(instance: AxonInstance) -> set[np.dtype]:
    return {
        np.dtype(section.membrane.dtype)
        for section in instance.axon.layout.sections
    }


__all__ = ["JaxExecutionContext", "jax_execution_context"]
