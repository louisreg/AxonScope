"""JAX execution-policy resolution.

This module is the runtime-owned bridge from public typed execution requests to
JAX runtime controls. Public layers should pass an ``ExecutionPolicy`` here
instead of importing JAX directly.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import jax
import numpy as np

from axonscope.benchmarking import benchmark_span, record_benchmark_metadata
from axonscope.axon_instance import AxonInstance
from axonscope.runtime.jax.policy.engine import (
    JaxSolverEngine,
    resolve_jax_solver_engine,
)
from axonscope.runtime import (
    Device,
    ExecutionPolicy,
    PrecisionPolicy,
    numpy as runtime_numpy,
)
from axonscope.runtime.policy import RuntimeKind


@dataclass(frozen=True)
class JaxExecutionContext:
    """Resolved JAX execution request used by orchestration code."""

    policy: ExecutionPolicy | None
    device: Any | None
    platform: str | None
    solver_engine: JaxSolverEngine | None = None


@dataclass(frozen=True)
class _ResolvedJaxExecutionPolicy:
    """Cached execution-policy lowering independent of the current cohort."""

    device: Any | None
    platform: str | None
    solver_engine: JaxSolverEngine | None


_RESOLVED_EXECUTION_POLICY_CACHE: OrderedDict[
    ExecutionPolicy,
    _ResolvedJaxExecutionPolicy,
] = OrderedDict()
_RESOLVED_EXECUTION_POLICY_CACHE_MAX_SIZE = 32
_PRECISION_VALIDATION_CACHE: OrderedDict[
    tuple[int, str, str, str],
    tuple[AxonInstance, ...],
] = OrderedDict()
_PRECISION_VALIDATION_CACHE_MAX_SIZE = 32


@contextmanager
def jax_execution_context(
    policy: ExecutionPolicy | None,
    *,
    instances: Sequence[AxonInstance],
) -> Iterator[JaxExecutionContext]:
    """Apply a public execution policy to the JAX runtime for one run."""

    if policy is None:
        yield JaxExecutionContext(policy=None, device=None, platform=None)
        return

    if not isinstance(policy, ExecutionPolicy):
        raise TypeError("execution_policy must be an axonscope.ExecutionPolicy value.")
    if policy.runtime is runtime_numpy:
        raise NotImplementedError(
            "axs.runtime.numpy is not implemented for executable simulations yet; "
            "use axs.runtime.auto or axs.runtime.jax."
        )
    if policy.runtime.kind not in {RuntimeKind.AUTO, RuntimeKind.JAX}:
        raise ValueError(f"Unsupported runtime: {policy.runtime!r}.")

    _validate_precision(policy.precision, instances=instances)

    resolved = _resolve_jax_execution_policy(policy)
    device = resolved.device
    solver_engine = resolved.solver_engine
    context = JaxExecutionContext(
        policy=policy,
        device=device,
        platform=resolved.platform,
        solver_engine=solver_engine,
    )
    if device is None:
        yield context
        return

    manager = jax.default_device(device)
    with benchmark_span("runtime.jax.execution.default_device.enter"):
        manager.__enter__()
    try:
        yield context
    except BaseException as exc:
        with benchmark_span("runtime.jax.execution.default_device.exit"):
            manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        with benchmark_span("runtime.jax.execution.default_device.exit"):
            manager.__exit__(None, None, None)


def _resolve_jax_execution_policy(
    policy: ExecutionPolicy,
) -> _ResolvedJaxExecutionPolicy:
    cached = _RESOLVED_EXECUTION_POLICY_CACHE.get(policy)
    if cached is not None:
        _RESOLVED_EXECUTION_POLICY_CACHE.move_to_end(policy)
        record_benchmark_metadata(jax_execution_policy_cache="hit")
        return cached

    with benchmark_span("runtime.jax.execution.resolve_policy"):
        device = _resolve_device(policy.device)
        platform = _platform_for_device(policy.device, device)
        solver_engine = resolve_jax_solver_engine(policy, platform=platform)
        resolved = _ResolvedJaxExecutionPolicy(
            device=device,
            platform=platform,
            solver_engine=solver_engine,
        )
    _RESOLVED_EXECUTION_POLICY_CACHE[policy] = resolved
    _RESOLVED_EXECUTION_POLICY_CACHE.move_to_end(policy)
    while (
        len(_RESOLVED_EXECUTION_POLICY_CACHE)
        > _RESOLVED_EXECUTION_POLICY_CACHE_MAX_SIZE
    ):
        _RESOLVED_EXECUTION_POLICY_CACHE.popitem(last=False)
    record_benchmark_metadata(jax_execution_policy_cache="miss")
    return resolved


def clear_jax_execution_policy_cache() -> None:
    """Clear cached JAX execution-policy lowering."""

    _RESOLVED_EXECUTION_POLICY_CACHE.clear()


def clear_jax_precision_validation_cache() -> None:
    """Clear cached exact-pool precision validation results."""

    _PRECISION_VALIDATION_CACHE.clear()


def clear_jax_execution_caches() -> None:
    """Clear cached JAX execution-policy and validation lowering."""

    clear_jax_execution_policy_cache()
    clear_jax_precision_validation_cache()


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
        return _canonical_jax_platform(jax.default_backend())
    platform = getattr(device, "platform", None)
    return None if platform is None else _canonical_jax_platform(str(platform))


def _canonical_jax_platform(platform: str) -> str:
    normalized = str(platform).lower()
    if normalized in {"cuda", "gpu", "rocm", "metal"}:
        return "gpu"
    if normalized == "cpu":
        return "cpu"
    return normalized


def jax_solver_engine_for_policy(
    policy: ExecutionPolicy | None,
) -> JaxSolverEngine | None:
    """Return the JAX solver engine selected by a public policy."""

    if policy is None:
        return None
    platform = (
        policy.device.kind
        if policy.device.kind in {"cpu", "gpu"}
        else _canonical_jax_platform(jax.default_backend())
    )
    return resolve_jax_solver_engine(policy, platform=platform)


def _validate_precision(
    precision: PrecisionPolicy | None,
    *,
    instances: Sequence[AxonInstance],
) -> None:
    if precision is None:
        return
    cache_key = _precision_validation_cache_key(precision, instances)
    if cache_key is not None:
        cached_instances = _PRECISION_VALIDATION_CACHE.get(cache_key)
        if cached_instances is instances:
            _PRECISION_VALIDATION_CACHE.move_to_end(cache_key)
            record_benchmark_metadata(jax_precision_validation_cache="hit")
            return

    with benchmark_span("runtime.jax.execution.validate_precision"):
        _validate_precision_uncached(precision, instances=instances)

    if cache_key is not None:
        _PRECISION_VALIDATION_CACHE[cache_key] = instances
        _PRECISION_VALIDATION_CACHE.move_to_end(cache_key)
        while len(_PRECISION_VALIDATION_CACHE) > _PRECISION_VALIDATION_CACHE_MAX_SIZE:
            _PRECISION_VALIDATION_CACHE.popitem(last=False)
        record_benchmark_metadata(jax_precision_validation_cache="miss")


def _precision_validation_cache_key(
    precision: PrecisionPolicy,
    instances: Sequence[AxonInstance],
) -> tuple[int, str, str, str] | None:
    if not isinstance(instances, tuple):
        return None
    return (
        id(instances),
        precision.state_dtype,
        precision.solver_dtype,
        precision.accumulation_dtype,
    )


def _validate_precision_uncached(
    precision: PrecisionPolicy,
    *,
    instances: Sequence[AxonInstance],
) -> None:
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


__all__ = [
    "clear_jax_execution_caches",
    "clear_jax_execution_policy_cache",
    "clear_jax_precision_validation_cache",
    "JaxExecutionContext",
    "jax_execution_context",
    "jax_solver_engine_for_policy",
]
