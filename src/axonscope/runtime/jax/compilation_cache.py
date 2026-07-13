"""AxonScope policy for JAX's persistent executable compilation cache."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})


@dataclass(frozen=True, slots=True)
class JaxCompilationCachePolicy:
    """Resolved process-wide JAX compilation-cache configuration."""

    enabled: bool
    directory: Path | None
    min_compile_time_s: float
    max_size_bytes: int


def configure_jax_compilation_cache(
    *,
    jax_config: Any | None = None,
) -> JaxCompilationCachePolicy:
    """Configure JAX's native persistent cache before AxonScope compiles."""

    policy = jax_compilation_cache_policy()
    if jax_config is None:
        from jax import config as jax_config

    jax_config.update("jax_enable_compilation_cache", policy.enabled)
    if not policy.enabled:
        return policy

    assert policy.directory is not None
    policy.directory.mkdir(parents=True, exist_ok=True)
    jax_config.update("jax_compilation_cache_dir", str(policy.directory))
    jax_config.update(
        "jax_persistent_cache_min_compile_time_secs",
        policy.min_compile_time_s,
    )
    jax_config.update("jax_compilation_cache_max_size", policy.max_size_bytes)
    return policy


def jax_compilation_cache_policy() -> JaxCompilationCachePolicy:
    """Resolve cache location and retention from AxonScope environment policy."""

    configured = os.environ.get("AXONSCOPE_JAX_COMPILATION_CACHE", "").strip()
    enabled = configured.lower() not in _FALSE_VALUES
    directory = None
    if enabled:
        directory = (
            Path(configured).expanduser().resolve()
            if configured
            else (
                Path.cwd()
                / ".axonscope_cache"
                / "runtime"
                / "jax"
                / "xla"
            ).resolve()
        )
    return JaxCompilationCachePolicy(
        enabled=enabled,
        directory=directory,
        min_compile_time_s=_environment_float(
            "AXONSCOPE_JAX_CACHE_MIN_COMPILE_TIME_S",
            default=0.5,
            minimum=0.0,
        ),
        max_size_bytes=_environment_int(
            "AXONSCOPE_JAX_CACHE_MAX_SIZE_BYTES",
            default=2 * 1024**3,
            minimum=-1,
        ),
    )


def _environment_float(name: str, *, default: float, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _environment_int(name: str, *, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


__all__ = [
    "JaxCompilationCachePolicy",
    "configure_jax_compilation_cache",
    "jax_compilation_cache_policy",
]
