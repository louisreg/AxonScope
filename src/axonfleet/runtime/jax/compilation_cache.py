"""AxonFleet policy for JAX's persistent executable compilation cache."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axonfleet.cache import persistent_directory

@dataclass(frozen=True, slots=True)
class JaxCompilationCachePolicy:
    """Resolved process-wide JAX compilation-cache configuration."""

    enabled: bool
    directory: Path | None
    min_compile_time_s: float
    min_entry_size_bytes: int
    max_size_bytes: int
    xla_caches: str


def configure_jax_compilation_cache(
    *,
    jax_config: Any | None = None,
) -> JaxCompilationCachePolicy:
    """Configure JAX's native persistent cache before AxonFleet compiles."""

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
    jax_config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        policy.min_entry_size_bytes,
    )
    jax_config.update("jax_compilation_cache_max_size", policy.max_size_bytes)
    jax_config.update(
        "jax_persistent_cache_enable_xla_caches",
        policy.xla_caches,
    )
    return policy


def jax_compilation_cache_policy() -> JaxCompilationCachePolicy:
    """Resolve cache location and retention from AxonFleet environment policy."""

    directory = persistent_directory("runtime", "jax", "xla")
    enabled = directory is not None
    return JaxCompilationCachePolicy(
        enabled=enabled,
        directory=directory,
        min_compile_time_s=_environment_float(
            "AXONFLEET_JAX_CACHE_MIN_COMPILE_TIME_S",
            default=0.5,
            minimum=0.0,
        ),
        min_entry_size_bytes=_environment_int(
            "AXONFLEET_JAX_CACHE_MIN_ENTRY_SIZE_BYTES",
            default=0,
            minimum=-1,
        ),
        max_size_bytes=_environment_int(
            "AXONFLEET_JAX_CACHE_MAX_SIZE_BYTES",
            default=2 * 1024**3,
            minimum=-1,
        ),
        xla_caches=_environment_choice(
            "AXONFLEET_JAX_PERSISTENT_XLA_CACHES",
            default="xla_gpu_per_fusion_autotune_cache_dir",
            choices=frozenset(
                {
                    "all",
                    "none",
                    "xla_gpu_kernel_cache_file",
                    "xla_gpu_per_fusion_autotune_cache_dir",
                }
            ),
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


def _environment_choice(
    name: str,
    *,
    default: str,
    choices: frozenset[str],
) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {expected}; got {value!r}.")
    return value


__all__ = [
    "JaxCompilationCachePolicy",
    "configure_jax_compilation_cache",
    "jax_compilation_cache_policy",
]
