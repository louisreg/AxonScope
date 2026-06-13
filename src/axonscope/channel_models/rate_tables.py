from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from axonscope.utils.settings import dtype


@dataclass(frozen=True)
class RateTableConfig:
    """Voltage grid used to tabulate alpha/beta rate constants."""

    v_min_mV: float = -120.0
    v_max_mV: float = 80.0
    step_mV: float = 0.05
    clamp: bool = True

    def validate(self) -> None:
        if self.step_mV <= 0:
            raise ValueError(f"step_mV must be positive, got {self.step_mV}.")
        if self.v_max_mV <= self.v_min_mV:
            raise ValueError(
                "v_max_mV must be greater than v_min_mV, "
                f"got {self.v_min_mV}..{self.v_max_mV}."
            )


@dataclass(frozen=True)
class RateTable:
    """Precomputed alpha/beta rates and linear interpolation metadata."""

    config: RateTableConfig
    alpha: jnp.ndarray
    beta: jnp.ndarray
    resolved_step_mV: float

    @classmethod
    def build(
        cls,
        config: RateTableConfig,
        *,
        dtype_local: jnp.dtype,
        exact_rate_constants: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    ) -> "RateTable":
        config.validate()
        n_points = int(
            np.ceil((float(config.v_max_mV) - float(config.v_min_mV)) / float(config.step_mV))
        ) + 1
        if n_points < 2:
            raise ValueError("rate table must contain at least two voltage points.")

        grid = jnp.linspace(
            dtype_local(config.v_min_mV),
            dtype_local(config.v_max_mV),
            n_points,
            dtype=dtype_local,
        )
        alpha, beta = exact_rate_constants(grid)
        return cls(
            config=config,
            alpha=jnp.asarray(alpha, dtype=dtype_local),
            beta=jnp.asarray(beta, dtype=dtype_local),
            resolved_step_mV=float(config.v_max_mV - config.v_min_mV) / float(n_points - 1),
        )

    def interpolate(self, V_mV: jnp.ndarray, *, dtype_local: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
        V = jnp.atleast_1d(jnp.asarray(V_mV, dtype=dtype_local))
        if self.config.clamp:
            V = jnp.clip(V, dtype_local(self.config.v_min_mV), dtype_local(self.config.v_max_mV))

        position = (V - dtype_local(self.config.v_min_mV)) / dtype_local(self.resolved_step_mV)
        lower = jnp.floor(position).astype(jnp.int32)
        lower = jnp.clip(lower, 0, int(self.alpha.shape[0]) - 2)
        upper = lower + 1
        weight = (position - lower.astype(dtype_local))[:, None]

        alpha = self.alpha[lower] + (self.alpha[upper] - self.alpha[lower]) * weight
        beta = self.beta[lower] + (self.beta[upper] - self.beta[lower]) * weight
        return alpha, beta


def make_rate_table_config(
    config: RateTableConfig | None = None,
    *,
    v_min_mV: float = -120.0,
    v_max_mV: float = 80.0,
    step_mV: float = 0.05,
    clamp: bool = True,
) -> RateTableConfig:
    if config is not None:
        config.validate()
        return config
    resolved = RateTableConfig(
        v_min_mV=float(v_min_mV),
        v_max_mV=float(v_max_mV),
        step_mV=float(step_mV),
        clamp=bool(clamp),
    )
    resolved.validate()
    return resolved


def enable_rate_tables(
    model: Any,
    config: RateTableConfig | None = None,
    *,
    recursive: bool | None = None,
    v_min_mV: float = -120.0,
    v_max_mV: float = 80.0,
    step_mV: float = 0.05,
    clamp: bool = True,
) -> int:
    """Enable alpha/beta lookup tables on a membrane model.

    Composite membrane models can usually be tabulated as one aggregate table.
    Heterogeneous compartment layouts need per-submodel tables instead; when
    `recursive` is left to `None`, this helper detects that case from the
    membrane facade shape.
    """
    resolved = make_rate_table_config(
        config,
        v_min_mV=v_min_mV,
        v_max_mV=v_max_mV,
        step_mV=step_mV,
        clamp=clamp,
    )

    if recursive is None:
        recursive = _looks_like_heterogeneous_membrane(model)
    if recursive:
        return _enable_unique_children(model, resolved)

    enable = getattr(model, "enable_rate_table", None)
    if callable(enable):
        enable(config=resolved)
        return 1
    return 0


def _looks_like_heterogeneous_membrane(model: Any) -> bool:
    return hasattr(model, "layout") and hasattr(model, "backend") and hasattr(model, "models")


def _enable_unique_children(model: Any, config: RateTableConfig) -> int:
    count = 0
    seen: set[int] = set()
    for child in getattr(model, "models", ()):
        ident = id(child)
        if ident in seen:
            continue
        seen.add(ident)
        enable = getattr(child, "enable_rate_table", None)
        if callable(enable):
            enable(config=config)
            count += 1
    return count


def disable_rate_tables(model: Any, *, recursive: bool = False) -> int:
    """Disable lookup tables on a model, optionally walking child models."""
    count = 0
    if recursive:
        seen: set[int] = set()
        for child in getattr(model, "models", ()):
            ident = id(child)
            if ident in seen:
                continue
            seen.add(ident)
            disable = getattr(child, "disable_rate_table", None)
            if callable(disable):
                disable()
                count += 1
        return count

    disable = getattr(model, "disable_rate_table", None)
    if callable(disable):
        disable()
        count += 1
    return count
