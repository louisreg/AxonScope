from __future__ import annotations

from typing import TypeAlias

import jax.numpy as jnp

Array: TypeAlias = jnp.ndarray


def apply_diffusion_operator(V: Array, lower: Array, upper: Array) -> Array:
    """
    Apply the sealed-end diffusion operator from its off-diagonal coefficients.
    """
    Nx = V.shape[0]
    out = jnp.zeros_like(V)

    if Nx >= 2:
        out = out.at[0].set(upper[0] * (V[1] - V[0]))
        out = out.at[-1].set(lower[-1] * (V[-2] - V[-1]))

    if Nx > 2:
        out = out.at[1:-1].set(
            lower[1:-1] * (V[:-2] - V[1:-1])
            + upper[1:-1] * (V[2:] - V[1:-1])
        )

    return out
