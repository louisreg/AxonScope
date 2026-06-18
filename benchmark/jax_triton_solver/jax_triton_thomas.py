"""Compatibility wrapper for the experimental jax-triton Thomas solver."""

from __future__ import annotations

from axonscope.solvers.jax_triton_thomas import (
    jax_triton_thomas_dependency_skip_reason,
    solve_block_tridiagonal_2x2_jax_triton_thomas_batched,
)


def solve_block_tridiagonal_2x2_jax_triton_thomas(*args, **kwargs):
    return solve_block_tridiagonal_2x2_jax_triton_thomas_batched(*args, **kwargs)


__all__ = [
    "jax_triton_thomas_dependency_skip_reason",
    "solve_block_tridiagonal_2x2_jax_triton_thomas",
    "solve_block_tridiagonal_2x2_jax_triton_thomas_batched",
]
