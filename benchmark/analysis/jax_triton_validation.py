"""Numerical validation helpers for benchmark-only Triton kernel changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate_double_cable_tiled_thomas(
    output_path: Path,
    *,
    nx: int = 22,
    batch_size: int = 7,
    block_b: int = 32,
    seed: int = 17,
) -> dict[str, float | int | bool]:
    """Compare the active GPU kernel with independent dense NumPy solves."""

    import jax
    import jax.numpy as jnp

    from axonfleet.runtime.jax.kernels.triton_double_cable import (
        solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb,
    )

    rng = np.random.default_rng(seed)
    shape = (nx, batch_size)
    edge_shape = (nx - 1, batch_size)
    a00_static = 4.0 + 0.2 * rng.random(shape)
    a11_static = 5.0 + 0.2 * rng.random(shape)
    cm_over_dt = 0.15 + 0.02 * rng.random(shape)
    cx_over_dt = 0.08 + 0.01 * rng.random(shape)
    off0 = -0.35 - 0.02 * rng.random(edge_shape)
    off1 = -0.30 - 0.02 * rng.random(edge_shape)
    vi = -70.0 + rng.normal(size=shape)
    ve = rng.normal(size=shape)
    gm_density = 0.01 + 0.005 * rng.random(shape)
    ge_density = rng.normal(size=shape)
    area = 0.5 + 0.1 * rng.random(shape)
    iinj_abs = rng.normal(size=shape)
    i_outward_abs = rng.normal(size=shape)
    i_corr_abs = rng.normal(size=shape)
    extracellular_drive_abs = rng.normal(size=shape)

    gm_abs = gm_density * area
    ge_abs = ge_density * area
    charge = cm_over_dt * (vi - ve)
    a00 = a00_static + gm_abs
    a01 = -(cm_over_dt + gm_abs)
    a10 = a01
    a11 = a11_static + gm_abs
    rhs0 = charge + ge_abs + iinj_abs - i_outward_abs - i_corr_abs
    rhs1 = (
        -charge
        - ge_abs
        + cx_over_dt * ve
        + extracellular_drive_abs
        + i_outward_abs
        + i_corr_abs
    )

    gpu0, gpu1 = solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb(
        *(
            jnp.asarray(value, dtype=jnp.float32)
            for value in (
                a00_static,
                a11_static,
                cm_over_dt,
                cx_over_dt,
                off0,
                off1,
                vi,
                ve,
                gm_density,
                ge_density,
                area,
                iinj_abs,
                i_outward_abs,
                i_corr_abs,
                extracellular_drive_abs,
            )
        ),
        block_b=block_b,
    )
    jax.block_until_ready((gpu0, gpu1))
    actual0 = np.asarray(gpu0)
    actual1 = np.asarray(gpu1)

    expected0 = np.empty(shape, dtype=np.float64)
    expected1 = np.empty(shape, dtype=np.float64)
    for batch_index in range(batch_size):
        matrix = np.zeros((2 * nx, 2 * nx), dtype=np.float64)
        rhs = np.empty((2 * nx,), dtype=np.float64)
        for x_index in range(nx):
            row = 2 * x_index
            matrix[row, row] = a00[x_index, batch_index]
            matrix[row, row + 1] = a01[x_index, batch_index]
            matrix[row + 1, row] = a10[x_index, batch_index]
            matrix[row + 1, row + 1] = a11[x_index, batch_index]
            rhs[row] = rhs0[x_index, batch_index]
            rhs[row + 1] = rhs1[x_index, batch_index]
            if x_index + 1 < nx:
                matrix[row, row + 2] = off0[x_index, batch_index]
                matrix[row + 2, row] = off0[x_index, batch_index]
                matrix[row + 1, row + 3] = off1[x_index, batch_index]
                matrix[row + 3, row + 1] = off1[x_index, batch_index]
        expected = np.linalg.solve(matrix, rhs)
        expected0[:, batch_index] = expected[0::2]
        expected1[:, batch_index] = expected[1::2]

    max_abs_error = float(
        max(
            np.max(np.abs(actual0 - expected0)),
            np.max(np.abs(actual1 - expected1)),
        )
    )
    scale = float(max(np.max(np.abs(expected0)), np.max(np.abs(expected1)), 1.0))
    payload: dict[str, float | int | bool] = {
        "nx": nx,
        "batch_size": batch_size,
        "block_b": block_b,
        "max_abs_error": max_abs_error,
        "max_scaled_error": max_abs_error / scale,
        "passed": bool(
            np.allclose(actual0, expected0, rtol=5e-5, atol=5e-5)
            and np.allclose(actual1, expected1, rtol=5e-5, atol=5e-5)
        ),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise AssertionError(
            "Triton double-cable validation failed: "
            f"max_abs_error={max_abs_error:.6g}"
        )
    return payload


__all__ = ["validate_double_cable_tiled_thomas"]
