from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ..cable_geometry import Array


class DoubleCableLinearSystem(NamedTuple):
    """Batch-first SoA coefficients/RHS for an exact double-cable solve."""

    a00: Array
    a01: Array
    a10: Array
    a11: Array
    off0: Array
    off1: Array
    rhs0: Array
    rhs1: Array


class DoubleCableLinearSystemXB(NamedTuple):
    """Node-first SoA coefficients/RHS for an exact double-cable solve."""

    a00: Array
    a01: Array
    a10: Array
    a11: Array
    off0: Array
    off1: Array
    rhs0: Array
    rhs1: Array


class DoubleCableLinearSystemStaticTerms(NamedTuple):
    """Static-ish batch terms used to assemble a double-cable linear system."""

    area: Array
    cm_over_dt: Array
    cx_over_dt: Array
    cx_plus_gx: Array
    a00_static: Array
    a11_static: Array
    off_i: Array
    off_e: Array
    background_abs: Array
    zero_abs: Array


class DoubleCableLinearSystemStaticTermsXB(NamedTuple):
    """Static-ish node-first terms used to assemble a double-cable system."""

    area: Array
    cm_over_dt: Array
    cx_over_dt: Array
    cx_plus_gx: Array
    a00_static: Array
    a11_static: Array
    off_i: Array
    off_e: Array
    background_abs: Array
    zero_abs: Array


def batch_double_cable_space(values: Array, *, batch_size: int, nx: int) -> Array:
    """Broadcast scalar or space-only values to batch-first double-cable space."""

    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (batch_size, nx))
    if arr.ndim == 1:
        return jnp.broadcast_to(arr[None, :], (batch_size, nx))
    return arr


def double_cable_space_to_xb(values: Array, *, batch_size: int, nx: int) -> Array:
    """Broadcast or transpose double-cable space values to node-first layout."""

    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (nx, batch_size))
    if arr.ndim == 1:
        if int(arr.shape[0]) != nx:
            raise ValueError(f"Expected space axis of length {nx}, got {arr.shape}.")
        return jnp.broadcast_to(arr[:, None], (nx, batch_size))
    if arr.ndim == 2:
        if tuple(arr.shape) == (nx, batch_size):
            return arr
        if tuple(arr.shape) == (batch_size, nx):
            return jnp.swapaxes(arr, 0, 1)
        if arr.shape[0] in (1, batch_size) and arr.shape[1] in (1, nx):
            return jnp.swapaxes(jnp.broadcast_to(arr, (batch_size, nx)), 0, 1)
    raise ValueError(
        f"Expected scalar, ({nx},), ({batch_size}, {nx}), "
        f"or ({nx}, {batch_size}); got {arr.shape}."
    )


def double_cable_edge_to_xb(values: Array, *, batch_size: int, nx: int) -> Array:
    """Broadcast or transpose double-cable edge values to node-first layout."""

    edge_count = nx - 1
    arr = jnp.asarray(values)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (edge_count, batch_size))
    if arr.ndim == 1:
        if int(arr.shape[0]) != edge_count:
            raise ValueError(
                f"Expected edge axis of length {edge_count}, got {arr.shape}."
            )
        return jnp.broadcast_to(arr[:, None], (edge_count, batch_size))
    if arr.ndim == 2:
        if tuple(arr.shape) == (edge_count, batch_size):
            return arr
        if tuple(arr.shape) == (batch_size, edge_count):
            return jnp.swapaxes(arr, 0, 1)
        if arr.shape[0] in (1, batch_size) and arr.shape[1] in (1, edge_count):
            return jnp.swapaxes(
                jnp.broadcast_to(arr, (batch_size, edge_count)),
                0,
                1,
            )
    raise ValueError(
        f"Expected scalar, ({edge_count},), ({batch_size}, {edge_count}), "
        f"or ({edge_count}, {batch_size}); got {arr.shape}."
    )


def double_cable_space_from_xb(values: Array) -> Array:
    """Convert node-first double-cable space values back to batch-first layout."""

    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"Expected node-first 2D values, got {arr.shape}.")
    return jnp.swapaxes(arr, 0, 1)


def prepare_double_cable_linear_system_static_terms(
    *,
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    dt_ms: Array,
    batch_size: int,
    nx: int,
) -> DoubleCableLinearSystemStaticTerms:
    """Prepare reusable batch-first terms for double-cable system assembly."""

    area = batch_double_cable_space(area_cm2, batch_size=batch_size, nx=nx)
    cm_over_dt = batch_double_cable_space(Cm_abs, batch_size=batch_size, nx=nx) / dt_ms
    cx_over_dt = batch_double_cable_space(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    gx_abs = batch_double_cable_space(Gx_abs, batch_size=batch_size, nx=nx)
    left_i_batch = batch_double_cable_space(left_i, batch_size=batch_size, nx=nx)
    right_i_batch = batch_double_cable_space(right_i, batch_size=batch_size, nx=nx)
    left_e_batch = batch_double_cable_space(left_e, batch_size=batch_size, nx=nx)
    right_e_batch = batch_double_cable_space(right_e, batch_size=batch_size, nx=nx)
    background = batch_double_cable_space(I_background, batch_size=batch_size, nx=nx)

    return DoubleCableLinearSystemStaticTerms(
        area=area,
        cm_over_dt=cm_over_dt,
        cx_over_dt=cx_over_dt,
        cx_plus_gx=cx_over_dt + gx_abs,
        a00_static=cm_over_dt + left_i_batch + right_i_batch,
        a11_static=cm_over_dt + cx_over_dt + gx_abs + left_e_batch + right_e_batch,
        off_i=-jnp.asarray(Gax_i),
        off_e=-jnp.asarray(Gax_e),
        background_abs=background * area,
        zero_abs=jnp.zeros_like(area),
    )


def prepare_double_cable_linear_system_static_terms_xb(
    *,
    area_cm2: Array,
    Cm_abs: Array,
    Cx_abs: Array,
    Gx_abs: Array,
    Gax_e: Array,
    Gax_i: Array,
    left_i: Array,
    right_i: Array,
    left_e: Array,
    right_e: Array,
    I_background: Array,
    dt_ms: Array,
    batch_size: int,
    nx: int,
) -> DoubleCableLinearSystemStaticTermsXB:
    """Prepare reusable node-first terms for double-cable system assembly."""

    area = double_cable_space_to_xb(area_cm2, batch_size=batch_size, nx=nx)
    cm_over_dt = double_cable_space_to_xb(Cm_abs, batch_size=batch_size, nx=nx) / dt_ms
    cx_over_dt = double_cable_space_to_xb(Cx_abs, batch_size=batch_size, nx=nx) / dt_ms
    gx_abs = double_cable_space_to_xb(Gx_abs, batch_size=batch_size, nx=nx)
    left_i_batch = double_cable_space_to_xb(left_i, batch_size=batch_size, nx=nx)
    right_i_batch = double_cable_space_to_xb(right_i, batch_size=batch_size, nx=nx)
    left_e_batch = double_cable_space_to_xb(left_e, batch_size=batch_size, nx=nx)
    right_e_batch = double_cable_space_to_xb(right_e, batch_size=batch_size, nx=nx)
    background = double_cable_space_to_xb(
        I_background,
        batch_size=batch_size,
        nx=nx,
    )

    return DoubleCableLinearSystemStaticTermsXB(
        area=area,
        cm_over_dt=cm_over_dt,
        cx_over_dt=cx_over_dt,
        cx_plus_gx=cx_over_dt + gx_abs,
        a00_static=cm_over_dt + left_i_batch + right_i_batch,
        a11_static=cm_over_dt + cx_over_dt + gx_abs + left_e_batch + right_e_batch,
        off_i=-double_cable_edge_to_xb(Gax_i, batch_size=batch_size, nx=nx),
        off_e=-double_cable_edge_to_xb(Gax_e, batch_size=batch_size, nx=nx),
        background_abs=background * area,
        zero_abs=jnp.zeros_like(area),
    )


def assemble_double_cable_linear_system(
    *,
    Vi: Array,
    Ve: Array,
    Gm_abs: Array,
    GE_abs: Array,
    static: DoubleCableLinearSystemStaticTerms,
    Iinj_abs: Array,
    I_outward_abs: Array,
    I_corr_abs: Array,
    extracellular_drive_abs: Array,
) -> DoubleCableLinearSystem:
    """Assemble the exact double-cable SoA block-tridiagonal system."""

    Vm = Vi - Ve
    cm_plus_gm = static.cm_over_dt + Gm_abs
    membrane_charge = static.cm_over_dt * Vm
    a01 = -cm_plus_gm
    return DoubleCableLinearSystem(
        a00=static.a00_static + Gm_abs,
        a01=a01,
        a10=a01,
        a11=static.a11_static + Gm_abs,
        off0=static.off_i,
        off1=static.off_e,
        rhs0=membrane_charge + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs,
        rhs1=(
            -membrane_charge
            - GE_abs
            + static.cx_over_dt * Ve
            + extracellular_drive_abs
            + I_outward_abs
            + I_corr_abs
        ),
    )


def assemble_double_cable_linear_system_xb(
    *,
    Vi: Array,
    Ve: Array,
    Gm_abs: Array,
    GE_abs: Array,
    static: DoubleCableLinearSystemStaticTermsXB,
    Iinj_abs: Array,
    I_outward_abs: Array,
    I_corr_abs: Array,
    extracellular_drive_abs: Array,
) -> DoubleCableLinearSystemXB:
    """Assemble the exact double-cable SoA system in node-first layout."""

    Vm = Vi - Ve
    cm_plus_gm = static.cm_over_dt + Gm_abs
    membrane_charge = static.cm_over_dt * Vm
    a01 = -cm_plus_gm
    return DoubleCableLinearSystemXB(
        a00=static.a00_static + Gm_abs,
        a01=a01,
        a10=a01,
        a11=static.a11_static + Gm_abs,
        off0=static.off_i,
        off1=static.off_e,
        rhs0=membrane_charge + GE_abs + Iinj_abs - I_outward_abs - I_corr_abs,
        rhs1=(
            -membrane_charge
            - GE_abs
            + static.cx_over_dt * Ve
            + extracellular_drive_abs
            + I_outward_abs
            + I_corr_abs
        ),
    )


def solve_double_cable_physical_system_jax_triton_loop_xb(
    *,
    static: DoubleCableLinearSystemStaticTermsXB,
    Vi: Array,
    Ve: Array,
    Gm_density: Array,
    GE_density: Array,
    Iinj_abs: Array,
    I_outward_abs: Array,
    I_corr_abs: Array,
    extracellular_drive_abs: Array,
    block_b: int = 32,
    return_node_first: bool = False,
) -> tuple[Array, Array]:
    """Assemble and solve a node-first system with the private GPU route."""

    from .triton_double_cable import (
        solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb,
    )

    Vi_xb, Ve_xb = solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb(
        static.a00_static,
        static.a11_static,
        static.cm_over_dt,
        static.cx_over_dt,
        static.off_i,
        static.off_e,
        Vi,
        Ve,
        Gm_density,
        GE_density,
        static.area,
        Iinj_abs,
        I_outward_abs,
        I_corr_abs,
        extracellular_drive_abs,
        block_b=block_b,
    )
    if return_node_first:
        return Vi_xb, Ve_xb
    return double_cable_space_from_xb(Vi_xb), double_cable_space_from_xb(Ve_xb)
