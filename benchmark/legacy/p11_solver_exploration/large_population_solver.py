from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp

from axonscope.runtime.jax.kernels.common import (
    solve_block_tridiagonal_2x2_pcr_soa_batched,
)
from benchmark.analysis.double_cable_solver_candidates import (
    solve_block_tridiagonal_2x2_pcr_soa_batched_transposed,
)


Array: TypeAlias = jnp.ndarray
LargePopulationLayoutName = Literal["BX", "XB", "TILED"]

DEFAULT_LARGE_POPULATION_NX_BUCKETS: tuple[int, ...] = (
    32,
    48,
    64,
    80,
    96,
    128,
    160,
    192,
    256,
)


class LargePopulationLayoutPlan(NamedTuple):
    """Backend-private layout contract for large-population double-cable gates."""

    layout: LargePopulationLayoutName
    batch_size: int
    batch_padded: int
    nx_true: int
    nx_pad: int
    block_b: int
    n_tiles: int


def select_large_population_nx_bucket(
    nx: int,
    *,
    buckets: tuple[int, ...] = DEFAULT_LARGE_POPULATION_NX_BUCKETS,
) -> int:
    """Return the smallest large-population bucket that can hold ``nx`` rows."""

    value = int(nx)
    if value < 1:
        raise ValueError("nx must be >= 1.")
    ordered = tuple(sorted(int(bucket) for bucket in buckets))
    for bucket in ordered:
        if value <= bucket:
            return bucket
    raise ValueError(f"nx={value} exceeds supported large-population buckets {ordered}.")


def block_b_candidates_for_nx_bucket(nx_pad: int) -> tuple[int, ...]:
    """Return benchmark candidates for the axon tile width at a given bucket."""

    bucket = int(nx_pad)
    if bucket < 1:
        raise ValueError("nx_pad must be >= 1.")
    if bucket <= 64:
        return (64, 128, 256)
    if bucket <= 128:
        return (32, 64, 128)
    return (16, 32, 64)


def make_large_population_layout_plan(
    *,
    batch_size: int,
    nx_true: int,
    nx_pad: int | None = None,
    block_b: int = 64,
    layout: LargePopulationLayoutName = "TILED",
    buckets: tuple[int, ...] = DEFAULT_LARGE_POPULATION_NX_BUCKETS,
) -> LargePopulationLayoutPlan:
    """Build a static layout plan for benchmark-private large-population solves."""

    batch = int(batch_size)
    nx = int(nx_true)
    block = int(block_b)
    if batch < 1:
        raise ValueError("batch_size must be >= 1.")
    if nx < 1:
        raise ValueError("nx_true must be >= 1.")
    if block < 1:
        raise ValueError("block_b must be >= 1.")
    if layout not in ("BX", "XB", "TILED"):
        raise ValueError(f"unsupported large-population layout: {layout!r}.")
    target_nx = select_large_population_nx_bucket(nx, buckets=buckets) if nx_pad is None else int(nx_pad)
    if target_nx < nx:
        raise ValueError(f"nx_pad must be >= nx_true={nx}, got {target_nx}.")
    n_tiles = (batch + block - 1) // block
    return LargePopulationLayoutPlan(
        layout=layout,
        batch_size=batch,
        batch_padded=n_tiles * block,
        nx_true=nx,
        nx_pad=target_nx,
        block_b=block,
        n_tiles=n_tiles,
    )


def pad_large_population_double_cable_system(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    plan: LargePopulationLayoutPlan,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Pad a batch-first double-cable system to a large-population layout plan."""

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )
    if int(rhs0.shape[0]) != plan.batch_size or int(rhs0.shape[1]) != plan.nx_true:
        raise ValueError(
            "layout plan does not match RHS shape: "
            f"plan=({plan.batch_size}, {plan.nx_true}), rhs={rhs0.shape}."
        )

    a00_b = _pad_space(_as_batch_space(a00, plan=plan), plan=plan, fill=1.0)
    a01_b = _pad_space(_as_batch_space(a01, plan=plan), plan=plan, fill=0.0)
    a10_b = _pad_space(_as_batch_space(a10, plan=plan), plan=plan, fill=0.0)
    a11_b = _pad_space(_as_batch_space(a11, plan=plan), plan=plan, fill=1.0)
    off0_b = _pad_edges(_as_batch_edges(off0, plan=plan), plan=plan, fill=0.0)
    off1_b = _pad_edges(_as_batch_edges(off1, plan=plan), plan=plan, fill=0.0)
    rhs0_b = _pad_space(rhs0, plan=plan, fill=0.0)
    rhs1_b = _pad_space(rhs1, plan=plan, fill=0.0)
    return a00_b, a01_b, a10_b, a11_b, off0_b, off1_b, rhs0_b, rhs1_b


def solve_large_population_exact_double_cable_jax(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    nx_pad: int | None = None,
    block_b: int = 64,
    layout: LargePopulationLayoutName = "TILED",
) -> tuple[Array, Array]:
    """Benchmark-private exact double-cable solve for large GPU populations.

    The current JAX prototype validates the P11C layout contract. It pads
    ``Nx`` and ``B`` explicitly, reshapes the solve into axon tiles, runs the
    exact PCR-SoA block solver per tile, then slices back to the true system.
    This is intentionally backend-private and should not be routed through the
    public solver options unless the P11C benchmark gates justify it.
    """

    rhs0 = jnp.asarray(rhs0)
    if rhs0.ndim != 2:
        raise ValueError("rhs0 must have shape (batch_size, Nx).")
    plan = make_large_population_layout_plan(
        batch_size=int(rhs0.shape[0]),
        nx_true=int(rhs0.shape[1]),
        nx_pad=nx_pad,
        block_b=block_b,
        layout=layout,
    )
    padded = pad_large_population_double_cable_system(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        plan=plan,
    )
    if plan.layout == "TILED":
        x0, x1 = _solve_tiled_bx(*padded, plan=plan)
    elif plan.layout == "BX":
        x0, x1 = solve_block_tridiagonal_2x2_pcr_soa_batched(*padded)
    elif plan.layout == "XB":
        x0, x1 = _solve_xb_packed(*padded)
    else:  # pragma: no cover - guarded by make_large_population_layout_plan.
        raise ValueError(f"unsupported large-population layout: {plan.layout!r}.")
    return x0[: plan.batch_size, : plan.nx_true], x1[: plan.batch_size, : plan.nx_true]


def _solve_tiled_bx(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    plan: LargePopulationLayoutPlan,
) -> tuple[Array, Array]:
    space_shape = (plan.n_tiles, plan.block_b, plan.nx_pad)
    edge_shape = (plan.n_tiles, plan.block_b, max(plan.nx_pad - 1, 0))

    def space(values: Array) -> Array:
        return jnp.reshape(values, space_shape)

    def edges(values: Array) -> Array:
        return jnp.reshape(values, edge_shape)

    def solve_tile(
        a00_t: Array,
        a01_t: Array,
        a10_t: Array,
        a11_t: Array,
        off0_t: Array,
        off1_t: Array,
        rhs0_t: Array,
        rhs1_t: Array,
    ) -> tuple[Array, Array]:
        return solve_block_tridiagonal_2x2_pcr_soa_batched(
            a00_t,
            a01_t,
            a10_t,
            a11_t,
            off0_t,
            off1_t,
            rhs0_t,
            rhs1_t,
        )

    x0_t, x1_t = jax.vmap(solve_tile)(
        space(a00),
        space(a01),
        space(a10),
        space(a11),
        edges(off0),
        edges(off1),
        space(rhs0),
        space(rhs1),
    )
    return jnp.reshape(x0_t, (plan.batch_padded, plan.nx_pad)), jnp.reshape(
        x1_t,
        (plan.batch_padded, plan.nx_pad),
    )


def _solve_xb_packed(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
) -> tuple[Array, Array]:
    """Prototype XB route using the existing exact transposed PCR body."""

    return solve_block_tridiagonal_2x2_pcr_soa_batched_transposed(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
    )


def _as_batch_space(values: Array, *, plan: LargePopulationLayoutPlan) -> Array:
    arr = jnp.asarray(values)
    if arr.ndim == 1:
        if int(arr.shape[0]) != plan.nx_true:
            raise ValueError(f"space value must have length {plan.nx_true}, got {arr.shape}.")
        return jnp.broadcast_to(arr[None, :], (plan.batch_size, plan.nx_true))
    if arr.ndim == 2 and tuple(arr.shape) == (plan.batch_size, plan.nx_true):
        return arr
    raise ValueError(
        f"space value must have shape ({plan.nx_true},) or "
        f"({plan.batch_size}, {plan.nx_true}), got {arr.shape}."
    )


def _as_batch_edges(values: Array, *, plan: LargePopulationLayoutPlan) -> Array:
    arr = jnp.asarray(values)
    edge_count = max(plan.nx_true - 1, 0)
    if arr.ndim == 1:
        if int(arr.shape[0]) != edge_count:
            raise ValueError(f"edge value must have length {edge_count}, got {arr.shape}.")
        return jnp.broadcast_to(arr[None, :], (plan.batch_size, edge_count))
    if arr.ndim == 2 and tuple(arr.shape) == (plan.batch_size, edge_count):
        return arr
    raise ValueError(
        f"edge value must have shape ({edge_count},) or "
        f"({plan.batch_size}, {edge_count}), got {arr.shape}."
    )


def _pad_space(values: Array, *, plan: LargePopulationLayoutPlan, fill: float) -> Array:
    arr = jnp.asarray(values)
    if tuple(arr.shape) != (plan.batch_size, plan.nx_true):
        raise ValueError(
            f"space array must have shape ({plan.batch_size}, {plan.nx_true}), got {arr.shape}."
        )
    return _pad_2d(arr, target_shape=(plan.batch_padded, plan.nx_pad), fill=fill)


def _pad_edges(values: Array, *, plan: LargePopulationLayoutPlan, fill: float) -> Array:
    edge_true = max(plan.nx_true - 1, 0)
    edge_pad = max(plan.nx_pad - 1, 0)
    arr = jnp.asarray(values)
    if tuple(arr.shape) != (plan.batch_size, edge_true):
        raise ValueError(
            f"edge array must have shape ({plan.batch_size}, {edge_true}), got {arr.shape}."
        )
    return _pad_2d(arr, target_shape=(plan.batch_padded, edge_pad), fill=fill)


def _pad_2d(values: Array, *, target_shape: tuple[int, int], fill: float) -> Array:
    arr = jnp.asarray(values)
    rows, cols = arr.shape
    target_rows, target_cols = target_shape
    if rows > target_rows or cols > target_cols:
        raise ValueError(f"cannot pad shape {arr.shape} to smaller target {target_shape}.")
    pad_width = ((0, target_rows - rows), (0, target_cols - cols))
    return jnp.pad(arr, pad_width, constant_values=jnp.asarray(fill, dtype=arr.dtype))
