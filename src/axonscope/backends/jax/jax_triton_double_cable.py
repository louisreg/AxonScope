"""Optional jax-triton exact double-cable block-Thomas solver.

This module is backend-private and experimental. It is intentionally not wired
into public solver routing or ``auto``. Benchmarks can opt into it to test the
JAX-to-Triton call path on GPU hardware.
"""

from __future__ import annotations

import importlib.util
from typing import Any

try:  # pragma: no cover - exercised on CUDA benchmark workers.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - common local CPU/dev path.
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _tiled_block_thomas_forward_kernel(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        N: tl.constexpr,
        B: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        tile = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_B)
        batch = tile * BLOCK_B + lanes
        mask = batch < B

        offset = batch
        m00 = tl.load(a00 + offset, mask=mask, other=1.0)
        m01 = tl.load(a01 + offset, mask=mask, other=0.0)
        m10 = tl.load(a10 + offset, mask=mask, other=0.0)
        m11 = tl.load(a11 + offset, mask=mask, other=1.0)
        r0 = tl.load(rhs0 + offset, mask=mask, other=0.0)
        r1 = tl.load(rhs1 + offset, mask=mask, other=0.0)
        det = m00 * m11 - m01 * m10
        inv00 = m11 / det
        inv01 = -m01 / det
        inv10 = -m10 / det
        inv11 = m00 / det

        u0 = tl.load(off0 + batch, mask=mask, other=0.0)
        u1 = tl.load(off1 + batch, mask=mask, other=0.0)
        cp00 = inv00 * u0
        cp01 = inv01 * u1
        cp10 = inv10 * u0
        cp11 = inv11 * u1
        dp0 = inv00 * r0 + inv01 * r1
        dp1 = inv10 * r0 + inv11 * r1
        tl.store(c00 + offset, cp00, mask=mask)
        tl.store(c01 + offset, cp01, mask=mask)
        tl.store(c10 + offset, cp10, mask=mask)
        tl.store(c11 + offset, cp11, mask=mask)
        tl.store(d0 + offset, dp0, mask=mask)
        tl.store(d1 + offset, dp1, mask=mask)

        prev_c00 = cp00
        prev_c01 = cp01
        prev_c10 = cp10
        prev_c11 = cp11
        prev_d0 = dp0
        prev_d1 = dp1

        for x in tl.static_range(1, N):
            offset = x * B + batch
            edge_prev = (x - 1) * B + batch
            l0 = tl.load(off0 + edge_prev, mask=mask, other=0.0)
            l1 = tl.load(off1 + edge_prev, mask=mask, other=0.0)
            m00 = tl.load(a00 + offset, mask=mask, other=1.0) - l0 * prev_c00
            m01 = tl.load(a01 + offset, mask=mask, other=0.0) - l0 * prev_c01
            m10 = tl.load(a10 + offset, mask=mask, other=0.0) - l1 * prev_c10
            m11 = tl.load(a11 + offset, mask=mask, other=1.0) - l1 * prev_c11
            r0 = tl.load(rhs0 + offset, mask=mask, other=0.0) - l0 * prev_d0
            r1 = tl.load(rhs1 + offset, mask=mask, other=0.0) - l1 * prev_d1

            det = m00 * m11 - m01 * m10
            inv00 = m11 / det
            inv01 = -m01 / det
            inv10 = -m10 / det
            inv11 = m00 / det

            u0 = tl.full((BLOCK_B,), 0.0, tl.float32)
            u1 = tl.full((BLOCK_B,), 0.0, tl.float32)
            if x < N - 1:
                edge_next = x * B + batch
                u0 = tl.load(off0 + edge_next, mask=mask, other=0.0)
                u1 = tl.load(off1 + edge_next, mask=mask, other=0.0)
            cp00 = inv00 * u0
            cp01 = inv01 * u1
            cp10 = inv10 * u0
            cp11 = inv11 * u1
            dp0 = inv00 * r0 + inv01 * r1
            dp1 = inv10 * r0 + inv11 * r1

            tl.store(c00 + offset, cp00, mask=mask)
            tl.store(c01 + offset, cp01, mask=mask)
            tl.store(c10 + offset, cp10, mask=mask)
            tl.store(c11 + offset, cp11, mask=mask)
            tl.store(d0 + offset, dp0, mask=mask)
            tl.store(d1 + offset, dp1, mask=mask)
            prev_c00 = cp00
            prev_c01 = cp01
            prev_c10 = cp10
            prev_c11 = cp11
            prev_d0 = dp0
            prev_d1 = dp1

    @triton.jit
    def _tiled_block_thomas_backward_kernel(
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        out0,
        out1,
        N: tl.constexpr,
        B: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        tile = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_B)
        batch = tile * BLOCK_B + lanes
        mask = batch < B
        last = (N - 1) * B + batch

        x0 = tl.load(d0 + last, mask=mask, other=0.0)
        x1 = tl.load(d1 + last, mask=mask, other=0.0)
        tl.store(out0 + last, x0, mask=mask)
        tl.store(out1 + last, x1, mask=mask)

        for rev in tl.static_range(0, N - 1):
            x = N - 2 - rev
            offset = x * B + batch
            cp00 = tl.load(c00 + offset, mask=mask, other=0.0)
            cp01 = tl.load(c01 + offset, mask=mask, other=0.0)
            cp10 = tl.load(c10 + offset, mask=mask, other=0.0)
            cp11 = tl.load(c11 + offset, mask=mask, other=0.0)
            next_x0 = x0
            next_x1 = x1
            x0 = tl.load(d0 + offset, mask=mask, other=0.0) - cp00 * next_x0 - cp01 * next_x1
            x1 = tl.load(d1 + offset, mask=mask, other=0.0) - cp10 * next_x0 - cp11 * next_x1
            tl.store(out0 + offset, x0, mask=mask)
            tl.store(out1 + offset, x1, mask=mask)

    @triton.jit
    def _block_thomas_forward_kernel(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        N: tl.constexpr,
    ):
        batch = tl.program_id(0)
        base = batch * N
        edge_base = batch * (N - 1)

        m00 = tl.load(a00 + base)
        m01 = tl.load(a01 + base)
        m10 = tl.load(a10 + base)
        m11 = tl.load(a11 + base)
        r0 = tl.load(rhs0 + base)
        r1 = tl.load(rhs1 + base)
        det = m00 * m11 - m01 * m10
        inv00 = m11 / det
        inv01 = -m01 / det
        inv10 = -m10 / det
        inv11 = m00 / det

        u0 = tl.load(off0 + edge_base)
        u1 = tl.load(off1 + edge_base)
        cp00 = inv00 * u0
        cp01 = inv01 * u1
        cp10 = inv10 * u0
        cp11 = inv11 * u1
        dp0 = inv00 * r0 + inv01 * r1
        dp1 = inv10 * r0 + inv11 * r1
        tl.store(c00 + base, cp00)
        tl.store(c01 + base, cp01)
        tl.store(c10 + base, cp10)
        tl.store(c11 + base, cp11)
        tl.store(d0 + base, dp0)
        tl.store(d1 + base, dp1)

        prev_c00 = cp00
        prev_c01 = cp01
        prev_c10 = cp10
        prev_c11 = cp11
        prev_d0 = dp0
        prev_d1 = dp1

        for i in tl.static_range(1, N):
            offset = base + i
            l0 = tl.load(off0 + edge_base + i - 1)
            l1 = tl.load(off1 + edge_base + i - 1)
            m00 = tl.load(a00 + offset) - l0 * prev_c00
            m01 = tl.load(a01 + offset) - l0 * prev_c01
            m10 = tl.load(a10 + offset) - l1 * prev_c10
            m11 = tl.load(a11 + offset) - l1 * prev_c11
            r0 = tl.load(rhs0 + offset) - l0 * prev_d0
            r1 = tl.load(rhs1 + offset) - l1 * prev_d1

            det = m00 * m11 - m01 * m10
            inv00 = m11 / det
            inv01 = -m01 / det
            inv10 = -m10 / det
            inv11 = m00 / det

            u0 = tl.full((), 0.0, tl.float32)
            u1 = tl.full((), 0.0, tl.float32)
            if i < N - 1:
                u0 = tl.load(off0 + edge_base + i)
                u1 = tl.load(off1 + edge_base + i)
            cp00 = inv00 * u0
            cp01 = inv01 * u1
            cp10 = inv10 * u0
            cp11 = inv11 * u1
            dp0 = inv00 * r0 + inv01 * r1
            dp1 = inv10 * r0 + inv11 * r1

            tl.store(c00 + offset, cp00)
            tl.store(c01 + offset, cp01)
            tl.store(c10 + offset, cp10)
            tl.store(c11 + offset, cp11)
            tl.store(d0 + offset, dp0)
            tl.store(d1 + offset, dp1)
            prev_c00 = cp00
            prev_c01 = cp01
            prev_c10 = cp10
            prev_c11 = cp11
            prev_d0 = dp0
            prev_d1 = dp1

    @triton.jit
    def _block_thomas_backward_kernel(
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        out0,
        out1,
        N: tl.constexpr,
    ):
        batch = tl.program_id(0)
        base = batch * N
        last = base + N - 1

        x0 = tl.load(d0 + last)
        x1 = tl.load(d1 + last)
        tl.store(out0 + last, x0)
        tl.store(out1 + last, x1)

        for rev in tl.static_range(0, N - 1):
            i = N - 2 - rev
            offset = base + i
            cp00 = tl.load(c00 + offset)
            cp01 = tl.load(c01 + offset)
            cp10 = tl.load(c10 + offset)
            cp11 = tl.load(c11 + offset)
            next_x0 = x0
            next_x1 = x1
            x0 = tl.load(d0 + offset) - cp00 * next_x0 - cp01 * next_x1
            x1 = tl.load(d1 + offset) - cp10 * next_x0 - cp11 * next_x1
            tl.store(out0 + offset, x0)
            tl.store(out1 + offset, x1)

else:
    _tiled_block_thomas_forward_kernel = None
    _tiled_block_thomas_backward_kernel = None
    _block_thomas_forward_kernel = None
    _block_thomas_backward_kernel = None


def jax_triton_thomas_dependency_skip_reason() -> str | None:
    """Return why the optional jax-triton Thomas backend cannot run."""

    if importlib.util.find_spec("triton") is None:
        return "Python package 'triton' is not installed."
    if importlib.util.find_spec("jax_triton") is None:
        return "Python package 'jax-triton' is not installed."
    try:
        import jax
        import jax_triton  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        return f"Could not import jax_triton: {exc}"
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        return f"JAX GPU backend is unavailable: {exc}"
    if not devices:
        return "jax.devices('gpu') returned no devices."
    return None


def solve_block_tridiagonal_2x2_jax_triton_thomas_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    """Solve batch-first 2x2 block-tridiagonal systems via jax-triton.

    The current prototype is a minimal custom-kernel bridge:

    - one Triton program solves one axon;
    - inputs are batch-first ``[B, Nx]`` arrays, with shared coefficient inputs
      broadcast before the kernel call;
    - only ``float32`` is supported;
    - the implementation is not yet the tiled/lane-coalesced PTA design from
      the literature notes.
    """

    skip_reason = jax_triton_thomas_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    import jax
    import jax.numpy as jnp
    import jax_triton as jt

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    _check_rhs_pair(rhs0, rhs1)
    if rhs0.dtype != jnp.float32:
        raise TypeError(f"jax_triton_thomas supports only float32, got {rhs0.dtype}.")
    batch_size, nx = int(rhs0.shape[0]), int(rhs0.shape[1])

    a00 = _space_tensor(a00, batch_size=batch_size, nx=nx, name="a00")
    a01 = _space_tensor(a01, batch_size=batch_size, nx=nx, name="a01")
    a10 = _space_tensor(a10, batch_size=batch_size, nx=nx, name="a10")
    a11 = _space_tensor(a11, batch_size=batch_size, nx=nx, name="a11")
    off0 = _edge_tensor(off0, batch_size=batch_size, nx=nx, name="off0")
    off1 = _edge_tensor(off1, batch_size=batch_size, nx=nx, name="off1")

    work_shape = jax.ShapeDtypeStruct(rhs0.shape, rhs0.dtype)
    c00, c01, c10, c11, d0, d1 = jt.triton_call(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        kernel=_block_thomas_forward_kernel,
        out_shape=(work_shape, work_shape, work_shape, work_shape, work_shape, work_shape),
        grid=(batch_size,),
        N=nx,
        num_warps=1,
        num_stages=1,
    )
    return jt.triton_call(
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        kernel=_block_thomas_backward_kernel,
        out_shape=(work_shape, work_shape),
        grid=(batch_size,),
        N=nx,
        num_warps=1,
        num_stages=1,
    )


def solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_xb(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int = 128,
) -> tuple[Any, Any]:
    """Solve ``[Nx, B]`` systems with a tile/lane Triton Thomas kernel.

    This is the first paper-inspired jax-triton candidate:

    - data layout is ``[x, batch]`` so lanes in a Triton program load adjacent
      axons at the same compartment index;
    - one program owns ``BLOCK_B`` axons;
    - the Thomas recurrence runs over ``x`` while lane vectors carry the axon
      batch.

    It still uses global-memory scratch for the forward coefficients and does
    not yet implement a PfSolve-equivalent shared-memory/scratch schedule.
    """

    skip_reason = jax_triton_thomas_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if int(block_b) < 1:
        raise ValueError("block_b must be >= 1.")

    import jax
    import jax.numpy as jnp
    import jax_triton as jt

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    _check_rhs_pair_xb(rhs0, rhs1)
    if rhs0.dtype != jnp.float32:
        raise TypeError(f"jax_triton_tiled_thomas supports only float32, got {rhs0.dtype}.")
    nx, batch_size = int(rhs0.shape[0]), int(rhs0.shape[1])

    a00 = _space_tensor_xb(a00, batch_size=batch_size, nx=nx, name="a00")
    a01 = _space_tensor_xb(a01, batch_size=batch_size, nx=nx, name="a01")
    a10 = _space_tensor_xb(a10, batch_size=batch_size, nx=nx, name="a10")
    a11 = _space_tensor_xb(a11, batch_size=batch_size, nx=nx, name="a11")
    off0 = _edge_tensor_xb(off0, batch_size=batch_size, nx=nx, name="off0")
    off1 = _edge_tensor_xb(off1, batch_size=batch_size, nx=nx, name="off1")

    work_shape = jax.ShapeDtypeStruct(rhs0.shape, rhs0.dtype)
    grid = ((batch_size + int(block_b) - 1) // int(block_b),)
    c00, c01, c10, c11, d0, d1 = jt.triton_call(
        a00,
        a01,
        a10,
        a11,
        off0,
        off1,
        rhs0,
        rhs1,
        kernel=_tiled_block_thomas_forward_kernel,
        out_shape=(work_shape, work_shape, work_shape, work_shape, work_shape, work_shape),
        grid=grid,
        N=nx,
        B=batch_size,
        BLOCK_B=int(block_b),
        num_warps=_num_warps_for_block_b(int(block_b)),
        num_stages=1,
    )
    return jt.triton_call(
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        kernel=_tiled_block_thomas_backward_kernel,
        out_shape=(work_shape, work_shape),
        grid=grid,
        N=nx,
        B=batch_size,
        BLOCK_B=int(block_b),
        num_warps=_num_warps_for_block_b(int(block_b)),
        num_stages=1,
    )


def solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_batched(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
    *,
    block_b: int = 128,
) -> tuple[Any, Any]:
    """Batch-first wrapper for the tile/lane jax-triton Thomas kernel."""

    import jax.numpy as jnp

    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)
    _check_rhs_pair(rhs0, rhs1)
    if rhs0.dtype != jnp.float32:
        raise TypeError(f"jax_triton_tiled_thomas supports only float32, got {rhs0.dtype}.")
    batch_size, nx = int(rhs0.shape[0]), int(rhs0.shape[1])

    a00_xb = jnp.swapaxes(_space_tensor(a00, batch_size=batch_size, nx=nx, name="a00"), 0, 1)
    a01_xb = jnp.swapaxes(_space_tensor(a01, batch_size=batch_size, nx=nx, name="a01"), 0, 1)
    a10_xb = jnp.swapaxes(_space_tensor(a10, batch_size=batch_size, nx=nx, name="a10"), 0, 1)
    a11_xb = jnp.swapaxes(_space_tensor(a11, batch_size=batch_size, nx=nx, name="a11"), 0, 1)
    off0_xb = jnp.swapaxes(_edge_tensor(off0, batch_size=batch_size, nx=nx, name="off0"), 0, 1)
    off1_xb = jnp.swapaxes(_edge_tensor(off1, batch_size=batch_size, nx=nx, name="off1"), 0, 1)
    out0_xb, out1_xb = solve_block_tridiagonal_2x2_jax_triton_tiled_thomas_xb(
        a00_xb,
        a01_xb,
        a10_xb,
        a11_xb,
        off0_xb,
        off1_xb,
        jnp.swapaxes(rhs0, 0, 1),
        jnp.swapaxes(rhs1, 0, 1),
        block_b=block_b,
    )
    return jnp.swapaxes(out0_xb, 0, 1), jnp.swapaxes(out1_xb, 0, 1)


def _check_rhs_pair(rhs0: Any, rhs1: Any) -> None:
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if tuple(rhs0.shape) != tuple(rhs1.shape):
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )
    if int(rhs0.shape[1]) < 2:
        raise ValueError("Nx must be >= 2 for the jax-triton Thomas backend.")


def _check_rhs_pair_xb(rhs0: Any, rhs1: Any) -> None:
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (Nx, batch_size).")
    if tuple(rhs0.shape) != tuple(rhs1.shape):
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )
    if int(rhs0.shape[0]) < 2:
        raise ValueError("Nx must be >= 2 for the jax-triton Thomas backend.")


def _space_tensor(values: Any, *, batch_size: int, nx: int, name: str) -> Any:
    import jax.numpy as jnp

    tensor = jnp.asarray(values, dtype=jnp.float32)
    if tensor.ndim == 1 and int(tensor.shape[0]) == nx:
        return jnp.broadcast_to(tensor[None, :], (batch_size, nx))
    if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, nx):
        return tensor
    raise ValueError(
        f"{name} must have shape ({nx},) or ({batch_size}, {nx}), got {tuple(tensor.shape)}."
    )


def _edge_tensor(values: Any, *, batch_size: int, nx: int, name: str) -> Any:
    import jax.numpy as jnp

    edge_n = nx - 1
    tensor = jnp.asarray(values, dtype=jnp.float32)
    if tensor.ndim == 1 and int(tensor.shape[0]) == edge_n:
        return jnp.broadcast_to(tensor[None, :], (batch_size, edge_n))
    if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, edge_n):
        return tensor
    raise ValueError(
        f"{name} must have shape ({edge_n},) or ({batch_size}, {edge_n}), got {tuple(tensor.shape)}."
    )


def _space_tensor_xb(values: Any, *, batch_size: int, nx: int, name: str) -> Any:
    import jax.numpy as jnp

    tensor = jnp.asarray(values, dtype=jnp.float32)
    if tensor.ndim == 1 and int(tensor.shape[0]) == nx:
        return jnp.broadcast_to(tensor[:, None], (nx, batch_size))
    if tensor.ndim == 2 and tuple(tensor.shape) == (nx, batch_size):
        return tensor
    raise ValueError(
        f"{name} must have shape ({nx},) or ({nx}, {batch_size}), got {tuple(tensor.shape)}."
    )


def _edge_tensor_xb(values: Any, *, batch_size: int, nx: int, name: str) -> Any:
    import jax.numpy as jnp

    edge_n = nx - 1
    tensor = jnp.asarray(values, dtype=jnp.float32)
    if tensor.ndim == 1 and int(tensor.shape[0]) == edge_n:
        return jnp.broadcast_to(tensor[:, None], (edge_n, batch_size))
    if tensor.ndim == 2 and tuple(tensor.shape) == (edge_n, batch_size):
        return tensor
    raise ValueError(
        f"{name} must have shape ({edge_n},) or ({edge_n}, {batch_size}), got {tuple(tensor.shape)}."
    )


def _num_warps_for_block_b(block_b: int) -> int:
    if block_b >= 128:
        return 4
    if block_b >= 64:
        return 2
    return 1
