"""Experimental Triton block-Thomas backend for batched double-cable solves.

This module is deliberately not wired into public solver routing yet. The
Triton kernel is useful only when the inputs already live on a CUDA device and
the call can avoid host round-trips.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised on CUDA benchmark workers.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - local CPU/dev env path.
    triton = None
    tl = None


if triton is not None and tl is not None:

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

        u0 = tl.full((), 0.0, tl.float32)
        u1 = tl.full((), 0.0, tl.float32)
        if N > 1:
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
    _block_thomas_forward_kernel = None
    _block_thomas_backward_kernel = None


def triton_thomas_dependency_skip_reason() -> str | None:
    """Return why the experimental Triton backend cannot run, if anything."""

    try:
        import torch
    except ModuleNotFoundError:
        return "Python package 'torch' is not installed."
    if triton is None:
        return "Python package 'triton' is not installed."
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is false."
    return None


def solve_block_tridiagonal_2x2_triton_thomas_torch(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    """Solve batched double-cable systems from CUDA Torch tensors.

    Inputs are batch-first ``[B, Nx]`` arrays, with edge coefficients either
    ``[B, Nx - 1]`` or shared ``[Nx - 1]``. Only ``float32`` CUDA tensors are
    supported in this first experimental backend.
    """

    skip_reason = triton_thomas_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    import torch

    rhs0 = _as_torch_tensor(rhs0, name="rhs0")
    rhs1 = _as_torch_tensor(rhs1, name="rhs1")
    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if tuple(rhs0.shape) != tuple(rhs1.shape):
        raise ValueError(f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}.")
    if int(rhs0.shape[1]) < 2:
        raise ValueError("Nx must be >= 2 for the Triton block-Thomas backend.")
    batch_size, nx = int(rhs0.shape[0]), int(rhs0.shape[1])
    device = rhs0.device
    dtype = rhs0.dtype
    if dtype != torch.float32:
        raise TypeError(f"rhs0 must be torch.float32, got {dtype}.")
    if not rhs0.is_cuda:
        raise RuntimeError("Triton block-Thomas inputs must be CUDA tensors.")

    rhs0 = rhs0.contiguous()
    rhs1 = _space_tensor(rhs1, batch_size=batch_size, nx=nx, name="rhs1", device=device)
    a00 = _space_tensor(a00, batch_size=batch_size, nx=nx, name="a00", device=device)
    a01 = _space_tensor(a01, batch_size=batch_size, nx=nx, name="a01", device=device)
    a10 = _space_tensor(a10, batch_size=batch_size, nx=nx, name="a10", device=device)
    a11 = _space_tensor(a11, batch_size=batch_size, nx=nx, name="a11", device=device)
    off0 = _edge_tensor(off0, batch_size=batch_size, nx=nx, name="off0", device=device)
    off1 = _edge_tensor(off1, batch_size=batch_size, nx=nx, name="off1", device=device)

    c00 = torch.empty_like(rhs0)
    c01 = torch.empty_like(rhs0)
    c10 = torch.empty_like(rhs0)
    c11 = torch.empty_like(rhs0)
    d0 = torch.empty_like(rhs0)
    d1 = torch.empty_like(rhs0)
    out0 = torch.empty_like(rhs0)
    out1 = torch.empty_like(rhs1)
    grid = (batch_size,)
    _block_thomas_forward_kernel[grid](
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
        N=nx,
        num_warps=1,
    )
    _block_thomas_backward_kernel[grid](
        c00,
        c01,
        c10,
        c11,
        d0,
        d1,
        out0,
        out1,
        N=nx,
        num_warps=1,
    )
    return out0, out1


def solve_block_tridiagonal_2x2_triton_thomas_jax(
    a00: Any,
    a01: Any,
    a10: Any,
    a11: Any,
    off0: Any,
    off1: Any,
    rhs0: Any,
    rhs1: Any,
) -> tuple[Any, Any]:
    """Solve from eager JAX arrays via DLPack.

    This bridge is experimental and cannot be called from inside ``jax.jit``.
    It is intended to measure integration overhead before adding any production
    routing.
    """

    skip_reason = triton_thomas_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    import jax
    import jax.numpy as jnp
    import torch

    arrays = [jnp.asarray(x) for x in (a00, a01, a10, a11, off0, off1, rhs0, rhs1)]
    torch_arrays = [torch.from_dlpack(x) for x in arrays]
    out0, out1 = solve_block_tridiagonal_2x2_triton_thomas_torch(*torch_arrays)
    return jax.dlpack.from_dlpack(out0), jax.dlpack.from_dlpack(out1)


def _as_torch_tensor(values: Any, *, name: str):
    import torch

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if values.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {values.dtype}.")
    if not values.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor.")
    return values


def _space_tensor(values: Any, *, batch_size: int, nx: int, name: str, device: Any):
    tensor = _as_torch_tensor(values, name=name)
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on the same CUDA device as rhs0.")
    if tensor.ndim == 1 and int(tensor.shape[0]) == nx:
        return tensor[None, :].expand(batch_size, nx).contiguous()
    if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, nx):
        return tensor.contiguous()
    raise ValueError(f"{name} must have shape ({nx},) or ({batch_size}, {nx}), got {tuple(tensor.shape)}.")


def _edge_tensor(values: Any, *, batch_size: int, nx: int, name: str, device: Any):
    tensor = _as_torch_tensor(values, name=name)
    edge_n = nx - 1
    if tensor.device != device:
        raise RuntimeError(f"{name} must be on the same CUDA device as rhs0.")
    if tensor.ndim == 1 and int(tensor.shape[0]) == edge_n:
        return tensor[None, :].expand(batch_size, edge_n).contiguous()
    if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, edge_n):
        return tensor.contiguous()
    raise ValueError(
        f"{name} must have shape ({edge_n},) or ({batch_size}, {edge_n}), got {tuple(tensor.shape)}."
    )
