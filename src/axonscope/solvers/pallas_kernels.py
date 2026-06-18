from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from axonscope.solvers.common import Array

try:
    _pallas_load = pl.load
    _pallas_store = pl.store
except AttributeError:
    from jax._src.pallas import primitives as _pallas_primitives

    _pallas_load = _pallas_primitives.load
    _pallas_store = _pallas_primitives.store


def solve_block_tridiagonal_2x2_pallas_thomas_batched(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    block_b: int = 128,
    interpret: bool | None = None,
    num_warps: int = 4,
) -> tuple[Array, Array]:
    """Benchmark-only Pallas block-Thomas solve for batch-first 2x2 systems.

    One Pallas program handles ``block_b`` fibers and the full ``Nx`` cable.
    This is a Phase 3 spike baseline, not a public solver backend.
    """

    a00 = jnp.asarray(a00)
    a01 = jnp.asarray(a01)
    a10 = jnp.asarray(a10)
    a11 = jnp.asarray(a11)
    off0 = jnp.asarray(off0)
    off1 = jnp.asarray(off1)
    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)

    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )
    batch_size, n = rhs0.shape
    if n < 2:
        raise ValueError("pallas_thomas requires Nx >= 2.")
    if batch_size % int(block_b) != 0:
        raise ValueError(
            f"batch_size must be divisible by block_b={int(block_b)}, got {batch_size}."
        )

    def as_batched(values: Array, *, length: int, name: str) -> Array:
        arr = jnp.asarray(values)
        if arr.shape[-1] != length:
            raise ValueError(f"{name} must have trailing length {length}, got {arr.shape}.")
        if arr.ndim == 1:
            return jnp.broadcast_to(arr[None, :], (batch_size, length))
        if arr.ndim == 2 and arr.shape[0] == batch_size:
            return arr
        raise ValueError(
            f"{name} must have shape ({length},) or ({batch_size}, {length}), got {arr.shape}."
        )

    a00_b = as_batched(a00, length=n, name="a00")
    a01_b = as_batched(a01, length=n, name="a01")
    a10_b = as_batched(a10, length=n, name="a10")
    a11_b = as_batched(a11, length=n, name="a11")
    off0_b = as_batched(off0, length=n - 1, name="off0")
    off1_b = as_batched(off1, length=n - 1, name="off1")

    if interpret is None:
        interpret = jax.default_backend() != "gpu"

    del num_warps
    block_b = int(block_b)
    n_storage = _round_up_to_multiple(n, 8)
    n_edge_storage = _round_up_to_multiple(n - 1, 8)
    a00_b = _pad_trailing_axis(a00_b, n_storage)
    a01_b = _pad_trailing_axis(a01_b, n_storage)
    a10_b = _pad_trailing_axis(a10_b, n_storage)
    a11_b = _pad_trailing_axis(a11_b, n_storage)
    off0_b = _pad_trailing_axis(off0_b, n_edge_storage)
    off1_b = _pad_trailing_axis(off1_b, n_edge_storage)
    rhs0_b = _pad_trailing_axis(rhs0, n_storage)
    rhs1_b = _pad_trailing_axis(rhs1, n_storage)
    in_specs = (
        _block_spec_2d(block_b, n_storage),
        _block_spec_2d(block_b, n_storage),
        _block_spec_2d(block_b, n_storage),
        _block_spec_2d(block_b, n_storage),
        _block_spec_2d(block_b, n_edge_storage),
        _block_spec_2d(block_b, n_edge_storage),
        _block_spec_2d(block_b, n_storage),
        _block_spec_2d(block_b, n_storage),
    )
    out_spec = pl.BlockSpec((block_b, n_storage, 2), lambda block_id: (block_id, 0, 0))
    scratch = _memory_ref(
        (block_b, n_storage, 6),
        rhs0.dtype,
        gpu_smem=not bool(interpret),
    )
    solve = pl.pallas_call(
        functools.partial(_pallas_thomas_2x2_kernel, n=n),
        out_shape=jax.ShapeDtypeStruct((batch_size, n_storage, 2), rhs0.dtype),
        grid=(batch_size // block_b,),
        in_specs=in_specs,
        out_specs=out_spec,
        scratch_shapes=(scratch,),
        interpret=bool(interpret),
        name=f"double_cable_pallas_thomas_b{block_b}",
    )
    out = solve(a00_b, a01_b, a10_b, a11_b, off0_b, off1_b, rhs0_b, rhs1_b)
    return out[:, :n, 0], out[:, :n, 1]


def solve_block_tridiagonal_2x2_pallas_pcr_batched(
    a00: Array,
    a01: Array,
    a10: Array,
    a11: Array,
    off0: Array,
    off1: Array,
    rhs0: Array,
    rhs1: Array,
    *,
    block_b: int = 128,
    interpret: bool | None = None,
) -> tuple[Array, Array]:
    """Benchmark-only Pallas PCR solve for batch-first 2x2 systems.

    This Phase 3B spike keeps each Pallas program at ``block_b`` fibers and one
    cable column. That shape is deliberate: Mosaic GPU strided loads want 128
    elements, while full-cable Thomas blocks exceeded P100 SMEM.
    """

    a00 = jnp.asarray(a00)
    a01 = jnp.asarray(a01)
    a10 = jnp.asarray(a10)
    a11 = jnp.asarray(a11)
    off0 = jnp.asarray(off0)
    off1 = jnp.asarray(off1)
    rhs0 = jnp.asarray(rhs0)
    rhs1 = jnp.asarray(rhs1)

    if rhs0.ndim != 2 or rhs1.ndim != 2:
        raise ValueError("rhs0 and rhs1 must have shape (batch_size, Nx).")
    if rhs0.shape != rhs1.shape:
        raise ValueError(
            f"rhs0 and rhs1 must have the same shape, got {rhs0.shape} and {rhs1.shape}."
        )
    batch_size, n = rhs0.shape
    if n < 2:
        raise ValueError("pallas_pcr requires Nx >= 2.")
    if batch_size % int(block_b) != 0:
        raise ValueError(
            f"batch_size must be divisible by block_b={int(block_b)}, got {batch_size}."
        )

    def as_batched(values: Array, *, length: int, name: str) -> Array:
        arr = jnp.asarray(values)
        if arr.shape[-1] != length:
            raise ValueError(f"{name} must have trailing length {length}, got {arr.shape}.")
        if arr.ndim == 1:
            return jnp.broadcast_to(arr[None, :], (batch_size, length))
        if arr.ndim == 2 and arr.shape[0] == batch_size:
            return arr
        raise ValueError(
            f"{name} must have shape ({length},) or ({batch_size}, {length}), got {arr.shape}."
        )

    diag00 = as_batched(a00, length=n, name="a00")
    diag01 = as_batched(a01, length=n, name="a01")
    diag10 = as_batched(a10, length=n, name="a10")
    diag11 = as_batched(a11, length=n, name="a11")
    off0_b = as_batched(off0, length=n - 1, name="off0")
    off1_b = as_batched(off1, length=n - 1, name="off1")

    zero_col = jnp.zeros((batch_size, 1), dtype=rhs0.dtype)
    zeros = jnp.zeros((batch_size, n), dtype=rhs0.dtype)
    lower00 = jnp.concatenate([zero_col, off0_b], axis=1)
    lower01 = zeros
    lower10 = zeros
    lower11 = jnp.concatenate([zero_col, off1_b], axis=1)
    upper00 = jnp.concatenate([off0_b, zero_col], axis=1)
    upper01 = zeros
    upper10 = zeros
    upper11 = jnp.concatenate([off1_b, zero_col], axis=1)
    r0 = rhs0
    r1 = rhs1

    if interpret is None:
        interpret = jax.default_backend() != "gpu"

    block_b = int(block_b)
    stage = functools.partial(
        _pallas_pcr_2x2_stage,
        batch_size=batch_size,
        n=n,
        block_b=block_b,
        interpret=bool(interpret),
    )
    stride = 1
    while stride < n:
        (
            lower00,
            lower01,
            lower10,
            lower11,
            upper00,
            upper01,
            upper10,
            upper11,
            diag00,
            diag01,
            diag10,
            diag11,
            r0,
            r1,
        ) = stage(
            lower00,
            lower01,
            lower10,
            lower11,
            upper00,
            upper01,
            upper10,
            upper11,
            diag00,
            diag01,
            diag10,
            diag11,
            r0,
            r1,
            stride=stride,
        )
        stride *= 2

    inv00, inv01, inv10, inv11 = _inv2_components(diag00, diag01, diag10, diag11)
    return _matvec2_components(inv00, inv01, inv10, inv11, r0, r1)


def _block_spec_2d(block_b: int, n: int) -> pl.BlockSpec:
    return pl.BlockSpec((block_b, n), lambda block_id: (block_id, 0))


def _block_spec_column(block_b: int, *, memory_space=None) -> pl.BlockSpec:
    return pl.BlockSpec(
        (block_b, 1),
        lambda batch_block, column: (batch_block, column),
        memory_space=memory_space,
    )


def _whole_array_spec(*, memory_space=None) -> pl.BlockSpec:
    return pl.BlockSpec(memory_space=memory_space)


def _gpu_memory_space(name: str, *, enabled: bool):
    if not enabled:
        return None
    try:
        from jax.experimental.pallas import mosaic_gpu as plgpu

        return getattr(plgpu, name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


def _inv2_components(
    m00: Array,
    m01: Array,
    m10: Array,
    m11: Array,
) -> tuple[Array, Array, Array, Array]:
    det = m00 * m11 - m01 * m10
    return m11 / det, -m01 / det, -m10 / det, m00 / det


def _matmul2_components(
    l00: Array,
    l01: Array,
    l10: Array,
    l11: Array,
    r00: Array,
    r01: Array,
    r10: Array,
    r11: Array,
) -> tuple[Array, Array, Array, Array]:
    return (
        l00 * r00 + l01 * r10,
        l00 * r01 + l01 * r11,
        l10 * r00 + l11 * r10,
        l10 * r01 + l11 * r11,
    )


def _matvec2_components(
    m00: Array,
    m01: Array,
    m10: Array,
    m11: Array,
    v0: Array,
    v1: Array,
) -> tuple[Array, Array]:
    return m00 * v0 + m01 * v1, m10 * v0 + m11 * v1


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _pad_trailing_axis(values: Array, target_length: int) -> Array:
    pad = int(target_length) - int(values.shape[-1])
    if pad <= 0:
        return values
    return jnp.pad(values, ((0, 0), (0, pad)))


def _memory_ref(shape: tuple[int, ...], dtype: jnp.dtype, *, gpu_smem: bool = False):
    if gpu_smem:
        try:
            from jax.experimental.pallas import mosaic_gpu as plgpu

            return plgpu.SMEM(shape, dtype)
        except (ImportError, AttributeError, TypeError):
            pass

    memory_ref = getattr(pl, "MemoryRef", None)
    memory_space = getattr(pl, "MemorySpace", None)
    if memory_ref is None or memory_space is None:
        from jax._src.pallas import core as pallas_core

        memory_ref = pallas_core.MemoryRef
        memory_space = pallas_core.MemorySpace
    spaces = tuple(
        space
        for name in ("DEFAULT", "ANY")
        if (space := getattr(memory_space, name, None)) is not None
    )
    for space in spaces:
        try:
            return space(shape, dtype)
        except TypeError:
            pass
        try:
            return memory_ref(jax.core.ShapedArray(shape, dtype), space)
        except (AttributeError, TypeError):
            pass
        try:
            return memory_ref(shape, dtype, space)
        except TypeError:
            pass
    return memory_ref(shape, dtype)


def _pallas_pcr_2x2_stage(
    lower00: Array,
    lower01: Array,
    lower10: Array,
    lower11: Array,
    upper00: Array,
    upper01: Array,
    upper10: Array,
    upper11: Array,
    diag00: Array,
    diag01: Array,
    diag10: Array,
    diag11: Array,
    r0: Array,
    r1: Array,
    *,
    stride: int,
    batch_size: int,
    n: int,
    block_b: int,
    interpret: bool,
) -> tuple[Array, ...]:
    gmem = _gpu_memory_space("GMEM", enabled=not bool(interpret))
    in_specs = (_whole_array_spec(memory_space=gmem),) * 14
    out_spec = _block_spec_column(block_b)
    out_specs = (out_spec,) * 14
    out_shape = tuple(
        jax.ShapeDtypeStruct((batch_size, n), lower00.dtype) for _ in range(14)
    )
    stage = pl.pallas_call(
        functools.partial(_pallas_pcr_2x2_stage_kernel, stride=int(stride), n=int(n)),
        out_shape=out_shape,
        grid=(batch_size // block_b, n),
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=bool(interpret),
        name=f"double_cable_pallas_pcr_s{int(stride)}",
    )
    return stage(
        lower00,
        lower01,
        lower10,
        lower11,
        upper00,
        upper01,
        upper10,
        upper11,
        diag00,
        diag01,
        diag10,
        diag11,
        r0,
        r1,
    )


def _pallas_pcr_2x2_stage_kernel(
    lower00_ref,
    lower01_ref,
    lower10_ref,
    lower11_ref,
    upper00_ref,
    upper01_ref,
    upper10_ref,
    upper11_ref,
    diag00_ref,
    diag01_ref,
    diag10_ref,
    diag11_ref,
    r0_ref,
    r1_ref,
    out_lower00_ref,
    out_lower01_ref,
    out_lower10_ref,
    out_lower11_ref,
    out_upper00_ref,
    out_upper01_ref,
    out_upper10_ref,
    out_upper11_ref,
    out_diag00_ref,
    out_diag01_ref,
    out_diag10_ref,
    out_diag11_ref,
    out_r0_ref,
    out_r1_ref,
    *,
    stride: int,
    n: int,
) -> None:
    batch_block = pl.program_id(0)
    col = pl.program_id(1)
    rows = pl.ds(batch_block * out_lower00_ref.shape[0], out_lower00_ref.shape[0])
    current = (rows, pl.ds(col, 1))
    left_col = jnp.maximum(col - int(stride), 0)
    right_col = jnp.minimum(col + int(stride), int(n) - 1)
    left = (rows, pl.ds(left_col, 1))
    right = (rows, pl.ds(right_col, 1))

    def load(ref, index):
        return ref[index][:, 0]

    l00 = load(lower00_ref, current)
    l01 = load(lower01_ref, current)
    l10 = load(lower10_ref, current)
    l11 = load(lower11_ref, current)
    u00 = load(upper00_ref, current)
    u01 = load(upper01_ref, current)
    u10 = load(upper10_ref, current)
    u11 = load(upper11_ref, current)

    left_inv = _inv2_components(
        load(diag00_ref, left),
        load(diag01_ref, left),
        load(diag10_ref, left),
        load(diag11_ref, left),
    )
    right_inv = _inv2_components(
        load(diag00_ref, right),
        load(diag01_ref, right),
        load(diag10_ref, right),
        load(diag11_ref, right),
    )
    lf00, lf01, lf10, lf11 = _matmul2_components(l00, l01, l10, l11, *left_inv)
    rf00, rf01, rf10, rf11 = _matmul2_components(u00, u01, u10, u11, *right_inv)

    zero = jnp.zeros_like(lf00)
    has_left = col >= int(stride)
    has_right = col + int(stride) < int(n)
    lf00 = jnp.where(has_left, lf00, zero)
    lf01 = jnp.where(has_left, lf01, zero)
    lf10 = jnp.where(has_left, lf10, zero)
    lf11 = jnp.where(has_left, lf11, zero)
    rf00 = jnp.where(has_right, rf00, zero)
    rf01 = jnp.where(has_right, rf01, zero)
    rf10 = jnp.where(has_right, rf10, zero)
    rf11 = jnp.where(has_right, rf11, zero)

    nl00, nl01, nl10, nl11 = _matmul2_components(
        lf00,
        lf01,
        lf10,
        lf11,
        load(lower00_ref, left),
        load(lower01_ref, left),
        load(lower10_ref, left),
        load(lower11_ref, left),
    )
    nu00, nu01, nu10, nu11 = _matmul2_components(
        rf00,
        rf01,
        rf10,
        rf11,
        load(upper00_ref, right),
        load(upper01_ref, right),
        load(upper10_ref, right),
        load(upper11_ref, right),
    )
    ldu00, ldu01, ldu10, ldu11 = _matmul2_components(
        lf00,
        lf01,
        lf10,
        lf11,
        load(upper00_ref, left),
        load(upper01_ref, left),
        load(upper10_ref, left),
        load(upper11_ref, left),
    )
    rdl00, rdl01, rdl10, rdl11 = _matmul2_components(
        rf00,
        rf01,
        rf10,
        rf11,
        load(lower00_ref, right),
        load(lower01_ref, right),
        load(lower10_ref, right),
        load(lower11_ref, right),
    )
    lr0, lr1 = _matvec2_components(
        lf00,
        lf01,
        lf10,
        lf11,
        load(r0_ref, left),
        load(r1_ref, left),
    )
    rr0, rr1 = _matvec2_components(
        rf00,
        rf01,
        rf10,
        rf11,
        load(r0_ref, right),
        load(r1_ref, right),
    )

    def store(ref, value):
        _pallas_store(ref, (slice(None), pl.ds(0, 1)), value[:, None])

    store(out_lower00_ref, -nl00)
    store(out_lower01_ref, -nl01)
    store(out_lower10_ref, -nl10)
    store(out_lower11_ref, -nl11)
    store(out_upper00_ref, -nu00)
    store(out_upper01_ref, -nu01)
    store(out_upper10_ref, -nu10)
    store(out_upper11_ref, -nu11)
    store(out_diag00_ref, load(diag00_ref, current) - ldu00 - rdl00)
    store(out_diag01_ref, load(diag01_ref, current) - ldu01 - rdl01)
    store(out_diag10_ref, load(diag10_ref, current) - ldu10 - rdl10)
    store(out_diag11_ref, load(diag11_ref, current) - ldu11 - rdl11)
    store(out_r0_ref, load(r0_ref, current) - lr0 - rr0)
    store(out_r1_ref, load(r1_ref, current) - lr1 - rr1)


def _pallas_thomas_2x2_kernel(
    a00_ref,
    a01_ref,
    a10_ref,
    a11_ref,
    off0_ref,
    off1_ref,
    rhs0_ref,
    rhs1_ref,
    out_ref,
    scratch_ref,
    *,
    n: int,
) -> None:
    zero = jnp.zeros((a00_ref.shape[0],), dtype=a00_ref.dtype)

    def inv_components(
        m00: Array,
        m01: Array,
        m10: Array,
        m11: Array,
    ) -> tuple[Array, Array, Array, Array]:
        det = m00 * m11 - m01 * m10
        return m11 / det, -m01 / det, -m10 / det, m00 / det

    inv00, inv01, inv10, inv11 = inv_components(
        a00_ref[:, 0],
        a01_ref[:, 0],
        a10_ref[:, 0],
        a11_ref[:, 0],
    )
    c00 = inv00 * off0_ref[:, 0]
    c01 = inv01 * off1_ref[:, 0]
    c10 = inv10 * off0_ref[:, 0]
    c11 = inv11 * off1_ref[:, 0]
    d0 = inv00 * rhs0_ref[:, 0] + inv01 * rhs1_ref[:, 0]
    d1 = inv10 * rhs0_ref[:, 0] + inv11 * rhs1_ref[:, 0]
    _store_forward_row(scratch_ref, 0, c00, c01, c10, c11, d0, d1)

    def forward_body(i: int, carry: tuple[Array, Array, Array, Array, Array, Array]):
        c00_prev, c01_prev, c10_prev, c11_prev, d0_prev, d1_prev = carry
        lower0 = off0_ref[:, i - 1]
        lower1 = off1_ref[:, i - 1]
        upper0 = off0_ref[:, i]
        upper1 = off1_ref[:, i]

        m00 = a00_ref[:, i] - lower0 * c00_prev
        m01 = a01_ref[:, i] - lower0 * c01_prev
        m10 = a10_ref[:, i] - lower1 * c10_prev
        m11 = a11_ref[:, i] - lower1 * c11_prev
        inv00_i, inv01_i, inv10_i, inv11_i = inv_components(m00, m01, m10, m11)

        r0 = rhs0_ref[:, i] - lower0 * d0_prev
        r1 = rhs1_ref[:, i] - lower1 * d1_prev
        c00_i = inv00_i * upper0
        c01_i = inv01_i * upper1
        c10_i = inv10_i * upper0
        c11_i = inv11_i * upper1
        d0_i = inv00_i * r0 + inv01_i * r1
        d1_i = inv10_i * r0 + inv11_i * r1
        _store_forward_row(
            scratch_ref,
            i,
            c00_i,
            c01_i,
            c10_i,
            c11_i,
            d0_i,
            d1_i,
        )
        return c00_i, c01_i, c10_i, c11_i, d0_i, d1_i

    c00, c01, c10, c11, d0, d1 = jax.lax.fori_loop(
        1,
        n - 1,
        forward_body,
        (c00, c01, c10, c11, d0, d1),
    )

    i = n - 1
    lower0 = off0_ref[:, i - 1]
    lower1 = off1_ref[:, i - 1]
    m00 = a00_ref[:, i] - lower0 * c00
    m01 = a01_ref[:, i] - lower0 * c01
    m10 = a10_ref[:, i] - lower1 * c10
    m11 = a11_ref[:, i] - lower1 * c11
    inv00, inv01, inv10, inv11 = inv_components(m00, m01, m10, m11)
    r0 = rhs0_ref[:, i] - lower0 * d0
    r1 = rhs1_ref[:, i] - lower1 * d1
    d0 = inv00 * r0 + inv01 * r1
    d1 = inv10 * r0 + inv11 * r1
    _store_forward_row(scratch_ref, i, zero, zero, zero, zero, d0, d1)
    _store_output_component(out_ref, i, 0, d0)
    _store_output_component(out_ref, i, 1, d1)

    def backward_body(k: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        next0, next1 = carry
        row = (n - 2) - k
        c00_i = _load_forward_component(scratch_ref, row, 0)
        c01_i = _load_forward_component(scratch_ref, row, 1)
        c10_i = _load_forward_component(scratch_ref, row, 2)
        c11_i = _load_forward_component(scratch_ref, row, 3)
        d0_i = _load_forward_component(scratch_ref, row, 4)
        d1_i = _load_forward_component(scratch_ref, row, 5)
        x0 = d0_i - c00_i * next0 - c01_i * next1
        x1 = d1_i - c10_i * next0 - c11_i * next1
        _store_output_component(out_ref, row, 0, x0)
        _store_output_component(out_ref, row, 1, x1)
        return x0, x1

    jax.lax.fori_loop(0, n - 1, backward_body, (d0, d1))


def _store_forward_row(
    scratch_ref,
    row: int,
    c00: Array,
    c01: Array,
    c10: Array,
    c11: Array,
    d0: Array,
    d1: Array,
) -> None:
    _store_forward_component(scratch_ref, row, 0, c00)
    _store_forward_component(scratch_ref, row, 1, c01)
    _store_forward_component(scratch_ref, row, 2, c10)
    _store_forward_component(scratch_ref, row, 3, c11)
    _store_forward_component(scratch_ref, row, 4, d0)
    _store_forward_component(scratch_ref, row, 5, d1)


def _store_forward_component(
    scratch_ref,
    row: int,
    component: int,
    value: Array,
) -> None:
    _pallas_store(
        scratch_ref,
        (slice(None), pl.ds(row, 1), pl.ds(component, 1)),
        value[:, None, None],
    )


def _load_forward_component(
    scratch_ref,
    row: int,
    component: int,
) -> Array:
    return _pallas_load(scratch_ref, (slice(None), pl.ds(row, 1), pl.ds(component, 1)))[
        :, 0, 0
    ]


def _store_output_component(
    out_ref,
    row: int,
    component: int,
    value: Array,
) -> None:
    _pallas_store(
        out_ref,
        (slice(None), pl.ds(row, 1), pl.ds(component, 1)),
        value[:, None, None],
    )
