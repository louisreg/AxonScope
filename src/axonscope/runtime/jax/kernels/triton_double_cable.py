"""Backend-private jax-triton double-cable tiled-Thomas kernel.

The active GPU runtime exposes one double-cable Triton route. It assembles and
solves node-first ``[Nx, B]`` systems with one Triton program per axon tile and
a looped ``tl.range`` Thomas recurrence to avoid materializing coefficients or
the static-unroll cold-start cost seen during P11C.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType
from typing import Any


_KERNEL_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

try:  # pragma: no cover - exercised on CUDA benchmark workers.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - common local CPU/dev path.
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _unused_generated_membrane_at(
        Vm,
        gates_ptr,
        dt_ptr,
        parameters_ptr,
        gates_out_ptr,
        offset,
        mask,
        linearize_previous: tl.constexpr,
    ):
        return 0.0, 0.0

    def _tiled_block_thomas_fused_loop_kernel_template(
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
        gates,
        static_gates,
        dt,
        membrane_parameters,
        area,
        iinj_abs,
        i_outward_abs,
        i_corr_abs,
        extracellular_drive_abs,
        c00,
        c01,
        c10,
        c11,
        gates_out,
        out0,
        out1,
        N: tl.constexpr,
        B: tl.constexpr,
        FUSED_MEMBRANE: tl.constexpr,
        BLEND_STATIC_GATES: tl.constexpr,
        LINEARIZE_PREVIOUS: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        tile = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_B)
        batch = tile * BLOCK_B + lanes
        mask = batch < B

        offset = batch
        cm = tl.load(cm_over_dt + offset, mask=mask, other=0.0)
        cx = tl.load(cx_over_dt + offset, mask=mask, other=0.0)
        vi_value = tl.load(vi + offset, mask=mask, other=0.0)
        ve_value = tl.load(ve + offset, mask=mask, other=0.0)
        area_value = tl.load(area + offset, mask=mask, other=0.0)
        if FUSED_MEMBRANE:
            gated_gm, gated_ge = _generated_membrane_at(
                vi_value - ve_value,
                gates,
                dt,
                membrane_parameters,
                gates_out,
                offset,
                mask,
                LINEARIZE_PREVIOUS,
            )
            if BLEND_STATIC_GATES:
                leak_offset = offset * 3
                leak_gm = tl.load(static_gates + leak_offset, mask=mask, other=0.0)
                leak_ge = tl.load(static_gates + leak_offset + 1, mask=mask, other=0.0)
                gated_mask = tl.load(static_gates + leak_offset + 2, mask=mask, other=0.0)
                gm_value = gated_mask * gated_gm + (1.0 - gated_mask) * leak_gm
                ge_value = gated_mask * gated_ge + (1.0 - gated_mask) * leak_ge
            else:
                gm_value = gated_gm
                ge_value = gated_ge
        else:
            gm_value = tl.load(gm_density + offset, mask=mask, other=0.0)
            ge_value = tl.load(ge_density + offset, mask=mask, other=0.0)
        gm_abs = gm_value * area_value
        ge_abs = ge_value * area_value
        charge = cm * (vi_value - ve_value)
        iout = tl.load(i_outward_abs + offset, mask=mask, other=0.0)
        icorr = tl.load(i_corr_abs + offset, mask=mask, other=0.0)
        m00 = tl.load(a00_static + offset, mask=mask, other=1.0) + gm_abs
        m01 = -(cm + gm_abs)
        m10 = m01
        m11 = tl.load(a11_static + offset, mask=mask, other=1.0) + gm_abs
        r0 = charge + ge_abs + tl.load(iinj_abs + offset, mask=mask, other=0.0) - iout - icorr
        r1 = (
            -charge
            - ge_abs
            + cx * ve_value
            + tl.load(extracellular_drive_abs + offset, mask=mask, other=0.0)
            + iout
            + icorr
        )
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
        tl.store(out0 + offset, dp0, mask=mask)
        tl.store(out1 + offset, dp1, mask=mask)

        prev_c00 = cp00
        prev_c01 = cp01
        prev_c10 = cp10
        prev_c11 = cp11
        prev_d0 = dp0
        prev_d1 = dp1

        if N > 2:
            for x in tl.range(1, N - 1):
                offset = x * B + batch
                edge_prev = (x - 1) * B + batch
                l0 = tl.load(off0 + edge_prev, mask=mask, other=0.0)
                l1 = tl.load(off1 + edge_prev, mask=mask, other=0.0)
                cm = tl.load(cm_over_dt + offset, mask=mask, other=0.0)
                cx = tl.load(cx_over_dt + offset, mask=mask, other=0.0)
                vi_value = tl.load(vi + offset, mask=mask, other=0.0)
                ve_value = tl.load(ve + offset, mask=mask, other=0.0)
                area_value = tl.load(area + offset, mask=mask, other=0.0)
                if FUSED_MEMBRANE:
                    gated_gm, gated_ge = _generated_membrane_at(
                        vi_value - ve_value,
                        gates,
                        dt,
                        membrane_parameters,
                        gates_out,
                        offset,
                        mask,
                        LINEARIZE_PREVIOUS,
                    )
                    if BLEND_STATIC_GATES:
                        leak_offset = offset * 3
                        leak_gm = tl.load(static_gates + leak_offset, mask=mask, other=0.0)
                        leak_ge = tl.load(static_gates + leak_offset + 1, mask=mask, other=0.0)
                        gated_mask = tl.load(static_gates + leak_offset + 2, mask=mask, other=0.0)
                        gm_value = gated_mask * gated_gm + (1.0 - gated_mask) * leak_gm
                        ge_value = gated_mask * gated_ge + (1.0 - gated_mask) * leak_ge
                    else:
                        gm_value = gated_gm
                        ge_value = gated_ge
                else:
                    gm_value = tl.load(gm_density + offset, mask=mask, other=0.0)
                    ge_value = tl.load(ge_density + offset, mask=mask, other=0.0)
                gm_abs = gm_value * area_value
                ge_abs = ge_value * area_value
                charge = cm * (vi_value - ve_value)
                iout = tl.load(i_outward_abs + offset, mask=mask, other=0.0)
                icorr = tl.load(i_corr_abs + offset, mask=mask, other=0.0)
                a01_value = -(cm + gm_abs)
                r0_value = (
                    charge
                    + ge_abs
                    + tl.load(iinj_abs + offset, mask=mask, other=0.0)
                    - iout
                    - icorr
                )
                r1_value = (
                    -charge
                    - ge_abs
                    + cx * ve_value
                    + tl.load(extracellular_drive_abs + offset, mask=mask, other=0.0)
                    + iout
                    + icorr
                )
                m00 = tl.load(a00_static + offset, mask=mask, other=1.0) + gm_abs - l0 * prev_c00
                m01 = a01_value - l0 * prev_c01
                m10 = a01_value - l1 * prev_c10
                m11 = tl.load(a11_static + offset, mask=mask, other=1.0) + gm_abs - l1 * prev_c11
                r0 = r0_value - l0 * prev_d0
                r1 = r1_value - l1 * prev_d1

                det = m00 * m11 - m01 * m10
                inv00 = m11 / det
                inv01 = -m01 / det
                inv10 = -m10 / det
                inv11 = m00 / det

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
                tl.store(out0 + offset, dp0, mask=mask)
                tl.store(out1 + offset, dp1, mask=mask)
                prev_c00 = cp00
                prev_c01 = cp01
                prev_c10 = cp10
                prev_c11 = cp11
                prev_d0 = dp0
                prev_d1 = dp1

        x = N - 1
        offset = x * B + batch
        edge_prev = (x - 1) * B + batch
        l0 = tl.load(off0 + edge_prev, mask=mask, other=0.0)
        l1 = tl.load(off1 + edge_prev, mask=mask, other=0.0)
        cm = tl.load(cm_over_dt + offset, mask=mask, other=0.0)
        cx = tl.load(cx_over_dt + offset, mask=mask, other=0.0)
        vi_value = tl.load(vi + offset, mask=mask, other=0.0)
        ve_value = tl.load(ve + offset, mask=mask, other=0.0)
        area_value = tl.load(area + offset, mask=mask, other=0.0)
        if FUSED_MEMBRANE:
            gated_gm, gated_ge = _generated_membrane_at(
                vi_value - ve_value,
                gates,
                dt,
                membrane_parameters,
                gates_out,
                offset,
                mask,
                LINEARIZE_PREVIOUS,
            )
            if BLEND_STATIC_GATES:
                leak_offset = offset * 3
                leak_gm = tl.load(static_gates + leak_offset, mask=mask, other=0.0)
                leak_ge = tl.load(static_gates + leak_offset + 1, mask=mask, other=0.0)
                gated_mask = tl.load(static_gates + leak_offset + 2, mask=mask, other=0.0)
                gm_value = gated_mask * gated_gm + (1.0 - gated_mask) * leak_gm
                ge_value = gated_mask * gated_ge + (1.0 - gated_mask) * leak_ge
            else:
                gm_value = gated_gm
                ge_value = gated_ge
        else:
            gm_value = tl.load(gm_density + offset, mask=mask, other=0.0)
            ge_value = tl.load(ge_density + offset, mask=mask, other=0.0)
        gm_abs = gm_value * area_value
        ge_abs = ge_value * area_value
        charge = cm * (vi_value - ve_value)
        iout = tl.load(i_outward_abs + offset, mask=mask, other=0.0)
        icorr = tl.load(i_corr_abs + offset, mask=mask, other=0.0)
        a01_value = -(cm + gm_abs)
        r0_value = charge + ge_abs + tl.load(iinj_abs + offset, mask=mask, other=0.0) - iout - icorr
        r1_value = (
            -charge
            - ge_abs
            + cx * ve_value
            + tl.load(extracellular_drive_abs + offset, mask=mask, other=0.0)
            + iout
            + icorr
        )
        m00 = tl.load(a00_static + offset, mask=mask, other=1.0) + gm_abs - l0 * prev_c00
        m01 = a01_value - l0 * prev_c01
        m10 = a01_value - l1 * prev_c10
        m11 = tl.load(a11_static + offset, mask=mask, other=1.0) + gm_abs - l1 * prev_c11
        r0 = r0_value - l0 * prev_d0
        r1 = r1_value - l1 * prev_d1

        det = m00 * m11 - m01 * m10
        inv00 = m11 / det
        inv01 = -m01 / det
        inv10 = -m10 / det
        inv11 = m00 / det

        cp00 = tl.full((BLOCK_B,), 0.0, tl.float32)
        cp01 = tl.full((BLOCK_B,), 0.0, tl.float32)
        cp10 = tl.full((BLOCK_B,), 0.0, tl.float32)
        cp11 = tl.full((BLOCK_B,), 0.0, tl.float32)
        dp0 = inv00 * r0 + inv01 * r1
        dp1 = inv10 * r0 + inv11 * r1

        tl.store(c00 + offset, cp00, mask=mask)
        tl.store(c01 + offset, cp01, mask=mask)
        tl.store(c10 + offset, cp10, mask=mask)
        tl.store(c11 + offset, cp11, mask=mask)
        tl.store(out0 + offset, dp0, mask=mask)
        tl.store(out1 + offset, dp1, mask=mask)

        x0 = dp0
        x1 = dp1

        for rev in tl.range(0, N - 1):
            x = N - 2 - rev
            offset = x * B + batch
            cp00 = tl.load(c00 + offset, mask=mask, other=0.0)
            cp01 = tl.load(c01 + offset, mask=mask, other=0.0)
            cp10 = tl.load(c10 + offset, mask=mask, other=0.0)
            cp11 = tl.load(c11 + offset, mask=mask, other=0.0)
            next_x0 = x0
            next_x1 = x1
            x0 = tl.load(out0 + offset, mask=mask, other=0.0) - cp00 * next_x0 - cp01 * next_x1
            x1 = tl.load(out1 + offset, mask=mask, other=0.0) - cp10 * next_x0 - cp11 * next_x1
            tl.store(out0 + offset, x0, mask=mask)
            tl.store(out1 + offset, x1, mask=mask)

    def _make_tiled_block_thomas_kernel(generated_membrane_at):
        namespace = dict(_tiled_block_thomas_fused_loop_kernel_template.__globals__)
        namespace["_generated_membrane_at"] = generated_membrane_at
        function = FunctionType(
            _tiled_block_thomas_fused_loop_kernel_template.__code__,
            namespace,
            name=_tiled_block_thomas_fused_loop_kernel_template.__name__,
            argdefs=_tiled_block_thomas_fused_loop_kernel_template.__defaults__,
            closure=_tiled_block_thomas_fused_loop_kernel_template.__closure__,
        )
        function.__module__ = __name__
        function.__qualname__ = _tiled_block_thomas_fused_loop_kernel_template.__qualname__
        return triton.jit(function)

    _tiled_block_thomas_fused_loop_kernel = _make_tiled_block_thomas_kernel(
        _unused_generated_membrane_at
    )
    _GENERATED_THOMAS_KERNELS = {}

else:
    _tiled_block_thomas_fused_loop_kernel = None
    _GENERATED_THOMAS_KERNELS = {}


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


def solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb(
    a00_static: Any,
    a11_static: Any,
    cm_over_dt: Any,
    cx_over_dt: Any,
    off0: Any,
    off1: Any,
    vi: Any,
    ve: Any,
    gm_density: Any,
    ge_density: Any,
    area: Any,
    iinj_abs: Any,
    i_outward_abs: Any,
    i_corr_abs: Any,
    extracellular_drive_abs: Any,
    *,
    block_b: int = 128,
    membrane_plan: Any | None = None,
    gates: Any | None = None,
    static_gates: Any | None = None,
    dt_ms: Any | None = None,
    linearize_previous: bool = False,
) -> tuple[Any, ...]:
    """Assemble and solve ``[Nx, B]`` systems with the retained GPU route."""

    skip_reason = jax_triton_thomas_dependency_skip_reason()
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if int(block_b) < 1:
        raise ValueError("block_b must be >= 1.")

    import jax
    import jax.numpy as jnp

    from .triton_call_cache import cached_triton_call

    vi = jnp.asarray(vi)
    ve = jnp.asarray(ve)
    _check_space_pair_xb(vi, ve)
    if vi.dtype != jnp.float32:
        raise TypeError(
            f"jax_triton_tiled_thomas_loop supports only float32, got {vi.dtype}."
        )
    nx, batch_size = int(vi.shape[0]), int(vi.shape[1])

    a00_static = _space_tensor_xb(a00_static, batch_size=batch_size, nx=nx, name="a00_static")
    a11_static = _space_tensor_xb(a11_static, batch_size=batch_size, nx=nx, name="a11_static")
    cm_over_dt = _space_tensor_xb(cm_over_dt, batch_size=batch_size, nx=nx, name="cm_over_dt")
    cx_over_dt = _space_tensor_xb(cx_over_dt, batch_size=batch_size, nx=nx, name="cx_over_dt")
    off0 = _edge_tensor_xb(off0, batch_size=batch_size, nx=nx, name="off0")
    off1 = _edge_tensor_xb(off1, batch_size=batch_size, nx=nx, name="off1")
    physical = tuple(
        _space_tensor_xb(value, batch_size=batch_size, nx=nx, name=name)
        for name, value in (
            ("area", area),
            ("iinj_abs", iinj_abs),
            ("i_outward_abs", i_outward_abs),
            ("i_corr_abs", i_corr_abs),
            ("extracellular_drive_abs", extracellular_drive_abs),
        )
    )

    dummy = jnp.zeros((1,), dtype=vi.dtype)
    if membrane_plan is None:
        gm_xb = _space_tensor_xb(
            gm_density, batch_size=batch_size, nx=nx, name="gm_density"
        )
        ge_xb = _space_tensor_xb(
            ge_density, batch_size=batch_size, nx=nx, name="ge_density"
        )
        gates_input = dummy
        static_input = dummy
        dt_input = dummy
        parameters_input = dummy
        gate_shape = jax.ShapeDtypeStruct(dummy.shape, dummy.dtype)
        kernel = _tiled_block_thomas_fused_loop_kernel
        source_hash = _KERNEL_SOURCE_HASH
    else:
        if gates is None or dt_ms is None:
            raise ValueError("Fused membrane execution requires gates and dt_ms.")
        gates_input = jnp.asarray(gates, dtype=vi.dtype)
        expected_gate_shape = (nx, batch_size, int(membrane_plan.gate_count))
        if tuple(gates_input.shape) != expected_gate_shape:
            raise ValueError(
                f"gates must have shape {expected_gate_shape}, got {gates_input.shape}."
            )
        if membrane_plan.blends_static_gates:
            if static_gates is None:
                raise ValueError("Static gate blending requires static_gates.")
            static_input = jnp.asarray(static_gates, dtype=vi.dtype)
            if tuple(static_input.shape) != (nx, batch_size, 3):
                raise ValueError(
                    "static_gates must have shape "
                    f"({nx}, {batch_size}, 3), got {static_input.shape}."
                )
        else:
            static_input = dummy
        dt_input = jnp.asarray(dt_ms, dtype=vi.dtype).reshape((1,))
        parameters_input = jnp.stack(
            tuple(
                jnp.asarray(value, dtype=vi.dtype)
                for value in membrane_plan.parameter_values
            )
        )
        gate_shape = jax.ShapeDtypeStruct(gates_input.shape, gates_input.dtype)
        cache_key = str(membrane_plan.module.CACHE_KEY)
        kernel = _GENERATED_THOMAS_KERNELS.get(cache_key)
        if kernel is None:
            kernel = _make_tiled_block_thomas_kernel(
                membrane_plan.module.advance_gates_and_membrane_terms_at
            )
            _GENERATED_THOMAS_KERNELS[cache_key] = kernel
        source_hash = (
            f"{_KERNEL_SOURCE_HASH}:{cache_key}:{membrane_plan.module.SOURCE_HASH}"
        )
        gm_xb = dummy
        ge_xb = dummy

    work_shape = jax.ShapeDtypeStruct(vi.shape, vi.dtype)
    grid = ((batch_size + int(block_b) - 1) // int(block_b),)
    *_, gates_out, out0, out1 = cached_triton_call(
        a00_static,
        a11_static,
        cm_over_dt,
        cx_over_dt,
        off0,
        off1,
        vi,
        ve,
        gm_xb,
        ge_xb,
        gates_input,
        static_input,
        dt_input,
        parameters_input,
        *physical,
        kernel=kernel,
        source_hash=source_hash,
        out_shape=((work_shape,) * 4 + (gate_shape, work_shape, work_shape)),
        grid=grid,
        name="axonscope_double_cable_tiled_thomas",
        N=nx,
        B=batch_size,
        FUSED_MEMBRANE=membrane_plan is not None,
        BLEND_STATIC_GATES=(
            False if membrane_plan is None else membrane_plan.blends_static_gates
        ),
        LINEARIZE_PREVIOUS=bool(linearize_previous),
        BLOCK_B=int(block_b),
        num_warps=_num_warps_for_block_b(int(block_b)),
        num_stages=1,
    )
    if membrane_plan is not None:
        return gates_out, out0, out1
    return out0, out1


def _check_space_pair_xb(first: Any, second: Any) -> None:
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("vi and ve must have shape (Nx, batch_size).")
    if tuple(first.shape) != tuple(second.shape):
        raise ValueError(
            f"vi and ve must have the same shape, got {first.shape} and {second.shape}."
        )
    if int(first.shape[0]) < 2:
        raise ValueError("Nx must be >= 2 for the jax-triton Thomas backend.")


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
        f"{name} must have shape ({edge_n},) or ({edge_n}, {batch_size}), "
        f"got {tuple(tensor.shape)}."
    )


def _num_warps_for_block_b(block_b: int) -> int:
    if block_b >= 128:
        return 4
    if block_b >= 64:
        return 2
    return 1


__all__ = [
    "jax_triton_thomas_dependency_skip_reason",
    "solve_double_cable_physical_jax_triton_tiled_thomas_loop_xb",
]
