"""Shared JAX batch-kernel input coercion helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from axonfleet.runtime.inputs.payloads import (
    FactorizedExtracellularPotentialBatch,
    SparseIntracellularCurrentDensityBatch,
)
from axonfleet.runtime.jax.cable_geometry import Array
from axonfleet.runtime.jax.preparation.caches import (
    get_batched_static_array,
    store_batched_static_array,
)
from axonfleet.solvers.options import BatchOptions, BatchRecording


def _cached_broadcast_batch_leading(values: Array, batch_size: int) -> Array:
    arr = jnp.asarray(values)
    key = _batched_static_array_cache_key(
        "leading",
        values,
        arr=arr,
        batch_size=batch_size,
    )
    cached = get_batched_static_array(key, sources=(values,))
    if cached is not None:
        return cached
    out = jnp.broadcast_to(arr, (batch_size, *arr.shape))
    store_batched_static_array(key, out, sources=(values,))
    return out

def _cached_constant_batched_space_array(
    name: str,
    value: float,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    key = (
        "constant_batched_space_v1",
        name,
        float(value),
        int(nx),
        str(jnp.dtype(dtype_local)),
        int(batch_size),
        _current_jax_device_key(),
    )
    cached = get_batched_static_array(key)
    if cached is not None:
        return cached
    out = jnp.full(
        (int(batch_size), int(nx)),
        jnp.asarray(value, dtype=dtype_local),
        dtype=dtype_local,
    )
    store_batched_static_array(key, out)
    return out

def _cached_single_cable_tridiagonal_coefficients(
    *,
    lower: Array,
    diag: Array,
    upper: Array,
    dt: Array,
    dt_ms: float,
) -> tuple[Array, Array, Array]:
    """Return cached Crank-Nicolson tridiagonal coefficients."""

    diag_arr = jnp.asarray(diag)
    key = (
        "single_cable_tridiagonal_coefficients_v1",
        id(lower),
        id(diag),
        id(upper),
        tuple(int(dim) for dim in diag_arr.shape),
        str(diag_arr.dtype),
        float(dt_ms),
        _current_jax_device_key(),
    )
    cached = get_batched_static_array(key, sources=(lower, diag, upper))
    if cached is not None:
        return cached
    dt_arr = jnp.asarray(dt, dtype=diag_arr.dtype)
    coeffs = (
        -dt_arr * lower,
        jnp.ones_like(diag_arr) - dt_arr * diag_arr,
        -dt_arr * upper,
    )
    store_batched_static_array(key, coeffs, sources=(lower, diag, upper))
    return coeffs

def _normalize_batch_options(options: BatchOptions | None) -> BatchOptions:
    return BatchOptions.full() if options is None else options

def _resolve_recording(recording: BatchRecording, *, nx: int) -> tuple[Array, bool]:
    indices = recording.indices_for(nx)
    if indices is None:
        return jnp.arange(nx, dtype=jnp.int32), True
    return jnp.asarray(indices, dtype=jnp.int32), False

def _resolve_output_recording(options: Any, *, nx: int) -> tuple[Array, bool]:
    row_indices = getattr(options, "row_record_indices", None)
    if row_indices is not None:
        indices = jnp.asarray(row_indices, dtype=jnp.int32)
        if indices.ndim != 2:
            raise ValueError("row_record_indices must have shape (batch, width).")
        return indices, False
    return _resolve_recording(options.recording, nx=nx)

def _record_vm_row(vm: Array, record_indices: Array, *, record_full: bool) -> Array:
    if record_full:
        return vm
    return jnp.take(vm, record_indices, axis=0)

def _record_vm_batch(vm: Array, record_indices: Array, *, record_full: bool) -> Array:
    if record_full:
        return vm
    indices = jnp.asarray(record_indices, dtype=jnp.int32)
    if indices.ndim == 1:
        return jnp.take(vm, indices, axis=1)
    if indices.ndim != 2:
        raise ValueError("batch record_indices must have shape (width,) or (batch, width).")
    return jnp.take_along_axis(vm, indices, axis=1)

def _as_sparse_intracellular_current_density_batch(
    name: str,
    values: SparseIntracellularCurrentDensityBatch,
    *,
    nt: int,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> SparseIntracellularCurrentDensityBatch:
    density_mid = jnp.asarray(values.density_mid, dtype=dtype_local)
    indices = jnp.asarray(values.indices, dtype=jnp.int32)
    mask = jnp.asarray(values.mask, dtype=bool)
    if int(values.target_nx) != int(nx):
        raise ValueError(f"{name}.target_nx must be {nx}, got {values.target_nx}.")
    if density_mid.ndim != 3:
        raise ValueError(f"{name}.density_mid must have shape (B, Nt, K).")
    if density_mid.shape[:2] != (batch_size, nt):
        raise ValueError(
            f"{name}.density_mid must have leading shape (B, Nt)="
            f"({batch_size}, {nt}), got {density_mid.shape}."
        )
    sparse_shape = (batch_size, int(density_mid.shape[2]))
    if indices.shape != sparse_shape:
        raise ValueError(f"{name}.indices must have shape {sparse_shape}, got {indices.shape}.")
    if mask.shape != sparse_shape:
        raise ValueError(f"{name}.mask must have shape {sparse_shape}, got {mask.shape}.")
    if int(indices.shape[1]) > 0 and bool(
        jnp.any(jnp.where(mask, (indices < 0) | (indices >= nx), False))
    ):
        raise ValueError(f"{name}.indices contains an out-of-range compartment index.")
    return SparseIntracellularCurrentDensityBatch(
        density_mid=density_mid,
        indices=indices,
        mask=mask,
        target_nx=nx,
    )

def _as_factorized_extracellular_potential_batch(
    name: str,
    values: FactorizedExtracellularPotentialBatch,
    *,
    nt: int,
    nx: int,
    dtype_local: jnp.dtype,
) -> FactorizedExtracellularPotentialBatch:
    current_mid_A = jnp.asarray(values.current_mid_A, dtype=dtype_local)
    current_initial_previous_A = (
        None
        if values.current_initial_previous_A is None
        else jnp.asarray(values.current_initial_previous_A, dtype=dtype_local)
    )
    current_row_indices = (
        None
        if values.current_row_indices is None
        else jnp.asarray(values.current_row_indices, dtype=jnp.int32)
    )
    current_row_scales = (
        None
        if values.current_row_scales is None
        else jnp.asarray(values.current_row_scales, dtype=dtype_local)
    )
    footprint_mV_per_A = jnp.asarray(values.footprint_mV_per_A, dtype=dtype_local)
    forcing_footprint_mV_per_A = (
        None
        if values.single_cable_forcing_footprint_mV_per_A is None
        else jnp.asarray(
            values.single_cable_forcing_footprint_mV_per_A,
            dtype=dtype_local,
        )
    )
    if int(values.target_nx) != int(nx):
        raise ValueError(f"{name}.target_nx must be {nx}, got {values.target_nx}.")
    if footprint_mV_per_A.ndim not in {2, 3} or footprint_mV_per_A.shape[-1] != nx:
        raise ValueError(
            f"{name}.footprint_mV_per_A must have shape (B, Nx) or "
            f"(B, K, Nx) with Nx={nx}, "
            f"got {footprint_mV_per_A.shape}."
        )
    batch_size = int(footprint_mV_per_A.shape[0])
    drive_count = 1 if footprint_mV_per_A.ndim == 2 else int(footprint_mV_per_A.shape[1])
    if current_row_indices is not None and current_row_scales is not None:
        raise ValueError(
            f"{name}.current_row_indices and current_row_scales are mutually exclusive."
        )
    if footprint_mV_per_A.ndim == 2 and current_mid_A.ndim == 1:
        if current_row_indices is not None:
            raise ValueError(f"{name}.current_row_indices require current_mid_A shape (U, Nt).")
        if current_mid_A.shape != (nt,):
            raise ValueError(
                f"{name}.current_mid_A must have shape (Nt,)=({nt},), "
                f"got {current_mid_A.shape}."
            )
        if current_row_scales is not None and current_row_scales.shape not in {
            (batch_size,),
            (batch_size, 1),
        }:
            raise ValueError(
                f"{name}.current_row_scales must have shape (B,) or (B, 1), "
                f"B={batch_size}, got {current_row_scales.shape}."
            )
    elif footprint_mV_per_A.ndim == 2 and current_mid_A.ndim == 2:
        if current_row_scales is not None:
            raise ValueError(
                f"{name}.current_row_scales with rank-1 footprints require "
                "current_mid_A shape (Nt,)."
            )
        if current_row_indices is None:
            valid_current = current_mid_A.shape == (batch_size, nt)
            expected = f"(B, Nt)=({batch_size}, {nt})"
        else:
            valid_current = (
                current_mid_A.shape[1] == nt
                and current_row_indices.shape == (batch_size,)
                and current_mid_A.shape[0] >= 1
            )
            expected = f"(U, Nt) with current_row_indices (B,), Nt={nt}, B={batch_size}"
        if not valid_current:
            raise ValueError(
                f"{name}.current_mid_A must have shape {expected}, "
                f"got current={current_mid_A.shape} and "
                f"indices={None if current_row_indices is None else current_row_indices.shape}."
            )
    elif footprint_mV_per_A.ndim == 3 and current_mid_A.ndim == 2:
        if current_row_indices is not None:
            raise ValueError(f"{name}.current_row_indices are only valid for rank-1 batches.")
        valid_current = current_mid_A.shape == (drive_count, nt)
        valid_scales = (
            current_row_scales is None
            or current_row_scales.shape == (batch_size, drive_count)
        )
        if not valid_current or not valid_scales:
            raise ValueError(
                f"{name}.current_mid_A must have shape (S, Nt)="
                f"({drive_count}, {nt}) and current_row_scales must be absent "
                f"or shape (B, S)=({batch_size}, {drive_count}); got "
                f"current={current_mid_A.shape}, "
                f"scales={None if current_row_scales is None else current_row_scales.shape}."
            )
    elif footprint_mV_per_A.ndim == 3 and current_mid_A.ndim == 3:
        if current_row_scales is not None:
            raise ValueError(
                f"{name}.current_row_scales with row-specific multi-drive current "
                "require current_mid_A shape (S, Nt)."
            )
        expected_rows = batch_size if current_row_indices is None else int(current_mid_A.shape[0])
        valid_current = (
            current_mid_A.shape == (expected_rows, drive_count, nt)
            and expected_rows >= 1
            and (
                current_row_indices is None
                or current_row_indices.shape == (batch_size,)
            )
        )
        if not valid_current:
            raise ValueError(
                f"{name}.current_mid_A must have shape (B, K, Nt) or "
                f"(U, K, Nt) with current_row_indices (B,), got "
                f"current={current_mid_A.shape} and "
                f"indices={None if current_row_indices is None else current_row_indices.shape}."
            )
    else:
        raise ValueError(
            f"{name}.current_mid_A shape {current_mid_A.shape} is incompatible "
            f"with footprint shape {footprint_mV_per_A.shape}."
        )
    if current_initial_previous_A is not None:
        if current_row_scales is not None:
            if footprint_mV_per_A.ndim == 2:
                valid_previous = current_initial_previous_A.ndim == 0
                expected = "scalar"
            else:
                valid_previous = current_initial_previous_A.shape == (drive_count,)
                expected = f"(S,)=({drive_count},)"
        elif footprint_mV_per_A.ndim == 2 and current_row_indices is None:
            valid_previous = current_initial_previous_A.ndim == 0 or (
                current_initial_previous_A.shape == (batch_size,)
            )
            expected = f"scalar or (B,)=({batch_size},)"
        elif footprint_mV_per_A.ndim == 2:
            valid_previous = current_initial_previous_A.shape == (
                int(current_mid_A.shape[0]),
            )
            expected = f"(U,), U={int(current_mid_A.shape[0])}"
        elif current_row_indices is not None:
            valid_previous = current_initial_previous_A.shape == (
                int(current_mid_A.shape[0]),
                drive_count,
            )
            expected = f"(U, S)=({int(current_mid_A.shape[0])}, {drive_count})"
        elif current_mid_A.ndim == 2:
            valid_previous = current_initial_previous_A.shape == (drive_count,)
            expected = f"(S,)=({drive_count},)"
        else:
            valid_previous = current_initial_previous_A.shape == (batch_size, drive_count)
            expected = f"(B, K)=({batch_size}, {drive_count})"
        if not valid_previous:
            raise ValueError(
                f"{name}.current_initial_previous_A must have shape {expected}, "
                f"got {current_initial_previous_A.shape}."
            )
    return FactorizedExtracellularPotentialBatch(
        current_mid_A=current_mid_A,
        footprint_mV_per_A=footprint_mV_per_A,
        target_nx=nx,
        current_initial_previous_A=current_initial_previous_A,
        single_cable_forcing_footprint_mV_per_A=forcing_footprint_mV_per_A,
        current_row_indices=current_row_indices,
        current_row_scales=current_row_scales,
    )

def _as_batched_time_space_array(
    name: str,
    values: Array,
    *,
    nt: int,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int | None = None,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 2:
        if arr.shape != (nt, nx):
            raise ValueError(
                f"{name} must have shape (Nt, Nx)=({nt}, {nx}) "
                f"or (B, Nt, Nx), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[1:] != (nt, nx):
            raise ValueError(
                f"{name} must have trailing shape (Nt, Nx)=({nt}, {nx}), "
                f"got {arr.shape}."
            )
    else:
        raise ValueError(
            f"{name} must have shape (Nt, Nx) or (B, Nt, Nx), got {arr.shape}."
        )

    if batch_size is None:
        return arr
    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, nt, nx))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")

def _as_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim != 1 or arr.shape != (nx,):
        raise ValueError(f"{name} must have shape (Nx,)=({nx},), got {arr.shape}.")
    return arr

def _as_edge_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> Array:
    edge_count = max(int(nx) - 1, 0)
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim != 1 or arr.shape != (edge_count,):
        raise ValueError(
            f"{name} must have shape (Nx-1,)=({edge_count},), got {arr.shape}."
        )
    return arr

def _as_scalar_or_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return arr
    return _as_space_array(name, arr, nx=nx, dtype_local=dtype_local)

def _as_batched_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int | None = None,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape != (nx,):
            raise ValueError(
                f"{name} must have shape (Nx,)=({nx},) or (B, Nx), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1:] != (nx,):
            raise ValueError(
                f"{name} must have trailing shape (Nx,)=({nx},), got {arr.shape}."
            )
    else:
        raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")

    if batch_size is None:
        return arr
    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, nx))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")

def _as_cached_batched_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape != (nx,):
            raise ValueError(
                f"{name} must have shape (Nx,)=({nx},) or (B, Nx), got {arr.shape}."
            )
        key = _batched_static_array_cache_key(
            "space",
            values,
            arr=arr,
            batch_size=batch_size,
        )
        cached = get_batched_static_array(key, sources=(values,))
        if cached is not None:
            return cached
        out = jnp.broadcast_to(arr[jnp.newaxis, :], (batch_size, nx))
        store_batched_static_array(key, out, sources=(values,))
        return out
    if arr.ndim == 2:
        if arr.shape[1:] != (nx,):
            raise ValueError(
                f"{name} must have trailing shape (Nx,)=({nx},), got {arr.shape}."
            )
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            key = _batched_static_array_cache_key(
                "space",
                values,
                arr=arr,
                batch_size=batch_size,
            )
            cached = get_batched_static_array(key, sources=(values,))
            if cached is not None:
                return cached
            out = jnp.broadcast_to(arr, (batch_size, nx))
            store_batched_static_array(key, out, sources=(values,))
            return out
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")
    raise ValueError(f"{name} must have shape (Nx,) or (B, Nx), got {arr.shape}.")

def _as_batched_edge_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    edge_count = max(int(nx) - 1, 0)
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 1:
        if arr.shape != (edge_count,):
            raise ValueError(
                f"{name} must have shape (Nx-1,)=({edge_count},) or "
                f"(B, Nx-1), got {arr.shape}."
            )
        arr = arr[jnp.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1:] != (edge_count,):
            raise ValueError(
                f"{name} must have trailing shape (Nx-1,)=({edge_count},), got {arr.shape}."
            )
    else:
        raise ValueError(f"{name} must have shape (Nx-1,) or (B, Nx-1), got {arr.shape}.")

    if arr.shape[0] == batch_size:
        return arr
    if arr.shape[0] == 1:
        return jnp.broadcast_to(arr, (batch_size, edge_count))
    raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {arr.shape[0]}.")

def _as_batched_scalar_or_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr[jnp.newaxis], (batch_size,))
    if arr.ndim == 1 and arr.shape == (batch_size,):
        return arr
    return _as_batched_space_array(
        name,
        arr,
        nx=nx,
        dtype_local=dtype_local,
        batch_size=batch_size,
    )

def _as_cached_batched_scalar_or_space_array(
    name: str,
    values: Array,
    *,
    nx: int,
    dtype_local: jnp.dtype,
    batch_size: int,
) -> Array:
    arr = jnp.asarray(values, dtype=dtype_local)
    if arr.ndim == 0:
        key = _batched_static_array_cache_key(
            "scalar",
            values,
            arr=arr,
            batch_size=batch_size,
        )
        cached = get_batched_static_array(key, sources=(values,))
        if cached is not None:
            return cached
        out = jnp.broadcast_to(arr[jnp.newaxis], (batch_size,))
        store_batched_static_array(key, out, sources=(values,))
        return out
    if arr.ndim == 1 and arr.shape == (batch_size,):
        return arr
    return _as_cached_batched_space_array(
        name,
        arr,
        nx=nx,
        dtype_local=dtype_local,
        batch_size=batch_size,
    )

def _batched_static_array_cache_key(
    kind: str,
    values: Array,
    *,
    arr: Array,
    batch_size: int,
) -> tuple[Any, ...]:
    return (
        "batched_static_array_v1",
        kind,
        id(values),
        tuple(int(dim) for dim in arr.shape),
        str(arr.dtype),
        int(batch_size),
        _current_jax_device_key(),
    )

def _current_jax_device_key() -> tuple[Any, ...]:
    device = getattr(jax.config, "jax_default_device", None)
    if device is None:
        try:
            devices = jax.devices(jax.default_backend())
        except Exception:
            devices = ()
        device = devices[0] if devices else None
    if device is None:
        return ("backend", jax.default_backend())
    return (
        "device",
        getattr(device, "platform", None),
        getattr(device, "id", None),
    )
