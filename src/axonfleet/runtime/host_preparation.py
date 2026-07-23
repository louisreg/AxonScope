"""Runtime-neutral host-array preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from axonfleet.preparation.axon_rows import MaterializedAxonRows


EXTRACELLULAR_SPACE_FIELDS = (
    "Cm_abs",
    "Cx_abs",
    "Gx_abs",
    "left_i",
    "right_i",
    "left_e",
    "right_e",
)
EXTRACELLULAR_EDGE_FIELDS = ("Gax_e", "Gax_i")


@dataclass(frozen=True)
class ExtracellularRuntimeArrays:
    """Padded double-cable extracellular arrays before runtime materialization."""

    Cm_abs: np.ndarray
    Cx_abs: np.ndarray
    Gx_abs: np.ndarray
    Gax_e: np.ndarray
    Gax_i: np.ndarray
    left_i: np.ndarray
    right_i: np.ndarray
    left_e: np.ndarray
    right_e: np.ndarray


@dataclass(frozen=True)
class CableRuntimeRows:
    """Template-major cable arrays before population gathering."""

    lower: np.ndarray
    diag: np.ndarray
    upper: np.ndarray
    area_cm2: np.ndarray


@dataclass(frozen=True)
class ExtracellularRuntimeRows:
    """Template-major double-cable arrays before population gathering."""

    Cm_abs: np.ndarray
    Cx_abs: np.ndarray
    Gx_abs: np.ndarray
    Gax_e: np.ndarray
    Gax_i: np.ndarray
    left_i: np.ndarray
    right_i: np.ndarray
    left_e: np.ndarray
    right_e: np.ndarray


def cable_runtime_rows_numpy(
    rows: MaterializedAxonRows,
    *,
    dtype: np.dtype,
    include_area: bool,
) -> CableRuntimeRows:
    """Lower all unique axon templates to cable arrays with NumPy."""

    lower, diag, upper = diffusion_operator_rows_numpy(rows, dtype=dtype)
    if include_area:
        area = compartment_area_rows_cm2_numpy(rows, dtype=dtype)
        area = _edge_pad_template_rows(area, rows.template_nx)
    else:
        area = np.zeros_like(lower)
    return CableRuntimeRows(
        lower=_readonly(lower),
        diag=_readonly(diag),
        upper=_readonly(upper),
        area_cm2=_readonly(area),
    )


def diffusion_operator_rows_numpy(
    rows: MaterializedAxonRows,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized diffusion coefficients for a materialized template table."""

    dtype = np.dtype(dtype)
    valid = np.asarray(rows.valid_mask, dtype=bool)
    template_nx = np.asarray(rows.template_nx, dtype=np.intp)
    template_count, nx = valid.shape
    edge_valid = valid[:, :-1] & valid[:, 1:]

    lengths_cm = np.asarray(rows.compartment_lengths_um, dtype=dtype) * dtype.type(1e-4)
    diam_um = np.asarray(rows.diam_um, dtype=dtype)
    ra_ohm_cm = np.asarray(rows.Ra_ohm_cm, dtype=dtype)
    cm_uF_cm2 = np.asarray(rows.Cm_uF_cm2, dtype=dtype)

    area_cm2 = np.pi * (diam_um * dtype.type(1e-4)) * lengths_cm
    radius_cm = dtype.type(0.5) * diam_um * dtype.type(1e-4)
    cross_section_cm2 = np.pi * radius_cm**2
    edge_resistance = (
        ra_ohm_cm[:, :-1]
        * (dtype.type(0.5) * lengths_cm[:, :-1])
        / np.maximum(cross_section_cm2[:, :-1], dtype.type(1e-30))
        + ra_ohm_cm[:, 1:]
        * (dtype.type(0.5) * lengths_cm[:, 1:])
        / np.maximum(cross_section_cm2[:, 1:], dtype.type(1e-30))
    )
    gax_i_mS = np.where(
        edge_valid,
        dtype.type(1e3) / np.maximum(edge_resistance, dtype.type(1e-18)),
        dtype.type(0),
    )
    cm_abs_uF = cm_uF_cm2 * area_cm2
    heterogeneous_lower = np.zeros((template_count, nx), dtype=dtype)
    heterogeneous_upper = np.zeros((template_count, nx), dtype=dtype)
    heterogeneous_lower[:, 1:] = np.where(
        edge_valid,
        gax_i_mS / np.maximum(cm_abs_uF[:, 1:], dtype.type(1e-30)),
        dtype.type(0),
    )
    heterogeneous_upper[:, :-1] = np.where(
        edge_valid,
        gax_i_mS / np.maximum(cm_abs_uF[:, :-1], dtype.type(1e-30)),
        dtype.type(0),
    )
    heterogeneous_diag = -(heterogeneous_lower + heterogeneous_upper)

    counts = np.maximum(template_nx, 1).astype(dtype)
    mean_diam = np.sum(np.where(valid, diam_um, 0), axis=1) / counts
    mean_ra = np.sum(np.where(valid, ra_ohm_cm, 0), axis=1) / counts
    mean_cm = np.sum(np.where(valid, cm_uF_cm2, 0), axis=1) / counts
    uniform_radius_cm = dtype.type(0.5) * mean_diam * dtype.type(1e-4)
    capacitance = (
        dtype.type(2.0)
        * np.pi
        * uniform_radius_cm
        * mean_cm
        * dtype.type(1e-6)
    )
    axial_resistance = mean_ra / (np.pi * uniform_radius_cm**2)
    diffusion = dtype.type(1.0) / (axial_resistance * capacitance) / dtype.type(1000.0)

    uniform_lower = np.zeros((template_count, nx), dtype=dtype)
    uniform_diag = np.zeros((template_count, nx), dtype=dtype)
    uniform_upper = np.zeros((template_count, nx), dtype=dtype)
    h = np.asarray(rows.h_cm, dtype=dtype)
    row_ids = np.arange(template_count)
    last = template_nx - 1
    last_edge = template_nx - 2
    first_h = h[:, 0]
    last_h = h[row_ids, last_edge]
    left_coef = dtype.type(2.0) * diffusion / (first_h**2)
    right_coef = dtype.type(2.0) * diffusion / (last_h**2)
    if nx > 2:
        h_left = h[:, :-1]
        h_right = h[:, 1:]
        interior_valid = valid[:, 1:-1] & valid[:, 2:]
        safe_h_left = np.where(interior_valid, h_left, dtype.type(1))
        safe_h_right = np.where(interior_valid, h_right, dtype.type(1))
        denom = safe_h_left + safe_h_right
        uniform_lower[:, 1:-1] = np.where(
            interior_valid,
            dtype.type(2.0) * diffusion[:, None] / (safe_h_left * denom),
            dtype.type(0),
        )
        uniform_diag[:, 1:-1] = np.where(
            interior_valid,
            -dtype.type(2.0)
            * diffusion[:, None]
            / (safe_h_left * safe_h_right),
            dtype.type(0),
        )
        uniform_upper[:, 1:-1] = np.where(
            interior_valid,
            dtype.type(2.0) * diffusion[:, None] / (safe_h_right * denom),
            dtype.type(0),
        )
    uniform_diag[:, 0] = -left_coef
    uniform_upper[:, 0] = left_coef
    uniform_lower[row_ids, last] = right_coef
    uniform_diag[row_ids, last] = -right_coef

    heterogeneous = np.asarray(
        rows.has_heterogeneous_cable_properties,
        dtype=bool,
    )[:, None]
    lower = np.where(heterogeneous, heterogeneous_lower, uniform_lower)
    diag = np.where(heterogeneous, heterogeneous_diag, uniform_diag)
    upper = np.where(heterogeneous, heterogeneous_upper, uniform_upper)
    return _readonly(lower), _readonly(diag), _readonly(upper)


def compartment_area_rows_cm2_numpy(
    rows: MaterializedAxonRows,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Return template-major membrane area arrays in cm2."""

    dtype = np.dtype(dtype)
    diam = np.asarray(rows.diam_um, dtype=dtype)
    length_cm = np.asarray(rows.compartment_lengths_um, dtype=dtype) * dtype.type(1e-4)
    area = np.pi * (diam * dtype.type(1e-4)) * length_cm
    return _readonly(np.where(rows.valid_mask, area, dtype.type(0)))


def extracellular_runtime_rows_numpy(
    rows: MaterializedAxonRows,
    *,
    dtype: np.dtype,
) -> ExtracellularRuntimeRows:
    """Lower all unique templates to padded double-cable arrays with NumPy."""

    dtype = np.dtype(dtype)
    valid = np.asarray(rows.valid_mask, dtype=bool)
    edge_valid = valid[:, :-1] & valid[:, 1:]
    area = compartment_area_rows_cm2_numpy(rows, dtype=dtype)
    cm_abs = np.asarray(rows.Cm_uF_cm2, dtype=dtype) * area
    cx_abs = np.asarray(rows.xc_uF_cm2, dtype=dtype) * area
    gx_abs = np.asarray(rows.xg_S_cm2, dtype=dtype) * dtype.type(1e3) * area

    xraxial = np.asarray(rows.xraxial_MOhm_per_cm, dtype=dtype)
    dx_cm = np.asarray(rows.dx_cm, dtype=dtype)
    edge_resistance = (
        xraxial[:, :-1] * dtype.type(0.5) * dx_cm[:, :-1]
        + xraxial[:, 1:] * dtype.type(0.5) * dx_cm[:, 1:]
    )
    gax_e = np.where(
        edge_valid,
        dtype.type(1e-3) / np.maximum(edge_resistance, dtype.type(1e-18)),
        dtype.type(0),
    )
    lower, _, upper = diffusion_operator_rows_numpy(rows, dtype=dtype)
    gax_i = np.where(
        edge_valid,
        dtype.type(0.5)
        * (upper[:, :-1] * cm_abs[:, :-1] + lower[:, 1:] * cm_abs[:, 1:]),
        dtype.type(0),
    )
    zero = np.zeros((rows.template_count, 1), dtype=dtype)
    left_i = np.concatenate((zero, gax_i), axis=1)
    right_i = np.concatenate((gax_i, zero), axis=1)
    left_e = np.concatenate((zero, gax_e), axis=1)
    right_e = np.concatenate((gax_e, zero), axis=1)
    return ExtracellularRuntimeRows(
        Cm_abs=_readonly(_edge_pad_template_rows(cm_abs, rows.template_nx)),
        Cx_abs=_readonly(_edge_pad_template_rows(cx_abs, rows.template_nx)),
        Gx_abs=_readonly(_edge_pad_template_rows(gx_abs, rows.template_nx)),
        Gax_e=_readonly(gax_e),
        Gax_i=_readonly(gax_i),
        left_i=_readonly(left_i),
        right_i=_readonly(right_i),
        left_e=_readonly(left_e),
        right_e=_readonly(right_e),
    )


def _edge_pad_template_rows(values: np.ndarray, template_nx: np.ndarray) -> np.ndarray:
    width = int(values.shape[1])
    indices = np.minimum(
        np.arange(width, dtype=np.intp)[None, :],
        np.asarray(template_nx, dtype=np.intp)[:, None] - 1,
    )
    return np.take_along_axis(values, indices, axis=1)


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    array.setflags(write=False)
    return array


def diffusion_operator_coeffs_numpy(
    axon: Any,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build diffusion-operator coefficients during host preparation."""

    nx = int(axon.n_compartments)
    lower = np.zeros((nx,), dtype=dtype)
    diag = np.zeros((nx,), dtype=dtype)
    upper = np.zeros((nx,), dtype=dtype)

    if bool(getattr(axon, "has_heterogeneous_cable_properties", False)):
        lengths_cm = np.asarray(axon.compartment_lengths_um, dtype=dtype) * dtype.type(1e-4)
        diam_um = np.asarray(axon.diam_um, dtype=dtype)
        ra_ohm_cm = np.asarray(axon.Ra_ohm_cm, dtype=dtype)
        cm_uF_cm2 = np.asarray(axon.Cm_uF_cm2, dtype=dtype)

        area_cm2 = np.pi * (diam_um * dtype.type(1e-4)) * lengths_cm
        radius_cm = dtype.type(0.5) * diam_um * dtype.type(1e-4)
        cross_section_cm2 = np.pi * radius_cm**2
        left_half_cm = dtype.type(0.5) * lengths_cm[:-1]
        right_half_cm = dtype.type(0.5) * lengths_cm[1:]
        edge_resistance_ohm = (
            ra_ohm_cm[:-1] * left_half_cm / cross_section_cm2[:-1]
            + ra_ohm_cm[1:] * right_half_cm / cross_section_cm2[1:]
        )
        gax_i_mS = dtype.type(1e3) / np.maximum(edge_resistance_ohm, dtype.type(1e-18))
        cm_abs_uF = cm_uF_cm2 * area_cm2
        lower[1:] = gax_i_mS / cm_abs_uF[1:]
        upper[:-1] = gax_i_mS / cm_abs_uF[:-1]
        diag = -(lower + upper)
        return lower, diag.astype(dtype, copy=False), upper

    h = np.asarray(axon.h_cm, dtype=dtype)
    diffusion = uniform_diffusion_coefficient_numpy(axon, dtype=dtype)
    if nx >= 2:
        left_coef = dtype.type(2.0) * diffusion / (h[0] ** 2)
        right_coef = dtype.type(2.0) * diffusion / (h[-1] ** 2)
        diag[0] = -left_coef
        upper[0] = left_coef
        lower[-1] = right_coef
        diag[-1] = -right_coef
    if nx > 2:
        h_left = h[:-1]
        h_right = h[1:]
        denom = h_left + h_right
        lower[1:-1] = dtype.type(2.0) * diffusion / (h_left * denom)
        diag[1:-1] = -dtype.type(2.0) * diffusion / (h_left * h_right)
        upper[1:-1] = dtype.type(2.0) * diffusion / (h_right * denom)
    return lower, diag, upper


def uniform_diffusion_coefficient_numpy(axon: Any, *, dtype: np.dtype) -> np.generic:
    """Return the uniform-cable diffusion coefficient for one host axon row."""

    diam_um = np.mean(np.asarray(axon.diam_um, dtype=dtype))
    ra_ohm_cm = np.mean(np.asarray(axon.Ra_ohm_cm, dtype=dtype))
    cm_uF_cm2 = np.mean(np.asarray(axon.Cm_uF_cm2, dtype=dtype))
    radius_cm = dtype.type(0.5) * diam_um * dtype.type(1e-4)
    cm = dtype.type(2.0) * np.pi * radius_cm * cm_uF_cm2 * dtype.type(1e-6)
    ra = ra_ohm_cm / (np.pi * radius_cm**2)
    return dtype.type(1.0) / (ra * cm) / dtype.type(1000.0)


def compartment_area_cm2_numpy(axon: Any, *, dtype: np.dtype) -> np.ndarray:
    """Return per-compartment membrane area in cm2 for one host axon row."""

    diam = np.asarray(axon.diam_um, dtype=dtype)
    length_cm = np.asarray(axon.compartment_lengths_um, dtype=dtype) * dtype.type(1e-4)
    return np.asarray(np.pi * (diam * dtype.type(1e-4)) * length_cm, dtype=dtype)


def extracellular_runtime_numpy(
    axon: Any,
    *,
    dtype: np.dtype,
    target_nx: int,
) -> ExtracellularRuntimeArrays:
    """Build one padded double-cable extracellular row with NumPy arrays."""

    area = compartment_area_cm2_numpy(axon, dtype=dtype)
    cm_uF_cm2 = np.asarray(axon.Cm_uF_cm2, dtype=dtype)
    Cm_abs = cm_uF_cm2 * area

    xg = np.asarray(axon.xg_S_cm2, dtype=dtype)
    xc = np.asarray(axon.xc_uF_cm2, dtype=dtype)
    xraxial = np.asarray(axon.xraxial_MOhm_per_cm, dtype=dtype)
    dx_cm = np.asarray(axon.dx_cm, dtype=dtype)

    Cx_abs = xc * area
    Gx_abs = (xg * dtype.type(1e3)) * area

    if int(axon.n_compartments) <= 1:
        Gax_e = np.zeros((0,), dtype=dtype)
    else:
        r_edge_mohm = (
            xraxial[:-1] * (dtype.type(0.5) * dx_cm[:-1])
            + xraxial[1:] * (dtype.type(0.5) * dx_cm[1:])
        )
        Gax_e = dtype.type(1e-3) / np.maximum(r_edge_mohm, dtype.type(1e-18))

    lower, _, upper = diffusion_operator_coeffs_numpy(axon, dtype=dtype)
    Gax_i = dtype.type(0.5) * (upper[:-1] * Cm_abs[:-1] + lower[1:] * Cm_abs[1:])
    left_i = np.concatenate([np.zeros((1,), dtype=dtype), Gax_i])
    right_i = np.concatenate([Gax_i, np.zeros((1,), dtype=dtype)])
    left_e = np.concatenate([np.zeros((1,), dtype=dtype), Gax_e])
    right_e = np.concatenate([Gax_e, np.zeros((1,), dtype=dtype)])

    return ExtracellularRuntimeArrays(
        Cm_abs=pad_space_array_numpy(Cm_abs, target_nx=target_nx, mode="edge"),
        Cx_abs=pad_space_array_numpy(Cx_abs, target_nx=target_nx, mode="edge"),
        Gx_abs=pad_space_array_numpy(Gx_abs, target_nx=target_nx, mode="edge"),
        Gax_e=pad_edge_array_numpy(Gax_e, target_nx=target_nx),
        Gax_i=pad_edge_array_numpy(Gax_i, target_nx=target_nx),
        left_i=pad_space_array_numpy(left_i, target_nx=target_nx, mode="zero"),
        right_i=pad_space_array_numpy(right_i, target_nx=target_nx, mode="zero"),
        left_e=pad_space_array_numpy(left_e, target_nx=target_nx, mode="zero"),
        right_e=pad_space_array_numpy(right_e, target_nx=target_nx, mode="zero"),
    )


def pad_space_array_numpy(
    values: np.ndarray,
    *,
    target_nx: int,
    mode: str,
) -> np.ndarray:
    """Pad one host compartment-space array to ``target_nx``."""

    arr = np.asarray(values)
    pad_count = int(target_nx) - int(arr.shape[0])
    if pad_count < 0:
        raise ValueError(
            f"target_nx must be >= array width, got target_nx={target_nx}, "
            f"width={arr.shape[0]}."
        )
    if pad_count == 0:
        return arr
    if mode == "zero":
        pad_values = np.zeros((pad_count,), dtype=arr.dtype)
    elif mode == "edge":
        pad_values = np.broadcast_to(arr[-1], (pad_count,)).astype(arr.dtype, copy=False)
    else:
        raise ValueError(f"unknown padding mode: {mode!r}.")
    return np.concatenate([arr, pad_values], axis=0)


def pad_edge_array_numpy(values: np.ndarray, *, target_nx: int) -> np.ndarray:
    """Pad one host edge-space array with zero coupling into padded compartments."""

    arr = np.asarray(values)
    target_edges = max(int(target_nx) - 1, 0)
    pad_count = target_edges - int(arr.shape[0])
    if pad_count < 0:
        raise ValueError(
            f"target_nx={target_nx} is too small for edge width={arr.shape[0]}."
        )
    if pad_count == 0:
        return arr
    return np.concatenate([arr, np.zeros((pad_count,), dtype=arr.dtype)], axis=0)


def pad_gate_array_numpy(
    values: np.ndarray,
    *,
    target_nx: int,
    target_gates: int,
) -> np.ndarray:
    """Pad one host gate matrix to shared spatial and gate widths."""

    arr = np.asarray(values)
    pad_nx = int(target_nx) - int(arr.shape[0])
    pad_gates = int(target_gates) - int(arr.shape[1])
    if pad_nx < 0 or pad_gates < 0:
        raise ValueError(
            "target_nx/target_gates must be >= gate array shape, got "
            f"targets=({target_nx}, {target_gates}) and shape={arr.shape}."
        )
    if pad_gates:
        arr = np.concatenate(
            [arr, np.zeros((arr.shape[0], pad_gates), dtype=arr.dtype)],
            axis=1,
        )
    if pad_nx:
        arr = np.concatenate(
            [arr, np.zeros((pad_nx, arr.shape[1]), dtype=arr.dtype)],
            axis=0,
        )
    return arr


__all__ = [
    "CableRuntimeRows",
    "EXTRACELLULAR_EDGE_FIELDS",
    "EXTRACELLULAR_SPACE_FIELDS",
    "ExtracellularRuntimeArrays",
    "ExtracellularRuntimeRows",
    "cable_runtime_rows_numpy",
    "compartment_area_cm2_numpy",
    "compartment_area_rows_cm2_numpy",
    "diffusion_operator_rows_numpy",
    "diffusion_operator_coeffs_numpy",
    "extracellular_runtime_numpy",
    "extracellular_runtime_rows_numpy",
    "pad_edge_array_numpy",
    "pad_gate_array_numpy",
    "pad_space_array_numpy",
    "uniform_diffusion_coefficient_numpy",
]
