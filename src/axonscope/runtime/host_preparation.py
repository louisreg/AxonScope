"""Runtime-neutral host-array preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


def diffusion_operator_coeffs_numpy(
    axon: Any,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy equivalent of ``diffusion_operator_coeffs`` for host preparation."""

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
    "EXTRACELLULAR_EDGE_FIELDS",
    "EXTRACELLULAR_SPACE_FIELDS",
    "ExtracellularRuntimeArrays",
    "compartment_area_cm2_numpy",
    "diffusion_operator_coeffs_numpy",
    "extracellular_runtime_numpy",
    "pad_edge_array_numpy",
    "pad_gate_array_numpy",
    "pad_space_array_numpy",
    "uniform_diffusion_coefficient_numpy",
]
