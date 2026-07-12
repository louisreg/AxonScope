from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

import axonscope as axs
from axonscope.axons.flattened import flatten_layout


AXONSCOPE_TO_NRV_CURRENT_SCALE = 1e-3
AXONSCOPE_TO_NRV_CONDUCTANCE_SCALE = 1e-3


def run_axonscope_simulation(
    axon,
    *,
    tsim: float,
    dt: float,
    record_observables: bool = False,
):
    """Run one AxonScope axon/instance through the public simulation path."""

    recording = None
    if record_observables:
        recording = axs.Recording(
            voltage=True,
            gates=True,
            currents=True,
            conductances=True,
        )
    return axs.AxonSimulation(
        axon,
        duration=tsim,
        dt=dt,
        recording=recording,
    ).run().single


def axonscope_x_um(axon) -> np.ndarray:
    """Return AxonScope compartment-center positions from the descriptive layout."""

    return np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float)


def axonscope_compartment_lengths_um(axon) -> np.ndarray:
    """Return AxonScope compartment control-volume lengths from the layout."""

    return np.asarray(axon.layout.compartment_length_values(unit="micrometer"), dtype=float)


def axonscope_section_names(axon) -> np.ndarray:
    """Return one section label per AxonScope compartment."""

    return np.asarray(flatten_layout(axon.layout).section_names, dtype=object)


def enable_nrv_recordings(axon_nrv) -> None:
    axon_nrv.record_V_mem = True
    axon_nrv.record_I_ions = True
    axon_nrv.record_particles = True
    axon_nrv.record_g_ions = True
    axon_nrv.record_g_mem = True
    if hasattr(axon_nrv, "record_particules"):
        axon_nrv.record_particules = True


def normalize_nrv_matrix(values: np.ndarray, t_ms: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D NRV array, got shape {arr.shape}.")
    if arr.shape == (x_um.size, t_ms.size):
        return arr
    if arr.shape == (t_ms.size, x_um.size):
        return arr.T
    if arr.shape[0] == x_um.size:
        return arr
    if arr.shape[1] == x_um.size:
        return arr.T
    raise ValueError(
        f"Could not align NRV array of shape {arr.shape} with x={x_um.size} and t={t_ms.size}."
    )


def interp_rows(values_by_space_time: np.ndarray, t_src_ms: np.ndarray, t_dst_ms: np.ndarray) -> np.ndarray:
    out = np.empty((values_by_space_time.shape[0], t_dst_ms.size), dtype=float)
    for i in range(values_by_space_time.shape[0]):
        out[i] = np.interp(t_dst_ms, t_src_ms, values_by_space_time[i])
    return out


def shifted_interp(
    t_dst_ms: np.ndarray,
    t_src_ms: np.ndarray,
    values: np.ndarray,
    shift_steps: int = 0,
) -> np.ndarray:
    t_src = np.asarray(t_src_ms, dtype=float).ravel()
    vals = np.asarray(values, dtype=float).ravel()
    if shift_steps == 0:
        return np.interp(t_dst_ms, t_src, vals)
    if shift_steps > 0:
        return np.interp(t_dst_ms, t_src[shift_steps:], vals[:-shift_steps])
    shift_steps = -shift_steps
    return np.interp(t_dst_ms, t_src[:-shift_steps], vals[shift_steps:])


def nrv_trace(
    results_nrv,
    key: str,
    row_index: int,
    t_dst_ms: np.ndarray,
    *,
    shift_steps: int = 0,
) -> np.ndarray:
    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_nrv = np.asarray(results_nrv["x_rec"], dtype=float)
    matrix = normalize_nrv_matrix(results_nrv[key], t_nrv, x_nrv)
    return shifted_interp(t_dst_ms, t_nrv, matrix[row_index], shift_steps=shift_steps)


def trace_metrics(ref: np.ndarray, test: np.ndarray) -> tuple[float, float, float]:
    diff = np.asarray(test, dtype=float) - np.asarray(ref, dtype=float)
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    q99_abs = float(np.quantile(np.abs(diff), 0.99))
    return rmse, max_abs, q99_abs


def sample_indices_from_position(
    as_x_um: np.ndarray,
    nrv_x_um: np.ndarray,
    sample_position_um: float | None,
) -> tuple[int, int]:
    if sample_position_um is None:
        return int(len(as_x_um) // 2), int(len(nrv_x_um) // 2)
    return (
        int(np.argmin(np.abs(np.asarray(as_x_um, dtype=float) - float(sample_position_um)))),
        int(np.argmin(np.abs(np.asarray(nrv_x_um, dtype=float) - float(sample_position_um)))),
    )


def select_nearest_rows(
    x_source_um: np.ndarray,
    matrix_source: np.ndarray,
    x_target_um: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    x_source = np.asarray(x_source_um, dtype=float).ravel()
    matrix = np.asarray(matrix_source, dtype=float)
    idx = [int(np.argmin(np.abs(x_source - float(xi)))) for xi in x_target_um]
    idx = np.asarray(idx, dtype=int)
    return x_source[idx], matrix[idx]


def align_rows_to_target_x(
    x_source_um: np.ndarray,
    matrix_source: np.ndarray,
    x_target_um: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_source = np.asarray(x_source_um, dtype=float).ravel()
    matrix = np.asarray(matrix_source, dtype=float)
    x_target = np.asarray(list(x_target_um), dtype=float).ravel()
    idx = np.asarray([int(np.argmin(np.abs(x_source - xi))) for xi in x_target], dtype=int)
    return x_target, matrix[idx], idx


def velocity_from_peak_times(
    x_um: np.ndarray,
    vm_space_time: np.ndarray,
    t_ms: np.ndarray,
    *,
    center_x_um: float,
    threshold_mV: float = 0.0,
    exclude_radius_um: float = 0.0,
) -> float:
    x = np.asarray(x_um, dtype=float).ravel()
    vm = np.asarray(vm_space_time, dtype=float)
    t = np.asarray(t_ms, dtype=float).ravel()

    if vm.ndim != 2:
        raise ValueError(f"Expected Vm matrix of shape (Nx, Nt), got {vm.shape}.")

    peak_idx = np.argmax(vm, axis=1)
    peak_t_ms = t[peak_idx]
    peak_v_mV = vm[np.arange(vm.shape[0]), peak_idx]

    active = peak_v_mV > float(threshold_mV)
    left_mask = active & (x < center_x_um - exclude_radius_um)
    right_mask = active & (x > center_x_um + exclude_radius_um)

    velocities = []
    for mask in (left_mask, right_mask):
        if int(np.sum(mask)) < 2:
            continue
        xs_m = x[mask] * 1e-6
        ts_s = peak_t_ms[mask] * 1e-3
        order = np.argsort(ts_s)
        xs_m = xs_m[order]
        ts_s = ts_s[order]
        coeff = np.polyfit(ts_s, xs_m, 1)
        velocities.append(abs(float(coeff[0])))

    if not velocities:
        return 0.0
    return float(np.mean(velocities))


def first_cross_time(trace_mV: np.ndarray, t_ms: np.ndarray, threshold_mV: float) -> float:
    trace = np.asarray(trace_mV, dtype=float)
    time = np.asarray(t_ms, dtype=float)
    above = trace >= threshold_mV
    idx = np.where(above[1:] & ~above[:-1])[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    t0, t1 = float(time[i]), float(time[i + 1])
    v0, v1 = float(trace[i]), float(trace[i + 1])
    if v1 == v0:
        return t1
    return t0 + (threshold_mV - v0) * (t1 - t0) / (v1 - v0)


def crossing_times(vm_space_time: np.ndarray, t_ms: np.ndarray, threshold_mV: float) -> np.ndarray:
    vm = np.asarray(vm_space_time, dtype=float)
    return np.asarray([first_cross_time(vm[i], t_ms, threshold_mV) for i in range(vm.shape[0])], dtype=float)


def velocity_from_crossing_times(
    x_um: np.ndarray,
    vm_space_time: np.ndarray,
    t_ms: np.ndarray,
    *,
    center_x_um: float,
    threshold_mV: float,
    exclude_radius_um: float = 0.0,
    fit_mode: Literal["direct", "symmetric"] = "direct",
) -> float:
    x = np.asarray(x_um, dtype=float).ravel()
    vm = np.asarray(vm_space_time, dtype=float)
    t = np.asarray(t_ms, dtype=float).ravel()
    tc_ms = crossing_times(vm, t, threshold_mV)

    active = np.isfinite(tc_ms)
    if fit_mode == "symmetric":
        center_idx = int(np.argmin(np.abs(x - float(center_x_um))))
        center_t_ms = float(tc_ms[center_idx])
        if not np.isfinite(center_t_ms):
            return 0.0
        dist_um = np.abs(x - float(center_x_um))
        delay_ms = tc_ms - center_t_ms
        mask = active & (dist_um > float(exclude_radius_um))
        if np.count_nonzero(mask) < 2:
            return 0.0
        dist = dist_um[mask]
        delay = delay_ms[mask]
        dist_round = np.round(dist, 6)
        uniq = np.unique(dist_round)
        if uniq.size < 2:
            return 0.0
        dist_u = np.asarray(uniq, dtype=float)
        delay_u = np.asarray([delay[dist_round == u].mean() for u in uniq], dtype=float)
        coeff = np.polyfit(delay_u * 1e-3, dist_u * 1e-6, 1)
        return abs(float(coeff[0]))

    left_mask = active & (x < float(center_x_um) - float(exclude_radius_um))
    right_mask = active & (x > float(center_x_um) + float(exclude_radius_um))
    velocities = []
    for mask in (left_mask, right_mask):
        if int(np.sum(mask)) < 2:
            continue
        xs_m = x[mask] * 1e-6
        ts_s = tc_ms[mask] * 1e-3
        order = np.argsort(ts_s)
        xs_m = xs_m[order]
        ts_s = ts_s[order]
        coeff = np.polyfit(ts_s, xs_m, 1)
        velocities.append(abs(float(coeff[0])))

    if not velocities:
        return 0.0
    return float(np.mean(velocities))
