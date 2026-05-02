from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrv

from axonscope.axons.myelinated import MRG
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.stimulus import Stimulus
from tests.nrv._helpers import interp_rows, normalize_nrv_matrix, select_nearest_rows


FIG_DIR = Path("figures/nrv_tests/velocity_vs_diameter")


def _first_cross_time(trace_mV: np.ndarray, t_ms: np.ndarray, threshold_mV: float) -> float:
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


def _crossing_times(vm_nodes: np.ndarray, t_ms: np.ndarray, threshold_mV: float) -> np.ndarray:
    vm = np.asarray(vm_nodes, dtype=float)
    return np.asarray(
        [_first_cross_time(vm[i], t_ms, threshold_mV) for i in range(vm.shape[0])],
        dtype=float,
    )


def _symmetric_delay_curve(
    x_nodes_um: np.ndarray,
    crossing_times_ms: np.ndarray,
    center_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_nodes_um, dtype=float)
    tc = np.asarray(crossing_times_ms, dtype=float)
    center_t = float(tc[center_index])
    dist = np.abs(x - x[center_index])
    delay = tc - center_t
    mask = np.isfinite(delay) & (dist > 0.0)
    d = dist[mask]
    y = delay[mask]
    d_round = np.round(d, 6)
    uniq = np.unique(d_round)
    d_u = np.asarray(uniq, dtype=float)
    y_u = np.asarray([y[d_round == u].mean() for u in uniq], dtype=float)
    return d_u, y_u


@pytest.mark.parametrize(
    ("diameter_um", "threshold_mV", "rmse_tol_ms"),
    [
        (5.7, -10.0, 0.025),
        (5.7, 0.0, 0.025),
        (10.0, -10.0, 0.045),
        (10.0, 0.0, 0.045),
        (14.0, -10.0, 0.015),
        (14.0, 0.0, 0.015),
    ],
)
def test_mrg_node_delay_vs_nrv(diameter_um: float, threshold_mV: float, rmse_tol_ms: float) -> None:
    axon = MRG(d=diameter_um, nodes=11)
    node_ids = np.asarray(axon.node_indices, dtype=int)
    center_index = int(node_ids.shape[0] // 2)
    stim_pos_um = float(np.asarray(axon.x, dtype=float)[int(node_ids[center_index])])
    axon.insert_I_Clamp(position=stim_pos_um, stimulus=Stimulus.pulse(start=1.0, duration=0.1, amplitude=2.0))

    res = CrankNicholson().solve(axon, tsim=4.0, dt=0.005)
    t_as = np.asarray(res.t, dtype=float)
    x_as = np.asarray(axon.x, dtype=float)[node_ids]
    vm_as = np.asarray(res.Vm, dtype=float)[:, node_ids].T

    axon_nrv = nrv.myelinated(
        0,
        0,
        diameter_um,
        float(axon.L),
        model="MRG",
        dt=0.005,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    axon_nrv.insert_I_Clamp_node(index=center_index, t_start=1.0, duration=0.1, amplitude=2.0)
    results_nrv = axon_nrv.simulate(t_sim=4.0)

    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_rec = np.asarray(results_nrv["x_rec"], dtype=float)
    vm_nrv = normalize_nrv_matrix(results_nrv["V_mem"], t_nrv, x_rec)
    x_nodes_nrv = np.asarray(results_nrv["x_nodes"], dtype=float).ravel()
    x_nrv, vm_nrv_nodes = select_nearest_rows(x_rec, vm_nrv, x_nodes_nrv)
    vm_nrv_i = interp_rows(vm_nrv_nodes, t_nrv, t_as)

    tc_as = _crossing_times(vm_as, t_as, threshold_mV)
    tc_nrv = _crossing_times(vm_nrv_i, t_as, threshold_mV)
    d_as, y_as = _symmetric_delay_curve(x_as, tc_as, center_index)
    d_nrv, y_nrv = _symmetric_delay_curve(x_nrv, tc_nrv, center_index)

    n = min(d_as.size, d_nrv.size)
    rmse_ms = float(np.sqrt(np.mean((y_as[:n] - y_nrv[:n]) ** 2)))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axs[0].plot(x_as, tc_as, "o-", lw=1.8, label="AxonScope")
    axs[0].plot(x_nrv, tc_nrv, "s--", lw=2.0, label="NRV")
    axs[0].set_title(f"MRG d={diameter_um:.1f} um crossing times @ {threshold_mV:+.0f} mV")
    axs[0].set_xlabel("Node position [um]")
    axs[0].set_ylabel("Crossing time [ms]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(d_as, y_as, "o-", lw=1.8, label="AxonScope")
    axs[1].plot(d_nrv, y_nrv, "s--", lw=2.0, label="NRV")
    axs[1].set_title(f"Symmetric node-delay curve | RMSE={rmse_ms:.4f} ms")
    axs[1].set_xlabel("Distance from center node [um]")
    axs[1].set_ylabel("Delay from center node [ms]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig_path = FIG_DIR / f"mrg_node_delay_d{diameter_um:.1f}_thr{threshold_mV:+.0f}.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    assert rmse_ms < rmse_tol_ms, (
        f"MRG node-delay mismatch too large for d={diameter_um:.1f} um, "
        f"thr={threshold_mV:+.1f} mV: {rmse_ms:.4f} ms >= {rmse_tol_ms:.4f} ms "
        f"(plot: {fig_path})"
    )
