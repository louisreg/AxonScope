from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
from scipy.signal import find_peaks
import nrv

from axonscope import AxonInstance, S_per_m, ms, um
from axonscope.axons.myelinated import MRG
from axonscope.stimulation import AnalyticalExtracellularContext, PointSourceElectrode
from axonscope.stimulation import Stimulus
from axonscope.solvers.crank_nicholson import CrankNicholson
from tests.nrv._helpers import axonscope_x_um, normalize_nrv_matrix

pytestmark = pytest.mark.nrv_extracellular


def _interp_rows(values_by_row: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.empty((values_by_row.shape[0], t_dst.size), dtype=float)
    for i in range(values_by_row.shape[0]):
        out[i] = np.interp(t_dst, t_src, values_by_row[i])
    return out


def test_myelinated_extracellular_ctx_api_vs_nrv(save_dir: str = "figures/physics_tests"):
    # Protocol tuned to robustly activate the MRG fiber in extracellular stimulation.
    diameter_um = 10.0
    nodes = 9
    dt = 0.005
    tsim = 4.0
    stim_start_ms = 1.0
    cathodic_uA = 80.0
    cathodic_duration_ms = 0.08
    anodic_uA = 20.0
    interphase_ms = 0.04
    elec_y_um = 100.0
    elec_z_um = 0.0
    sigma_S_m = 0.2  # endoneurium_bhadra

    # --- AxonScope ---
    ax_as = MRG(diameter=diameter_um * um, nodes=nodes)
    sim_as = AxonInstance(ax_as)
    x0_um = float(ax_as.length / 2.0)
    electrode_as = PointSourceElectrode(
        x=x0_um * um,
        y=elec_y_um * um,
        z=elec_z_um * um,
    )
    stim_as = Stimulus.biphasic(
        start=stim_start_ms * ms,
        cathodic_amplitude=cathodic_uA * 1e-6,
        cathodic_duration=cathodic_duration_ms * ms,
        anodic_amplitude=anodic_uA * 1e-6,
        interphase=interphase_ms * ms,
    )
    sim_as.add_extracellular_context(
        context=AnalyticalExtracellularContext(
            electrodes=[electrode_as.with_stimulus(stim_as)],
            sigma=sigma_S_m * S_per_m,
        ),
        replace=True,
    )

    res_as = CrankNicholson().solve(sim_as, tsim=tsim, dt=dt)
    t_as = np.asarray(res_as.t)
    x_all_as = axonscope_x_um(ax_as)
    vm_all_as = np.asarray(res_as.Vm, dtype=float).T
    vext_all_as = np.stack(
        [np.asarray(sim_as.extracellular_potential_mV(float(t))) for t in t_as],
        axis=0,
    ).T
    node_idx_as = np.asarray(ax_as.node_indices, dtype=int)
    x_nodes_as = x_all_as[node_idx_as]

    # --- NRV reference ---
    ax_nrv = nrv.myelinated(
        0,
        0,
        diameter_um,
        float(ax_as.length),
        model="MRG",
        dt=dt,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    elec_nrv = nrv.point_source_electrode(x0_um, elec_y_um, elec_z_um)
    stim_nrv = nrv.stimulus()
    stim_nrv.biphasic_pulse(
        stim_start_ms,
        cathodic_uA,
        cathodic_duration_ms,
        anodic_uA,
        interphase_ms,
    )
    extra_nrv = nrv.stimulation("endoneurium_bhadra")
    extra_nrv.add_electrode(elec_nrv, stim_nrv)
    ax_nrv.attach_extracellular_stimulation(extra_nrv)

    out = ax_nrv.simulate(t_sim=tsim)
    t_nrv = np.asarray(out["t"]).ravel()
    x_all_nrv = np.asarray(out["x_rec"], dtype=float)
    vm_all_nrv = normalize_nrv_matrix(np.asarray(out["V_mem"]), t_nrv, x_all_nrv)
    idx_nrv = np.asarray([int(np.argmin(np.abs(x_all_nrv - xi))) for xi in x_all_as], dtype=int)
    x_all_nrv = x_all_nrv[idx_nrv]
    vm_all_nrv = vm_all_nrv[idx_nrv]

    # --- Align arrays ---
    n = min(vm_all_as.shape[0], vm_all_nrv.shape[0], vext_all_as.shape[0], x_all_as.size, x_all_nrv.size)
    vm_all_as = vm_all_as[:n]
    vm_all_nrv = vm_all_nrv[:n]
    vext_all_as = vext_all_as[:n]
    x_all_as = x_all_as[:n]
    x_all_nrv = x_all_nrv[:n]

    vm_all_nrv_i = _interp_rows(vm_all_nrv, t_nrv, t_as)
    err = vm_all_as - vm_all_nrv_i

    # --- Metrics ---
    rmse_global = float(np.sqrt(np.mean(err**2)))
    center_node = len(x_nodes_as) // 2
    sample_pos_um = float(x_nodes_as[center_node])
    sample_as_idx = int(np.argmin(np.abs(x_all_as - sample_pos_um)))
    sample_nrv_idx = int(np.argmin(np.abs(x_all_nrv - sample_pos_um)))
    rmse_center = float(np.sqrt(np.mean((vm_all_as[sample_as_idx] - vm_all_nrv_i[sample_nrv_idx]) ** 2)))
    corr = float(np.corrcoef(vm_all_as.ravel(), vm_all_nrv_i.ravel())[0, 1])

    peak_as = float(np.max(vm_all_as[sample_as_idx]))
    peak_nrv = float(np.max(vm_all_nrv_i[sample_nrv_idx]))
    d_peak = peak_as - peak_nrv

    min_peak_distance_pts = max(1, int(0.5 / dt))
    peaks_as, _ = find_peaks(vm_all_as[sample_as_idx], height=0.0, distance=min_peak_distance_pts)
    peaks_nrv, _ = find_peaks(vm_all_nrv_i[sample_nrv_idx], height=0.0, distance=min_peak_distance_pts)

    # Vext validation vs analytical point-source profile at pulse peak.
    i_vext_peak = int(np.argmax(np.max(np.abs(vext_all_as), axis=0)))
    x_m = x_all_as * 1e-6
    x0_m = x0_um * 1e-6
    y_m = elec_y_um * 1e-6
    z_m = elec_z_um * 1e-6
    r = np.sqrt((x_m - x0_m) ** 2 + y_m**2 + z_m**2)
    vext_ref_mV = (1.0 / (4.0 * np.pi * sigma_S_m * np.maximum(r, 1e-12))) * (-cathodic_uA * 1e-6) * 1e3
    rmse_vext = float(np.sqrt(np.mean((vext_all_as[:, i_vext_peak] - vext_ref_mV) ** 2)))

    # --- Visual validation figure ---
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    axs[0, 0].plot(t_as, vm_all_as[sample_as_idx], lw=2, label="AxonScope")
    axs[0, 0].plot(t_as, vm_all_nrv_i[sample_nrv_idx], "--", lw=2, label="NRV (interp)")
    axs[0, 0].set_title("Center node trace")
    axs[0, 0].set_xlabel("Time [ms]")
    axs[0, 0].set_ylabel("Vm [mV]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    im0 = axs[0, 1].imshow(
        vm_all_as,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, vm_all_as.shape[0] - 1],
        cmap="viridis",
    )
    axs[0, 1].set_title("AxonScope full heatmap")
    axs[0, 1].set_xlabel("Time [ms]")
    axs[0, 1].set_ylabel("Compartment index")
    fig.colorbar(im0, ax=axs[0, 1], label="Vm [mV]")

    im1 = axs[0, 2].imshow(
        vm_all_nrv_i,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, vm_all_nrv_i.shape[0] - 1],
        cmap="viridis",
    )
    axs[0, 2].set_title("NRV full heatmap (interp)")
    axs[0, 2].set_xlabel("Time [ms]")
    axs[0, 2].set_ylabel("Compartment index")
    fig.colorbar(im1, ax=axs[0, 2], label="Vm [mV]")

    vmax_abs = float(np.max(np.abs(err)))
    im2 = axs[1, 0].imshow(
        err,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, err.shape[0] - 1],
        cmap="coolwarm",
        vmin=-vmax_abs,
        vmax=vmax_abs,
    )
    axs[1, 0].set_title(f"Error heatmap (RMSE={rmse_global:.2f} mV, corr={corr:.3f})")
    axs[1, 0].set_xlabel("Time [ms]")
    axs[1, 0].set_ylabel("Compartment index")
    fig.colorbar(im2, ax=axs[1, 0], label="ΔVm [mV]")

    im3 = axs[1, 1].imshow(
        vext_all_as,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, vext_all_as.shape[0] - 1],
        cmap="magma",
    )
    axs[1, 1].set_title("Applied Vext (AxonScope)")
    axs[1, 1].set_xlabel("Time [ms]")
    axs[1, 1].set_ylabel("Compartment index")
    fig.colorbar(im3, ax=axs[1, 1], label="Vext [mV]")

    axs[1, 2].plot(x_all_as, np.max(np.abs(vext_all_as), axis=1), "k-", lw=2, label="AxonScope")
    axs[1, 2].plot(x_all_as, np.abs(vext_ref_mV), "r--", lw=2, label="Analytical/NRV ref")
    axs[1, 2].set_title(f"Peak |Vext| profile (RMSE={rmse_vext:.3e} mV)")
    axs[1, 2].set_xlabel("x [um]")
    axs[1, 2].set_ylabel("Peak |Vext| [mV]")
    axs[1, 2].grid(True, alpha=0.3)
    axs[1, 2].legend()

    fig_path = os.path.join(save_dir, "myelinated_mrg_extracellular_ctx_api_vs_nrv.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("=== Extracellular ctx API vs NRV (myelinated MRG) ===")
    print(f"RMSE Vm global={rmse_global:.3f} mV | center={rmse_center:.3f} mV | corr={corr:.3f}")
    print(f"Δpeak center={d_peak:+.3f} mV | peaks center AS/NRV={len(peaks_as)}/{len(peaks_nrv)}")
    print(f"Vext RMSE vs analytical NRV ref={rmse_vext:.3e} mV")
    print(f"Figure saved -> {fig_path}")

    # Strict on extracellular profile/API correctness; moderate on Vm (known model-level gap).
    assert np.isfinite(vm_all_as).all()
    assert np.isfinite(vm_all_nrv_i).all()
    assert len(peaks_as) == 1 and len(peaks_nrv) == 1
    assert corr > 0.88
    assert rmse_global < 15.0
    assert rmse_center < 15.0
    assert abs(d_peak) < 40.0
    assert rmse_vext < 1e-3
