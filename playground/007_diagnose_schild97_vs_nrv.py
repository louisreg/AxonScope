"""
Diagnostic script for Schild_97 parity (AxonScope vs NRV).

Focus:
- Vm comparison at one compartment
- AxonScope membrane-current budget proxy via -Cm*dV/dt
- NRV grouped ionic currents (I_na, I_k, I_ca)

Usage:
    mamba run -n Axonscope-env python playground/007_diagnose_schild97_vs_nrv.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from axonscope.axons.unmyelinated import Schild97
from axonscope.solvers.CrankNicholson import CrankNicholson
import nrv


def main():
    L = 5000.0
    D = 1.0
    NX = 101
    DT_AS = 0.025
    DT_NRV = 0.005
    TSIM = 30.0
    T = 37.0
    AMP = 2.0
    T_START = 5.0
    DUR = 1.0
    idx_as = int(NX * 3 / 4)
    x_probe_um = idx_as * L / NX

    # AxonScope
    ax_as = Schild97(L=L, d=D, Nx=NX)
    ax_as.insert_I_Clamp(position=L / 2, t_start=T_START, duration=DUR, amplitude=AMP)
    res_as = CrankNicholson().solve(ax_as, tsim=TSIM, dt=DT_AS, record_diagnostics=True)
    t_as = np.asarray(res_as.t)
    v_as = np.asarray(res_as.Vm)[:, idx_as]
    # Proxy for total membrane current from Vm slope.
    # Units: µA/cm² because Cm is µF/cm² and dV/dt is mV/ms.
    i_mem_as = -ax_as.Cm * np.gradient(v_as, DT_AS)
    diag = res_as.diagnostics if res_as.diagnostics is not None else {}
    i_na_as = np.asarray(diag.get("I_na_total_uAcm2"))[:, idx_as]
    i_k_as = np.asarray(diag.get("I_k_total_uAcm2"))[:, idx_as]
    i_ca_as = np.asarray(diag.get("I_ca_total_uAcm2"))[:, idx_as]
    i_tot_rhs_as = np.asarray(diag.get("I_total_rhs_uAcm2"))[:, idx_as]

    # NRV
    ax_nrv = nrv.unmyelinated(
        0,
        0,
        D,
        L,
        dt=DT_NRV,
        Nsec=1,
        Nseg_per_sec=NX,
        model="Schild_97",
        v_init=-48.0,
        T=T,
    )
    ax_nrv.insert_I_Clamp(0.5, T_START, DUR, AMP)
    out = ax_nrv.simulate(t_sim=TSIM, record_I_ions=True)

    t_nrv = np.asarray(out["t"]).ravel()
    x_nrv = np.asarray(out["x_rec"])
    idx_nrv = int(np.argmin(np.abs(x_nrv - x_probe_um)))
    v_nrv = np.asarray(out["V_mem"])[idx_nrv]

    i_na = np.asarray(out["I_na"])
    i_k = np.asarray(out["I_k"])
    i_ca = np.asarray(out["I_ca"])
    if i_na.shape[0] == len(x_nrv):
        i_na = i_na[idx_nrv]
        i_k = i_k[idx_nrv]
        i_ca = i_ca[idx_nrv]
    else:
        i_na = i_na[:, idx_nrv]
        i_k = i_k[:, idx_nrv]
        i_ca = i_ca[:, idx_nrv]
    # NRV currents are in mA/cm² -> convert to µA/cm² for direct comparison.
    i_mem_nrv = (i_na + i_k + i_ca) * 1e3

    print(f"NRV mesh: Nsec={ax_nrv.Nsec}, Nseg={ax_nrv.Nseg}, len(x_rec)={len(x_nrv)}")

    # Metrics
    d_peak = float(np.max(v_as) - np.max(v_nrv))
    d_ahp = float(np.min(v_as) - np.min(v_nrv))
    rmse_v = float(np.sqrt(np.mean((np.interp(t_nrv, t_as, v_as) - v_nrv) ** 2)))

    print(f"Schild97 @ x≈{x_probe_um/1000:.3f} mm")
    print(f"  Δpeak = {d_peak:+.3f} mV")
    print(f"  ΔAHP  = {d_ahp:+.3f} mV")
    print(f"  RMSE(V) over full trace = {rmse_v:.3f} mV")

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    axs[0].plot(t_as, v_as, label="AxonScope Vm", lw=2)
    axs[0].plot(t_nrv, v_nrv, "--", label="NRV Vm", lw=1.5)
    axs[0].set_title("Schild_97 Vm trace")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Vm (mV)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(t_as, i_tot_rhs_as, label="AxonScope I_na+I_k+I_ca (instrumented)", lw=1.8)
    axs[1].plot(t_nrv, i_mem_nrv, "--", label="NRV I_na+I_k+I_ca", lw=1.4)
    axs[1].set_title("Grouped ionic current budget")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Current density (µA/cm²)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[2].plot(t_as, i_na_as, label="AS I_na", lw=1.2)
    axs[2].plot(t_as, i_k_as, label="AS I_k", lw=1.2)
    axs[2].plot(t_as, i_ca_as, label="AS I_ca", lw=1.2)
    axs[2].plot(t_nrv, i_na * 1e3, "--", label="NRV I_na", lw=1.2)
    axs[2].plot(t_nrv, i_k * 1e3, "--", label="NRV I_k", lw=1.2)
    axs[2].plot(t_nrv, i_ca * 1e3, "--", label="NRV I_ca", lw=1.2)
    axs[2].set_title("Channel-group comparison (AxonScope vs NRV)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylabel("Current density (µA/cm²)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    out_dir = "figures/physics_tests"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Schild97_diagnostic_axonscope_vs_nrv.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")


if __name__ == "__main__":
    main()
