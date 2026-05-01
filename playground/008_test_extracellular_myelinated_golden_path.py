from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import nrv

from axonscope.axons.myelinated import MRG
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers.CrankNicholson import CrankNicholson
from axonscope.stimulus import Stimulus


@dataclass
class Config:
    diameter_um: float = 10.0
    nodes: int = 9
    dt_ms: float = 0.005
    tsim_ms: float = 4.0
    stim_start_ms: float = 1.0
    cathodic_uA: float = 80.0
    cathodic_duration_ms: float = 0.08
    anodic_uA: float = 20.0
    interphase_ms: float = 0.04
    elec_y_um: float = 100.0
    elec_z_um: float = 0.0
    sigma_S_m: float = 0.2
    save_dir: str = "figures/physics_tests"
    no_plot: bool = False


def _interp_rows(values_by_row: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.empty((values_by_row.shape[0], t_dst.size), dtype=float)
    for i in range(values_by_row.shape[0]):
        out[i] = np.interp(t_dst, t_src, values_by_row[i])
    return out


def run_axonscope(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    ax = MRG(d=cfg.diameter_um, nodes=cfg.nodes)
    x0_um = float(ax.L / 2.0)

    electrode = PointSourceElectrode(
        x0_m=x0_um * 1e-6,
        y0_m=cfg.elec_y_um * 1e-6,
        z0_m=cfg.elec_z_um * 1e-6,
        sigma_S_m=cfg.sigma_S_m,
    )
    stim = Stimulus.biphasic(
        start=cfg.stim_start_ms,
        cathodic_amplitude=cfg.cathodic_uA * 1e-6,
        cathodic_duration=cfg.cathodic_duration_ms,
        anodic_amplitude=cfg.anodic_uA * 1e-6,
        interphase=cfg.interphase_ms,
    )
    ax.attach_extracellular_stimulus(electrode.attach_stimulus(stim))
    ax.use_extracellular = True

    res = CrankNicholson().solve(ax, tsim=cfg.tsim_ms, dt=cfg.dt_ms)

    t_as = np.asarray(res.t)
    node_idx_as = np.asarray(ax.node_indices, dtype=int)
    vm_nodes_as = np.asarray(res.Vm)[:, node_idx_as].T
    vext_full = np.stack([np.asarray(ax.Vext_mV(float(t))) for t in t_as], axis=0).T
    vext_mV_as = vext_full[node_idx_as]
    return t_as, vm_nodes_as, vext_mV_as, np.asarray(ax.x)[node_idx_as], float(ax.L)


def run_nrv(cfg: Config, L_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax_nrv = nrv.myelinated(
        0,
        0,
        cfg.diameter_um,
        float(L_um),
        model="MRG",
        dt=cfg.dt_ms,
        node_shift=0,
        Nseg_per_sec=1,
        rec="nodes",
        T=37.0,
        v_init=-80.0,
    )

    elec = nrv.point_source_electrode(L_um / 2.0, cfg.elec_y_um, cfg.elec_z_um)
    stim = nrv.stimulus()
    stim.biphasic_pulse(
        cfg.stim_start_ms,
        cfg.cathodic_uA,
        cfg.cathodic_duration_ms,
        cfg.anodic_uA,
        cfg.interphase_ms,
    )
    extra = nrv.stimulation("endoneurium_bhadra")
    extra.add_electrode(elec, stim)
    ax_nrv.attach_extracellular_stimulation(extra)

    out = ax_nrv.simulate(t_sim=cfg.tsim_ms)
    t_nrv = np.asarray(out["t"]).ravel()
    vm_nrv = np.asarray(out["V_mem"])
    if vm_nrv.shape[0] > vm_nrv.shape[1]:
        vm_nrv = vm_nrv.T
    x_nrv = np.asarray(out["x_rec"], dtype=float)
    return t_nrv, vm_nrv, x_nrv


def main(cfg: Config) -> None:
    t_as, vm_nodes_as, vext_mV_as, x_nodes_as, L_um = run_axonscope(cfg)
    t_nrv, vm_nodes_nrv, x_nodes_nrv = run_nrv(cfg, L_um=L_um)

    n = min(vm_nodes_as.shape[0], vm_nodes_nrv.shape[0], x_nodes_nrv.size, x_nodes_as.size)
    vm_nodes_as = vm_nodes_as[:n]
    vm_nodes_nrv = vm_nodes_nrv[:n]
    x_nodes_as = x_nodes_as[:n]
    x_nodes_nrv = x_nodes_nrv[:n]

    vm_nodes_nrv_i = _interp_rows(vm_nodes_nrv, t_nrv, t_as)
    err = vm_nodes_as - vm_nodes_nrv_i

    # Compare extracellular profiles at pulse peak time.
    i_peak = int(np.argmax(np.max(np.abs(vext_mV_as[:n]), axis=0)))
    vext_peak_as = vext_mV_as[:n, i_peak]
    t_peak = float(t_as[i_peak])
    # Analytical point-source profile used by NRV for isotropic material.
    x_m = x_nodes_nrv[:n] * 1e-6
    x0_m = 0.5 * L_um * 1e-6
    y_m = cfg.elec_y_um * 1e-6
    z_m = cfg.elec_z_um * 1e-6
    r = np.sqrt((x_m - x0_m) ** 2 + y_m**2 + z_m**2)
    fp_v_per_a = 1.0 / (4.0 * np.pi * cfg.sigma_S_m * np.maximum(r, 1e-12))
    vext_peak_nrv = fp_v_per_a * (-cfg.cathodic_uA * 1e-6) * 1e3
    rmse_vext = float(np.sqrt(np.mean((vext_peak_as - vext_peak_nrv) ** 2)))

    rmse_global = float(np.sqrt(np.mean(err**2)))
    center = n // 2
    rmse_center = float(np.sqrt(np.mean((vm_nodes_as[center] - vm_nodes_nrv_i[center]) ** 2)))
    peak_as = float(np.max(vm_nodes_as[center]))
    peak_nrv = float(np.max(vm_nodes_nrv_i[center]))
    ahp_as = float(np.min(vm_nodes_as[center]))
    ahp_nrv = float(np.min(vm_nodes_nrv_i[center]))
    vmax_abs = float(np.max(np.abs(err)))

    print("=== Extracellular Golden Path (MRG) ===")
    print(f"d={cfg.diameter_um:.2f} um | nodes={cfg.nodes} | dt={cfg.dt_ms:.4f} ms | tsim={cfg.tsim_ms:.3f} ms")
    print(
        "Stimulus: biphasic "
        f"({cfg.cathodic_uA:.1f} uA cathodic for {cfg.cathodic_duration_ms:.3f} ms, "
        f"{cfg.anodic_uA:.1f} uA anodic, interphase {cfg.interphase_ms:.3f} ms)"
    )
    print(f"Electrode: point source at y={cfg.elec_y_um:.1f} um, z={cfg.elec_z_um:.1f} um, sigma={cfg.sigma_S_m:.3f} S/m")
    print(f"AxonScope Vext peak: {float(np.max(np.abs(vext_mV_as))):.3f} mV")
    print(f"Center node Δpeak: {peak_as - peak_nrv:+.3f} mV")
    print(f"Center node ΔAHP : {ahp_as - ahp_nrv:+.3f} mV")
    print(f"RMSE global: {rmse_global:.3f} mV | RMSE center: {rmse_center:.3f} mV | max|err|: {vmax_abs:.3f} mV")
    print(f"Vext profile check @ t={t_peak:.3f} ms: RMSE={rmse_vext:.3f} mV")

    if cfg.no_plot:
        return

    os.makedirs(cfg.save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    axs[0, 0].plot(t_as, vm_nodes_as[center], lw=2, label="AxonScope")
    axs[0, 0].plot(t_as, vm_nodes_nrv_i[center], "--", lw=2, label="NRV (interp)")
    axs[0, 0].set_title("Center node trace")
    axs[0, 0].set_xlabel("Time [ms]")
    axs[0, 0].set_ylabel("Vm [mV]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    im0 = axs[0, 1].imshow(
        vm_nodes_as,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, n - 1],
        cmap="viridis",
    )
    axs[0, 1].set_title("AxonScope node heatmap")
    axs[0, 1].set_xlabel("Time [ms]")
    axs[0, 1].set_ylabel("Node index")
    fig.colorbar(im0, ax=axs[0, 1], label="Vm [mV]")

    im1 = axs[0, 2].imshow(
        vm_nodes_nrv_i,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, n - 1],
        cmap="viridis",
    )
    axs[0, 2].set_title("NRV node heatmap (interp)")
    axs[0, 2].set_xlabel("Time [ms]")
    axs[0, 2].set_ylabel("Node index")
    fig.colorbar(im1, ax=axs[0, 2], label="Vm [mV]")

    im2 = axs[1, 0].imshow(
        err,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, n - 1],
        cmap="coolwarm",
        vmin=-vmax_abs,
        vmax=vmax_abs,
    )
    axs[1, 0].set_title(f"Error heatmap (RMSE={rmse_global:.2f} mV)")
    axs[1, 0].set_xlabel("Time [ms]")
    axs[1, 0].set_ylabel("Node index")
    fig.colorbar(im2, ax=axs[1, 0], label="ΔVm [mV]")

    im3 = axs[1, 1].imshow(
        vext_mV_as[:n],
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), 0, n - 1],
        cmap="magma",
    )
    axs[1, 1].set_title("Applied Vext (AxonScope)")
    axs[1, 1].set_xlabel("Time [ms]")
    axs[1, 1].set_ylabel("Node index")
    fig.colorbar(im3, ax=axs[1, 1], label="Vext [mV]")

    axs[1, 2].plot(x_nodes_as, np.max(np.abs(vext_mV_as[:n]), axis=1), "k-", lw=2, label="AxonScope")
    axs[1, 2].plot(x_nodes_nrv, np.abs(vext_peak_nrv), "r--", lw=2, label="NRV (footprint)")
    axs[1, 2].set_title("Peak |Vext| profile along nodes")
    axs[1, 2].set_xlabel("x [um]")
    axs[1, 2].set_ylabel("Peak |Vext| [mV]")
    axs[1, 2].grid(True, alpha=0.3)
    axs[1, 2].legend()

    out_path = os.path.join(cfg.save_dir, "myelinated_mrg_extracellular_axonscope_vs_nrv.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Golden-path extracellular test for myelinated MRG (AxonScope vs NRV).")
    p.add_argument("--diameter-um", type=float, default=10.0)
    p.add_argument("--nodes", type=int, default=9)
    p.add_argument("--dt-ms", type=float, default=0.005)
    p.add_argument("--tsim-ms", type=float, default=4.0)
    p.add_argument("--stim-start-ms", type=float, default=1.0)
    p.add_argument("--cathodic-ua", type=float, default=80.0)
    p.add_argument("--cathodic-duration-ms", type=float, default=0.08)
    p.add_argument("--anodic-ua", type=float, default=20.0)
    p.add_argument("--interphase-ms", type=float, default=0.04)
    p.add_argument("--elec-y-um", type=float, default=100.0)
    p.add_argument("--elec-z-um", type=float, default=0.0)
    p.add_argument("--sigma-s-m", type=float, default=0.2)
    p.add_argument("--save-dir", type=str, default="figures/physics_tests")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    cfg = Config(
        diameter_um=args.diameter_um,
        nodes=args.nodes,
        dt_ms=args.dt_ms,
        tsim_ms=args.tsim_ms,
        stim_start_ms=args.stim_start_ms,
        cathodic_uA=args.cathodic_ua,
        cathodic_duration_ms=args.cathodic_duration_ms,
        anodic_uA=args.anodic_ua,
        interphase_ms=args.interphase_ms,
        elec_y_um=args.elec_y_um,
        elec_z_um=args.elec_z_um,
        sigma_S_m=args.sigma_s_m,
        save_dir=args.save_dir,
        no_plot=args.no_plot,
    )
    main(cfg)
