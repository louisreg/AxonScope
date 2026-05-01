from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import nrv


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Config:
    diameter_um: float = 10.0
    nodes: int = 9
    dt: float = 0.001
    tsim: float = 6.0
    stim_amp_nA: float = 5.0
    stim_start_ms: float = 1.0
    stim_duration_ms: float = 0.1
    celsius: float = 37.0
    ap_threshold_mV: float = 0.0
    stim_node: int | None = None
    zoom_tmin_ms: float | None = None
    zoom_tmax_ms: float | None = None
    with_plot: bool = True


def _velocity_from_peaks(x_um: np.ndarray, vm_nodes: np.ndarray, t_ms: np.ndarray, threshold_mV: float) -> float:
    center_x = float(x_um[len(x_um) // 2])
    peaks = np.max(vm_nodes, axis=1)
    tpk = t_ms[np.argmax(vm_nodes, axis=1)]
    dist_um = np.abs(x_um - center_x)
    mask = (dist_um > 0.0) & (peaks > threshold_mV)
    if np.count_nonzero(mask) < 2:
        return float("nan")

    d = dist_um[mask]
    tp = tpk[mask]

    # Symmetric stimulation creates left/right nodes with the same distance.
    # Average arrival time per distance to avoid sign artifacts from tiny jitter.
    d_round = np.round(d, 6)
    uniq = np.unique(d_round)
    d_u = np.array([u for u in uniq], dtype=float)
    t_u = np.array([tp[d_round == u].mean() for u in uniq], dtype=float)

    if d_u.size < 2:
        return float("nan")

    coeff = np.polyfit(t_u * 1e-3, d_u * 1e-6, 1)
    return float(coeff[0])


def _run_axonscope(cfg: Config):
    m003 = _load_module("m003", Path("playground/003_test_standalone_MRG.py"))
    stim_node = cfg.stim_node if cfg.stim_node is not None else cfg.nodes // 2
    morph, t_ms, _vi, _ve, vm = m003.simulate(
        diameter_um=cfg.diameter_um,
        nodes=cfg.nodes,
        tsim=cfg.tsim,
        dt=cfg.dt,
        stim_node=stim_node,
        stim_amp_nA=cfg.stim_amp_nA,
        stim_start_ms=cfg.stim_start_ms,
        stim_duration_ms=cfg.stim_duration_ms,
        celsius=cfg.celsius,
    )
    node_idx = morph.node_indices
    x_nodes_um = morph.x_um[node_idx]
    vm_nodes = vm[:, node_idx].T
    x_full_um = morph.x_um
    vm_full = vm.T
    return (
        np.asarray(t_ms),
        np.asarray(x_nodes_um),
        np.asarray(vm_nodes),
        np.asarray(x_full_um),
        np.asarray(vm_full),
        m003,
    )


def _run_nrv(cfg: Config, m003):
    p = m003.get_mrg_params(cfg.diameter_um)
    length_um = float(math.ceil(p.deltax * (cfg.nodes - 1)))
    stim_node = cfg.stim_node if cfg.stim_node is not None else cfg.nodes // 2
    ax = nrv.myelinated(
        0,
        0,
        cfg.diameter_um,
        length_um,
        model="MRG",
        dt=cfg.dt,
        node_shift=0,
        Nseg_per_sec=1,
        rec="nodes",
        T=cfg.celsius,
        v_init=-80.0,
    )
    ax.insert_I_Clamp_node(
        index=stim_node,
        t_start=cfg.stim_start_ms,
        duration=cfg.stim_duration_ms,
        amplitude=cfg.stim_amp_nA,
    )
    out = ax.simulate(t_sim=cfg.tsim)
    t_ms = np.asarray(out["t"]).ravel()
    x_nodes_um = np.asarray(out["x_rec"]).ravel()
    vm_nodes = np.asarray(out["V_mem"])
    if vm_nodes.shape[0] != x_nodes_um.size and vm_nodes.shape[1] == x_nodes_um.size:
        vm_nodes = vm_nodes.T
    return t_ms, x_nodes_um, vm_nodes


def _interp_rows(y_rows: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.empty((y_rows.shape[0], t_dst.size), dtype=float)
    for i in range(y_rows.shape[0]):
        out[i] = np.interp(t_dst, t_src, y_rows[i])
    return out


def compare(cfg: Config):
    t_as, x_as, vm_as, x_full_as, vm_full_as, m003 = _run_axonscope(cfg)
    t_nrv, x_nrv, vm_nrv = _run_nrv(cfg, m003)

    n = min(vm_as.shape[0], vm_nrv.shape[0])
    vm_as = vm_as[:n]
    vm_nrv = vm_nrv[:n]
    x_as = x_as[:n]
    x_nrv = x_nrv[:n]

    vm_nrv_i = _interp_rows(vm_nrv, t_nrv, t_as)

    center = n // 2
    rmse_center = float(np.sqrt(np.mean((vm_as[center] - vm_nrv_i[center]) ** 2)))
    rmse_global = float(np.sqrt(np.mean((vm_as - vm_nrv_i) ** 2)))
    mae_global = float(np.mean(np.abs(vm_as - vm_nrv_i)))
    peak_as = np.max(vm_as, axis=1)
    peak_nrv = np.max(vm_nrv_i, axis=1)
    tpk_as = t_as[np.argmax(vm_as, axis=1)]
    tpk_nrv = t_as[np.argmax(vm_nrv_i, axis=1)]
    dpeak = float(np.mean(np.abs(peak_as - peak_nrv)))
    v_as = _velocity_from_peaks(x_as, vm_as, t_as, cfg.ap_threshold_mV)
    v_nrv = _velocity_from_peaks(x_nrv, vm_nrv_i, t_as, cfg.ap_threshold_mV)

    print("=== MRG AxonScope vs NRV ===")
    print(f"nodes compared: {n}")
    print(f"stim node index: {cfg.stim_node if cfg.stim_node is not None else cfg.nodes // 2}")
    print(f"finite AxonScope: {bool(np.isfinite(vm_as).all())}")
    print(f"finite NRV: {bool(np.isfinite(vm_nrv_i).all())}")
    print(f"center-node RMSE [mV]: {rmse_center:.3f}")
    print(f"global RMSE [mV]: {rmse_global:.3f}")
    print(f"global MAE [mV]: {mae_global:.3f}")
    print(f"mean |peak error| [mV]: {dpeak:.3f}")
    print(f"velocity AxonScope [m/s]: {v_as:.4f}")
    print(f"velocity NRV [m/s]: {v_nrv:.4f}")
    if np.isfinite(v_as) and np.isfinite(v_nrv):
        print(f"|velocity error| [m/s]: {abs(v_as - v_nrv):.4f}")
    print("")
    print("node  x_as[um]  peak_as[mV]  peak_nrv[mV]")
    for i in range(n):
        print(f"{i:>4d} {x_as[i]:>9.2f} {peak_as[i]:>12.3f} {peak_nrv[i]:>13.3f}")

    if cfg.with_plot:
        fig, axs = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)

        axs[0, 0].plot(t_as, vm_as[center], lw=2, label="AxonScope center node")
        axs[0, 0].plot(t_as, vm_nrv_i[center], "--", lw=2, label="NRV center node")
        axs[0, 0].set_xlabel("Time [ms]")
        axs[0, 0].set_ylabel("Vm [mV]")
        axs[0, 0].set_title("Center node trace")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend(fontsize=8)

        axs[0, 1].plot(x_as, peak_as, "o-", lw=1.8, label="AxonScope peak Vm")
        axs[0, 1].plot(x_nrv, peak_nrv, "s--", lw=1.8, label="NRV peak Vm")
        axs[0, 1].set_xlabel("Node position [um]")
        axs[0, 1].set_ylabel("Peak Vm [mV]")
        axs[0, 1].set_title("Peak-by-node comparison")
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend(fontsize=8)

        axs[0, 2].plot(x_as, tpk_as, "o-", lw=1.8, label="AxonScope peak time")
        axs[0, 2].plot(x_nrv, tpk_nrv, "s--", lw=1.8, label="NRV peak time")
        axs[0, 2].set_xlabel("Node position [um]")
        axs[0, 2].set_ylabel("Peak time [ms]")
        axs[0, 2].set_title("Peak timing by node")
        axs[0, 2].grid(True, alpha=0.3)
        axs[0, 2].legend(fontsize=8)

        vmin = float(min(np.min(vm_as), np.min(vm_nrv_i)))
        vmax = float(max(np.max(vm_as), np.max(vm_nrv_i)))
        err = vm_as - vm_nrv_i
        emax = float(np.max(np.abs(err)))

        im0 = axs[1, 0].imshow(
            vm_as,
            aspect="auto",
            origin="lower",
            extent=[float(t_as[0]), float(t_as[-1]), float(x_as[0]), float(x_as[-1])],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axs[1, 0].set_title("Heatmap AxonScope")
        axs[1, 0].set_xlabel("Time [ms]")
        axs[1, 0].set_ylabel("Node position [um]")
        fig.colorbar(im0, ax=axs[1, 0], label="Vm [mV]")

        im1 = axs[1, 1].imshow(
            vm_nrv_i,
            aspect="auto",
            origin="lower",
            extent=[float(t_as[0]), float(t_as[-1]), float(x_nrv[0]), float(x_nrv[-1])],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axs[1, 1].set_title("Heatmap NRV (interp on t)")
        axs[1, 1].set_xlabel("Time [ms]")
        axs[1, 1].set_ylabel("Node position [um]")
        fig.colorbar(im1, ax=axs[1, 1], label="Vm [mV]")

        im2 = axs[1, 2].imshow(
            err,
            aspect="auto",
            origin="lower",
            extent=[float(t_as[0]), float(t_as[-1]), float(x_as[0]), float(x_as[-1])],
            vmin=-emax,
            vmax=emax,
            cmap="coolwarm",
        )
        axs[1, 2].set_title("Heatmap error (AxonScope - NRV)")
        axs[1, 2].set_xlabel("Time [ms]")
        axs[1, 2].set_ylabel("Node position [um]")
        fig.colorbar(im2, ax=axs[1, 2], label="ΔVm [mV]")

        # Full-compartment AxonScope heatmap: this is where saltatory structure is easiest to see.
        im3 = axs[2, 0].imshow(
            vm_full_as,
            aspect="auto",
            origin="lower",
            extent=[float(t_as[0]), float(t_as[-1]), float(x_full_as[0]), float(x_full_as[-1])],
            cmap="viridis",
        )
        axs[2, 0].set_title("Heatmap AxonScope (full compartments)")
        axs[2, 0].set_xlabel("Time [ms]")
        axs[2, 0].set_ylabel("Position [um]")
        fig.colorbar(im3, ax=axs[2, 0], label="Vm [mV]")

        # Same full heatmap with node markers to emphasize node-to-node regeneration.
        im4 = axs[2, 1].imshow(
            vm_full_as,
            aspect="auto",
            origin="lower",
            extent=[float(t_as[0]), float(t_as[-1]), float(x_full_as[0]), float(x_full_as[-1])],
            cmap="viridis",
        )
        for xn in x_as:
            axs[2, 1].axhline(float(xn), color="white", lw=0.5, alpha=0.5)
        axs[2, 1].set_title("Full heatmap + node lines")
        axs[2, 1].set_xlabel("Time [ms]")
        axs[2, 1].set_ylabel("Position [um]")
        fig.colorbar(im4, ax=axs[2, 1], label="Vm [mV]")

        # Zoomed node heatmap for visual front-tracking.
        im5 = axs[2, 2].imshow(
            vm_as,
            aspect="auto",
            origin="lower",
            extent=[float(t_as[0]), float(t_as[-1]), float(x_as[0]), float(x_as[-1])],
            cmap="viridis",
        )
        axs[2, 2].set_title("Node heatmap (zoom target)")
        axs[2, 2].set_xlabel("Time [ms]")
        axs[2, 2].set_ylabel("Node position [um]")
        fig.colorbar(im5, ax=axs[2, 2], label="Vm [mV]")

        # Optional time zoom to reveal saltatory timing.
        if cfg.zoom_tmin_ms is not None or cfg.zoom_tmax_ms is not None:
            tmin = cfg.zoom_tmin_ms if cfg.zoom_tmin_ms is not None else float(t_as[0])
            tmax = cfg.zoom_tmax_ms if cfg.zoom_tmax_ms is not None else float(t_as[-1])
            for r in range(3):
                for c in range(3):
                    axs[r, c].set_xlim(tmin, tmax)

        fig.suptitle(
            (
                f"MRG comparison | RMSEc={rmse_center:.3f} mV | "
                f"RMSEg={rmse_global:.3f} mV | MAEg={mae_global:.3f} mV"
            ),
            fontsize=11,
        )
        plt.show()


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Standalone MRG comparison: AxonScope prototype vs NRV.")
    p.add_argument("--diameter-um", type=float, default=10.0)
    p.add_argument("--nodes", type=int, default=9)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--tsim", type=float, default=6.0)
    p.add_argument("--stim-amp-na", type=float, default=5.0)
    p.add_argument("--stim-start-ms", type=float, default=1.0)
    p.add_argument("--stim-duration-ms", type=float, default=0.1)
    p.add_argument("--celsius", type=float, default=37.0)
    p.add_argument("--ap-threshold-mv", type=float, default=0.0)
    p.add_argument("--stim-node", type=int, default=None, help="Stimulated node index (default: middle node)")
    p.add_argument("--zoom-tmin-ms", type=float, default=None, help="Optional plot zoom start time [ms]")
    p.add_argument("--zoom-tmax-ms", type=float, default=None, help="Optional plot zoom end time [ms]")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    return Config(
        diameter_um=args.diameter_um,
        nodes=args.nodes,
        dt=args.dt,
        tsim=args.tsim,
        stim_amp_nA=args.stim_amp_na,
        stim_start_ms=args.stim_start_ms,
        stim_duration_ms=args.stim_duration_ms,
        celsius=args.celsius,
        ap_threshold_mV=args.ap_threshold_mv,
        stim_node=args.stim_node,
        zoom_tmin_ms=args.zoom_tmin_ms,
        zoom_tmax_ms=args.zoom_tmax_ms,
        with_plot=not args.no_plot,
    )


if __name__ == "__main__":
    compare(_parse_args())
