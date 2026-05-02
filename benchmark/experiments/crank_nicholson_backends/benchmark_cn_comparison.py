"""
benchmark_cn_comparison.py
==========================

Compare the main implicit and exponential solver variants on a
Hodgkin-Huxley axon:

  CrankNicholson             – reference Hines tridiagonal solver (Neumann BCs)
  CrankNicholsonSemiImplicit – linearized ionic currents
  CrankNicholsonImplicit     – Newton iteration per step
  CrankNicholsonImplicitFast – semi-implicit fast tridiagonal form
  CrankNicholsonImplicitFastMultiStep – fast tridiagonal form batched by blocks
  CrankNicholsonQuasiNewtonFast – quasi-Newton fast tridiagonal form

Two sweeps are run:
  fine   – dt = 0.001 ms for all fixed-step solvers  → all should agree closely
  coarse – dt = 0.1 ms                               → shows stability advantage
                                                         of implicit methods

Outputs
-------
  benchmark/figures/cn_comparison.png   (comparison figure)
  benchmark/cn_comparison.csv           (timing + accuracy table)
"""

from __future__ import annotations

import csv
import os
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import jax.numpy as jnp

from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.solvers.CrankNicholson import (
    CrankNicholson,
    CrankNicholsonImplicit,
    CrankNicholsonSemiImplicit,
    CrankNicholsonImplicitFast,
    CrankNicholsonImplicitFastMultiStep,
    CrankNicholsonQuasiNewtonFast,
)

# ── Simulation parameters ─────────────────────────────────────────────────────
L_UM      = 1000.0   # axon length [µm]
D_UM      = 0.5      # diameter [µm]
NX        = 101      # compartments
CELSIUS   = 6.3      # temperature [°C]
T_START   = 1.0      # stimulus onset [ms]
T_DUR     = 1.0      # stimulus duration [ms]
AMP       = 2.0      # stimulus amplitude [nA]
TSIM      = 150.0     # simulation time [ms]

DT_FINE   = 0.001    # fine time step [ms]
DT_COARSE = 0.1        # coarse time step [ms]

# Compartment index used for trace plots (~3/4 of axon length)
IDX_PLOT = int(NX * 3 / 4)

COLORS = {
    "CN":            "tab:blue",
    "SemiImplicit":  "tab:orange",
    "Implicit":      "tab:green",
    "Adaptive":      "tab:red",
    "CrankNicholsonImplicitFast": "tab:purple",
    "CrankNicholsonImplicitFastMultiStep": "tab:brown",
    "CrankNicholsonQuasiNewtonFast": "tab:pink",
}


def color_for(name: str) -> str:
    """Return a stable plotting color for a solver name."""
    return COLORS.get(name, "tab:gray")

# ── Helper ────────────────────────────────────────────────────────────────────

def make_axon() -> HodgkinHuxley:
    axon = HodgkinHuxley(L=L_UM, d=D_UM, Nx=NX, celsius=CELSIUS)
    axon.insert_I_Clamp(
        position=L_UM / 2,
        t_start=T_START,
        duration=T_DUR,
        amplitude=AMP,
    )
    return axon


def run_solver(name: str, solver, dt: float) -> dict:
    """
    Warm up (triggers JIT), then time one production run.

    Parameters
    ----------
    name : str
    solver : Solver instance
    dt : float
        Time step for fixed-step solvers; used as dt_save for Adaptive.

    Returns
    -------
    dict with keys: name, dt, runtime_s, has_nan, velocity_ms, res
    """
    # Warmup – JIT compilation happens here
    _ = solver.solve(make_axon(), tsim=TSIM, dt=dt)
    np.asarray(_.Vm)  # block until JAX finishes

    # Timed production run
    t0 = time.perf_counter()
    res = solver.solve(make_axon(), tsim=TSIM, dt=dt)
    np.asarray(res.Vm)
    elapsed = time.perf_counter() - t0

    has_nan = bool(np.any(np.isnan(np.asarray(res.Vm))))
    velocity = res.average_velocity() if not has_nan else float("nan")

    return dict(name=name, dt=dt, runtime_s=elapsed,
                has_nan=has_nan, velocity_ms=velocity, res=res)


# ── Solver configs ─────────────────────────────────────────────────────────────
# (name, solver_instance, dt)   – for Adaptive, dt is dt_save

def build_configs(dt: float) -> list[tuple[str, object, float]]:
    return [
        ("CN",           CrankNicholson(),                             dt),
        ("SemiImplicit", CrankNicholsonSemiImplicit(),                 dt),
        ("Implicit",     CrankNicholsonImplicit(n_newton=3),           dt),
        ("CrankNicholsonImplicitFast", CrankNicholsonImplicitFast(), dt),
        ("CrankNicholsonImplicitFastMultiStep", CrankNicholsonImplicitFastMultiStep(), dt),
        ("CrankNicholsonQuasiNewtonFast", CrankNicholsonQuasiNewtonFast(), dt),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    fine_data:   list[dict] = []
    coarse_data: list[dict] = []

    print("\n── Fine dt benchmark  (dt = {} ms) ──────────────────────────".format(DT_FINE))
    for name, solver, dt in build_configs(DT_FINE):
        print(f"  {name:<18} ...", end="", flush=True)
        row = run_solver(name, solver, dt)
        fine_data.append(row)
        if row["has_nan"]:
            print("  NaN!")
        else:
            print(f"  {row['runtime_s']:.3f} s   v = {row['velocity_ms']:.3f} m/s")

    print("\n── Coarse dt benchmark (dt = {} ms) ─────────────────────────".format(DT_COARSE))
    for name, solver, dt in build_configs(DT_COARSE):
        print(f"  {name:<18} ...", end="", flush=True)
        row = run_solver(name, solver, dt)
        coarse_data.append(row)
        if row["has_nan"]:
            print("  NaN!")
        else:
            print(f"  {row['runtime_s']:.3f} s   v = {row['velocity_ms']:.3f} m/s")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    ax_fine   = fig.add_subplot(gs[0, 0])
    ax_coarse = fig.add_subplot(gs[0, 1])
    ax_raster = fig.add_subplot(gs[0, 2])
    ax_time   = fig.add_subplot(gs[1, 0])
    ax_vel    = fig.add_subplot(gs[1, 1])
    ax_err    = fig.add_subplot(gs[1, 2])

    # Reference trace: CN at fine dt
    ref = fine_data[0]["res"]

    # ── Panel 1 – AP traces, fine dt ─────────────────────────────────────────
    ax_fine.set_title(f"AP traces – fine dt ({DT_FINE} ms)", fontsize=10)
    for row in fine_data:
        if row["has_nan"]:
            continue
        t  = np.asarray(row["res"].t)
        Vm = np.asarray(row["res"].Vm)[:, IDX_PLOT]
        lw = 2.5 if row["name"] == "CN" else 1.2
        ls = "-"  if row["name"] == "CN" else "--"
        ax_fine.plot(t, Vm, color=color_for(row["name"]), lw=lw, ls=ls,
                     label=row["name"])
    ax_fine.set_xlabel("Time (ms)")
    ax_fine.set_ylabel("Vm (mV)")
    ax_fine.legend(fontsize=8)
    ax_fine.grid(True, alpha=0.3)

    # ── Panel 2 – AP traces, coarse dt ───────────────────────────────────────
    ax_coarse.set_title(f"AP traces – coarse dt ({DT_COARSE} ms)", fontsize=10)
    for row in coarse_data:
        if row["has_nan"]:
            # Mark as failed in the legend
            ax_coarse.plot([], [], color=color_for(row["name"]),
                           label=f"{row['name']} [NaN]")
            continue
        t  = np.asarray(row["res"].t)
        Vm = np.asarray(row["res"].Vm)[:, IDX_PLOT]
        lw = 2.5 if row["name"] == "CN" else 1.2
        ls = "-"  if row["name"] == "CN" else "--"
        ax_coarse.plot(t, Vm, color=color_for(row["name"]), lw=lw, ls=ls,
                       label=row["name"])
    ax_coarse.set_xlabel("Time (ms)")
    ax_coarse.set_ylabel("Vm (mV)")
    ax_coarse.legend(fontsize=8)
    ax_coarse.grid(True, alpha=0.3)

    # ── Panel 3 – Raster (CN fine dt reference) ───────────────────────────────
    ax_raster.set_title(f"Raster – CN (dt = {DT_FINE} ms)", fontsize=10)
    ref.rasterplot(ax_raster)
    ax_raster.grid(True, alpha=0.3)

    # ── Panel 4 – Runtime bar chart ───────────────────────────────────────────
    names = [r["name"] for r in fine_data]
    x = np.arange(len(names))
    w = 0.35

    t_fine   = [r["runtime_s"] if not r["has_nan"] else 0.0 for r in fine_data]
    t_coarse = [r["runtime_s"] if not r["has_nan"] else 0.0 for r in coarse_data]

    bars_f = ax_time.bar(x - w / 2, t_fine,   w, label=f"dt={DT_FINE}",   alpha=0.8)
    bars_c = ax_time.bar(x + w / 2, t_coarse, w, label=f"dt={DT_COARSE}", alpha=0.8)

    for bar, row in zip(bars_f, fine_data):
        bar.set_color(color_for(row["name"]))
    for bar, row in zip(bars_c, coarse_data):
        bar.set_color(color_for(row["name"]))
        bar.set_alpha(0.45)

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(names, fontsize=9)
    ax_time.set_ylabel("Wall time (s)")
    ax_time.set_title("Runtime (after JIT warmup)", fontsize=10)
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3, axis="y")

    # ── Panel 5 – Conduction velocity ─────────────────────────────────────────
    v_fine   = [r["velocity_ms"] if not r["has_nan"] else 0.0 for r in fine_data]
    v_coarse = [r["velocity_ms"] if not r["has_nan"] else 0.0 for r in coarse_data]

    bars_vf = ax_vel.bar(x - w / 2, v_fine,   w, label=f"dt={DT_FINE}",   alpha=0.8)
    bars_vc = ax_vel.bar(x + w / 2, v_coarse, w, label=f"dt={DT_COARSE}", alpha=0.8)

    for bar, row in zip(bars_vf, fine_data):
        bar.set_color(color_for(row["name"]))
    for bar, row in zip(bars_vc, coarse_data):
        bar.set_color(color_for(row["name"]))
        bar.set_alpha(0.45)

    ax_vel.set_xticks(x)
    ax_vel.set_xticklabels(names, fontsize=9)
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.set_title("Conduction velocity", fontsize=10)
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, alpha=0.3, axis="y")

    # ── Panel 6 – MAE vs CN reference (fine dt, at IDX_PLOT) ─────────────────
    # Interpolate all fine traces onto the reference CN time grid.
    ax_err.set_title(f"MAE vs CN ref – fine dt ({DT_FINE} ms)", fontsize=10)
    t_ref = np.asarray(ref.t)
    Vm_ref = np.asarray(ref.Vm)[:, IDX_PLOT]

    mae_labels = []
    mae_values = []
    for row in fine_data[1:]:   # skip CN itself (MAE = 0)
        if row["has_nan"]:
            mae_labels.append(row["name"])
            mae_values.append(float("nan"))
            continue
        t_other  = np.asarray(row["res"].t)
        Vm_other = np.asarray(row["res"].Vm)[:, IDX_PLOT]
        # Interpolate onto ref grid (handles different t-vector conventions)
        Vm_interp = np.interp(t_ref, t_other, Vm_other)
        mae = np.mean(np.abs(Vm_interp - Vm_ref))
        mae_labels.append(row["name"])
        mae_values.append(mae)

    bar_colors = [color_for(n) for n in mae_labels]
    ax_err.bar(mae_labels, mae_values, color=bar_colors, alpha=0.8)
    ax_err.set_ylabel("MAE (mV)")
    ax_err.set_title(f"Mean |ΔVm| vs CN ref\n(at x ≈ {IDX_PLOT * L_UM / NX:.0f} µm)", fontsize=10)
    ax_err.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Crank-Nicolson solver variants – HH axon benchmark\n"
        f"L={L_UM} µm   d={D_UM} µm   Nx={NX}   T={CELSIUS}°C   tsim={TSIM} ms",
        fontsize=11,
    )

    os.makedirs("benchmark/figures", exist_ok=True)
    fig_path = "benchmark/figures/cn_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {fig_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = "benchmark/cn_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "solver", "dt_ms", "runtime_s", "has_nan", "velocity_ms"])
        for row in fine_data:
            writer.writerow(["fine",   row["name"], row["dt"],
                             f"{row['runtime_s']:.4f}", row["has_nan"],
                             f"{row['velocity_ms']:.4f}"])
        for row in coarse_data:
            writer.writerow(["coarse", row["name"], row["dt"],
                             f"{row['runtime_s']:.4f}", row["has_nan"],
                             f"{row['velocity_ms']:.4f}"])
    print(f"CSV saved   → {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
