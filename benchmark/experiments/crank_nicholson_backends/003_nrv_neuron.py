from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyswarms.utils import Reporter as _Reporter

import settings as s
import utils as u


_LOCAL_LOG = Path(__file__).resolve().parent / "NRV.log"
_ORIG_REPORTER_INIT = _Reporter.__init__


def _patched_reporter_init(self, *args, log_path=None, **kwargs):
    if log_path and str(log_path).endswith("NRV.log"):
        log_path = str(_LOCAL_LOG)
    return _ORIG_REPORTER_INIT(self, *args, log_path=log_path, **kwargs)


_Reporter.__init__ = _patched_reporter_init

import nrv


def nrv_cache_path(Nx: int):
    from pathlib import Path

    return Path(__file__).resolve().parent / f"nrv_reference_Nx{Nx}.npz"


def run_nrv_reference(Nx: int):
    axon = nrv.unmyelinated(
        0,
        0,
        s.d,
        s.L,
        model="HH",
        Nsec=Nx,
        dt=s.dt,
        v_init=s.Vinit,
        T=6.3,
    )
    axon.insert_I_Clamp(s.position / s.L, s.t_start, s.duration, s.amplitude)

    t0 = time.perf_counter()
    results = axon.simulate(t_sim=s.tsim)
    elapsed = time.perf_counter() - t0

    t_vec = np.asarray(results["t"]).ravel()
    Vm_raw = np.asarray(results["V_mem"])
    x_rec = np.asarray(results.get("x_rec", results.get("x")))
    if Vm_raw.ndim != 2:
        raise ValueError(f"Unexpected NRV V_mem shape: {Vm_raw.shape}")
    if Vm_raw.shape[0] == t_vec.shape[0] and Vm_raw.shape[1] == x_rec.shape[0]:
        Vm = Vm_raw
    elif Vm_raw.shape[1] == t_vec.shape[0] and Vm_raw.shape[0] == x_rec.shape[0]:
        Vm = Vm_raw.T
    else:
        raise ValueError(
            f"Cannot normalize NRV V_mem shape {Vm_raw.shape} with "
            f"t={t_vec.shape[0]} and x={x_rec.shape[0]}"
        )
    x_vec = x_rec
    np.savez(nrv_cache_path(Nx), t=t_vec, Vm=Vm, x=x_vec)
    return t_vec, Vm, x_vec, elapsed


if __name__ == "__main__":
    Nx_values = list(s.Nx_v)
    timing_rows = []
    last = None
    for Nx in Nx_values:
        t_vec, Vm, x_vec, elapsed = run_nrv_reference(Nx)
        timing_rows.append(elapsed)
        print(f"Nx={Nx:<4d}  time={elapsed:.4f}s  cache={nrv_cache_path(Nx)}")
        last = (t_vec, Vm, x_vec)

    u.append_to_csv(u.res_to_df(Nx_values, timing_rows, label="nrv_neuron"))

    if last is not None:
        t_vec, Vm, x_vec = last
        fig_dir = nrv_cache_path(Nx_values[-1]).parent / "figures"
        fig_dir.mkdir(exist_ok=True)
        fig_path = fig_dir / "nrv_reference.png"
        x_positions = [s.L / 4.0, s.L / 3.0, s.L / 2.0, 2.0 * s.L / 3.0, 3.0 * s.L / 4.0]
        idx = [int(np.argmin(np.abs(x_vec - xp))) for xp in x_positions]
        fig, ax = plt.subplots(1, figsize=(6, 4))
        for i, xp in zip(idx, x_positions):
            ax.plot(t_vec, Vm[:, i], label=f"x={xp:.0f}um")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Vm (mV)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"NRV figure: {fig_path}")
