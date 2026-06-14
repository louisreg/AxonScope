import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from axonscope import AxonInstance, degC, mV, ms, um
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.stimulation import Stimulus

import nrv

pytestmark = pytest.mark.nrv_numerics


def _endpoint_metrics(trace: np.ndarray) -> tuple[float, float]:
    """Return endpoint mean and excursion around the initial value."""
    baseline = float(trace[0])
    excursion = float(np.max(np.abs(trace - baseline)))
    return baseline, excursion


def test_boundary_conditions_vs_nrv(save_dir: str = "figures/physics_tests"):
    """
    Diagnostic comparison of endpoint behavior in AxonScope vs NRV.

    This test does not enforce a boundary-condition model. It saves a figure and
    prints endpoint excursions so we can inspect whether NRV behaves like a
    fixed-voltage, sealed-end, or other boundary treatment.
    """
    os.makedirs(save_dir, exist_ok=True)

    L = 300.0
    d = 1.0
    Nx = 51
    tsim = 4.0
    dt = 0.001
    t_start = 0.5
    duration = 0.5
    amplitude = 2.0

    axon = HodgkinHuxley(
        length=L * um,
        diameter=d * um,
        compartments=Nx,
        celsius=6.3 * degC,
        v_init=-70.0 * mV,
        include_passive_leak=True,
        g_pas=0.001,
        e_pas=-70.0,
    )
    sim = AxonInstance(axon)
    sim.add_current_clamp(position=(L / 2) * um, current=Stimulus.pulse(start=t_start * ms, duration=duration * ms, amplitude=amplitude))
    res_ax = CrankNicholson().solve(sim, tsim=tsim, dt=dt)

    axon_nrv = nrv.unmyelinated(
        y=0,
        z=0,
        d=d,
        L=L,
        dt=dt,
        Nsec=Nx,
        model="HH",
        v_init=axon.v_init,
        T=axon.temperature,
    )
    axon_nrv.insert_I_Clamp(0.5, t_start, duration, amplitude)
    res_nrv = axon_nrv.simulate(t_sim=tsim)

    t_nrv = np.asarray(res_nrv["t"]).ravel()
    Vm_nrv = np.asarray(res_nrv["V_mem"])

    traces = {
        "AxonScope left": np.asarray(res_ax.Vm[:, 0]),
        "AxonScope center": np.asarray(res_ax.Vm[:, Nx // 2]),
        "AxonScope right": np.asarray(res_ax.Vm[:, -1]),
        "NRV left": Vm_nrv[0, :],
        "NRV center": Vm_nrv[Nx // 2, :],
        "NRV right": Vm_nrv[-1, :],
    }

    for label in ("AxonScope left", "AxonScope right", "NRV left", "NRV right"):
        baseline, excursion = _endpoint_metrics(traces[label])
        print(f"{label}: baseline={baseline:.4f} mV, max excursion={excursion:.4f} mV")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(res_ax.t, traces["AxonScope left"], label="AxonScope left")
    axes[0].plot(t_nrv, traces["NRV left"], "--", label="NRV left")
    axes[0].set_title("Left Endpoint")
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Vm [mV]")
    axes[0].grid(True, lw=0.3)
    axes[0].legend()

    axes[1].plot(res_ax.t, traces["AxonScope center"], label="AxonScope center")
    axes[1].plot(t_nrv, traces["NRV center"], "--", label="NRV center")
    axes[1].set_title("Center Reference")
    axes[1].set_xlabel("Time [ms]")
    axes[1].grid(True, lw=0.3)
    axes[1].legend()

    axes[2].plot(res_ax.t, traces["AxonScope right"], label="AxonScope right")
    axes[2].plot(t_nrv, traces["NRV right"], "--", label="NRV right")
    axes[2].set_title("Right Endpoint")
    axes[2].set_xlabel("Time [ms]")
    axes[2].grid(True, lw=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "boundary_conditions_axonscope_vs_nrv.png"), dpi=120)
    plt.close(fig)

    assert np.isfinite(np.asarray(res_ax.Vm)).all()
    assert np.isfinite(Vm_nrv).all()
