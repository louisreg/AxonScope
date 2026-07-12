"""
Numerics — CN solver variants vs NRV.

test_cn_solver_smoke_traces:  visual CN traces on 3 axon types.
test_cn_fine_mesh_vs_nrv:  fine-mesh stability check (Nx=501, dx≈2µm) vs NRV.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytest

from axonscope import AxonInstance, degC, mV, ms, um
from axonscope import membranes
from axonscope.axons import Axon, Layout, Section
from axonscope.axons.unmyelinated import RattayAberham, HodgkinHuxley
from axonscope.stimulation import Stimulus
from axonscope.utils import units
from tests.nrv._helpers import run_axonscope_simulation
import nrv

pytestmark = pytest.mark.nrv_numerics


def test_cn_solver_smoke_traces(save_dir="figures/nrv_tests"):
    L, d, Nx = 1000, 1, 101
    tsim, dt = 25.0, 0.001
    t_start, duration, amplitude = 1.0, 1.0, 5

    axons = {
        "Passive": Axon(
            layout=Layout.single_uniform(
                Section(
                    "passive",
                    membrane=membranes.Passive(),
                    diameter=units.Q_(d, "micrometer"),
                ),
                length=units.Q_(L, "micrometer"),
                compartments=Nx,
            ),
        ),
        "RattayAberham": RattayAberham(length=L * um, diameter=d * um, compartments=Nx),
        "HodgkinHuxley": HodgkinHuxley(length=L * um, diameter=d * um, compartments=Nx),
    }
    x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]

    simulations = {}
    for name, axon in axons.items():
        simulation = AxonInstance(axon)
        simulation.add_current_clamp(position=(L / 2) * um, current=Stimulus.pulse(start=t_start * ms, duration=duration * ms, amplitude=amplitude))
        simulations[name] = simulation

    results = {}
    for name, simulation in simulations.items():
        results[name] = run_axonscope_simulation(simulation, tsim=tsim, dt=dt)

    fig, axs = plt.subplots(len(axons), 2, figsize=(14, 12), sharex="col")
    for i, (name, res_cn) in enumerate(results.items()):
        x_arr = np.linspace(0, L, Nx)
        indices = [np.argmin(np.abs(x_arr - xp)) for xp in x_positions]
        for idx, xp in zip(indices, x_positions):
            axs[i, 0].plot(res_cn.t, res_cn.Vm[:, idx], label=f"x={xp:.0f}µm CN")
        axs[i, 0].set_title(f"{name} — traces")
        axs[i, 0].set_ylabel("Vm [mV]")
        axs[i, 1].imshow(res_cn.Vm.T, aspect="auto", origin="lower",
                         extent=[0, tsim, 0, L], cmap="viridis")
        axs[i, 1].set_title(f"{name} — space-time (CN)")
    axs[-1, 0].set_xlabel("Time [ms]")
    axs[-1, 1].set_xlabel("Time [ms]")
    fig.tight_layout()
    import os; os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/compare_three_axons_CN.png")
    plt.close(fig)


def _is_unstable(Vm: np.ndarray) -> bool:
    return bool(np.any(Vm < -150.0) or np.any(Vm > 100.0))


@pytest.mark.nrv_numerics
def test_cn_fine_mesh_vs_nrv(save_dir="figures/nrv_tests"):
    """Fine-mesh stability check (Nx=501, dx≈2µm) — AxonScope vs NRV."""
    L, d, Nx, tsim, dt = 1000, 1.0, 501, 10.0, 0.001
    t_start, duration, amplitude = 1.0, 1.0, 5.0
    x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]

    axon_ra = RattayAberham(length=L * um, diameter=d * um, compartments=Nx, celsius=37.0 * degC)
    axon_hh = HodgkinHuxley(length=L * um, diameter=d * um, compartments=Nx, celsius=6.3 * degC, v_init=-70.0 * mV,
                             include_passive_leak=True, g_pas=0.001, e_pas=-70.0)
    sim_ra = AxonInstance(axon_ra)
    sim_hh = AxonInstance(axon_hh)
    for sim in (sim_ra, sim_hh):
        sim.add_current_clamp(position=(L / 2) * um, current=Stimulus.pulse(start=t_start * ms, duration=duration * ms, amplitude=amplitude))

    res_ra = run_axonscope_simulation(sim_ra, tsim=tsim, dt=dt)
    res_hh = run_axonscope_simulation(sim_hh, tsim=tsim, dt=dt)

    nrv_ra = nrv.unmyelinated(0, 0, d, L, dt=dt, Nsec=Nx, V_init=axon_ra.v_init, T=axon_ra.temperature)
    nrv_ra.insert_I_Clamp(0.5, t_start, duration, amplitude)
    res_nrv_ra = nrv_ra.simulate(t_sim=tsim)

    nrv_hh = nrv.unmyelinated(0, 0, d, L, dt=dt, Nsec=Nx, model="HH",
                               v_init=axon_hh.v_init, T=axon_hh.temperature)
    nrv_hh.insert_I_Clamp(0.5, t_start, duration, amplitude)
    res_nrv_hh = nrv_hh.simulate(t_sim=tsim)

    checks = {
        "AxonScope RattayAberham": _is_unstable(np.array(res_ra.Vm)),
        "AxonScope HodgkinHuxley": _is_unstable(np.array(res_hh.Vm)),
        "NRV RattayAberham":       _is_unstable(res_nrv_ra["V_mem"].T),
        "NRV HodgkinHuxley":       _is_unstable(res_nrv_hh["V_mem"].T),
    }
    for label, unstable in checks.items():
        print(f"  {label}: {'UNSTABLE' if unstable else 'stable'}")
