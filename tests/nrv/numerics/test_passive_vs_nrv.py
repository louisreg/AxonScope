from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nrv
import numpy as np

from axonfleet import AxonInstance, membranes, ms, um
from axonfleet.axons import Axon, Layout, Section
from axonfleet.stimulation import Stimulus
from axonfleet.utils import units
from tests.nrv._helpers import run_axonfleet_simulation


def test_passive_current_definition_matches_nrv_pas(save_dir="figures/physics_tests"):
    length_um = 1_000.0
    diameter_um = 5.0
    compartments = 101
    capacitance_uF_cm2 = 10.0
    leak_S_cm2 = 1e-4
    reversal_mV = -70.0
    axial_resistivity_ohm_cm = 100.0
    current_nA = 10.0
    start_ms = 0.5
    duration_ms = 0.5
    tsim_ms = 4.0
    dt_ms = 0.01

    axon = Axon(
        layout=Layout.single_uniform(
            Section(
                "passive",
                membrane=membranes.Passive(Rm=1.0 / leak_S_cm2, EL=reversal_mV),
                diameter=units.Q_(diameter_um, "micrometer"),
                Ra=units.Q_(axial_resistivity_ohm_cm, "ohm * centimeter"),
                Cm=units.Q_(capacitance_uF_cm2, "microfarad / centimeter ** 2"),
            ),
            length=units.Q_(length_um, "micrometer"),
            compartments=compartments,
        ),
        v_init=units.Q_(reversal_mV, "millivolt"),
    )
    simulation = AxonInstance(axon)
    simulation.add_current_clamp(
        position=0.5 * length_um * um,
        current=Stimulus.pulse(
            start=start_ms * ms,
            duration=duration_ms * ms,
            amplitude=current_nA,
        ),
    )
    result = run_axonfleet_simulation(
        simulation,
        tsim=tsim_ms,
        dt=dt_ms,
        record_observables=True,
    )

    axon_nrv = nrv.unmyelinated(
        0,
        0,
        diameter_um,
        length_um,
        model="HH",
        dt=dt_ms,
        Nsec=1,
        Nseg_per_sec=compartments,
        v_init=reversal_mV,
        T=32.0,
        include_passive_leak=True,
        g_pas=leak_S_cm2,
        e_pas=reversal_mV,
    )
    for section in axon_nrv.unmyelinated_sections:
        section.gnabar_hh = 0.0
        section.gkbar_hh = 0.0
        section.gl_hh = 0.0
        section.g_pas = leak_S_cm2
        section.e_pas = reversal_mV
        section.cm = capacitance_uF_cm2
        section.Ra = axial_resistivity_ohm_cm
    axon_nrv.insert_I_Clamp(0.5, start_ms, duration_ms, current_nA)
    axon_nrv.record_V_mem = True
    reference = axon_nrv.simulate(t_sim=tsim_ms)

    t_as = np.asarray(result.t, dtype=float)
    t_nrv = np.asarray(reference["t"], dtype=float)
    vm_nrv = np.asarray(reference["V_mem"], dtype=float)
    center_as = compartments // 2
    center_nrv = vm_nrv.shape[0] // 2
    vm_reference = np.interp(t_as, t_nrv, vm_nrv[center_nrv])
    current_reference = leak_S_cm2 * (vm_reference - reversal_mV)
    assert result.recordings is not None
    current_as = 1e-3 * np.asarray(result.recordings["currents"]["I_l"][:, center_as])

    expected_current_as = leak_S_cm2 * (result.Vm[:, center_as] - reversal_mV)
    np.testing.assert_allclose(current_as, expected_current_as, rtol=2e-5, atol=2e-6)
    assert all(section.g_pas == leak_S_cm2 for section in axon_nrv.unmyelinated_sections)
    assert all(section.e_pas == reversal_mV for section in axon_nrv.unmyelinated_sections)

    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t_as, result.Vm[:, center_as], label="AxonFleet")
    axes[0].plot(t_as, vm_reference, "--", label="NRV passive reference")
    axes[0].set_ylabel("Vm [mV]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_as, current_as, label="AxonFleet")
    axes[1].plot(t_as, current_reference, "--", label="NRV pas equation")
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("I_l [mA/cm2]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "axon_compare_passive_nrv.png", dpi=150)
    plt.close(figure)
