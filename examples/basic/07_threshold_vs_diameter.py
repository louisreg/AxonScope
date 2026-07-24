"""Estimate extracellular activation threshold versus diameter.

Run:
    python examples/basic/07_threshold_vs_diameter.py

Each threshold curve uses a point-source electrode and a batched binary search.
At every bisection step, AxonFleet simulates the whole diameter pool together
with one tested electrode-current amplitude per fiber.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Two pulse widths are enough to show that threshold is a protocol result,
    # not a fixed property of the axon. Shorter pulses usually need more
    # current to activate the same fiber.
    pulse_widths = (0.05 * axs.ms, 0.10 * axs.ms)
    stim_start = 0.20 * axs.ms
    sigma = 0.3 * axs.S_per_m
    temperature = 37.0 * axs.degC

    rattay_length = 1000.0 * axs.um
    rattay_diameters_um = np.concatenate(
        [
            np.linspace(0.1, 1.0, 10),
            np.asarray([1.5, 2.0]),
        ]
    )
    rattay_diameters = rattay_diameters_um * axs.um
    mrg_diameters_um = np.linspace(3.0, 20.0, 18)
    mrg_diameters = mrg_diameters_um * axs.um
    rattay_bounds_by_pulse_us = {
        50: (20.0 * axs.uA, 650.0 * axs.uA),
        100: (20.0 * axs.uA, 350.0 * axs.uA),
    }
    mrg_bounds_by_pulse_us = {
        50: (5.0 * axs.uA, 150.0 * axs.uA),
        100: (1.0 * axs.uA, 50.0 * axs.uA),
    }

    # The activation criterion says what counts as "activated" during the
    # threshold search. `DISTAL` is resolved over the positions that are actually
    # recorded; below, the MRG run records node indices so this means distal node.
    criterion = axs.analysis.Activation(
        threshold=0.0 * axs.mV,
        blanking=stim_start,
        target=axs.positions.DISTAL,
    )
    runner = axs.Runner()

    # The protocol calls this function with a candidate current amplitude at
    # every bisection step. Only the drive stimulus changes; the sampled
    # footprint and axon model stay fixed.
    def update_point_source_current(
        sim: axs.AxonInstance,
        current_magnitude: Any,
        *,
        pulse_width: Any,
    ) -> None:
        stimulation = sim.extracellular_stimulation
        if stimulation is None:
            raise ValueError("simulation has no extracellular stimulation to update.")
        drive = stimulation.drives[0]
        updated = stimulation.replace_drive(
            drive.id,
            stimulus=axs.Stimulus.pulse(
                start=stim_start,
                duration=pulse_width,
                amplitude=-current_magnitude,
            ),
        )
        sim.add_extracellular_stimulation(stimulation=updated, replace=True)

    results: dict[str, dict[float, axs.protocols.ThresholdCurve]] = {
        "Rattay-Aberham": {},
        "MRG": {},
    }
    show_cold_solver_progress = {
        "Rattay-Aberham": True,
        "MRG": True,
    }

    for pulse_width in pulse_widths:
        pulse_us = int(round(float(pulse_width.to(axs.us).magnitude)))

        # Build the unmyelinated pool for this pulse width. The initial stimulus
        # amplitude is zero because the threshold protocol will set it.
        rattay_pool = []
        for diameter in rattay_diameters:
            axon = axs.axons.RattayAberham(
                length=rattay_length,
                diameter=diameter,
                compartments=101,
                celsius=temperature,
            )
            electrode = axs.analytical.PointSourceElectrode(
                x=rattay_length / 2.0,
                y=0.0 * axs.um,
                z=100.0 * axs.um,
            )
            stimulus = axs.Stimulus.pulse(
                start=stim_start,
                duration=pulse_width,
                amplitude=0.0 * axs.uA,
            )
            positions = axon.layout.position_values(unit=axs.um) * axs.um
            extracellular = axs.analytical.point_source_stimulation(
                electrode,
                positions,
                stimulus=stimulus,
                sigma=sigma,
            )
            sim = axs.AxonInstance(axon)
            sim.add_extracellular_stimulation(stimulation=extracellular)
            rattay_pool.append(sim)

        rattay_curve = runner.run(
            axs.protocols.find_threshold(
                tuple(rattay_pool),
                rows=rattay_diameters,
                update=lambda sim, current, pw=pulse_width: update_point_source_current(
                    sim,
                    current,
                    pulse_width=pw,
                ),
                bounds=rattay_bounds_by_pulse_us[pulse_us],
                duration=6.0 * axs.ms,
                dt=0.01 * axs.ms,
                criterion=criterion,
                tolerance=0.01 * axs.uA,
                relative_tolerance=0.01,
                max_iterations=20,
                recording=axs.Recording.probes(axs.signals.Vm, count=9),
                progress=True,
                solver_progress=(
                    "plain" if show_cold_solver_progress["Rattay-Aberham"] else False
                ),
            )
        )
        show_cold_solver_progress["Rattay-Aberham"] = False
        results["Rattay-Aberham"][pulse_us] = rattay_curve

        print(f"\n=== Rattay-Aberham, PW={pulse_us:.0f} us ===")
        print(
            rattay_curve.to_dataframe(
                row_name="diameter_um",
                row_unit=axs.um,
                threshold_unit=axs.uA,
            ).to_string(index=False)
        )

        # Build the myelinated pool. Each diameter has its own MRG internode
        # spacing, so the electrode is aligned to each row's central Ranvier node.
        mrg_pool = []
        for diameter in mrg_diameters:
            axon = axs.axons.MRG(
                diameter=diameter,
                nodes=9,
                compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
            )
            electrode = axs.analytical.PointSourceElectrode(
                x=axon.node_position("center", unit=axs.um),
                y=0.0 * axs.um,
                z=100.0 * axs.um,
            )
            stimulus = axs.Stimulus.pulse(
                start=stim_start,
                duration=pulse_width,
                amplitude=0.0 * axs.uA,
            )
            positions = axon.layout.position_values(unit=axs.um) * axs.um
            extracellular = axs.analytical.point_source_stimulation(
                electrode,
                positions,
                stimulus=stimulus,
                sigma=sigma,
            )
            sim = axs.AxonInstance(axon)
            sim.add_extracellular_stimulation(stimulation=extracellular)
            mrg_pool.append(sim)

        mrg_node_indices = tuple(int(value) for value in mrg_pool[0].node_indices)
        mrg_curve = runner.run(
            axs.protocols.find_threshold(
                tuple(mrg_pool),
                rows=mrg_diameters,
                update=lambda sim, current, pw=pulse_width: update_point_source_current(
                    sim,
                    current,
                    pulse_width=pw,
                ),
                bounds=mrg_bounds_by_pulse_us[pulse_us],
                duration=5.0 * axs.ms,
                dt=0.01 * axs.ms,
                criterion=criterion,
                tolerance=0.01 * axs.uA,
                relative_tolerance=0.01,
                max_iterations=20,
                recording=axs.Recording.indices(mrg_node_indices, axs.signals.Vm),
                progress=True,
                solver_progress="plain" if show_cold_solver_progress["MRG"] else False,
            )
        )
        show_cold_solver_progress["MRG"] = False
        results["MRG"][pulse_us] = mrg_curve

        print(f"\n=== MRG, PW={pulse_us:.0f} us ===")
        print(
            mrg_curve.to_dataframe(
                row_name="diameter_um",
                row_unit=axs.um,
                threshold_unit=axs.uA,
            ).to_string(index=False)
        )

    # The plotting block deliberately reads from the same `results` dictionary
    # created above, so users can see the shape of the protocol output.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for label, diameters, ax in (
        ("Rattay-Aberham", rattay_diameters, axes[0]),
        ("MRG", mrg_diameters, axes[1]),
    ):
        for pulse_width in pulse_widths:
            pulse_us = int(round(float(pulse_width.to(axs.us).magnitude)))
            curve = results[label][pulse_us]
            curve.plot(
                ax=ax,
                row_unit=axs.um,
                threshold_unit=axs.uA,
                label=f"{pulse_us:.0f} us",
            )
        ax.set_title(label)
        ax.set_xlabel("diameter [um]")
        ax.set_ylabel("threshold current magnitude [uA]")
        ax.grid(True, alpha=0.3)
        ax.legend(title="pulse width")
    plt.show()


if __name__ == "__main__":
    main()
