"""Estimate activation thresholds over diameter and waveform family.

Run:
    python examples/advanced/protocols/01_threshold_vs_parameters.py

This example focuses on threshold protocols only. Each curve keeps the same
diameter sweep and changes one protocol parameter: the extracellular waveform
family. The threshold search still mutates only the tested electrode current.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: define the scientific parameter grid explicitly. This advanced
    # example uses several temporal waveforms over the same diameter sweep.
    length = 1000.0 * axs.um
    diameters_um = np.asarray([0.4, 0.6, 0.8, 1.0, 1.2])
    diameters = diameters_um * axs.um
    pulse_start = 0.20 * axs.ms
    phase_duration = 0.20 * axs.ms
    interphase = 0.04 * axs.ms
    sinus_duration = 0.40 * axs.ms
    sigma = 0.3 * axs.S_per_m
    waveforms = (
        "monophasic cathodic",
        "biphasic charge-balanced",
        "linear ramp",
        "sinusoidal burst",
        "sampled triphasic",
    )

    # Step 2: the electrode is defined in the analytical helper space. The helper
    # will convert it into each axon-local context without storing world
    # coordinates on AxonInstance.
    electrode = axs.PointSourceElectrode(
        x=length / 2.0,
        y=0.0 * axs.um,
        z=80.0 * axs.um,
    )

    # Step 3: activation is evaluated at the distal recorded probe after the
    # stimulus starts.
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=pulse_start,
        target=axs.positions.DISTAL,
    )

    # Step 4: this local builder centralizes the waveform definition. Each branch
    # receives the same positive current magnitude and maps it to a temporal
    # electrode current shape.
    def build_waveform_stimulus(waveform_name: str, current_magnitude: Any) -> axs.Stimulus:
        if waveform_name == "monophasic cathodic":
            return axs.Stimulus.pulse(
                start=pulse_start,
                duration=phase_duration,
                amplitude=-current_magnitude,
            )

        if waveform_name == "biphasic charge-balanced":
            return axs.Stimulus.biphasic(
                start=pulse_start,
                cathodic_duration=phase_duration,
                cathodic_amplitude=current_magnitude,
                interphase=interphase,
            )

        if waveform_name == "linear ramp":
            current_uA = float(current_magnitude.to(axs.uA).magnitude)
            start_ms = float(pulse_start.to(axs.ms).magnitude)
            phase_ms = float(phase_duration.to(axs.ms).magnitude)
            return axs.Stimulus.from_samples(
                t=np.asarray(
                    [0.0, start_ms, start_ms + phase_ms, start_ms + phase_ms + 0.02]
                )
                * axs.ms,
                y=np.asarray([0.0, 0.0, -current_uA, 0.0]),
                mode="linear",
                unit=axs.uA,
            )

        if waveform_name == "sinusoidal burst":
            return axs.Stimulus.sinus(
                start=pulse_start,
                duration=sinus_duration,
                amplitude=current_magnitude,
                frequency_khz=5.0,
                phase=0.0,
                dt=0.005 * axs.ms,
            )

        if waveform_name == "sampled triphasic":
            current_uA = float(current_magnitude.to(axs.uA).magnitude)
            return axs.Stimulus.from_samples(
                t=np.asarray([0.0, 0.20, 0.30, 0.42, 0.58, 0.72]) * axs.ms,
                y=np.asarray([0.0, 0.0, -1.0, 0.45, -0.25, 0.0]) * current_uA,
                mode="linear",
                unit=axs.uA,
            )

        raise ValueError(f"Unknown threshold waveform: {waveform_name!r}")

    # Step 5: this local callback is the only thing the threshold protocol
    # changes during bisection. Geometry, diameter, and waveform family stay fixed.
    def update_point_source_current(
        simulation: axs.AxonInstance,
        current_magnitude: Any,
        *,
        waveform_name: str,
    ) -> None:
        context = simulation.extracellular_context
        if context is None:
            raise ValueError("simulation has no extracellular context to update.")
        context.electrodes[0].set_stimulus(
            build_waveform_stimulus(waveform_name, current_magnitude)
        )

    curves: dict[str, axs.protocols.ThresholdCurve] = {}
    preview_stimuli: dict[str, axs.Stimulus] = {}
    preview_current = 100.0 * axs.uA
    bounds_by_waveform = {
        "monophasic cathodic": (10.0 * axs.uA, 100.0 * axs.uA),
        "biphasic charge-balanced": (20.0 * axs.uA, 250.0 * axs.uA),
        "linear ramp": (20.0 * axs.uA, 250.0 * axs.uA),
        "sinusoidal burst": (20.0 * axs.uA, 250.0 * axs.uA),
        "sampled triphasic": (20.0 * axs.uA, 250.0 * axs.uA),
    }

    for waveform in waveforms:
        # Step 6: build one pool for this waveform. Each row owns its axon
        # diameter and a local point-source context initialized at zero current.
        pool: list[axs.AxonInstance] = []
        for diameter in diameters:
            axon = axs.axons.RattayAberham(
                length=length,
                diameter=diameter,
                compartments=101,
                celsius=37.0 * axs.degC,
            )
            initial_stimulus = build_waveform_stimulus(waveform, 0.0 * axs.uA)
            context = axs.analytical.local_point_source_context(
                electrode,
                stimulus=initial_stimulus,
                sigma=sigma,
            )
            simulation = axs.AxonInstance(axon)
            simulation.add_extracellular_context(context=context)
            pool.append(simulation)

        # Step 7: batched threshold search returns one threshold per row.
        curve = axs.protocols.find_activation_threshold_curve(
            tuple(pool),
            rows=diameters,
            update=lambda sim, current, name=waveform: update_point_source_current(
                sim,
                current,
                waveform_name=name,
            ),
            bounds=bounds_by_waveform[waveform],
            duration=6.0 * axs.ms,
            dt=0.01 * axs.ms,
            criterion=criterion,
            tolerance=0.01 * axs.uA,
            relative_tolerance=0.01,
            max_iterations=20,
            recording=axs.Recording.probes(axs.signals.Vm, count=9),
            progress=True,
        )
        curves[waveform] = curve
        preview_stimuli[waveform] = build_waveform_stimulus(waveform, preview_current)

        print(f"\n=== Rattay-Aberham threshold: {waveform} ===")
        for diameter_um, threshold_uA, status in zip(
            diameters_um,
            curve.threshold_uA,
            curve.status,
            strict=True,
        ):
            value = "outside range" if np.isnan(threshold_uA) else f"{threshold_uA:.1f} uA"
            print(f"d={diameter_um:>4.1f} um: {value:>14s} ({status})")

    # Step 8: plot the waveform shapes and threshold curves together so users can
    # connect the temporal drive to the activation boundary.
    fig, (ax_waveforms, ax_curves) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.0),
        constrained_layout=True,
    )
    preview_time = np.linspace(0.0, 1.0, 400) * axs.ms
    for waveform, stimulus in preview_stimuli.items():
        stimulus.plot(
            preview_time,
            ax=ax_waveforms,
            time_unit=axs.ms,
            amplitude_unit=axs.uA,
            label=waveform,
        )
    ax_waveforms.set_title("Waveforms at 100 uA")
    ax_waveforms.set_xlabel("Time [ms]")
    ax_waveforms.set_ylabel("Electrode current [uA]")
    ax_waveforms.legend(fontsize=8)

    for waveform, curve in curves.items():
        curve.plot(
            ax=ax_curves,
            row_unit=axs.um,
            threshold_unit=axs.uA,
            label=waveform,
        )
    ax_curves.set_title("Activation threshold versus waveform")
    ax_curves.set_xlabel("Diameter [um]")
    ax_curves.set_ylabel("Threshold current magnitude [uA]")
    ax_curves.legend(fontsize=8)
    plt.show()


if __name__ == "__main__":
    main()
