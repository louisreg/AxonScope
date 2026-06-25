"""Compare recruitment curves for several extracellular waveforms.

Run:
    python examples/advanced/protocols/02_recruitment_waveforms.py

This example focuses on recruitment protocols only. The axon pool is fixed; the
protocol sweeps current magnitude for several temporal waveform families.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: build a tiny cohort at increasing point-source distances. These
    # transverse offsets are teaching inputs to the analytical helper, not
    # coordinates stored on AxonInstance.
    length = 500.0 * axs.um
    y_positions = np.asarray([12.0, 24.0, 40.0, 70.0, 110.0, 160.0]) * axs.um
    pulse_start = 0.20 * axs.ms
    phase_duration = 0.20 * axs.ms
    interphase = 0.04 * axs.ms
    sinus_duration = 0.40 * axs.ms
    sigma = 0.3 * axs.S_per_m
    amplitudes = np.asarray([2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 100.0, 150.0, 200.0]) * axs.uA
    waveforms = (
        "monophasic cathodic",
        "biphasic charge-balanced",
        "linear ramp",
        "sinusoidal burst",
        "sampled triphasic",
    )

    electrode = axs.analytical.PointSourceElectrode(
        x=length / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )

    # Step 2: activation at any recorded position counts as recruitment for this
    # population-level view.
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=pulse_start,
        target=axs.positions.ALL,
    )

    curves: dict[str, axs.protocols.RecruitmentCurve] = {}
    preview_stimuli: dict[str, axs.Stimulus] = {}
    show_cold_solver_progress = True

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
        raise ValueError(f"Unknown recruitment waveform: {waveform_name!r}")

    for waveform in waveforms:
        # Step 3: rebuild an identical local pool for each waveform. This keeps
        # the mutable electrode stimulus history separate between protocol runs.
        pool: list[axs.AxonInstance] = []
        for y_position in y_positions:
            axon = axs.axons.RattayAberham(
                length=length,
                diameter=0.8 * axs.um,
                compartments=51,
                celsius=37.0 * axs.degC,
            )
            positions = axon.layout.position_values(unit=axs.um) * axs.um
            stimulation = axs.analytical.point_source_stimulation(
                electrode,
                positions,
                stimulus=axs.Stimulus.constant(0.0 * axs.uA),
                sigma=sigma,
                axon_y=y_position,
            )
            simulation = axs.AxonInstance(axon)
            simulation.add_extracellular_stimulation(stimulation=stimulation)
            pool.append(simulation)

        # Step 4: recruitment_sweep calls this callback once per row and sampled
        # amplitude. Only the temporal waveform changes; the amplitude samples
        # keep the same positive "current magnitude" meaning for all waveforms.
        def update_waveform_current(
            simulation: axs.AxonInstance,
            current_magnitude: Any,
            *,
            waveform_name: str,
        ) -> None:
            stimulation = simulation.extracellular_stimulation
            if stimulation is None:
                raise ValueError("simulation has no extracellular stimulation to update.")
            drive = stimulation.drives[0]
            updated = stimulation.replace_drive(
                drive.id,
                stimulus=build_waveform_stimulus(waveform_name, current_magnitude),
            )
            simulation.add_extracellular_stimulation(stimulation=updated, replace=True)

        curve = axs.protocols.recruitment_sweep(
            tuple(pool),
            update=lambda sim, current, name=waveform: update_waveform_current(
                sim,
                current,
                waveform_name=name,
            ),
            amplitudes=amplitudes,
            duration=3.0 * axs.ms,
            dt=0.02 * axs.ms,
            criterion=criterion,
            recording=axs.Recording.probes(axs.signals.Vm, count=5),
            progress=True,
            solver_progress="plain" if show_cold_solver_progress else False,
        )
        show_cold_solver_progress = False
        curves[waveform] = curve
        preview_stimuli[waveform] = build_waveform_stimulus(waveform, amplitudes[-1])

        print(f"\n=== Recruitment: {waveform} ===")
        for amplitude_uA, count, fraction in zip(
            curve.amplitudes_uA,
            curve.count,
            curve.fraction,
            strict=True,
        ):
            print(f"{amplitude_uA:>5.1f} uA: {int(count)}/{len(pool)} fibers ({fraction:.2f})")

    # Step 5: plot the waveform shapes and recruitment curves together so users
    # can connect the temporal drive to the cohort response.
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
    ax_waveforms.set_title("Waveforms at last tested amplitude")
    ax_waveforms.set_xlabel("Time [ms]")
    ax_waveforms.set_ylabel("Electrode current [uA]")
    ax_waveforms.legend(fontsize=8)

    for waveform, curve in curves.items():
        curve.plot(ax=ax_curves, unit=axs.uA, label=waveform)
    ax_curves.set_title("Recruitment versus waveform")
    ax_curves.set_xlabel("Current magnitude [uA]")
    ax_curves.legend(fontsize=8)
    plt.show()


if __name__ == "__main__":
    main()
