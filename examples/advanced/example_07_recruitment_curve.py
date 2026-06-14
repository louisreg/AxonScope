"""Advanced example 07: activation threshold and recruitment curve.

Run:
    python examples/advanced/example_07_recruitment_curve.py

This example uses one point-source electrode and a monophasic extracellular
pulse family. The threshold search builds one simulation per tested current.
The recruitment sweep keeps a stable pool and changes only the electrode
stimulus between sampled currents.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


LENGTH = 500.0 * axs.um
ELECTRODE = axs.PointSourceElectrode(
    x=LENGTH / 2.0,
    y=0.0 * axs.um,
    z=0.0 * axs.um,
)


def make_extracellular_context(electrode_current) -> axs.AnalyticalExtracellularContext:
    """Create one extracellular context for the sampled electrode current."""

    stimulus = axs.Stimulus.pulse(
        start=0.2 * axs.ms,
        duration=0.3 * axs.ms,
        amplitude=electrode_current,
    )
    return axs.AnalyticalExtracellularContext(
        electrodes=[ELECTRODE.with_stimulus(stimulus)],
        sigma=0.3 * axs.S_per_m,
    )


def make_simulation(
    *,
    electrode_current,
    y_position=40.0 * axs.um,
) -> axs.AxonInstance:
    """Create one extracellularly stimulated unmyelinated axon."""

    axon = axs.axons.RattayAberham(
        length=LENGTH,
        diameter=0.8 * axs.um,
        compartments=51,
        celsius=37.0 * axs.degC,
    )
    sim = axs.AxonInstance(axon, y=y_position, z=0.0 * axs.um)
    sim.add_extracellular_context(
        context=make_extracellular_context(electrode_current)
    )
    return sim


def make_pool(electrode_current) -> tuple[axs.AxonInstance, ...]:
    """Create a tiny pool at different electrode distances."""

    y_positions = np.asarray([10.0, 20.0, 40.0, 80.0]) * axs.um
    return tuple(
        make_simulation(electrode_current=electrode_current, y_position=y_position)
        for y_position in y_positions
    )


def update_point_source_current(sim: axs.AxonInstance, electrode_current) -> None:
    """Change only the point-source stimulus attached to one simulation."""

    context = sim.extracellular_context
    if context is None:
        raise ValueError("simulation has no extracellular context to update.")
    electrode = context.electrodes[0]
    electrode.set_stimulus(
        axs.Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.3 * axs.ms,
            amplitude=electrode_current,
        )
    )


def main() -> None:
    criterion = axs.analysis.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=0.2 * axs.ms,
        target=axs.positions.DISTAL,
    )
    duration = 4.0 * axs.ms
    dt = 0.02 * axs.ms

    threshold = axs.protocols.find_activation_threshold(
        lambda tested_current: make_simulation(electrode_current=tested_current),
        bounds=(1.0 * axs.uA, 80.0 * axs.uA),
        duration=duration,
        dt=dt,
        criterion=criterion,
        tolerance=1.0 * axs.uA,
        max_iterations=8,
    )

    pool = make_pool(electrode_current=0.0 * axs.uA)
    curve = axs.protocols.recruitment_sweep(
        pool,
        update=update_point_source_current,
        amplitudes=np.asarray([1.0, 5.0, 10.0, 20.0, 40.0, 80.0]) * axs.uA,
        duration=duration,
        dt=dt,
        criterion=criterion,
    )

    print_threshold_summary(threshold)
    print_recruitment_summary(curve)
    plot_protocol_summary(threshold, curve)
    plt.show()


def print_threshold_summary(threshold: axs.protocols.ThresholdSearchResult) -> None:
    """Print threshold-search status."""

    print("=== Activation threshold ===")
    print(f"status={threshold.status}")
    print(f"lower={threshold.lower_bound.to(axs.uA).magnitude:.2f} uA")
    print(f"upper={threshold.upper_bound.to(axs.uA).magnitude:.2f} uA")
    if threshold.amplitude is not None:
        print(f"threshold={threshold.amplitude.to(axs.uA).magnitude:.2f} uA")


def print_recruitment_summary(curve: axs.protocols.RecruitmentCurve) -> None:
    """Print recruitment count at each sampled amplitude."""

    print("=== Recruitment sweep ===")
    for amplitude_uA, count, fraction in zip(
        curve.amplitudes_uA,
        curve.count,
        curve.fraction,
        strict=True,
    ):
        print(
            f"{amplitude_uA:.1f} uA: {int(count)} fibers ({fraction:.2f})"
        )


def plot_protocol_summary(
    threshold: axs.protocols.ThresholdSearchResult,
    curve: axs.protocols.RecruitmentCurve,
) -> None:
    """Plot threshold decisions and recruitment fraction."""

    fig, (ax_threshold, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(10, 3.6),
        constrained_layout=True,
    )
    threshold.plot(ax=ax_threshold, unit=axs.uA)
    ax_threshold.set_title("Binary threshold search")
    ax_threshold.set_xlabel("Electrode current amplitude [uA]")

    curve.plot(ax=ax_curve, unit=axs.uA)
    ax_curve.set_title("Recruitment curve")
    ax_curve.set_xlabel("Electrode current amplitude [uA]")


if __name__ == "__main__":
    main()
