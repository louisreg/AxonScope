"""Example 07: extracellular activation threshold versus diameter.

Run:
    python examples/basic/example_07_threshold_vs_diameter.py

Each threshold curve uses a point-source electrode and a batched binary search.
At every bisection step, AxonScope simulates the whole diameter pool together
with one tested electrode-current amplitude per fiber.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


PULSE_WIDTHS = (0.05 * axs.ms, 0.10 * axs.ms)
STIM_START = 0.20 * axs.ms
SIGMA = 0.3 * axs.S_per_m
TEMPERATURE = 37.0 * axs.degC

RATTAY_LENGTH = 1000.0 * axs.um
RATTAY_DIAMETERS = np.asarray([0.5, 0.8, 1.1, 1.5, 2.0]) * axs.um
MRG_DIAMETERS = np.asarray([5.7, 7.3, 10.0, 12.8, 15.0]) * axs.um


def point_source_context(
    *,
    length: Any,
    pulse_width: Any,
    electrode_z: Any,
) -> axs.AnalyticalExtracellularContext:
    """Create one cathodic point-source context."""

    electrode = axs.PointSourceElectrode(
        x=length / 2.0,
        y=0.0 * axs.um,
        z=electrode_z,
    )
    stimulus = axs.Stimulus.pulse(
        start=STIM_START,
        duration=pulse_width,
        amplitude=0.0 * axs.uA,
    )
    return axs.AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(stimulus)],
        sigma=SIGMA,
    )


def make_rattay_simulation(
    diameter: Any,
    *,
    pulse_width: Any,
) -> axs.AxonInstance:
    """Create one Rattay-Aberham fiber."""

    axon = axs.axons.RattayAberham(
        length=RATTAY_LENGTH,
        diameter=diameter,
        compartments=101,
        celsius=TEMPERATURE,
    )
    sim = axs.AxonInstance(axon)
    sim.add_extracellular_context(
        context=point_source_context(
            length=RATTAY_LENGTH,
            pulse_width=pulse_width,
            electrode_z=100.0 * axs.um,
        )
    )
    return sim


def make_mrg_simulation(
    diameter: Any,
    *,
    pulse_width: Any,
) -> axs.AxonInstance:
    """Create one MRG fiber."""

    axon = axs.axons.MRG(
        diameter=diameter,
        nodes=9,
        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
    )
    sim = axs.AxonInstance(axon)
    sim.add_extracellular_context(
        context=point_source_context(
            length=axon.length * axs.um,
            pulse_width=pulse_width,
            electrode_z=100.0 * axs.um,
        )
    )
    return sim


def update_point_source_current(
    sim: axs.AxonInstance,
    current_magnitude: Any,
    *,
    pulse_width: Any,
) -> None:
    """Change only the point-source stimulus attached to one simulation."""

    context = sim.extracellular_context
    if context is None:
        raise ValueError("simulation has no extracellular context to update.")
    electrode = context.electrodes[0]
    electrode.set_stimulus(
        axs.Stimulus.pulse(
            start=STIM_START,
            duration=pulse_width,
            amplitude=-current_magnitude,
        )
    )


def print_curve(
    label: str,
    pulse_width: Any,
    diameters: Any,
    curve: axs.protocols.ThresholdCurve,
) -> None:
    """Print one threshold curve."""

    print(f"\n=== {label}, PW={pulse_width.to(axs.us).magnitude:.0f} us ===")
    for diameter, threshold_uA, status in zip(
        diameters.to(axs.um).magnitude,
        curve.threshold_uA,
        curve.status,
        strict=True,
    ):
        value = "outside range" if np.isnan(threshold_uA) else f"{threshold_uA:.1f} uA"
        print(f"d={diameter:>5.2f} um: {value:>14s} ({status})")


def plot_curves(results: dict[str, dict[float, axs.protocols.ThresholdCurve]]) -> None:
    """Plot threshold versus diameter."""

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    model_data = {
        "Rattay-Aberham": (RATTAY_DIAMETERS, axes[0]),
        "MRG": (MRG_DIAMETERS, axes[1]),
    }
    for label, (diameters, ax) in model_data.items():
        for pulse_width in PULSE_WIDTHS:
            pulse_us = float(pulse_width.to(axs.us).magnitude)
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


def main() -> None:
    criterion = axs.results.ActivationCriterion(
        threshold=0.0 * axs.mV,
        blanking=STIM_START,
        target=axs.positions.DISTAL,
    )
    results: dict[str, dict[float, axs.protocols.ThresholdCurve]] = {
        "Rattay-Aberham": {},
        "MRG": {},
    }

    for pulse_width in PULSE_WIDTHS:
        pulse_us = float(pulse_width.to(axs.us).magnitude)

        rattay_pool = tuple(
            make_rattay_simulation(diameter, pulse_width=pulse_width)
            for diameter in RATTAY_DIAMETERS
        )
        rattay_curve = axs.protocols.find_activation_threshold_curve(
            rattay_pool,
            rows=RATTAY_DIAMETERS,
            update=lambda sim, current, pw=pulse_width: update_point_source_current(
                sim,
                current,
                pulse_width=pw,
            ),
            bounds=(20.0 * axs.uA, 250.0 * axs.uA),
            duration=6.0 * axs.ms,
            dt=0.01 * axs.ms,
            criterion=criterion,
            tolerance = 0.01,
            relative_tolerance=0.01,
            max_iterations=20,
            recording=axs.Recording.probes(axs.signals.Vm, count=9),
            progress=True,
        )
        results["Rattay-Aberham"][pulse_us] = rattay_curve
        print_curve("Rattay-Aberham", pulse_width, RATTAY_DIAMETERS, rattay_curve)

        mrg_pool = tuple(
            make_mrg_simulation(diameter, pulse_width=pulse_width)
            for diameter in MRG_DIAMETERS
        )
        mrg_curve = axs.protocols.find_activation_threshold_curve(
            mrg_pool,
            rows=MRG_DIAMETERS,
            update=lambda sim, current, pw=pulse_width: update_point_source_current(
                sim,
                current,
                pulse_width=pw,
            ),
            bounds=(5.0 * axs.uA, 100.0 * axs.uA),
            duration=5.0 * axs.ms,
            dt=0.01 * axs.ms,
            criterion=criterion,
            tolerance = 0.01,
            relative_tolerance=0.01,
            max_iterations=20,
            recording=axs.Recording.probes(axs.signals.Vm, count=9),
            progress=True,
        )
        results["MRG"][pulse_us] = mrg_curve
        print_curve("MRG", pulse_width, MRG_DIAMETERS, mrg_curve)

    plot_curves(results)


if __name__ == "__main__":
    main()
