"""Advanced example 18: solver-side observers without stored Vm traces.

Run:
    python examples/advanced/example_18_solver_side_observers.py

This example compares the usual post-hoc analysis path with the observer-only
path. The observer-only simulation updates compact observer state at every
solver step and returns `result.observations` without storing `result.Vm`.
"""

from __future__ import annotations

import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: build one small Hodgkin-Huxley axon.
    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=31,
        celsius=6.3 * axs.degC,
    )

    # Step 2: attach a short intracellular pulse to the axon instance.
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.10 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )

    # Step 3: describe the compact quantities we want from the solver loop.
    peak_voltage = axs.analysis.PeakVoltage(target=axs.positions.CENTER)
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        blanking=0.0 * axs.ms,
        target=axs.positions.CENTER,
    )

    # Step 4: run once with a normal Vm recording so we have a post-hoc reference.
    recorded = axs.simulate(
        instance,
        duration=0.30 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    )

    # Step 5: run again without Vm storage; only observer states are retained.
    compact = axs.simulate(
        instance,
        duration=0.30 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.none(),
        observers=[peak_voltage, activation],
    )

    # Step 6: post-hoc analyses consume the recorded Vm trace.
    posthoc_peak = recorded.analyze(peak_voltage)
    posthoc_activation = recorded.analyze(activation)

    # Step 7: solver-side observations are already finalized in the result.
    if compact.observations is None:
        raise RuntimeError("observer-only simulation returned no observations.")
    compact_peak = compact.observations["peak_voltage"]
    compact_activation = compact.observations["activation"]

    # Step 8: the compact result intentionally has no Vm recording.
    print(f"recorded Vm shape: {recorded.Vm.shape}")
    print(f"compact recordings: {compact.recordings}")

    # Step 9: both paths return the same public AnalysisResult surface.
    print(f"post-hoc peak [mV]: {posthoc_peak.value:.3f}")
    print(f"observer peak [mV]: {compact_peak.value:.3f}")
    print(f"post-hoc activated: {bool(posthoc_activation.value)}")
    print(f"observer activated: {bool(compact_activation.value)}")

    # Step 10: use ordinary NumPy checks when you want a regression assertion.
    np.testing.assert_allclose(compact_peak.values, posthoc_peak.values, rtol=1e-6)
    assert bool(compact_activation.value) == bool(posthoc_activation.value)
    assert (
        compact_activation.events[0].first_index
        == posthoc_activation.events[0].first_index
    )


if __name__ == "__main__":
    main()
