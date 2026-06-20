"""Advanced example 18: solver-side VmRaster without stored Vm traces.

Run:
    python examples/advanced/example_18_solver_side_observers.py

This example compares a normal Vm recording with the observer-only VmRaster
path. The observer-only simulation thresholds selected Vm probes at every solver
step and returns packed bits in `result.observations` without storing `result.Vm`.
"""

from __future__ import annotations

import numpy as np

import axonscope as axs
from axonscope.solvers.observer_runtime import VM_RASTER_OBSERVATION_KEY


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

    # Step 3: describe the threshold/probe sets for the solver-side VmRaster.
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        blanking=0.0 * axs.ms,
        target=axs.positions.CENTER,
    )
    latency = axs.analysis.Latency(
        threshold=-80.0 * axs.mV,
        blanking=0.0 * axs.ms,
        target=axs.positions.CENTER,
        name="latency_center",
    )

    # Step 4: run once with a normal Vm recording so we have a post-hoc reference.
    recorded = axs.simulate(
        instance,
        duration=0.30 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    )

    # Step 5: run again without Vm storage; only packed VmRaster words are retained.
    compact = axs.simulate(
        instance,
        duration=0.30 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation, latency],
    )

    # Step 6: post-hoc analyses still consume the recorded Vm trace.
    posthoc_activation = recorded.analyze(activation)

    # Step 7: solver-side output is one packed VmRaster.
    if compact.observations is None:
        raise RuntimeError("observer-only simulation returned no observations.")
    raster = compact.observations[VM_RASTER_OBSERVATION_KEY]
    activation_raster = raster.unpack()[0, 0, 0, :]
    activation_from_raster = bool(np.any(activation_raster))
    probe_index = int(np.asarray(raster.original_indices)[0, 0])

    # Step 8: the compact result intentionally has no Vm recording.
    print(f"recorded Vm shape: {recorded.Vm.shape}")
    print(f"compact recordings: {compact.recordings}")
    print(f"raster words shape: {raster.words.shape}")

    # Step 9: higher-level analyses are post-processing of the raster.
    print(f"post-hoc activated: {bool(posthoc_activation.value)}")
    print(f"raster activated: {activation_from_raster}")

    # Step 10: use ordinary NumPy checks when you want a regression assertion.
    assert activation_from_raster == bool(posthoc_activation.value)
    np.testing.assert_array_equal(
        activation_raster,
        np.asarray(recorded.Vm)[:, probe_index] >= -80.0,
    )


if __name__ == "__main__":
    main()
