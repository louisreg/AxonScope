"""Advanced example 14: hotpath benchmarking for observer-only runs.

Run:
    python examples/advanced/example_14_hotpath_benchmarking.py

This diagnostic mode estimates array memory before execution, then times the
major execution stages. Here the simulation keeps a packed solver-side VmRaster
instead of storing the full Vm[time, position] trace.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import axonscope as axs
from axonscope.solvers.observer_runtime import VM_RASTER_OBSERVATION_KEY


def main() -> None:
    # Step 1: keep the workload small enough for a quick local diagnostic run.
    axon = axs.axons.HodgkinHuxley(
        length=80.0 * axs.um,
        diameter=1.0 * axs.um,
        compartments=51,
        celsius=6.3 * axs.degC,
    )

    # Step 2: build two rows so the pool dispatcher can use a batch path.
    axon_a = axs.AxonInstance(axon)
    axon_a.add_current_clamp(
        position=40.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.10 * axs.ms,
            duration=0.10 * axs.ms,
            amplitude=0.6 * axs.nA,
        ),
    )

    axon_b = axs.AxonInstance(axon)
    axon_b.add_current_clamp(
        position=40.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.10 * axs.ms,
            duration=0.10 * axs.ms,
            amplitude=0.9 * axs.nA,
        ),
    )

    # Step 3: choose threshold/probe definitions for the solver-side VmRaster.
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    latency = axs.analysis.Latency(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
        name="latency_center",
    )

    # Step 4: request no stored traces; only packed VmRaster words are retained.
    simulation = axs.AxonSimulation(
        axs.AxonPopulation([axon_a, axon_b]),
        duration=0.30 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation, latency],
    )

    # Step 5: inspect the estimated array footprint before running anything.
    estimate = simulation.estimate(
        runtime=axs.Runtime.JAX,
        device=axs.Device.auto(),
        precision=axs.PrecisionPolicy.float32(),
    )
    print(estimate.format())
    print(f"intracellular input: {estimate.metadata['intracellular_input_format']}")

    # Step 6: write diagnostic files into a temporary folder for inspection.
    output_dir = Path(tempfile.mkdtemp(prefix="axonscope-hotpaths-"))

    # Step 7: run with benchmark instrumentation enabled around the hotpaths.
    axs.enable_benchmark(output_dir, print_summary=False)
    results = simulation.run()
    report = axs.disable_benchmark(print_summary=True)

    # Step 8: read the packed VmRaster directly from the pool result.
    observations = results.observations
    if observations is None:
        raise RuntimeError("observer-only run did not return observations.")
    raster = observations[VM_RASTER_OBSERVATION_KEY]
    unpacked = raster.unpack()
    activated = np.any(unpacked[:, 0, 0, :], axis=1)

    print(f"results: {len(results)} axons")
    print(f"raster names: {raster.names}")
    print(f"raster words shape: {raster.words.shape}")
    print(f"activated from raster: {activated}")
    print(f"trace files: {output_dir}")
    print(f"events: {0 if report is None else len(report.events)}")
    print(f"estimated retained Vm: {estimate.retained_mib:.3f} MiB")


if __name__ == "__main__":
    main()
