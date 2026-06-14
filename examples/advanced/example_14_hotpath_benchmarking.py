"""Advanced example 14: opt-in hotpath benchmarking.

Run:
    python examples/advanced/example_14_hotpath_benchmarking.py

This diagnostic mode times the major execution stages without changing the
simulation result. It is useful before deciding whether a refactor should
target planning, preprocessing, kernel execution, synchronization, or packaging.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import axonscope as axs


def main() -> None:
    # Step 1: keep the workload small enough for a quick local diagnostic run.
    axon = axs.axons.HodgkinHuxley(
        length=80.0 * axs.um,
        diameter=1.0 * axs.um,
        compartments=7,
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

    # Step 3: keep the public result light; the benchmark records the hotpaths.
    simulation = axs.AxonSimulation(
        axs.AxonPopulation([axon_a, axon_b]),
        duration=0.30 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    # Step 4: write diagnostic files into a temporary folder for inspection.
    output_dir = Path(tempfile.mkdtemp(prefix="axonscope-hotpaths-"))

    axs.enable_benchmark(output_dir, print_summary=False)
    results = simulation.run()
    report = axs.disable_benchmark(print_summary=True)

    print(f"results: {len(results)} axons")
    print(f"trace files: {output_dir}")
    print(f"events: {0 if report is None else len(report.events)}")


if __name__ == "__main__":
    main()
