"""Work with canonical pool results.

Run:
    python examples/advanced/simulation_workflow/03_pool_results.py

Pool simulations return one canonical result object instead of a plain list.
Indexing or iterating over that object gives one-axon views in the original
input order.
"""

from __future__ import annotations

import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: build one descriptive axon model shared by two concrete rows.
    axon_model = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )

    # Step 2: create the first executable axon instance.
    axon_a = axs.AxonInstance(axon_model)

    # Step 3: give the first row a small central current pulse.
    axon_a.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=0.30 * axs.nA,
        ),
    )

    # Step 4: create a second row from the same axon model.
    axon_b = axs.AxonInstance(axon_model)

    # Step 5: give the second row a stronger pulse.
    axon_b.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=0.60 * axs.nA,
        ),
    )

    # Step 6: ask the pool run to keep only the center voltage trace.
    recording = axs.Recording.center(axs.signals.Vm)

    # Step 7: run the two rows as one pool.
    results = axs.AxonSimulation(
        [axon_a, axon_b],
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=recording,
    ).run()

    # Step 8: the top-level result is not a list.
    print(f"result type: {type(results).__name__}")
    print(f"rows: {len(results)}")
    print(f"recording rows: {len(results.recordings)}")
    print(f"final states: {sum(state is not None for state in results.final_states)} retained")

    # Step 9: the recording manifest says what signals are actually available.
    manifest = results.recording_manifest
    print(f"requested signals: {[signal.id.value for signal in manifest.requested_signals]}")
    print(f"available signals: {[signal.id.value for signal in manifest.available_signals]}")

    # Step 10: indexing returns a one-axon view in the original pool order.
    first = results[0]
    second = results[1]
    print(f"first view input index: {first.index}")
    print(f"second view input index: {second.index}")

    # Step 11: the view still behaves like a familiar single-axon result.
    time_ms, first_trace_mV = first.trace_values(index=0)
    _, second_trace_mV = second.trace_values(index=0)
    print(f"time samples: {time_ms.tolist()} ms")
    print(f"first peak: {float(np.max(first_trace_mV)):.3f} mV")
    print(f"second peak: {float(np.max(second_trace_mV)):.3f} mV")

    # Step 12: homogeneous recordings can also expose one dense signal block.
    dense_vm = results.signal(axs.signals.Vm)
    print(f"dense Vm shape: {dense_vm.shape}  # axon, time, recorded_position")

    # Step 13: the view is the public one-axon result surface.
    print(f"second simulation is original row: {second.simulation is axon_b}")


if __name__ == "__main__":
    main()
