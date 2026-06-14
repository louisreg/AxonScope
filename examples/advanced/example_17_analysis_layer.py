"""Advanced example 17: structured scientific analyses.

Run:
    python examples/advanced/example_17_analysis_layer.py

Phase 6 introduces `axs.analysis` as the public place for scientific analysis
definitions. A definition declares what it needs, then returns values with
per-axon statuses and population denominators.
"""

from __future__ import annotations

import axonscope as axs


def main() -> None:
    # Step 1: build one descriptive axon shared by two pool rows.
    axon_model = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )

    # Step 2: create a first row with a weak current pulse.
    weak = axs.AxonInstance(axon_model, y=-20.0 * axs.um)
    weak.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=0.20 * axs.nA,
        ),
    )

    # Step 3: create a second row with a stronger current pulse.
    strong = axs.AxonInstance(axon_model, y=20.0 * axs.um)
    strong.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=0.80 * axs.nA,
        ),
    )

    # Step 4: record the full membrane-voltage map because the analyses below
    # inspect selected positions after the simulation.
    recording = axs.Recording(signals=[axs.signals.Vm])

    # Step 5: run both rows as one population.
    result = axs.simulate_pool(
        [weak, strong],
        duration=0.12 * axs.ms,
        dt=0.02 * axs.ms,
        recording=recording,
    )

    # Step 6: define an activation analysis at the center compartment.
    activation = axs.analysis.Activation(
        threshold=-50.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    # Step 7: define a peak-voltage analysis over the same position.
    peak = axs.analysis.PeakVoltage(target=axs.positions.CENTER)

    # Step 8: each definition exposes its recording and algorithm contract.
    required = [signal.id.value for signal in activation.requirements.required_signals]
    print(f"activation requires: {required}")
    print(f"activation algorithm: {activation.requirements.algorithm_version}")

    # Step 9: run both analyses and keep them attached to the same result.
    report = result.report(activation, peak)

    # Step 10: every metric has explicit population denominators.
    activation_result = report["activation"]
    print(f"rows: {activation_result.population.n_total}")
    print(f"valid activation rows: {activation_result.population.n_valid}")
    print(f"failed activation rows: {activation_result.population.n_failed}")

    # Step 11: per-row values and statuses stay side by side.
    for row_index, (value, status) in enumerate(
        zip(activation_result.values, activation_result.statuses, strict=True)
    ):
        print(f"row {row_index}: activated={bool(value)} status={status.value}")

    # Step 12: the peak-voltage metric uses the same status/result shape.
    peak_result = report["peak_voltage"]
    for row_index, (value, status) in enumerate(
        zip(peak_result.values, peak_result.statuses, strict=True)
    ):
        print(f"row {row_index}: peak={float(value):.3f} mV status={status.value}")

    # Step 13: an online observer can consume Vm chunks and produce the same
    # result shape as the post-hoc definition.
    first_row = result[0]
    observer = activation.online_observer(
        positions=first_row.position_values(unit=axs.um) * axs.um,
        original_indices=first_row.record_indices,
    )

    # Step 14: here the chunks come from the recorded trace; a solver-side
    # implementation would feed the same observer during execution.
    time_ms = first_row.time_values(unit=axs.ms)
    voltage_mV = first_row.voltage_values(unit=axs.mV)
    split = len(time_ms) // 2
    observer.update(time_ms[:split] * axs.ms, voltage_mV[:split] * axs.mV)
    observer.update(time_ms[split:] * axs.ms, voltage_mV[split:] * axs.mV)

    # Step 15: compare online-style and post-hoc activation on the same row.
    online_activation = observer.finalize()
    posthoc_activation = first_row.analyze(activation)
    print(
        "online/posthoc first row: "
        f"{bool(online_activation.value)} / {bool(posthoc_activation.value)}"
    )


if __name__ == "__main__":
    main()
