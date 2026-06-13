"""Advanced example 05: recording policies and observable groups.

Run:
    python examples/advanced/example_05_recording_options.py

Single-axon runs can record Vm plus observable groups such as gates, currents,
and conductances. Pool runs currently record Vm only, but can retain all
compartments, the center compartment, evenly spaced probes, or explicit
compartment indices.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def make_hh_simulation(*, y_um=0.0 * axs.um) -> axs.AxonSimulation:
    """Create a small Hodgkin-Huxley simulation with one current clamp."""

    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    sim = axs.AxonSimulation(axon, y_um=y_um)
    sim.add_current_clamp(
        position_um=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    return sim


def main() -> None:
    single_result = axs.simulate(
        make_hh_simulation(),
        duration_ms=2.0 * axs.ms,
        dt_ms=0.02 * axs.ms,
        recording=axs.Recording.full(),
    )
    gates_only = axs.simulate(
        make_hh_simulation(),
        duration_ms=0.2 * axs.ms,
        dt_ms=0.05 * axs.ms,
        recording=axs.Recording.only("Vm", "gates"),
    )

    pool = [
        make_hh_simulation(y_um=0.0 * axs.um),
        make_hh_simulation(y_um=25.0 * axs.um),
        make_hh_simulation(y_um=50.0 * axs.um),
    ]
    recording_modes = {
        "full": axs.Recording.voltage(),
        "center": axs.Recording.center("Vm"),
        "probes": axs.Recording.probes("Vm", count=5),
        "indices": axs.Recording.indices([0, 10, 20], "Vm"),
    }
    pool_results = {
        label: axs.simulate_pool(
            pool,
            duration_ms=0.2 * axs.ms,
            dt_ms=0.05 * axs.ms,
            recording=recording,
        )
        for label, recording in recording_modes.items()
    }

    print_recording_summary(single_result, gates_only, pool_results)
    plot_recording_summary(single_result, pool_results)
    plt.show()


def print_recording_summary(
    full_result: axs.SimResult,
    gates_only: axs.SimResult,
    pool_results: dict[str, list[axs.SimResult]],
) -> None:
    """Print the observable groups and retained Vm widths."""

    full_groups = summarize_recordings(full_result)
    gate_groups = summarize_recordings(gates_only)
    print("=== Single-axon observable groups ===")
    print(f"Recording.full(): {full_groups}")
    print(f"Recording.only('Vm', 'gates'): {gate_groups}")

    print("=== Pool Vm recording widths ===")
    for label, results in pool_results.items():
        first = results[0]
        print(
            f"{label:>7}: Vm shape={np.asarray(first.Vm).shape} "
            f"record_indices={first.record_indices}"
        )


def summarize_recordings(result: axs.SimResult) -> dict[str, str | tuple[str, ...]]:
    """Return compact labels for recorded variables."""

    summary: dict[str, str | tuple[str, ...]] = {}
    for name, value in (result.recordings or {}).items():
        if isinstance(value, dict):
            summary[name] = tuple(value)
        else:
            summary[name] = str(np.asarray(value).shape)
    return summary


def plot_recording_summary(
    full_result: axs.SimResult,
    pool_results: dict[str, list[axs.SimResult]],
) -> None:
    """Plot one Vm trace, local observables, and pool retained widths."""

    center_index = full_result.nearest_position_index(50.0 * axs.um)
    t_ms = full_result.time_values(unit=axs.ms)
    recordings = full_result.recordings or {}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax_vm, ax_gates, ax_currents, ax_widths = axes.ravel()

    full_result.plot_trace(
        ax=ax_vm,
        index=center_index,
        voltage_unit=axs.mV,
        title="Center Vm",
    )

    for name, values in recordings.get("gates", {}).items():
        ax_gates.plot(t_ms, np.asarray(values)[:, center_index], label=name)
    ax_gates.set_title("Center gates")
    ax_gates.set_xlabel("Time [ms]")
    ax_gates.set_ylabel("Gate value")
    ax_gates.grid(True, alpha=0.3)
    ax_gates.legend(frameon=False)

    for name, values in recordings.get("currents", {}).items():
        ax_currents.plot(t_ms, np.asarray(values)[:, center_index], label=name)
    ax_currents.set_title("Center current densities")
    ax_currents.set_xlabel("Time [ms]")
    ax_currents.set_ylabel("Current density [mA/cm2]")
    ax_currents.grid(True, alpha=0.3)
    ax_currents.legend(frameon=False)

    labels = tuple(pool_results)
    widths = [np.asarray(results[0].Vm).shape[1] for results in pool_results.values()]
    ax_widths.bar(labels, widths, color=["0.45", "C0", "C2", "C3"])
    ax_widths.set_title("Pool retained Vm columns")
    ax_widths.set_ylabel("Recorded compartments")
    ax_widths.set_ylim(0, max(widths) + 2)
    ax_widths.grid(True, axis="y", alpha=0.3)


if __name__ == "__main__":
    main()
