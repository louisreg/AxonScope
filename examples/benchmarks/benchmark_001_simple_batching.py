"""Example 05: compare batch and simulation-by-simulation dispatch.

The benchmark compares:

1. Batch:
       one simulate_pool(...) call containing every simulation.

2. Simulation by simulation:
       one simulate_pool(...) call per simulation, using singleton pools.

Simulation construction and plotting are excluded from the timings.

Run:
    python examples/basic/example_05_pool_dispatch_timing.py
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


REPEATS = 3


def make_extracellular_context(length):
    """Return one analytical context shared by every axon in a pool."""

    electrode = axs.PointSourceElectrode(
        x_um=length / 2.0,
        y_um=0.0 * axs.um,
        z_um=0.0 * axs.um,
    )
    current = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.50 * axs.ms,
        amplitude=-50.0 * axs.uA,
    )
    context = axs.AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(current)],
        sigma=0.3 * axs.S_per_m,
    )
    return context, electrode


def make_simulations(y_positions, *, length, extracellular_context):
    """Create one positioned simulation per y coordinate."""

    simulations = []

    for index, y_position in enumerate(y_positions):
        axon = axs.axons.RattayAberham(
            length=length,
            diameter=0.5 * axs.um,
            compartments=51,
            celsius=37.0 * axs.degC,
        )
        simulation = axs.AxonSimulation(
            axon,
            y_um=y_position,
            z_um=0.0 * axs.um,
        )
        simulation.add_extracellular_context(
            context=extracellular_context,
        )
        simulation.label = f"fiber {index}"
        simulations.append(simulation)

    return tuple(simulations)


def run_batch(
    simulations,
    *,
    duration,
    dt,
    recording,
):
    """Run every simulation in a single pool dispatch."""

    return tuple(
        axs.simulate_pool(
            simulations,
            duration_ms=duration,
            dt_ms=dt,
            recording=recording,
        )
    )


def run_sim_by_sim(
    simulations,
    *,
    duration,
    dt,
    recording,
):
    """Run simulations sequentially through singleton pool dispatches."""

    return tuple(
        axs.simulate_pool(
            (simulation,),
            duration_ms=duration,
            dt_ms=dt,
            recording=recording,
        )[0]
        for simulation in simulations
    )


def benchmark_modes(
    simulation_factory: Callable[[], Sequence],
    *,
    duration,
    dt,
    recording,
    repeats: int,
):
    """Benchmark both dispatch modes using fresh simulations each time."""

    runners = {
        "batch": run_batch,
        "sim_by_sim": run_sim_by_sim,
    }
    timings_s = {
        name: []
        for name in runners
    }
    latest_outputs = {}

    # Warm up any lazy initialization or compilation shared by both modes.
    warmup_simulations = simulation_factory()
    run_batch(
        warmup_simulations[:1],
        duration=duration,
        dt=dt,
        recording=recording,
    )

    for repeat in range(repeats):
        # Alternate execution order to reduce systematic first/second-run bias.
        mode_order = (
            ("batch", "sim_by_sim")
            if repeat % 2 == 0
            else ("sim_by_sim", "batch")
        )

        for mode_name in mode_order:
            # Construction is intentionally outside the timed section.
            simulations = simulation_factory()

            # Avoid charging unrelated pending garbage collection to one mode.
            gc.collect()

            start = perf_counter()
            results = runners[mode_name](
                simulations,
                duration=duration,
                dt=dt,
                recording=recording,
            )
            elapsed_s = perf_counter() - start

            timings_s[mode_name].append(elapsed_s)
            latest_outputs[mode_name] = (
                simulations,
                results,
            )

    timings_s = {
        name: np.asarray(values, dtype=float)
        for name, values in timings_s.items()
    }

    return latest_outputs, timings_s


def peak_center_voltages(results):
    """Extract the peak center voltage of every simulation."""

    return np.asarray(
        [
            float(result.peak_voltage_values(unit=axs.mV)[0])
            for result in results
        ],
        dtype=float,
    )


def compare_results(batch_results, sim_by_sim_results):
    """Return numerical agreement information for both execution modes."""

    if len(batch_results) != len(sim_by_sim_results):
        raise RuntimeError(
            "Batch and sim-by-sim modes returned different result counts."
        )

    batch_peaks = peak_center_voltages(batch_results)
    sim_by_sim_peaks = peak_center_voltages(sim_by_sim_results)

    peak_difference_mV = np.max(
        np.abs(batch_peaks - sim_by_sim_peaks),
        initial=0.0,
    )

    trace_difference_mV = 0.0

    for batch_result, sequential_result in zip(
        batch_results,
        sim_by_sim_results,
        strict=True,
    ):
        batch_time, batch_voltage = batch_result.trace_values(
            index=0,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
        )
        sequential_time, sequential_voltage = (
            sequential_result.trace_values(
                index=0,
                time_unit=axs.ms,
                voltage_unit=axs.mV,
            )
        )

        if batch_time.shape != sequential_time.shape:
            raise RuntimeError(
                "Batch and sim-by-sim traces have different shapes."
            )

        if not np.allclose(batch_time, sequential_time):
            raise RuntimeError(
                "Batch and sim-by-sim traces use different time samples."
            )

        current_difference = np.max(
            np.abs(batch_voltage - sequential_voltage),
            initial=0.0,
        )
        trace_difference_mV = max(
            trace_difference_mV,
            float(current_difference),
        )

    equivalent = np.allclose(
        batch_peaks,
        sim_by_sim_peaks,
        rtol=1e-6,
        atol=1e-8,
    )

    return {
        "batch_peaks_mV": batch_peaks,
        "sim_by_sim_peaks_mV": sim_by_sim_peaks,
        "max_peak_difference_mV": float(peak_difference_mV),
        "max_trace_difference_mV": trace_difference_mV,
        "equivalent": equivalent,
    }


def print_benchmark_summary(timings_s, comparison):
    """Print a compact benchmark report."""

    batch_median_s = float(np.median(timings_s["batch"]))
    sequential_median_s = float(
        np.median(timings_s["sim_by_sim"])
    )
    speedup = sequential_median_s / batch_median_s

    print()
    print("Dispatch benchmark")
    print("------------------")
    print(
        f"Batch median:       {batch_median_s * 1e3:9.3f} ms"
    )
    print(
        f"Sim-by-sim median:  {sequential_median_s * 1e3:9.3f} ms"
    )
    print(f"Batch speed-up:     {speedup:9.3f} x")
    print()
    print("Numerical comparison")
    print("--------------------")
    print(
        "Equivalent peaks:   "
        f"{comparison['equivalent']}"
    )
    print(
        "Maximum peak error: "
        f"{comparison['max_peak_difference_mV']:.3e} mV"
    )
    print(
        "Maximum trace error:"
        f" {comparison['max_trace_difference_mV']:.3e} mV"
    )

    return speedup


def plot_results(
    simulations,
    batch_results,
    *,
    electrode,
    timings_s,
    comparison,
    speedup,
):
    """Plot pool geometry, traces, and benchmark timings."""

    y_positions_um = np.asarray(
        [simulation.y_um for simulation in simulations],
        dtype=float,
    )
    z_positions_um = np.asarray(
        [simulation.z_um for simulation in simulations],
        dtype=float,
    )
    peak_vm_mV = comparison["batch_peaks_mV"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
        constrained_layout=True,
    )
    ax_pool, ax_traces, ax_timing = axes

    # Pool geometry
    scatter = ax_pool.scatter(
        z_positions_um,
        y_positions_um,
        c=peak_vm_mV,
        s=160,
        cmap="viridis",
    )
    ax_pool.scatter(
        [electrode.z_um],
        [electrode.y_um],
        marker="*",
        s=200,
        color="tab:red",
        label="electrode",
    )
    ax_pool.set_xlabel("z [um]")
    ax_pool.set_ylabel("y [um]")
    ax_pool.set_title("Pool geometry")
    ax_pool.set_aspect("equal", adjustable="datalim")
    ax_pool.legend()

    fig.colorbar(
        scatter,
        ax=ax_pool,
        label="Peak center Vm [mV]",
    )

    # Batch traces
    for y_um, result in zip(
        y_positions_um,
        batch_results,
        strict=True,
    ):
        time_ms, voltage_mV = result.trace_values(
            index=0,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
        )
        ax_traces.plot(
            time_ms,
            voltage_mV,
            label=f"y={y_um:.0f} um",
        )

    ax_traces.set_xlabel("Time [ms]")
    ax_traces.set_ylabel("Center Vm [mV]")
    ax_traces.set_title(
        "Batch results\n"
        f"max |ΔVm| = "
        f"{comparison['max_trace_difference_mV']:.2e} mV"
    )
    ax_traces.grid(True, alpha=0.3)
    ax_traces.legend()

    # Timing comparison
    labels = ["Batch", "Sim by sim"]
    timing_values_ms = [
        timings_s["batch"] * 1e3,
        timings_s["sim_by_sim"] * 1e3,
    ]

    medians_ms = np.asarray(
        [np.median(values) for values in timing_values_ms]
    )
    minimums_ms = np.asarray(
        [np.min(values) for values in timing_values_ms]
    )
    maximums_ms = np.asarray(
        [np.max(values) for values in timing_values_ms]
    )
    error_ms = np.vstack(
        (
            medians_ms - minimums_ms,
            maximums_ms - medians_ms,
        )
    )

    x_positions = np.arange(len(labels))
    ax_timing.bar(
        x_positions,
        medians_ms,
        yerr=error_ms,
        capsize=5,
    )

    for x_position, values in zip(
        x_positions,
        timing_values_ms,
        strict=True,
    ):
        ax_timing.scatter(
            np.full(values.shape, x_position),
            values,
            zorder=3,
        )

    ax_timing.set_xticks(x_positions, labels)
    ax_timing.set_ylabel("Execution time [ms]")
    ax_timing.set_title(
        f"Median over {len(timings_s['batch'])} runs\n"
        f"batch speed-up: {speedup:.2f}x"
    )
    ax_timing.grid(True, axis="y", alpha=0.3)

    plt.show()


def main() -> None:
    length = 100.0 * axs.um
    dt = 0.01 * axs.ms
    duration = 20.0 * axs.ms
    recording = axs.Recording.center("Vm")

    y_positions = (
        np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 250.0])
        * axs.um
    )

    # This context is used only to obtain the electrode for plotting.
    _, electrode = make_extracellular_context(length)

    def simulation_factory():
        # Each benchmark measurement receives a fresh context and pool.
        context, _ = make_extracellular_context(length)
        return make_simulations(
            y_positions,
            length=length,
            extracellular_context=context,
        )

    outputs, timings_s = benchmark_modes(
        simulation_factory,
        duration=duration,
        dt=dt,
        recording=recording,
        repeats=REPEATS,
    )

    batch_simulations, batch_results = outputs["batch"]
    _, sim_by_sim_results = outputs["sim_by_sim"]

    comparison = compare_results(
        batch_results,
        sim_by_sim_results,
    )
    speedup = print_benchmark_summary(
        timings_s,
        comparison,
    )

    plot_results(
        batch_simulations,
        batch_results,
        electrode=electrode,
        timings_s=timings_s,
        comparison=comparison,
        speedup=speedup,
    )


if __name__ == "__main__":
    main()