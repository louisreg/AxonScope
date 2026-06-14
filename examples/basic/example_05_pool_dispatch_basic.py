"""Example 05: run a tiny Pint-aware axon pool without NRV.

Run:
    python examples/basic/example_05_pool_dispatch_basic.py
"""

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def make_extracellular_context(length):
    """Return one analytical context shared by every axon in the pool."""

    electrode = axs.PointSourceElectrode(
        x=length / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
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
        sim = axs.AxonInstance(
            axon,
            y=y_position,
            z=0.0 * axs.um,
        )
        sim.add_extracellular_context(context=extracellular_context)
        sim.label = f"fiber {index}"
        simulations.append(sim)
    return tuple(simulations)


def main() -> None:
    length = 100.0 * axs.um
    dt = 0.01 * axs.ms
    duration = 20.0 * axs.ms
    y_positions = np.asarray([10.0, 20.0, 60.0, 120.0, 250.0]) * axs.um

    extracellular_context, electrode = make_extracellular_context(length)
    simulations = make_simulations(
        y_positions,
        length=length,
        extracellular_context=extracellular_context,
    )

    results = axs.simulate_pool(
        simulations,
        duration=duration,
        dt=dt,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    y_positions_um = np.asarray([sim.y_um for sim in simulations], dtype=float)
    z_positions_um = np.asarray([sim.z_um for sim in simulations], dtype=float)
    peak_vm_mV = np.asarray(
        [float(res.peak_voltage_values(unit=axs.mV)[0]) for res in results]
    )

    fig, (ax_pool, ax_traces) = plt.subplots(1, 2, figsize=(10, 3.5))
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
    fig.colorbar(scatter, ax=ax_pool, label="Peak center Vm [mV]")

    for y_um, result in zip(y_positions_um, results, strict=True):
        t_ms, vm_mV = result.trace_values(
            index=0,
            time_unit=axs.ms,
            voltage_unit=axs.mV,
        )
        ax_traces.plot(t_ms, vm_mV, label=f"y={y_um:.0f} um")
    ax_traces.set_xlabel("Time [ms]")
    ax_traces.set_ylabel("Center Vm [mV]")
    ax_traces.set_title("simulate_pool keeps input order")
    ax_traces.grid(True, alpha=0.3)
    ax_traces.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
