"""Run a tiny Pint-aware axon population without NRV.

Run:
    python examples/basic/05_population_pool_run.py
"""

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # This example is the smallest population workflow:
    # one shared electrode, five axons at different y positions, and one
    # `AxonSimulation(...).run()` call that preserves the input order in the result.
    length = 100.0 * axs.um
    dt = 0.01 * axs.ms
    duration = 20.0 * axs.ms
    y_positions = np.asarray([10.0, 20.0, 60.0, 120.0, 250.0]) * axs.um

    # The electrode is centered along the axon. Its temporal current is a short
    # cathodic pulse. The transverse fiber positions below are teaching data for
    # the analytical point-source helper; they are not stored on AxonInstance.
    electrode = axs.analytical.PointSourceElectrode(
        x=length / 2.0,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    current = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.50 * axs.ms,
        amplitude=-50.0 * axs.uA,
    )

    # Build the pool explicitly so it is clear what AxonScope receives:
    # a sequence of local AxonInstance rows, not a geometry object.
    simulations = []
    for index, y_position in enumerate(y_positions):
        axon = axs.axons.RattayAberham(
            length=length,
            diameter=0.5 * axs.um,
            compartments=51,
            celsius=37.0 * axs.degC,
        )
        positions = axon.layout.position_values(unit=axs.um) * axs.um
        extracellular = axs.analytical.point_source_stimulation(
            electrode,
            positions,
            stimulus=current,
            sigma=0.3 * axs.S_per_m,
            axon_y=y_position,
            axon_z=0.0 * axs.um,
        )
        sim = axs.AxonInstance(axon)
        sim.add_extracellular_stimulation(stimulation=extracellular)
        sim.label = f"fiber {index}"
        simulations.append(sim)

    # `Recording.center(...)` keeps one Vm column per axon. That is a good first
    # population default: the output is small, but each row still has a trace.
    results = axs.AxonSimulation(
        simulations,
        duration=duration,
        dt=dt,
        recording=axs.Recording.center(axs.signals.Vm),
        progress=True,
    ).run()

    # The result is ordered like the input pool, so we can zip the positions and
    # result rows directly.
    y_positions_um = np.asarray(y_positions.to(axs.um).magnitude, dtype=float)
    z_positions_um = np.zeros_like(y_positions_um)
    peak_vm_mV = np.asarray(
        [float(res.peak_voltage_values(unit=axs.mV)[0]) for res in results]
    )

    print("Peak center Vm by row:")
    for index, (y_um, peak_mV) in enumerate(
        zip(y_positions_um, peak_vm_mV, strict=True)
    ):
        print(f"  row {index}: y={y_um:6.1f} um, peak={peak_mV:7.2f} mV")

    # The left panel shows the tiny pool geometry; color encodes the peak
    # voltage measured from the center trace. The right panel shows the retained
    # traces in the same input order.
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

    results.plot_traces(
        ax=ax_traces,
        index=0,
        labels=tuple(f"y={y_um:.0f} um" for y_um in y_positions_um),
        voltage_unit=axs.mV,
        title="AxonSimulation keeps input order",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
