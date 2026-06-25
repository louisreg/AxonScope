"""Build a non-uniform axon discretization from an activation-function proxy.

Run:
    python examples/advanced/axon_models/04_non_uniform_activation_function.py

Non-uniform layouts are ordinary descriptive layouts with explicit compartment
centers. This example derives those centers from a point-source activating
function proxy: compartments are denser where the sampled extracellular
footprint has the strongest spatial curvature.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonscope as axs


def main() -> None:
    # Step 1: define a one-dimensional axon-local problem. The point-source
    # helper is only used to sample a footprint over intrinsic axon positions.
    length = 1000.0 * axs.um
    diameter = 1.0 * axs.um
    sigma = 0.3 * axs.S_per_m
    n_compartments = 61
    electrode = axs.analytical.PointSourceElectrode(
        x=500.0 * axs.um,
        z=90.0 * axs.um,
    )

    # Step 2: sample a fine reference footprint and compute a Rattay-style
    # activation-function proxy. The proxy is only a meshing heuristic here; it
    # is not an activation criterion.
    fine_x_um = np.linspace(0.0, float(length.to(axs.um).magnitude), 1201)
    footprint_V_per_A = electrode.footprint(
        fine_x_um * 1e-6,
        sigma_S_m=float(sigma.to(axs.S_per_m).magnitude),
    )
    footprint_mV_per_uA = footprint_V_per_A * 1e-3
    activation_proxy = np.gradient(
        np.gradient(footprint_mV_per_uA, fine_x_um),
        fine_x_um,
    )
    activation_weight = np.abs(activation_proxy)
    activation_weight /= np.max(activation_weight)

    # Step 3: convert the proxy into a mesh density. The baseline term prevents
    # the tails from becoming too coarse; the curvature term pulls compartments
    # toward the electrode.
    mesh_density = 1.0 + 9.0 * np.sqrt(activation_weight)
    cumulative_density = np.zeros_like(fine_x_um)
    cumulative_density[1:] = np.cumsum(
        0.5
        * (mesh_density[:-1] + mesh_density[1:])
        * np.diff(fine_x_um)
    )
    adaptive_edges_um = np.interp(
        np.linspace(0.0, cumulative_density[-1], n_compartments + 1),
        cumulative_density,
        fine_x_um,
    )
    adaptive_centers_um = 0.5 * (
        adaptive_edges_um[:-1] + adaptive_edges_um[1:]
    )

    uniform_edges_um = np.linspace(0.0, float(length.to(axs.um).magnitude), n_compartments + 1)
    uniform_centers_um = 0.5 * (uniform_edges_um[:-1] + uniform_edges_um[1:])

    # Step 4: build two otherwise identical axons. The non-uniform model uses
    # the public `x=` constructor, which maps to `Layout.single_non_uniform`.
    uniform_axon = axs.axons.RattayAberham(
        length=length,
        diameter=diameter,
        compartments=n_compartments,
        celsius=37.0 * axs.degC,
    )
    adaptive_axon = axs.axons.RattayAberham(
        x=adaptive_centers_um * axs.um,
        diameter=diameter,
        celsius=37.0 * axs.degC,
    )

    # Step 5: print the compact modeling summary before running the solver. The
    # minimum local spacing should be near the electrode for the adaptive layout.
    adaptive_dx_um = np.diff(adaptive_edges_um)
    uniform_dx_um = np.diff(uniform_edges_um)
    print("=== Non-uniform activation-function layout ===")
    print(f"uniform compartments : {uniform_axon.n_compartments}")
    print(f"adaptive compartments: {adaptive_axon.n_compartments}")
    print(
        "adaptive dx range    : "
        f"{np.min(adaptive_dx_um):.2f}-{np.max(adaptive_dx_um):.2f} um"
    )
    print(
        "smallest dx near     : "
        f"{adaptive_centers_um[np.argmin(adaptive_dx_um)]:.1f} um"
    )

    # Step 6: stimulate both axons with the same analytical setup. Each
    # stimulation is sampled on the axon's own intrinsic compartment centers.
    stimulus = axs.Stimulus.biphasic(
        start=0.4 * axs.ms,
        cathodic_amplitude=240.0 * axs.uA,
        cathodic_duration=0.08 * axs.ms,
        interphase=0.04 * axs.ms,
    )
    simulations = []
    for axon in (uniform_axon, adaptive_axon):
        positions = axon.layout.position_values(unit=axs.um) * axs.um
        stimulation = axs.analytical.point_source_stimulation(
            electrode,
            positions,
            sigma=sigma,
            stimulus=stimulus,
        )
        simulation = axs.AxonInstance(axon)
        simulation.add_extracellular_stimulation(stimulation=stimulation)
        simulations.append(simulation)

    results = axs.simulate_pool(
        tuple(simulations),
        duration=2.5 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
        progress="plain",
    )

    # Step 7: visualize both the meshing heuristic and the simulated response.
    fig = plt.figure(figsize=(13.0, 8.2), constrained_layout=True)
    subfig_left, subfig_right = fig.subfigures(1, 2, width_ratios=[1.0, 1.15])
    left_axes = subfig_left.subplots(3, 1)
    right_axes = subfig_right.subplots(2, 1, sharex=False, sharey=True)
    ax_activation, ax_spacing, ax_layout = left_axes
    ax_uniform_vm, ax_adaptive_vm = right_axes

    ax_activation.plot(fine_x_um, activation_proxy, color="C3", linewidth=2.0)
    ax_activation.axvline(electrode.x_um, color="black", linestyle="--", linewidth=1.0)
    ax_activation.scatter(
        adaptive_centers_um,
        np.full_like(adaptive_centers_um, np.min(activation_proxy)),
        s=10,
        color="C0",
        alpha=0.65,
        label="adaptive centers",
    )
    ax_activation.set_title("Activation-function proxy")
    ax_activation.set_xlabel("Intrinsic axon position [um]")
    ax_activation.set_ylabel("d2 footprint / dx2")
    ax_activation.grid(True, alpha=0.3)
    ax_activation.legend(frameon=False)

    ax_spacing.plot(uniform_centers_um, uniform_dx_um, label="uniform", color="C1")
    ax_spacing.plot(adaptive_centers_um, adaptive_dx_um, label="adaptive", color="C0")
    ax_spacing.axvline(electrode.x_um, color="black", linestyle="--", linewidth=1.0)
    ax_spacing.set_title("Local compartment width")
    ax_spacing.set_xlabel("Intrinsic axon position [um]")
    ax_spacing.set_ylabel("dx [um]")
    ax_spacing.grid(True, alpha=0.3)
    ax_spacing.legend(frameon=False)

    adaptive_axon.layout.plot(
        ax=ax_layout,
        position_unit=axs.um,
        title="Adaptive non-uniform layout",
        compartment_labels="auto",
        max_compartment_labels=80,
    )
    ax_layout.axvline(electrode.x_um, color="black", linestyle="--", linewidth=1.0)

    vm_uniform = results[0].voltage_values(unit=axs.mV)
    vm_adaptive = results[1].voltage_values(unit=axs.mV)
    time_ms = results[0].time_values(unit=axs.ms)
    if time_ms.size > 1:
        time_edges_ms = np.empty(time_ms.size + 1, dtype=float)
        time_edges_ms[1:-1] = 0.5 * (time_ms[:-1] + time_ms[1:])
        time_edges_ms[0] = max(0.0, time_ms[0] - 0.5 * (time_ms[1] - time_ms[0]))
        time_edges_ms[-1] = time_ms[-1] + 0.5 * (time_ms[-1] - time_ms[-2])
    else:
        time_edges_ms = np.asarray([time_ms[0] - 0.5, time_ms[0] + 0.5])
    vmin = min(float(np.min(vm_uniform)), float(np.min(vm_adaptive)))
    vmax = max(float(np.max(vm_uniform)), float(np.max(vm_adaptive)))

    for ax, title, vm_mV, edges_um in (
        (ax_uniform_vm, "Uniform Vm map", vm_uniform, uniform_edges_um),
        (ax_adaptive_vm, "Adaptive Vm map", vm_adaptive, adaptive_edges_um),
    ):
        image = ax.pcolormesh(
            time_edges_ms,
            edges_um,
            vm_mV.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axhline(electrode.x_um, color="white", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("Intrinsic axon position [um]")
    ax_adaptive_vm.set_xlabel("Time [ms]")
    subfig_right.colorbar(image, ax=right_axes, label="Vm [mV]")

    plt.show()


if __name__ == "__main__":
    main()
