"""Compare descriptive axon and layout construction options.

Run:
    python examples/advanced/axon_models/01_layout_options.py

This example stays in the descriptive layer:

- `Section` defines what a cable piece is.
- `Layout` places sections in space and assigns compartment counts.
- Axon templates such as `HodgkinHuxley`, `RattayAberham`, and `MRG` build the
  same kind of descriptive layout for you.
- Repeated motifs can be phase-shifted. For MRG this is the documented way to
  import NRV-style node shifts without adding world coordinates to the axon.
- `layout.plot(...)` and `layout.position_values(...)` expose the geometry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import axonfleet as axs


def main() -> None:
    # Step 1: start with membrane descriptions. They define local membrane
    # dynamics, but they do not yet say where compartments live in space.
    passive_membrane = axs.membranes.Passive(
        Rm=10_000.0 * axs.ohm_cm2,
        EL=-70.0 * axs.mV,
    )
    hh_node_membrane = axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC)

    # Step 2: wrap each membrane into a named Section. A section carries the
    # passive electrical geometry used when it becomes part of a cable layout.
    axon_section = axs.axons.Section(
        "axon",
        membrane=passive_membrane,
        diameter=0.8 * axs.um,
        Ra=100.0 * axs.ohm_cm,
        Cm=1.0 * axs.uF_per_cm2,
    )
    node_section = axs.axons.Section(
        "node",
        membrane=hh_node_membrane,
        diameter=1.0 * axs.um,
        Ra=100.0 * axs.ohm_cm,
        Cm=1.0 * axs.uF_per_cm2,
    )
    mysa_section = axs.axons.Section(
        "mysa",
        membrane=passive_membrane,
        diameter=1.4 * axs.um,
        Ra=100.0 * axs.ohm_cm,
        Cm=0.1 * axs.uF_per_cm2,
    )
    flut_section = axs.axons.Section(
        "flut",
        membrane=passive_membrane,
        diameter=2.0 * axs.um,
        Ra=100.0 * axs.ohm_cm,
        Cm=0.1 * axs.uF_per_cm2,
    )
    stin_section = axs.axons.Section(
        "stin",
        membrane=passive_membrane,
        diameter=2.5 * axs.um,
        Ra=100.0 * axs.ohm_cm,
        Cm=0.1 * axs.uF_per_cm2,
    )

    # Step 3: build explicit layouts. These are useful when teaching or when a
    # model needs hand-authored sections rather than a built-in template.
    uniform_layout = axs.axons.Layout.single_uniform(
        axon_section,
        length=1.0 * axs.mm,
        compartments=20,
    )
    non_uniform_layout = axs.axons.Layout.single_non_uniform(
        axon_section,
        x=np.asarray([0.0, 15.0, 40.0, 120.0, 300.0, 700.0, 1000.0]) * axs.um,
    )
    simple_motif_layout = axs.axons.Layout.sequence(
        [node_section, stin_section],
        section_lengths=np.asarray([1.0, 199.0]) * axs.um,
        compartments=[1, 8],
        lengths=800.0 * axs.um,
    )
    # A phase shift rotates the repeated motif before cropping it to the
    # requested length. This is still intrinsic 1D axon geometry.
    phased_motif_layout = axs.axons.Layout.sequence(
        [node_section, stin_section],
        section_lengths=np.asarray([1.0, 199.0]) * axs.um,
        compartments=[1, 8],
        lengths=800.0 * axs.um,
        phase_shift=65.0 * axs.um,
    )
    heterogeneous_motif_layout = axs.axons.Layout.sequence(
        [node_section, mysa_section, flut_section, stin_section],
        section_lengths=np.asarray([1.0, 3.0, 20.0, 80.0]) * axs.um,
        compartments=[1, 1, 2, 4],
        lengths=312.0 * axs.um,
    )

    # Step 4: compare the explicit layouts with built-in axon templates. A
    # template is still descriptive; it just creates the sections/layout for you.
    hh_template = axs.axons.HodgkinHuxley(
        length=1.0 * axs.mm,
        diameter=0.8 * axs.um,
        compartments=20,
        celsius=6.3 * axs.degC,
    )
    rattay_template = axs.axons.RattayAberham(
        x=np.asarray([0.0, 15.0, 40.0, 120.0, 300.0, 700.0, 1000.0]) * axs.um,
        diameter=0.8 * axs.um,
        celsius=37.0 * axs.degC,
    )
    mrg_default = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=5,
    )
    # MRG exposes the same idea as `x_shift`: the distance from the axon start
    # to the first node start. It phases the node motif; it is not anatomical
    # placement.
    mrg_phased_nodes = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=5,
        x_shift=120.0 * axs.um,
    )
    mrg_compartment_map = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=5,
        compartments={
            "node": 1,
            "MYSA": 1,
            "FLUT": 2,
            "STIN": 4,
        },
    )

    plots = {
        "single_uniform": uniform_layout,
        "single_non_uniform": non_uniform_layout,
        "sequence, simple repeated motif": simple_motif_layout,
        "sequence, phase-shifted repeated motif": phased_motif_layout,
        "sequence, heterogeneous repeated motif": heterogeneous_motif_layout,
        "axon template, uniform HH": hh_template.layout,
        "axon template, non-uniform Rattay": rattay_template.layout,
        "MRG template, one compartment per section": mrg_default.layout,
        "MRG template, node phase shift": mrg_phased_nodes.layout,
        "MRG template, section compartment map": mrg_compartment_map.layout,
    }

    # Step 5: print one compact table before plotting, so the example is useful
    # in a terminal or notebook cell even before the figure appears.
    print("=== Layout options ===")
    for title, layout in plots.items():
        positions_um = layout.position_values(unit=axs.um)
        first_section = layout.elements[0].section.name
        print(
            f"{title}: {positions_um.size} compartments, "
            f"x=[{positions_um[0]:.1f}, {positions_um[-1]:.1f}] um, "
            f"first section={first_section}"
        )

    print("\n=== Phase checks ===")
    print(
        "sequence phase_shift=65 um starts inside the internode/STIN motif; "
        f"first compartment at {phased_motif_layout.position_values(unit=axs.um)[0]:.1f} um"
    )
    print(
        "MRG x_shift=120 um places the first node center at "
        f"{mrg_phased_nodes.node_position_values(unit=axs.um)[0]:.1f} um"
    )

    # Step 6: plot each layout. Compartment labels make the difference between
    # uniform, non-uniform, and section-sequence construction visible.
    fig, axes = plt.subplots(len(plots), 1, figsize=(12, 15), constrained_layout=True)
    for ax, (title, layout) in zip(np.ravel(axes), plots.items(), strict=True):
        layout.plot(
            ax=ax,
            position_unit=axs.um,
            title=title,
            compartment_labels=True,
            max_compartment_labels=180,
        )
    fig.suptitle("AxonFleet descriptive layout options")
    plt.show()


if __name__ == "__main__":
    main()
