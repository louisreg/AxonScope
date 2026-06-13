"""Compare descriptive axon and layout construction options.

This example stays in the descriptive layer:

- `Section` defines what a cable piece is.
- `Layout` places sections in space and assigns compartment counts.
- Axon templates such as `HodgkinHuxley`, `RattayAberham`, and `MRG` build the
  same kind of descriptive layout for you.
- `layout.plot(...)` and `layout.position_values(...)` expose the geometry.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import axonscope as axs


def make_sections() -> dict[str, axs.axons.Section]:
    passive = axs.membranes.Passive(Rm=10_000.0 * axs.ohm_cm2, EL=-70.0 * axs.mV)
    node = axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC)

    return {
        "axon": axs.axons.Section(
            "axon",
            membrane=passive,
            diameter=0.8 * axs.um,
            Ra=100.0 * axs.ohm_cm,
            Cm=1.0 * axs.uF_per_cm2,
        ),
        "node": axs.axons.Section(
            "node",
            membrane=node,
            diameter=1.0 * axs.um,
            Ra=100.0 * axs.ohm_cm,
            Cm=1.0 * axs.uF_per_cm2,
        ),
        "mysa": axs.axons.Section(
            "mysa",
            membrane=passive,
            diameter=1.4 * axs.um,
            Ra=100.0 * axs.ohm_cm,
            Cm=0.1 * axs.uF_per_cm2,
        ),
        "flut": axs.axons.Section(
            "flut",
            membrane=passive,
            diameter=2.0 * axs.um,
            Ra=100.0 * axs.ohm_cm,
            Cm=0.1 * axs.uF_per_cm2,
        ),
        "stin": axs.axons.Section(
            "stin",
            membrane=passive,
            diameter=2.5 * axs.um,
            Ra=100.0 * axs.ohm_cm,
            Cm=0.1 * axs.uF_per_cm2,
        ),
    }


def build_layouts(sections: dict[str, axs.axons.Section]) -> dict[str, axs.axons.Layout]:
    axon = sections["axon"]
    node = sections["node"]
    mysa = sections["mysa"]
    flut = sections["flut"]
    stin = sections["stin"]

    return {
        "single_uniform": axs.axons.Layout.single_uniform(
            axon,
            length=1.0 * axs.mm,
            compartments=20,
        ),
        "single_non_uniform": axs.axons.Layout.single_non_uniform(
            axon,
            x=np.asarray([0.0, 15.0, 40.0, 120.0, 300.0, 700.0, 1000.0]) * axs.um,
        ),
        "sequence, simple repeated motif": axs.axons.Layout.sequence(
            [node, stin],
            section_lengths=np.asarray([1.0, 199.0]) * axs.um,
            compartments=[1, 8],
            lengths=800.0 * axs.um,
        ),
        "sequence, heterogeneous repeated motif": axs.axons.Layout.sequence(
            [node, mysa, flut, stin],
            section_lengths=np.asarray([1.0, 3.0, 20.0, 80.0]) * axs.um,
            compartments=[1, 1, 2, 4],
            lengths=312.0 * axs.um,
        ),
    }


def build_axons() -> dict[str, axs.axons.Axon]:
    return {
        "axon template, uniform HH": axs.axons.HodgkinHuxley(
            length=1.0 * axs.mm,
            diameter=0.8 * axs.um,
            compartments=20,
            celsius=6.3 * axs.degC,
        ),
        "axon template, non-uniform Rattay": axs.axons.RattayAberham(
            x=np.asarray([0.0, 15.0, 40.0, 120.0, 300.0, 700.0, 1000.0]) * axs.um,
            diameter=0.8 * axs.um,
            celsius=37.0 * axs.degC,
        ),
        "MRG template, one compartment per section": axs.axons.MRG(
            diameter=10.0 * axs.um,
            nodes=5,
        ),
        "MRG template, section compartment map": axs.axons.MRG(
            diameter=10.0 * axs.um,
            nodes=5,
            compartments={
                "node": 1,
                "MYSA": 1,
                "FLUT": 2,
                "STIN": 4,
            },
        ),
    }


def main() -> None:
    sections = make_sections()
    layouts = build_layouts(sections)
    axons = build_axons()
    plots = {
        **layouts,
        **{title: axon.layout for title, axon in axons.items()},
    }

    fig, axes = plt.subplots(len(plots), 1, figsize=(12, 15), constrained_layout=True)
    for ax, (title, layout) in zip(np.ravel(axes), plots.items(), strict=True):
        layout.plot(
            ax=ax,
            position_unit=axs.um,
            title=title,
            compartment_labels=True,
            max_compartment_labels=180,
        )
    fig.suptitle("AxonScope descriptive layout options")
    plt.show()


if __name__ == "__main__":
    main()
