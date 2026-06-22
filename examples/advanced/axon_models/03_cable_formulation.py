"""Use typed cable formulations.

Run:
    python examples/advanced/axon_models/03_cable_formulation.py

Phase 2 replaces raw formulation strings with ``axs.axons.CableFormulation``.
Most users can rely on template defaults; the enum is useful for custom layouts.
"""

from __future__ import annotations

import axonscope as axs


def main() -> None:
    # Step 1: build a custom single-cable section layout.
    section = axs.axons.Section(
        "axon",
        membrane=axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC),
        diameter=0.5 * axs.um,
    )
    layout = axs.axons.Layout.single_uniform(
        section,
        length=120.0 * axs.um,
        compartments=21,
    )

    # Step 2: use the typed formulation selector instead of "single-cable".
    custom = axs.axons.Axon(
        layout=layout,
        formulation=axs.axons.CableFormulation.SINGLE_CABLE,
        temperature=6.3 * axs.degC,
    )

    # Step 3: templates already choose their formulation.
    hh = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    mrg = axs.axons.MRG(diameter=5.7 * axs.um, nodes=3)

    print(f"custom formulation: {custom.resolved_formulation}")
    print(f"HH formulation: {hh.resolved_formulation}")
    print(f"MRG formulation: {mrg.resolved_formulation}")


if __name__ == "__main__":
    main()
