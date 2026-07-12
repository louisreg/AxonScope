"""Show how labelled composite membrane outputs are named.

Run:
    python examples/advanced/axon_models/06_composite_recording_names.py

This example focuses on naming, not on a scientific simulation result.
`Composite` component labels become the public namespace for gates, states, and
generic observables. Currents and conductances keep their semantic aggregate
names.
"""

from __future__ import annotations

import axonscope as axs


def main() -> None:
    # A labelled mapping is the clearest form for composites. The keys are the
    # public component labels that will appear in recording names such as
    # "hh.m". Labels should be stable snake_case names chosen by the model
    # author.
    membrane = axs.membranes.Composite(
        {
            "hh": axs.membranes.HodgkinHuxley(celsius=6.3 * axs.degC),
            "leak_weak": axs.membranes.Passive(
                Rm=20_000.0 * axs.ohm_cm2,
                EL=-70.0 * axs.mV,
            ),
            "leak_strong": axs.membranes.Passive(
                Rm=5_000.0 * axs.ohm_cm2,
                EL=-65.0 * axs.mV,
            ),
        }
    )

    # If the same model kind appears twice in a sequence, AxonScope refuses to
    # invent arbitrary suffixes. Use a labelled mapping instead.
    try:
        axs.membranes.Composite([axs.membranes.Passive(), axs.membranes.Passive()])
    except ValueError as exc:
        print(f"Duplicate sequence rejected: {exc}")

    report = membrane.explain()
    outputs = report.recording_outputs
    print("=== Composite output names ===")
    print(f"gates: {outputs.gates}")
    print(f"currents: {outputs.currents}")
    print(f"conductances: {outputs.conductances}")
    print(f"generic observables: {outputs.observables}")

    print("\n=== What to notice ===")
    print("Gates are namespaced by component label: hh.m, hh.h, hh.n.")
    print("Currents stay semantic aggregates: I_l combines all leak terms.")
    print("Conductances do the same: g_l combines all leak conductances.")
    print("There is no leak_weak.g_l conductance column because g_l is aggregated.")


if __name__ == "__main__":
    main()
