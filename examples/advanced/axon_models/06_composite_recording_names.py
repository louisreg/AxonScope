"""Show how labelled composite membranes appear in recordings.

Run:
    python examples/advanced/axon_models/06_composite_recording_names.py

This example focuses on naming, not on a scientific simulation result.
`Composite` component labels become the public namespace for gates, states, and
generic observables. Currents and conductances keep their semantic aggregate
names.
"""

from __future__ import annotations

import numpy as np

import axonscope as axs


def _recording_summary(recordings: dict[str, object] | None) -> dict[str, object]:
    summary: dict[str, object] = {}
    for group_name, values in (recordings or {}).items():
        if isinstance(values, dict):
            summary[group_name] = tuple(values)
        else:
            summary[group_name] = tuple(np.asarray(values).shape)
    return summary


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

    # The rest of the script is a tiny single-section axon whose membrane is the
    # composite above. Recording.full() keeps Vm plus observable groups so the
    # naming rule is visible in the result.
    section = axs.axons.Section(
        "labelled composite membrane",
        membrane=membrane,
        diameter=0.5 * axs.um,
        Ra=100.0 * axs.ohm_cm,
        Cm=1.0 * axs.uF_per_cm2,
    )
    axon = axs.axons.Axon(
        layout=axs.axons.Layout.single_uniform(
            section,
            length=120.0 * axs.um,
            compartments=21,
        ),
        formulation=axs.axons.CableFormulation.SINGLE_CABLE,
        v_init=-70.0 * axs.mV,
    )

    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.10 * axs.ms,
            duration=0.10 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )

    result = axs.AxonSimulation(
        instance,
        duration=0.30 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.full(),
    ).run().single

    summary = _recording_summary(result.recordings)
    print("=== Composite recording groups ===")
    for group_name, names in summary.items():
        print(f"{group_name}: {names}")

    print("\n=== What to notice ===")
    print("Gates are namespaced by component label: hh.m, hh.h, hh.n.")
    print("Currents stay semantic aggregates: I_l combines all leak terms.")
    print("Conductances do the same: g_l combines all leak conductances.")
    print("There is no leak_weak.g_l conductance column because g_l is aggregated.")


if __name__ == "__main__":
    main()
