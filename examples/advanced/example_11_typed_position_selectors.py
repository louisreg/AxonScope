"""Advanced example 11: typed activation position selectors.

Run:
    python examples/advanced/example_11_typed_position_selectors.py

Phase 2 replaces activation strings such as ``"distal"`` with typed selectors
from ``axs.positions``.
"""

from __future__ import annotations

import axonscope as axs


def main() -> None:
    # Step 1: build one axon and stimulate near the middle.
    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=21,
        celsius=6.3 * axs.degC,
    )
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.15 * axs.ms,
            amplitude=0.9 * axs.nA,
        ),
    )

    # Step 2: record Vm everywhere so post-hoc position selectors can compare.
    result = axs.AxonSimulation(
        instance,
        duration=1.0 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.voltage(),
    ).run()

    # Step 3: evaluate the same threshold at different typed targets.
    criteria = {
        "anywhere": axs.results.ActivationCriterion(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.ALL,
        ),
        "distal": axs.results.ActivationCriterion(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.DISTAL,
        ),
        "near clamp": axs.results.ActivationCriterion(
            threshold=-20.0 * axs.mV,
            blanking=0.1 * axs.ms,
            target=axs.positions.At(60.0 * axs.um),
        ),
    }

    # Step 4: print compact event summaries.
    for label, criterion in criteria.items():
        event = criterion.evaluate(result)
        print(
            f"{label}: activated={event.activated}, "
            f"index={event.first_index}, "
            f"position={event.first_position_um} um"
        )


if __name__ == "__main__":
    main()
