"""Run a simulation with an explicit public execution policy.

Run:
    python examples/advanced/runtime/01_runtime_policy.py

Runtime, device, and precision are public choices. They should be expressed
through `ExecutionPolicy` rather than by importing backend/JAX internals from an
example or notebook.
"""

from __future__ import annotations

import axonscope as axs


def main() -> None:
    # Step 1: build the same ordinary single-axon simulation used in many basic
    # examples. The runtime policy is orthogonal to the scientific model.
    axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.6 * axs.um,
        compartments=9,
        celsius=6.3 * axs.degC,
    )
    simulation = axs.AxonInstance(axon)
    simulation.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=0.4 * axs.nA,
        ),
    )

    # Step 2: declare backend/runtime choices with the public policy object. This
    # is where a script says "JAX, CPU, float32" without reaching into internals.
    policy = axs.ExecutionPolicy(
        runtime=axs.Runtime.JAX,
        device=axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
    )
    recording = axs.Recording.voltage()

    # Step 3: pass the policy to execution. The same policy can be used for
    # estimation or inspection so all runtime-facing reports agree.
    result = axs.simulate(
        simulation,
        duration=10 * axs.ms,
        dt=0.001 * axs.ms,
        recording=recording,
        execution_policy=policy,
    )
    estimate = axs.AxonSimulation(
        simulation,
        duration=10 * axs.ms,
        dt=0.001 * axs.ms,
        recording=recording,
        execution_policy=policy,
    ).estimate()

    print(f"Vm shape: {result.Vm.shape}")
    print("estimate:")
    print(estimate.format())


if __name__ == "__main__":
    main()
