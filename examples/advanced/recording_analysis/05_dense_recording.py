"""Record dense membrane observables through the public batch result path.

Run:
    python examples/advanced/recording_analysis/05_dense_recording.py
"""

from __future__ import annotations

import axonfleet as axs


def _stimulated_hh(amplitude_na: float) -> axs.AxonInstance:
    axon = axs.AxonInstance(
        axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon.add_current_clamp(
        position=50.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=amplitude_na * axs.nA,
        ),
    )
    return axon


def _mrg() -> axs.AxonInstance:
    return axs.AxonInstance(
        axs.axons.MRG(
            diameter=5.7 * axs.um,
            nodes=3,
            compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
        )
    )


def main() -> None:
    rows = [_stimulated_hh(0.30), _stimulated_hh(0.60)]

    full = axs.AxonSimulation(
        rows,
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.full(),
    ).run()

    print(f"available: {[signal.id.value for signal in full.recording_manifest.available_signals]}")
    print(f"Vm block: {full.signal(axs.signals.Vm).shape}")
    print(f"gate names: {list(full.signal(axs.signals.GATES))}")
    print(f"m gate block: {full.signal(axs.signals.GATES)['hodgkin_huxley.m'].shape}")
    print(f"current names: {list(full.signal(axs.signals.CURRENTS))}")
    print(f"conductance names: {list(full.signal(axs.signals.CONDUCTANCES))}")

    gates_only = axs.AxonSimulation(
        rows,
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.only(axs.signals.GATES),
    ).run()

    print(f"gates-only available: {[signal.id.value for signal in gates_only.recording_manifest.available_signals]}")
    print(f"gates-only m gate block: {gates_only.signal(axs.signals.GATES)['hodgkin_huxley.m'].shape}")

    double_cable = axs.AxonSimulation(
        _mrg(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.probes(
            (
                axs.signals.Vm,
                axs.signals.GATES,
                axs.signals.CURRENTS,
                axs.signals.CONDUCTANCES,
            ),
            count=3,
        ),
    ).run().single

    print(f"double-cable route: {double_cable.diagnostics['dispatch_method']}")
    print(f"double-cable Vm probes: {double_cable.Vm.shape}")
    print(f"double-cable gate names: {list(double_cable.signal(axs.signals.GATES))}")


if __name__ == "__main__":
    main()
