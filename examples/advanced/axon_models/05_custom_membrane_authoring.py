"""Author custom membrane models as plain Python classes.

Run:
    python examples/advanced/axon_models/05_custom_membrane_authoring.py

This example covers the public P7 membrane-authoring contract:

- a passive leak class with named intermediate equations;
- a small gated HH-style class with a mechanism section;
- a non-gate state with step diagnostics;
- generated-code and source explain reports;
- a short simulation using the custom membrane;
- one intentionally invalid model to show the compiler error shape.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import axonscope as axs
from axonscope.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Rate,
    ResistanceArea,
    Time,
    Voltage,
)


class DemoLeak(axs.membranes.Model):
    """Minimal passive leak model with explicit intermediate equations."""

    model_kind = "demo_leak"

    Rm: ResistanceArea = 10_000.0 * axs.ohm_cm2
    EL: Voltage = -70.0 * axs.mV

    @axs.membranes.currents
    def currents(self, Vm: Voltage):
        drive: Voltage = Vm - self.EL
        g_l: ConductanceDensity = 1.0 / self.Rm
        I_l: CurrentDensity = g_l * drive
        return I_l, g_l, drive


class DemoSodiumLeak(axs.membranes.Model):
    """Small HH-style membrane with one gated sodium activation and leak."""

    model_kind = "demo_sodium_leak"

    gna: ConductanceDensity = 20.0 * axs.mS_per_cm2
    gl: ConductanceDensity = 0.1 * axs.mS_per_cm2
    ena: Voltage = 45.0 * axs.mV
    el: Voltage = -70.0 * axs.mV

    @axs.membranes.mechanism("sodium_activation")
    def sodium_activation(self, Vm: Voltage):
        alpha_m: Rate = 0.1 / (axs.ms * axs.mV) * (Vm + 35.0 * axs.mV)
        beta_m: Rate = 4.0 / axs.ms
        self.keep(alpha_m, beta_m)

    @axs.membranes.currents(
        outputs=("I_na", "I_l"),
        observables=("g_na", "g_l"),
        internal=("drive_na", "drive_l"),
    )
    def currents(self, Vm: Voltage, m: Gate):
        drive_na: Voltage = Vm - self.ena
        drive_l: Voltage = Vm - self.el
        g_na: ConductanceDensity = self.gna * m
        g_l: ConductanceDensity = self.gl
        I_na: CurrentDensity = g_na * drive_na
        I_l: CurrentDensity = g_l * drive_l
        return I_na, I_l, g_na, g_l


class RelaxingLeak(axs.membranes.Model):
    """Leak model with one non-gate state and a step diagnostic."""

    model_kind = "relaxing_leak"

    Rm: ResistanceArea = 12_000.0 * axs.ohm_cm2
    EL: Voltage = -68.0 * axs.mV
    adaptation: Dimensionless = axs.membranes.state(
        0.0,
        description="Slow dimensionless leak modulation.",
    )

    @axs.membranes.initials(updates={"adaptation": "adaptation_initial"})
    def initials(self):
        adaptation_initial: Dimensionless = 0.0
        return adaptation_initial

    @axs.membranes.currents(
        outputs=("I_l",),
        observables=("g_l", "adaptation_observable"),
    )
    def currents(self, Vm: Voltage, adaptation: Dimensionless):
        drive: Voltage = Vm - self.EL
        adaptation_observable: Dimensionless = adaptation
        g_l: ConductanceDensity = 1.0 / self.Rm
        I_l: CurrentDensity = g_l * drive
        return I_l, g_l, adaptation_observable

    @axs.membranes.step(
        prepare={"adaptation": "adaptation_next"},
        total_outward_current="total_outward_current",
        explicit_outward_current="explicit_outward_current",
        correction_current="correction_current",
        diagnostics={"adaptation_drive": "adaptation_drive"},
    )
    def step(self, dt: Time):
        adaptation_drive: Dimensionless = 0.5 - self.adaptation
        adaptation_next: Dimensionless = self.adaptation + dt * (
            adaptation_drive / (5.0 * axs.ms)
        )
        total_outward_current: CurrentDensity = self.I_l
        explicit_outward_current: CurrentDensity = self.I_l
        correction_current: CurrentDensity = 0.0 * self.I_l
        return (
            adaptation_drive,
            adaptation_next,
            total_outward_current,
            explicit_outward_current,
            correction_current,
        )


class InvalidMissingSymbol(axs.membranes.Model):
    """Broken on purpose: the compiler should report the unknown symbol."""

    model_kind = "invalid_missing_symbol"

    @axs.membranes.currents
    def currents(self, Vm: Voltage):
        I_l: CurrentDensity = missing_current
        return I_l


def main() -> None:
    print(DemoLeak().explain().format())
    print(DemoSodiumLeak().inspect_generated_code().format())
    print(RelaxingLeak().explain().format())

    try:
        InvalidMissingSymbol().explain()
    except Exception as exc:
        print(f"Invalid model error: {exc}")

    length = 400.0 * axs.um
    axon = axs.axons.Axon(
        layout=axs.axons.Layout.single_uniform(
            axs.axons.Section(
                "custom relaxing leak",
                membrane=RelaxingLeak(),
                diameter=0.6 * axs.um,
                Ra=120.0 * axs.ohm_cm,
                Cm=1.0 * axs.uF_per_cm2,
                tags=("custom", "stateful"),
            ),
            length=length,
            compartments=41,
        ),
        formulation=axs.axons.CableFormulation.SINGLE_CABLE,
        v_init=-68.0 * axs.mV,
    )

    sim = axs.AxonInstance(axon)
    clamp = axs.IntracellularCurrentClamp(
        position=length / 2.0,
        current=axs.Stimulus.pulse(
            start=0.5 * axs.ms,
            duration=0.4 * axs.ms,
            amplitude=0.4 * axs.nA,
        ),
    )
    sim.add_intracellular_context(context=clamp)

    run = axs.AxonSimulation(sim, duration=2.0 * axs.ms, dt=0.02 * axs.ms).run()
    result = run.single

    fig, (ax_layout, ax_trace, ax_map) = plt.subplots(1, 3, figsize=(13, 3.5))
    axon.layout.plot(
        ax=ax_layout,
        position_unit=axs.um,
        title="Custom membrane layout",
        compartment_labels="auto",
        max_compartment_labels=50,
    )
    ax_layout.axvline(
        clamp.position_um,
        color="C3",
        linestyle="--",
        linewidth=1.2,
        label="clamp",
    )
    ax_layout.legend(frameon=False)

    result.plot_trace(
        ax=ax_trace,
        position=length / 2.0,
        voltage_unit=axs.mV,
        title="RelaxingLeak trace",
    )
    result.plot_map(
        ax=ax_map,
        voltage_unit=axs.mV,
        position_unit=axs.um,
        title="RelaxingLeak Vm",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
