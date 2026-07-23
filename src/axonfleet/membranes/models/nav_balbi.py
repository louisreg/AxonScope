"""Shared Balbi six-state human voltage-gated sodium channel equations."""

from __future__ import annotations

from axonfleet.membranes.math import exp, q10
from axonfleet.membranes.model import Model, currents, markov, state
from axonfleet.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Occupancy,
    Rate,
    Temperature,
    Voltage,
)
from axonfleet.utils.units import cm2, degC, mS, mV, ms


class BalbiNav(Model):
    """Single six-state topology shared by the human Nav1.1-Nav1.9 family."""

    model_kind = "balbi_nav"
    parameter_aliases = {"temperature": "celsius"}
    metadata = {
        "display_name": "Balbi six-state human Nav channel",
        "family": "balbi_nav",
        "source_reference": "Balbi, Massobrio, Hellgren Kotaleski 2017",
        "source_doi": "10.1371/journal.pcbi.1005737",
        "source_modeldb": "230137",
        "temperature_reference": "20 degC",
        "current_sign_convention": "outward_positive",
        "notes": "Shared C1/C2/O1/O2/I1/I2 topology; parameters select the isoform.",
    }

    celsius: Temperature = 22.0 * degC
    gbar: ConductanceDensity = 100.0 * mS / cm2
    ena: Voltage = 65.0 * mV

    C1C2b2: Rate = 18.0 / ms
    C1C2v2: Voltage = -7.0 * mV
    C1C2k2: Voltage = -10.0 * mV
    C2C1b1: Rate = 3.0 / ms
    C2C1v1: Voltage = -37.0 * mV
    C2C1k1: Voltage = 10.0 * mV
    C2C1b2: Rate = 18.0 / ms
    C2C1v2: Voltage = -7.0 * mV
    C2C1k2: Voltage = -10.0 * mV
    C2O1b2: Rate = 18.0 / ms
    C2O1v2: Voltage = -7.0 * mV
    C2O1k2: Voltage = -10.0 * mV
    O1C2b1: Rate = 3.0 / ms
    O1C2v1: Voltage = -37.0 * mV
    O1C2k1: Voltage = 10.0 * mV
    O1C2b2: Rate = 18.0 / ms
    O1C2v2: Voltage = -7.0 * mV
    O1C2k2: Voltage = -10.0 * mV
    C2O2b2: Rate = 0.08 / ms
    C2O2v2: Voltage = -10.0 * mV
    C2O2k2: Voltage = -15.0 * mV
    O2C2b1: Rate = 2.0 / ms
    O2C2v1: Voltage = -50.0 * mV
    O2C2k1: Voltage = 7.0 * mV
    O2C2b2: Rate = 0.2 / ms
    O2C2v2: Voltage = -20.0 * mV
    O2C2k2: Voltage = -10.0 * mV
    O1I1b1: Rate = 8.0 / ms
    O1I1v1: Voltage = -37.0 * mV
    O1I1k1: Voltage = 13.0 * mV
    O1I1b2: Rate = 17.0 / ms
    O1I1v2: Voltage = -7.0 * mV
    O1I1k2: Voltage = -15.0 * mV
    I1O1b1: Rate = 0.00001 / ms
    I1O1v1: Voltage = -37.0 * mV
    I1O1k1: Voltage = 10.0 * mV
    I1C1b1: Rate = 0.21 / ms
    I1C1v1: Voltage = -61.0 * mV
    I1C1k1: Voltage = 7.0 * mV
    C1I1b2: Rate = 0.3 / ms
    C1I1v2: Voltage = -61.0 * mV
    C1I1k2: Voltage = -5.5 * mV
    I1I2b2: Rate = 0.0015 / ms
    I1I2v2: Voltage = -90.0 * mV
    I1I2k2: Voltage = -5.0 * mV
    I2I1b1: Rate = 0.0075 / ms
    I2I1v1: Voltage = -90.0 * mV
    I2I1k1: Voltage = 15.0 * mV

    C1: Occupancy = state(1.0)
    C2: Occupancy = state(0.0)
    O1: Occupancy = state(0.0)
    O2: Occupancy = state(0.0)
    I1: Occupancy = state(0.0)
    I2: Occupancy = state(0.0)

    @markov(
        "sodium",
        states=("C1", "C2", "O1", "O2", "I1", "I2"),
        transitions=(
            ("C1", "C2", "C1C2", "C2C1"),
            ("C2", "O1", "C2O1", "O1C2"),
            ("C2", "O2", "C2O2", "O2C2"),
            ("O1", "I1", "O1I1", "I1O1"),
            ("I1", "C1", "I1C1", "C1I1"),
            ("I1", "I2", "I1I2", "I2I1"),
        ),
        initialization="stationary",
        conserve_probability=True,
    )
    def sodium(self, Vm: Voltage):
        temperature_factor: Dimensionless = q10(3.0, self.celsius, 20.0 * degC)
        C1C2: Rate = temperature_factor * self.C1C2b2 / (
            1.0 + exp((Vm - self.C1C2v2) / self.C1C2k2)
        )
        C2C1: Rate = temperature_factor * (
            self.C2C1b1 / (1.0 + exp((Vm - self.C2C1v1) / self.C2C1k1))
            + self.C2C1b2 / (1.0 + exp((Vm - self.C2C1v2) / self.C2C1k2))
        )
        C2O1: Rate = temperature_factor * self.C2O1b2 / (
            1.0 + exp((Vm - self.C2O1v2) / self.C2O1k2)
        )
        O1C2: Rate = temperature_factor * (
            self.O1C2b1 / (1.0 + exp((Vm - self.O1C2v1) / self.O1C2k1))
            + self.O1C2b2 / (1.0 + exp((Vm - self.O1C2v2) / self.O1C2k2))
        )
        C2O2: Rate = temperature_factor * self.C2O2b2 / (
            1.0 + exp((Vm - self.C2O2v2) / self.C2O2k2)
        )
        O2C2: Rate = temperature_factor * (
            self.O2C2b1 / (1.0 + exp((Vm - self.O2C2v1) / self.O2C2k1))
            + self.O2C2b2 / (1.0 + exp((Vm - self.O2C2v2) / self.O2C2k2))
        )
        O1I1: Rate = temperature_factor * (
            self.O1I1b1 / (1.0 + exp((Vm - self.O1I1v1) / self.O1I1k1))
            + self.O1I1b2 / (1.0 + exp((Vm - self.O1I1v2) / self.O1I1k2))
        )
        I1O1: Rate = temperature_factor * self.I1O1b1 / (
            1.0 + exp((Vm - self.I1O1v1) / self.I1O1k1)
        )
        I1C1: Rate = temperature_factor * self.I1C1b1 / (
            1.0 + exp((Vm - self.I1C1v1) / self.I1C1k1)
        )
        C1I1: Rate = temperature_factor * self.C1I1b2 / (
            1.0 + exp((Vm - self.C1I1v2) / self.C1I1k2)
        )
        I1I2: Rate = temperature_factor * self.I1I2b2 / (
            1.0 + exp((Vm - self.I1I2v2) / self.I1I2k2)
        )
        I2I1: Rate = temperature_factor * self.I2I1b1 / (
            1.0 + exp((Vm - self.I2I1v1) / self.I2I1k1)
        )
        self.keep(C1C2, C2C1, C2O1, O1C2, C2O2, O2C2)
        self.keep(O1I1, I1O1, I1C1, C1I1, I1I2, I2I1)

    @currents(
        outputs=("I_na",),
        observables=("g_na", "open_probability"),
    )
    def currents(
        self,
        Vm: Voltage,
        C1: Occupancy,
        C2: Occupancy,
        O1: Occupancy,
        O2: Occupancy,
        I1: Occupancy,
        I2: Occupancy,
    ):
        open_probability: Dimensionless = O1 + O2
        g_na: ConductanceDensity = self.gbar * open_probability
        I_na: CurrentDensity = g_na * (Vm - self.ena)
        return I_na, g_na, open_probability


__all__ = ["BalbiNav"]
