"""Rattay-Aberham membrane equations written as plain Python source."""

from __future__ import annotations

from axonscope.membranes.model import Model, currents, rates
from axonscope.membranes.math import exp, vtrap
from axonscope.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Rate,
    Temperature,
    Voltage,
)
from axonscope.utils.units import cm2, degC, mS, mV, ms


class RattayAberham(Model):
    """Rattay-Aberham active membrane equations."""

    model_kind = "rattay_aberham"
    parameter_aliases = {"temperature": "celsius"}
    metadata = {
        "display_name": "Rattay-Aberham active membrane",
        "family": "rattay_aberham",
        "source_reference": "Rattay and Aberham 1993 electrical stimulation axon model",
        "final_gate_update": "post_solve_voltage",
        "temperature_reference": "6.3 degC with fitted q10 at 37 degC defaults",
        "current_sign_convention": "outward_positive",
        "notes": "HH-like Na, K, and leak membrane used for stimulation examples.",
    }

    celsius: Temperature = 37.0 * degC
    gnabar: ConductanceDensity = 120.0 * mS / cm2
    gkbar: ConductanceDensity = 36.0 * mS / cm2
    gl: ConductanceDensity = 0.3 * mS / cm2
    el: Voltage = -59.4 * mV
    ena: Voltage = 45.0 * mV
    ek: Voltage = -82.0 * mV

    @rates
    def rates(self, Vm: Voltage):
        """HH-like m/h/n gate rates shifted to the Rattay-Aberham voltage convention."""

        q10: Dimensionless = 2.24659524757 ** (
            (self.celsius - 6.3 * degC) / (10.0 * degC)
        )
        v70: Voltage = Vm + 70.0 * mV
        v70_over_10: Dimensionless = v70 / (10.0 * mV)

        alpha_m: Rate = 1.0 / ms * vtrap(2.5 - v70_over_10, 1.0)
        beta_m: Rate = 4.0 / ms * exp(-(v70 / (18.0 * mV)))

        alpha_h: Rate = 0.07 / ms * exp(-(v70 / (20.0 * mV)))
        beta_h: Rate = 1.0 / ms / (exp(3.0 - v70_over_10) + 1.0)

        alpha_n: Rate = 0.1 / ms * vtrap(1.0 - v70_over_10, 1.0)
        beta_n: Rate = 0.125 / ms * exp(-(v70 / (80.0 * mV)))
        self.keep(q10, alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n)

    @currents
    def currents(
        self,
        Vm: Voltage,
        m: Gate,
        h: Gate,
        n: Gate,
    ):
        """Na, K, and passive leak current densities plus conductance observables."""

        g_na: ConductanceDensity = self.gnabar * (m**3) * h
        g_k: ConductanceDensity = self.gkbar * (n**4)
        g_l: ConductanceDensity = self.gl

        I_na: CurrentDensity = g_na * (Vm - self.ena)
        I_k: CurrentDensity = g_k * (Vm - self.ek)
        I_l: CurrentDensity = g_l * (Vm - self.el)
        return I_na, I_k, I_l, g_na, g_k, g_l
