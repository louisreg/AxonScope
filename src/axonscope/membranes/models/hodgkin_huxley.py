"""Hodgkin-Huxley membrane equations written as plain Python source."""

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


class HodgkinHuxley(Model):
    """Classical Hodgkin-Huxley squid axon membrane equations."""

    model_kind = "hodgkin_huxley"
    parameter_aliases = {"temperature": "celsius"}
    metadata = {
        "display_name": "Hodgkin-Huxley squid axon",
        "family": "hodgkin_huxley",
        "source_reference": "Hodgkin and Huxley 1952 squid giant axon",
        "source_doi": "10.1113/jphysiol.1952.sp004764",
        "temperature_reference": "6.3 degC",
        "current_sign_convention": "outward_positive",
        "notes": "Classical Na, K, and leak conductance model.",
    }

    celsius: Temperature = 6.3 * degC
    gnabar: ConductanceDensity = 120.0 * mS / cm2
    gkbar: ConductanceDensity = 36.0 * mS / cm2
    gl: ConductanceDensity = 0.3 * mS / cm2
    el: Voltage = -54.3 * mV
    ena: Voltage = 50.0 * mV
    ek: Voltage = -77.0 * mV

    @rates
    def rates(self, Vm: Voltage):
        """Gate rates for m/h sodium activation-inactivation and n potassium activation."""

        q10: Dimensionless = 3.0 ** ((self.celsius - 6.3 * degC) / (10.0 * degC))

        alpha_m: Rate = 0.1 / (ms * mV) * vtrap(
            -(Vm + 40.0 * mV),
            10.0 * mV,
        )
        beta_m: Rate = 4.0 / ms * exp(-((Vm + 65.0 * mV) / (18.0 * mV)))

        alpha_h: Rate = 0.07 / ms * exp(-((Vm + 65.0 * mV) / (20.0 * mV)))
        beta_h: Rate = 1.0 / ms / (
            exp(-((Vm + 35.0 * mV) / (10.0 * mV))) + 1.0
        )

        alpha_n: Rate = 0.01 / (ms * mV) * vtrap(
            -(Vm + 55.0 * mV),
            10.0 * mV,
        )
        beta_n: Rate = 0.125 / ms * exp(-((Vm + 65.0 * mV) / (80.0 * mV)))
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
