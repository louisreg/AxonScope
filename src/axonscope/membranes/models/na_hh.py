"""Sundt NaHH sodium membrane equations written as plain Python source."""

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


class NaHH(Model):
    """Sundt NaHH sodium component equations."""

    model_kind = "na_hh"
    metadata = {
        "display_name": "Sundt sodium component",
        "family": "sundt_component",
        "component": "NaHH",
        "source_reference": "Sundt et al. conductance model component",
        "final_gate_update": "post_solve_voltage",
        "current_sign_convention": "outward_positive",
    }

    celsius: Temperature = 36.0 * degC
    mshift: Voltage = -6.0 * mV
    hshift: Voltage = 6.0 * mV
    ishift: Voltage = 0.0 * mV
    gnabar: ConductanceDensity = 300.0 * mS / cm2
    ena: Voltage = 50.0 * mV

    @rates
    def rates(self, Vm: Voltage):
        """Sodium m/h gate rates used by the Sundt composite membrane."""

        V_m: Voltage = Vm + 65.0 * mV
        q10: Dimensionless = 3.0 ** ((self.celsius - 30.0 * degC) / (10.0 * degC))

        alpha_m: Rate = 0.32 / (ms * mV) * vtrap(
            13.1 * mV - (V_m + self.mshift),
            4.0 * mV,
        )
        beta_m: Rate = 0.28 / (ms * mV) * vtrap(
            (V_m + self.mshift) - 40.1 * mV,
            5.0 * mV,
        )

        alpha_h: Rate = 0.128 / ms * exp(
            (17.0 * mV - (V_m + self.hshift) + self.ishift) / (18.0 * mV)
        )
        beta_h: Rate = 4.0 / ms / (
            exp((40.0 * mV - (V_m + self.hshift)) / (5.0 * mV)) + 1.0
        )
        self.keep(q10, alpha_m, beta_m, alpha_h, beta_h)

    @currents
    def currents(
        self,
        Vm: Voltage,
        m: Gate,
        h: Gate,
    ):
        """Fast sodium current and conductance observable."""

        g_na: ConductanceDensity = self.gnabar * (m**3) * h
        I_na: CurrentDensity = g_na * (Vm - self.ena)
        return I_na, g_na
