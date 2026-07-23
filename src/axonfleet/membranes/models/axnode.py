"""AxNode/MRG nodal membrane equations written as plain Python source."""

from __future__ import annotations

from axonfleet.membranes.model import Model, currents, rates
from axonfleet.membranes.math import safe_exp, vtrap
from axonfleet.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Rate,
    Temperature,
    Voltage,
)
from axonfleet.utils.units import cm2, degC, mS, mV, ms


class AxNode(Model):
    """MRG-like active nodal membrane equations."""

    model_kind = "axnode"
    parameter_aliases = {"temperature": "celsius"}
    metadata = {
        "display_name": "MRG-like active node",
        "family": "mrg_node",
        "source_reference": "McIntyre-Richardson-Grill style nodal membrane",
        "temperature_reference": "20/36 degC q10 references by gate family",
        "current_sign_convention": "outward_positive",
        "notes": "Persistent sodium, transient sodium, potassium, and leak currents.",
    }

    celsius: Temperature = 37.0 * degC
    gnapbar: ConductanceDensity = 10.0 * mS / cm2
    gnabar: ConductanceDensity = 3000.0 * mS / cm2
    gkbar: ConductanceDensity = 80.0 * mS / cm2
    gl: ConductanceDensity = 7.0 * mS / cm2
    ena: Voltage = 50.0 * mV
    ek: Voltage = -90.0 * mV
    el: Voltage = -90.0 * mV

    @rates
    def rates(self, Vm: Voltage):
        """Persistent sodium, transient sodium, and potassium gate rates."""

        q10_1: Dimensionless = 2.2 ** ((self.celsius - 20.0 * degC) / (10.0 * degC))
        q10_2: Dimensionless = 2.9 ** ((self.celsius - 20.0 * degC) / (10.0 * degC))
        q10_3: Dimensionless = 3.0 ** ((self.celsius - 36.0 * degC) / (10.0 * degC))

        alpha_mp: Rate = q10_1 * (0.01 / (ms * mV)) * vtrap(
            -(Vm + 27.0 * mV),
            10.2 * mV,
        )
        beta_mp: Rate = q10_1 * (0.00025 / (ms * mV)) * vtrap(
            Vm + 34.0 * mV,
            10.0 * mV,
        )
        alpha_m: Rate = q10_1 * (1.86 / (ms * mV)) * vtrap(
            -(Vm + 21.4 * mV),
            10.3 * mV,
        )
        beta_m: Rate = q10_1 * (0.086 / (ms * mV)) * vtrap(
            Vm + 25.7 * mV,
            9.16 * mV,
        )
        alpha_h: Rate = q10_2 * (0.062 / (ms * mV)) * vtrap(
            Vm + 114.0 * mV,
            11.0 * mV,
        )
        beta_h: Rate = q10_2 * (2.3 / ms) / (
            1.0 + safe_exp(-((Vm + 31.8 * mV) / (13.4 * mV)))
        )

        v2: Voltage = Vm + 80.0 * mV
        alpha_s: Rate = q10_3 * (0.3 / ms) / (
            safe_exp((v2 - 27.0 * mV) / (-5.0 * mV)) + 1.0
        )
        beta_s: Rate = q10_3 * (0.03 / ms) / (
            safe_exp((v2 + 10.0 * mV) / (-1.0 * mV)) + 1.0
        )
        self.keep(
            q10_1,
            q10_2,
            q10_3,
            alpha_mp,
            beta_mp,
            alpha_m,
            beta_m,
            alpha_h,
            beta_h,
            alpha_s,
            beta_s,
        )

    @currents
    def currents(
        self,
        Vm: Voltage,
        mp: Gate,
        m: Gate,
        h: Gate,
        s: Gate,
    ):
        """Persistent Na, transient Na, K, and leak currents plus conductances."""

        g_nap: ConductanceDensity = self.gnapbar * (mp**3)
        g_na: ConductanceDensity = self.gnabar * (m**3) * h
        g_k: ConductanceDensity = self.gkbar * s
        g_l: ConductanceDensity = self.gl

        I_nap: CurrentDensity = g_nap * (Vm - self.ena)
        I_na: CurrentDensity = g_na * (Vm - self.ena)
        I_k: CurrentDensity = g_k * (Vm - self.ek)
        I_l: CurrentDensity = g_l * (Vm - self.el)
        return I_nap, I_na, I_k, I_l, g_nap, g_na, g_k, g_l
