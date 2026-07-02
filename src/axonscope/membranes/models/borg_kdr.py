"""Sundt/Borg KDR potassium membrane equations written as plain Python source."""

from __future__ import annotations

from axonscope.membranes.model import Model, currents, rates
from axonscope.membranes.math import exp
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


class BorgKDR(Model):
    """Sundt/Borg delayed-rectifier potassium component equations."""

    model_kind = "borg_kdr"
    metadata = {
        "display_name": "Sundt/Borg delayed rectifier potassium component",
        "family": "sundt_component",
        "component": "BorgKDR",
        "source_reference": "Borg-Graham delayed rectifier component used by Sundt",
        "final_gate_update": "post_solve_voltage",
        "current_sign_convention": "outward_positive",
    }

    celsius: Temperature = 36.0 * degC
    vhalfn: Voltage = -32.0 * mV
    vhalfl: Voltage = -61.0 * mV
    a0n: Rate = 0.03 / ms
    a0l: Rate = 0.001 / ms
    zetan: Dimensionless = -5.0
    zetal: Dimensionless = 2.0
    gmn: Dimensionless = 0.4
    gml: Dimensionless = 1.0
    gkdrbar: ConductanceDensity = 3.0 * mS / cm2
    ek: Voltage = -77.0 * mV

    @rates
    def rates(self, Vm: Voltage):
        """Delayed-rectifier n/l gate rates with temperature scaling."""

        q10: Dimensionless = 3.0 ** ((self.celsius - 30.0 * degC) / (10.0 * degC))
        abs_temperature: Dimensionless = (273.16 * degC + self.celsius) / (1.0 * degC)
        thermal_scale: Dimensionless = (1e-3 * 9.648e4 / 8.315) / abs_temperature

        v_n: Dimensionless = (Vm - self.vhalfn) / (1.0 * mV)
        v_l: Dimensionless = (Vm - self.vhalfl) / (1.0 * mV)

        alpn: Dimensionless = exp(self.zetan * v_n * thermal_scale)
        alpl: Dimensionless = exp(self.zetal * v_l * thermal_scale)
        betn: Dimensionless = exp(self.zetan * self.gmn * v_n * thermal_scale)
        betl: Dimensionless = exp(self.zetal * self.gml * v_l * thermal_scale)

        alpha_n: Rate = self.a0n / betn
        beta_n: Rate = self.a0n * alpn / betn
        alpha_l: Rate = self.a0l / betl
        beta_l: Rate = self.a0l * alpl / betl
        self.keep(q10, alpha_n, beta_n, alpha_l, beta_l)

    @currents
    def currents(
        self,
        Vm: Voltage,
        n: Gate,
        l: Gate,
    ):
        """Delayed-rectifier potassium current and conductance observable."""

        g_k: ConductanceDensity = self.gkdrbar * (n**3) * l
        I_k: CurrentDensity = g_k * (Vm - self.ek)
        return I_k, g_k
