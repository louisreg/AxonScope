"""Sundt composite membrane equations written as one standalone source model."""

from __future__ import annotations

from axonscope.membranes.model import Model, currents, mechanism
from axonscope.membranes.math import exp, vtrap
from axonscope.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Rate,
    ResistanceArea,
    Temperature,
    Voltage,
)
from axonscope.utils.units import cm2, degC, mS, mV, ms, ohm


class Sundt(Model):
    """Sundt composite NaHH + BorgKDR + passive leak equations."""

    model_kind = "sundt"
    parameter_aliases = {"temperature": "celsius"}
    metadata = {
        "display_name": "Sundt composite membrane",
        "family": "sundt",
        "source_reference": "Sundt et al. composite NaHH + BorgKDR + passive leak model",
        "final_gate_update": "post_solve_voltage",
        "current_sign_convention": "outward_positive",
        "notes": "Standalone source version of the NaHH, BorgKDR, and leak composition.",
    }

    celsius: Temperature = 37.0 * degC
    mshift: Voltage = -6.0 * mV
    hshift: Voltage = 6.0 * mV
    ishift: Voltage = 0.0 * mV
    vhalfn: Voltage = -32.0 * mV
    vhalfl: Voltage = -61.0 * mV
    a0n: Rate = 0.03 / ms
    a0l: Rate = 0.001 / ms
    zetan: Dimensionless = -5.0
    zetal: Dimensionless = 2.0
    gmn: Dimensionless = 0.4
    gml: Dimensionless = 1.0
    gnabar: ConductanceDensity = 40.0 * mS / cm2
    gkdrbar: ConductanceDensity = 40.0 * mS / cm2
    ena: Voltage = 45.0 * mV
    ek: Voltage = -90.0 * mV
    Rm: ResistanceArea = 1.0e4 * ohm * cm2
    El: Voltage = -70.0 * mV

    @mechanism("na_hh")
    def na_hh_rates(self, Vm: Voltage):
        """Sundt sodium m/h gate rates."""

        V_m: Voltage = Vm + 65.0 * mV
        q10_m: Dimensionless = 3.0 ** ((self.celsius - 30.0 * degC) / (10.0 * degC))
        q10_h: Dimensionless = q10_m

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
        self.keep(q10_m, q10_h, alpha_m, beta_m, alpha_h, beta_h)

    @mechanism("borg_kdr")
    def borg_kdr_rates(self, Vm: Voltage):
        """Borg-Graham delayed-rectifier n/l gate rates."""

        q10_n: Dimensionless = 3.0 ** ((self.celsius - 30.0 * degC) / (10.0 * degC))
        q10_l: Dimensionless = q10_n
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
        self.keep(q10_n, q10_l, alpha_n, beta_n, alpha_l, beta_l)

    @currents
    def currents(
        self,
        Vm: Voltage,
        m: Gate,
        h: Gate,
        n: Gate,
        l: Gate,
    ):
        """Sodium, delayed-rectifier potassium, and passive leak current densities."""

        g_na: ConductanceDensity = self.gnabar * (m**3) * h
        g_k: ConductanceDensity = self.gkdrbar * (n**3) * l
        g_l: ConductanceDensity = 1.0 / self.Rm

        I_na: CurrentDensity = g_na * (Vm - self.ena)
        I_k: CurrentDensity = g_k * (Vm - self.ek)
        I_l: CurrentDensity = g_l * (Vm - self.El)
        return I_na, I_k, I_l, g_na, g_k, g_l
