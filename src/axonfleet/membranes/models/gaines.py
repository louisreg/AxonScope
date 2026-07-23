"""Gaines motor and sensory myelinated membrane parameterizations.

The public motor/sensory classes delegate to one nodal and one internodal
source topology.  Family and section differences therefore remain numeric
parameters instead of creating parallel compiler or runtime paths.
"""

from __future__ import annotations

from axonfleet.membranes.math import safe_exp, vtrap
from axonfleet.membranes.model import Model, currents, rates
from axonfleet.membranes.types import (
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Rate,
    RatePerVoltage,
    Temperature,
    Voltage,
)
from axonfleet.utils.units import degC, mS_per_cm2, mV, ms, per_ms_per_mV


class _GainesNodeSource(Model):
    """Shared Gaines nodal equation topology."""

    model_kind = "gaines_node_source"
    parameter_aliases = {"temperature": "celsius"}
    metadata = {
        "display_name": "Gaines active node",
        "family": "gaines",
        "source_reference": "Gaines et al. 2018 motor/sensory myelinated axon model",
        "doi": "10.1007/s10827-018-0689-5",
        "temperature_reference": "20/36 degC q10 references by gate family",
        "current_sign_convention": "outward_positive",
        "notes": "One shared topology parameterized for motor and sensory nodes.",
    }

    celsius: Temperature = 37.0 * degC
    gnapbar: ConductanceDensity = 10.0 * mS_per_cm2
    gnabar: ConductanceDensity = 3000.0 * mS_per_cm2
    gkbar: ConductanceDensity = 80.0 * mS_per_cm2
    gl: ConductanceDensity = 7.0 * mS_per_cm2
    gkfbar: ConductanceDensity = 25.68 * mS_per_cm2
    ena: Voltage = 50.0 * mV
    ek: Voltage = -90.0 * mV
    el: Voltage = -90.0 * mV
    ekf: Voltage = -90.0 * mV

    amp_a: RatePerVoltage = 0.01 * per_ms_per_mV
    amp_b: Voltage = 27.0 * mV
    amp_c: Voltage = 10.2 * mV
    bmp_a: RatePerVoltage = 0.00025 * per_ms_per_mV
    bmp_b: Voltage = 34.0 * mV
    bmp_c: Voltage = 10.0 * mV
    am_a: RatePerVoltage = 1.86 * per_ms_per_mV
    am_b: Voltage = 20.4 * mV
    am_c: Voltage = 10.3 * mV
    bm_a: RatePerVoltage = 0.086 * per_ms_per_mV
    bm_b: Voltage = 25.7 * mV
    bm_c: Voltage = 9.16 * mV
    ah_a: RatePerVoltage = 0.062 * per_ms_per_mV
    ah_b: Voltage = 114.0 * mV
    ah_c: Voltage = 11.0 * mV
    bh_a: Rate = 2.3 / ms
    bh_b: Voltage = 31.8 * mV
    bh_c: Voltage = 13.4 * mV

    @rates
    def rates(self, Vm: Voltage):
        """Persistent Na, transient Na, slow K, and fast K gate rates."""

        q10_na: Dimensionless = 2.2 ** ((self.celsius - 20.0 * degC) / (10.0 * degC))
        q10_inactivation: Dimensionless = 2.9 ** (
            (self.celsius - 20.0 * degC) / (10.0 * degC)
        )
        q10_k: Dimensionless = 3.0 ** ((self.celsius - 36.0 * degC) / (10.0 * degC))

        alpha_mp: Rate = q10_na * self.amp_a * vtrap(
            -(Vm + self.amp_b),
            self.amp_c,
        )
        beta_mp: Rate = q10_na * self.bmp_a * vtrap(
            Vm + self.bmp_b,
            self.bmp_c,
        )
        alpha_m: Rate = q10_na * self.am_a * vtrap(
            -(Vm + self.am_b),
            self.am_c,
        )
        beta_m: Rate = q10_na * self.bm_a * vtrap(
            Vm + self.bm_b,
            self.bm_c,
        )
        alpha_h: Rate = q10_inactivation * self.ah_a * vtrap(
            Vm + self.ah_b,
            self.ah_c,
        )
        beta_h: Rate = q10_inactivation * self.bh_a / (
            1.0 + safe_exp(-((Vm + self.bh_b) / self.bh_c))
        )

        v_traub: Voltage = Vm + 80.0 * mV
        alpha_s: Rate = q10_k * (0.3 / ms) / (
            safe_exp((v_traub - 27.0 * mV) / (-5.0 * mV)) + 1.0
        )
        beta_s: Rate = q10_k * (0.03 / ms) / (
            safe_exp((v_traub + 10.0 * mV) / (-1.0 * mV)) + 1.0
        )
        alpha_n: Rate = q10_k * (0.0462 * per_ms_per_mV) * vtrap(
            -(Vm + 83.2 * mV),
            1.1 * mV,
        )
        beta_n: Rate = q10_k * (0.0824 * per_ms_per_mV) * vtrap(
            Vm + 66.0 * mV,
            10.5 * mV,
        )
        self.keep(
            q10_na,
            q10_inactivation,
            q10_k,
            alpha_mp,
            beta_mp,
            alpha_m,
            beta_m,
            alpha_h,
            beta_h,
            alpha_s,
            beta_s,
            alpha_n,
            beta_n,
        )

    @currents
    def currents(
        self,
        Vm: Voltage,
        mp: Gate,
        m: Gate,
        h: Gate,
        s: Gate,
        n: Gate,
    ):
        """Nodal sodium, potassium, and leak currents."""

        g_nap: ConductanceDensity = self.gnapbar * (mp**3)
        g_na: ConductanceDensity = self.gnabar * (m**3) * h
        g_k: ConductanceDensity = self.gkbar * s
        g_l: ConductanceDensity = self.gl
        g_kf: ConductanceDensity = self.gkfbar * (n**4)
        I_nap: CurrentDensity = g_nap * (Vm - self.ena)
        I_na: CurrentDensity = g_na * (Vm - self.ena)
        I_k: CurrentDensity = g_k * (Vm - self.ek)
        I_l: CurrentDensity = g_l * (Vm - self.el)
        I_kf: CurrentDensity = g_kf * (Vm - self.ekf)
        return I_nap, I_na, I_k, I_l, I_kf, g_nap, g_na, g_k, g_l, g_kf


class GainesMotorNode(_GainesNodeSource):
    """Gaines motor-axon nodal membrane."""

    model_kind = "gaines_motor_node"
    source_model = _GainesNodeSource


class GainesSensoryNode(_GainesNodeSource):
    """Gaines sensory-axon nodal membrane."""

    model_kind = "gaines_sensory_node"
    source_model = _GainesNodeSource
    gkbar: ConductanceDensity = 41.06 * mS_per_cm2
    gl: ConductanceDensity = 6.005 * mS_per_cm2
    gkfbar: ConductanceDensity = 27.37 * mS_per_cm2
    amp_a: RatePerVoltage = 0.00957 * per_ms_per_mV
    amp_b: Voltage = 26.852 * mV
    bmp_a: RatePerVoltage = 0.0002401 * per_ms_per_mV
    bmp_b: Voltage = 33.8333 * mV
    am_a: RatePerVoltage = 1.77753 * per_ms_per_mV
    am_b: Voltage = 20.1795 * mV
    bm_a: RatePerVoltage = 0.0823 * per_ms_per_mV
    bm_b: Voltage = 25.4746 * mV
    ah_a: RatePerVoltage = 0.075286 * per_ms_per_mV
    ah_b: Voltage = 112.7124 * mV
    ah_c: Voltage = 8.3910 * mV
    bh_a: Rate = 2.8083 / ms
    bh_b: Voltage = 30.5435 * mV
    bh_c: Voltage = 10.2263 * mV


class _GainesInternodeSource(Model):
    """Shared Gaines MYSA/FLUT/STIN membrane equation topology."""

    model_kind = "gaines_internode_source"
    parameter_aliases = {"temperature": "celsius"}
    metadata = {
        "display_name": "Gaines active internode",
        "family": "gaines",
        "source_reference": "Gaines et al. 2018 motor/sensory myelinated axon model",
        "doi": "10.1007/s10827-018-0689-5",
        "temperature_reference": "36 degC q10 reference",
        "current_sign_convention": "outward_positive",
        "notes": "One shared MYSA/FLUT/STIN topology with section parameters.",
    }

    celsius: Temperature = 37.0 * degC
    gkbar: ConductanceDensity = 2.581 * mS_per_cm2
    gl: ConductanceDensity = 0.2 * mS_per_cm2
    gqbar: ConductanceDensity = 2.232 * mS_per_cm2
    gkfbar: ConductanceDensity = 25.68 * mS_per_cm2
    ek: Voltage = -90.0 * mV
    el: Voltage = -90.0 * mV
    eq: Voltage = -54.9 * mV
    ekf: Voltage = -90.0 * mV
    hcn_midpoint: Voltage = -107.3 * mV

    @rates
    def rates(self, Vm: Voltage):
        """Slow K, HCN, and fast K gate rates."""

        q10_k: Dimensionless = 3.0 ** ((self.celsius - 36.0 * degC) / (10.0 * degC))
        v_traub: Voltage = Vm + 80.0 * mV
        alpha_s: Rate = q10_k * (0.3 / ms) / (
            safe_exp((v_traub - 27.0 * mV) / (-5.0 * mV)) + 1.0
        )
        beta_s: Rate = q10_k * (0.03 / ms) / (
            safe_exp((v_traub + 10.0 * mV) / (-1.0 * mV)) + 1.0
        )
        alpha_q: Rate = q10_k * (0.00522 / ms) * safe_exp(
            (Vm - self.hcn_midpoint) / (-12.2 * mV)
        )
        beta_q: Rate = q10_k * (0.00522 / ms) * safe_exp(
            -((Vm - self.hcn_midpoint) / (-12.2 * mV))
        )
        alpha_n: Rate = q10_k * (0.0462 * per_ms_per_mV) * vtrap(
            -(Vm + 83.2 * mV),
            1.1 * mV,
        )
        beta_n: Rate = q10_k * (0.0824 * per_ms_per_mV) * vtrap(
            Vm + 66.0 * mV,
            10.5 * mV,
        )
        self.keep(
            q10_k,
            alpha_s,
            beta_s,
            alpha_q,
            beta_q,
            alpha_n,
            beta_n,
        )

    @currents
    def currents(self, Vm: Voltage, s: Gate, q: Gate, n: Gate):
        """Internodal slow K, leak, HCN, and fast K currents."""

        g_k: ConductanceDensity = self.gkbar * s
        g_l: ConductanceDensity = self.gl
        g_q: ConductanceDensity = self.gqbar * q
        g_kf: ConductanceDensity = self.gkfbar * (n**4)
        I_k: CurrentDensity = g_k * (Vm - self.ek)
        I_l: CurrentDensity = g_l * (Vm - self.el)
        I_q: CurrentDensity = g_q * (Vm - self.eq)
        I_kf: CurrentDensity = g_kf * (Vm - self.ekf)
        return I_k, I_l, I_q, I_kf, g_k, g_l, g_q, g_kf


class GainesMotorInternode(_GainesInternodeSource):
    """Gaines motor-axon active internodal membrane."""

    model_kind = "gaines_motor_internode"
    source_model = _GainesInternodeSource


class GainesSensoryInternode(_GainesInternodeSource):
    """Gaines sensory-axon active internodal membrane."""

    model_kind = "gaines_sensory_internode"
    source_model = _GainesInternodeSource
    gkbar: ConductanceDensity = 1.324 * mS_per_cm2
    gl: ConductanceDensity = 0.1716 * mS_per_cm2
    gqbar: ConductanceDensity = 3.102 * mS_per_cm2
    gkfbar: ConductanceDensity = 27.37 * mS_per_cm2
    hcn_midpoint: Voltage = -94.2 * mV


__all__ = [
    "GainesMotorInternode",
    "GainesMotorNode",
    "GainesSensoryInternode",
    "GainesSensoryNode",
]
