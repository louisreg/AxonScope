"""Schild 1994 DRG C-fiber membrane equations as standalone source."""

from __future__ import annotations

import math
from axonfleet.membranes.model import Model, currents, initials, mechanism, state, step
from axonfleet.membranes.math import clip, exp, log, maximum, q10, rates_from_tau_inf, where
from axonfleet.membranes.models.schild_common import nernst_mV
from axonfleet.membranes.types import (
    Concentration,
    ConcentrationPerCurrentDensityTime,
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Length,
    Rate,
    RatePerConcentration,
    Temperature,
    Time,
    Voltage,
)
from axonfleet.utils.units import degC, mM, mM_per_uA_cm2_ms, mS_per_cm2, mV, ms, per_ms_per_mM, uA_per_cm2, um


_CAI0_MM = 0.000117
_CAO0_MM = 2.0
_KM_CA_MM = 0.0005
_NAO_MM = 154.0
_NAI_MM = 8.9
_KO_MM = 5.4
_KI_MM = 145.0
_NAO_ION_MM = 140.0
_NAI_ION_MM = 10.0
_KO_ION_MM = 2.5
_D_NACA = 0.0036
_GAMMA_NACA = 0.5
_R_NACA = 3.0


def derive_parameters(
    *,
    diameter_um: float = 1.0,
    celsius: float = 37.0,
    vinit_mV: float = -48.0,
) -> dict[str, float]:
    ena = nernst_mV(celsius, 1.0, _NAO_MM, _NAI_MM)
    ek = nernst_mV(celsius, 1.0, _KO_MM, _KI_MM)
    eca_static = nernst_mV(celsius, 2.0, _CAO0_MM, _CAI0_MM) - 78.7
    rt_f = 8314.0 * (celsius + 273.15) / 96500.0

    radius_cm = diameter_um / 2.0 * 1e-4
    cai_factor = 1e-3 / (96500.0 * radius_cm)
    outer_radius_cm = (diameter_um / 2.0 + 0.03 / 2.0) * 1e-4
    peri_volume_per_l = math.pi * (outer_radius_cm**2 - radius_cm**2)
    surface_area_per_l = 2.0 * math.pi * radius_cm
    cao_factor = 1e-3 * surface_area_per_l / (2.0 * peri_volume_per_l * 96500.0)

    na_term = (_NAI_ION_MM / (_NAI_ION_MM + 5.46)) ** 3
    ko_term = (_KO_ION_MM / (_KO_ION_MM + 0.621)) ** 2
    inak_max = 0.009726135 * (1.16 ** ((22.85 - celsius) / 10.0))
    fnk0 = (vinit_mV + 150.0) / (vinit_mV + 200.0)
    i_nak_pump = inak_max * fnk0 * na_term * ko_term * 1e3
    inak_max_na_ko = inak_max * na_term * ko_term * 1e3

    ica_pump_max = 0.000859437 * (2.30 ** ((22.85 - celsius) / 10.0))
    ica_pump = ica_pump_max * _CAI0_MM / (_CAI0_MM + _KM_CA_MM) * 1e3
    ica_pump_max_uA = ica_pump_max * 1e3

    knaca = 1.27324e-6 * (2.20 ** ((22.85 - celsius) / 10.0))
    fac_v = 96500.0 / (1000.0 * 8.314 * (celsius + 273.15))
    dfin = _NAI_ION_MM**3 * _CAO0_MM * math.exp(
        (_R_NACA - 2.0) * _GAMMA_NACA * vinit_mV * fac_v
    )
    dfout = _NAO_ION_MM**3 * _CAI0_MM * math.exp(
        (_R_NACA - 2.0) * (_GAMMA_NACA - 1.0) * vinit_mV * fac_v
    )
    s_naca = 1.0 + _D_NACA * (_CAI0_MM * _NAO_ION_MM**3 + _CAO0_MM * _NAI_ION_MM**3)
    inca0 = knaca * (dfin - dfout) / s_naca * 1e3

    return {
        "celsius": celsius,
        "ena": ena,
        "ek": ek,
        "eca_static": eca_static,
        "rt_f": rt_f,
        "cai_factor": cai_factor,
        "cao_factor": cao_factor,
        "cao_bath": _CAO0_MM,
        "cao_txfer": 4511.0,
        "ICaPmax": ica_pump_max_uA,
        "KmCa": _KM_CA_MM,
        "KNaCa": knaca * 1e3,
        "fac_V": fac_v,
        "nai_ion": _NAI_ION_MM,
        "nao_ion": _NAO_ION_MM,
        "DNaCa": _D_NACA,
        "gamma": _GAMMA_NACA,
        "r_naca": _R_NACA,
        "INaKmax_na_ko": inak_max_na_ko,
        "I_NaKpump_bg": i_nak_pump,
        "I_CaPump_bg": ica_pump,
        "inca0": inca0,
        "background_current": i_nak_pump + ica_pump + inca0,
    }


class Schild94(Model):
    """Schild 1994 DRG C-fiber membrane equations."""

    model_kind = "schild94"
    parameter_aliases = {
        "diameter": "diameter_um",
        "temperature": "celsius",
        "v_init": "vinit_mV",
    }
    metadata = {
        "display_name": "Schild 1994 DRG C-fiber",
        "family": "schild",
        "source_reference": "Schild et al. 1994 DRG C-fiber membrane model with calcium dynamics",
        "final_gate_update": "post_solve_voltage",
        "stateful": "calcium_pools",
        "current_sign_convention": "outward_positive",
        "notes": "Standalone source model compiled to Model IR; no historical channel-model runtime.",
    }
    diameter_um: Length = 1.0 * um
    celsius: Temperature = 37.0 * degC
    vinit_mV: Voltage = -48.0 * mV

    cai: Concentration = state(
        0.000117 * mM,
        description="Intracellular calcium concentration.",
    )
    Oc: Dimensionless = state(
        0.05,
        description="Fraction of occupied calcium buffer.",
    )
    cao: Concentration = state(
        2.0 * mM,
        description="Periaxonal calcium concentration.",
    )
    c_kca: Dimensionless = state(
        0.0,
        description="Calcium-activated potassium gate.",
    )

    @initials(updates={"c_kca": "c_kca_initial"})
    def initials(self, Vm: Voltage):
        """Voltage-dependent initial occupancy for the calcium-activated K gate."""

        alpha_c0: Rate = (750.0 * per_ms_per_mM) * (0.000117 * mM) * exp(
            (Vm - 10.0 * mV) / (12.0 * mV)
        )
        beta_c0: Rate = (0.05 / ms) * exp((Vm - 10.0 * mV) / (-60.0 * mV))
        c_sum0: Rate = maximum(alpha_c0 + beta_c0, 1e-12 / ms)
        c_kca_initial: Dimensionless = alpha_c0 / c_sum0
        self.keep(c_kca_initial)


    @mechanism("naf")
    def naf(self, Vm: Voltage, celsius: Temperature = 37.0 * degC):
        """Fast sodium activation, fast inactivation, and slow inactivation."""

        qt_m_naf: Dimensionless = q10(2.30, celsius, 22.85 * degC)
        qt_h_naf: Dimensionless = q10(1.50, celsius, 22.85 * degC)
        tau_m_naf: Time = (
            0.75 * ms * exp(-((0.0635 * ((Vm + 40.35 * mV) / (1.0 * mV))) ** 2))
            + 0.12 * ms
        ) / qt_m_naf
        inf_m_naf: Dimensionless = 1.0 / (1.0 + exp((Vm + 23.85 * mV) / (-4.75 * mV)))
        alpha_m_naf, beta_m_naf = rates_from_tau_inf(inf_m_naf, maximum(tau_m_naf, 1e-9 * ms))

        tau_h_naf: Time = (
            6.5 * ms * exp(-((0.0295 * ((Vm + 75.0 * mV) / (1.0 * mV))) ** 2))
            + 0.55 * ms
        ) / qt_h_naf
        inf_h_naf: Dimensionless = 1.0 / (1.0 + exp((Vm + 44.5 * mV) / (4.5 * mV)))
        alpha_h_naf, beta_h_naf = rates_from_tau_inf(inf_h_naf, maximum(tau_h_naf, 1e-9 * ms))

        tau_l_naf: Time = 25.0 * ms / (1.0 + exp((Vm - 20.0 * mV) / (4.5 * mV))) + 0.01 * ms
        inf_l_naf: Dimensionless = 1.0 / (1.0 + exp((Vm + 40.0 * mV) / (1.5 * mV)))
        alpha_l_naf, beta_l_naf = rates_from_tau_inf(inf_l_naf, maximum(tau_l_naf, 1e-9 * ms))
        self.keep(alpha_m_naf, beta_m_naf, alpha_h_naf, beta_h_naf, alpha_l_naf, beta_l_naf)


    @mechanism("nas")
    def nas(self, Vm: Voltage, celsius: Temperature = 37.0 * degC):
        """Slow sodium activation and inactivation."""

        qt_m_nas: Dimensionless = q10(2.30, celsius, 22.85 * degC)
        qt_h_nas: Dimensionless = q10(1.50, celsius, 22.85 * degC)
        tau_m_nas: Time = (
            1.50 * ms * exp(-((0.0595 * ((Vm + 20.35 * mV) / (1.0 * mV))) ** 2))
            + 0.15 * ms
        ) / qt_m_nas
        inf_m_nas: Dimensionless = 1.0 / (1.0 + exp((Vm + 0.35 * mV) / (-4.45 * mV)))
        alpha_m_nas, beta_m_nas = rates_from_tau_inf(inf_m_nas, maximum(tau_m_nas, 1e-9 * ms))

        tau_h_nas: Time = (
            4.95 * ms * exp(-((0.0335 * ((Vm + 20.0 * mV) / (1.0 * mV))) ** 2))
            + 0.75 * ms
        ) / qt_h_nas
        inf_h_nas: Dimensionless = 1.0 / (1.0 + exp((Vm - 2.0 * mV) / (4.5 * mV)))
        alpha_h_nas, beta_h_nas = rates_from_tau_inf(inf_h_nas, maximum(tau_h_nas, 1e-9 * ms))
        self.keep(alpha_m_nas, beta_m_nas, alpha_h_nas, beta_h_nas)


    @mechanism("kd")
    def kd(self, Vm: Voltage, celsius: Temperature = 37.0 * degC):
        """Delayed rectifier potassium activation."""

        qt_n_kd: Dimensionless = q10(1.40, celsius, 22.85 * degC)
        x_kd: Voltage = Vm + 14.273 * mV
        alpha_n_kd_raw: Rate = where(
            abs(x_kd) < 1e-6 * mV,
            0.01265 / ms,
            (0.001265 / (ms * mV)) * x_kd / (1.0 - exp(x_kd / (-10.0 * mV))),
        )
        beta_n_kd_raw: Rate = (0.125 / ms) * exp((Vm + 55.0 * mV) / (-2.5 * mV))
        inf_n_kd: Dimensionless = 1.0 / (1.0 + exp((Vm + 17.62 * mV) / (-18.38 * mV)))
        tau_n_kd: Time = (1.0 / maximum(alpha_n_kd_raw + beta_n_kd_raw, 1e-9 / ms) + 1.0 * ms) / qt_n_kd
        alpha_n_kd, beta_n_kd = rates_from_tau_inf(inf_n_kd, maximum(tau_n_kd, 1e-9 * ms))
        self.keep(alpha_n_kd, beta_n_kd)


    @mechanism("ka")
    def ka(self, Vm: Voltage, celsius: Temperature = 37.0 * degC):
        """A-type potassium activation and inactivation."""

        qt_ka: Dimensionless = q10(1.93, celsius, 22.85 * degC)
        tau_p_ka: Time = (
            5.0 * ms * exp(-((0.022 * ((Vm + 65.0 * mV) / (1.0 * mV))) ** 2))
            + 2.5 * ms
        ) / qt_ka
        inf_p_ka: Dimensionless = 1.0 / (1.0 + exp((Vm + 31.0 * mV) / (-28.0 * mV)))
        alpha_p_ka, beta_p_ka = rates_from_tau_inf(inf_p_ka, maximum(tau_p_ka, 1e-9 * ms))

        tau_q_ka: Time = (
            100.0 * ms * exp(-((0.035 * ((Vm + 30.0 * mV) / (1.0 * mV))) ** 2))
            + 10.5 * ms
        ) / qt_ka
        inf_q_ka: Dimensionless = 1.0 / (1.0 + exp((Vm + 61.0 * mV) / (7.0 * mV)))
        alpha_q_ka, beta_q_ka = rates_from_tau_inf(inf_q_ka, maximum(tau_q_ka, 1e-9 * ms))
        self.keep(alpha_p_ka, beta_p_ka, alpha_q_ka, beta_q_ka)


    @mechanism("kds")
    def kds(self, Vm: Voltage, celsius: Temperature = 37.0 * degC):
        """Sustained potassium activation and inactivation."""

        qt_kds: Dimensionless = q10(1.93, celsius, 22.85 * degC)
        tau_x_kds: Time = (
            5.0 * ms * exp(-((0.022 * ((Vm + 65.0 * mV) / (1.0 * mV))) ** 2))
            + 2.5 * ms
        ) / qt_kds
        inf_x_kds: Dimensionless = 1.0 / (1.0 + exp((Vm + 42.59 * mV) / (-14.68 * mV)))
        alpha_x_kds, beta_x_kds = rates_from_tau_inf(inf_x_kds, maximum(tau_x_kds, 1e-9 * ms))

        inf_y1_kds: Dimensionless = 1.0 / (1.0 + exp((Vm + 51.0 * mV) / (7.0 * mV)))
        tau_y1_kds: Time = (7500.0 * ms) / qt_kds
        alpha_y1_kds, beta_y1_kds = rates_from_tau_inf(inf_y1_kds, maximum(tau_y1_kds, 1e-9 * ms))
        self.keep(alpha_x_kds, beta_x_kds, alpha_y1_kds, beta_y1_kds)


    @mechanism("can")
    def can(self, Vm: Voltage, celsius: Temperature = 37.0 * degC):
        """N-type calcium activation and dual inactivation."""

        qt_can: Dimensionless = q10(4.30, celsius, 22.85 * degC)
        tau_d_can: Time = (
            3.25 * ms * exp(-((0.042 * ((Vm + 31.0 * mV) / (1.0 * mV))) ** 2))
            + 0.395 * ms
        ) / qt_can
        inf_d_can: Dimensionless = 1.0 / (1.0 + exp((Vm + 13.0 * mV) / (-4.5 * mV)))
        alpha_d_can, beta_d_can = rates_from_tau_inf(inf_d_can, maximum(tau_d_can, 1e-9 * ms))

        tau_f1_can: Time = (
            33.5 * ms * exp(-((0.0395 * ((Vm + 30.0 * mV) / (1.0 * mV))) ** 2))
            + 5.0 * ms
        ) / qt_can
        inf_f1_can: Dimensionless = 1.0 / (1.0 + exp((Vm + 13.0 * mV) / (25.0 * mV)))
        alpha_f1_can, beta_f1_can = rates_from_tau_inf(inf_f1_can, maximum(tau_f1_can, 1e-9 * ms))

        tau_f2_can: Time = (
            225.0 * ms * exp(-((0.0275 * ((Vm + 40.0 * mV) / (1.0 * mV))) ** 2))
            + 75.0 * ms
        ) / qt_can
        rn_f2_can: Dimensionless = 0.2 / (1.0 + exp((Vm - 2.0 * mV) / (-10.0 * mV)))
        inf_f2_can: Dimensionless = clip(
            rn_f2_can + 1.0 / (1.0 + exp((Vm + 33.0 * mV) / (10.0 * mV))),
            0.0,
            1.0,
        )
        alpha_f2_can, beta_f2_can = rates_from_tau_inf(inf_f2_can, maximum(tau_f2_can, 1e-9 * ms))
        self.keep(alpha_d_can, beta_d_can, alpha_f1_can, beta_f1_can, alpha_f2_can, beta_f2_can)


    @mechanism("cat")
    def cat(self, Vm: Voltage, celsius: Temperature = 37.0 * degC):
        """T-type calcium activation and inactivation."""

        qt_d_cat: Dimensionless = q10(1.90, celsius, 22.85 * degC)
        qt_f_cat: Dimensionless = q10(2.20, celsius, 22.85 * degC)
        tau_d_cat: Time = (
            22.0 * ms * exp(-((0.052 * ((Vm + 68.0 * mV) / (1.0 * mV))) ** 2))
            + 2.5 * ms
        ) / qt_d_cat
        inf_d_cat: Dimensionless = 1.0 / (1.0 + exp((Vm + 47.0 * mV) / (-5.75 * mV)))
        alpha_d_cat, beta_d_cat = rates_from_tau_inf(inf_d_cat, maximum(tau_d_cat, 1e-9 * ms))

        tau_f_cat: Time = (
            103.0 * ms * exp(-((0.050 * ((Vm + 58.0 * mV) / (1.0 * mV))) ** 2))
            + 12.5 * ms
        ) / qt_f_cat
        inf_f_cat: Dimensionless = 1.0 / (1.0 + exp((Vm + 61.0 * mV) / (6.0 * mV)))
        alpha_f_cat, beta_f_cat = rates_from_tau_inf(inf_f_cat, maximum(tau_f_cat, 1e-9 * ms))
        self.keep(alpha_d_cat, beta_d_cat, alpha_f_cat, beta_f_cat)


    @currents(
        outputs=(
            "I_na_leak",
            "I_ca_leak",
            "I_na_naf",
            "I_na_nas",
            "I_k_kd",
            "I_k_ka",
            "I_k_kds",
            "I_ca_can",
            "I_ca_cat",
        ),
        observables=(
            "g_leak_na",
            "g_leak_ca",
            "g_naf",
            "g_nas",
            "g_kd",
            "g_ka",
            "g_kds",
            "g_kca",
            "g_can",
            "g_cat",
            "g_na",
            "g_k",
            "g_ca",
        ),
        internal=(
            "cai_mid",
            "Oc_mid",
            "cao_mid",
            "c_kca_new",
            "cai_new",
            "Oc_new",
            "cao_new",
            "total_outward_current",
            "explicit_outward_current",
            "ca_correction",
            "I_na_total",
            "I_k_total",
            "I_ca_total",
            "I_total_rhs",
        ),
    )
    def currents(self,
        Vm: Voltage,
        m_naf: Gate,
        h_naf: Gate,
        l_naf: Gate,
        m_nas: Gate,
        h_nas: Gate,
        n_kd: Gate,
        p_ka: Gate,
        q_ka: Gate,
        x_kds: Gate,
        y1_kds: Gate,
        d_can: Gate,
        f1_can: Gate,
        f2_can: Gate,
        d_cat: Gate,
        f_cat: Gate,
        cai: Concentration,
        Oc: Dimensionless,
        cao: Concentration,
        c_kca: Dimensionless,
        gbna: ConductanceDensity = 0.0185681 * mS_per_cm2,
        gbca: ConductanceDensity = 0.00300626 * mS_per_cm2,
        gbar_naf: ConductanceDensity = 68.967142 * mS_per_cm2,
        gbar_nas: ConductanceDensity = 1.043349 * mS_per_cm2,
        gbar_kd: ConductanceDensity = 0.180376 * mS_per_cm2,
        gbar_ka: ConductanceDensity = 0.141471 * mS_per_cm2,
        gbar_kds: ConductanceDensity = 0.106103 * mS_per_cm2,
        gbar_can: ConductanceDensity = 0.106103 * mS_per_cm2,
        gbar_cat: ConductanceDensity = 0.0123787 * mS_per_cm2,
        gbar_kca: ConductanceDensity = 0.141471 * mS_per_cm2,
        ena: Voltage = 76.1792474793765 * mV,
        ek: Voltage = -87.92139731824584 * mV,
        eca_static: Voltage = 51.518432158797836 * mV,
    ):
        """Static channel conductances and public current components."""

        g_leak_na: ConductanceDensity = gbna
        g_leak_ca: ConductanceDensity = gbca
        g_naf: ConductanceDensity = gbar_naf * (m_naf ** 3) * h_naf * l_naf
        g_nas: ConductanceDensity = gbar_nas * (m_nas ** 3) * h_nas
        g_kd: ConductanceDensity = gbar_kd * n_kd
        g_ka: ConductanceDensity = gbar_ka * (p_ka ** 3) * q_ka
        g_kds: ConductanceDensity = gbar_kds * (x_kds ** 3) * y1_kds
        g_can: ConductanceDensity = gbar_can * d_can * (0.55 * f1_can + 0.45 * f2_can)
        g_cat: ConductanceDensity = gbar_cat * d_cat * f_cat
        g_kca: ConductanceDensity = gbar_kca * c_kca
        i_kca: CurrentDensity = g_kca * (Vm - ek)

        I_na_leak: CurrentDensity = g_leak_na * (Vm - ena)
        I_ca_leak: CurrentDensity = g_leak_ca * (Vm - eca_static)
        I_na_naf: CurrentDensity = g_naf * (Vm - ena)
        I_na_nas: CurrentDensity = g_nas * (Vm - ena)
        I_k_kd: CurrentDensity = g_kd * (Vm - ek)
        I_k_ka: CurrentDensity = g_ka * (Vm - ek)
        I_k_kds: CurrentDensity = g_kds * (Vm - ek)
        I_ca_can: CurrentDensity = g_can * (Vm - eca_static)
        I_ca_cat: CurrentDensity = g_cat * (Vm - eca_static)
        g_na: ConductanceDensity = g_leak_na + g_naf + g_nas
        g_k: ConductanceDensity = g_kd + g_ka + g_kds + g_kca
        g_ca: ConductanceDensity = g_leak_ca + g_can + g_cat
        self.keep(i_kca)
        return (
            I_na_leak,
            I_ca_leak,
            I_na_naf,
            I_na_nas,
            I_k_kd,
            I_k_ka,
            I_k_kds,
            I_ca_can,
            I_ca_cat,
            g_leak_na,
            g_leak_ca,
            g_naf,
            g_nas,
            g_kd,
            g_ka,
            g_kds,
            g_kca,
            g_can,
            g_cat,
            g_na,
            g_k,
            g_ca,
        )


    @step(
        prepare={"cai": "cai_mid", "Oc": "Oc_mid", "cao": "cao_mid"},
        finalize={
            "c_kca": "c_kca_new",
            "cai": "cai_new",
            "Oc": "Oc_new",
            "cao": "cao_new",
        },
        total_outward_current="total_outward_current",
        explicit_outward_current="explicit_outward_current",
        correction_current="ca_correction",
        prepare_gate_source="previous",
        linearization_gate_source="previous",
        diagnostics={
            "I_na": "I_na_total",
            "I_k": "I_k_total",
            "I_ca": "I_ca_total",
            "I_total_rhs_uAcm2": "I_total_rhs",
        },
    )
    def step(self,
        dt: Time,
        celsius: Temperature = 37.0 * degC,
        rt_f: Voltage = 26.72110984455958 * mV,
        gbca: ConductanceDensity = 0.00300626 * mS_per_cm2,
        eca_static: Voltage = 51.518432158797836 * mV,
        ek: Voltage = -87.92139731824584 * mV,
        cai_factor: ConcentrationPerCurrentDensityTime = 0.0002072538860103627 * mM_per_uA_cm2_ms,
        cao_factor: ConcentrationPerCurrentDensityTime = 0.003403183678331066 * mM_per_uA_cm2_ms,
        cao_bath: Concentration = 2.0 * mM,
        cao_txfer: Time = 4511.0 * ms,
        ICaPmax: CurrentDensity = 0.26446558112363094 * uA_per_cm2,
        KmCa: Concentration = 0.0005 * mM,
        KNaCa: CurrentDensity = 0.0004172363913458649 * uA_per_cm2,
        fac_V: Dimensionless = 0.037423595270448695,
        nai_ion: Dimensionless = 10.0,
        nao_ion: Dimensionless = 140.0,
        DNaCa: Dimensionless = 0.0036,
        gamma: Dimensionless = 0.5,
        r_naca: Dimensionless = 3.0,
        INaKmax_na_ko: CurrentDensity = 1.3689773094685371 * uA_per_cm2,
        I_NaKpump_bg: CurrentDensity = 0.9186558260907288 * uA_per_cm2,
        I_CaPump_bg: CurrentDensity = 0.05014987518875984 * uA_per_cm2,
        inca0: CurrentDensity = 0.0011785902539344772 * uA_per_cm2,
        background_current: CurrentDensity = 0.9699842915334231 * uA_per_cm2,
    ):
        """Advance calcium pools, KCa state, and solver correction terms."""

        half_dt: Time = 0.5 * dt
        cai_safe_prepare: Concentration = maximum(self.cai, 1e-9 * mM)
        cao_safe_prepare: Concentration = maximum(self.cao, 1e-9 * mM)
        eca_prepare: Voltage = 0.5 * rt_f * log(cao_safe_prepare / cai_safe_prepare) - 78.7 * mV
        i_chan_prepare: CurrentDensity = (self.g_can + self.g_cat) * (self.Vm - eca_prepare)
        i_leak_prepare: CurrentDensity = gbca * (self.Vm - eca_prepare)
        i_cap_prepare: CurrentDensity = ICaPmax * (self.cai / (self.cai + KmCa))
        v_norm_prepare: Dimensionless = self.Vm / (1.0 * mV)
        cai_norm_prepare: Dimensionless = self.cai / (1.0 * mM)
        cao_norm_prepare: Dimensionless = self.cao / (1.0 * mM)
        dfin_prepare: Dimensionless = (nai_ion ** 3) * cao_norm_prepare * exp((r_naca - 2.0) * gamma * v_norm_prepare * fac_V)
        dfout_prepare: Dimensionless = (nao_ion ** 3) * cai_norm_prepare * exp((r_naca - 2.0) * (gamma - 1.0) * v_norm_prepare * fac_V)
        s_nca_prepare: Dimensionless = 1.0 + DNaCa * (cai_norm_prepare * (nao_ion ** 3) + cao_norm_prepare * (nai_ion ** 3))
        inca_prepare: CurrentDensity = KNaCa * (dfin_prepare - dfout_prepare) / maximum(s_nca_prepare, 1e-12)
        ca_budget_prepare: CurrentDensity = i_chan_prepare + i_leak_prepare + i_cap_prepare - 2.0 * inca_prepare

        ku: RatePerConcentration = 100.0 * per_ms_per_mM
        kr: Rate = 0.238 / ms
        nb_Bi: Concentration = 0.004 * mM
        diff0_prepare: Rate = ku * self.cai * (1.0 - self.Oc) - kr * self.Oc
        Oc1_prepare: Dimensionless = clip(self.Oc + half_dt * diff0_prepare, 0.0, 1.0)
        cai1_prepare: Concentration = maximum(self.cai + half_dt * (-ca_budget_prepare * cai_factor - nb_Bi * diff0_prepare), 1e-9 * mM)
        diff1_prepare: Rate = ku * cai1_prepare * (1.0 - Oc1_prepare) - kr * Oc1_prepare
        Oc_mid: Dimensionless = clip(self.Oc + half_dt * diff1_prepare, 0.0, 1.0)
        cai_mid: Concentration = maximum(self.cai + half_dt * (-ca_budget_prepare * cai_factor - nb_Bi * diff1_prepare), 1e-9 * mM)
        dcao_dt_prepare: Concentration = ca_budget_prepare * cao_factor + (cao_bath - self.cao) / cao_txfer
        cao_mid: Concentration = maximum(self.cao + half_dt * dcao_dt_prepare, 1e-9 * mM)

        cai_safe_corr: Concentration = maximum(cai_mid, 1e-9 * mM)
        cao_safe_corr: Concentration = maximum(cao_mid, 1e-9 * mM)
        eca_corr_dyn: Voltage = 0.5 * rt_f * log(cao_safe_corr / cai_safe_corr) - 78.7 * mV
        i_eca_corr: CurrentDensity = (self.g_can + self.g_cat + gbca) * (eca_static - eca_corr_dyn)
        fnk_dyn: Dimensionless = (self.Vm + 150.0 * mV) / (self.Vm + 200.0 * mV)
        i_nak_corr: CurrentDensity = INaKmax_na_ko * fnk_dyn - I_NaKpump_bg
        i_cap_corr_dyn: CurrentDensity = ICaPmax * (cai_mid / (cai_mid + KmCa))
        i_cap_corr: CurrentDensity = i_cap_corr_dyn - I_CaPump_bg
        v_norm_corr: Dimensionless = self.Vm / (1.0 * mV)
        cai_norm_corr: Dimensionless = cai_mid / (1.0 * mM)
        cao_norm_corr: Dimensionless = cao_mid / (1.0 * mM)
        dfin_corr: Dimensionless = (nai_ion ** 3) * cao_norm_corr * exp((r_naca - 2.0) * gamma * v_norm_corr * fac_V)
        dfout_corr: Dimensionless = (nao_ion ** 3) * cai_norm_corr * exp((r_naca - 2.0) * (gamma - 1.0) * v_norm_corr * fac_V)
        s_nca_corr: Dimensionless = 1.0 + DNaCa * (cai_norm_corr * (nao_ion ** 3) + cao_norm_corr * (nai_ion ** 3))
        inca_corr: CurrentDensity = KNaCa * (dfin_corr - dfout_corr) / maximum(s_nca_corr, 1e-12)
        i_nca_corr: CurrentDensity = inca_corr - inca0
        ca_correction: CurrentDensity = i_eca_corr + i_nak_corr + i_cap_corr + i_nca_corr

        qt_kca: Dimensionless = q10(2.30, celsius, 22.85 * degC)
        alpha_c: Rate = (750.0 * per_ms_per_mM) * self.cai * exp((self.Vm - 10.0 * mV) / (12.0 * mV)) * qt_kca
        beta_c: Rate = (0.05 / ms) * exp((self.Vm - 10.0 * mV) / (-60.0 * mV)) * qt_kca
        c_sum: Rate = maximum(alpha_c + beta_c, 1e-12 / ms)
        c_inf: Dimensionless = alpha_c / c_sum
        c_kca_new: Dimensionless = c_inf - (c_inf - self.c_kca) * exp(-(dt * (c_sum / 4.5)))

        diff0_finalize: Rate = ku * self.cai * (1.0 - self.Oc) - kr * self.Oc
        Oc1_finalize: Dimensionless = clip(self.Oc + half_dt * diff0_finalize, 0.0, 1.0)
        cai1_finalize: Concentration = maximum(self.cai + half_dt * (-ca_budget_prepare * cai_factor - nb_Bi * diff0_finalize), 1e-9 * mM)
        diff1_finalize: Rate = ku * cai1_finalize * (1.0 - Oc1_finalize) - kr * Oc1_finalize
        Oc_new: Dimensionless = clip(self.Oc + half_dt * diff1_finalize, 0.0, 1.0)
        cai_new: Concentration = maximum(self.cai + half_dt * (-ca_budget_prepare * cai_factor - nb_Bi * diff1_finalize), 1e-9 * mM)
        cao_new: Concentration = maximum(self.cao + half_dt * dcao_dt_prepare, 1e-9 * mM)

        total_outward_current: CurrentDensity = self.I_ion + self.i_kca + background_current
        explicit_outward_current: CurrentDensity = self.i_kca + background_current

        cai_safe_diag: Concentration = maximum(self.cai, 1e-9 * mM)
        cao_safe_diag: Concentration = maximum(self.cao, 1e-9 * mM)
        eca_diag: Voltage = 0.5 * rt_f * log(cao_safe_diag / cai_safe_diag) - 78.7 * mV
        fnk_diag: Dimensionless = (self.Vm_prev + 150.0 * mV) / (self.Vm_prev + 200.0 * mV)
        ink_diag: CurrentDensity = INaKmax_na_ko * fnk_diag
        v_norm_diag: Dimensionless = self.Vm_prev / (1.0 * mV)
        cai_norm_diag: Dimensionless = self.cai / (1.0 * mM)
        cao_norm_diag: Dimensionless = self.cao / (1.0 * mM)
        dfin_diag: Dimensionless = (nai_ion ** 3) * cao_norm_diag * exp((r_naca - 2.0) * gamma * v_norm_diag * fac_V)
        dfout_diag: Dimensionless = (nao_ion ** 3) * cai_norm_diag * exp((r_naca - 2.0) * (gamma - 1.0) * v_norm_diag * fac_V)
        s_nca_diag: Dimensionless = 1.0 + DNaCa * (cai_norm_diag * (nao_ion ** 3) + cao_norm_diag * (nai_ion ** 3))
        inca_diag: CurrentDensity = KNaCa * (dfin_diag - dfout_diag) / maximum(s_nca_diag, 1e-12)
        i_eca_corr_diag: CurrentDensity = (self.g_can + self.g_cat + gbca) * (eca_static - eca_diag)
        i_nak_corr_diag: CurrentDensity = INaKmax_na_ko * fnk_diag - I_NaKpump_bg
        i_cap_corr_diag: CurrentDensity = ICaPmax * (self.cai / (self.cai + KmCa)) - I_CaPump_bg
        i_nca_corr_diag: CurrentDensity = inca_diag - inca0
        ca_correction_diag: CurrentDensity = (
            i_eca_corr_diag + i_nak_corr_diag + i_cap_corr_diag + i_nca_corr_diag
        )
        I_na_total: CurrentDensity = (
            self.g_naf * (self.Vm_prev - self.ena)
            + self.g_nas * (self.Vm_prev - self.ena)
            + self.g_leak_na * (self.Vm_prev - self.ena)
            + 3.0 * ink_diag
            + 3.0 * inca_diag
        )
        I_k_total: CurrentDensity = (
            self.g_kd * (self.Vm_prev - ek)
            + self.g_ka * (self.Vm_prev - ek)
            + self.g_kds * (self.Vm_prev - ek)
            + self.g_kca * (self.Vm_prev - ek)
            - 2.0 * ink_diag
        )
        I_ca_total: CurrentDensity = (
            self.g_can * (self.Vm_prev - eca_diag)
            + self.g_cat * (self.Vm_prev - eca_diag)
            + self.g_leak_ca * (self.Vm_prev - eca_diag)
            + ICaPmax * (self.cai / (self.cai + KmCa))
            - 2.0 * inca_diag
        )
        I_total_rhs: CurrentDensity = self.I_ion + self.g_kca * (self.Vm_prev - ek) + background_current + ca_correction_diag
        self.keep(
            cai_mid,
            Oc_mid,
            cao_mid,
            c_kca_new,
            cai_new,
            Oc_new,
            cao_new,
            total_outward_current,
            explicit_outward_current,
            ca_correction,
            I_na_total,
            I_k_total,
            I_ca_total,
            I_total_rhs,
        )
