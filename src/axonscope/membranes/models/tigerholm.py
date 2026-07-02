"""Tigerholm C-fiber membrane equations written as plain Python source."""

from __future__ import annotations

from axonscope.membranes.model import Model, currents, initials, mechanism, state, step
from axonscope.membranes.math import (
    alpha_from_inf_tau,
    beta_from_inf_tau,
    exp,
    log,
    maximum,
    q10,
    where,
)
from axonscope.membranes.types import (
    Concentration,
    ConductanceDensity,
    CurrentDensity,
    Dimensionless,
    Gate,
    Length,
    Rate,
    Temperature,
    Time,
    Voltage,
)
from axonscope.utils.units import cm2, degC, mM, mM_per_uA_cm2_ms, mS, mV, ms, uA, um


class Tigerholm(Model):
    """Tigerholm et al. 2014 C-fiber membrane equations."""

    model_kind = "tigerholm"
    parameter_aliases = {
        "diameter": "diameter_um",
        "temperature": "celsius",
    }
    metadata = {
        "display_name": "Tigerholm C-fiber",
        "family": "tigerholm",
        "source_reference": "Tigerholm et al. 2014 C-fiber membrane model",
        "final_gate_update": "post_solve_voltage",
        "stateful": "concentrations",
        "gate_trace_observables": ("w_kna",),
        "temperature_reference": "mixed channel-specific q10 references",
        "current_sign_convention": "outward_positive",
        "notes": (
            "Stateful Na/K concentration dynamics with static public component "
            "currents and dynamic correction terms for the solver."
        ),
    }
    nai_fixed: Concentration = 11.4 * mM
    pump_ko: Concentration = 5.6 * mM
    diameter_um: Length = 1.0 * um
    celsius: Temperature = 37.0 * degC
    ena: Voltage = 71.5 * mV
    ek: Voltage = -87.0 * mV
    gbar_nav17: ConductanceDensity = 106.64 * mS / cm2
    gbar_nav18: ConductanceDensity = 242.71 * mS / cm2
    gbar_nav19: ConductanceDensity = 0.094779 * mS / cm2
    gbar_ks: ConductanceDensity = 6.9733 * mS / cm2
    gbar_kf: ConductanceDensity = 12.756 * mS / cm2
    gbar_kdr: ConductanceDensity = 18.002 * mS / cm2
    gbar_h: ConductanceDensity = 2.5377 * mS / cm2
    gbar_kna: ConductanceDensity = 0.42 * mS / cm2
    pump_smalla: CurrentDensity = -4.7891 * uA / cm2

    nai: Concentration = state(
        11.4 * mM,
        description="Intracellular sodium concentration.",
    )
    nao: Concentration = state(
        154.0 * mM,
        description="Extracellular/periaxonal sodium concentration.",
    )
    ki: Concentration = state(
        144.9 * mM,
        description="Intracellular potassium concentration.",
    )
    ko: Concentration = state(
        5.6 * mM,
        description="Extracellular/periaxonal potassium concentration.",
    )

    @initials(updates={"nai": "nai_initial", "ko": "ko_initial"})
    def initials(self):
        """Parameter-dependent initial Na/K concentration states."""

        nai_initial: Concentration = self.nai_fixed
        ko_initial: Concentration = self.pump_ko
        self.keep(nai_initial, ko_initial)


    @mechanism("nav17")
    def nav17(self, Vm: Voltage):
        """Nav1.7 / NaTTXs activation, fast inactivation, and slow inactivation rates."""

        qt_nav17: Dimensionless = q10(2.5, self.celsius, 21.0 * degC)
        alpha_m_nattxs: Rate = (15.5 / ms) / (
            1.0 + exp((Vm - 5.0 * mV) / (-12.08 * mV))
        ) * qt_nav17
        beta_m_nattxs: Rate = (35.2 / ms) / (
            1.0 + exp((Vm + 72.7 * mV) / (16.7 * mV))
        ) * qt_nav17
        alpha_h_nattxs: Rate = (0.38685 / ms) / (
            1.0 + exp((Vm + 122.35 * mV) / (15.29 * mV))
        ) * qt_nav17
        beta_h_nattxs: Rate = maximum(
            (-0.00283 / ms)
            + (2.00283 / ms)
            / (1.0 + exp((Vm + 5.5266 * mV) / (-12.70195 * mV))),
            0.0 / ms,
        ) * qt_nav17
        alpha_s_nattxs: Rate = (
            (0.00003 / ms)
            + (0.00092 / ms)
            / (1.0 + exp((Vm + 93.9 * mV) / (16.6 * mV)))
        ) * qt_nav17
        beta_s_nattxs: Rate = (
            (132.05 / ms)
            - (132.05 / ms)
            / (1.0 + exp((Vm - 384.9 * mV) / (28.5 * mV)))
        ) * qt_nav17
        self.keep(
            alpha_m_nattxs,
            beta_m_nattxs,
            alpha_h_nattxs,
            beta_h_nattxs,
            alpha_s_nattxs,
            beta_s_nattxs,
        )


    @mechanism("nav18")
    def nav18(self, Vm: Voltage):
        """Nav1.8 activation plus h/s/u inactivation rates."""

        qt_nav18: Dimensionless = q10(2.5, self.celsius, 22.0 * degC)
        alpha_m_nav18: Rate = (
            (2.85 / ms)
            - (2.839 / ms) / (1.0 + exp((Vm - 1.159 * mV) / (13.95 * mV)))
        ) * qt_nav18
        beta_m_nav18: Rate = (7.6205 / ms) / (
            1.0 + exp((Vm + 46.463 * mV) / (8.8289 * mV))
        ) * qt_nav18
        h_inf_nav18: Dimensionless = 1.0 / (
            1.0 + exp((Vm + 32.2 * mV) / (4.0 * mV))
        )
        tau_h_nav18: Time = 1.218 * ms + 42.043 * ms * exp(
            -(((Vm + 38.1 * mV) / (15.19 * mV)) ** 2) / 2.0
        )
        alpha_h_nav18: Rate = alpha_from_inf_tau(h_inf_nav18, tau_h_nav18) * qt_nav18
        beta_h_nav18: Rate = beta_from_inf_tau(h_inf_nav18, tau_h_nav18) * qt_nav18
        s_inf_nav18: Dimensionless = 1.0 / (
            1.0 + exp((Vm + 45.0 * mV) / (8.0 * mV))
        )
        alpha_s_nav18_raw: Rate = ((0.001 * 5.4203) / ms) / (
            1.0 + exp((Vm + 79.816 * mV) / (16.269 * mV))
        )
        beta_s_nav18_raw: Rate = ((0.001 * 5.0757) / ms) / (
            1.0 + exp(-((Vm + 15.968 * mV) / (11.542 * mV)))
        )
        sum_s_nav18: Rate = maximum(alpha_s_nav18_raw + beta_s_nav18_raw, 1e-12 / ms)
        alpha_s_nav18: Rate = s_inf_nav18 * sum_s_nav18 * qt_nav18
        beta_s_nav18: Rate = (1.0 - s_inf_nav18) * sum_s_nav18 * qt_nav18
        u_inf_nav18: Dimensionless = 1.0 / (
            1.0 + exp((Vm + 51.0 * mV) / (8.0 * mV))
        )
        alpha_u_nav18_raw: Rate = ((0.0002 * 2.0434) / ms) / (
            1.0 + exp((Vm + 67.499 * mV) / (19.51 * mV))
        )
        beta_u_nav18_raw: Rate = ((0.0002 * 1.9952) / ms) / (
            1.0 + exp(-((Vm + 30.963 * mV) / (14.792 * mV)))
        )
        sum_u_nav18: Rate = maximum(alpha_u_nav18_raw + beta_u_nav18_raw, 1e-12 / ms)
        alpha_u_nav18: Rate = u_inf_nav18 * sum_u_nav18 * qt_nav18
        beta_u_nav18: Rate = (1.0 - u_inf_nav18) * sum_u_nav18 * qt_nav18
        self.keep(
            alpha_m_nav18,
            beta_m_nav18,
            alpha_h_nav18,
            beta_h_nav18,
            alpha_s_nav18,
            beta_s_nav18,
            alpha_u_nav18,
            beta_u_nav18,
        )


    @mechanism("nav19")
    def nav19(self, Vm: Voltage):
        """Nav1.9 activation and inactivation rates."""

        qt_nav19: Dimensionless = q10(2.5, self.celsius, 21.0 * degC)
        alpha_m_nav19: Rate = (1.032 / ms) / (
            1.0 + exp((Vm + 6.99 * mV) / (-14.87115 * mV))
        ) * qt_nav19
        beta_m_nav19: Rate = (5.79 / ms) / (
            1.0 + exp((Vm + 130.4 * mV) / (22.9 * mV))
        ) * qt_nav19
        alpha_h_nav19: Rate = (0.06435 / ms) / (
            1.0 + exp((Vm + 73.26415 * mV) / (3.71928 * mV))
        ) * qt_nav19
        beta_h_nav19: Rate = (0.13496 / ms) / (
            1.0 + exp((Vm + 10.27853 * mV) / (-9.09334 * mV))
        ) * qt_nav19
        alpha_s_nav19: Rate = (0.00000016 / ms) * exp(-(Vm / (12.0 * mV))) * qt_nav19
        beta_s_nav19: Rate = (0.0005 / ms) / (
            1.0 + exp(-((Vm + 32.0 * mV) / (23.0 * mV)))
        ) * qt_nav19
        self.keep(
            alpha_m_nav19,
            beta_m_nav19,
            alpha_h_nav19,
            beta_h_nav19,
            alpha_s_nav19,
            beta_s_nav19,
        )


    @mechanism("ks")
    def ks(self, Vm: Voltage):
        """Slow potassium Ks fast/slow gate rates."""

        qt_ks: Dimensionless = q10(3.3, self.celsius, 21.0 * degC)
        n_inf_ks: Dimensionless = 1.0 / (
            1.0 + exp(-((Vm + 30.0 * mV) / (6.0 * mV)))
        )
        tau_ns_ks: Time = where(
            Vm >= -60.0 * mV,
            13.0 * ms * (Vm / (1.0 * mV)) + 1000.0 * ms,
            219.0 * ms,
        )
        a_ks: Rate = (0.00395 / ms) * exp((Vm + 30.0 * mV) / (40.0 * mV))
        b_ks: Rate = (0.00395 / ms) * exp(-((Vm + 30.0 * mV) / (20.0 * mV)))
        tau_nf_ks: Time = 1.0 / maximum(a_ks + b_ks, 1e-12 / ms)
        alpha_ns_ks: Rate = alpha_from_inf_tau(n_inf_ks, tau_ns_ks) * qt_ks
        beta_ns_ks: Rate = beta_from_inf_tau(n_inf_ks, tau_ns_ks) * qt_ks
        alpha_nf_ks: Rate = alpha_from_inf_tau(n_inf_ks, tau_nf_ks) * qt_ks
        beta_nf_ks: Rate = beta_from_inf_tau(n_inf_ks, tau_nf_ks) * qt_ks
        self.keep(
            alpha_ns_ks,
            beta_ns_ks,
            alpha_nf_ks,
            beta_nf_ks,
        )


    @mechanism("kf")
    def kf(self, Vm: Voltage):
        """Fast potassium Kf activation and inactivation rates."""

        qt_kf: Dimensionless = q10(3.3, self.celsius, 23.0 * degC)
        m_inf_kf: Dimensionless = (
            1.0
            / (
                1.0
                + exp(-((Vm - (-5.4 * mV) + (-15.0 * mV)) / (16.4 * mV)))
            )
        ) ** 4
        tau_m_kf: Time = 0.25 * ms + 10.04 * ms * exp(
            -(((Vm + 24.67 * mV) / (34.8 * mV)) ** 2) / 2.0
        )
        h_inf_kf: Dimensionless = 1.0 / (
            1.0 + exp((Vm - (-49.9 * mV) + (-15.0 * mV)) / (4.6 * mV))
        )
        tau_h_kf: Time = maximum(
            20.0 * ms
            + 50.0 * ms * exp(-(((Vm + 40.0 * mV) / (40.0 * mV)) ** 2) / 2.0),
            5.0 * ms,
        )
        alpha_m_kf: Rate = alpha_from_inf_tau(m_inf_kf, tau_m_kf) * qt_kf
        beta_m_kf: Rate = beta_from_inf_tau(m_inf_kf, tau_m_kf) * qt_kf
        alpha_h_kf: Rate = alpha_from_inf_tau(h_inf_kf, tau_h_kf) * qt_kf
        beta_h_kf: Rate = beta_from_inf_tau(h_inf_kf, tau_h_kf) * qt_kf
        self.keep(
            alpha_m_kf,
            beta_m_kf,
            alpha_h_kf,
            beta_h_kf,
        )


    @mechanism("kdr")
    def kdr(self, Vm: Voltage):
        """Delayed rectifier potassium Kdr activation rates."""

        qt_kdr: Dimensionless = q10(3.3, self.celsius, 22.0 * degC)
        n_inf_kdr: Dimensionless = 1.0 / (
            1.0 + exp((Vm + 35.0 * mV - 10.0 * mV) / (-15.4 * mV))
        )
        tau_high_kdr: Time = 0.16 * ms + 0.8 * ms * exp(
            -0.0267 * ((Vm + 11.0 * mV) / (1.0 * mV))
        )
        exp1_kdr: Dimensionless = exp((Vm + 75.2 * mV) / (6.5 * mV))
        exp2_kdr: Dimensionless = exp((Vm - 131.5 * mV) / (-34.8 * mV))
        tau_low_kdr: Time = 1000.0 * ms * (
            0.000688 + 1.0 / maximum(exp1_kdr + exp2_kdr, 1e-12)
        )
        tau_kdr: Time = where(Vm >= -31.0 * mV, tau_high_kdr, tau_low_kdr)
        alpha_n_kdr: Rate = alpha_from_inf_tau(n_inf_kdr, tau_kdr) * qt_kdr
        beta_n_kdr: Rate = beta_from_inf_tau(n_inf_kdr, tau_kdr) * qt_kdr
        self.keep(
            alpha_n_kdr,
            beta_n_kdr,
        )


    @mechanism("hcn")
    def hcn(self, Vm: Voltage):
        """HCN mixed Na/K inward rectifier slow and fast gate rates."""

        qt_h: Dimensionless = q10(3.0, self.celsius, 22.0 * degC)
        n_inf_h: Dimensionless = 1.0 / (
            1.0 + exp((Vm + 87.2 * mV) / (9.7 * mV))
        )
        tau_ns_h: Time = where(
            Vm >= -70.0 * mV,
            300.0 * ms + 542.0 * ms * exp((Vm + 25.0 * mV) / (-20.0 * mV)),
            2500.0 * ms + 100.0 * ms * exp((Vm + 240.0 * mV) / (50.0 * mV)),
        )
        tau_nf_h: Time = where(
            Vm >= -70.0 * mV,
            140.0 * ms + 50.0 * ms * exp((Vm + 25.0 * mV) / (-20.0 * mV)),
            250.0 * ms + 12.0 * ms * exp((Vm + 240.0 * mV) / (50.0 * mV)),
        )
        alpha_ns_h: Rate = alpha_from_inf_tau(n_inf_h, tau_ns_h) * qt_h
        beta_ns_h: Rate = beta_from_inf_tau(n_inf_h, tau_ns_h) * qt_h
        alpha_nf_h: Rate = alpha_from_inf_tau(n_inf_h, tau_nf_h) * qt_h
        beta_nf_h: Rate = beta_from_inf_tau(n_inf_h, tau_nf_h) * qt_h
        self.keep(
            alpha_ns_h,
            beta_ns_h,
            alpha_nf_h,
            beta_nf_h,
        )


    @currents(
        outputs=(
            "I_na_nav17",
            "I_na_nav18",
            "I_na_nav19",
            "I_k_ks",
            "I_k_kf",
            "I_k_kdr",
            "I_k_kna",
            "I_k_h",
        ),
        observables=("g_na", "g_k", "w_kna"),
        internal=(
            "i_na_dyn",
            "i_k_dyn",
            "total_outward_current",
            "explicit_outward_current",
            "correction_current",
        ),
    )
    def currents(self,
        Vm: Voltage,
        m_nattxs: Gate,
        h_nattxs: Gate,
        s_nattxs: Gate,
        m_nav18: Gate,
        h_nav18: Gate,
        s_nav18: Gate,
        u_nav18: Gate,
        m_nav19: Gate,
        h_nav19: Gate,
        s_nav19: Gate,
        ns_ks: Gate,
        nf_ks: Gate,
        m_kf: Gate,
        h_kf: Gate,
        n_kdr: Gate,
        ns_h: Gate,
        nf_h: Gate,
        nai: Concentration,
        nao: Concentration,
        ki: Concentration,
        ko: Concentration,
    ):
        """Public channel currents, aggregate conductances, and ion-dynamics solver terms."""

        g_nav17: ConductanceDensity = self.gbar_nav17 * (m_nattxs**3) * h_nattxs * s_nattxs
        g_nav18: ConductanceDensity = self.gbar_nav18 * (m_nav18**3) * h_nav18 * s_nav18 * u_nav18
        g_nav19: ConductanceDensity = self.gbar_nav19 * m_nav19 * h_nav19 * s_nav19
        g_na: ConductanceDensity = g_nav17 + g_nav18 + g_nav19

        g_ks: ConductanceDensity = self.gbar_ks * (0.25 * ns_ks + 0.75 * nf_ks)
        g_kf: ConductanceDensity = self.gbar_kf * m_kf * h_kf
        g_kdr: ConductanceDensity = self.gbar_kdr * (n_kdr**4)
        g_h: ConductanceDensity = self.gbar_h * (0.5 * ns_h + 0.5 * nf_h)
        w_kna_static: Dimensionless = 0.37 / (1.0 + (38.7 / (self.nai_fixed / (1.0 * mM))) ** 3.5)
        g_kna_static: ConductanceDensity = self.gbar_kna * w_kna_static
        g_k_without_h: ConductanceDensity = g_ks + g_kf + g_kdr
        eh_static: Voltage = 0.5 * (self.ena + self.ek)

        pump_f_nai_static: Dimensionless = 1.62 / (
            1.0 + (6.7 / ((self.nai_fixed / (1.0 * mM)) + 8.0)) ** 3
        ) + 1.0 / (1.0 + (67.6 / ((self.nai_fixed / (1.0 * mM)) + 8.0)) ** 3)
        pump_static: CurrentDensity = (
            -0.5
            * self.pump_smalla
            / ((1.0 + 1.0 / (self.pump_ko / (1.0 * mM))) ** 2)
            * pump_f_nai_static
        )

        rt_over_f: Voltage = (
            8.314
            * ((self.celsius + 273.15 * degC) / (1.0 * degC))
            / 96485.0
            * 1000.0
            * mV
        )
        e_na_dyn: Voltage = rt_over_f * log(nao / nai)
        e_k_dyn: Voltage = rt_over_f * log(ko / ki)
        w_kna: Dimensionless = 0.37 / (1.0 + (38.7 / (nai / (1.0 * mM))) ** 3.5)
        g_kna_dyn: ConductanceDensity = self.gbar_kna * w_kna

        I_na_nav17: CurrentDensity = g_nav17 * (Vm - self.ena)
        I_na_nav18: CurrentDensity = g_nav18 * (Vm - self.ena)
        I_na_nav19: CurrentDensity = g_nav19 * (Vm - self.ena)
        I_k_ks: CurrentDensity = g_ks * (Vm - self.ek)
        I_k_kf: CurrentDensity = g_kf * (Vm - self.ek)
        I_k_kdr: CurrentDensity = g_kdr * (Vm - self.ek)
        I_k_kna: CurrentDensity = g_kna_static * (Vm - self.ek)
        I_k_h: CurrentDensity = g_h * (Vm - eh_static)

        i_na_dyn: CurrentDensity = g_na * (Vm - e_na_dyn)
        i_k_dyn: CurrentDensity = (
            g_k_without_h * (Vm - e_k_dyn)
            + 0.5 * g_h * (Vm - e_k_dyn)
            + g_kna_dyn * (Vm - e_k_dyn)
        )
        pump_f_nai_dyn: Dimensionless = 1.62 / (
            1.0 + (6.7 / ((nai / (1.0 * mM)) + 8.0)) ** 3
        ) + 1.0 / (1.0 + (67.6 / ((nai / (1.0 * mM)) + 8.0)) ** 3)
        pump_dyn: CurrentDensity = (
            -0.5 * self.pump_smalla / ((1.0 + 1.0 / (ko / (1.0 * mM))) ** 2) * pump_f_nai_dyn
        )

        diameter_value_um: Dimensionless = self.diameter_um / (1.0 * um)
        nai_factor: Dimensionless = (40.0 / (96485.0 * diameter_value_um)) * mM_per_uA_cm2_ms
        nao_factor: Dimensionless = (10.0 / (96485.0 * 0.029)) * mM_per_uA_cm2_ms
        ki_factor: Dimensionless = (40.0 / (96485.0 * diameter_value_um)) * mM_per_uA_cm2_ms
        ko_factor: Dimensionless = (10.0 / (96485.0 * 0.029)) * mM_per_uA_cm2_ms
        nao_inf: Concentration = 154.0 * mM
        ko_inf: Concentration = self.pump_ko

        delta_e_na: Voltage = e_na_dyn - self.ena
        delta_e_k: Voltage = e_k_dyn - self.ek
        correction_current: CurrentDensity = (
            -g_na * delta_e_na
            - (g_k_without_h + g_kna_dyn) * delta_e_k
            - 0.5 * g_h * (delta_e_na + delta_e_k)
            + (g_kna_dyn - g_kna_static) * (Vm - self.ek)
            + (pump_dyn - pump_static)
        )
        ion_current: CurrentDensity = (
            g_na * (Vm - self.ena)
            + g_k_without_h * (Vm - self.ek)
            + g_kna_static * (Vm - self.ek)
            + g_h * (Vm - eh_static)
        )
        total_outward_current: CurrentDensity = ion_current + pump_static
        explicit_outward_current: CurrentDensity = pump_static
        g_k: ConductanceDensity = g_k_without_h + g_kna_static + g_h
        self.keep(
            i_na_dyn,
            i_k_dyn,
            nai_factor,
            nao_factor,
            ki_factor,
            ko_factor,
            nao_inf,
            ko_inf,
            total_outward_current,
            explicit_outward_current,
            correction_current,
        )
        return (
            I_na_nav17,
            I_na_nav18,
            I_na_nav19,
            I_k_ks,
            I_k_kf,
            I_k_kdr,
            I_k_kna,
            I_k_h,
            g_na,
            g_k,
            w_kna,
        )


    @step
    def step(self, dt: Time):
        """Advance Na/K concentration states using dynamic Na and K current budgets."""

        nai_next: Concentration = self.nai - self.i_na_dyn * self.nai_factor * dt
        nao_next: Concentration = self.nao + dt * (
            self.i_na_dyn * self.nao_factor - (self.nao - self.nao_inf) * ((1e-4 / 0.029) / ms)
        )
        ki_next: Concentration = self.ki - self.i_k_dyn * self.ki_factor * dt
        ko_next: Concentration = self.ko + dt * (
            self.i_k_dyn * self.ko_factor - (self.ko - self.ko_inf) * ((1e-4 / 0.029) / ms)
        )
        self.keep(
            nai_next,
            nao_next,
            ki_next,
            ko_next,
            self.total_outward_current,
            self.explicit_outward_current,
            self.correction_current,
        )
