from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp

from axonscope.utils.settings import dtype
from axonscope.channel_models.base_channel_model import (
    CompositeICM,
    MembraneStateSpec,
    MembraneStepPlan,
)
from axonscope.channel_models.passive import PassiveICM
from axonscope.channel_models.nav17 import NaV17ICM
from axonscope.channel_models.nav18 import NaV18ICM
from axonscope.channel_models.nav19 import NaV19ICM
from axonscope.channel_models.ks_tigerholm import KSlowICM
from axonscope.channel_models.kf_tigerholm import KFastICM
from axonscope.channel_models.kdr_tigerholm import KDRTigerICM
from axonscope.channel_models.hcn import HCNICM
from axonscope.channel_models.nakpump import NaKPumpICM
from axonscope.channel_models.schild import (
    LeakSchildICM,
    NafSchildICM,
    NasSchildICM,
    Naf97ICM,
    Nas97ICM,
    KdSchildICM,
    KaSchildICM,
    KdsSchildICM,
    CaNICM,
    CaTICM,
)


def _schild_nernst(T_K, z, conc_out, conc_in, R=8314.0, F=96500.0):
    """Schild Nernst potential [mV]: (R*T/(z*F)) * log(out/in)."""
    return (R * T_K / (z * F)) * float(jnp.log(jnp.array(conc_out / conc_in)).item())


class TigerholmCompositeICM(CompositeICM):
    """Tigerholm membrane with Na/K concentration dynamics attached to the channel model."""

    def __init__(
        self,
        *,
        diameter_um: float,
        celsius: float = 37.0,
        ena: float = 71.5,
        ek: float = -87.0,
        gbar_nav17: float = 0.10664,
        gbar_nav18: float = 0.24271,
        gbar_nav19: float = 9.4779e-05,
        gbar_ks: float = 0.0069733,
        gbar_kf: float = 0.012756,
        gbar_kdr: float = 0.018002,
        gbar_h: float = 0.0025377,
        gbar_kna: float = 0.00042,
        nai_fixed: float = 11.4,
        pump_smalla: float = -0.0047891,
        pump_ko: float = 5.6,
    ) -> None:
        w_kna0 = 0.37 / (1.0 + (38.7 / nai_fixed) ** 3.5)
        g_kna_eff = gbar_kna * w_kna0
        pump = NaKPumpICM(smalla=pump_smalla, ko=pump_ko, nai=nai_fixed)

        super().__init__(
            [
                NaV17ICM(gbar=gbar_nav17, ena=ena, celsius=celsius),
                NaV18ICM(gbar=gbar_nav18, ena=ena, celsius=celsius),
                NaV19ICM(gbar=gbar_nav19, ena=ena, celsius=celsius),
                KSlowICM(gbar=gbar_ks, ek=ek, celsius=celsius),
                KFastICM(gbar=gbar_kf, ek=ek, celsius=celsius),
                KDRTigerICM(gbar=gbar_kdr, ek=ek, celsius=celsius),
                HCNICM(gbar=gbar_h, ena=ena, ek=ek, celsius=celsius),
                PassiveICM(Rm=1.0 / g_kna_eff if g_kna_eff > 0 else 1e12, EL=ek),
                pump,
            ]
        )

        self.nai0 = float(nai_fixed)
        self.nao0 = 154.0
        self.ko0 = float(pump_ko)
        self.ki0 = 144.9

        self._nai_factor = float(dtype(40.0 / (96485.0 * diameter_um)))
        theta_naoi = 0.029
        self._nao_factor = float(10.0 / (96485.0 * theta_naoi))
        self._naoinf = float(self.nao0)
        self._nao_tau_inv = float(1e-4 / theta_naoi)

        self._ena = float(ena)
        self._ek = float(ek)
        self._gbar_kna = float(gbar_kna)
        self._kna_w0 = float(w_kna0)
        self._pump_I_bg_static = float(pump._I_bg_uA)
        self._pump_smalla = float(pump_smalla)
        self._pump_b1 = 1.0

        theta_koi = 0.029
        self._koinf = float(pump_ko)
        self._ki = float(self.ki0)
        self._RT_F = float(8.314 * (celsius + 273.15) / 96485.0 * 1000.0)
        self._ko_factor = float(10.0 / (96485.0 * theta_koi))
        self._ko_tau_inv = float(1e-4 / theta_koi)
        self._ki_factor = float(dtype(40.0 / (96485.0 * diameter_um)))

    def membrane_state_specs(self) -> tuple[MembraneStateSpec, ...]:
        return (
            MembraneStateSpec("nai"),
            MembraneStateSpec("nao"),
            MembraneStateSpec("ki"),
            MembraneStateSpec("ko"),
        )

    def gate_names(self) -> tuple[str, ...]:
        return (
            "m_nav18",
            "h_nav18",
            "s_nav18",
            "u_nav18",
            "m_nav19",
            "h_nav19",
            "s_nav19",
            "m_nattxs",
            "h_nattxs",
            "s_nattxs",
            "n_kdr",
            "m_kf",
            "h_kf",
            "ns_ks",
            "nf_ks",
            "w_kna",
            "ns_h",
            "nf_h",
        )

    def gate_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        nav17 = gates[:, self.cum_sizes[0] : self.cum_sizes[1]]
        nav18 = gates[:, self.cum_sizes[1] : self.cum_sizes[2]]
        nav19 = gates[:, self.cum_sizes[2] : self.cum_sizes[3]]
        ks = gates[:, self.cum_sizes[3] : self.cum_sizes[4]]
        kf = gates[:, self.cum_sizes[4] : self.cum_sizes[5]]
        kdr = gates[:, self.cum_sizes[5] : self.cum_sizes[6]]
        hcn = gates[:, self.cum_sizes[6] : self.cum_sizes[7]]

        if state:
            nai = state[0]
        else:
            nai = jnp.full((gates.shape[0],), self.nai0, dtype=dtype)
        w_kna = dtype(0.37) / (dtype(1.0) + (dtype(38.7) / nai) ** dtype(3.5))

        return jnp.concatenate(
            [
                nav18,
                nav19,
                nav17,
                kdr,
                kf,
                ks,
                w_kna[:, None],
                hcn,
            ],
            axis=1,
        )

    def conductance_names(self) -> tuple[str, ...]:
        return ("g_na", "g_k")

    def current_names(self) -> tuple[str, ...]:
        return ("I_na", "I_k")

    def conductance_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        G_Na = jnp.zeros(gates.shape[0], dtype=dtype)
        for i in [0, 1, 2]:
            model = self.models[i]
            i0, i1 = self.cum_sizes[i], self.cum_sizes[i + 1]
            G_Na = G_Na + model.g_funcs(gates[:, i0:i1], model.g_bar)[:, 0]

        G_K = jnp.zeros(gates.shape[0], dtype=dtype)
        for i in [3, 4, 5]:
            model = self.models[i]
            i0, i1 = self.cum_sizes[i], self.cum_sizes[i + 1]
            G_K = G_K + model.g_funcs(gates[:, i0:i1], model.g_bar)[:, 0]

        i0, i1 = self.cum_sizes[6], self.cum_sizes[7]
        g_hcn = self.models[6].g_funcs(gates[:, i0:i1], self.models[6].g_bar)[:, 0]
        G_K = G_K + dtype(0.5) * g_hcn

        if state:
            nai = state[0]
        else:
            nai = jnp.full((gates.shape[0],), self.nai0, dtype=dtype)
        w_dyn = dtype(0.37) / (dtype(1.0) + (dtype(38.7) / nai) ** dtype(3.5))
        G_K = G_K + dtype(self._gbar_kna) * w_dyn * dtype(1e3)
        return jnp.stack([G_Na, G_K], axis=1)

    def ionic_current_trace_matrix(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        if state:
            nai, nao, ki, ko = state
        else:
            nai = jnp.full((gates.shape[0],), self.nai0, dtype=dtype)
            nao = jnp.full((gates.shape[0],), self.nao0, dtype=dtype)
            ki = jnp.full((gates.shape[0],), self.ki0, dtype=dtype)
            ko = jnp.full((gates.shape[0],), self.ko0, dtype=dtype)
        I_Na = self.compute_I_Na_dyn(V_mV, gates, nai, nao)
        I_K = self.compute_I_K_dyn(V_mV, gates, nai, ko, ki)
        return jnp.stack([I_Na, I_K], axis=1)

    def init_membrane_state(
        self, Nx: int, dtype_local: jnp.dtype, V0_mV: jnp.ndarray
    ) -> tuple[jnp.ndarray, ...]:
        _ = V0_mV
        nai0_vec = jnp.full((Nx,), self.nai0, dtype=dtype_local)
        nao0_vec = jnp.full((Nx,), self.nao0, dtype=dtype_local)
        ki0_vec = jnp.full((Nx,), self.ki0, dtype=dtype_local)
        ko0_vec = jnp.full((Nx,), self.ko0, dtype=dtype_local)
        return nai0_vec, nao0_vec, ki0_vec, ko0_vec

    def prepare_membrane_step(
        self,
        V_mV: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state: tuple[jnp.ndarray, ...],
        dt: float,
        I_ion: jnp.ndarray,
        I_background: jnp.ndarray,
    ) -> MembraneStepPlan:
        _ = gates_prev
        nai, nao, ki, ko = state
        I_Na_dyn = self.compute_I_Na_dyn(V_mV, gates_new, nai, nao)
        I_K_dyn = self.compute_I_K_dyn(V_mV, gates_new, nai, ko, ki)
        nai_new = nai - I_Na_dyn * self._nai_factor * dt
        nao_new = nao + dt * (
            I_Na_dyn * self._nao_factor - (nao - self._naoinf) * self._nao_tau_inv
        )
        ki_new = ki - I_K_dyn * self._ki_factor * dt
        ko_new = ko + dt * (
            I_K_dyn * self._ko_factor - (ko - self._koinf) * self._ko_tau_inv
        )
        I_corr = self.dynamics_correction(V_mV, gates_new, nai, ko, nao, ki)
        return MembraneStepPlan(
            state=(nai_new, nao_new, ki_new, ko_new),
            linearization_gates=gates_new,
            total_outward_current=I_background + I_ion,
            explicit_outward_current=I_background,
            correction_current=I_corr,
        )

    def compute_I_Na(self, V_mV: jnp.ndarray, gates: jnp.ndarray) -> jnp.ndarray:
        I_Na = jnp.zeros(V_mV.shape[0], dtype=dtype)
        for i in range(3):
            model = self.models[i]
            i0, i1 = self.cum_sizes[i], self.cum_sizes[i + 1]
            g_i = model.g_funcs(gates[:, i0:i1], model.g_bar)[:, 0]
            I_Na = I_Na + g_i * (V_mV - dtype(getattr(model, "ena", self._ena)))
        return I_Na

    def compute_I_Na_dyn(
        self, V_mV: jnp.ndarray, gates: jnp.ndarray, nai: jnp.ndarray, nao: jnp.ndarray
    ) -> jnp.ndarray:
        E_Na_dyn = dtype(self._RT_F) * jnp.log(nao / nai)
        I_Na = jnp.zeros(V_mV.shape[0], dtype=dtype)
        for i in range(3):
            model = self.models[i]
            i0, i1 = self.cum_sizes[i], self.cum_sizes[i + 1]
            g_i = model.g_funcs(gates[:, i0:i1], model.g_bar)[:, 0]
            I_Na = I_Na + g_i * (V_mV - E_Na_dyn)
        return I_Na

    def _I_K_from_ek(
        self, V_mV: jnp.ndarray, gates: jnp.ndarray, nai: jnp.ndarray, E_K: jnp.ndarray
    ) -> jnp.ndarray:
        I_K = jnp.zeros(V_mV.shape[0], dtype=dtype)
        for i in [3, 4, 5]:
            model = self.models[i]
            i0, i1 = self.cum_sizes[i], self.cum_sizes[i + 1]
            g_i = model.g_funcs(gates[:, i0:i1], model.g_bar)[:, 0]
            I_K = I_K + g_i * (V_mV - E_K)
        i0, i1 = self.cum_sizes[6], self.cum_sizes[7]
        g_hcn = self.models[6].g_funcs(gates[:, i0:i1], self.models[6].g_bar)[:, 0]
        I_K = I_K + dtype(0.5) * g_hcn * (V_mV - E_K)
        w_dyn = dtype(0.37) / (dtype(1.0) + (dtype(38.7) / nai) ** dtype(3.5))
        G_kna_dyn = dtype(self._gbar_kna) * w_dyn * dtype(1e3)
        I_K = I_K + G_kna_dyn * (V_mV - E_K)
        return I_K

    def compute_I_K_total(self, V_mV: jnp.ndarray, gates: jnp.ndarray, nai: jnp.ndarray) -> jnp.ndarray:
        return self._I_K_from_ek(V_mV, gates, nai, dtype(self._ek))

    def compute_I_K_dyn(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        nai: jnp.ndarray,
        ko: jnp.ndarray,
        ki: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        ki_ref = dtype(self._ki) if ki is None else ki
        E_K_dyn = dtype(self._RT_F) * jnp.log(ko / ki_ref)
        return self._I_K_from_ek(V_mV, gates, nai, E_K_dyn)

    def dynamics_correction(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        nai: jnp.ndarray,
        ko: jnp.ndarray,
        nao: Optional[jnp.ndarray] = None,
        ki: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        nao_ref = dtype(self.nao0) if nao is None else nao
        ki_ref = dtype(self._ki) if ki is None else ki
        E_Na_dyn = dtype(self._RT_F) * jnp.log(nao_ref / nai)
        E_K_dyn = dtype(self._RT_F) * jnp.log(ko / ki_ref)
        delta_E_Na = E_Na_dyn - dtype(self._ena)
        delta_E_K = E_K_dyn - dtype(self._ek)

        G_Na = jnp.zeros(V_mV.shape[0], dtype=dtype)
        for i in [0, 1, 2]:
            model = self.models[i]
            i0, i1 = self.cum_sizes[i], self.cum_sizes[i + 1]
            g_i = model.g_funcs(gates[:, i0:i1], model.g_bar)[:, 0]
            G_Na = G_Na + g_i

        G_K_pure = jnp.zeros(V_mV.shape[0], dtype=dtype)
        for i in [3, 4, 5]:
            model = self.models[i]
            i0, i1 = self.cum_sizes[i], self.cum_sizes[i + 1]
            g_i = model.g_funcs(gates[:, i0:i1], model.g_bar)[:, 0]
            G_K_pure = G_K_pure + g_i

        i0, i1 = self.cum_sizes[6], self.cum_sizes[7]
        g_hcn = self.models[6].g_funcs(gates[:, i0:i1], self.models[6].g_bar)[:, 0]

        w_dyn = dtype(0.37) / (dtype(1.0) + (dtype(38.7) / nai) ** dtype(3.5))
        G_kna_dyn = dtype(self._gbar_kna) * w_dyn * dtype(1e3)
        G_kna_stat = dtype(self._gbar_kna * self._kna_w0 * 1e3)

        I_ENa_corr = -G_Na * delta_E_Na
        I_EK_corr = -(G_K_pure + G_kna_dyn) * delta_E_K
        I_hcn_corr = -dtype(0.5) * g_hcn * (delta_E_Na + delta_E_K)
        I_kna_nai = (G_kna_dyn - G_kna_stat) * (V_mV - dtype(self._ek))

        f_nai = (
            dtype(1.62) / (dtype(1.0) + (dtype(6.7) / (nai + dtype(8.0))) ** dtype(3.0))
            + dtype(1.0) / (dtype(1.0) + (dtype(67.6) / (nai + dtype(8.0))) ** dtype(3.0))
        )
        denom_dyn = (dtype(1.0) + dtype(self._pump_b1) / ko) ** 2
        ikpump_dyn = dtype(self._pump_smalla) / denom_dyn * f_nai
        I_pump_dyn = dtype(-0.5) * ikpump_dyn * dtype(1e3)
        I_pump_stat = jnp.full(V_mV.shape[0], dtype(self._pump_I_bg_static))

        return I_ENa_corr + I_EK_corr + I_hcn_corr + I_kna_nai + (I_pump_dyn - I_pump_stat)


class SchildCompositeICM(CompositeICM):
    """Shared Schild 1994/1997 dynamic membrane with Ca pool dynamics."""

    _ku: float = 100.0
    _kr: float = 0.238
    _nb: int = 4
    _Bi: float = 0.001
    _fhspace_um: float = 0.03
    _txfer: float = 4511.0

    def __init__(
        self,
        *,
        models,
        diameter_um: float,
        temp_c: float,
        vinit_mV: float,
        nao: float = 154.0,
        nai: float = 8.9,
        ko: float = 5.4,
        ki: float = 145.0,
        nao_ion: float = 140.0,
        nai_ion: float = 10.0,
        ko_ion: float = 2.5,
        ki_ion: float = 54.4,
        cai0: float = 0.000117,
        cao0: float = 2.0,
        gbar_kca: float = 0.000141471,
        idx_can: int = 6,
        idx_cat: int = 7,
    ) -> None:
        super().__init__(models)

        T_K = float(temp_c + 273.15)
        R, F = 8314.0, 96500.0

        self._idx_can = int(idx_can)
        self._idx_cat = int(idx_cat)

        self._ek_schild = float(_schild_nernst(T_K, 1, ko, ki, R, F))
        self._ena_schild = float(_schild_nernst(T_K, 1, nao, nai, R, F))
        self._eca_static = float(_schild_nernst(T_K, 2, cao0, cai0, R, F)) - 78.7
        self._RT_F_schild = float(R * T_K / F)

        self._gbar_kca = float(gbar_kca * 1e3)
        self._q10_kca = float(2.30 ** ((temp_c - 22.85) / 10.0))

        self.cai0 = float(cai0)
        self.Oc0 = float(0.05)
        self.cao0 = float(cao0)
        self._cabath = float(cao0)

        a_cm = float(diameter_um / 2.0 * 1e-4)
        self._cai_factor = float(1e-3 / (96500.0 * a_cm))

        a_out_cm = float((diameter_um / 2.0 + self._fhspace_um / 2.0) * 1e-4)
        vol_peri_per_l = float(math.pi * (a_out_cm ** 2 - a_cm ** 2))
        sa_per_l = float(2.0 * math.pi * a_cm)
        self._cao_factor = float(1e-3 * sa_per_l / (2.0 * vol_peri_per_l * 96500.0))

        INaKmax = 0.009726135 * (1.16 ** ((22.85 - temp_c) / 10.0))
        fnk0 = (vinit_mV + 150.0) / (vinit_mV + 200.0)
        na_term = (nai_ion / (nai_ion + 5.46)) ** 3
        ko_term = (ko_ion / (ko_ion + 0.621)) ** 2
        ink0 = INaKmax * fnk0 * na_term * ko_term
        I_NaKpump = float(ink0 * 1e3)

        ICaPmax = 0.000859437 * (2.30 ** ((22.85 - temp_c) / 10.0))
        icap0 = ICaPmax * cai0 / (cai0 + 0.0005)
        I_CaPump = float(icap0 * 1e3)

        KNaCa = 1.27324e-6 * (2.20 ** ((22.85 - temp_c) / 10.0))
        gamma, r = 0.5, 3.0
        R_si = 8.314
        fac = F / (1000.0 * R_si * T_K)
        DFin = nai_ion**3 * cao0 * float(jnp.exp(jnp.array((r - 2) * gamma * vinit_mV * fac)).item())
        DFout = nao_ion**3 * cai0 * float(jnp.exp(jnp.array((r - 2) * (gamma - 1) * vinit_mV * fac)).item())
        S_nca = 1.0 + 0.0036 * (cai0 * nao_ion**3 + cao0 * nai_ion**3)
        inca0 = KNaCa * (DFin - DFout) / S_nca
        I_NaCaExch = float(inca0 * 1e3)
        self._inca0_uA = float(inca0 * 1e3)

        self._I_pump_bg = float(dtype(I_NaKpump + I_CaPump + I_NaCaExch))
        self._nao = float(nao_ion)
        self._nai = float(nai_ion)
        self._ko = float(ko_ion)
        self._ki_schild = float(ki_ion)
        self._KNaCa = float(KNaCa)
        self._KNaCa_uA = float(KNaCa * 1e3)
        self._DNaCa = float(0.0036)
        self._gamma = float(gamma)
        self._r = float(r)
        self._fac_V = float(fac)
        self._ICaPmax = float(ICaPmax)
        self._gbca_leak = float(self.models[0]._gbca)
        self._INaKmax_na_ko = float(INaKmax * na_term * ko_term * 1e3)
        self._I_NaKpump_bg = float(I_NaKpump)
        self._KmCa = float(0.0005)
        self._ICaPmax_uA = float(ICaPmax * 1e3)
        self._I_CaPump_bg = float(I_CaPump)

    def membrane_state_specs(self) -> tuple[MembraneStateSpec, ...]:
        return (
            MembraneStateSpec("cai"),
            MembraneStateSpec("Oc"),
            MembraneStateSpec("cao"),
            MembraneStateSpec("c_kca"),
        )

    def gate_names(self) -> tuple[str, ...]:
        names = [
            "d_can",
            "f1_can",
            "f2_can",
            "d_cat",
            "f_cat",
            "p_ka",
            "q_ka",
            "n_kd",
            "x_kds",
            "y1_kds",
            "m_naf",
            "h_naf",
            "m_nas",
            "h_nas",
        ]
        if isinstance(self, Schild94CompositeICM):
            return (
                "d_can",
                "f1_can",
                "f2_can",
                "d_cat",
                "f_cat",
                "p_ka",
                "q_ka",
                "n_kd",
                "x_kds",
                "y1_kds",
                "m_naf",
                "h_naf",
                "l_naf",
                "m_nas",
                "h_nas",
            )
        return tuple(names)

    def gate_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        naf = gates[:, self.cum_sizes[1] : self.cum_sizes[2]]
        nas = gates[:, self.cum_sizes[2] : self.cum_sizes[3]]
        kd = gates[:, self.cum_sizes[3] : self.cum_sizes[4]]
        ka = gates[:, self.cum_sizes[4] : self.cum_sizes[5]]
        kds = gates[:, self.cum_sizes[5] : self.cum_sizes[6]]
        can = gates[:, self.cum_sizes[self._idx_can] : self.cum_sizes[self._idx_can + 1]]
        cat = gates[:, self.cum_sizes[self._idx_cat] : self.cum_sizes[self._idx_cat + 1]]

        parts = [can, cat, ka, kd, kds, naf, nas]
        return jnp.concatenate(parts, axis=1)

    def conductance_names(self) -> tuple[str, ...]:
        return ("g_na", "g_k", "g_ca")

    def current_names(self) -> tuple[str, ...]:
        return ("I_na", "I_k", "I_ca")

    def conductance_trace_matrix(
        self,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        _ = state
        g_naf = self._model_g(1, gates)
        g_nas = self._model_g(2, gates)
        g_kd = self._model_g(3, gates)
        g_ka = self._model_g(4, gates)
        g_kds = self._model_g(5, gates)
        g_can = self._model_g(self._idx_can, gates)
        g_cat = self._model_g(self._idx_cat, gates)
        g_na = g_naf + g_nas + dtype(self.models[0]._gbna)
        g_k = g_kd + g_ka + g_kds
        g_ca = g_can + g_cat + dtype(self.models[0]._gbca)
        return jnp.stack([g_na, g_k, g_ca], axis=1)

    def ionic_current_trace_matrix(
        self,
        V_mV: jnp.ndarray,
        gates: jnp.ndarray,
        state: tuple[jnp.ndarray, ...] = (),
    ) -> jnp.ndarray:
        if state:
            cai, _, cao, c_kca = state
        else:
            cai = jnp.full((gates.shape[0],), self.cai0, dtype=dtype)
            cao = jnp.full((gates.shape[0],), self.cao0, dtype=dtype)
            c_kca = self.init_c_kca(V_mV, cai)

        eca_dyn = self._eca_dyn(cai, cao)
        g_naf = self._model_g(1, gates)
        g_nas = self._model_g(2, gates)
        g_kd = self._model_g(3, gates)
        g_ka = self._model_g(4, gates)
        g_kds = self._model_g(5, gates)
        g_can = self._model_g(self._idx_can, gates)
        g_cat = self._model_g(self._idx_cat, gates)

        I_kca = self.compute_I_kca(V_mV, c_kca)
        I_na = (
            g_naf * (V_mV - dtype(self._ena_schild))
            + g_nas * (V_mV - dtype(self._ena_schild))
            + dtype(self.models[0]._gbna) * (V_mV - dtype(self._ena_schild))
        )
        I_k = (
            g_kd * (V_mV - dtype(self._ek_schild))
            + g_ka * (V_mV - dtype(self._ek_schild))
            + g_kds * (V_mV - dtype(self._ek_schild))
            + I_kca
        )
        I_ca = (
            g_can * (V_mV - eca_dyn)
            + g_cat * (V_mV - eca_dyn)
            + dtype(self.models[0]._gbca) * (V_mV - eca_dyn)
        )
        return jnp.stack([I_na, I_k, I_ca], axis=1)

    def init_membrane_state(
        self, Nx: int, dtype_local: jnp.dtype, V0_mV: jnp.ndarray
    ) -> tuple[jnp.ndarray, ...]:
        cai0_vec = jnp.full((Nx,), self.cai0, dtype=dtype_local)
        Oc0_vec = jnp.full((Nx,), self.Oc0, dtype=dtype_local)
        cao0_vec = jnp.full((Nx,), self.cao0, dtype=dtype_local)
        c_kca0 = self.init_c_kca(V0_mV, cai0_vec)
        return cai0_vec, Oc0_vec, cao0_vec, c_kca0

    def init_c_kca(self, V0: jnp.ndarray, cai0_vec: jnp.ndarray) -> jnp.ndarray:
        a = self.alpha_c_kca(V0, cai0_vec)
        b = self.beta_c_kca(V0)
        return a / jnp.maximum(a + b, jnp.array(1e-12, dtype=a.dtype))

    def alpha_c_kca(self, V_mV: jnp.ndarray, cai: jnp.ndarray) -> jnp.ndarray:
        return dtype(750.0) * cai * jnp.exp((V_mV - dtype(10.0)) / dtype(12.0))

    def beta_c_kca(self, V_mV: jnp.ndarray) -> jnp.ndarray:
        return dtype(0.05) * jnp.exp((V_mV - dtype(10.0)) / dtype(-60.0))

    def compute_I_kca(self, V_mV: jnp.ndarray, c_kca: jnp.ndarray) -> jnp.ndarray:
        return dtype(self._gbar_kca) * c_kca * (V_mV - dtype(self._ek_schild))

    def _eca_dyn(self, cai: jnp.ndarray, cao: jnp.ndarray) -> jnp.ndarray:
        cai_safe = jnp.maximum(cai, dtype(1e-9))
        cao_safe = jnp.maximum(cao, dtype(1e-9))
        return dtype(0.5 * self._RT_F_schild) * jnp.log(cao_safe / cai_safe) - dtype(78.7)

    def compute_I_Ca_dyn(
        self, V_mV: jnp.ndarray, gates: jnp.ndarray, cai: jnp.ndarray, cao: jnp.ndarray
    ) -> jnp.ndarray:
        eca = self._eca_dyn(cai, cao)
        i0, i1 = self.cum_sizes[self._idx_can], self.cum_sizes[self._idx_can + 1]
        g_can = self.models[self._idx_can].g_funcs(gates[:, i0:i1], self.models[self._idx_can].g_bar)[:, 0]
        i0, i1 = self.cum_sizes[self._idx_cat], self.cum_sizes[self._idx_cat + 1]
        g_cat = self.models[self._idx_cat].g_funcs(gates[:, i0:i1], self.models[self._idx_cat].g_bar)[:, 0]
        return (g_can + g_cat) * (V_mV - eca)

    def compute_I_Ca_budget(
        self, V_mV: jnp.ndarray, gates: jnp.ndarray, cai: jnp.ndarray, cao: jnp.ndarray
    ) -> jnp.ndarray:
        I_chan = self.compute_I_Ca_dyn(V_mV, gates, cai, cao)
        eca_dyn = self._eca_dyn(cai, cao)
        I_leak = dtype(self._gbca_leak) * (V_mV - eca_dyn)
        I_cap = dtype(self._ICaPmax_uA) * cai / (cai + dtype(self._KmCa))
        nai3 = dtype(self._nai ** 3)
        nao3 = dtype(self._nao ** 3)
        fac = dtype(self._fac_V)
        DFin = nai3 * cao * jnp.exp(dtype((self._r - 2) * self._gamma) * V_mV * fac)
        DFout = nao3 * cai * jnp.exp(dtype((self._r - 2) * (self._gamma - 1)) * V_mV * fac)
        S_nca = dtype(1.0) + dtype(self._DNaCa) * (cai * nao3 + cao * nai3)
        inca = dtype(self._KNaCa_uA) * (DFin - DFout) / jnp.maximum(S_nca, dtype(1e-12))
        I_nca = -dtype(2.0) * inca
        return I_chan + I_leak + I_cap + I_nca

    def dynamics_correction_ca(
        self, V_mV: jnp.ndarray, gates: jnp.ndarray, cai: jnp.ndarray, cao: jnp.ndarray
    ) -> jnp.ndarray:
        eca_static = dtype(self._eca_static)
        eca_dyn = self._eca_dyn(cai, cao)
        delta_eca = eca_static - eca_dyn
        i0, i1 = self.cum_sizes[self._idx_can], self.cum_sizes[self._idx_can + 1]
        g_can = self.models[self._idx_can].g_funcs(gates[:, i0:i1], self.models[self._idx_can].g_bar)[:, 0]
        i0, i1 = self.cum_sizes[self._idx_cat], self.cum_sizes[self._idx_cat + 1]
        g_cat = self.models[self._idx_cat].g_funcs(gates[:, i0:i1], self.models[self._idx_cat].g_bar)[:, 0]
        I_eca_corr = (g_can + g_cat + dtype(self._gbca_leak)) * delta_eca

        fnk_dyn = (V_mV + dtype(150.0)) / (V_mV + dtype(200.0))
        I_nak_corr = dtype(self._INaKmax_na_ko) * fnk_dyn - dtype(self._I_NaKpump_bg)

        I_cap_dyn = dtype(self._ICaPmax_uA) * cai / (cai + dtype(self._KmCa))
        I_cap_corr = I_cap_dyn - dtype(self._I_CaPump_bg)

        fac = dtype(self._fac_V)
        nai3 = dtype(self._nai ** 3)
        nao3 = dtype(self._nao ** 3)
        DFin = nai3 * cao * jnp.exp(dtype((self._r - 2) * self._gamma) * V_mV * fac)
        DFout = nao3 * cai * jnp.exp(dtype((self._r - 2) * (self._gamma - 1)) * V_mV * fac)
        S_nca = dtype(1.0) + dtype(self._DNaCa) * (cai * nao3 + cao * nai3)
        inca = dtype(self._KNaCa_uA) * (DFin - DFout) / jnp.maximum(S_nca, dtype(1e-12))
        I_nca_corr = inca - dtype(self._inca0_uA)
        return I_eca_corr + I_nak_corr + I_cap_corr + I_nca_corr

    def update_cai_Oc(
        self, cai: jnp.ndarray, Oc: jnp.ndarray, I_Ca: jnp.ndarray, dt: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ku = dtype(self._ku)
        kr = dtype(self._kr)
        nb_Bi = dtype(self._nb * self._Bi)
        factor = dtype(self._cai_factor)
        dt_ = dtype(dt)

        cai_it = cai
        Oc_it = Oc
        for _ in range(2):
            diffOc = ku * cai_it * (dtype(1.0) - Oc_it) - kr * Oc_it
            Oc_it = jnp.clip(Oc + dt_ * diffOc, dtype(0.0), dtype(1.0))
            cai_it = jnp.maximum(cai + dt_ * (-I_Ca * factor - nb_Bi * diffOc), dtype(1e-9))
        return cai_it, Oc_it

    def update_cao(self, cao: jnp.ndarray, I_Ca: jnp.ndarray, dt: float) -> jnp.ndarray:
        factor = dtype(self._cao_factor)
        txfer = dtype(self._txfer)
        bath = dtype(self._cabath)
        dt_ = dtype(dt)
        dcao_dt = I_Ca * factor + (bath - cao) / txfer
        return jnp.maximum(cao + dt_ * dcao_dt, dtype(1e-9))

    def I_background_static(self, Nx: int) -> jnp.ndarray:
        return jnp.full((Nx,), self._I_pump_bg, dtype=self.dtype)

    def prepare_membrane_step(
        self,
        V_mV: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state: tuple[jnp.ndarray, ...],
        dt: float,
        I_ion: jnp.ndarray,
        I_background: jnp.ndarray,
    ) -> MembraneStepPlan:
        _ = I_background
        cai, Oc, cao, c_kca = state
        half_dt = dtype(0.5) * dtype(dt)

        I_Ca_budget_0 = self.compute_I_Ca_budget(V_mV, gates_prev, cai, cao)
        cai_mid, Oc_mid = self.update_cai_Oc(cai, Oc, I_Ca_budget_0, half_dt)
        cao_mid = self.update_cao(cao, I_Ca_budget_0, half_dt)

        I_kca = self.compute_I_kca(V_mV, c_kca)
        I_ca_corr = self.dynamics_correction_ca(V_mV, gates_new, cai_mid, cao_mid)
        I_bg = self.I_background_static(V_mV.shape[0])
        return MembraneStepPlan(
            state=(cai_mid, Oc_mid, cao_mid, c_kca),
            linearization_gates=gates_prev,
            total_outward_current=I_bg + I_ion + I_kca,
            explicit_outward_current=I_bg + I_kca,
            correction_current=I_ca_corr,
        )

    def finalize_membrane_step(
        self,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state_prev: tuple[jnp.ndarray, ...],
        step_plan: MembraneStepPlan,
        dt: float,
    ) -> tuple[jnp.ndarray, ...]:
        _ = V_mV_prev, gates_prev, state_prev
        cai_mid, Oc_mid, cao_mid, c_kca = step_plan.state
        q10_kca = dtype(self._q10_kca)

        alpha_c = self.alpha_c_kca(V_mV_new, cai_mid) * q10_kca
        beta_c = self.beta_c_kca(V_mV_new) * q10_kca
        eff_rate = (alpha_c + beta_c) / dtype(4.5)
        cinf = alpha_c / jnp.maximum(alpha_c + beta_c, dtype(1e-12))
        c_kca_new = cinf - (cinf - c_kca) * jnp.exp(-dtype(dt) * eff_rate)

        half_dt = dtype(0.5) * dtype(dt)
        I_Ca_budget_1 = self.compute_I_Ca_budget(V_mV_new, gates_new, cai_mid, cao_mid)
        cai_new, Oc_new = self.update_cai_Oc(cai_mid, Oc_mid, I_Ca_budget_1, half_dt)
        cao_new = self.update_cao(cao_mid, I_Ca_budget_1, half_dt)
        return (cai_new, Oc_new, cao_new, c_kca_new)

    def diagnostic_names(self) -> tuple[str, ...]:
        return (
            "I_na_total_uAcm2",
            "I_k_total_uAcm2",
            "I_ca_total_uAcm2",
            "I_total_rhs_uAcm2",
        )

    def _model_g(self, idx_model: int, gates_local: jnp.ndarray) -> jnp.ndarray:
        i0, i1 = self.cum_sizes[idx_model], self.cum_sizes[idx_model + 1]
        return self.models[idx_model].g_funcs(gates_local[:, i0:i1], self.models[idx_model].g_bar)[:, 0]

    def compute_step_diagnostics(
        self,
        V_mV_prev: jnp.ndarray,
        V_mV_new: jnp.ndarray,
        gates_prev: jnp.ndarray,
        gates_new: jnp.ndarray,
        state_prev: tuple[jnp.ndarray, ...],
        state_new: tuple[jnp.ndarray, ...],
        step_plan: MembraneStepPlan,
        I_ion: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        _ = V_mV_new, gates_prev, state_prev
        cai_mid, _, cao_mid, c_kca_old = step_plan.state
        _, _, _, c_kca_new = state_new

        eca_dyn = self._eca_dyn(cai_mid, cao_mid)
        g_naf = self._model_g(1, gates_new)
        g_nas = self._model_g(2, gates_new)
        g_kd = self._model_g(3, gates_new)
        g_ka = self._model_g(4, gates_new)
        g_kds = self._model_g(5, gates_new)
        g_can = self._model_g(self._idx_can, gates_new)
        g_cat = self._model_g(self._idx_cat, gates_new)

        i_naf_cur = g_naf * (V_mV_prev - dtype(self._ena_schild))
        i_nas_cur = g_nas * (V_mV_prev - dtype(self._ena_schild))
        i_kd_cur = g_kd * (V_mV_prev - dtype(self._ek_schild))
        i_ka_cur = g_ka * (V_mV_prev - dtype(self._ek_schild))
        i_kds_cur = g_kds * (V_mV_prev - dtype(self._ek_schild))
        i_can_cur = g_can * (V_mV_prev - eca_dyn)
        i_cat_cur = g_cat * (V_mV_prev - eca_dyn)
        i_leak_na = dtype(self.models[0]._gbna) * (V_mV_prev - dtype(self._ena_schild))
        i_leak_ca = dtype(self.models[0]._gbca) * (V_mV_prev - eca_dyn)

        fnk_dyn = (V_mV_prev + dtype(150.0)) / (V_mV_prev + dtype(200.0))
        ink = dtype(self._INaKmax_na_ko) * fnk_dyn
        i_nak_na = dtype(3.0) * ink
        i_nak_k = dtype(-2.0) * ink

        i_cap = dtype(self._ICaPmax_uA) * cai_mid / (cai_mid + dtype(self._KmCa))

        fac = dtype(self._fac_V)
        nai3 = dtype(self._nai ** 3)
        nao3 = dtype(self._nao ** 3)
        dfin = nai3 * cao_mid * jnp.exp(dtype((self._r - 2) * self._gamma) * V_mV_prev * fac)
        dfout = nao3 * cai_mid * jnp.exp(
            dtype((self._r - 2) * (self._gamma - 1)) * V_mV_prev * fac
        )
        s_nca = dtype(1.0) + dtype(self._DNaCa) * (cai_mid * nao3 + cao_mid * nai3)
        inca = dtype(self._KNaCa_uA) * (dfin - dfout) / jnp.maximum(s_nca, dtype(1e-12))
        i_ncx_na = dtype(3.0) * inca
        i_ncx_ca = dtype(-2.0) * inca

        I_kca = self.compute_I_kca(V_mV_prev, c_kca_old)
        i_na_total = i_naf_cur + i_nas_cur + i_leak_na + i_nak_na + i_ncx_na
        i_k_total = i_kd_cur + i_ka_cur + i_kds_cur + I_kca + i_nak_k
        i_ca_total = i_can_cur + i_cat_cur + i_leak_ca + i_cap + i_ncx_ca
        i_total_rhs = I_ion + I_kca + self.I_background_static(V_mV_prev.shape[0]) + step_plan.correction_current
        _ = c_kca_new
        return i_na_total, i_k_total, i_ca_total, i_total_rhs


class Schild94CompositeICM(SchildCompositeICM):
    def __init__(self, *, diameter_um: float, temp_c: float = 37.0, vinit_mV: float = -48.0) -> None:
        T_K = float(temp_c + 273.15)
        R, F = 8314.0, 96500.0
        nao, nai, ko, ki = 154.0, 8.9, 5.4, 145.0
        cai0, cao0 = 0.000117, 2.0
        ena = float(_schild_nernst(T_K, 1, nao, nai, R, F))
        ek = float(_schild_nernst(T_K, 1, ko, ki, R, F))
        eca = float(_schild_nernst(T_K, 2, cao0, cai0, R, F)) - 78.7
        super().__init__(
            models=[
                LeakSchildICM(gbna=1.85681e-5, gbca=3.00626e-6, ena=ena, eca=eca),
                NafSchildICM(gbar=0.068967142, ena=ena, celsius=temp_c),
                NasSchildICM(gbar=0.001043349, ena=ena, celsius=temp_c),
                KdSchildICM(gbar=0.000180376, ek=ek, celsius=temp_c),
                KaSchildICM(gbar=0.000141471, ek=ek, celsius=temp_c),
                KdsSchildICM(gbar=0.000106103, ek=ek, celsius=temp_c),
                CaNICM(gbar=0.000106103, eca=eca, celsius=temp_c),
                CaTICM(gbar=1.23787e-5, eca=eca, celsius=temp_c),
            ],
            diameter_um=diameter_um,
            temp_c=temp_c,
            vinit_mV=vinit_mV,
            nao=nao,
            nai=nai,
            ko=ko,
            ki=ki,
            cai0=cai0,
            cao0=cao0,
            gbar_kca=0.000141471,
            idx_can=6,
            idx_cat=7,
        )


class Schild97CompositeICM(SchildCompositeICM):
    def __init__(self, *, diameter_um: float, temp_c: float = 37.0, vinit_mV: float = -48.0) -> None:
        T_K = float(temp_c + 273.15)
        R, F = 8314.0, 96500.0
        nao, nai, ko, ki = 154.0, 8.9, 5.4, 145.0
        cai0, cao0 = 0.000117, 2.0
        ena = float(_schild_nernst(T_K, 1, nao, nai, R, F))
        ek = float(_schild_nernst(T_K, 1, ko, ki, R, F))
        eca = float(_schild_nernst(T_K, 2, cao0, cai0, R, F)) - 78.7
        super().__init__(
            models=[
                LeakSchildICM(gbna=1.8261e-5, gbca=9.13049e-6, ena=ena, eca=eca),
                Naf97ICM(gbar=0.022434928, ena=ena, celsius=temp_c),
                Nas97ICM(gbar=0.022434928, ena=ena, celsius=temp_c),
                KdSchildICM(gbar=0.001956534, ek=ek, celsius=temp_c),
                KaSchildICM(gbar=0.001304356, ek=ek, celsius=temp_c),
                KdsSchildICM(gbar=0.000782614, ek=ek, celsius=temp_c),
                CaNICM(gbar=0.000521743, eca=eca, celsius=temp_c),
                CaTICM(gbar=0.00018261, eca=eca, celsius=temp_c),
            ],
            diameter_um=diameter_um,
            temp_c=temp_c,
            vinit_mV=vinit_mV,
            nao=nao,
            nai=nai,
            ko=ko,
            ki=ki,
            cai0=cai0,
            cao0=cao0,
            gbar_kca=0.000913049,
            idx_can=6,
            idx_cat=7,
        )
