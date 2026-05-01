from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrv

from axonscope.axons.myelinated import MRG
from axonscope.axons.unmyelinated import (
    HodgkinHuxley,
    RattayAberham,
    Schild94,
    Schild97,
    Sundt,
    Tigerholm,
)
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers.CrankNicholson import CrankNicholson
from axonscope.stimulus import Stimulus
from tests.nrv._helpers import (
    AXONSCOPE_TO_NRV_CONDUCTANCE_SCALE,
    AXONSCOPE_TO_NRV_CURRENT_SCALE,
    align_rows_to_target_x,
    enable_nrv_recordings,
    interp_rows,
    normalize_nrv_matrix,
    nrv_trace,
    sample_indices_from_position,
    trace_metrics,
)

pytestmark = pytest.mark.nrv_extracellular

FIG_DIR = Path("figures/nrv_tests/extracellular")
SIGMA_S_M = 0.2
ELECTRODE_Y_UM = 100.0
ELECTRODE_Z_UM = 0.0


@dataclass(frozen=True)
class ExtracellularSpec:
    name: str
    diameters_um: tuple[float, ...]
    axonscope_factory: Callable[[float], object]
    nrv_factory: Callable[[float, object, float], object]
    tsim_ms: float
    dt_ms: float
    current_pairs: tuple[tuple[str, str], ...]
    gate_pairs: tuple[tuple[str, str], ...]
    state_pairs: tuple[tuple[str, str], ...]
    vm_rmse_atol_mV: float
    vm_peak_atol_mV: float
    vm_matrix_rmse_atol_mV: float | None
    vm_matrix_corr_min: float | None
    current_rmse_atol: float
    current_max_atol: float
    gate_rmse_atol: float
    gate_max_atol: float
    state_rmse_atol: float
    state_max_atol: float
    nrv_key_overrides: dict[str, str] | None = None
    vext_rmse_atol_mV: float = 1.0
    vext_max_atol_mV: float = 2.5
    conductance_pairs: tuple[tuple[str, str], ...] = ()
    conductance_rmse_atol: float = 0.0
    conductance_max_atol: float = 0.0
    gate_rmse_atol_by_name: dict[str, float] | None = None
    gate_max_atol_by_name: dict[str, float] | None = None
    state_rmse_atol_by_name: dict[str, float] | None = None
    state_max_atol_by_name: dict[str, float] | None = None
    current_time_shift_steps: int = 0
    conductance_time_shift_steps: int = 0
    gate_time_shift_steps: int = 0
    state_time_shift_steps: int = 0


HH_EXTRA_AMP_UA = 160.0
RATTAY_EXTRA_AMP_UA = 80.0
SUNDT_EXTRA_AMP_UA = 120.0
TIGERHOLM_EXTRA_AMP_UA = 100.0
SCHILD94_EXTRA_AMP_UA = 160.0
SCHILD97_EXTRA_AMP_UA = 160.0


def _attach_point_source_extra_as(
    axon,
    *,
    amp_uA: float,
    start_ms: float,
    duration_ms: float,
) -> None:
    x0_um = float(axon.L / 2.0)
    electrode = PointSourceElectrode(
        x0_m=x0_um * 1e-6,
        y0_m=ELECTRODE_Y_UM * 1e-6,
        z0_m=ELECTRODE_Z_UM * 1e-6,
        sigma_S_m=SIGMA_S_M,
    )
    stim = Stimulus.pulse(start=start_ms, amplitude=-amp_uA * 1e-6, duration=duration_ms)
    axon.add_extracellular_ctx(electrode, stim, replace=True)


def _attach_point_source_extra_nrv(
    axon_nrv,
    *,
    center_x_um: float,
    amp_uA: float,
    start_ms: float,
    duration_ms: float,
) -> None:
    elec = nrv.point_source_electrode(center_x_um, ELECTRODE_Y_UM, ELECTRODE_Z_UM)
    stim = nrv.stimulus()
    stim.pulse(start=start_ms, value=-amp_uA, duration=duration_ms)
    extra = nrv.stimulation("endoneurium_bhadra")
    extra.add_electrode(elec, stim)
    axon_nrv.attach_extracellular_stimulation(extra)


def _make_hh_extra_axon(d: float):
    ax = HodgkinHuxley(
        L=1000.0,
        d=d,
        Nx=101,
        celsius=6.3,
        Vinit=-70.0,
        include_passive_leak=True,
        g_pas=0.001,
        e_pas=-70.0,
    )
    _attach_point_source_extra_as(ax, amp_uA=HH_EXTRA_AMP_UA, start_ms=1.0, duration_ms=0.1)
    ax.comparison_sample_position_um = 500.0
    return ax


def _make_hh_extra_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=1000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-70.0, T=6.3, model="HH")
    _attach_point_source_extra_nrv(ax, center_x_um=500.0, amp_uA=HH_EXTRA_AMP_UA, start_ms=1.0, duration_ms=0.1)
    return ax


def _make_rattay_extra_axon(d: float):
    ax = RattayAberham(L=1000.0, d=d, Nx=101, celsius=37.0)
    _attach_point_source_extra_as(ax, amp_uA=RATTAY_EXTRA_AMP_UA, start_ms=1.0, duration_ms=0.1)
    ax.comparison_sample_position_um = 500.0
    return ax


def _make_rattay_extra_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=1000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-70.0, T=37.0, model="Rattay_Aberham")
    _attach_point_source_extra_nrv(ax, center_x_um=500.0, amp_uA=RATTAY_EXTRA_AMP_UA, start_ms=1.0, duration_ms=0.1)
    return ax


def _make_sundt_extra_axon(d: float):
    ax = Sundt(L=2000.0, d=d, Nx=101, celsius=37.0)
    _attach_point_source_extra_as(ax, amp_uA=SUNDT_EXTRA_AMP_UA, start_ms=1.0, duration_ms=0.1)
    ax.comparison_sample_position_um = 1000.0
    return ax


def _make_sundt_extra_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=2000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-60.0, T=37.0, model="Sundt")
    _attach_point_source_extra_nrv(ax, center_x_um=1000.0, amp_uA=SUNDT_EXTRA_AMP_UA, start_ms=1.0, duration_ms=0.1)
    return ax


def _make_tigerholm_extra_axon(d: float):
    ax = Tigerholm(L=5000.0, d=d, Nx=101, celsius=37.0)
    _attach_point_source_extra_as(ax, amp_uA=TIGERHOLM_EXTRA_AMP_UA, start_ms=5.0, duration_ms=0.1)
    ax.comparison_sample_position_um = 2500.0
    return ax


def _make_tigerholm_extra_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=5000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-62.0, T=37.0, model="Tigerholm")
    _attach_point_source_extra_nrv(ax, center_x_um=2500.0, amp_uA=TIGERHOLM_EXTRA_AMP_UA, start_ms=5.0, duration_ms=0.1)
    return ax


def _make_schild94_extra_axon(d: float):
    ax = Schild94(L=3000.0, d=d, Nx=51)
    _attach_point_source_extra_as(ax, amp_uA=SCHILD94_EXTRA_AMP_UA, start_ms=2.0, duration_ms=0.1)
    ax.comparison_sample_position_um = 1500.0
    return ax


def _make_schild94_extra_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=3000.0, Nsec=1, Nseg_per_sec=51, dt=dt_ms, v_init=-48.0, T=37.0, model="Schild_94")
    _attach_point_source_extra_nrv(ax, center_x_um=1500.0, amp_uA=SCHILD94_EXTRA_AMP_UA, start_ms=2.0, duration_ms=0.1)
    return ax


def _make_schild97_extra_axon(d: float):
    ax = Schild97(L=3000.0, d=d, Nx=51)
    _attach_point_source_extra_as(ax, amp_uA=SCHILD97_EXTRA_AMP_UA, start_ms=2.0, duration_ms=0.1)
    ax.comparison_sample_position_um = 1500.0
    return ax


def _make_schild97_extra_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=3000.0, Nsec=1, Nseg_per_sec=51, dt=dt_ms, v_init=-48.0, T=37.0, model="Schild_97")
    _attach_point_source_extra_nrv(ax, center_x_um=1500.0, amp_uA=SCHILD97_EXTRA_AMP_UA, start_ms=2.0, duration_ms=0.1)
    return ax


def _mrg_center_node_pos_um(ax: MRG) -> tuple[int, float]:
    node_ids = np.asarray(ax.node_indices, dtype=int)
    stim_node = int(node_ids.shape[0] // 2)
    stim_pos_um = float(np.asarray(ax.x, dtype=float)[int(node_ids[stim_node])])
    return stim_node, stim_pos_um


def _make_mrg_extra_axon(d: float):
    ax = MRG(d=d, nodes=9)
    _, center_node_pos = _mrg_center_node_pos_um(ax)
    x0_um = float(ax.L / 2.0)
    electrode = PointSourceElectrode(
        x0_m=x0_um * 1e-6,
        y0_m=ELECTRODE_Y_UM * 1e-6,
        z0_m=ELECTRODE_Z_UM * 1e-6,
        sigma_S_m=SIGMA_S_M,
    )
    stim = Stimulus.biphasic(
        start=1.0,
        cathodic_amplitude=80.0 * 1e-6,
        cathodic_duration=0.08,
        anodic_amplitude=20.0 * 1e-6,
        interphase=0.04,
    )
    ax.add_extracellular_ctx(electrode, stim, replace=True)
    ax.comparison_sample_position_um = center_node_pos
    return ax


def _make_mrg_extra_nrv(d: float, axon_as, dt_ms: float):
    ax = nrv.myelinated(
        0,
        0,
        d,
        float(axon_as.L),
        model="MRG",
        dt=dt_ms,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    x0_um = float(axon_as.L / 2.0)
    elec = nrv.point_source_electrode(x0_um, ELECTRODE_Y_UM, ELECTRODE_Z_UM)
    stim = nrv.stimulus()
    stim.biphasic_pulse(1.0, 80.0, 0.08, 20.0, 0.04)
    extra = nrv.stimulation("endoneurium_bhadra")
    extra.add_electrode(elec, stim)
    ax.attach_extracellular_stimulation(extra)
    return ax


SPECS = [
    ExtracellularSpec(
        name="hh",
        diameters_um=(0.5, 0.75, 1.0),
        axonscope_factory=_make_hh_extra_axon,
        nrv_factory=_make_hh_extra_nrv,
        tsim_ms=8.0,
        dt_ms=0.001,
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("m", "m"), ("n", "n"), ("h", "h")),
        state_pairs=(),
        vm_rmse_atol_mV=5.0,
        vm_peak_atol_mV=10.0,
        vm_matrix_rmse_atol_mV=None,
        vm_matrix_corr_min=None,
        current_rmse_atol=4.0,
        current_max_atol=15.0,
        gate_rmse_atol=0.06,
        gate_max_atol=0.18,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
        vext_rmse_atol_mV=1.35,
        vext_max_atol_mV=3.3,
    ),
    ExtracellularSpec(
        name="rattay",
        diameters_um=(0.4, 0.6, 0.8),
        axonscope_factory=_make_rattay_extra_axon,
        nrv_factory=_make_rattay_extra_nrv,
        tsim_ms=10.0,
        dt_ms=0.01,
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("m", "m"), ("n", "n"), ("h", "h")),
        state_pairs=(),
        vm_rmse_atol_mV=6.0,
        vm_peak_atol_mV=13.0,
        vm_matrix_rmse_atol_mV=None,
        vm_matrix_corr_min=None,
        current_rmse_atol=5.0,
        current_max_atol=18.0,
        gate_rmse_atol=0.07,
        gate_max_atol=0.20,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
        gate_max_atol_by_name={"m": 0.32},
    ),
    ExtracellularSpec(
        name="sundt",
        diameters_um=(0.5, 0.65, 0.8),
        axonscope_factory=_make_sundt_extra_axon,
        nrv_factory=_make_sundt_extra_nrv,
        tsim_ms=8.0,
        dt_ms=0.001,
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("m", "m"), ("n", "n"), ("h", "h")),
        state_pairs=(),
        vm_rmse_atol_mV=6.0,
        vm_peak_atol_mV=12.0,
        vm_matrix_rmse_atol_mV=None,
        vm_matrix_corr_min=None,
        current_rmse_atol=5.0,
        current_max_atol=18.0,
        gate_rmse_atol=0.08,
        gate_max_atol=0.22,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
        gate_rmse_atol_by_name={"m": 0.10},
        gate_max_atol_by_name={"m": 0.50},
    ),
    ExtracellularSpec(
        name="tigerholm",
        diameters_um=(0.5, 0.75, 1.0),
        axonscope_factory=_make_tigerholm_extra_axon,
        nrv_factory=_make_tigerholm_extra_nrv,
        tsim_ms=20.0,
        dt_ms=0.025,
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k")),
        gate_pairs=(
            ("m_nav18", "m_nav18"),
            ("h_nav18", "h_nav18"),
            ("s_nav18", "s_nav18"),
            ("u_nav18", "u_nav18"),
            ("m_nav19", "m_nav19"),
            ("h_nav19", "h_nav19"),
            ("s_nav19", "s_nav19"),
            ("m_nattxs", "m_nattxs"),
            ("h_nattxs", "h_nattxs"),
            ("s_nattxs", "s_nattxs"),
            ("n_kdr", "n_kdr"),
            ("m_kf", "m_kf"),
            ("h_kf", "h_kf"),
            ("ns_ks", "ns_ks"),
            ("nf_ks", "nf_ks"),
            ("w_kna", "w_kna"),
            ("ns_h", "ns_h"),
            ("nf_h", "nf_h"),
        ),
        state_pairs=(),
        vm_rmse_atol_mV=8.0,
        vm_peak_atol_mV=15.0,
        vm_matrix_rmse_atol_mV=None,
        vm_matrix_corr_min=None,
        current_rmse_atol=12.0,
        current_max_atol=40.0,
        gate_rmse_atol=0.10,
        gate_max_atol=0.25,
        state_rmse_atol=2.0,
        state_max_atol=5.0,
    ),
    ExtracellularSpec(
        name="schild94",
        diameters_um=(0.8, 0.9, 1.0),
        axonscope_factory=_make_schild94_extra_axon,
        nrv_factory=_make_schild94_extra_nrv,
        tsim_ms=12.0,
        dt_ms=0.005,
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_ca", "I_ca")),
        gate_pairs=(
            ("d_can", "d_can"),
            ("f1_can", "f1_can"),
            ("f2_can", "f2_can"),
            ("d_cat", "d_cat"),
            ("f_cat", "f_cat"),
            ("p_ka", "p_ka"),
            ("q_ka", "q_ka"),
            ("n_kd", "n_kd"),
            ("x_kds", "x_kds"),
            ("y1_kds", "y1_kds"),
            ("m_naf", "m_naf"),
            ("h_naf", "h_naf"),
            ("l_naf", "l_naf"),
            ("m_nas", "m_nas"),
            ("h_nas", "h_nas"),
        ),
        state_pairs=(("c_kca", "c_ka"),),
        vm_rmse_atol_mV=8.0,
        vm_peak_atol_mV=16.0,
        vm_matrix_rmse_atol_mV=None,
        vm_matrix_corr_min=None,
        current_rmse_atol=15.0,
        current_max_atol=50.0,
        gate_rmse_atol=0.12,
        gate_max_atol=0.30,
        state_rmse_atol=0.12,
        state_max_atol=0.35,
        nrv_key_overrides={"l_naf": "h_nas", "m_nas": "l_naf", "h_nas": "m_nas"},
        vext_rmse_atol_mV=1.8,
        vext_max_atol_mV=5.0,
    ),
    ExtracellularSpec(
        name="schild97",
        diameters_um=(0.8, 0.9, 1.0),
        axonscope_factory=_make_schild97_extra_axon,
        nrv_factory=_make_schild97_extra_nrv,
        tsim_ms=12.0,
        dt_ms=0.005,
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_ca", "I_ca")),
        gate_pairs=(
            ("d_can", "d_can"),
            ("f1_can", "f1_can"),
            ("f2_can", "f2_can"),
            ("d_cat", "d_cat"),
            ("f_cat", "f_cat"),
            ("p_ka", "p_ka"),
            ("q_ka", "q_ka"),
            ("n_kd", "n_kd"),
            ("x_kds", "x_kds"),
            ("y1_kds", "y1_kds"),
            ("m_naf", "m_naf"),
            ("h_naf", "h_naf"),
            ("m_nas", "m_nas"),
            ("h_nas", "h_nas"),
        ),
        state_pairs=(("c_kca", "c_ka"),),
        vm_rmse_atol_mV=8.0,
        vm_peak_atol_mV=16.0,
        vm_matrix_rmse_atol_mV=None,
        vm_matrix_corr_min=None,
        current_rmse_atol=15.0,
        current_max_atol=50.0,
        gate_rmse_atol=0.12,
        gate_max_atol=0.30,
        state_rmse_atol=0.12,
        state_max_atol=0.35,
        vext_rmse_atol_mV=1.8,
        vext_max_atol_mV=5.0,
    ),
    ExtracellularSpec(
        name="mrg",
        diameters_um=(8.7, 10.0),
        axonscope_factory=_make_mrg_extra_axon,
        nrv_factory=_make_mrg_extra_nrv,
        tsim_ms=4.0,
        dt_ms=0.005,
        current_pairs=(("I_na", "I_na"), ("I_nap", "I_nap"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("mp", "mp"), ("m", "m"), ("h", "h"), ("s", "s")),
        state_pairs=(),
        vm_rmse_atol_mV=12.0,
        vm_peak_atol_mV=30.0,
        vm_matrix_rmse_atol_mV=4.0,
        vm_matrix_corr_min=0.95,
        current_rmse_atol=5.0,
        current_max_atol=240.0,
        gate_rmse_atol=0.10,
        gate_max_atol=0.28,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
        conductance_pairs=(("g_na", "g_na"), ("g_nap", "g_nap"), ("g_k", "g_k"), ("g_l", "g_l")),
        conductance_rmse_atol=0.20,
        conductance_max_atol=2.0,
        gate_max_atol_by_name={"m": 0.40},
        current_time_shift_steps=-1,
        gate_time_shift_steps=1,
    ),
]


def _recorded_trace(res, group: str, name: str, compartment_index: int) -> np.ndarray:
    assert res.recordings is not None
    return np.asarray(res.recordings[group][name], dtype=float)[:, compartment_index]


def _resolve_nrv_key(spec: ExtracellularSpec, key: str) -> str:
    if spec.nrv_key_overrides is None:
        return key
    return spec.nrv_key_overrides.get(key, key)


def _compare_vext_profiles(
    axon_as,
    axon_nrv,
    x_nrv_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    extra_nrv = getattr(axon_nrv, "extra_stim", None)
    if extra_nrv is None:
        return None

    if not extra_nrv.synchronised:
        extra_nrv.synchronise_stimuli()

    t_edges_ms = np.asarray(extra_nrv.global_time_serie, dtype=float).ravel()
    if t_edges_ms.size < 2:
        return None

    x_as_um = np.asarray(axon_as.x, dtype=float)
    x_nrv_um = np.asarray(x_nrv_um, dtype=float).ravel()
    if x_nrv_um.size == 0:
        return None

    t_probe_ms = 0.5 * (t_edges_ms[:-1] + t_edges_ms[1:])
    vext_as_mV = np.stack(
        [np.asarray(axon_as.Vext_mV(float(ti)), dtype=float) for ti in t_probe_ms],
        axis=1,
    )
    vext_nrv_mV = np.stack(
        [np.asarray(extra_nrv.compute_vext(i), dtype=float) for i in range(t_probe_ms.size)],
        axis=1,
    )

    def _best_vext_x_candidate(
        x_nrv_full_um: np.ndarray,
        n_rows: int,
        x_target_um: np.ndarray,
    ) -> np.ndarray | None:
        candidates: list[np.ndarray] = []
        if x_nrv_full_um.size == n_rows:
            candidates.append(x_nrv_full_um)
        if x_nrv_full_um.size == n_rows + 2:
            candidates.append(x_nrv_full_um[1:-1])
        if x_nrv_full_um[1::2].size == n_rows:
            candidates.append(x_nrv_full_um[1::2])
        if x_nrv_full_um[::2].size == n_rows:
            candidates.append(x_nrv_full_um[::2])

        if not candidates:
            return None

        best_x: np.ndarray | None = None
        best_score = float("inf")
        for x_candidate in candidates:
            idx = np.asarray(
                [int(np.argmin(np.abs(x_candidate - xi))) for xi in x_target_um],
                dtype=int,
            )
            score = float(np.mean(np.abs(x_candidate[idx] - x_target_um)))
            if score < best_score:
                best_score = score
                best_x = x_candidate
        return best_x

    x_nrv_vext_um = _best_vext_x_candidate(x_nrv_um, vext_nrv_mV.shape[0], x_as_um)
    if x_nrv_vext_um is None:
        return None
    _, vext_nrv_aligned_mV, _ = align_rows_to_target_x(x_nrv_vext_um, vext_nrv_mV, x_as_um)

    diff = vext_as_mV - vext_nrv_aligned_mV
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    return t_probe_ms, vext_as_mV, vext_nrv_aligned_mV, rmse, max_abs


def _plot_extracellular_report(
    spec: ExtracellularSpec,
    diameter_um: float,
    t_as: np.ndarray,
    x_as_um: np.ndarray,
    vm_as_matrix: np.ndarray,
    vm_nrv_interp: np.ndarray,
    vm_local_as: np.ndarray,
    vm_local_nrv: np.ndarray,
    t_vext_ms: np.ndarray | None,
    vext_local_as: np.ndarray | None,
    vext_local_nrv: np.ndarray | None,
    current_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    conductance_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    gate_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    state_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    metrics_lines: list[str],
) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)

    axs[0, 0].plot(t_as, vm_local_as, lw=2.0, zorder=2, label="AxonScope")
    axs[0, 0].plot(t_as, vm_local_nrv, "--", lw=2.2, zorder=4, label="NRV")
    axs[0, 0].set_title(f"{spec.name} d={diameter_um:.3f} um local Vm")
    axs[0, 0].set_xlabel("Time [ms]")
    axs[0, 0].set_ylabel("Vm [mV]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    ax_vext = axs[0, 1]
    if t_vext_ms is not None and vext_local_as is not None and vext_local_nrv is not None:
        ax_vext.plot(t_vext_ms, vext_local_as, lw=2.0, zorder=2, label="AxonScope")
        ax_vext.plot(t_vext_ms, vext_local_nrv, "--", lw=2.2, zorder=4, label="NRV")
        ax_vext.legend()
    else:
        ax_vext.text(0.5, 0.5, "No Vext comparison", ha="center", va="center")
    ax_vext.set_title("Local extracellular potential")
    ax_vext.set_xlabel("Time [ms]")
    ax_vext.set_ylabel("Vext [mV]")
    ax_vext.grid(True, alpha=0.3)

    im_as = axs[1, 0].imshow(
        vm_as_matrix,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), float(x_as_um[0]), float(x_as_um[-1])],
        cmap="viridis",
    )
    axs[1, 0].set_title("AxonScope heatmap")
    axs[1, 0].set_xlabel("Time [ms]")
    axs[1, 0].set_ylabel("Position [um]")
    fig.colorbar(im_as, ax=axs[1, 0], label="Vm [mV]")

    im_nrv = axs[1, 1].imshow(
        vm_nrv_interp,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), float(x_as_um[0]), float(x_as_um[-1])],
        cmap="viridis",
    )
    axs[1, 1].set_title("NRV heatmap (aligned on AxonScope x)")
    axs[1, 1].set_xlabel("Time [ms]")
    axs[1, 1].set_ylabel("Position [um]")
    fig.colorbar(im_nrv, ax=axs[1, 1], label="Vm [mV]")

    ax_curr = axs[2, 0]
    if current_pairs:
        for i, (label, as_trace, nrv_trace_ref) in enumerate(current_pairs):
            color = plt.cm.tab20(i % 20)
            ax_curr.plot(t_as, as_trace, color=color, lw=1.8, zorder=2, label=f"{label} AS")
            ax_curr.plot(t_as, nrv_trace_ref, color=color, lw=2.0, ls="--", zorder=4, label=f"{label} NRV")
    else:
        ax_curr.text(0.5, 0.5, "No current comparison", ha="center", va="center")
    ax_curr.set_title("Local ionic currents")
    ax_curr.set_xlabel("Time [ms]")
    ax_curr.set_ylabel("Current density [mA/cm²]")
    ax_curr.grid(True, alpha=0.3)
    if current_pairs:
        ax_curr.legend(fontsize=8, ncol=2)

    ax_cond = axs[2, 1]
    if conductance_pairs:
        for i, (label, as_trace, nrv_trace_ref) in enumerate(conductance_pairs):
            color = plt.cm.tab20(i % 20)
            ax_cond.plot(t_as, as_trace, color=color, lw=1.8, zorder=2, label=f"{label} AS")
            ax_cond.plot(t_as, nrv_trace_ref, color=color, lw=2.0, ls="--", zorder=4, label=f"{label} NRV")
    else:
        ax_cond.text(0.5, 0.5, "No conductance comparison", ha="center", va="center")
    ax_cond.set_title("Local conductances")
    ax_cond.set_xlabel("Time [ms]")
    ax_cond.set_ylabel("Conductance [S/cm²]")
    ax_cond.grid(True, alpha=0.3)
    if conductance_pairs:
        ax_cond.legend(fontsize=8, ncol=2)

    ax_states = axs[3, 0]
    merged_pairs = gate_pairs + state_pairs
    if merged_pairs:
        for i, (label, as_trace, nrv_trace_ref) in enumerate(merged_pairs):
            color = plt.cm.tab20(i % 20)
            ax_states.plot(t_as, as_trace, color=color, lw=1.8, zorder=2, label=f"{label} AS")
            ax_states.plot(t_as, nrv_trace_ref, color=color, lw=2.0, ls="--", zorder=4, label=f"{label} NRV")
    else:
        ax_states.text(0.5, 0.5, "No gate/state comparison", ha="center", va="center")
    ax_states.set_title("Local gates / states")
    ax_states.set_xlabel("Time [ms]")
    ax_states.set_ylabel("State value")
    ax_states.grid(True, alpha=0.3)
    if merged_pairs:
        ax_states.legend(fontsize=7, ncol=2)

    axs[3, 1].axis("off")
    axs[3, 1].text(
        0.01,
        0.99,
        "\n".join(metrics_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
    )

    fig_path = FIG_DIR / f"{spec.name}_d{diameter_um:.3f}_extracellular_vs_nrv.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _matrix_metrics(as_matrix: np.ndarray, nrv_matrix: np.ndarray) -> tuple[float, float]:
    diff = as_matrix - nrv_matrix
    rmse = float(np.sqrt(np.mean(diff**2)))
    corr = float(np.corrcoef(as_matrix.ravel(), nrv_matrix.ravel())[0, 1])
    return rmse, corr


def _run_extracellular_case(spec: ExtracellularSpec, diameter_um: float) -> None:
    axon = spec.axonscope_factory(float(diameter_um))
    res = CrankNicholson().solve(axon, tsim=spec.tsim_ms, dt=spec.dt_ms, record_observables=True)
    assert res.recordings is not None

    axon_nrv = spec.nrv_factory(float(diameter_um), axon, spec.dt_ms)
    enable_nrv_recordings(axon_nrv)
    results_nrv = axon_nrv.simulate(t_sim=spec.tsim_ms)

    t_as = np.asarray(res.t, dtype=float)
    as_x_um = np.asarray(axon.x, dtype=float)
    vm_as_matrix = np.asarray(res.Vm, dtype=float).T
    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_nrv = np.asarray(results_nrv["x_rec"], dtype=float)
    vm_nrv_matrix = normalize_nrv_matrix(results_nrv["V_mem"], t_nrv, x_nrv)
    _, vm_nrv_matrix_aligned, _ = align_rows_to_target_x(x_nrv, vm_nrv_matrix, as_x_um)
    vm_nrv_interp = interp_rows(vm_nrv_matrix_aligned, t_nrv, t_as)

    sample_position_um = getattr(axon, "comparison_sample_position_um", None)
    sample_as_idx, sample_nrv_idx = sample_indices_from_position(as_x_um, x_nrv, sample_position_um)

    vext_cmp = _compare_vext_profiles(axon, axon_nrv, x_nrv)
    if vext_cmp is None:
        t_vext_ms = None
        vext_local_as = None
        vext_local_nrv = None
        vext_rmse = float("nan")
        vext_max_abs = float("nan")
        failures = [f"{spec.name} d={diameter_um:.3f} could not align Vext profiles between AxonScope and NRV"]
    else:
        t_vext_ms, vext_as_matrix, vext_nrv_matrix = vext_cmp[:3]
        vext_local_as = np.asarray(vext_as_matrix[sample_as_idx], dtype=float)
        vext_local_nrv = np.asarray(vext_nrv_matrix[sample_as_idx], dtype=float)
        vext_rmse = float(vext_cmp[3])
        vext_max_abs = float(vext_cmp[4])
        failures: list[str] = []

    vm_local_as = np.asarray(res.Vm, dtype=float)[:, sample_as_idx]
    vm_local_nrv = np.interp(t_as, t_nrv, vm_nrv_matrix_aligned[sample_as_idx])
    vm_rmse, _, _ = trace_metrics(vm_local_nrv, vm_local_as)
    vm_peak_diff = float(abs(float(vm_local_as.max()) - float(vm_local_nrv.max())))
    matrix_rmse, matrix_corr = _matrix_metrics(vm_as_matrix, vm_nrv_interp)

    metrics_lines = [
        f"diameter [um] : {diameter_um:.4f}",
        f"Vext RMSE     : {vext_rmse:8.4e} mV",
        f"Vext max |Δ|  : {vext_max_abs:8.4e} mV",
        f"Vm RMSE       : {vm_rmse:8.4f} mV",
        f"Vm peak diff  : {vm_peak_diff:8.4f} mV",
        f"Vm matrix RMSE: {matrix_rmse:8.4f} mV",
        f"Vm matrix corr: {matrix_corr:8.4f}",
        f"Shift steps   : I={spec.current_time_shift_steps:+d}, g={spec.conductance_time_shift_steps:+d}, gates={spec.gate_time_shift_steps:+d}, states={spec.state_time_shift_steps:+d}",
        "Currents: AxonScope traces scaled by 1e-3 to match NRV current units",
        "Conductances: AxonScope traces scaled by 1e-3 to match NRV conductance units",
    ]

    node_mask = np.asarray(getattr(axon, "node_mask", np.array([], dtype=bool)), dtype=bool).ravel()
    if node_mask.size == as_x_um.size and np.any(node_mask) and np.any(~node_mask):
        node_rmse, node_corr = _matrix_metrics(vm_as_matrix[node_mask], vm_nrv_interp[node_mask])
        internode_rmse, internode_corr = _matrix_metrics(vm_as_matrix[~node_mask], vm_nrv_interp[~node_mask])
        metrics_lines.extend(
            [
                f"Node RMSE     : {node_rmse:8.4f} mV",
                f"Node corr     : {node_corr:8.4f}",
                f"Internode RMSE: {internode_rmse:8.4f} mV",
                f"Internode corr: {internode_corr:8.4f}",
            ]
        )

    current_plot_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for as_name, nrv_name in spec.current_pairs:
        as_trace = AXONSCOPE_TO_NRV_CURRENT_SCALE * _recorded_trace(res, "currents", as_name, sample_as_idx)
        nrv_trace_ref = nrv_trace(
            results_nrv,
            _resolve_nrv_key(spec, nrv_name),
            sample_nrv_idx,
            t_as,
            shift_steps=spec.current_time_shift_steps,
        )
        rmse, max_abs, q99_abs = trace_metrics(nrv_trace_ref, as_trace)
        metrics_lines.append(f"{as_name:12s}: rmse={rmse:8.4f} q99={q99_abs:8.4f} max={max_abs:8.4f}")
        if rmse >= spec.current_rmse_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} RMSE {rmse:.4f} > {spec.current_rmse_atol:.4f}")
        if max_abs >= spec.current_max_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} max |Δ| {max_abs:.4f} > {spec.current_max_atol:.4f}")
        current_plot_pairs.append((as_name, as_trace, nrv_trace_ref))

    conductance_plot_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for as_name, nrv_name in spec.conductance_pairs:
        as_trace = AXONSCOPE_TO_NRV_CONDUCTANCE_SCALE * _recorded_trace(
            res, "conductances", as_name, sample_as_idx
        )
        nrv_trace_ref = nrv_trace(
            results_nrv,
            _resolve_nrv_key(spec, nrv_name),
            sample_nrv_idx,
            t_as,
            shift_steps=spec.conductance_time_shift_steps,
        )
        rmse, max_abs, q99_abs = trace_metrics(nrv_trace_ref, as_trace)
        metrics_lines.append(f"{as_name:12s}: rmse={rmse:8.4f} q99={q99_abs:8.4f} max={max_abs:8.4f}")
        if rmse >= spec.conductance_rmse_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} RMSE {rmse:.4f} > {spec.conductance_rmse_atol:.4f}")
        if max_abs >= spec.conductance_max_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} max |Δ| {max_abs:.4f} > {spec.conductance_max_atol:.4f}")
        conductance_plot_pairs.append((as_name, as_trace, nrv_trace_ref))

    gate_plot_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for as_name, nrv_name in spec.gate_pairs:
        as_trace = _recorded_trace(res, "gates", as_name, sample_as_idx)
        nrv_trace_ref = nrv_trace(
            results_nrv,
            _resolve_nrv_key(spec, nrv_name),
            sample_nrv_idx,
            t_as,
            shift_steps=spec.gate_time_shift_steps,
        )
        rmse, max_abs, q99_abs = trace_metrics(nrv_trace_ref, as_trace)
        gate_rmse_atol = spec.gate_rmse_atol
        if spec.gate_rmse_atol_by_name is not None:
            gate_rmse_atol = spec.gate_rmse_atol_by_name.get(as_name, gate_rmse_atol)
        gate_max_atol = spec.gate_max_atol
        if spec.gate_max_atol_by_name is not None:
            gate_max_atol = spec.gate_max_atol_by_name.get(as_name, gate_max_atol)
        metrics_lines.append(f"{as_name:12s}: rmse={rmse:8.4f} q99={q99_abs:8.4f} max={max_abs:8.4f}")
        if rmse >= gate_rmse_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} RMSE {rmse:.4f} > {gate_rmse_atol:.4f}")
        if q99_abs >= gate_max_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} q99 |Δ| {q99_abs:.4f} > {gate_max_atol:.4f}")
        gate_plot_pairs.append((as_name, as_trace, nrv_trace_ref))

    state_plot_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for as_name, nrv_name in spec.state_pairs:
        as_trace = _recorded_trace(res, "states", as_name, sample_as_idx)
        nrv_trace_ref = nrv_trace(
            results_nrv,
            _resolve_nrv_key(spec, nrv_name),
            sample_nrv_idx,
            t_as,
            shift_steps=spec.state_time_shift_steps,
        )
        rmse, max_abs, q99_abs = trace_metrics(nrv_trace_ref, as_trace)
        state_rmse_atol = spec.state_rmse_atol
        if spec.state_rmse_atol_by_name is not None:
            state_rmse_atol = spec.state_rmse_atol_by_name.get(as_name, state_rmse_atol)
        state_max_atol = spec.state_max_atol
        if spec.state_max_atol_by_name is not None:
            state_max_atol = spec.state_max_atol_by_name.get(as_name, state_max_atol)
        metrics_lines.append(f"{as_name:12s}: rmse={rmse:8.4f} q99={q99_abs:8.4f} max={max_abs:8.4f}")
        if rmse >= state_rmse_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} RMSE {rmse:.4f} > {state_rmse_atol:.4f}")
        if q99_abs >= state_max_atol:
            failures.append(f"{spec.name} d={diameter_um:.3f} {as_name} q99 |Δ| {q99_abs:.4f} > {state_max_atol:.4f}")
        state_plot_pairs.append((as_name, as_trace, nrv_trace_ref))

    fig_path = _plot_extracellular_report(
        spec,
        diameter_um,
        t_as,
        as_x_um,
        vm_as_matrix,
        vm_nrv_interp,
        vm_local_as,
        vm_local_nrv,
        t_vext_ms,
        vext_local_as,
        vext_local_nrv,
        current_plot_pairs,
        conductance_plot_pairs,
        gate_plot_pairs,
        state_plot_pairs,
        metrics_lines,
    )

    if vm_rmse >= spec.vm_rmse_atol_mV:
        failures.append(
            f"{spec.name} d={diameter_um:.3f} Vm RMSE {vm_rmse:.4f} mV > {spec.vm_rmse_atol_mV:.4f} mV (plot: {fig_path})"
        )
    if np.isfinite(vext_rmse) and vext_rmse >= spec.vext_rmse_atol_mV:
        failures.append(
            f"{spec.name} d={diameter_um:.3f} Vext RMSE {vext_rmse:.4e} mV > {spec.vext_rmse_atol_mV:.4e} mV (plot: {fig_path})"
        )
    if np.isfinite(vext_max_abs) and vext_max_abs >= spec.vext_max_atol_mV:
        failures.append(
            f"{spec.name} d={diameter_um:.3f} Vext max |Δ| {vext_max_abs:.4e} mV > {spec.vext_max_atol_mV:.4e} mV (plot: {fig_path})"
        )
    if vm_peak_diff >= spec.vm_peak_atol_mV:
        failures.append(
            f"{spec.name} d={diameter_um:.3f} Vm peak diff {vm_peak_diff:.4f} mV > {spec.vm_peak_atol_mV:.4f} mV (plot: {fig_path})"
        )
    if spec.vm_matrix_rmse_atol_mV is not None and matrix_rmse >= spec.vm_matrix_rmse_atol_mV:
        failures.append(
            f"{spec.name} d={diameter_um:.3f} Vm matrix RMSE {matrix_rmse:.4f} mV > {spec.vm_matrix_rmse_atol_mV:.4f} mV (plot: {fig_path})"
        )
    if spec.vm_matrix_corr_min is not None and matrix_corr <= spec.vm_matrix_corr_min:
        failures.append(
            f"{spec.name} d={diameter_um:.3f} Vm matrix corr {matrix_corr:.4f} < {spec.vm_matrix_corr_min:.4f} (plot: {fig_path})"
        )

    if failures:
        raise AssertionError("\n".join(failures))


@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_extracellular_models_vs_nrv(spec: ExtracellularSpec):
    failures: list[str] = []
    for diameter_um in spec.diameters_um:
        try:
            _run_extracellular_case(spec, float(diameter_um))
        except AssertionError as exc:
            failures.append(str(exc))
    if failures:
        raise AssertionError("\n\n".join(failures))
