from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrv

from axonscope import AxonInstance, degC, mV, ms, um
from axonscope.axons.myelinated import GainesMotor, GainesSensory, MRG
from axonscope.axons.unmyelinated import (
    HodgkinHuxley,
    RattayAberham,
    Schild94,
    Schild97,
    Sundt,
    Tigerholm,
)
from axonscope.analysis import conduction_velocity
from axonscope.stimulation import Stimulus
from tests.helpers import FakeSingleAxonResult
from tests.nrv._helpers import (
    align_rows_to_target_x,
    axonscope_x_um,
    crossing_times,
    interp_rows,
    normalize_nrv_matrix,
    run_axonscope_simulation,
    select_nearest_rows,
    symmetric_crossing_curve,
    velocity_from_crossing_times,
    velocity_from_peak_times,
)

pytestmark = pytest.mark.nrv_velocity

FIG_DIR = Path("figures/nrv_tests/velocity_vs_diameter")
VELOCITY_RTOL = 0.02


@dataclass(frozen=True)
class VelocitySpec:
    name: str
    diameters_um: tuple[float, ...]
    axonscope_factory: Callable[[float], object]
    nrv_factory: Callable[[float, object, float], object]
    tsim_ms: float
    dt_ms: float
    velocity_mode: Literal["all", "nodes"]
    plot_mode: Literal["all", "nodes"]
    threshold_mV: float
    exclude_radius_um: float
    velocity_fit_mode: Literal["direct", "symmetric", "raster", "crossing", "crossing_symmetric"] = "crossing_symmetric"
    raster_min_distance_ms: float = 0.2
    representative_index: int = 1
    require_propagation: bool = False
    vm_rmse_tolerance_mV: float | None = None


def _axonscope_matrix(axon, res, mode: Literal["all", "nodes"]) -> tuple[np.ndarray, np.ndarray]:
    vm = np.asarray(res.Vm, dtype=float).T
    x = axonscope_x_um(axon)
    if mode == "all":
        return x, vm
    idx = np.asarray(axon.node_indices, dtype=int)
    return x[idx], vm[idx]


def _nrv_matrix(
    results_nrv,
    mode: Literal["all", "nodes"],
    *,
    target_x_um: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_nrv = np.asarray(results_nrv["x_rec"], dtype=float)
    vm_nrv = normalize_nrv_matrix(results_nrv["V_mem"], t_nrv, x_nrv)

    if mode == "all":
        x_sel = x_nrv
        vm_sel = vm_nrv
    else:
        x_nodes = np.asarray(results_nrv.get("x_nodes", x_nrv), dtype=float).ravel()
        x_sel, vm_sel = select_nearest_rows(x_nrv, vm_nrv, x_nodes)

    if target_x_um is not None:
        x_sel, vm_sel, _ = align_rows_to_target_x(x_sel, vm_sel, target_x_um)
    return x_sel, vm_sel, t_nrv


def _mrg_stim_position(axon: MRG) -> float:
    center_node = int(axon.node_indices.shape[0] // 2)
    return float(axonscope_x_um(axon)[int(np.asarray(axon.node_indices)[center_node])])


def _velocity_from_symmetric_distances(
    x_um: np.ndarray,
    vm_space_time: np.ndarray,
    t_ms: np.ndarray,
    *,
    threshold_mV: float,
) -> float:
    x = np.asarray(x_um, dtype=float).ravel()
    vm = np.asarray(vm_space_time, dtype=float)
    t = np.asarray(t_ms, dtype=float).ravel()

    center_x = float(x[len(x) // 2])
    peaks = np.max(vm, axis=1)
    tpk = t[np.argmax(vm, axis=1)]
    dist_um = np.abs(x - center_x)
    mask = (dist_um > 0.0) & (peaks > float(threshold_mV))
    if np.count_nonzero(mask) < 2:
        return 0.0

    d = dist_um[mask]
    tp = tpk[mask]
    d_round = np.round(d, 6)
    uniq = np.unique(d_round)
    if uniq.size < 2:
        return 0.0
    d_u = np.asarray(uniq, dtype=float)
    t_u = np.asarray([tp[d_round == u].mean() for u in uniq], dtype=float)
    coeff = np.polyfit(t_u * 1e-3, d_u * 1e-6, 1)
    return float(coeff[0])


def _velocity_from_rasterized_matrix(
    x_um: np.ndarray,
    vm_space_time: np.ndarray,
    t_ms: np.ndarray,
    *,
    threshold_mV: float,
    min_distance_ms: float,
) -> float:
    class _DummyAxon:
        pass

    dummy = _DummyAxon()
    positions = np.asarray(x_um, dtype=float)

    class _DummyLayout:
        def position_values(self, *, unit="micrometer"):
            return positions

    dummy.layout = _DummyLayout()
    result = FakeSingleAxonResult(
        dummy,
        np.asarray(vm_space_time, dtype=float).T,
        np.asarray(t_ms, dtype=float),
    )
    return float(
        conduction_velocity(
            result,
            threshold_mV=threshold_mV,
            min_distance_ms=min_distance_ms,
        )
    )


def _make_hh_axon(d: float):
    ax = HodgkinHuxley(
        length=1000.0 * um,
        diameter=d * um,
        compartments=101,
        celsius=6.3 * degC,
        v_init=-70.0 * mV,
        include_passive_leak=True,
        g_pas=0.001,
        e_pas=-70.0,
    )
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=500.0 * um, current=Stimulus.pulse(start=1.0 * ms, duration=0.5 * ms, amplitude=1.0))
    sim.comparison_sample_position_um = 500.0
    return sim


def _make_hh_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=1000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-70.0, T=6.3, model="HH")
    ax.insert_I_Clamp(0.5, 1.0, 0.5, 1.0)
    return ax


def _make_rattay_axon(d: float):
    ax = RattayAberham(length=1000.0 * um, diameter=d * um, compartments=101, celsius=37.0 * degC)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=500.0 * um, current=Stimulus.pulse(start=1.0 * ms, duration=0.5 * ms, amplitude=1.0))
    sim.comparison_sample_position_um = 500.0
    return sim


def _make_rattay_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=1000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-70.0, T=37.0, model="Rattay_Aberham")
    ax.insert_I_Clamp(0.5, 1.0, 0.5, 1.0)
    return ax


def _make_sundt_axon(d: float):
    ax = Sundt(length=2000.0 * um, diameter=d * um, compartments=101, celsius=37.0 * degC)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=1000.0 * um, current=Stimulus.pulse(start=1.0 * ms, duration=0.5 * ms, amplitude=0.5))
    sim.comparison_sample_position_um = 1000.0
    return sim


def _make_sundt_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=2000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-60.0, T=37.0, model="Sundt")
    ax.insert_I_Clamp(0.5, 1.0, 0.5, 0.5)
    return ax


def _make_tigerholm_axon(d: float):
    ax = Tigerholm(length=5000.0 * um, diameter=d * um, compartments=101, celsius=37.0 * degC)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=2500.0 * um, current=Stimulus.pulse(start=5.0 * ms, duration=1.0 * ms, amplitude=0.5))
    sim.comparison_sample_position_um = 2500.0
    return sim


def _make_tigerholm_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=5000.0, Nsec=1, Nseg_per_sec=101, dt=dt_ms, v_init=-62.0, T=37.0, model="Tigerholm")
    ax.insert_I_Clamp(0.5, 5.0, 1.0, 0.5)
    return ax


def _make_schild94_axon(d: float):
    ax = Schild94(length=3000.0 * um, diameter=d * um, compartments=51)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=1500.0 * um, current=Stimulus.pulse(start=2.0 * ms, duration=1.0 * ms, amplitude=1.0))
    sim.comparison_sample_position_um = 1500.0
    return sim


def _make_schild94_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=3000.0, Nsec=1, Nseg_per_sec=51, dt=dt_ms, v_init=-48.0, T=37.0, model="Schild_94")
    ax.insert_I_Clamp(0.5, 2.0, 1.0, 1.0)
    return ax


def _make_schild97_axon(d: float):
    ax = Schild97(length=3000.0 * um, diameter=d * um, compartments=51)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=1500.0 * um, current=Stimulus.pulse(start=2.0 * ms, duration=1.0 * ms, amplitude=1.0))
    sim.comparison_sample_position_um = 1500.0
    return sim


def _make_schild97_nrv(d: float, _axon_as, dt_ms: float):
    ax = nrv.unmyelinated(y=0, z=0, d=d, L=3000.0, Nsec=1, Nseg_per_sec=51, dt=dt_ms, v_init=-48.0, T=37.0, model="Schild_97")
    ax.insert_I_Clamp(0.5, 2.0, 1.0, 1.0)
    return ax


def _make_mrg_axon(d: float):
    ax = MRG(diameter=d * um, nodes=11)
    stim_pos_um = _mrg_stim_position(ax)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=stim_pos_um * um, current=Stimulus.pulse(start=1.0 * ms, duration=0.1 * ms, amplitude=2.0))
    sim.comparison_sample_position_um = stim_pos_um
    return sim


def _make_mrg_nrv(d: float, axon_as, dt_ms: float):
    center_node = int(axon_as.node_indices.shape[0] // 2)
    ax = nrv.myelinated(
        0,
        0,
        d,
        float(axon_as.length),
        model="MRG",
        dt=dt_ms,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    ax.insert_I_Clamp_node(index=center_node, t_start=1.0, duration=0.1, amplitude=2.0)
    return ax


def _make_gaines_motor_axon(d: float):
    ax = GainesMotor(diameter=d * um, nodes=11)
    stim_pos_um = _mrg_stim_position(ax)
    sim = AxonInstance(ax)
    sim.add_current_clamp(
        position=stim_pos_um * um,
        current=Stimulus.pulse(start=1.0 * ms, duration=0.1 * ms, amplitude=5.0),
    )
    sim.comparison_sample_position_um = stim_pos_um
    return sim


def _make_gaines_sensory_axon(d: float):
    ax = GainesSensory(diameter=d * um, nodes=11)
    stim_pos_um = _mrg_stim_position(ax)
    sim = AxonInstance(ax)
    sim.add_current_clamp(
        position=stim_pos_um * um,
        current=Stimulus.pulse(start=1.0 * ms, duration=0.1 * ms, amplitude=5.0),
    )
    sim.comparison_sample_position_um = stim_pos_um
    return sim


def _make_gaines_nrv(model: str, d: float, axon_as, dt_ms: float):
    center_node = int(axon_as.node_indices.shape[0] // 2)
    ax = nrv.myelinated(
        0,
        0,
        d,
        float(axon_as.length),
        model=model,
        dt=dt_ms,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=None,
    )
    ax.insert_I_Clamp_node(index=center_node, t_start=1.0, duration=0.1, amplitude=5.0)
    return ax


def _make_gaines_motor_nrv(d: float, axon_as, dt_ms: float):
    return _make_gaines_nrv("Gaines_motor", d, axon_as, dt_ms)


def _make_gaines_sensory_nrv(d: float, axon_as, dt_ms: float):
    return _make_gaines_nrv("Gaines_sensory", d, axon_as, dt_ms)


SPECS = [
    VelocitySpec(
        name="hh",
        diameters_um=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        axonscope_factory=_make_hh_axon,
        nrv_factory=_make_hh_nrv,
        tsim_ms=10.0,
        dt_ms=0.001,
        velocity_mode="all",
        plot_mode="all",
        threshold_mV=0.0,
        exclude_radius_um=25.0,
        representative_index=3,
    ),
    VelocitySpec(
        name="rattay",
        diameters_um=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        axonscope_factory=_make_rattay_axon,
        nrv_factory=_make_rattay_nrv,
        tsim_ms=12.0,
        dt_ms=0.005,
        velocity_mode="all",
        plot_mode="all",
        threshold_mV=-30.0,
        exclude_radius_um=25.0,
        representative_index=4,
    ),
    VelocitySpec(
        name="sundt",
        diameters_um=(0.5, 0.6, 0.8, 1.0),
        axonscope_factory=_make_sundt_axon,
        nrv_factory=_make_sundt_nrv,
        tsim_ms=10.0,
        dt_ms=0.001,
        velocity_mode="all",
        plot_mode="all",
        threshold_mV=0.0,
        exclude_radius_um=50.0,
        representative_index=2,
    ),
    VelocitySpec(
        name="tigerholm",
        diameters_um=(0.5, 0.8, 1.0, 1.2),
        axonscope_factory=_make_tigerholm_axon,
        nrv_factory=_make_tigerholm_nrv,
        tsim_ms=30.0,
        dt_ms=0.025,
        velocity_mode="all",
        plot_mode="all",
        threshold_mV=-10.0,
        exclude_radius_um=200.0,
        representative_index=2,
    ),
    VelocitySpec(
        name="schild94",
        diameters_um=(0.6, 0.7, 0.8, 0.9, 1.0),
        axonscope_factory=_make_schild94_axon,
        nrv_factory=_make_schild94_nrv,
        tsim_ms=12.0,
        dt_ms=0.005,
        velocity_mode="all",
        plot_mode="all",
        threshold_mV=-10.0,
        exclude_radius_um=100.0,
        representative_index=2,
    ),
    VelocitySpec(
        name="schild97",
        diameters_um=(0.6, 0.7, 0.8, 0.9, 1.0),
        axonscope_factory=_make_schild97_axon,
        nrv_factory=_make_schild97_nrv,
        tsim_ms=12.0,
        dt_ms=0.005,
        velocity_mode="all",
        plot_mode="all",
        threshold_mV=-10.0,
        exclude_radius_um=100.0,
        representative_index=1,
    ),
    VelocitySpec(
        name="mrg",
        diameters_um=(5.7, 10.0, 14.0),
        axonscope_factory=_make_mrg_axon,
        nrv_factory=_make_mrg_nrv,
        tsim_ms=4.0,
        dt_ms=0.005,
        velocity_mode="nodes",
        plot_mode="all",
        threshold_mV=0.0,
        exclude_radius_um=100.0,
        velocity_fit_mode="crossing_symmetric",
        representative_index=1,
    ),
    VelocitySpec(
        name="gaines_motor",
        diameters_um=(5.7, 10.0, 14.0),
        axonscope_factory=_make_gaines_motor_axon,
        nrv_factory=_make_gaines_motor_nrv,
        tsim_ms=4.0,
        dt_ms=0.005,
        velocity_mode="nodes",
        plot_mode="all",
        threshold_mV=0.0,
        exclude_radius_um=100.0,
        velocity_fit_mode="crossing_symmetric",
        representative_index=1,
        require_propagation=True,
        vm_rmse_tolerance_mV=0.05,
    ),
    VelocitySpec(
        name="gaines_sensory",
        diameters_um=(5.7, 10.0, 14.0),
        axonscope_factory=_make_gaines_sensory_axon,
        nrv_factory=_make_gaines_sensory_nrv,
        tsim_ms=4.0,
        dt_ms=0.005,
        velocity_mode="nodes",
        plot_mode="all",
        threshold_mV=0.0,
        exclude_radius_um=100.0,
        velocity_fit_mode="crossing_symmetric",
        representative_index=1,
        require_propagation=True,
        vm_rmse_tolerance_mV=0.05,
    ),
]


def _plot_velocity_report(
    spec: VelocitySpec,
    diameters_um: list[float],
    vel_as: list[float],
    vel_nrv: list[float],
    rep_t_as: np.ndarray,
    rep_x_um: np.ndarray,
    rep_vm_as: np.ndarray,
    rep_t_nrv: np.ndarray,
    rep_vm_nrv: np.ndarray,
    rep_velocity_payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None,
) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rep_vm_nrv_i = interp_rows(rep_vm_nrv, rep_t_nrv, rep_t_as)
    err_pct = 100.0 * (np.asarray(vel_as) - np.asarray(vel_nrv)) / np.maximum(np.asarray(vel_nrv), 1e-12)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    axs[0, 0].plot(diameters_um, vel_as, "o-", lw=2, label="AxonScope")
    axs[0, 0].plot(diameters_um, vel_nrv, "s--", lw=2, label="NRV")
    axs[0, 0].set_title(f"{spec.name} velocity vs diameter")
    axs[0, 0].set_xlabel("Diameter [um]")
    axs[0, 0].set_ylabel("Velocity [m/s]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(diameters_um, err_pct, "o-", color="tab:red", lw=2)
    axs[0, 1].axhline(0.0, color="k", lw=1)
    axs[0, 1].axhline(100.0 * VELOCITY_RTOL, color="gray", ls="--", lw=1)
    axs[0, 1].axhline(-100.0 * VELOCITY_RTOL, color="gray", ls="--", lw=1)
    axs[0, 1].set_title("Relative error")
    axs[0, 1].set_xlabel("Diameter [um]")
    axs[0, 1].set_ylabel("Error [%]")
    axs[0, 1].grid(True, alpha=0.3)

    im0 = axs[1, 0].imshow(
        rep_vm_as,
        aspect="auto",
        origin="lower",
        extent=[float(rep_t_as[0]), float(rep_t_as[-1]), float(rep_x_um[0]), float(rep_x_um[-1])],
        cmap="viridis",
    )
    axs[1, 0].set_title("Representative AxonScope Vm heatmap")
    axs[1, 0].set_xlabel("Time [ms]")
    axs[1, 0].set_ylabel("Position [um]")
    fig.colorbar(im0, ax=axs[1, 0], label="Vm [mV]")

    im1 = axs[1, 1].imshow(
        rep_vm_nrv_i,
        aspect="auto",
        origin="lower",
        extent=[float(rep_t_as[0]), float(rep_t_as[-1]), float(rep_x_um[0]), float(rep_x_um[-1])],
        cmap="viridis",
    )
    axs[1, 1].set_title("Representative NRV Vm heatmap (aligned on AxonScope x)")
    axs[1, 1].set_xlabel("Time [ms]")
    axs[1, 1].set_ylabel("Position [um]")
    fig.colorbar(im1, ax=axs[1, 1], label="Vm [mV]")

    if rep_velocity_payload is not None:
        fig2, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
        x_um, tc_as, tc_nrv, diameter_um, center_x_um = rep_velocity_payload
        max_radius_um = 0.9 * float(
            np.max(np.abs(np.asarray(x_um, dtype=float) - float(center_x_um)))
        )
        for crossings, marker, linestyle, label in (
            (tc_as, "o", "-", "AxonScope"),
            (tc_nrv, "s", "--", "NRV"),
        ):
            dist, crossing = symmetric_crossing_curve(
                x_um,
                crossings,
                center_x_um=center_x_um,
                exclude_radius_um=spec.exclude_radius_um,
                max_radius_um=max_radius_um,
            )
            slope, intercept = np.polyfit(dist, crossing, 1)
            delay = crossing - intercept
            ax.plot(dist, delay, marker=marker, ls=linestyle, lw=2, label=label)
            ax.plot(
                dist,
                slope * dist,
                color=ax.lines[-1].get_color(),
                alpha=0.35,
                lw=1,
            )
        ax.set_title(f"{spec.name} d={diameter_um:.1f} um crossing-delay curve")
        ax.set_xlabel("Distance from center [um]")
        ax.set_ylabel("Delay from fitted propagation origin [ms]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig2_path = FIG_DIR / f"{spec.name}_velocity_vs_diameter_delay_curve.png"
        fig2.savefig(fig2_path, dpi=140, bbox_inches="tight")
        plt.close(fig2)

    fig_path = FIG_DIR / f"{spec.name}_velocity_vs_diameter.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _run_velocity_spec(spec: VelocitySpec) -> None:
    vel_as: list[float] = []
    vel_nrv: list[float] = []
    rep_payload = None
    rep_velocity_payload = None

    for i, d in enumerate(spec.diameters_um):
        axon = spec.axonscope_factory(float(d))
        res = run_axonscope_simulation(axon, tsim=spec.tsim_ms, dt=spec.dt_ms)

        axon_nrv = spec.nrv_factory(float(d), axon, spec.dt_ms)
        results_nrv = axon_nrv.simulate(t_sim=spec.tsim_ms)

        center_x_um = float(getattr(axon, "comparison_sample_position_um", axon.length / 2.0))

        x_as_vel, vm_as_vel = _axonscope_matrix(axon, res, spec.velocity_mode)
        x_nrv_vel, vm_nrv_vel, t_nrv = _nrv_matrix(
            results_nrv,
            spec.velocity_mode,
            target_x_um=x_as_vel,
        )
        t_as = np.asarray(res.t, dtype=float)
        vm_nrv_vel_i = interp_rows(vm_nrv_vel, t_nrv, t_as)

        if spec.velocity_fit_mode == "symmetric":
            v_as = _velocity_from_symmetric_distances(
                x_as_vel,
                vm_as_vel,
                t_as,
                threshold_mV=spec.threshold_mV,
            )
            v_nrv = _velocity_from_symmetric_distances(
                x_nrv_vel,
                vm_nrv_vel_i,
                t_as,
                threshold_mV=spec.threshold_mV,
            )
        elif spec.velocity_fit_mode == "raster":
            v_as = _velocity_from_rasterized_matrix(
                x_as_vel,
                vm_as_vel,
                t_as,
                threshold_mV=spec.threshold_mV,
                min_distance_ms=spec.raster_min_distance_ms,
            )
            v_nrv = _velocity_from_rasterized_matrix(
                x_nrv_vel,
                vm_nrv_vel_i,
                t_as,
                threshold_mV=spec.threshold_mV,
                min_distance_ms=spec.raster_min_distance_ms,
            )
        elif spec.velocity_fit_mode == "crossing":
            v_as = velocity_from_crossing_times(
                x_as_vel,
                vm_as_vel,
                t_as,
                center_x_um=center_x_um,
                threshold_mV=spec.threshold_mV,
                exclude_radius_um=spec.exclude_radius_um,
                fit_mode="direct",
            )
            v_nrv = velocity_from_crossing_times(
                x_nrv_vel,
                vm_nrv_vel_i,
                t_as,
                center_x_um=center_x_um,
                threshold_mV=spec.threshold_mV,
                exclude_radius_um=spec.exclude_radius_um,
                fit_mode="direct",
            )
        elif spec.velocity_fit_mode == "crossing_symmetric":
            max_radius_um = 0.9 * float(np.max(np.abs(x_as_vel - center_x_um)))
            v_as = velocity_from_crossing_times(
                x_as_vel,
                vm_as_vel,
                t_as,
                center_x_um=center_x_um,
                threshold_mV=spec.threshold_mV,
                exclude_radius_um=spec.exclude_radius_um,
                max_radius_um=max_radius_um,
                fit_mode="symmetric",
            )
            v_nrv = velocity_from_crossing_times(
                x_nrv_vel,
                vm_nrv_vel_i,
                t_as,
                center_x_um=center_x_um,
                threshold_mV=spec.threshold_mV,
                exclude_radius_um=spec.exclude_radius_um,
                max_radius_um=max_radius_um,
                fit_mode="symmetric",
            )
        else:
            v_as = velocity_from_peak_times(
                x_as_vel,
                vm_as_vel,
                t_as,
                center_x_um=center_x_um,
                threshold_mV=spec.threshold_mV,
                exclude_radius_um=spec.exclude_radius_um,
            )
            v_nrv = velocity_from_peak_times(
                x_nrv_vel,
                vm_nrv_vel_i,
                t_as,
                center_x_um=center_x_um,
                threshold_mV=spec.threshold_mV,
                exclude_radius_um=spec.exclude_radius_um,
            )

        print(f"{spec.name} d={d:.3f} um | AxonScope={v_as:.4f} m/s | NRV={v_nrv:.4f} m/s")
        vel_as.append(v_as)
        vel_nrv.append(v_nrv)

        if i == spec.representative_index:
            x_as_plot, vm_as_plot = _axonscope_matrix(axon, res, spec.plot_mode)
            _, vm_nrv_plot, _ = _nrv_matrix(
                results_nrv,
                spec.plot_mode,
                target_x_um=x_as_plot,
            )
            rep_payload = (t_as, x_as_plot, vm_as_plot, t_nrv, vm_nrv_plot)
            if spec.velocity_fit_mode in {"crossing", "crossing_symmetric"}:
                rep_velocity_payload = (
                    x_as_vel,
                    crossing_times(vm_as_vel, t_as, spec.threshold_mV),
                    crossing_times(vm_nrv_vel_i, t_as, spec.threshold_mV),
                    float(d),
                    center_x_um,
                )

    assert rep_payload is not None
    fig_path = _plot_velocity_report(
        spec,
        list(spec.diameters_um),
        vel_as,
        vel_nrv,
        rep_payload[0],
        rep_payload[1],
        rep_payload[2],
        rep_payload[3],
        rep_payload[4],
        rep_velocity_payload,
    )

    failures = []
    for d, v_as, v_nrv in zip(spec.diameters_um, vel_as, vel_nrv, strict=True):
        err = abs(v_as - v_nrv)
        tol = max(0.001, VELOCITY_RTOL * max(abs(v_nrv), 1e-12))
        if not np.isfinite(v_as) or not np.isfinite(v_nrv):
            failures.append(f"{spec.name} d={d:.3f} um produced non-finite velocity.")
            continue
        if spec.require_propagation and (v_as <= 0.0 or v_nrv <= 0.0):
            failures.append(
                f"{spec.name} d={d:.3f} um did not propagate in both implementations."
            )
            continue
        if err > tol:
            failures.append(
                f"{spec.name} d={d:.3f} um |Δv|={err:.4f} m/s > tol={tol:.4f} m/s "
                f"(plot: {fig_path})"
            )

    if failures:
        raise AssertionError("\n".join(failures))

    if spec.vm_rmse_tolerance_mV is not None:
        vm_nrv_aligned = interp_rows(rep_payload[4], rep_payload[3], rep_payload[0])
        vm_rmse = float(np.sqrt(np.mean((rep_payload[2] - vm_nrv_aligned) ** 2)))
        print(f"{spec.name} representative Vm RMSE={vm_rmse:.6f} mV")
        if vm_rmse > spec.vm_rmse_tolerance_mV:
            raise AssertionError(
                f"{spec.name} representative Vm RMSE={vm_rmse:.6f} mV > "
                f"{spec.vm_rmse_tolerance_mV:.6f} mV (plot: {fig_path})"
            )


@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_velocity_models_vs_nrv(spec: VelocitySpec):
    _run_velocity_spec(spec)
