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

from axonscope import AxonInstance, degC, mV, ms, um
from axonscope.axons.myelinated import MRG
from axonscope.axons.unmyelinated import (
    HodgkinHuxley,
    RattayAberham,
    Schild94,
    Schild97,
    Sundt,
    Tigerholm,
)
from axonscope.stimulation import Stimulus
from tests.nrv._helpers import axonscope_x_um, run_axonscope_simulation

pytestmark = pytest.mark.nrv_intracellular

FIG_DIR = Path("figures/nrv_tests/intracellular")
AXONSCOPE_TO_NRV_CURRENT_SCALE = 1e-3


@dataclass(frozen=True)
class IntracellularSpec:
    name: str
    axonscope_factory: Callable[[], object]
    nrv_factory: Callable[[object, float], object]
    tsim_ms: float
    dt_ms: float
    matrix_mode: str
    current_pairs: tuple[tuple[str, str], ...]
    gate_pairs: tuple[tuple[str, str], ...]
    state_pairs: tuple[tuple[str, str], ...]
    vm_rmse_atol_mV: float
    vm_peak_atol_mV: float
    current_rmse_atol: float
    current_max_atol: float
    gate_rmse_atol: float
    gate_max_atol: float
    state_rmse_atol: float
    state_max_atol: float
    nrv_only_observables: tuple[str, ...] = ()
    nrv_key_overrides: dict[str, str] | None = None


def _enable_nrv_recordings(axon_nrv) -> None:
    axon_nrv.record_V_mem = True
    axon_nrv.record_I_ions = True
    axon_nrv.record_particles = True
    if hasattr(axon_nrv, "record_particules"):
        axon_nrv.record_particules = True


def _normalize_nrv_matrix(values: np.ndarray, t_ms: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D NRV array, got shape {arr.shape}.")
    if arr.shape == (x_um.size, t_ms.size):
        return arr
    if arr.shape == (t_ms.size, x_um.size):
        return arr.T
    if arr.shape[0] == x_um.size:
        return arr
    if arr.shape[1] == x_um.size:
        return arr.T
    raise ValueError(
        f"Could not align NRV array of shape {arr.shape} with x={x_um.size} and t={t_ms.size}."
    )


def _interp_rows(values_by_space_time: np.ndarray, t_src_ms: np.ndarray, t_dst_ms: np.ndarray) -> np.ndarray:
    out = np.empty((values_by_space_time.shape[0], t_dst_ms.size), dtype=float)
    for i in range(values_by_space_time.shape[0]):
        out[i] = np.interp(t_dst_ms, t_src_ms, values_by_space_time[i])
    return out


def _align_rows_to_target_x(
    x_source_um: np.ndarray,
    matrix_source: np.ndarray,
    x_target_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_source = np.asarray(x_source_um, dtype=float).ravel()
    matrix = np.asarray(matrix_source, dtype=float)
    x_target = np.asarray(x_target_um, dtype=float).ravel()
    idx = np.asarray([int(np.argmin(np.abs(x_source - xi))) for xi in x_target], dtype=int)
    return x_target, matrix[idx], idx


def _axonspace_vm_matrix(axon, res, matrix_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Vm = np.asarray(res.Vm, dtype=float)
    x = axonscope_x_um(axon)
    if matrix_mode == "all":
        indices = np.arange(axon.n_compartments, dtype=int)
        return Vm.T, x, indices
    if matrix_mode == "nodes":
        indices = np.asarray(axon.node_indices, dtype=int)
        return Vm[:, indices].T, x[indices], indices
    raise ValueError(f"Unsupported matrix mode: {matrix_mode}")


def _nrv_vm_matrix(results_nrv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_nrv = np.asarray(results_nrv["x_rec"], dtype=float)
    Vm_nrv = _normalize_nrv_matrix(results_nrv["V_mem"], t_nrv, x_nrv)
    return Vm_nrv, x_nrv, t_nrv


def _sample_indices(
    as_x_um: np.ndarray,
    nrv_x_um: np.ndarray,
    sample_position_um: float | None = None,
) -> tuple[int, int]:
    if sample_position_um is None:
        return int(len(as_x_um) // 2), int(len(nrv_x_um) // 2)
    return (
        int(np.argmin(np.abs(np.asarray(as_x_um, dtype=float) - float(sample_position_um)))),
        int(np.argmin(np.abs(np.asarray(nrv_x_um, dtype=float) - float(sample_position_um)))),
    )


def _trace_metrics(ref: np.ndarray, test: np.ndarray) -> tuple[float, float, float]:
    diff = np.asarray(test, dtype=float) - np.asarray(ref, dtype=float)
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    q99_abs = float(np.quantile(np.abs(diff), 0.99))
    return rmse, max_abs, q99_abs


def _recorded_trace(res, group: str, name: str, compartment_index: int) -> np.ndarray:
    assert res.recordings is not None
    return np.asarray(res.recordings[group][name], dtype=float)[:, compartment_index]


def _nrv_trace(results_nrv, key: str, row_index: int, t_dst_ms: np.ndarray) -> np.ndarray:
    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_nrv = np.asarray(results_nrv["x_rec"], dtype=float)
    matrix = _normalize_nrv_matrix(results_nrv[key], t_nrv, x_nrv)
    return np.interp(t_dst_ms, t_nrv, matrix[row_index])


def _resolve_nrv_key(spec: IntracellularSpec, key: str) -> str:
    if spec.nrv_key_overrides is None:
        return key
    return spec.nrv_key_overrides.get(key, key)


def _plot_intracellular_report(
    spec: IntracellularSpec,
    axon,
    res,
    vm_local_nrv: np.ndarray,
    vm_as_matrix: np.ndarray,
    vm_nrv_interp: np.ndarray,
    sample_as_idx: int,
    metrics_lines: list[str],
    current_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    gate_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    state_pairs: list[tuple[str, np.ndarray, np.ndarray]],
) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_as = np.asarray(res.t, dtype=float)
    as_x = axonscope_x_um(axon)
    vm_local_as = np.asarray(res.Vm, dtype=float)[:, sample_as_idx]

    fig, axs = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)

    axs[0, 0].plot(t_as, vm_local_as, lw=2.0, zorder=2, label="AxonScope")
    axs[0, 0].plot(t_as, vm_local_nrv, "--", lw=2.2, zorder=4, label="NRV")
    axs[0, 0].set_title(f"{spec.name} local Vm")
    axs[0, 0].set_xlabel("Time [ms]")
    axs[0, 0].set_ylabel("Vm [mV]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    im_as = axs[0, 1].imshow(
        vm_as_matrix,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), float(as_x[0]), float(as_x[-1])],
        cmap="viridis",
    )
    axs[0, 1].set_title("AxonScope heatmap")
    axs[0, 1].set_xlabel("Time [ms]")
    axs[0, 1].set_ylabel("Position [um]")
    fig.colorbar(im_as, ax=axs[0, 1], label="Vm [mV]")

    im_nrv = axs[1, 0].imshow(
        vm_nrv_interp,
        aspect="auto",
        origin="lower",
        extent=[float(t_as[0]), float(t_as[-1]), float(as_x[0]), float(as_x[-1])],
        cmap="viridis",
    )
    axs[1, 0].set_title("NRV heatmap (aligned on AxonScope x/time)")
    axs[1, 0].set_xlabel("Time [ms]")
    axs[1, 0].set_ylabel("Position [um]")
    fig.colorbar(im_nrv, ax=axs[1, 0], label="Vm [mV]")

    ax_curr = axs[1, 1]
    if current_pairs:
        for i, (label, as_trace, nrv_trace) in enumerate(current_pairs):
            color = plt.cm.tab20(i % 20)
            ax_curr.plot(
                t_as,
                as_trace,
                color=color,
                lw=1.8,
                zorder=2,
                label=f"{label} AS",
            )
            ax_curr.plot(
                t_as,
                nrv_trace,
                color=color,
                lw=2.0,
                ls="--",
                zorder=4,
                label=f"{label} NRV",
            )
    else:
        ax_curr.text(0.5, 0.5, "No current comparison", ha="center", va="center")
    ax_curr.set_title("Local ionic currents")
    ax_curr.set_xlabel("Time [ms]")
    ax_curr.set_ylabel("Current density [mA/cm²]")
    ax_curr.grid(True, alpha=0.3)
    if current_pairs:
        ax_curr.legend(fontsize=8, ncol=2)

    ax_states = axs[2, 0]
    merged_pairs = gate_pairs + state_pairs
    if merged_pairs:
        for i, (label, as_trace, nrv_trace) in enumerate(merged_pairs):
            color = plt.cm.tab20(i % 20)
            ax_states.plot(
                t_as,
                as_trace,
                color=color,
                lw=1.8,
                zorder=2,
                label=f"{label} AS",
            )
            ax_states.plot(
                t_as,
                nrv_trace,
                color=color,
                lw=2.0,
                ls="--",
                zorder=4,
                label=f"{label} NRV",
            )
    else:
        ax_states.text(0.5, 0.5, "No gate/state comparison", ha="center", va="center")
    ax_states.set_title("Local gates / states")
    ax_states.set_xlabel("Time [ms]")
    ax_states.set_ylabel("State value")
    ax_states.grid(True, alpha=0.3)
    if merged_pairs:
        ax_states.legend(fontsize=7, ncol=2)

    axs[2, 1].axis("off")
    axs[2, 1].text(
        0.01,
        0.99,
        "\n".join(metrics_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
    )

    fig.suptitle(
        f"{spec.name} intracellular validation vs NRV\n"
        f"sample x={float(as_x[sample_as_idx]):.1f} µm | dt={spec.dt_ms} ms | tsim={spec.tsim_ms} ms",
        fontsize=13,
    )

    out_path = FIG_DIR / f"{spec.name.lower()}_intracellular_vs_nrv.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _run_intracellular_spec(spec: IntracellularSpec) -> None:
    pytest.skip("dense observable NRV comparison awaits batch-native observables")
    axon = spec.axonscope_factory()
    res = run_axonscope_simulation(
        axon,
        tsim=spec.tsim_ms,
        dt=spec.dt_ms,
        record_observables=True,
    )
    assert res.recordings is not None
    failures: list[str] = []

    axon_nrv = spec.nrv_factory(axon, spec.dt_ms)
    _enable_nrv_recordings(axon_nrv)
    results_nrv = axon_nrv.simulate(t_sim=spec.tsim_ms)

    t_as = np.asarray(res.t, dtype=float)
    vm_as_matrix, as_x_um, _ = _axonspace_vm_matrix(axon, res, spec.matrix_mode)
    vm_nrv_matrix, x_nrv, t_nrv = _nrv_vm_matrix(results_nrv)
    _, vm_nrv_matrix_aligned, _ = _align_rows_to_target_x(x_nrv, vm_nrv_matrix, as_x_um)
    vm_nrv_interp = _interp_rows(vm_nrv_matrix_aligned, t_nrv, t_as)
    sample_position_um = getattr(axon, "comparison_sample_position_um", None)
    sample_as_idx, sample_nrv_idx = _sample_indices(as_x_um, x_nrv, sample_position_um)

    vm_local_as = np.asarray(res.Vm, dtype=float)[:, sample_as_idx]
    vm_local_nrv = np.interp(t_as, t_nrv, vm_nrv_matrix_aligned[sample_as_idx])
    vm_rmse, _, _ = _trace_metrics(vm_local_nrv, vm_local_as)
    peak_diff = float(abs(float(vm_local_as.max()) - float(vm_local_nrv.max())))

    metrics_lines = [
        f"Vm RMSE      : {vm_rmse:8.4f} mV",
        f"Vm peak diff : {peak_diff:8.4f} mV",
        "Currents: AxonScope traces scaled by 1e-3 to match NRV current units",
    ]

    current_plot_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for as_name, nrv_name in spec.current_pairs:
        as_trace = AXONSCOPE_TO_NRV_CURRENT_SCALE * _recorded_trace(
            res, "currents", as_name, sample_as_idx
        )
        nrv_trace = _nrv_trace(
            results_nrv,
            _resolve_nrv_key(spec, nrv_name),
            sample_nrv_idx,
            t_as,
        )
        rmse, max_abs, q99_abs = _trace_metrics(nrv_trace, as_trace)
        metrics_lines.append(
            f"{as_name:12s}: rmse={rmse:8.4f} q99={q99_abs:8.4f} max={max_abs:8.4f}"
        )
        if not (rmse < spec.current_rmse_atol):
            failures.append(
                f"{spec.name} {as_name} RMSE {rmse:.4f} > {spec.current_rmse_atol:.4f}"
            )
        if not (max_abs < spec.current_max_atol):
            failures.append(
                f"{spec.name} {as_name} max |Δ| {max_abs:.4f} > {spec.current_max_atol:.4f}"
            )
        current_plot_pairs.append((as_name, as_trace, nrv_trace))

    gate_plot_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for as_name, nrv_name in spec.gate_pairs:
        as_trace = _recorded_trace(res, "gates", as_name, sample_as_idx)
        nrv_trace = _nrv_trace(results_nrv, _resolve_nrv_key(spec, nrv_name), sample_nrv_idx, t_as)
        rmse, max_abs, q99_abs = _trace_metrics(nrv_trace, as_trace)
        metrics_lines.append(
            f"{as_name:12s}: rmse={rmse:8.4f} q99={q99_abs:8.4f} max={max_abs:8.4f}"
        )
        if not (rmse < spec.gate_rmse_atol):
            failures.append(
                f"{spec.name} {as_name} RMSE {rmse:.4f} > {spec.gate_rmse_atol:.4f}"
            )
        if not (q99_abs < spec.gate_max_atol):
            failures.append(
                f"{spec.name} {as_name} q99 |Δ| {q99_abs:.4f} > {spec.gate_max_atol:.4f}"
            )
        gate_plot_pairs.append((as_name, as_trace, nrv_trace))

    state_plot_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for as_name, nrv_name in spec.state_pairs:
        as_trace = _recorded_trace(res, "states", as_name, sample_as_idx)
        nrv_trace = _nrv_trace(results_nrv, _resolve_nrv_key(spec, nrv_name), sample_nrv_idx, t_as)
        rmse, max_abs, q99_abs = _trace_metrics(nrv_trace, as_trace)
        metrics_lines.append(
            f"{as_name:12s}: rmse={rmse:8.4f} q99={q99_abs:8.4f} max={max_abs:8.4f}"
        )
        if not (rmse < spec.state_rmse_atol):
            failures.append(
                f"{spec.name} {as_name} RMSE {rmse:.4f} > {spec.state_rmse_atol:.4f}"
            )
        if not (q99_abs < spec.state_max_atol):
            failures.append(
                f"{spec.name} {as_name} q99 |Δ| {q99_abs:.4f} > {spec.state_max_atol:.4f}"
            )
        state_plot_pairs.append((as_name, as_trace, nrv_trace))

    if spec.nrv_only_observables:
        metrics_lines.append("")
        metrics_lines.append("NRV-only observables:")
        metrics_lines.extend(f"  - {name}" for name in spec.nrv_only_observables)

    figure_path = _plot_intracellular_report(
        spec,
        axon,
        res,
        vm_local_nrv,
        vm_as_matrix,
        vm_nrv_interp,
        sample_as_idx,
        metrics_lines,
        current_plot_pairs,
        gate_plot_pairs,
        state_plot_pairs,
    )

    assert np.isfinite(np.asarray(res.Vm)).all()
    if not (vm_rmse < spec.vm_rmse_atol_mV):
        failures.append(
            f"{spec.name} Vm RMSE {vm_rmse:.4f} mV > {spec.vm_rmse_atol_mV:.4f} mV "
            f"(plot: {figure_path})"
        )
    if not (peak_diff < spec.vm_peak_atol_mV):
        failures.append(
            f"{spec.name} Vm peak diff {peak_diff:.4f} mV > {spec.vm_peak_atol_mV:.4f} mV "
            f"(plot: {figure_path})"
        )
    if failures:
        raise AssertionError("\n".join(failures))


def _make_hh_axon():
    ax = HodgkinHuxley(
        length=1000.0 * um,
        diameter=0.5 * um,
        compartments=101,
        celsius=6.3 * degC,
        v_init=-70.0 * mV,
    )
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=500.0 * um, current=Stimulus.pulse(start=1.0 * ms, duration=1.0 * ms, amplitude=2.0))
    return sim


def _make_hh_nrv(_, dt_ms: float):
    ax = nrv.unmyelinated(
        0,
        0,
        0.5,
        1000.0,
        dt=dt_ms,
        Nsec=1,
        Nseg_per_sec=101,
        model="HH",
        v_init=-70.0,
        T=6.3,
    )
    ax.insert_I_Clamp(0.5, 1.0, 1.0, 2.0)
    return ax


def _make_rattay_axon():
    ax = RattayAberham(length=1000.0 * um, diameter=0.8 * um, compartments=51, celsius=37.0 * degC)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=500.0 * um, current=Stimulus.pulse(start=1.0 * ms, duration=1.0 * ms, amplitude=2.0))
    return sim


def _make_rattay_nrv(_, dt_ms: float):
    ax = nrv.unmyelinated(
        0,
        0,
        0.8,
        1000.0,
        dt=dt_ms,
        Nsec=1,
        Nseg_per_sec=51,
        model="Rattay_Aberham",
        v_init=-70.0,
        T=37.0,
    )
    ax.insert_I_Clamp(0.5, 1.0, 1.0, 2.0)
    return ax


def _make_sundt_axon():
    ax = Sundt(length=1000.0 * um, diameter=0.5 * um, compartments=101, celsius=37.0 * degC)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=500.0 * um, current=Stimulus.pulse(start=1.0 * ms, duration=1.0 * ms, amplitude=2.0))
    return sim


def _make_sundt_nrv(_, dt_ms: float):
    ax = nrv.unmyelinated(
        0,
        0,
        0.5,
        1000.0,
        dt=dt_ms,
        Nsec=1,
        Nseg_per_sec=101,
        model="Sundt",
        v_init=-60.0,
        T=37.0,
    )
    ax.insert_I_Clamp(0.5, 1.0, 1.0, 2.0)
    return ax


def _make_tigerholm_axon():
    ax = Tigerholm(length=5000.0 * um, diameter=1.0 * um, compartments=101, celsius=37.0 * degC)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=2500.0 * um, current=Stimulus.pulse(start=5.0 * ms, duration=1.0 * ms, amplitude=2.0))
    return sim


def _make_tigerholm_nrv(_, dt_ms: float):
    ax = nrv.unmyelinated(
        0,
        0,
        1.0,
        5000.0,
        dt=dt_ms,
        Nsec=1,
        Nseg_per_sec=101,
        model="Tigerholm",
        v_init=-62.0,
        T=37.0,
    )
    ax.insert_I_Clamp(0.5, 5.0, 1.0, 2.0)
    return ax


def _make_schild94_axon():
    ax = Schild94(length=3000.0 * um, diameter=0.8 * um, compartments=51)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=1500.0 * um, current=Stimulus.pulse(start=2.0 * ms, duration=1.0 * ms, amplitude=1.0))
    return sim


def _make_schild94_nrv(_, dt_ms: float):
    ax = nrv.unmyelinated(
        0,
        0,
        0.8,
        3000.0,
        dt=dt_ms,
        Nsec=1,
        Nseg_per_sec=51,
        model="Schild_94",
        v_init=-48.0,
        T=37.0,
    )
    ax.insert_I_Clamp(0.5, 2.0, 1.0, 1.0)
    return ax


def _make_schild97_axon():
    ax = Schild97(length=3000.0 * um, diameter=0.8 * um, compartments=51)
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=1500.0 * um, current=Stimulus.pulse(start=2.0 * ms, duration=1.0 * ms, amplitude=1.0))
    return sim


def _make_schild97_nrv(_, _dt_ms: float):
    ax = nrv.unmyelinated(
        0,
        0,
        0.8,
        3000.0,
        dt=0.005,
        Nsec=1,
        Nseg_per_sec=51,
        model="Schild_97",
        v_init=-48.0,
        T=37.0,
    )
    ax.insert_I_Clamp(0.5, 2.0, 1.0, 1.0)
    return ax


def _make_mrg_axon():
    ax = MRG(diameter=10.0 * um, nodes=7)
    stim_node = int(ax.node_indices.shape[0] // 2)
    stim_pos_um = float(axonscope_x_um(ax)[int(ax.node_indices[stim_node])])
    sim = AxonInstance(ax)
    sim.add_current_clamp(position=stim_pos_um * um, current=Stimulus.pulse(start=1.0 * ms, duration=0.1 * ms, amplitude=2.0))
    sim.comparison_sample_position_um = stim_pos_um
    return sim


def _make_mrg_nrv(axon_as, dt_ms: float):
    stim_node = int(axon_as.node_indices.shape[0] // 2)
    ax = nrv.myelinated(
        0,
        0,
        10.0,
        float(axon_as.length),
        model="MRG",
        dt=dt_ms,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    ax.insert_I_Clamp_node(index=stim_node, t_start=1.0, duration=0.1, amplitude=2.0)
    return ax


SPECS = [
    IntracellularSpec(
        name="hh",
        axonscope_factory=_make_hh_axon,
        nrv_factory=_make_hh_nrv,
        tsim_ms=10.0,
        dt_ms=0.001,
        matrix_mode="all",
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("m", "m"), ("n", "n"), ("h", "h")),
        state_pairs=(),
        vm_rmse_atol_mV=5.0,
        vm_peak_atol_mV=5.0,
        current_rmse_atol=4.0,
        current_max_atol=15.0,
        gate_rmse_atol=0.05,
        gate_max_atol=0.15,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
    ),
    IntracellularSpec(
        name="rattay",
        axonscope_factory=_make_rattay_axon,
        nrv_factory=_make_rattay_nrv,
        tsim_ms=15.0,
        dt_ms=0.01,
        matrix_mode="all",
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("m", "m"), ("n", "n"), ("h", "h")),
        state_pairs=(),
        vm_rmse_atol_mV=5.0,
        vm_peak_atol_mV=5.0,
        current_rmse_atol=4.0,
        current_max_atol=15.0,
        gate_rmse_atol=0.05,
        gate_max_atol=0.15,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
    ),
    IntracellularSpec(
        name="sundt",
        axonscope_factory=_make_sundt_axon,
        nrv_factory=_make_sundt_nrv,
        tsim_ms=10.0,
        dt_ms=0.001,
        matrix_mode="all",
        current_pairs=(("I_na", "I_na"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("m", "m"), ("n", "n"), ("h", "h")),
        state_pairs=(),
        vm_rmse_atol_mV=5.0,
        vm_peak_atol_mV=6.0,
        current_rmse_atol=5.0,
        current_max_atol=18.0,
        gate_rmse_atol=0.06,
        gate_max_atol=0.18,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
    ),
    IntracellularSpec(
        name="tigerholm",
        axonscope_factory=_make_tigerholm_axon,
        nrv_factory=_make_tigerholm_nrv,
        tsim_ms=30.0,
        dt_ms=0.025,
        matrix_mode="all",
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
        vm_rmse_atol_mV=5.0,
        vm_peak_atol_mV=10.0,
        current_rmse_atol=12.0,
        current_max_atol=40.0,
        gate_rmse_atol=0.10,
        gate_max_atol=0.30,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
        nrv_only_observables=("I_ca",),
    ),
    IntracellularSpec(
        name="schild94",
        axonscope_factory=_make_schild94_axon,
        nrv_factory=_make_schild94_nrv,
        tsim_ms=20.0,
        dt_ms=0.01,
        matrix_mode="all",
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
        vm_rmse_atol_mV=5.0,
        vm_peak_atol_mV=10.0,
        current_rmse_atol=15.0,
        current_max_atol=45.0,
        gate_rmse_atol=0.12,
        gate_max_atol=0.35,
        state_rmse_atol=0.12,
        state_max_atol=0.35,
        nrv_key_overrides={
            "l_naf": "h_nas",
            "m_nas": "l_naf",
            "h_nas": "m_nas",
        },
    ),
    IntracellularSpec(
        name="schild97",
        axonscope_factory=_make_schild97_axon,
        nrv_factory=_make_schild97_nrv,
        tsim_ms=20.0,
        dt_ms=0.01,
        matrix_mode="all",
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
        vm_rmse_atol_mV=5.0,
        vm_peak_atol_mV=10.0,
        current_rmse_atol=15.0,
        current_max_atol=45.0,
        gate_rmse_atol=0.12,
        gate_max_atol=0.35,
        state_rmse_atol=0.12,
        state_max_atol=0.35,
    ),
    IntracellularSpec(
        name="mrg",
        axonscope_factory=_make_mrg_axon,
        nrv_factory=_make_mrg_nrv,
        tsim_ms=4.0,
        dt_ms=0.005,
        matrix_mode="all",
        current_pairs=(("I_na", "I_na"), ("I_nap", "I_nap"), ("I_k", "I_k"), ("I_l", "I_l")),
        gate_pairs=(("mp", "mp"), ("m", "m"), ("h", "h"), ("s", "s")),
        state_pairs=(),
        vm_rmse_atol_mV=6.0,
        vm_peak_atol_mV=12.0,
        current_rmse_atol=60.0,
        current_max_atol=180.0,
        gate_rmse_atol=0.08,
        gate_max_atol=0.25,
        state_rmse_atol=0.0,
        state_max_atol=0.0,
    ),
]


@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_intracellular_models_vs_nrv(spec: IntracellularSpec):
    _run_intracellular_spec(spec)
