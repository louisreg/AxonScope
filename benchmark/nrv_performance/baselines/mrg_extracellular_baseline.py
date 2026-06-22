from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import numpy as np

from axonscope import AxonInstance, um
from axonscope.axons import MRG
from axonscope.stimulation import AnalyticalExtracellularContext, PointSourceElectrode
from axonscope.solvers import CrankNicholson
from axonscope.stimulation import Stimulus


SIGMA_S_M = 0.2
ELECTRODE_Y_UM = 100.0
ELECTRODE_Z_UM = 0.0


def _enable_nrv_recordings(axon_nrv) -> None:
    axon_nrv.record_V_mem = True
    axon_nrv.record_I_ions = True
    axon_nrv.record_particles = True
    axon_nrv.record_g_ions = True
    axon_nrv.record_g_mem = True
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
    raise ValueError(f"Could not align NRV array of shape {arr.shape} with x={x_um.size} and t={t_ms.size}.")


def _align_rows_to_target_x(
    x_source_um: np.ndarray,
    matrix_source: np.ndarray,
    x_target_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_source = np.asarray(x_source_um, dtype=float).ravel()
    matrix = np.asarray(matrix_source, dtype=float)
    x_target = np.asarray(x_target_um, dtype=float).ravel()
    idx = np.asarray([int(np.argmin(np.abs(x_source - xi))) for xi in x_target], dtype=int)
    return matrix[idx], idx


def _interp_rows(values_by_space_time: np.ndarray, t_src_ms: np.ndarray, t_dst_ms: np.ndarray) -> np.ndarray:
    values = np.asarray(values_by_space_time, dtype=float)
    out = np.empty((values.shape[0], t_dst_ms.size), dtype=float)
    for i in range(values.shape[0]):
        out[i] = np.interp(t_dst_ms, t_src_ms, values[i])
    return out


def _shifted_interp_rows(
    values_by_space_time: np.ndarray,
    t_src_ms: np.ndarray,
    t_dst_ms: np.ndarray,
    *,
    shift_steps: int,
) -> np.ndarray:
    values = np.asarray(values_by_space_time, dtype=float)
    t_src = np.asarray(t_src_ms, dtype=float).ravel()
    if shift_steps == 0:
        return _interp_rows(values, t_src, t_dst_ms)
    if abs(shift_steps) >= t_src.size:
        raise ValueError(f"shift_steps={shift_steps} is too large for t_src size {t_src.size}.")
    if shift_steps > 0:
        return _interp_rows(values[:, :-shift_steps], t_src[shift_steps:], t_dst_ms)
    shift = -shift_steps
    return _interp_rows(values[:, shift:], t_src[:-shift], t_dst_ms)


def _array_metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    ref_arr = np.asarray(ref, dtype=float)
    test_arr = np.asarray(test, dtype=float)
    diff = test_arr - ref_arr
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    q99_abs = float(np.quantile(np.abs(diff), 0.99))
    ref_flat = ref_arr.ravel()
    test_flat = test_arr.ravel()
    if ref_flat.size > 1 and float(np.std(ref_flat)) > 0.0 and float(np.std(test_flat)) > 0.0:
        corr = float(np.corrcoef(ref_flat, test_flat)[0, 1])
    else:
        corr = float("nan")
    return {
        "rmse": rmse,
        "max_abs": max_abs,
        "q99_abs": q99_abs,
        "corr": corr,
    }


def _trace_metrics(ref: np.ndarray, test: np.ndarray, t_ms: np.ndarray) -> dict[str, float]:
    metrics = _array_metrics(ref, test)
    diff = np.asarray(test, dtype=float) - np.asarray(ref, dtype=float)
    idx = int(np.argmax(np.abs(diff)))
    metrics.update(
        {
            "time_of_max_abs_ms": float(np.asarray(t_ms, dtype=float).ravel()[idx]),
            "ref_peak": float(np.max(ref)),
            "test_peak": float(np.max(test)),
            "peak_diff": float(np.max(test) - np.max(ref)),
        }
    )
    return metrics


def _matrix_metrics(ref: np.ndarray, test: np.ndarray, t_ms: np.ndarray, x_um: np.ndarray) -> dict[str, float]:
    metrics = _array_metrics(ref, test)
    diff = np.asarray(test, dtype=float) - np.asarray(ref, dtype=float)
    row_idx, col_idx = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
    metrics.update(
        {
            "x_of_max_abs_um": float(np.asarray(x_um, dtype=float).ravel()[row_idx]),
            "time_of_max_abs_ms": float(np.asarray(t_ms, dtype=float).ravel()[col_idx]),
        }
    )
    return metrics


def _center_node(axon: MRG) -> tuple[int, float]:
    node_ids = np.asarray(axon.node_indices, dtype=int)
    node_pos = int(node_ids.shape[0] // 2)
    comp_idx = int(node_ids[node_pos])
    return comp_idx, float(np.asarray(axon.layout.position_values(unit="micrometer"))[comp_idx])


def _build_axonscope_case(
    *,
    diameter_um: float,
    nodes: int,
    tsim_ms: float,
    dt_ms: float,
    cathodic_uA: float,
    cathodic_duration_ms: float,
    anodic_uA: float,
    interphase_ms: float,
) -> tuple[AxonInstance, Any]:
    axon = MRG(diameter=diameter_um * um, nodes=nodes)
    x0_um = float(axon.length / 2.0)
    electrode = PointSourceElectrode(
        x=x0_um * um,
        y=ELECTRODE_Y_UM * um,
        z=ELECTRODE_Z_UM * um,
    )
    stim = Stimulus.biphasic(
        start=1.0,
        cathodic_amplitude=cathodic_uA * 1e-6,
        cathodic_duration=cathodic_duration_ms,
        anodic_amplitude=anodic_uA * 1e-6,
        interphase=interphase_ms,
    )
    simulation = AxonInstance(axon)
    simulation.add_extracellular_context(
        context=AnalyticalExtracellularContext(
            electrodes=[electrode.with_stimulus(stim)],
            sigma=SIGMA_S_M,
        ),
        replace=True,
    )
    result = CrankNicholson().solve(simulation, tsim=tsim_ms, dt=dt_ms, record_observables=True)
    if result.recordings is None:
        raise RuntimeError("AxonScope result does not contain observable recordings.")
    return simulation, result


def _build_nrv_case(
    axon_as: MRG,
    *,
    diameter_um: float,
    tsim_ms: float,
    dt_ms: float,
    cathodic_uA: float,
    cathodic_duration_ms: float,
    anodic_uA: float,
    interphase_ms: float,
) -> tuple[Any, dict[str, Any]]:
    import nrv

    axon_nrv = nrv.myelinated(
        0,
        0,
        diameter_um,
        float(axon_as.length),
        model="MRG",
        dt=dt_ms,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    x0_um = float(axon_as.length / 2.0)
    electrode = nrv.point_source_electrode(x0_um, ELECTRODE_Y_UM, ELECTRODE_Z_UM)
    stim = nrv.stimulus()
    stim.biphasic_pulse(1.0, cathodic_uA, cathodic_duration_ms, anodic_uA, interphase_ms)
    extra = nrv.stimulation("endoneurium_bhadra")
    extra.add_electrode(electrode, stim)
    axon_nrv.attach_extracellular_stimulation(extra)
    _enable_nrv_recordings(axon_nrv)
    return axon_nrv, axon_nrv.simulate(t_sim=tsim_ms)


def _recording_matrix(result: Any, group: str, name: str) -> np.ndarray:
    return np.asarray(result.recordings[group][name], dtype=float).T


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

    best_x: np.ndarray | None = None
    best_score = float("inf")
    for candidate in candidates:
        idx = np.asarray([int(np.argmin(np.abs(candidate - xi))) for xi in x_target_um], dtype=int)
        score = float(np.mean(np.abs(candidate[idx] - x_target_um)))
        if score < best_score:
            best_score = score
            best_x = candidate
    return best_x


def _compare_vext_profiles(
    axon_as: MRG,
    axon_nrv: Any,
    x_nrv_um: np.ndarray,
    x_as_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    extra_nrv = getattr(axon_nrv, "extra_stim", None)
    if extra_nrv is None:
        return None
    if not extra_nrv.synchronised:
        extra_nrv.synchronise_stimuli()

    t_edges_ms = np.asarray(extra_nrv.global_time_serie, dtype=float).ravel()
    if t_edges_ms.size < 2:
        return None
    t_probe_ms = 0.5 * (t_edges_ms[:-1] + t_edges_ms[1:])
    vext_as_mV = np.stack(
        [np.asarray(axon_as.extracellular_potential_mV(float(ti)), dtype=float) for ti in t_probe_ms],
        axis=1,
    )
    vext_nrv_mV = np.stack(
        [np.asarray(extra_nrv.compute_vext(i), dtype=float) for i in range(t_probe_ms.size)],
        axis=1,
    )
    x_nrv_vext_um = _best_vext_x_candidate(np.asarray(x_nrv_um, dtype=float), vext_nrv_mV.shape[0], x_as_um)
    if x_nrv_vext_um is None:
        return None
    vext_nrv_aligned_mV, _ = _align_rows_to_target_x(x_nrv_vext_um, vext_nrv_mV, x_as_um)
    return t_probe_ms, vext_as_mV, vext_nrv_aligned_mV


def _gate_metrics(
    *,
    gate_name: str,
    gate_as: np.ndarray,
    gate_nrv_aligned: np.ndarray,
    t_nrv_ms: np.ndarray,
    t_as_ms: np.ndarray,
    x_as_um: np.ndarray,
    node_mask: np.ndarray,
    sample_idx: int,
    shift_steps: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    gate_nrv_raw = _shifted_interp_rows(gate_nrv_aligned, t_nrv_ms, t_as_ms, shift_steps=0)
    gate_nrv_shifted = _shifted_interp_rows(gate_nrv_aligned, t_nrv_ms, t_as_ms, shift_steps=shift_steps)

    metrics = {
        "local_raw": _trace_metrics(gate_nrv_raw[sample_idx], gate_as[sample_idx], t_as_ms),
        f"local_shift_{shift_steps:+d}": _trace_metrics(gate_nrv_shifted[sample_idx], gate_as[sample_idx], t_as_ms),
        "node_matrix_raw": _matrix_metrics(gate_nrv_raw[node_mask], gate_as[node_mask], t_as_ms, x_as_um[node_mask]),
        f"node_matrix_shift_{shift_steps:+d}": _matrix_metrics(
            gate_nrv_shifted[node_mask],
            gate_as[node_mask],
            t_as_ms,
            x_as_um[node_mask],
        ),
    }
    arrays = {
        f"gate_{gate_name}_as": gate_as,
        f"gate_{gate_name}_nrv_raw": gate_nrv_raw,
        f"gate_{gate_name}_nrv_shift_{shift_steps:+d}": gate_nrv_shifted,
    }
    return metrics, arrays


def run_baseline(
    *,
    diameter_um: float,
    nodes: int,
    tsim_ms: float,
    dt_ms: float,
    cathodic_uA: float,
    cathodic_duration_ms: float,
    anodic_uA: float,
    interphase_ms: float,
    gate_shift_steps: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    axon_as, result_as = _build_axonscope_case(
        diameter_um=diameter_um,
        nodes=nodes,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        cathodic_uA=cathodic_uA,
        cathodic_duration_ms=cathodic_duration_ms,
        anodic_uA=anodic_uA,
        interphase_ms=interphase_ms,
    )
    axon_nrv, result_nrv = _build_nrv_case(
        axon_as,
        diameter_um=diameter_um,
        tsim_ms=tsim_ms,
        dt_ms=dt_ms,
        cathodic_uA=cathodic_uA,
        cathodic_duration_ms=cathodic_duration_ms,
        anodic_uA=anodic_uA,
        interphase_ms=interphase_ms,
    )

    t_as_ms = np.asarray(result_as.t, dtype=float).ravel()
    t_nrv_ms = np.asarray(result_nrv["t"], dtype=float).ravel()
    x_as_um = np.asarray(axon_as.layout.position_values(unit="micrometer"), dtype=float).ravel()
    x_nrv_um = np.asarray(result_nrv["x_rec"], dtype=float).ravel()
    sample_idx, sample_pos_um = _center_node(axon_as)
    node_mask = np.asarray(axon_as.node_mask, dtype=bool).ravel()

    vm_as = np.asarray(result_as.Vm, dtype=float).T
    vm_nrv = _normalize_nrv_matrix(np.asarray(result_nrv["V_mem"], dtype=float), t_nrv_ms, x_nrv_um)
    vm_nrv_aligned, nrv_row_indices = _align_rows_to_target_x(x_nrv_um, vm_nrv, x_as_um)
    vm_nrv_interp = _interp_rows(vm_nrv_aligned, t_nrv_ms, t_as_ms)

    metrics: dict[str, Any] = {
        "case": {
            "model": "MRG",
            "diameter_um": float(diameter_um),
            "nodes": int(nodes),
            "tsim_ms": float(tsim_ms),
            "dt_ms": float(dt_ms),
            "cathodic_uA": float(cathodic_uA),
            "cathodic_duration_ms": float(cathodic_duration_ms),
            "anodic_uA": float(anodic_uA),
            "interphase_ms": float(interphase_ms),
            "electrode_y_um": float(ELECTRODE_Y_UM),
            "electrode_z_um": float(ELECTRODE_Z_UM),
            "sigma_S_m": float(SIGMA_S_M),
        },
        "sample": {
            "axon_scope_index": int(sample_idx),
            "position_um": float(sample_pos_um),
            "nrv_index": int(nrv_row_indices[sample_idx]),
            "nrv_position_um": float(x_nrv_um[int(nrv_row_indices[sample_idx])]),
        },
        "vm": {
            "local": _trace_metrics(vm_nrv_interp[sample_idx], vm_as[sample_idx], t_as_ms),
            "matrix": _matrix_metrics(vm_nrv_interp, vm_as, t_as_ms, x_as_um),
            "node_matrix": _matrix_metrics(vm_nrv_interp[node_mask], vm_as[node_mask], t_as_ms, x_as_um[node_mask]),
        },
        "gates": {},
    }

    arrays: dict[str, np.ndarray] = {
        "t_as_ms": t_as_ms,
        "t_nrv_ms": t_nrv_ms,
        "x_as_um": x_as_um,
        "x_nrv_um": x_nrv_um,
        "nrv_row_indices_for_x_as": nrv_row_indices,
        "node_mask": node_mask,
        "vm_as_mV": vm_as,
        "vm_nrv_interp_mV": vm_nrv_interp,
    }

    vext_cmp = _compare_vext_profiles(axon_as, axon_nrv, x_nrv_um, x_as_um)
    if vext_cmp is not None:
        t_vext_ms, vext_as_mV, vext_nrv_mV = vext_cmp
        metrics["vext"] = {
            "local": _trace_metrics(vext_nrv_mV[sample_idx], vext_as_mV[sample_idx], t_vext_ms),
            "matrix": _matrix_metrics(vext_nrv_mV, vext_as_mV, t_vext_ms, x_as_um),
        }
        arrays.update(
            {
                "t_vext_ms": t_vext_ms,
                "vext_as_mV": vext_as_mV,
                "vext_nrv_mV": vext_nrv_mV,
            }
        )
    else:
        metrics["vext"] = None

    for gate_name in ("mp", "m", "h", "s"):
        gate_as = _recording_matrix(result_as, "gates", gate_name)
        gate_nrv = _normalize_nrv_matrix(np.asarray(result_nrv[gate_name], dtype=float), t_nrv_ms, x_nrv_um)
        gate_nrv_aligned, _ = _align_rows_to_target_x(x_nrv_um, gate_nrv, x_as_um)
        gate_metric, gate_arrays = _gate_metrics(
            gate_name=gate_name,
            gate_as=gate_as,
            gate_nrv_aligned=gate_nrv_aligned,
            t_nrv_ms=t_nrv_ms,
            t_as_ms=t_as_ms,
            x_as_um=x_as_um,
            node_mask=node_mask,
            sample_idx=sample_idx,
            shift_steps=gate_shift_steps,
        )
        metrics["gates"][gate_name] = gate_metric
        arrays.update(gate_arrays)

    return metrics, arrays


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _plot_report(metrics: dict[str, Any], arrays: dict[str, np.ndarray], fig_path: Path, gate_shift_steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = arrays["t_as_ms"]
    x = arrays["x_as_um"]
    node_mask = arrays["node_mask"].astype(bool)
    sample_idx = int(metrics["sample"]["axon_scope_index"])
    shift_key = f"gate_m_nrv_shift_{gate_shift_steps:+d}"

    fig, axs = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)

    axs[0, 0].plot(t, arrays["vm_as_mV"][sample_idx], lw=2.0, label="AxonScope")
    axs[0, 0].plot(t, arrays["vm_nrv_interp_mV"][sample_idx], "--", lw=2.0, label="NRV")
    axs[0, 0].set_title("Center node Vm")
    axs[0, 0].set_xlabel("Time [ms]")
    axs[0, 0].set_ylabel("Vm [mV]")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(t, arrays["gate_m_as"][sample_idx], lw=2.0, label="AxonScope m")
    axs[0, 1].plot(t, arrays["gate_m_nrv_raw"][sample_idx], "--", lw=2.0, label="NRV m raw")
    axs[0, 1].plot(t, arrays[shift_key][sample_idx], ":", lw=2.4, label=f"NRV m shift {gate_shift_steps:+d}")
    axs[0, 1].set_title("Center node m gate")
    axs[0, 1].set_xlabel("Time [ms]")
    axs[0, 1].set_ylabel("m")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    vm_err = arrays["vm_as_mV"] - arrays["vm_nrv_interp_mV"]
    vmax = float(np.max(np.abs(vm_err)))
    im0 = axs[1, 0].imshow(
        vm_err,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axs[1, 0].set_title("Vm error")
    axs[1, 0].set_xlabel("Time [ms]")
    axs[1, 0].set_ylabel("x [um]")
    fig.colorbar(im0, ax=axs[1, 0], label="AxonScope - NRV [mV]")

    m_err_nodes = arrays["gate_m_as"][node_mask] - arrays[shift_key][node_mask]
    vmax_m = float(np.max(np.abs(m_err_nodes)))
    im1 = axs[1, 1].imshow(
        m_err_nodes,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), 0.0, float(np.count_nonzero(node_mask) - 1)],
        cmap="coolwarm",
        vmin=-vmax_m,
        vmax=vmax_m,
    )
    axs[1, 1].set_title(f"m error on nodes, shift {gate_shift_steps:+d}")
    axs[1, 1].set_xlabel("Time [ms]")
    axs[1, 1].set_ylabel("Node index")
    fig.colorbar(im1, ax=axs[1, 1], label="AxonScope - NRV")

    for i, gate_name in enumerate(("mp", "h", "s")):
        axs[2, 0].plot(t, arrays[f"gate_{gate_name}_as"][sample_idx], lw=1.8, label=f"{gate_name} AS")
        axs[2, 0].plot(t, arrays[f"gate_{gate_name}_nrv_raw"][sample_idx], "--", lw=1.8, label=f"{gate_name} NRV")
    axs[2, 0].set_title("Other center-node gates")
    axs[2, 0].set_xlabel("Time [ms]")
    axs[2, 0].set_ylabel("Gate value")
    axs[2, 0].grid(True, alpha=0.3)
    axs[2, 0].legend(fontsize=8, ncol=2)

    summary = [
        f"Vm local RMSE: {metrics['vm']['local']['rmse']:.4f} mV",
        f"Vm matrix RMSE: {metrics['vm']['matrix']['rmse']:.4f} mV",
        f"m local raw RMSE: {metrics['gates']['m']['local_raw']['rmse']:.6f}",
        f"m local shift RMSE: {metrics['gates']['m'][f'local_shift_{gate_shift_steps:+d}']['rmse']:.6f}",
        f"m node raw RMSE: {metrics['gates']['m']['node_matrix_raw']['rmse']:.6f}",
        f"m node shift RMSE: {metrics['gates']['m'][f'node_matrix_shift_{gate_shift_steps:+d}']['rmse']:.6f}",
    ]
    axs[2, 1].axis("off")
    axs[2, 1].text(0.02, 0.98, "\n".join(summary), va="top", family="monospace", fontsize=11)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _write_artifacts(
    *,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
    out_dir: Path,
    prefix: str,
    plot: bool,
    gate_shift_steps: int,
) -> tuple[Path, Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}_metrics.json"
    npz_path = out_dir / f"{prefix}_traces.npz"
    json_path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)

    fig_path = None
    if plot:
        fig_path = out_dir / f"{prefix}_report.png"
        _plot_report(metrics, arrays, fig_path, gate_shift_steps)
    return json_path, npz_path, fig_path


def _print_summary(metrics: dict[str, Any], paths: tuple[Path, Path, Path | None] | None) -> None:
    gate_shift_steps = int(next(k for k in metrics["gates"]["m"] if k.startswith("local_shift_")).split("_")[-1])
    local_shift_key = f"local_shift_{gate_shift_steps:+d}"
    node_shift_key = f"node_matrix_shift_{gate_shift_steps:+d}"
    print("=== MRG extracellular AxonScope vs NRV baseline ===")
    print(
        f"d={metrics['case']['diameter_um']:.3f} um | nodes={metrics['case']['nodes']} | "
        f"dt={metrics['case']['dt_ms']:.4f} ms | tsim={metrics['case']['tsim_ms']:.3f} ms"
    )
    print(
        f"Vm local RMSE={metrics['vm']['local']['rmse']:.4f} mV | "
        f"matrix RMSE={metrics['vm']['matrix']['rmse']:.4f} mV | "
        f"corr={metrics['vm']['matrix']['corr']:.5f}"
    )
    print(
        f"m local RMSE raw={metrics['gates']['m']['local_raw']['rmse']:.6f} | "
        f"shifted={metrics['gates']['m'][local_shift_key]['rmse']:.6f}"
    )
    print(
        f"m node RMSE raw={metrics['gates']['m']['node_matrix_raw']['rmse']:.6f} | "
        f"shifted={metrics['gates']['m'][node_shift_key]['rmse']:.6f}"
    )
    if metrics.get("vext") is not None:
        print(
            f"Vext matrix RMSE={metrics['vext']['matrix']['rmse']:.6e} mV | "
            f"max={metrics['vext']['matrix']['max_abs']:.6e} mV"
        )
    if paths is not None:
        json_path, npz_path, fig_path = paths
        print(f"metrics: {json_path}")
        print(f"traces : {npz_path}")
        if fig_path is not None:
            print(f"figure : {fig_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diameter", type=float, default=10.0, help="MRG fiber diameter in um.")
    parser.add_argument("--nodes", type=int, default=9, help="Number of MRG nodes.")
    parser.add_argument("--tsim", type=float, default=4.0, help="Simulation duration in ms.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step in ms.")
    parser.add_argument("--cathodic-uA", type=float, default=80.0, help="Cathodic phase amplitude in uA.")
    parser.add_argument("--cathodic-duration", type=float, default=0.08, help="Cathodic phase duration in ms.")
    parser.add_argument("--anodic-uA", type=float, default=20.0, help="Anodic phase amplitude in uA.")
    parser.add_argument("--interphase", type=float, default=0.04, help="Interphase duration in ms.")
    parser.add_argument("--gate-shift-steps", type=int, default=1, help="NRV gate shift, in time steps, used in diagnostics.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/nrv_performance/mrg_extracellular_baseline"),
        help="Directory for metrics, traces, and optional report figure.",
    )
    parser.add_argument("--no-artifacts", action="store_true", help="Run and print metrics without writing files.")
    parser.add_argument("--no-plot", action="store_true", help="Do not write the PNG report.")
    args = parser.parse_args(argv)

    metrics, arrays = run_baseline(
        diameter_um=args.diameter,
        nodes=args.nodes,
        tsim_ms=args.tsim,
        dt_ms=args.dt,
        cathodic_uA=args.cathodic_uA,
        cathodic_duration_ms=args.cathodic_duration,
        anodic_uA=args.anodic_uA,
        interphase_ms=args.interphase,
        gate_shift_steps=args.gate_shift_steps,
    )

    paths = None
    if not args.no_artifacts:
        prefix = f"mrg_d{args.diameter:.3f}_nodes{args.nodes}_dt{args.dt:.4f}".replace(".", "p")
        paths = _write_artifacts(
            metrics=metrics,
            arrays=arrays,
            out_dir=args.out_dir,
            prefix=prefix,
            plot=not args.no_plot,
            gate_shift_steps=args.gate_shift_steps,
        )
    _print_summary(metrics, paths)


if __name__ == "__main__":
    main()
