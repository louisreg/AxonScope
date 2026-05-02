from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from axonscope.axons import HodgkinHuxley, MRG
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers import CrankNicholson
from axonscope.stimulus import Stimulus


ModelName = Literal["hh_intracellular", "mrg_intracellular", "mrg_extracellular"]

MODEL_ALIASES = {
    "hh": "hh_intracellular",
    "hh_intracellular": "hh_intracellular",
    "mrg_intracellular": "mrg_intracellular",
    "mrg_extracellular": "mrg_extracellular",
}

PROFILE_DEFAULTS = {
    "smoke": {
        "models": ("hh_intracellular",),
        "dt_ms": (0.01,),
        "nx": (21,),
        "nodes": (5,),
        "tsim_ms": (1.0,),
        "hh_diameter_um": (0.5,),
        "mrg_diameter_um": (10.0,),
        "repeats": 1,
        "warmups": 0,
    },
    "full": {
        "models": ("hh_intracellular", "mrg_intracellular", "mrg_extracellular"),
        "dt_ms": (0.005, 0.01),
        "nx": (51, 101),
        "nodes": (5, 9),
        "tsim_ms": (4.0, 8.0),
        "hh_diameter_um": (0.5,),
        "mrg_diameter_um": (8.7, 10.0),
        "repeats": 1,
        "warmups": 0,
    },
}


@dataclass(frozen=True)
class GridCase:
    model: ModelName
    dt_ms: float
    tsim_ms: float
    diameter_um: float
    nx: int | None = None
    nodes: int | None = None


@dataclass(frozen=True)
class TimedValue:
    elapsed_s: float
    value: Any


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare AxonScope and NRV accuracy/runtime over a small grid."
    )
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=tuple(MODEL_ALIASES),
        default=None,
        help="Model(s) to run. 'hh' is kept as an alias for hh_intracellular.",
    )
    parser.add_argument("--dt", nargs="+", type=float, default=None)
    parser.add_argument("--nx", nargs="+", type=int, default=None, help="Unmyelinated compartment counts.")
    parser.add_argument("--nodes", nargs="+", type=int, default=None, help="MRG node counts.")
    parser.add_argument("--tsim", nargs="+", type=float, default=None)
    parser.add_argument("--diameter", nargs="+", type=float, default=None, help="Override model diameters in um.")
    parser.add_argument("--repeats", type=int, default=None, help="Total measured solves/simulations per case.")
    parser.add_argument("--warmups", type=int, default=None, help="Throwaway warm solves before repeated timings.")
    parser.add_argument("--threshold", type=float, default=-10.0, help="Spike threshold in mV.")
    parser.add_argument("--record-gates", action="store_true", help="Record and compare the m gate when available.")
    parser.add_argument("--gate-shift-steps", type=int, default=1, help="NRV gate shift used for shifted m metrics.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/validation/nrv_axonscope_grid"),
    )
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--list", action="store_true", help="List model/profile defaults and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded cases without running simulations.")
    args = parser.parse_args(argv)

    if args.list:
        _print_available()
        return

    cases = _build_cases(args)
    if args.dry_run:
        _print_cases(cases)
        return

    rows = [
        run_case(
            case,
            repeats=_arg_or_default(args.repeats, args.profile, "repeats"),
            warmups=_arg_or_default(args.warmups, args.profile, "warmups"),
            threshold_mV=float(args.threshold),
            record_gates=bool(args.record_gates),
            gate_shift_steps=int(args.gate_shift_steps),
        )
        for case in cases
    ]
    prefix = args.prefix or datetime.now().strftime("nrv_axonscope_grid_%Y%m%d_%H%M%S")
    json_path, csv_path = write_results(rows, args.out_dir, prefix)

    print("=== NRV/AxonScope grid ===")
    for row in rows:
        speedup = _fmt_optional(row.get("speedup_nrv_over_as_total_first"))
        warm = _fmt_optional(row.get("as_warm_total_median_s"))
        print(
            f"{row['model']:20s} dt={row['dt_ms']:g} ms "
            f"nx={row['axon_nx']:4d} tsim={row['tsim_ms']:g} ms "
            f"rmse={row['vm_rmse_mV']:.4f} mV max={row['vm_max_abs_mV']:.4f} mV "
            f"AS_total={row['as_total_first_s']:.3f}s AS_warm={warm}s "
            f"NRV_total={row['nrv_total_s']:.3f}s speedup={speedup}"
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")


def run_case(
    case: GridCase,
    *,
    repeats: int,
    warmups: int,
    threshold_mV: float,
    record_gates: bool,
    gate_shift_steps: int,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")
    if warmups < 0:
        raise ValueError(f"warmups must be >= 0, got {warmups}.")

    as_build = _time_call(lambda: _make_axonscope_case(case))
    axon_as = as_build.value
    as_solve = _time_axonscope_solve(axon_as, case, record_gates=record_gates)
    result_as = as_solve.value
    as_materialize = _time_call(lambda: _materialize_axonscope_result(result_as, record_gates=record_gates))
    as_data = as_materialize.value

    for _ in range(warmups):
        axon_warm = _make_axonscope_case(case)
        warm_solve = _time_axonscope_solve(axon_warm, case, record_gates=record_gates)
        _time_call(lambda res=warm_solve.value: _materialize_axonscope_result(res, record_gates=record_gates))

    as_warm_samples = []
    as_warm_materialize_samples = []
    as_warm_total_samples = []
    for _ in range(max(0, repeats - 1)):
        axon_repeat = _make_axonscope_case(case)
        repeat_solve = _time_axonscope_solve(axon_repeat, case, record_gates=record_gates)
        repeat_materialize = _time_call(
            lambda res=repeat_solve.value: _materialize_axonscope_result(res, record_gates=record_gates)
        )
        as_warm_samples.append(repeat_solve.elapsed_s)
        as_warm_materialize_samples.append(repeat_materialize.elapsed_s)
        as_warm_total_samples.append(repeat_solve.elapsed_s + repeat_materialize.elapsed_s)

    nrv_build = _time_call(lambda: _make_nrv_case(case, axon_as))
    axon_nrv = nrv_build.value
    _enable_nrv_recordings(axon_nrv, record_gates=record_gates)
    nrv_sim = _time_call(lambda: axon_nrv.simulate(t_sim=case.tsim_ms))
    result_nrv = nrv_sim.value
    nrv_materialize = _time_call(lambda: _materialize_nrv_result(result_nrv, record_gates=record_gates))
    nrv_data = nrv_materialize.value

    for _ in range(warmups):
        axon_warm = _make_nrv_case(case, axon_as)
        _enable_nrv_recordings(axon_warm, record_gates=record_gates)
        warm_sim = _time_call(lambda ax=axon_warm: ax.simulate(t_sim=case.tsim_ms))
        _time_call(lambda res=warm_sim.value: _materialize_nrv_result(res, record_gates=record_gates))

    nrv_repeat_samples = []
    nrv_repeat_materialize_samples = []
    nrv_repeat_total_samples = []
    for _ in range(max(0, repeats - 1)):
        axon_repeat = _make_nrv_case(case, axon_as)
        _enable_nrv_recordings(axon_repeat, record_gates=record_gates)
        repeat_sim = _time_call(lambda ax=axon_repeat: ax.simulate(t_sim=case.tsim_ms))
        repeat_materialize = _time_call(
            lambda res=repeat_sim.value: _materialize_nrv_result(res, record_gates=record_gates)
        )
        nrv_repeat_samples.append(repeat_sim.elapsed_s)
        nrv_repeat_materialize_samples.append(repeat_materialize.elapsed_s)
        nrv_repeat_total_samples.append(repeat_sim.elapsed_s + repeat_materialize.elapsed_s)

    vm_as = as_data["Vm"].T
    t_as = as_data["t"]
    x_as = np.asarray(axon_as.x, dtype=float).ravel()
    vm_nrv, x_nrv, t_nrv = _nrv_vm_matrix(nrv_data)
    x_target, vm_nrv_aligned, row_idx = _align_rows_to_target_x(x_nrv, vm_nrv, x_as)
    vm_nrv_interp = _interp_rows(vm_nrv_aligned, t_nrv, t_as)

    vm_rmse, vm_max_abs, vm_q99_abs = _trace_metrics(vm_nrv_interp, vm_as)
    sample_idx = _sample_index(axon_as, x_as)
    sample_metrics = _sample_spike_metrics(
        vm_ref=vm_nrv_interp[sample_idx],
        vm_test=vm_as[sample_idx],
        t_ms=t_as,
        threshold_mV=threshold_mV,
    )
    velocity_as = _velocity_from_crossing_times(
        x_as,
        vm_as,
        t_as,
        center_x_um=float(x_as[sample_idx]),
        threshold_mV=threshold_mV,
    )
    velocity_nrv = _velocity_from_crossing_times(
        x_as,
        vm_nrv_interp,
        t_as,
        center_x_um=float(x_as[sample_idx]),
        threshold_mV=threshold_mV,
    )

    x_alignment_error = np.asarray(x_nrv, dtype=float).ravel()[row_idx] - x_target
    row = {
        "model": case.model,
        "dt_ms": float(case.dt_ms),
        "tsim_ms": float(case.tsim_ms),
        "input_nx": None if case.nx is None else int(case.nx),
        "nodes": None if case.nodes is None else int(case.nodes),
        "diameter_um": float(case.diameter_um),
        "axon_nx": int(vm_as.shape[0]),
        "record_gates": bool(record_gates),
        "threshold_mV": float(threshold_mV),
        "as_build_s": float(as_build.elapsed_s),
        "as_first_solve_s": float(as_solve.elapsed_s),
        "as_materialize_first_s": float(as_materialize.elapsed_s),
        "as_total_first_s": float(as_solve.elapsed_s + as_materialize.elapsed_s),
        "nrv_build_s": float(nrv_build.elapsed_s),
        "nrv_simulate_s": float(nrv_sim.elapsed_s),
        "nrv_materialize_s": float(nrv_materialize.elapsed_s),
        "nrv_total_s": float(nrv_sim.elapsed_s + nrv_materialize.elapsed_s),
        "as_warm_solve_repeats": int(len(as_warm_samples)),
        "nrv_repeat_repeats": int(len(nrv_repeat_samples)),
        "as_warm_solve_mean_s": _mean_or_none(as_warm_samples),
        "as_warm_solve_median_s": _median_or_none(as_warm_samples),
        "as_warm_solve_min_s": _min_or_none(as_warm_samples),
        "as_warm_materialize_mean_s": _mean_or_none(as_warm_materialize_samples),
        "as_warm_materialize_median_s": _median_or_none(as_warm_materialize_samples),
        "as_warm_total_mean_s": _mean_or_none(as_warm_total_samples),
        "as_warm_total_median_s": _median_or_none(as_warm_total_samples),
        "as_warm_total_min_s": _min_or_none(as_warm_total_samples),
        "nrv_repeat_mean_s": _mean_or_none(nrv_repeat_samples),
        "nrv_repeat_median_s": _median_or_none(nrv_repeat_samples),
        "nrv_repeat_min_s": _min_or_none(nrv_repeat_samples),
        "nrv_repeat_materialize_mean_s": _mean_or_none(nrv_repeat_materialize_samples),
        "nrv_repeat_materialize_median_s": _median_or_none(nrv_repeat_materialize_samples),
        "nrv_repeat_total_mean_s": _mean_or_none(nrv_repeat_total_samples),
        "nrv_repeat_total_median_s": _median_or_none(nrv_repeat_total_samples),
        "nrv_repeat_total_min_s": _min_or_none(nrv_repeat_total_samples),
        "speedup_nrv_over_as_first": _safe_ratio(nrv_sim.elapsed_s, as_solve.elapsed_s),
        "speedup_nrv_over_as_total_first": _safe_ratio(
            nrv_sim.elapsed_s + nrv_materialize.elapsed_s,
            as_solve.elapsed_s + as_materialize.elapsed_s,
        ),
        "speedup_nrv_over_as_warm": _safe_ratio(_median_or_none(nrv_repeat_samples), _median_or_none(as_warm_samples)),
        "speedup_nrv_over_as_total_warm": _safe_ratio(
            _median_or_none(nrv_repeat_total_samples),
            _median_or_none(as_warm_total_samples),
        ),
        "vm_rmse_mV": vm_rmse,
        "vm_max_abs_mV": vm_max_abs,
        "vm_q99_abs_mV": vm_q99_abs,
        "vm_peak_diff_mV": sample_metrics["peak_diff_mV"],
        "sample_position_um": float(x_as[sample_idx]),
        "sample_as_index": int(sample_idx),
        "vm_cross_time_diff_ms": sample_metrics["cross_time_diff_ms"],
        "as_sample_peak_mV": sample_metrics["test_peak_mV"],
        "nrv_sample_peak_mV": sample_metrics["ref_peak_mV"],
        "as_sample_activated": sample_metrics["test_activated"],
        "nrv_sample_activated": sample_metrics["ref_activated"],
        "velocity_as_m_s": velocity_as,
        "velocity_nrv_m_s": velocity_nrv,
        "velocity_diff_m_s": None if np.isnan(velocity_as) or np.isnan(velocity_nrv) else float(velocity_as - velocity_nrv),
        "x_alignment_max_um": float(np.max(np.abs(x_alignment_error))) if x_alignment_error.size else None,
        "x_alignment_rmse_um": float(np.sqrt(np.mean(x_alignment_error**2))) if x_alignment_error.size else None,
        "axonscope_shape": tuple(int(v) for v in as_data["Vm"].shape),
        "nrv_shape": tuple(int(v) for v in vm_nrv.shape),
    }
    row.update(
        _m_gate_metrics(
            as_data=as_data,
            nrv_data=nrv_data,
            t_as_ms=t_as,
            t_nrv_ms=t_nrv,
            x_nrv_um=x_nrv,
            x_as_um=x_as,
            sample_idx=sample_idx,
            node_mask=np.asarray(getattr(axon_as, "node_mask", np.ones_like(x_as, dtype=bool)), dtype=bool),
            shift_steps=gate_shift_steps,
            enabled=record_gates,
        )
    )
    return row


def _build_cases(args) -> list[GridCase]:
    defaults = PROFILE_DEFAULTS[args.profile]
    models = tuple(MODEL_ALIASES[name] for name in (args.model or defaults["models"]))
    dt_values = tuple(args.dt or defaults["dt_ms"])
    tsim_values = tuple(args.tsim or defaults["tsim_ms"])
    nx_values = tuple(args.nx or defaults["nx"])
    node_values = tuple(args.nodes or defaults["nodes"])

    cases: list[GridCase] = []
    for model in models:
        if model == "hh_intracellular":
            diameters = tuple(args.diameter or defaults["hh_diameter_um"])
            for dt_ms in dt_values:
                for nx in nx_values:
                    for tsim_ms in tsim_values:
                        for diameter_um in diameters:
                            cases.append(
                                GridCase(
                                    model=model,
                                    dt_ms=float(dt_ms),
                                    nx=int(nx),
                                    tsim_ms=float(tsim_ms),
                                    diameter_um=float(diameter_um),
                                )
                            )
            continue

        diameters = tuple(args.diameter or defaults["mrg_diameter_um"])
        for dt_ms in dt_values:
            for nodes in node_values:
                for tsim_ms in tsim_values:
                    for diameter_um in diameters:
                        cases.append(
                            GridCase(
                                model=model,
                                dt_ms=float(dt_ms),
                                nodes=int(nodes),
                                tsim_ms=float(tsim_ms),
                                diameter_um=float(diameter_um),
                            )
                        )
    return cases


def _make_axonscope_case(case: GridCase):
    if case.model == "hh_intracellular":
        if case.nx is None:
            raise ValueError("HH cases require nx.")
        axon = HodgkinHuxley(
            L=1000.0,
            d=case.diameter_um,
            Nx=case.nx,
            celsius=6.3,
            Vinit=-70.0,
            include_passive_leak=True,
            g_pas=0.001,
            e_pas=-70.0,
        )
        axon.insert_I_Clamp(position=500.0, t_start=1.0, duration=1.0, amplitude=2.0)
        axon.comparison_sample_position_um = 500.0
        return axon

    if case.nodes is None:
        raise ValueError("MRG cases require nodes.")
    axon = MRG(d=case.diameter_um, nodes=case.nodes)
    center_node_idx, center_node_pos_um = _mrg_center_node(axon)
    axon.comparison_sample_position_um = center_node_pos_um

    if case.model == "mrg_intracellular":
        axon.insert_I_Clamp(position=center_node_pos_um, t_start=1.0, duration=0.1, amplitude=2.0)
        return axon

    if case.model == "mrg_extracellular":
        _ = center_node_idx
        x0_um = float(axon.L / 2.0)
        electrode = PointSourceElectrode(
            x0_m=x0_um * 1e-6,
            y0_m=100e-6,
            z0_m=0.0,
            sigma_S_m=0.2,
        )
        stim = Stimulus.biphasic(
            start=1.0,
            cathodic_amplitude=80e-6,
            cathodic_duration=0.08,
            anodic_amplitude=20e-6,
            interphase=0.04,
        )
        axon.add_extracellular_ctx(electrode, stim, replace=True)
        return axon

    raise ValueError(f"Unsupported model: {case.model}")


def _make_nrv_case(case: GridCase, axon_as):
    import nrv

    if case.model == "hh_intracellular":
        if case.nx is None:
            raise ValueError("HH cases require nx.")
        axon = nrv.unmyelinated(
            0,
            0,
            case.diameter_um,
            1000.0,
            dt=case.dt_ms,
            Nsec=1,
            Nseg_per_sec=case.nx,
            model="HH",
            v_init=-70.0,
            T=6.3,
        )
        axon.insert_I_Clamp(0.5, 1.0, 1.0, 2.0)
        return axon

    axon = nrv.myelinated(
        0,
        0,
        case.diameter_um,
        float(axon_as.L),
        model="MRG",
        dt=case.dt_ms,
        node_shift=0,
        Nseg_per_sec=1,
        rec="all",
        T=37.0,
        v_init=-80.0,
    )
    if case.model == "mrg_intracellular":
        center_node = int(np.asarray(axon_as.node_indices).shape[0] // 2)
        axon.insert_I_Clamp_node(index=center_node, t_start=1.0, duration=0.1, amplitude=2.0)
        return axon

    if case.model == "mrg_extracellular":
        x0_um = float(axon_as.L / 2.0)
        electrode = nrv.point_source_electrode(x0_um, 100.0, 0.0)
        stim = nrv.stimulus()
        stim.biphasic_pulse(1.0, 80.0, 0.08, 20.0, 0.04)
        extra = nrv.stimulation("endoneurium_bhadra")
        extra.add_electrode(electrode, stim)
        axon.attach_extracellular_stimulation(extra)
        return axon

    raise ValueError(f"Unsupported model: {case.model}")


def _time_axonscope_solve(axon, case: GridCase, *, record_gates: bool) -> TimedValue:
    start = time.perf_counter()
    result = CrankNicholson().solve(
        axon,
        tsim=case.tsim_ms,
        dt=case.dt_ms,
        record_observables=record_gates,
    )
    _block_until_ready(result)
    return TimedValue(time.perf_counter() - start, result)


def _time_call(func) -> TimedValue:
    start = time.perf_counter()
    value = func()
    return TimedValue(time.perf_counter() - start, value)


def _block_until_ready(result) -> None:
    for attr in ("Vm", "t"):
        arr = getattr(result, attr, None)
        if hasattr(arr, "block_until_ready"):
            arr.block_until_ready()
    recordings = getattr(result, "recordings", None)
    if isinstance(recordings, dict):
        for group in recordings.values():
            if isinstance(group, dict):
                for arr in group.values():
                    if hasattr(arr, "block_until_ready"):
                        arr.block_until_ready()
    diagnostics = getattr(result, "diagnostics", None)
    if isinstance(diagnostics, dict):
        for arr in diagnostics.values():
            if hasattr(arr, "block_until_ready"):
                arr.block_until_ready()


def _materialize_axonscope_result(result, *, record_gates: bool) -> dict[str, np.ndarray]:
    data = {
        "Vm": np.array(result.Vm, dtype=float, copy=True),
        "t": np.array(result.t, dtype=float, copy=True).ravel(),
    }
    if record_gates:
        recordings = getattr(result, "recordings", None)
        if isinstance(recordings, dict) and "gates" in recordings and "m" in recordings["gates"]:
            data["m"] = np.array(recordings["gates"]["m"], dtype=float, copy=True)
    return data


def _materialize_nrv_result(result_nrv, *, record_gates: bool) -> dict[str, np.ndarray]:
    data = {
        "t": np.array(result_nrv["t"], dtype=float, copy=True).ravel(),
        "x_rec": np.array(result_nrv["x_rec"], dtype=float, copy=True),
        "V_mem": np.array(result_nrv["V_mem"], dtype=float, copy=True),
    }
    if record_gates and "m" in result_nrv:
        data["m"] = np.array(result_nrv["m"], dtype=float, copy=True)
    return data


def _enable_nrv_recordings(axon_nrv, *, record_gates: bool) -> None:
    axon_nrv.record_V_mem = True
    if not record_gates:
        return
    axon_nrv.record_particles = True
    axon_nrv.record_g_ions = True
    if hasattr(axon_nrv, "record_particules"):
        axon_nrv.record_particules = True


def _nrv_vm_matrix(results_nrv: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_nrv = np.asarray(results_nrv["x_rec"], dtype=float)
    vm_nrv = _normalize_nrv_matrix(results_nrv["V_mem"], t_nrv, x_nrv)
    return vm_nrv, x_nrv, t_nrv


def _m_gate_metrics(
    *,
    as_data: dict[str, np.ndarray],
    nrv_data: dict[str, np.ndarray],
    t_as_ms: np.ndarray,
    t_nrv_ms: np.ndarray,
    x_nrv_um: np.ndarray,
    x_as_um: np.ndarray,
    sample_idx: int,
    node_mask: np.ndarray,
    shift_steps: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {}
    if "m" not in as_data or "m" not in nrv_data:
        return {"m_gate_available": False}

    gate_as = np.asarray(as_data["m"], dtype=float).T
    gate_nrv = _normalize_nrv_matrix(np.asarray(nrv_data["m"], dtype=float), t_nrv_ms, x_nrv_um)
    _, gate_nrv_aligned, _ = _align_rows_to_target_x(x_nrv_um, gate_nrv, x_as_um)
    gate_nrv_raw = _interp_rows(gate_nrv_aligned, t_nrv_ms, t_as_ms)
    gate_nrv_shifted = _shifted_interp_rows(gate_nrv_aligned, t_nrv_ms, t_as_ms, shift_steps=shift_steps)

    node_mask = np.asarray(node_mask, dtype=bool)
    if node_mask.shape != x_as_um.shape or not np.any(node_mask):
        node_mask = np.ones_like(x_as_um, dtype=bool)

    local_raw = _trace_metrics(gate_nrv_raw[sample_idx], gate_as[sample_idx])
    local_shifted = _trace_metrics(gate_nrv_shifted[sample_idx], gate_as[sample_idx])
    node_raw = _trace_metrics(gate_nrv_raw[node_mask], gate_as[node_mask])
    node_shifted = _trace_metrics(gate_nrv_shifted[node_mask], gate_as[node_mask])
    return {
        "m_gate_available": True,
        "m_gate_shift_steps": int(shift_steps),
        "m_gate_local_rmse_raw": local_raw[0],
        "m_gate_local_max_raw": local_raw[1],
        "m_gate_local_q99_raw": local_raw[2],
        "m_gate_local_rmse_shifted": local_shifted[0],
        "m_gate_local_max_shifted": local_shifted[1],
        "m_gate_local_q99_shifted": local_shifted[2],
        "m_gate_node_rmse_raw": node_raw[0],
        "m_gate_node_max_raw": node_raw[1],
        "m_gate_node_q99_raw": node_raw[2],
        "m_gate_node_rmse_shifted": node_shifted[0],
        "m_gate_node_max_shifted": node_shifted[1],
        "m_gate_node_q99_shifted": node_shifted[2],
    }


def _mrg_center_node(axon: MRG) -> tuple[int, float]:
    node_ids = np.asarray(axon.node_indices, dtype=int)
    node_pos = int(node_ids.shape[0] // 2)
    comp_idx = int(node_ids[node_pos])
    return comp_idx, float(np.asarray(axon.x, dtype=float)[comp_idx])


def _sample_index(axon_as, x_as_um: np.ndarray) -> int:
    sample_position_um = getattr(axon_as, "comparison_sample_position_um", None)
    if sample_position_um is None:
        return int(x_as_um.size // 2)
    return int(np.argmin(np.abs(x_as_um - float(sample_position_um))))


def _sample_spike_metrics(
    *,
    vm_ref: np.ndarray,
    vm_test: np.ndarray,
    t_ms: np.ndarray,
    threshold_mV: float,
) -> dict[str, Any]:
    ref_peak = float(np.max(vm_ref))
    test_peak = float(np.max(vm_test))
    ref_cross = _first_cross_time(vm_ref, t_ms, threshold_mV)
    test_cross = _first_cross_time(vm_test, t_ms, threshold_mV)
    if np.isfinite(ref_cross) and np.isfinite(test_cross):
        cross_diff = float(test_cross - ref_cross)
    else:
        cross_diff = None
    return {
        "ref_peak_mV": ref_peak,
        "test_peak_mV": test_peak,
        "peak_diff_mV": float(abs(test_peak - ref_peak)),
        "ref_activated": bool(ref_peak >= threshold_mV),
        "test_activated": bool(test_peak >= threshold_mV),
        "cross_time_diff_ms": cross_diff,
    }


def _velocity_from_crossing_times(
    x_um: np.ndarray,
    vm_space_time: np.ndarray,
    t_ms: np.ndarray,
    *,
    center_x_um: float,
    threshold_mV: float,
) -> float:
    crossings = np.asarray(
        [_first_cross_time(trace, t_ms, threshold_mV) for trace in np.asarray(vm_space_time, dtype=float)],
        dtype=float,
    )
    x = np.asarray(x_um, dtype=float).ravel()
    active = np.isfinite(crossings)
    velocities = []
    for mask in (active & (x < center_x_um), active & (x > center_x_um)):
        if int(np.sum(mask)) < 2:
            continue
        xs_m = x[mask] * 1e-6
        ts_s = crossings[mask] * 1e-3
        order = np.argsort(ts_s)
        coeff = np.polyfit(ts_s[order], xs_m[order], 1)
        velocities.append(abs(float(coeff[0])))
    if not velocities:
        return float("nan")
    return float(statistics.fmean(velocities))


def _first_cross_time(trace_mV: np.ndarray, t_ms: np.ndarray, threshold_mV: float) -> float:
    trace = np.asarray(trace_mV, dtype=float).ravel()
    time = np.asarray(t_ms, dtype=float).ravel()
    above = trace >= threshold_mV
    idx = np.where(above[1:] & ~above[:-1])[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    t0, t1 = float(time[i]), float(time[i + 1])
    v0, v1 = float(trace[i]), float(trace[i + 1])
    if v1 == v0:
        return t1
    return t0 + (threshold_mV - v0) * (t1 - t0) / (v1 - v0)


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


def _trace_metrics(ref: np.ndarray, test: np.ndarray) -> tuple[float, float, float]:
    diff = np.asarray(test, dtype=float) - np.asarray(ref, dtype=float)
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    q99_abs = float(np.quantile(np.abs(diff), 0.99))
    return rmse, max_abs, q99_abs


def write_results(rows: Iterable[dict[str, Any]], out_dir: Path, prefix: str) -> tuple[Path, Path]:
    row_list = list(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    json_path.write_text(json.dumps(_jsonable(row_list), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = sorted({key for row in row_list for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({_jsonable(key): _jsonable(value) for key, value in row.items()} for row in row_list)
    return json_path, csv_path


def _arg_or_default(value: int | None, profile: str, key: str) -> int:
    if value is not None:
        return int(value)
    return int(PROFILE_DEFAULTS[profile][key])


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def _fmt_optional(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _print_available() -> None:
    print("Models:")
    for name in MODEL_ALIASES:
        print(f"  {name}")
    print("Profiles:")
    for name, defaults in PROFILE_DEFAULTS.items():
        print(
            f"  {name}: models={list(defaults['models'])} "
            f"dt={list(defaults['dt_ms'])} nx={list(defaults['nx'])} "
            f"nodes={list(defaults['nodes'])} tsim={list(defaults['tsim_ms'])}"
        )


def _print_cases(cases: list[GridCase]) -> None:
    print(f"Expanded cases: {len(cases)}")
    for case in cases:
        nx_label = f"nx={case.nx}" if case.nx is not None else f"nodes={case.nodes}"
        print(
            f"  {case.model:20s} dt={case.dt_ms:g} ms "
            f"{nx_label:10s} tsim={case.tsim_ms:g} ms d={case.diameter_um:g} um"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
