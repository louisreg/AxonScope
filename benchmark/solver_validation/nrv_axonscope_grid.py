from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from axonscope.axons import HodgkinHuxley
from axonscope.solvers import CrankNicholson


@dataclass(frozen=True)
class GridCase:
    model: str
    dt_ms: float
    nx: int
    tsim_ms: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AxonScope and NRV over a dt/Nx/Tsim grid."
    )
    parser.add_argument("--model", choices=["hh"], default="hh")
    parser.add_argument("--dt", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--nx", nargs="+", type=int, default=[51, 101])
    parser.add_argument("--tsim", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument(
        "--output-mode",
        choices=["full_trace", "probes"],
        default="probes",
        help="AxonScope voltage output mode. Use probes for large sweeps.",
    )
    parser.add_argument(
        "--probe-positions-um",
        nargs="+",
        type=float,
        default=[0.0, 500.0, 1000.0],
        help="Probe positions used with --output-mode probes.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/results/nrv_axonscope_grid"),
    )
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    cases = [
        GridCase(args.model, dt_ms=dt, nx=nx, tsim_ms=tsim)
        for dt in args.dt
        for nx in args.nx
        for tsim in args.tsim
    ]
    rows = [
        run_case(
            case,
            output_mode=args.output_mode,
            probe_positions_um=tuple(args.probe_positions_um),
        )
        for case in cases
    ]
    prefix = args.prefix or datetime.now().strftime("nrv_axonscope_grid_%Y%m%d_%H%M%S")
    json_path, csv_path = write_results(rows, args.out_dir, prefix)

    print("=== NRV/AxonScope grid ===")
    for row in rows:
        print(
            f"{row['model']:8s} dt={row['dt_ms']:g} ms nx={row['nx']:4d} "
            f"tsim={row['tsim_ms']:g} ms "
            f"rmse={row['vm_rmse_mV']:.4f} mV max={row['vm_max_abs_mV']:.4f} mV "
            f"AS={row['axonscope_runtime_s']:.3f}s NRV={row['nrv_runtime_s']:.3f}s"
        )
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")


def run_case(
    case: GridCase,
    *,
    output_mode: str,
    probe_positions_um: tuple[float, ...],
) -> dict[str, Any]:
    if case.model != "hh":
        raise ValueError(f"Unsupported model: {case.model}")

    axon_as = _make_hh_axonscope(case.nx)
    solve_kwargs: dict[str, Any] = {"output_mode": output_mode}
    if output_mode == "probes":
        solve_kwargs["probe_positions_um"] = probe_positions_um

    start = time.perf_counter()
    res_as = CrankNicholson().solve(
        axon_as,
        tsim=case.tsim_ms,
        dt=case.dt_ms,
        **solve_kwargs,
    )
    axonscope_runtime_s = time.perf_counter() - start

    axon_nrv = _make_hh_nrv(case.nx, case.dt_ms)
    _enable_nrv_voltage_recording(axon_nrv)
    start = time.perf_counter()
    res_nrv = axon_nrv.simulate(t_sim=case.tsim_ms)
    nrv_runtime_s = time.perf_counter() - start

    vm_as = np.asarray(res_as.Vm, dtype=float).T
    t_as = np.asarray(res_as.t, dtype=float).ravel()
    x_as = res_as.spatial_positions_um()
    vm_nrv, x_nrv, t_nrv = _nrv_vm_matrix(res_nrv)
    _, vm_nrv_aligned, _ = _align_rows_to_target_x(x_nrv, vm_nrv, x_as)
    vm_nrv_interp = _interp_rows(vm_nrv_aligned, t_nrv, t_as)
    rmse, max_abs, q99_abs = _trace_metrics(vm_nrv_interp, vm_as)
    peak_diff = float(abs(float(np.max(vm_as)) - float(np.max(vm_nrv_interp))))

    return {
        "model": case.model,
        "dt_ms": float(case.dt_ms),
        "nx": int(case.nx),
        "tsim_ms": float(case.tsim_ms),
        "output_mode": output_mode,
        "probe_positions_um": tuple(float(x) for x in x_as),
        "axonscope_runtime_s": float(axonscope_runtime_s),
        "nrv_runtime_s": float(nrv_runtime_s),
        "vm_rmse_mV": rmse,
        "vm_max_abs_mV": max_abs,
        "vm_q99_abs_mV": q99_abs,
        "vm_peak_diff_mV": peak_diff,
        "axonscope_shape": tuple(int(v) for v in np.asarray(res_as.Vm).shape),
        "nrv_shape": tuple(int(v) for v in vm_nrv.shape),
    }


def _make_hh_axonscope(nx: int):
    axon = HodgkinHuxley(
        L=1000.0,
        d=0.5,
        Nx=nx,
        celsius=6.3,
        Vinit=-70.0,
        include_passive_leak=True,
        g_pas=0.001,
        e_pas=-70.0,
    )
    axon.insert_I_Clamp(position=500.0, t_start=1.0, duration=1.0, amplitude=2.0)
    return axon


def _make_hh_nrv(nx: int, dt_ms: float):
    import nrv

    axon = nrv.unmyelinated(
        0,
        0,
        0.5,
        1000.0,
        dt=dt_ms,
        Nsec=1,
        Nseg_per_sec=nx,
        model="HH",
        v_init=-70.0,
        T=6.3,
    )
    axon.insert_I_Clamp(0.5, 1.0, 1.0, 2.0)
    return axon


def _enable_nrv_voltage_recording(axon_nrv) -> None:
    axon_nrv.record_V_mem = True


def _nrv_vm_matrix(results_nrv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_nrv = np.asarray(results_nrv["t"], dtype=float).ravel()
    x_nrv = np.asarray(results_nrv["x_rec"], dtype=float)
    vm_nrv = _normalize_nrv_matrix(results_nrv["V_mem"], t_nrv, x_nrv)
    return vm_nrv, x_nrv, t_nrv


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
